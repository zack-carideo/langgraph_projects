"""
Asynchronous RSS Feed Hunter using LangGraph
This module implements a state-based agent that discovers RSS feeds using Google Search API.

The agent works in three stages:
1. Generate search queries based on a theme
2. Search and discover RSS feeds
3. Validate and store results
"""
from IPython.display import Image, display
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import asyncio
import datetime
import logging
import os
import sys
from typing import Dict, List, Optional, TypedDict, Annotated
from operator import add
import json 
import pandas as pd 

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import nest_asyncio
from contextlib import AsyncExitStack
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from random_header_generator import HeaderGenerator

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Add project root to path
_root = "/home/zjc1002/Mounts/code/"
sys.path.append(_root)

# Import custom modules
from admin.api_keys import _api_keys
from admin.sys_ops import _set_env
from langgraph_projects.utils.tools import GoogleRSSDiscovery , read_multiple_rss_feeds

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ===== CONFIGURATION =====
class Config:
    """Centralized configuration for the RSS Hunter agent."""
    
    # API Keys - Set these in environment
    API_KEYS = [
        "TAVILY_API_KEY", 
        "ANTHROPIC_API_KEY", 
        "OPENAI_API_KEY", 
        "LANGSMITH_API_KEY",
        "GOOGLE_API_KEY", 
        "GOOGLE_CSE_ID"
    ]
    
    # LangSmith Configuration
    LANGSMITH_TRACE = 'true'
    LANGSMITH_PROJECT = 'rss_hunter_v2'
    LANGGRAPH_PERSISTANT_DB_PATH =  "/home/zjc1002/Mounts/code/langgraph_projects/custom_implementations/App1/my_agent/sqlite:/home/zjc1002/Mounts/code/langgraph_projects/custom_implementations/rss_results.db"

    # LLM Configuration
    LLM_MODEL = 'claude-3-haiku-20240307'
    LLM_TEMPERATURE = 0
    LLM_MAX_TOKENS = 4000
    LLM_STREAMING = True
    
    # RSS Discovery Configuration
    MAX_SEARCH_RESULTS = 4
    EXCLUDE_DOMAINS = ["https://www.feedspot.com","https://feedvisor.com", 'https://facebook.com']
    


    @classmethod
    def setup_environment(cls):
        """Set up all required environment variables."""
        for key in cls.API_KEYS:
            _set_env(key, value=_api_keys.get(key, ""))
        
        # Set LangSmith environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = cls.LANGSMITH_TRACE
        os.environ["LANGCHAIN_PROJECT"] = cls.LANGSMITH_PROJECT


# Set up environment
Config.setup_environment()

# Initialize LLM
llm = ChatAnthropic(
    model=Config.LLM_MODEL,
    temperature=Config.LLM_TEMPERATURE,
    max_tokens=Config.LLM_MAX_TOKENS,
    streaming=Config.LLM_STREAMING,
)


# ===== STATE DEFINITION =====
class RSSFeed(TypedDict):
    """Structure for RSS feed information."""
    url: str
    title: Optional[str]
    content_type: Optional[str]
    payload_size: int
    discovered_from: str  # URL or search query that led to this feed
    search_query: Optional[str] = None  #  search query used to find this feed

class RSSHunterState(TypedDict):
    """
    State definition for the RSS Hunter agent.
    
    Attributes:
        theme: The main topic/theme to search RSS feeds for
        search_queries: List of search queries generated based on the theme
        discovered_feeds: List of discovered RSS feeds with metadata
        processed_urls: Set of URLs already processed to avoid duplicates
        error_log: List of errors encountered during processing
        status: Current status of the agent
    """

    theme: str
    search_queries: Annotated[List[str], add]
    discovered_feeds: Annotated[List[RSSFeed], add]
    processed_urls: Annotated[List[str], add]
    error_log: Annotated[List[str], add]
    status: str

 # Type hint for async node
# ===== NODE FUNCTIONS =====

def generate_search_queries(state: RSSHunterState) -> Dict:
    """
    Generate diverse search queries for finding RSS feeds based on the theme.
    
    This node uses an LLM to create multiple search queries that explore
    different aspects of the main theme to maximize RSS feed discovery.
    """
    logger.info(f"Generating search queries for theme: {state['theme']}")
    
    # Define structured output for LLM
    class SearchQueries(BaseModel):
        queries: List[str] = Field(
            min_length=3, 
            max_length=5,
            description="List of search queries optimized for finding RSS feeds"
        )
    
    # Create structured LLM
    structured_llm = llm.with_structured_output(SearchQueries)
    
    # Prepare prompt
    existing_queries = state.get('search_queries', [])
    current_year = datetime.datetime.now().year
    
    prompt = f"""You are an expert at finding RSS feeds on specific topics.

Theme: {state['theme']}
Current Year: {current_year}

Generate {Config.MAX_SEARCH_RESULTS} search queries to find RSS feeds about this theme.

Requirements:
1. Each query should explore a different aspect or sub-topic of the theme
2. Include relevant keywords like "RSS", "feed", "XML feed", or "news feed"
3. Mix broad and specific queries for comprehensive coverage
4. Consider industry publications, news sites, and blogs
5. Avoid queries similar to these already used: {existing_queries}

Examples of good queries:
- "artificial intelligence RSS feeds technology news"
- "machine learning blog feed XML"
- "AI research updates RSS subscription"

Generate diverse, high-quality queries that will help discover active RSS feeds."""

    try:
        # Generate queries
        result = structured_llm.invoke([
            SystemMessage("You are an RSS feed discovery expert."),
            HumanMessage(prompt)
        ])
        
        new_queries = [q for q in result.queries if q not in existing_queries]
        logger.info(f"Generated {len(new_queries)} new search queries")
        
        #the outputs get saved to predefined state attributes 
        return {
            "search_queries": new_queries,
            "status": f"Generated {len(new_queries)} search queries"
        }
        
    except Exception as e:
        error_msg = f"Error generating queries: {str(e)}"
        logger.error(error_msg)
        return {
            "error_log": [error_msg],
            "status": "Error in query generation"
        }


async def search_and_discover_feeds(state: RSSHunterState) -> Dict:
    """
    Search for and discover RSS feeds using Google Custom Search API.
    
    This async node processes all search queries concurrently to find RSS feeds
    efficiently. It handles both direct RSS URLs and pages containing RSS links.

    await search_and_discover_feeds(RSSHunterState(theme = 'zack test'
                                         ,search_queries = ['government','buisness'] 
                                         )) 
    """
    logger.info(f"Starting RSS discovery for {len(state.get('search_queries', []))} queries")
    
    #try:

    # Process all queries concurrently
    assert state.get('search_queries',[]) != [], 'no queries provided'
    logger.info(f"Googling {len(state.get('search_queries', []))} queries")

    # Create discovery tasks for each search query
    discovery = GoogleRSSDiscovery(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        cse_id=os.environ.get("GOOGLE_CSE_ID"),
        n_results=Config.MAX_SEARCH_RESULTS,
        exclude_domains=Config.EXCLUDE_DOMAINS
    )

    #gather results 
    #[x['rss_links'] for x in all_results if x['rss_links']!=[]]
    #[xx[0].keys() for xx in [x['rss_links'] for x in all_results if x['rss_links']!=[]]]
    #parent keys: ['title', 'url', 'snippet', 'publisher', 'search_query', 'rss_links']
    #child keys(rss_links): ['url', 'response_code', 'response_method', 'response', 'content_type', 'is_rss']
    #ISSUE: searcher has RSS information in it as well as all_results
    searcher, all_results = await discovery.end2end_googler(
        state['search_queries'],
        max_links=Config.MAX_SEARCH_RESULTS
    )
    
    print(f"Total RSS Feeds in Searher output: {sum([x.get('rss_links',[])!=[] for x in searcher])}")

    # Process results
    discovered_feeds, errors = [],[]
    processed_urls = state.get('processed_urls', [])
    processed_urls.extend([x['url'] for x in searcher])
    
    logger.info(f"THESE URLS HAVE BEEN PROCESSED: {processed_urls}")
    #loop over queries and their results
    logger.info("Processing discovered feeds...")
    print(state['search_queries'])

    # Loop through all results and extract RSS links
    for result in all_results: 
    
        print(f"QUERY:{result['search_query']}")
        print("RESULT:", result['rss_links'])

        #verify each result is a dict
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

        if isinstance(result, Exception):
            error_msg = f"ERROR processing query '{result['search_query']}': {str(result)}"
            logger.error(error_msg)
            errors.append(error_msg)
            print("ERROR")
            continue
        
        # Process each discovered feed
        if ((result.get('rss_links') and (len(result.get('rss_links'))>0))):

            
            for feed_info in result['rss_links']:
                url = feed_info.get('url', '')
                
                # Skip if already processed
                if url in processed_urls:
                    continue
                else: 
                    processed_urls.append(url)
                
                # Create RSS feed entry
                rss_feed = RSSFeed(
                    url=url,
                    title=feed_info.get('title', 'No Title'),
                    content_type=feed_info.get('content_type'),
                    payload_size=len(feed_info.get('response', b'').text),
                    discovered_from=result['search_query']  
                )
                
                discovered_feeds.append(rss_feed)
                logger.info(f"Discovered feed: {url} (size: {rss_feed['payload_size']} bytes)")

    
    # Summary statistics
    total_feeds = len(discovered_feeds)
    logger.info(f"Discovery complete: {total_feeds} feeds found, {len(errors)} errors")
    
    return {
        "discovered_feeds": discovered_feeds,
        "processed_urls": list(processed_urls) if not  isinstance(processed_urls, list) else processed_urls,
        "error_log": errors,
        "status": f"Discovered {total_feeds} RSS feeds"
    }
        


def analyze_results(state: RSSHunterState) -> Dict:
    """
    Analyze and summarize the discovered RSS feeds.
    
    This node provides insights about the discovered feeds and can be extended
    to perform additional validation or categorization.
    """
    feeds = state.get('discovered_feeds', [])
    errors = state.get('error_log', [])
    
    # Group feeds by source query
    feeds_by_query = {}
    for feed in feeds:
        query = feed['discovered_from']
        if query not in feeds_by_query:
            feeds_by_query[query] = []
        feeds_by_query[query].append(feed)
    
    # Create summary
    summary_lines = [
        f"RSS Feed Discovery Summary for '{state['theme']}'",
        f"=" * 50,
        f"Total feeds discovered: {len(feeds)}",
        f"Total errors encountered: {len(errors)}",
        f"",
        "Feeds by search query:"
    ]
    
    for query, query_feeds in feeds_by_query.items():
        summary_lines.append(f"\n'{query}': {len(query_feeds)} feeds")
        for feed in query_feeds[:3]:  # Show first 3 feeds per query
            summary_lines.append(f"  - {feed['url']} ({feed['payload_size']} bytes)")
    
    summary = "\n".join(summary_lines)
    logger.info(summary)
    
    return {"status": f"Analysis complete: {len(feeds)} feeds discovered"}


# ===== GRAPH BUILDER =====

def build_rss_hunter_graph() -> StateGraph:
    """
    Build the LangGraph state machine for RSS feed discovery.
    
    The graph consists of three main nodes:
    1. generate_queries: Creates search queries based on theme
    2. search_and_discover: Finds RSS feeds (async)
    3. analyze_results: Summarizes findings
    
    Returns:
        Compiled StateGraph with memory checkpoint
    """

    # Create graph
    graph = StateGraph(RSSHunterState)
    
    # Add nodes
    graph.add_node("generate_queries", generate_search_queries)
    graph.add_node("search_and_discover", search_and_discover_feeds) # Async node
    graph.add_node("analyze_results", analyze_results)
    
    # Define flow
    graph.add_edge(START, "generate_queries")
    graph.add_edge("generate_queries", "search_and_discover")
    graph.add_edge("search_and_discover", "analyze_results")
    graph.add_edge("analyze_results", END)
    
    return graph



# ===== MAIN EXECUTION =====

async def run_rss_hunter(theme: str
                         , thread_id: str = "default2"
                         , db_path = "sqlite:////home/zjc1002/Mounts/code/langgraph_projects/custom_implementations/rss_results.db"
                         ) -> RSSHunterState:
    """
    Run the RSS Hunter agent asynchronously. optionally save results to db 
    
    Args:
        theme: The topic to search RSS feeds for
        thread_id: Unique identifier for conversation memory
        db_path: Optional path to SQLite database for saving results
        
    Returns:
        Final state containing all discovered RSS feeds
    """

    # Initial state
    initial_state = RSSHunterState(
        theme=theme,
        search_queries=[],
        discovered_feeds=[],
        processed_urls=[],
        error_log=[],
        status="Initialized"
    )
    
    # Configuration for memory
    config = {"configurable": {"thread_id": thread_id}}
    
    # Build graph
    graph = build_rss_hunter_graph()

    # add memory and invoke async graph
    if db_path:

        async with AsyncExitStack() as stack:
            # Initialize AsyncSqliteSaver with the database path
            saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(db_path))
            logger.info(f"Using SQLite database at: {db_path}")
            
            # Compile graph with the saver
            g_comp = graph.compile(checkpointer=saver)
            logger.info(f"Starting RSS Hunter for theme: {theme}")
            final_state = await g_comp.ainvoke(initial_state, config)

    else:

       # Compile and Run with MemorySaver
        memory = MemorySaver()
        g_comp = graph.compile(checkpointer=memory)
        logger.info("RSS Hunter graph built successfully")
        logger.info(f"Starting RSS Hunter for theme: {theme}")
        final_state = await g_comp.ainvoke(initial_state, config)

    #show the graph and return the final state
    logger.info("RSS Hunter completed successfully")
    display(Image(g_comp.get_graph().draw_mermaid_png()))

    return final_state, g_comp


# ===== USAGE EXAMPLE =====

async def main(db_path: str = None) -> RSSHunterState:
    """Example usage of the RSS Hunter agent."""
    
    # Example themes to explore
    themes = [
        "artificial intelligence and machine learning trends",
        "financial markets and algorithmic trading",
        "consumer behavior and retail analytics", 
        "audit and compliance in financial services",
    ]
    
    # Run hunter for first theme
    theme = themes[0]
    result, graph = await run_rss_hunter(theme, thread_id="ai_ml_feeds_001", db_path=db_path)

    # Display results
    print(f"\n{'='*60}")
    print(f"RSS FEED DISCOVERY RESULTS")
    print(f"{'='*60}")
    print(f"Theme: {theme}")
    print(f"Status: {result['status']}")
    print(f"Total Feeds: {len(result['discovered_feeds'])}")
    print(f"Errors: {len(result['error_log'])}")
    
    # Show sample feeds
    print(f"\nSample RSS Feeds:")
    for feed in result['discovered_feeds'][:10]:
        print(f"\n- URL: {feed['url']}")
        print(f"  Size: {feed['payload_size']:,} bytes")
        print(f"  Type: {feed['content_type']}")
        print(f"  From: {feed['discovered_from']}")

    return result , graph


def load_rss_results(db_path:str)-> pd.DataFrame:

    #EXPLORING HOW TO STORE / PERSIST RESULTS IN SQLITE
    data_col = 'metadata'

    #load langgraphs checkpoint sqlite db
    conn = sqlite3.connect(Config.LANGGRAPH_PERSISTANT_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM checkpoints ", conn)
    conn.close()

    #convert to string
    df[data_col] = df[data_col].apply(
        lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    #filter for search_and_discover results
    df['results_flag'] = df[data_col].apply(
        lambda x: 1 if isinstance(x, str) and 'search_and_discover' in x else 0)
    
    #extract search_and_discover results
    df[f"{data_col}_info"] = df.apply(
        lambda x: json.loads(x[data_col])['writes']['search_and_discover'] if x['results_flag'] == 1 else {}, axis=1)

    #extract discovered feeds from the results
    rss_feeds = pd.DataFrame(
        df[df['results_flag'] == 1][f"{data_col}_info"].tolist()
        )['discovered_feeds']

    return  pd.concat([pd.DataFrame(x) for x in rss_feeds if isinstance(x, list)], ignore_index=True)



if __name__ == "__main__":

    # Run the async main function
    result, graph = asyncio.run(main(db_path=Config.LANGGRAPH_PERSISTANT_DB_PATH))

    #load ALL rss feeds from ALL HISTORICAL runs 
    rss_feeds = load_rss_results(Config.LANGGRAPH_PERSISTANT_DB_PATH)
    feeds = [x for x in rss_feeds['url'].values if isinstance(x, str) and x.startswith('http')]
    #NEXT STEP: iterate over each RSS feed , extract the payload and save to sqlite db
    #generate proxy headers
    header_generator = HeaderGenerator()
    headers = header_generator()

    #try and parse RSS feeds with traditional feedparser (i think its working good ebough)
    feed_data = read_multiple_rss_feeds(feeds)





