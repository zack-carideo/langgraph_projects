import requests, os , httpx
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from datetime import datetime
from typing import Dict, Any, Iterator , List 
from langchain_community.document_loaders import RSSFeedLoader, NewsURLLoader
from langchain_core.documents import Document
from fake_useragent import UserAgent
import asyncio
import logging
from random_header_generator import HeaderGenerator 
import aiohttp
import asyncio
import feedparser
from typing import List, Dict, Optional
from random_header_generator import HeaderGenerator

logger = logging.getLogger(__name__)

#local imports of custom rss handlers
from .rss_handlers import generate_headers, extract_rss_links_from_url_async, is_rss, is_valid_url, cancel_all_tasks

@tool
def get_weather(city: str) -> str:
    """
    Returns the current weather description for a given city using the wttr.in service.
    Args:
        city (str): The city name.
    Returns:
        str: Weather description or error message.
    """
    url = f"https://wttr.in/{city}?format=3"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.text.strip()
    except Exception as e:
        return f"Could not retrieve weather data: {e}"
    

@tool
def get_date() -> str:
    """
    Returns the current date and time.
    Returns:
        str: Current date and time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


@tool
def internet_search(query: str, n_results: int = 2): 
    """search internet for a query
    using TavilySearch
    This function is a wrapper around the TavilySearch class from langchain_tavily.
    It allows you to perform a search using the Tavily API.
    The function takes a query string and an optional number of results to return.

    Args:
        query (str): _description_
        n_results (int, optional): _description_. Defaults to 2.s

    Returns:
        _type_: _description_
    """
    return TavilySearch(max_results=n_results)


#this is new (testing)
class GoogleRSSDiscovery:
    """
    GoogleRSSDiscovery is a utility class for discovering RSS feeds related to a specific genre using the Google Custom Search API.
    
    This class searches for RSS feeds using Google Custom Search, then validates and extracts RSS links from the discovered pages.
    
    Attributes:
        api_key (str): Google Custom Search API key.
        cse_id (str): Custom Search Engine ID.
        base_url (str): Base URL for the Google Custom Search API.
        exclude_domains (list): List of domains to exclude from search results. Defaults to feedspot.com, feedvisor.com, and facebook.com.
        n_results (int): Maximum number of results to fetch per search query. Default is 3.
        max_links (int): Maximum number of RSS links to extract from each discovered page. Default is 10.

    Methods:
        search_rss_feeds(genre, num_results=10, exclude_domains=None): 
            Search for RSS feeds related to a specific genre.
            
        main_runner(url_meta_dict, max_links=2):
            Process discovered URLs to validate RSS feeds and extract RSS links from pages.

    Example:
        >>> api_key = "YOUR_GOOGLE_API_KEY"
        >>> cse_id = "YOUR_CUSTOM_SEARCH_ENGINE_ID"
        >>> rss_discovery = GoogleRSSDiscovery(api_key, cse_id, n_results=5, max_links=10)
        >>> search_results = await rss_discovery.search_rss_feeds("technology", num_results=5)
        >>> parsed_feeds = await rss_discovery.main_runner(search_results, max_links=10)
        >>> for result in parsed_feeds:
        ...     print(f"Page: {result['title']}")
        ...     for rss_link in result['rss_links']:
        ...         print(f"  RSS: {rss_link['title']} - {rss_link['url']}")
        
    """
    def __init__(self
                 , api_key:str
                 , cse_id:str
                 , n_results:int =3
                 , max_links: int = 10
                 , exclude_domains:list=None
                 ):
        
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.exclude_domains = ["https://www.feedspot.com","https://feedvisor.com", 'https://facebook.com'] if exclude_domains is None else exclude_domains
        self.n_results = n_results
        self.max_links = max_links
        
        """_summary_
        """
        # Log the initialization parameters
        for attr, value in self.__dict__.items():
            if attr not in ['api_key', 'cse_id']:  
                print(f"{attr}: {value}")


    async def search_rss_feeds(self
                         , genre: list or str
                         , num_results=10
                         , exclude_domains:list=None):
        
        """Search for RSS feeds related to a specific genre"""

        search_queries = [f'"{genre}" RSS feeds',] if isinstance(genre, str) else [f'{g} RSS feeds' for g in genre]

        if exclude_domains is None:
            exclude_domains = self.exclude_domains


        # Prepare exclusion string for query
        exclusion = ""
        if exclude_domains:
            for domain in exclude_domains:
                exclusion += f" -site:{domain}"

        all_results = []
        
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            
            for query in search_queries:
                full_query = query + exclusion
                logger.info(f"Searching for: {full_query}")
                try:
                    params = {
                        'key': self.api_key,
                        'cx': self.cse_id,
                        'q': full_query,
                        'num': min(num_results, self.n_results),
                    #    'sort': 'date',  # Sort by date (most recent first)
                    #    'dateRestrict': 'm6'  # Limit results to last 3 months
                    }
                    response = await client.get(self.base_url, params=params)
                    response.raise_for_status()
                    data = response.json()

                    for item in data.get('items', []):
                        all_results.append({
                            'title': item.get('title'),
                            'url': item.get('link'),
                            'snippet': item.get('snippet'),
                            'publisher':item.get('article:publisher'),
                            'search_query': query
                        })
                        
                except httpx.RequestError as e:
                    print(f"Error searching for '{query}': {e}")
                    continue

        
        return all_results



    async def main_runner(self, url_meta_dict: List[dict] ,  max_links: int = 2 
                        ) -> List[dict]:
            """
            Main function to discover RSS feeds based on a query.
            This function takes a list of dictionaries containing metadata about URLs,checks if each URL is an RSS feed, and if not, attempts to extract RSS links from the page.
            Args:
            url_meta_dict = [{
                            'title': <'title of webpage'>
                            ,'url': <'url to webpage'>
                            ,'snippet': <'summary of  content on webpage'>
                            ,'publisher': <'publisher of webpage'>
                            ,'search_query': <'query used to find this webpage'>
                        }, ....  , ]

            NOTE: The input url_meta_dict is generated via 
                #get payloads for google search reesults 
                    url_meta_dict = GoogleRSSDiscovery.search_rss_feeds(
                        query
                        , num_results=self.n_results
                        , exclude_domains=self.exclude_domains
                        )

            """
            
            # Extract all URLs from the response that may contain RSS feeds
            rss_candidate_urls = []
            used_feeds = []
            
            for _d in url_meta_dict:
                
                if _d['url'] in used_feeds:
                    logger.info(f"SKIPPING {_d['url']} it has  been processed.")
                    continue

                if not is_valid_url(_d['url']):  
                    logger.warning(f"Invalid URL IN MAIN RUNNER: {_d['url']}")
                    continue

        
                async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                    

                    try: 
                        response = await client.get(_d['url'], headers=generate_headers())
                    except: 
                        response = await client.head(_d['url'], headers=generate_headers())
                        
                # Check if the URL is already in the used feeds
                if is_rss(response):
                    logger.info(f"RSS feed found directly at {_d['url']}")
                    _d['rss_links'] = [{'url': _d['url'], 'title': _d['title']}]
                    rss_candidate_urls.append(_d)
                    used_feeds.append(_d['url'])
                    continue

                # If not, try to extract RSS links from the URL
                rss_links, parsed_urls = await extract_rss_links_from_url_async(_d['url'], headers=generate_headers(), max_links=max_links)

                # add the list of rss links found in the page to the dictionary (add a copy of the search query used to find this page)
                _d['rss_links'] = [_dd for _dd in rss_links if _dd['url'] not in used_feeds]
                
                # update the memory of used feeds 
                rss_candidate_urls.append(_d)
                used_feeds.extend(parsed_urls)
                used_feeds.extend([x['url'] for x in rss_links])
                logger.info(f"Extracted {len(rss_links)} RSS links from {_d['url']}")

            return rss_candidate_urls

    async def end2end_googler(self
                                , query_list: List[str]
                                ,  max_links: int = 2 
                            ) -> List[dict]:
        
        # Search for RSS feeds related to a specific genre
        searcher = await self.search_rss_feeds(
            genre = query_list 
            , num_results=5
            , exclude_domains=self.exclude_domains
        )

        # Process discovered URLs to validate RSS feeds and extract RSS links returned searcher results 
        results = await self.main_runner(searcher, max_links=max_links)

        return searcher, results



class CustomNewsURLLoader(NewsURLLoader):
    def __init__(self, urls, user_agent=None, **kwargs):
        super().__init__(urls, **kwargs)
        self.user_agent = UserAgent().random or user_agent
        logger.info(f"Using User-Agent: {self.user_agent}")

    def _fetch(self, url):
        #generate proxy headers
        header_generator = HeaderGenerator()
        headers = header_generator()

        # Make a HEAD request to check if the URL is valid
        resp = requests.head(url, allow_redirects=True, timeout=10, headers=headers)

        if resp.status_code in (404, 405):

            # Some servers don't support HEAD for RSS, try GET
            resp = requests.get(url, allow_redirects=True, timeout=10, headers=headers, stream=True)
        
        resp.raise_for_status()
        return resp.content  

# class CustomRSSFeedLoader(RSSFeedLoader):
#     """
#     Custom RSS Feed Loader that takes a single dictionary mapping URLs to their metadata.
#     Args:
#         url_metadata_dict: Dictionary where keys are URLs and values are metadata dicts.
#         **kwargs: Additional keyword arguments passed to the base class.

    
#     Example usage:
#         url_metadata_dict = {
#             "https://rss.nytimes.com/services/xml/rss/nyt/World.xml": {"source": "NYTimes", "category": "World","filter_phrases": ["subscri"]}

#             ,"https://feeds.bbci.co.uk/news/rss.xml": {"source": "BBC", "category": "News", "filter_phrases": ["subscribe", "premium"]}
#         }

#         loader = CustomRSSFeedLoader(url_metadata_dict=url_metadata_dict, nlp=True)trwi

#         for doc in loader.lazy_load():
#             print(doc.type)
#             print(f"Title: {doc.metadata.get('title', 'No Title')}")
#             print(f"Source: {doc.metadata.get('source')}")
#             print(f"Category: {doc.metadata.get('category')}")
#             print(f"Feed URL: {doc.metadata.get('feed')}")
#             print(f"Content: {doc.page_content[:100]}...\n")

#     """

#     def __init__(
#         self,
#         url_metadata_dict: Dict[str, Dict[str, Any]]
#         , headers: Dict[str, str] = None,
#         **kwargs,
#     ):
#         #generate random proxy headers
#         header_generator = HeaderGenerator()
#         self.headers = header_generator() or headers
#         self.url_metadata_dict = url_metadata_dict or {}
#         self.headers = header_generator() or headers
#         urls = list(self.url_metadata_dict.keys())
#         print(f"Proxy headers for CustomRSSLoader: {self.headers}")

#         # Initialize the base class with the URLs
#         super().__init__(urls=urls, **kwargs)

#     def _fetch(self, url: str):
#         headers = self.headers

#         resp = requests.head(url, allow_redirects=True, timeout=10, headers=headers)

#         if resp.status_code != 200:
#             # Some servers don't support HEAD for RSS, try GET
#             resp = requests.get(url, allow_redirects=True, timeout=10, headers=headers, stream=True)

        
#         resp.raise_for_status()
#         return resp.content

#     def lazy_load(self) -> Iterator[Document]:
#         try:
#             import feedparser
#         except ImportError:
#             raise ImportError(
#                 "feedparser package not found, please install it with "
#                 "`pip install feedparser`"
#             )

#         for url in self._get_urls:
#             custom_meta = self.url_metadata_dict.get(url, {})
#             try:
#                 feed = feedparser.parse(self._fetch(url))

#                 if getattr(feed, "bozo", False):
#                     raise ValueError(
#                         f"Error fetching {url}, exception: {feed.bozo_exception}"
#                     )
#             except Exception as e:
#                 if self.continue_on_failure:
#                     logger.error(f"Error fetching {url}, exception: {e}")
#                     continue
#                 else:
#                     raise e
#             try:
#                 for entry in feed.entries:
#                     loader = CustomNewsURLLoader(
#                         urls=[entry.link],
#                         **self.newsloader_kwargs,
#                     )

#                     article = loader.load()[0]
#                     article.metadata["feed"] = url
#                     article.metadata.update(custom_meta)
#                     yield article
#             except Exception as e:
#                 if self.continue_on_failure:
#                     logger.error(f"Error processing entry {entry.link}, exception: {e}")
#                     continue
#                 else:
#                     raise e
                



async def read_rss_async(url: str, session: aiohttp.ClientSession, headers: Optional[Dict] = None) -> Dict:
    """Async RSS feed reader with custom headers support"""
    try:
        if not headers: 
            header_generator = HeaderGenerator()
            headers = header_generator()

        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            content = await response.text()
            feed = feedparser.parse(content)

            return {
                'url': url,
                'status_code': response.status,
                'title': feed.feed.get('title', 'N/A'),
                'description': feed.feed.get('description', 'N/A'),
                'link': feed.feed.get('link', 'N/A'),
                'language': feed.feed.get('language', 'N/A'),
                'last_build_date': feed.feed.get('lastbuilddate', 'N/A'),
                'total_entries': len(feed.entries),
                'entries': [
                    {
                        'title': entry.get('title', 'N/A'),
                        'link': entry.get('link', 'N/A'),
                        'published': entry.get('published', 'N/A'),
                        'summary': entry.get('summary', 'N/A'),
                        'author': entry.get('author', 'N/A'),
                        'tags': [tag.term for tag in entry.get('tags', [])]
                    }
                    for entry in feed.entries[:10]  # Limit to 10 entries
                ]
            }
    except aiohttp.ClientError as e:
        return {
            'url': url,
            'error': f'HTTP Error: {str(e)}',
            'error_type': 'http_error',
            'entries': []
        }
    except Exception as e:
        return {
            'url': url,
            'error': str(e),
            'error_type': 'parse_error',
            'entries': []
        }

async def read_multiple_rss_feeds(urls: List[str], headers: Optional[Dict] = None, timeout: int = 30):
    """Read multiple RSS feeds concurrently with custom headers"""
    
    # Create session with timeout
    timeout_config = aiohttp.ClientTimeout(total=timeout)
    
    async with aiohttp.ClientSession(timeout=timeout_config) as session:
        tasks = [read_rss_async(url, session, headers) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
if __name__ == "__main__":
    
    """
    Simplified working RSS Hunter - Combines fixed discovery with minimal agent
    This is a complete, working example that you can run immediately.
    """

    import asyncio, httpx, logging, os, sys
    from typing import List, Dict, Any, Optional
    from operator import add
    from pydantic import BaseModel, Field
    from pathlib import Path
    # Add project root
    _root = "/home/zjc1002/Mounts/code/"
    sys.path.append(_root)

    sys.path.append(Path(_root) / "langgraph_projects" / "utils")
    
    from admin.api_keys import _api_keys
    from admin.sys_ops import _set_env
    from langgraph_projects.utils.tools import GoogleRSSDiscovery


    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

async    def _test(): 
        
        from admin.sys_ops import _set_env
        from admin.api_keys import _api_keys
        
        # Set up environment
        for key in ["GOOGLE_API_KEY", "GOOGLE_CSE_ID", "ANTHROPIC_API_KEY"]:
            _set_env(key, _api_keys.get(key, ""))


        ### ===== GOOGLE RSS DISCOVERY CLASS =====
        google_search = GoogleRSSDiscovery(
            api_key=os.environ.get("GOOGLE_API_KEY"),
            cse_id=os.environ.get("GOOGLE_CSE_ID")
        )


        queries = ['Financial Regulatory Institutions', 'Financial Regulatory Agencies', 'Financial Regulatory Bodies']


        results = await google_search.search_rss_feeds(queries, num_results=5)

        #results = await asyncio.gather(*tasks, return_exceptions=True)

        search_results = await google_search.search_rss_feeds("Financial Regulatory Institutions", num_results=5)


        parsed_details = await google_search.main_runner(results, max_links =100)


asyncio.run(_test())
















