import sys
import logging
import asyncio
import re
from typing import List
from functools import wraps
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from random_header_generator import HeaderGenerator 
import httpx 
import requests
import validators

def is_valid_url(url):
    return validators.url(url) is True

def async_timeout(timeout_seconds: float, default_return=None):
    """Decorator to add timeout to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(f"{func.__name__} timed out after {timeout_seconds} seconds")
                return default_return
        return wrapper
    return decorator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)

# Import API Keys (stored in a separate file so I run into less issues with git)
_root = "/home/zjc1002/Mounts/code/"
sys.path.append(_root)
from admin.api_keys import _api_keys
from admin.sys_ops import _set_env


def generate_headers():
    """Generate headers for HTTP requests."""

    try:
        #generate proxy headers
        header_generator = HeaderGenerator()
        headers = header_generator()

    except ImportError:
        # Fallback to basic headers if fake_headers is not available
        headers = {
        "User-Agent": UserAgent().random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive"
        }
    return headers

    

def define_content_type(content_type):
    """Determine the appropriate BeautifulSoup parser based on content type.
    # Usage
    await cancel_all_tasks()
    """

    if any(sub in content_type for sub in ["xml", "rss", "atom"]):
        return 'xml'
    elif 'html' in content_type:
        return 'html.parser'
    else:
        return None


# ===== HELPER FUNCTIONS =====
def is_rss(resp: requests.Response) -> bool:

    """Check if the response content is an RSS/Atom feed based on content-type and root tag.""" 

    content_type = resp.headers.get("Content-Type", "").lower()
    logger.debug(f"Checking content type: {content_type}")
    if any(sub in content_type.lower() for sub in ["xml","feed" , "rss","atom"]):        
        
        logger.debug(f"LOOKS LIKE AN RSS FEED: {content_type}")
        soup = BeautifulSoup(resp.content, define_content_type(content_type))
        root_tag = soup.find().name.lower() if soup.find() else ""
        logger.debug(f"Root tag found: {root_tag}")
        if root_tag in ["rss", "feed","xml"]:
            return True
    else: 
        return False
    
async def cancel_all_tasks():
    """Cancel all currently running tasks except the current one."""
    current_task = asyncio.current_task()
    tasks = [task for task in asyncio.all_tasks() if task is not current_task]
    
    if not tasks:
        print("No tasks to cancel")
        return
    
    print(f"Cancelling {len(tasks)} tasks...")
    for task in tasks:
        task.cancel()
    
    # Wait for all tasks to be cancelled
    await asyncio.gather(*tasks, return_exceptions=True)
    print("All tasks cancelled")

async def async_get_url_payload(url: str, timeout: int = 5 , headers:dict = None
                    ) -> dict:
    
    """Check if URL is valid by making HTTP request.

    url = 'https://americangerman.institute/by-program/business-economics/feed/'
 

    get_url_payload(url,  headers = generate_headers())
    """
    
    try:
        
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response_method= 'head'
            response = await client.head(url
                                    , follow_redirects=True
                                    )

            #if HEAD fails, try GET
            if ((response.status_code !=200) or len(response.content) <10):
                response_method= 'get'
                response = await client.get(url
                                        , follow_redirects=True
                                        ,)
            
            #verify that the response is valid
            response.raise_for_status()

        return {'url':url
                ,'response_code':response.status_code
                , 'response_method': response_method
                , 'response': response
                , 'content_type': response.headers.get("Content-Type", "").lower()
                , 'is_rss': is_rss(response)
                }
    
    except Exception:
        return {'url':url
                ,'response_code': None
                , 'response_method': None
                , 'response': None
                , 'content_type': None
                , 'is_rss': False
                }

@async_timeout(300.0, default_return=([], []))
async def link_parser(url:str,headers:dict,max_links: int = 400):

    """Parse links from a webpage and return a list of URLs."""
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:

        # Go to the parent URL and get the content
        logger.info(f"0) PARENT URL: {url}")
        resp = await client.get(url)
        resp.raise_for_status()

        #parse the content with BeautifulSoup
        content_type = define_content_type(resp.headers.get("Content-Type", "").lower())
        soup = BeautifulSoup(resp.content, content_type)

        # loop through all links on the page and check if they are RSS feeds
        rss_links = []
        urls_parsed = []
        links = soup.find_all(["link", "a"], href=True)
        
        link_pattern = re.compile(f'(feed|rss)', re.IGNORECASE)
        links = sorted(links, key=lambda x: 'rss' in x.get('href','').lower() or 'feed'  in x.get('href','').lower() , reverse=True)

        canidate_urls =[]
        for idx,link in enumerate(links[:max_links]):        
            
            #get href and text from the link
            href = link.get("href")
            link_text = link.get_text(strip=True)   

            if not href.startswith("http"):
                href = urljoin(url, href)
                logger.debug(f"ABSOLUTE HREF {link}:: => {href}")
            else: 
                logger.info(f"PROCESSING {str(idx)} LINK: {link}")
            
            if link not in urls_parsed: 
                urls_parsed.append(href)
                logger.debug(f"ADDED {href} to URLs to parse.")
            else: 
                continue

            if (link_pattern.search(href)  or link_pattern.search(link_text)):
                canidate_urls.append(href)
        


        # Process all canidates concurrently
        tasks = [async_get_url_payload(url, headers=generate_headers()) for url in canidate_urls]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        rss_links = [res for res in results if isinstance(res, dict)
                     and res.get('is_rss', False)
                     ]

        return urls_parsed, rss_links


    

async def extract_rss_links_from_url_async(url, headers: dict = None, max_links= 20):

    """Navigate to a URL and extract all RSS feed links found asynchronously."""
    
    parsed_urls = []
    if headers is None:
        headers = generate_headers()
    try:
        
        logger.info(f"LINK PARSER RUNNING FOR: {url}")
        urls_parsed, rss_links = await link_parser(url
                                                   , headers
                                                   , max_links=max_links)
        
        #sort to try and prioritize rss feeds first
        logger.info(f"RSS LINKS FOUND: {len(rss_links)}")
        logger.info(f"TOTAL URLS PARSED: {len(urls_parsed)}")
    
        #sort em 
        urls_parsed = sorted(urls_parsed
                             , key=lambda x: 'rss' in x.lower() or 'href' in x.lower()
                             )

        
        if len(urls_parsed) != 0:
            parsed_urls.extend(urls_parsed)
            return rss_links, parsed_urls
        
        #the big loop 
        #if you cant find any rss links, keep trying until you find some
        n_links = 0
        while ((len(rss_links) == 0) & (n_links < max_links) & (not all(x in parsed_urls for x in urls_parsed))

        ):
            logger.info(f"NO RSS LINKS FOUND AT PRIMARY URL:{url}")
            for url in urls_parsed:
                
                if url not in parsed_urls:

                    logger.info(f"SECONDARY PARSER: {url}")
                    new_urls, new_rss_links = await link_parser(url, headers, max_links=max_links)
                    
                    # Add new RSS links to the list
                    rss_links.extend(new_rss_links)
                    urls_parsed.extend(new_urls)
                    
                    # Update the list of URLs to parse
                    #urls_parsed.extend(new_urls)
                    parsed_urls.append(url)
                    n_links += 1
                
            
        return rss_links, parsed_urls
    
    except Exception as e:
        logger.error(f"Failed to extract RSS links from {url}: {e}")
        cancel_all_tasks()
        return [] , []




# async def main_runner(url_meta_dict: List[dict] ,  max_links: int = 2 
#                       ) -> List[dict]:
#         """
#         Main function to discover RSS feeds based on a query.
#         This function takes a list of dictionaries containing metadata about URLs,checks if each URL is an RSS feed, and if not, attempts to extract RSS links from the page.
#         Args:
#         url_meta_dict = [{
#                         'title': <'title of webpage'>
#                         ,'url': <'url to webpage'>
#                         ,'snippet': <'summary of  content on webpage'>
#                         ,'publisher': <'publisher of webpage'>
#                         ,'search_query': <'query used to find this webpage'>
#                     }, ....  , ]

#         NOTE: The input url_meta_dict is generated via 
#             #get payloads for google search reesults 
#                 url_meta_dict = GoogleRSSDiscovery.search_rss_feeds(
#                     query
#                     , num_results=self.n_results
#                     , exclude_domains=self.exclude_domains
#                     )

#         """
        
#         # Extract all URLs from the response that may contain RSS feeds
#         rss_candidate_urls = []
#         used_feeds = []
        
#         for _d in url_meta_dict:
            
#             if _d['url'] in used_feeds:
#                 logger.info(f"SKIPPING {_d['url']} it has  been processed.")
#                 continue

#             if not is_valid_url(_d['url']):  
#                 logger.warning(f"Invalid URL IN MAIN RUNNER: {_d['url']}")
#                 continue

     
#             async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                

#                 try: 
#                     response = await client.get(_d['url'], headers=generate_headers())
#                 except: 
#                     response = await client.head(_d['url'], headers=generate_headers())
                    
#             # Check if the URL is already in the used feeds
#             import pdb;pdb.set_trace(); 
#             if is_rss(response):
#                 logger.info(f"RSS feed found directly at {_d['url']}")
#                 _d['rss_links'] = [{'url': _d['url'], 'title': _d['title']}]
#                 rss_candidate_urls.append(_d)
#                 used_feeds.append(_d['url'])
#                 continue

#             # If not, try to extract RSS links from the URL
#             rss_links, parsed_urls = await extract_rss_links_from_url_async(_d['url'], headers=generate_headers(), max_links=max_links)

#             # add the list of rss links found in the page to the dictionary
#             _d['rss_links'] = [_dd for _dd in rss_links if _dd['url'] not in used_feeds]
            
#             # update the memory of used feeds 
#             rss_candidate_urls.append(_d)
#             used_feeds.extend(parsed_urls)
#             used_feeds.extend([x['url'] for x in rss_links])
#             logger.info(f"Extracted {len(rss_links)} RSS links from {_d['url']}")

#         return rss_candidate_urls


if __name__ == "__main__":


    # Example usage
    url = "https://americangerman.institute/by-program/business-economics/feed/"
    headers = generate_headers()
    
    # Run the async function
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(async_get_url_payload(url, headers=headers))
    
    print(result)
    
    # Example of running the main runner
    url_meta_dict = [{'title': 'Example Title', 'url': url, 'snippet': 'Example Snippet', 'publisher': 'Example Publisher', 'search_query': 'Example Query'}]
    rss_candidate_urls = loop.run_until_complete(main_runner(url_meta_dict))
    
    print(rss_candidate_urls)



