import requests
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from datetime import datetime

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
        n_results (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    return TavilySearch(max_results=n_results)
