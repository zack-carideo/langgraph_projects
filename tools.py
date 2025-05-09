import requests
from langchain_core.tools import tool
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