from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import requests

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from datetime import datetime 

def get_date() -> str:
    """
    Returns the current date and time.
    Returns:
        str: Current date and time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

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
    
tools = [add, multiply, divide,get_weather,get_date]

# Define LLM with bound tools
#llm = ChatOpenAI(model="gpt-4o")
llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
    max_tokens=2000,
    streaming=True,
    verbose=True,
)

#bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs or fetching the weather for a user defined location.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()
