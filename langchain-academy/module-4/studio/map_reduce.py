import operator, os
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, field_validator

from langgraph.constants import Send
from langgraph.graph import END, StateGraph, START

from langchain_anthropic import ChatAnthropic

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"


# config inputs 
_root = "/home/zjc1002/Mounts/code/"
_langsmith_trace = 'true'
_langsmith_project = 'zjc_custom_v1'
_anthropic_model = 'claude-3-haiku-20240307'
_temparate = 0
_max_tokens = 4000
_streaming = True
_system_message = None
_db_path = "/home/zjc1002/Mounts/data/langgraph_memory/state_db/example1.db"
# This is a basic config to enable tracing of conversations and  utilization of memory
_config = {'configurable': {'thread_id':"1"}}


# Prompts we will use
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}. Only output the list of sub-topics, no additional information and do not include the overall topic in the list. The list must be only 3 elements long"""

#this prompt is used to generate a joke about a specific subject(subtopic)
joke_prompt = """Generate a joke about {subject}"""

best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

# LLM
model =  ChatAnthropic(
    model=_anthropic_model,
    temperature=_temparate,
    #max_tokens=_max_tokens,
    #streaming=_streaming,
    verbose=True,
)

class Subjects(BaseModel):
    subjects: list[str]
    
    @field_validator('subjects')
    def validate_subjects_length(cls, v):
        if len(v) != 3:
            raise ValueError('subjects must contain exactly 3 elements')
        return v

class BestJoke(BaseModel):
    id: int
    
class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


class JokeState(TypedDict):
    subject: str
    joke_prompt: str

class Joke(BaseModel):
    joke: str


#TOPIC GENERATION GRAPH
#NOTE: with_structured_output used to enforce the output of the model to be a specific type
def generate_topics(state: OverallState):
    
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
   
    #return response key aligned to the Subjects model and the overall state
    return {"subjects": response.subjects}

# Parllell generation of jokes for each subject using send()
def continue_to_jokes(state: OverallState):
    return [Send(
        "generate_joke"
            , {"subject": s
            }
           ) for s in state["subjects"]]


def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])

    #enforce  joke output to be a string
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}


def best_joke(state: OverallState):

    #jokes in annotated list generated from  each parallel joke generation
    jokes = "\n\n".join(state["jokes"])
    
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)

    # enforce best joke output to be an integer ID
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}


#build the graph 
# Construct the graph: here we put everything together to construct our graph
graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")

#pass topics to SEND() via continue_to_jokes mapper function, and then use the SEND() function to pass each subject to the generate_joke node
graph.add_conditional_edges(
    "generate_topics"
    , continue_to_jokes
    , ["generate_joke"])

# pass the list of  jokes generated to the best_joke node
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)

# Compile the graph
app = graph.compile()
