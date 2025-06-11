from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGragh
from langchain_core.messages import ToolMessage # passes data back to LLM after it calls a tool such as the content of a file
from langchain_core.messages import SystemMessage # provides instructions to the LLM about how to behave
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv  # for storing API keys in .env file

load_dotenv()  

class AgentState(TypedDict):
    """State of the agent, containing messages."""
    messages : Annotated[Sequence[BaseMessage], add_messages] #A sequence of messages, which can include human, AI, and tool messages
    
@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """multiplication function."""
    return a * b

@tool
def subtract_numbers(a: int, b: int) -> int:
    """subtraction function."""
    return a - b


tools = [add_numbers, multiply_numbers, subtract_numbers] #tools created

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    """Calls the model with the current state and returns the updated state."""
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my querry to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> bool:
    """Determines whether to continue the conversation based on the last message."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools = tools) #adding tools
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            
input_message = {"messages": [("user", "add 40 + 2 and then substract 2. and also tell me joke after that")]} #input message to the agent
print_stream(app.stream(input_message, stream_mode="values"))
