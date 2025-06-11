import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv  # for storing API keys in .env file

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    """This node processes the messages and returns a response"""
    response = llm.invoke(state["messages"])
    
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])
    
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

coversation_history = []

user_input = input("Enter your message here: ")
while user_input != "exit":
    coversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": coversation_history})
    conversation_history = result["messages"]
    user_input = input("\nEnter your message here: ")
    
with open("conversation_history.txt", "w") as f:
    f.write("Conversation History:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n\n")
    f.write("End of conversation history.")
            
print("Conversation history saved to conversation_history.txt")
