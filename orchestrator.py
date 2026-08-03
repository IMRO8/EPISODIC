import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 1. Initialize Free Model
llm = ChatOpenAI(
    model="deepseek-v4-flash-free",
    base_url="https://opencode.ai/zen/v1",
    api_key=os.environ.get("OPENCODE_API_KEY"),
    temperature=0,
)

# 2. State & Graph Definition
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# 3. Test memory context
config = {"configurable": {"thread_id": "session_user_123"}}

# Turn 1
input_1 = {"messages": [HumanMessage(content="Hi! My name is Rohith and I'm working on an ERP system.")]}
output_1 = graph.invoke(input_1, config)
print("Assistant:", output_1["messages"][-1].content)

print("\n" + "="*50 + "\n")

# Turn 2
input_2 = {"messages": [HumanMessage(content="What is my name and what project am I working on?")]}
output_2 = graph.invoke(input_2, config)
print("Assistant:", output_2["messages"][-1].content)