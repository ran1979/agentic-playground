import json
import urllib.parse
import urllib.request
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# need to run 1st 'pip install langgraph langchain-ollama'

# 1. DEFINE TOOLS USING LANGCHAIN DECORATOR
@tool
def get_live_weather(location: str) -> str:
    """
    Fetch real-time weather information for any given city.

    Args:
        location: The city or region name, e.g. "Tel Aviv", "Tokyo", "Kfar Saba"
    """
    try:
        encoded_location = urllib.parse.quote(location)
        url = f"https://wttr.in/{encoded_location}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status != 200:
                return f"Error: Unable to reach weather service (status {response.status})."

            data = json.loads(response.read().decode("utf-8"))
            current = data["current_condition"][0]
            temp_c = current["temp_C"]
            condition = current["weatherDesc"][0]["value"]
            humidity = current["humidity"]
            wind_kmh = current["windspeedKmph"]

            return f"Current weather in {location}: {temp_c}°C, {condition}. Humidity: {humidity}%, Wind: {wind_kmh} km/h."

    except Exception as e:
        return f"Failed to retrieve weather data for {location}: {str(e)}"


# 2. DEFINE THE STATE SCHEMA
class AgentState(TypedDict):
    # `add_messages` automatically appends new messages rather than overwriting the list
    messages: Annotated[list[BaseMessage], add_messages]


# 3. SETUP MODEL & BIND TOOLS
tools = [get_live_weather]
llm = ChatOllama(model="llama3.2:latest", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# 4. DEFINE GRAPH NODES
def call_model(state: AgentState) -> dict:
    """Node: Invokes the LLM with the current list of messages."""
    response = llm_with_tools.invoke(state["messages"])
    # Returning a list under key "messages" gets appended to AgentState automatically
    return {"messages": [response]}


# 5. BUILD THE GRAPH
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools)) # Prebuilt node that executes tool calls automatically

# Add Edges
workflow.set_entry_point("agent")

# `tools_condition` inspects the last message:
# -> If the LLM output `tool_calls`, route to "tools" node
# -> If no `tool_calls`, route to END
workflow.add_conditional_edges("agent", tools_condition)

# Loop back: after tools execute, return control to the agent node
workflow.add_edge("tools", "agent")

# Compile the execution graph
app = workflow.compile()


# 6. RUN THE AGENT
if __name__ == "__main__":
    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful assistant with access to real-time tools."),
            HumanMessage(content="What is the weather right now in Tokyo?")
        ]
    }

    print("--- Executing LangGraph Agent Loop ---")
    for event in app.stream(initial_state):
        for node_name, output in event.items():
            print(f"\n[Node Executed]: {node_name}")
            latest_msg = output["messages"][-1]
            
            if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                print(f"  Tool Calls Requested: {latest_msg.tool_calls}")
            elif latest_msg.type == "tool":
                print(f"  Tool Result: {latest_msg.content}")
            else:
                print(f"  Agent Output: {latest_msg.content}")