"""Wizard for the agent."""

from typing import Literal

from langchain.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.agent import create_custom_agent
from agent.role.context import UserContext
from agent.tools import tools_by_name


# Nodes
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not."""
    llm_with_tools = create_custom_agent()

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: dict):
    """Perform the tool call."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
wizard_builder = StateGraph(MessagesState, context_schema=UserContext)

# Add nodes
wizard_builder.add_node("llm_call", llm_call)
wizard_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
wizard_builder.add_edge(START, "llm_call")
wizard_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
wizard_builder.add_edge("tool_node", "llm_call")

# Compile the agent
graph = wizard_builder.compile()
