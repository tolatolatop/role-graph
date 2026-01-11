"""Wizard for the agent."""

from langchain.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.agent import create_custom_agent
from agent.role.context import UserContext
from agent.tools import tools


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


# Build workflow
wizard_builder = StateGraph(MessagesState, context_schema=UserContext)

# Add nodes
wizard_builder.add_node("llm_call", llm_call)
wizard_builder.add_node("tools", ToolNode(tools))

# Add edges to connect nodes
wizard_builder.add_edge(START, "llm_call")
wizard_builder.add_edge("llm_call", END)
wizard_builder.add_conditional_edges("llm_call", tools_condition)
wizard_builder.add_edge("tools", "llm_call")

# Compile the agent
graph = wizard_builder.compile()
