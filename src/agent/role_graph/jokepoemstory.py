"""Joke, story and poem workflow."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph import MessagesState
from langchain.messages import SystemMessage
from langchain.messages import HumanMessage
from langgraph.types import interrupt
from langgraph.types import Command
from typing import Literal

from agent.agent import create_custom_agent


class State(MessagesState):
    """State for the joke, story and poem workflow."""

    topic: str = ""


def catch_topic(state: State) -> Command[Literal["continue_node", END]]:
    """Catch the topic from the user."""
    llm = create_custom_agent()
    msg = llm.invoke(
        [SystemMessage(content="从用户输入中提取对话主题。")] + state["messages"]
    )
    answer = interrupt(f"是否开始生成有关 {msg.content} 的故事、笑话和诗歌？(y/n)")
    if answer == "y":
        return Command(update={"topic": msg.content}, goto="continue_node")
    else:
        return Command(goto=END)


def continue_node(state: State):
    """Continue the node."""
    return {"messages": [HumanMessage(content="请根据对话主题生成故事、笑话和诗歌。")]}


# Nodes
def call_llm_1(state: State):
    """First LLM call to generate initial joke."""
    llm = create_custom_agent()

    msg = llm.invoke(f"写一个关于 {state['topic']} 的笑话")
    return {"messages": [msg]}


def call_llm_2(state: State):
    """Second LLM call to generate story."""
    llm = create_custom_agent()

    msg = llm.invoke(f"写一个关于 {state['topic']} 的故事")
    return {"messages": [msg]}


def call_llm_3(state: State):
    """Third LLM call to generate poem."""
    llm = create_custom_agent()

    msg = llm.invoke(f"写一首关于 {state['topic']} 的诗")
    return {"messages": [msg]}


def aggregator(state: State):
    """Combine the joke, story and poem into a single output."""
    combined = f"这是一个关于 {state['topic']} 的故事、笑话和诗歌！\n\n"
    combined += f"故事：\n{state['messages'][1].content}\n\n"
    combined += f"笑话：\n{state['messages'][2].content}\n\n"
    combined += f"诗歌：\n{state['messages'][3].content}"
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("catch_topic", catch_topic)
parallel_builder.add_node("continue_node", continue_node)
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes
parallel_builder.add_edge(START, "catch_topic")
parallel_test = True

if parallel_test:
    parallel_builder.add_edge("continue_node", "call_llm_1")
    parallel_builder.add_edge("continue_node", "call_llm_2")
    parallel_builder.add_edge("continue_node", "call_llm_3")
    parallel_builder.add_edge("call_llm_1", "aggregator")
    parallel_builder.add_edge("call_llm_2", "aggregator")
    parallel_builder.add_edge("call_llm_3", "aggregator")
    parallel_builder.add_edge("aggregator", END)
else:
    parallel_builder.add_edge("continue_node", "call_llm_1")
    parallel_builder.add_edge("call_llm_1", "call_llm_2")
    parallel_builder.add_edge("call_llm_2", "call_llm_3")
    parallel_builder.add_edge("call_llm_3", "aggregator")
    parallel_builder.add_edge("aggregator", END)

parallel_builder.add_edge("aggregator", END)
graph = parallel_builder.compile()
