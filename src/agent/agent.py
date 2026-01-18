"""Agent for the application."""

import os
from typing import Set

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.tools import tools


def create_custom_agent(
    output_model: BaseModel | None = None, use_tools: bool | Set[str] = True
):
    """Create an agent for a user."""
    load_dotenv()
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY")
    # )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    if isinstance(use_tools, Set):
        tools_to_use = [tool for tool in tools if tool.name in use_tools]
    elif use_tools:
        tools_to_use = tools

    if not use_tools:
        return llm
    if output_model:
        return llm.with_structured_output(output_model, tools=tools_to_use)
    else:
        return llm.bind_tools(tools_to_use)
