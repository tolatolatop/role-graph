"""Agent for the application."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.tools import tools


def create_custom_agent(output_model: BaseModel | None = None):
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
    if output_model:
        return llm.with_structured_output(output_model, tools=tools)
    else:
        return llm.bind_tools(tools)
