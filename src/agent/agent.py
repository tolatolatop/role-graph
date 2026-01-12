"""Agent for the application."""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from agent.tools import tools


def create_custom_agent():
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
    return llm.bind_tools(tools)
