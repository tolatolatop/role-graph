"""User tools."""

from langchain.tools import ToolRuntime, tool

from agent.role.context import UserContext


@tool
def get_user_id(runtime: ToolRuntime[UserContext]) -> str:
    """Get the user ID."""
    return runtime.context["user_id"]
