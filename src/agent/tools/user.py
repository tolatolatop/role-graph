"""User tools."""

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel

from agent.role.context import UserContext


class GetUserEmailArgs(BaseModel):
    """Arguments for the get_user_email tool."""

    user_id: str
    other_args: dict


@tool
def get_user_id(runtime: ToolRuntime[UserContext]) -> str:
    """Get the user ID."""
    return runtime.context["user_id"]


@tool(args_schema=GetUserEmailArgs)
def get_user_email(user_id: str, other_args: dict, runtime: ToolRuntime) -> str:
    """Get the user email."""
    return f"{user_id}@example.com"
