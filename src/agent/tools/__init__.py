"""Tools for the agent."""

from .filesystem import fs_opt
from .user import get_user_id

__all__ = ["fs_opt", "get_user_id"]

tools = [get_user_id]

tools_by_name = {tool.name: tool for tool in tools}
