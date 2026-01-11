"""Tools for the agent."""

from .filesystem import fs_opt
from .user import get_user_id, get_user_email

__all__ = ["fs_opt", "get_user_id", "get_user_email"]

tools = [get_user_id, get_user_email, fs_opt]

tools_by_name = {tool.name: tool for tool in tools}
