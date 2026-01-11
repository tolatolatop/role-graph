"""Tools for the agent."""

from .filesystem import fs_opt

__all__ = ["fs_opt"]

tools = [fs_opt]

tools_by_name = {tool.name: tool for tool in tools}
