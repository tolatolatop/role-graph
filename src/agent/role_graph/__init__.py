"""LangGraph graph for the agent."""

from .standalone import graph as standalone_graph
from .wizard import graph as wizard_graph

__all__ = ["standalone_graph", "wizard_graph"]
