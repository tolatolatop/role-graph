from unittest.mock import patch, MagicMock

from langchain.tools import ToolRuntime
from agent.tools import user as user_tools


def test_get_user_email_with_runtime() -> None:
    """Test the get_user_email tool with runtime."""
    runtime = MagicMock(spec=ToolRuntime)
    runtime.context = {"user_id": "test_user"}
    runtime.state = MagicMock()
    runtime.config = MagicMock()
    runtime.stream_writer = MagicMock()
    runtime.tool_call_id = "test_tool_call_id"
    runtime.store = MagicMock()
    assert (
        user_tools.get_user_email.invoke(
            {
                "user_id": "test_user",
                "other_args": {},
                "runtime": runtime,
            }
        )
        == "test_user@example.com"
    )
