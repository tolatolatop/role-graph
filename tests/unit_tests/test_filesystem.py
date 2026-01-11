"""Test the filesystem tools."""

import pytest

from agent.tools.filesystem import FSOperation


def test_fsoperation_validation() -> None:
    """Test the validation of the FSOperation."""
    with pytest.raises(ValueError):
        FSOperation(operation="read", path=None)

    with pytest.raises(ValueError):
        FSOperation(operation="write", content=None)

    with pytest.raises(ValueError):
        FSOperation(operation="patch", content=None)

    with pytest.raises(ValueError):
        FSOperation(operation="search", query=None)

    with pytest.raises(ValueError):
        FSOperation(operation="glob", glob_pattern=None)

    with pytest.raises(ValueError):
        FSOperation(operation="list", path=None)

    with pytest.raises(ValueError):
        FSOperation(operation="read", path="", query="", glob_pattern="", content="")

    with pytest.raises(ValueError):
        FSOperation(operation="write", path="", content="")

    with pytest.raises(ValueError):
        FSOperation(operation="patch", path="", content="")

    with pytest.raises(ValueError):
        FSOperation(operation="search", path="", query="")

    op = FSOperation(operation="read", path="test.txt")
    assert op.operation == "read"

    op = FSOperation(operation="write", path="test.txt", content="test content")
    assert op.operation == "write"

    op = FSOperation(operation="patch", path="test.txt", content="test content")
    assert op.operation == "patch"

    op = FSOperation(operation="search", query="test")
    assert op.operation == "search"

    op = FSOperation(operation="glob", glob_pattern="test.txt")
    assert op.operation == "glob"

    op = FSOperation(operation="list", path="test.txt")
    assert op.operation == "list"

    op = FSOperation(operation="delete", path="test.txt")
    assert op.operation == "delete"
