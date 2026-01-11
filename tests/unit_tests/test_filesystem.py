"""Test the filesystem tools."""

import os
from unittest.mock import patch, MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent.tools import filesystem as fs
from agent.tools.filesystem import FSOperation


def test_fsoperation_validation() -> None:
    """Test the validation of the FSOperation."""
    with pytest.raises(ValueError):
        FSOperation(operation="read", args={})

    with pytest.raises(ValueError):
        FSOperation(operation="write", args={})

    with pytest.raises(ValueError):
        FSOperation(operation="patch", args={})

    with pytest.raises(ValueError):
        FSOperation(operation="search", args={})

    with pytest.raises(ValueError):
        FSOperation(operation="glob", args={})

    with pytest.raises(ValueError):
        FSOperation(operation="list", args={})

    with pytest.raises(ValueError):
        FSOperation(
            operation="read",
            args={"path": "", "query": "", "glob_pattern": "", "content": ""},
        )

    with pytest.raises(ValueError):
        FSOperation(operation="write", args={"path": "", "content": ""})

    with pytest.raises(ValueError):
        FSOperation(operation="patch", args={"path": "", "content": ""})

    with pytest.raises(ValueError):
        FSOperation(operation="search", path="", query="")

    with pytest.raises(ValueError):
        FSOperation(operation="replace", args={})

    op = FSOperation(
        operation="read", args={"path": "test.txt", "read_offset": 1, "read_length": 1}
    )
    assert op.operation == "read"

    op = FSOperation(
        operation="write",
        args={"path": "test.txt", "content": "test content", "write_append": True},
    )
    assert op.operation == "write"

    op = FSOperation(
        operation="patch", args={"path": "test.txt", "content": "test content"}
    )
    assert op.operation == "patch"

    op = FSOperation(operation="search", args={"query": "test"})
    assert op.operation == "search"

    op = FSOperation(operation="glob", args={"glob_pattern": "test.txt"})
    assert op.operation == "glob"

    op = FSOperation(operation="list", args={"path": "test.txt"})
    assert op.operation == "list"

    op = FSOperation(operation="delete", args={"path": "test.txt"})
    assert op.operation == "delete"

    op = FSOperation(
        operation="replace",
        args={
            "replace_pattern": "test",
            "glob_pattern": "test.txt",
            "content": "test content",
        },
    )
    assert op.operation == "replace"


@pytest.fixture
def root_temp_path():
    """Root temporary path."""
    with TemporaryDirectory() as temp_dir:
        cwd = os.getcwd()
        os.chdir(temp_dir)
        yield Path(".")
        os.chdir(cwd)


def test_all_fs_opt(root_temp_path: Path) -> None:
    """Test the all filesystem operations."""

    # list empty directory
    op = FSOperation(operation="list", path=root_temp_path)
    assert fs.list_file(op) == f"Directory: {root_temp_path}\n(empty)"

    # write first test file
    op = FSOperation(
        operation="write",
        path=root_temp_path / "test1.txt",
        content="test content 1",
        write_append=False,
    )
    assert fs.write_file(op) == "Content written to file 'test1.txt'"

    # list the directory again
    op = FSOperation(operation="list", path=root_temp_path)
    result = fs.list_file(op)
    assert result.startswith(f"Directory: {root_temp_path}\n")
    assert "test1.txt" in result

    # read the test file
    op = FSOperation(
        operation="read",
        path=root_temp_path / "test1.txt",
        read_offset=0,
        read_length=10,
    )
    assert fs.read_file(op) == "test content 1"

    # patch the test file
    patch_content = """--- test1.txt       2026-01-11 10:53:58.947771242 +0000
+++ test1.txt       2026-01-11 10:54:09.072830904 +0000
@@ -1 +1 @@
-test content 1
\ No newline at end of file
+test content 1 patched
\ No newline at end of file
"""
    op = FSOperation(
        operation="patch",
        path=root_temp_path / "test1.txt",
        content=patch_content,
    )
    assert fs.patch_file(op) == "Content patched to file 'test1.txt'"

    # replace the test file
    op = FSOperation(
        operation="replace",
        path=root_temp_path / "test1.txt",
        replace_pattern="test content 1 patched",
        glob_pattern="test1.txt",
        content="test content 1 replaced",
    )
    result = fs.replace_file(op)
    assert result.startswith("Replacements made in 1 file(s):")
    assert "Replacements made in 1 file(s):\n'test1.txt': 1 replacement(s)" == result

    # read the test file again
    op = FSOperation(
        operation="read",
        path=root_temp_path / "test1.txt",
        read_offset=0,
        read_length=10,
    )
    result = fs.read_file(op)
    # Replace replaces all matches, so result should contain the replacement
    assert "test content 1 replaced" == result

    # delete the test file
    op = FSOperation(operation="delete", path=root_temp_path / "test1.txt")
    assert fs.delete_file(op) == "File 'test1.txt' deleted successfully"

    # list the directory again
    op = FSOperation(operation="list", path=root_temp_path)
    assert fs.list_file(op) == f"Directory: {root_temp_path}\n(empty)"

    # write second test file
    op = FSOperation(
        operation="write",
        path=root_temp_path / "test2.txt",
        content="test content 2",
        write_append=False,
    )
    assert fs.write_file(op) == "Content written to file 'test2.txt'"
    # write third test file

    op = FSOperation(
        operation="write",
        path=root_temp_path / "test3.txt",
        content="test content 3",
        write_append=False,
    )
    assert fs.write_file(op) == "Content written to file 'test3.txt'"

    # glob file
    op = FSOperation(operation="glob", glob_pattern="test*.txt")
    assert fs.glob_file(op) == "test2.txt\ntest3.txt"

    # search content in the files
    op = FSOperation(operation="search", query="test content")
    result = fs.search_file(op)
    # search_file returns full paths, check that both files are present
    assert "test2.txt:1: test content 2\ntest3.txt:1: test content 3" == result
