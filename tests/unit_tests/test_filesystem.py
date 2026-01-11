"""Test the filesystem tools."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent.tools import filesystem as fs
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

    with pytest.raises(ValueError):
        FSOperation(operation="replace", replace_pattern=None)

    op = FSOperation(operation="read", path="test.txt", read_offset=1, read_length=1)
    assert op.operation == "read"

    op = FSOperation(
        operation="write", path="test.txt", content="test content", write_append=True
    )
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

    op = FSOperation(
        operation="replace",
        replace_pattern="test",
        glob_pattern="test.txt",
        content="test content",
    )
    assert op.operation == "replace"


@pytest.fixture
def root_temp_path():
    """
    只是一个测试专用的简化上下文，在正式使用时需要配合其他方法保证路径正确。
    """
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
    # op = FSOperation(
    #     operation="patch",
    #     path=root_temp_path / "test1.txt",
    #     content="test content 1 patched",
    # )
    # assert fs.patch_file(op) == "Content patched to file 'test1.txt'"

    # replace the test file
    op = FSOperation(
        operation="replace",
        path=root_temp_path / "test1.txt",
        replace_pattern="test content 1",
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
