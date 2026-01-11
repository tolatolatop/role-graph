"""Filesystem tools."""

import glob as glob_module
import multiprocessing as mp
import os
import re
import subprocess as sp
from pathlib import Path
from typing import Literal

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field, field_validator, model_validator

from agent.role.context import UserContext

operation_types = Literal[
    "read", "write", "delete", "search", "glob", "patch", "list", "replace"
]

NEED_WRITE_PERMISSION_TYPES = ("write", "patch", "replace")

operation_type_descriptions = """The operation to perform. One of:
read: Read the content of the file.
write: Write the content to the file.
delete: Delete the file.
search: Search content in the file.
glob: Glob the file path or directory path.
patch: Patch the content of the file.
list: List recursively the file or directory.
"""


class FSOperationArgs(BaseModel):
    """Filesystem operation."""

    path: str | None = Field(
        None,
        description="The path to the file or directory. Required for read, write, delete, patch, list operations.",
    )

    query: str | None = Field(
        None,
        description="Regex pattern to search in the file. Required for search operation.",
    )

    replace_pattern: str | None = Field(
        None,
        description="Regex pattern to replace in the file. Required for replace operation.",
    )

    glob_pattern: str | None = Field(
        None,
        description="The pattern to glob or replace in the file or directory. Required for glob and replace operations.",
    )

    content: str | None = Field(
        None,
        description="The content to write/patch/replace. Required for write and patch operations.",
    )

    read_offset: int | None = Field(
        None,
        description="The offset line number to read from the file. Required for read operation.",
    )

    read_length: int | None = Field(
        None,
        description="The length of lines to read from the file. Required for read operation.",
    )

    write_append: bool | None = Field(
        None,
        description="Whether to append to the file. **Required for write operation.**",
    )

    @field_validator("path", mode="before")
    def validate_path(cls, v) -> str:
        """Validate path."""
        if isinstance(v, Path):
            return v.as_posix()
        return v


class FSOperation(BaseModel):
    """Filesystem operation."""

    operation: operation_types = Field(..., description=operation_type_descriptions)
    args: FSOperationArgs = Field(
        ..., description="The arguments for the filesystem operation."
    )

    @model_validator(mode="after")
    def validate(self):
        """Validate required fields based on operation."""
        op = self.operation

        if op in ("read", "write", "delete", "patch", "list") and not self.args.path:
            raise ValueError(f"path is required for '{op}' operation")

        if op == "read" and (
            self.args.read_offset is None or self.args.read_length is None
        ):
            raise ValueError(
                "read_offset and read_length are required for 'read' operation"
            )

        if op == "write" and self.args.write_append is None:
            raise ValueError("write_append is required for 'write' operation")

        if op == "search" and not self.args.query:
            raise ValueError("query is required for 'search' operation")

        if op in ("glob", "replace") and not self.args.glob_pattern:
            raise ValueError("glob_pattern is required for 'glob' operation")

        if op in ("write", "patch", "replace") and self.args.content is None:
            # 允许 content=""（空字符串）作为合法内容，因此用 is None 判断
            raise ValueError(f"content is required for '{op}' operation")

        if op == "replace" and self.args.replace_pattern is None:
            raise ValueError("replace_pattern is required for 'replace' operation")

        return self


@tool(args_schema=FSOperation)
def fs_opt(
    operation: operation_types,
    args: dict,
    runtime: ToolRuntime[UserContext],
) -> str:
    """Perform a filesystem operation."""
    if runtime is None:
        raise ValueError("runtime is required")
    operation = FSOperation(
        operation=operation,
        args=args,
    )
    path = Path("user_data") / runtime.context["user_id"]
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return mp.Pool(processes=1).apply(_fs_opt, (operation, path))


def _fs_opt(operation: FSOperation, cwd: Path) -> str:
    """Perform a filesystem operation."""
    os.chdir(cwd)
    opt_map = {
        "read": read_file,
        "write": write_file,
        "delete": delete_file,
        "search": search_file,
        "glob": glob_file,
        "patch": patch_file,
        "list": list_file,
        "replace": replace_file,
    }
    return opt_map[operation.operation](operation)


def read_file(operation: FSOperation) -> str:
    """Read the content of the file."""
    file_path = Path(operation.args.path)
    if not file_path.exists():
        return f"Error: File '{operation.args.path}' does not exist"

    try:
        with file_path.open("r", encoding="utf-8") as f:
            for _ in range(operation.args.read_offset):
                f.readline()
            content = f.readlines(operation.args.read_length)
            return "\n".join(content)
    except Exception as e:
        return f"Error reading file '{operation.args.path}': {str(e)}"


def write_file(operation: FSOperation) -> str:
    """Write the content to the file."""
    file_path = Path(operation.args.path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if operation.args.write_append else "w"
        with file_path.open(mode, encoding="utf-8") as f:
            f.write(operation.args.content)
        action = "appended to" if operation.args.write_append else "written to"
        return f"Content {action} file '{operation.args.path}'"
    except Exception as e:
        return f"Error writing file '{operation.args.path}': {str(e)}"


def delete_file(operation: FSOperation) -> str:
    """Delete the file."""
    file_path = Path(operation.args.path)
    if not file_path.exists():
        return f"Error: File '{operation.args.path}' does not exist"

    try:
        if file_path.is_file():
            file_path.unlink()
            return f"File '{operation.args.path}' deleted successfully"
        elif file_path.is_dir():
            import shutil

            shutil.rmtree(file_path)
            return f"Directory '{operation.args.path}' deleted successfully"
        else:
            return f"Error: '{operation.args.path}' is neither a file nor a directory"
    except Exception as e:
        return f"Error deleting '{operation.args.path}': {str(e)}"


def search_file(operation: FSOperation) -> str:
    """Search the content of the file."""
    if not operation.args.path:
        # 如果没有指定路径，在当前工作目录递归搜索
        search_path = Path(".")
        results = []
        try:
            pattern = re.compile(operation.args.query)
            for file_path in search_path.rglob("*"):
                if file_path.is_file():
                    try:
                        with file_path.open(
                            "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            for line_num, line in enumerate(f, 1):
                                if pattern.search(line):
                                    results.append(
                                        f"{file_path}:{line_num}: {line.strip()}"
                                    )
                    except Exception:
                        continue
        except re.error as e:
            return f"Error: Invalid regex pattern '{operation.args.query}': {str(e)}"
        except Exception as e:
            return f"Error searching files: {str(e)}"

        if results:
            return "\n".join(results)
        else:
            return f"No matches found for pattern '{operation.args.query}'"
    else:
        # 在指定文件中搜索
        file_path = Path(operation.args.path)
        if not file_path.exists():
            return f"Error: File '{operation.args.path}' does not exist"

        try:
            pattern = re.compile(operation.args.query)
            matches = []
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if pattern.search(line):
                        matches.append(f"{line_num}: {line.strip()}")
            if matches:
                return "\n".join(matches)
            else:
                return f"No matches found for pattern '{operation.args.query}' in '{operation.args.path}'"
        except re.error as e:
            return f"Error: Invalid regex pattern '{operation.args.query}': {str(e)}"
        except Exception as e:
            return f"Error searching file '{operation.args.path}': {str(e)}"


def glob_file(operation: FSOperation) -> str:
    """Glob the file."""
    try:
        matches = sorted(glob_module.glob(operation.args.glob_pattern, recursive=True))
        if matches:
            return "\n".join(matches)
        else:
            return f"No files matched pattern '{operation.args.glob_pattern}'"
    except Exception as e:
        return f"Error globbing pattern '{operation.args.glob_pattern}': {str(e)}"


def patch_file(operation: FSOperation) -> str:
    """Patch the content of the file."""
    file_path = Path(operation.args.path)
    if not file_path.exists():
        return f"Error: File '{operation.args.path}' does not exist"

    try:
        # patch 操作通常是在文件末尾追加内容
        cmd = ["patch", "-p", "0"]
        proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.STDOUT)
        proc.stdin.write(operation.args.content.encode("utf-8"))
        proc.stdin.close()
        stdout, _ = proc.communicate(timeout=10)
        if proc.returncode != 0:
            return f"Error: Failed to patch file '{operation.args.path}': {stdout.decode('utf-8', errors='ignore')}"
        return f"Content patched to file '{operation.args.path}'"
    except Exception as e:
        return f"Error patching file '{operation.args.path}': {str(e)}"


def list_file(operation: FSOperation) -> str:
    """List the content of the file."""
    file_path = Path(operation.args.path)
    if not file_path.exists():
        return f"Error: Path '{operation.args.path}' does not exist"

    try:
        if file_path.is_file():
            return (
                f"File: {operation.args.path}\nSize: {file_path.stat().st_size} bytes"
            )
        elif file_path.is_dir():
            items = []

            def _list_recursive(path: Path, prefix: str = "") -> None:
                """Recursively list directory contents."""
                try:
                    entries = sorted(path.iterdir())
                    for i, item in enumerate(entries):
                        is_last = i == len(entries) - 1
                        current_prefix = "└── " if is_last else "├── "
                        full_prefix = prefix + current_prefix

                        if item.is_file():
                            size = item.stat().st_size
                            items.append(f"{full_prefix}{item.name} ({size} bytes)")
                        elif item.is_dir():
                            items.append(f"{full_prefix}{item.name}/")
                            next_prefix = prefix + ("    " if is_last else "│   ")
                            _list_recursive(item, next_prefix)
                        else:
                            items.append(f"{full_prefix}{item.name} (?)")
                except PermissionError:
                    items.append(f"{prefix}[Permission denied]")

            _list_recursive(file_path)
            if items:
                return f"Directory: {operation.args.path}\n" + "\n".join(items)
            else:
                return f"Directory: {operation.args.path}\n(empty)"
        else:
            return f"Error: '{operation.args.path}' is neither a file nor a directory"
    except Exception as e:
        return f"Error listing '{operation.args.path}': {str(e)}"


def replace_file(operation: FSOperation) -> str:
    """Replace the content of the file."""
    try:
        # 使用 glob_pattern 查找文件
        matched_files = sorted(
            glob_module.glob(operation.args.glob_pattern, recursive=True)
        )
        if not matched_files:
            return f"No files matched pattern '{operation.args.glob_pattern}'"

        pattern = re.compile(operation.args.replace_pattern)
        results = []
        total_replacements = 0

        for file_path_str in matched_files:
            file_path = Path(file_path_str)
            if not file_path.is_file():
                continue

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    original_content = f.read()

                new_content, count = pattern.subn(
                    operation.args.content, original_content
                )

                if count > 0:
                    with file_path.open("w", encoding="utf-8") as f:
                        f.write(new_content)
                    results.append(f"'{file_path_str}': {count} replacement(s)")
                    total_replacements += count
                else:
                    results.append(f"'{file_path_str}': No matches found")
            except Exception as e:
                results.append(f"'{file_path_str}': Error - {str(e)}")

        if total_replacements > 0:
            return (
                f"Replacements made in {len([r for r in results if 'replacement(s)' in r])} file(s):\n"
                + "\n".join(results)
            )
        else:
            return "No replacements made:\n" + "\n".join(results)
    except re.error as e:
        return (
            f"Error: Invalid regex pattern '{operation.args.replace_pattern}': {str(e)}"
        )
    except Exception as e:
        return f"Error replacing in files: {str(e)}"
