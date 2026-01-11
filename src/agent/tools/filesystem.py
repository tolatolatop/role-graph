"""Filesystem tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

operation_types = Literal[
    "read", "write", "delete", "search", "glob", "patch", "list", "replace"
]

operation_type_descriptions = """The operation to perform. One of:
read: Read the content of the file.
write: Write the content to the file.
delete: Delete the file.
search: Search content in the file.
glob: Glob the file path or directory path.
patch: Patch the content of the file.
list: List recursively the file or directory.
"""


class FSOperation(BaseModel):
    """Filesystem operation."""

    operation: operation_types = Field(..., description=operation_type_descriptions)

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
        description="The offset to read from the file. Required for read operation.",
    )

    read_length: int | None = Field(
        None,
        description="The length to read from the file. Required for read operation.",
    )

    write_append: bool | None = Field(
        None,
        description="Whether to append to the file. Required for write operation.",
    )

    @model_validator(mode="after")
    def validate_required_fields(self) -> FSOperation:
        """Validate required fields based on operation."""
        op = self.operation

        if op in ("read", "write", "delete", "patch", "list") and not self.path:
            raise ValueError(f"path is required for '{op}' operation")

        if op == "read" and (self.read_offset is None or self.read_length is None):
            raise ValueError(
                "read_offset and read_length are required for 'read' operation"
            )

        if op == "write" and self.write_append is None:
            raise ValueError("write_append is required for 'write' operation")

        if op == "search" and not self.query:
            raise ValueError("query is required for 'search' operation")

        if op in ("glob", "replace") and not self.glob_pattern:
            raise ValueError("glob_pattern is required for 'glob' operation")

        if op in ("write", "patch", "replace") and self.content is None:
            # 允许 content=""（空字符串）作为合法内容，因此用 is None 判断
            raise ValueError(f"content is required for '{op}' operation")

        if op == "replace" and self.replace_pattern is None:
            raise ValueError("replace_pattern is required for 'replace' operation")

        return self
