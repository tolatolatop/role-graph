"""Filesystem tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

operation_types = Literal["read", "write", "delete", "search", "glob", "patch", "list"]

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

    glob_pattern: str | None = Field(
        None,
        description="The pattern to glob the file or directory. Required for glob operation.",
    )

    content: str | None = Field(
        None,
        description="The content to write/patch. Required for write and patch operations.",
    )

    @model_validator(mode="after")
    def validate_required_fields(self) -> FSOperation:
        """Validate required fields based on operation."""
        op = self.operation

        if op in ("read", "write", "delete", "patch", "list") and not self.path:
            raise ValueError(f"path is required for '{op}' operation")

        if op == "search" and not self.query:
            raise ValueError("query is required for 'search' operation")

        if op == "glob" and not self.glob_pattern:
            raise ValueError("glob_pattern is required for 'glob' operation")

        if op in ("write", "patch") and self.content is None:
            # 允许 content=""（空字符串）作为合法内容，因此用 is None 判断
            raise ValueError(f"content is required for '{op}' operation")

        return self
