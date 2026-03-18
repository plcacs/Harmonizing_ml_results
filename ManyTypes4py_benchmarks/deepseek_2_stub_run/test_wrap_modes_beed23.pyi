```python
from typing import Any, List

def _wrap_mode_interface(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...

def noqa(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...

def backslash_grid(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...

def grid(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...

def hanging_indent(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...

def hanging_indent_with_parentheses(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...

def vertical(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...

def vertical_grid(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...

def vertical_grid_grouped(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...

def vertical_hanging_indent(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...

def vertical_hanging_indent_bracket(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...

def vertical_prefix_from_module_import(
    statement: str,
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> str: ...
```