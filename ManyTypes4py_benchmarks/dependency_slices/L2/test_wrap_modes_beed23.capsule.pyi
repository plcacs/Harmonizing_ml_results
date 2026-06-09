from typing import Any

# === Third-party dependency: hypothesis ===
# Used symbols: given, reject, strategies

# === Internal dependency: isort ===
# re-export: from .api import sort_code_string as code

# === Internal dependency: isort.wrap_modes ===
def _wrap_mode_interface(statement: str, imports: List[str], white_space: str, indent: str, line_length: int, comments: List[str], line_separator: str, comment_prefix: str, include_trailing_comma: bool, remove_comments: bool) -> str: ...
def noqa(**interface: Any) -> str: ...

# === Third-party dependency: pytest ===
# Used symbols: mark