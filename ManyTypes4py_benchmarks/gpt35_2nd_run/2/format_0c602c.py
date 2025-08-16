from pathlib import Path
from typing import Optional, TextIO

def show_unified_diff(*, file_input: str, file_output: str, file_path: Optional[Path], output: Optional[TextIO] = None, color_output: bool = False) -> None:
    ...

def ask_whether_to_apply_changes_to_file(file_path: str) -> bool:
    ...

def remove_whitespace(content: str, line_separator: str = '\n') -> str:
    ...
