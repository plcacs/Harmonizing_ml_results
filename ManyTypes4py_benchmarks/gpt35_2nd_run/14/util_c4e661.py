from typing import Any, Optional, List, Tuple

class TestCaseArgs:
    mode: black.Mode
    fast: bool = False
    minimum_version: Optional[Tuple[int, int]] = None
    lines: List[int] = []
    no_preview_line_length_1: bool = False

def _assert_format_equal(expected: str, actual: str) -> None:
    ...

def assert_format(source: str, expected: str, mode: black.Mode, *, fast: bool = False, minimum_version: Optional[Tuple[int, int]] = None, lines: List[int] = [], no_preview_line_length_1: bool = False) -> None:
    ...

def _assert_format_inner(source: str, expected: Optional[str], mode: black.Mode, *, fast: bool = False, minimum_version: Optional[Tuple[int, int]] = None, lines: List[int] = []) -> None:
    ...

def dump_to_stderr(*output: str) -> str:
    ...

def get_base_dir(data: bool) -> Path:
    ...

def all_data_cases(subdir_name: str, data: bool) -> List[str]:
    ...

def get_case_path(subdir_name: str, name: str, data: bool, suffix: str = PYTHON_SUFFIX) -> Path:
    ...

def read_data_with_mode(subdir_name: str, name: str, data: bool) -> Tuple[black.Mode, str, str]:
    ...

def read_data(subdir_name: str, name: str, data: bool) -> Tuple[str, str]:
    ...

def _parse_minimum_version(version: str) -> Tuple[int, int]:
    ...

def get_flags_parser() -> argparse.ArgumentParser:
    ...

def parse_mode(flags_line: str) -> TestCaseArgs:
    ...

def read_data_from_file(file_name: str) -> Tuple[black.Mode, str, str]:
    ...

def read_jupyter_notebook(subdir_name: str, name: str, data: bool) -> str:
    ...

def read_jupyter_notebook_from_file(file_name: str) -> str:
    ...

def change_directory(path: str) -> None:
    ...
