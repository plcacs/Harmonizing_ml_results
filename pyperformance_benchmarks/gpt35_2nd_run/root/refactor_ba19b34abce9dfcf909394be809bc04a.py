from typing import List, Dict, Union, Any

class RefactoringTool:
    _default_options: Dict[str, bool]
    CLASS_PREFIX: str
    FILE_PREFIX: str

    def __init__(self, fixer_names: List[str], options: Dict[str, Any] = None, explicit: List[str] = None) -> None:
        ...

    def get_fixers(self) -> Tuple[List[Any], List[Any]]:
        ...

    def log_error(self, msg: str, *args, **kwds) -> None:
        ...

    def log_message(self, msg: str, *args) -> None:
        ...

    def log_debug(self, msg: str, *args) -> None:
        ...

    def print_output(self, old_text: str, new_text: str, filename: str, equal: bool) -> None:
        ...

    def refactor(self, items: List[str], write: bool = False, doctests_only: bool = False) -> None:
        ...

    def refactor_dir(self, dir_name: str, write: bool = False, doctests_only: bool = False) -> None:
        ...

    def _read_python_source(self, filename: str) -> Tuple[str, str]:
        ...

    def refactor_file(self, filename: str, write: bool = False, doctests_only: bool = False) -> None:
        ...

    def refactor_string(self, data: str, name: str) -> Any:
        ...

    def refactor_stdin(self, doctests_only: bool = False) -> None:
        ...

    def refactor_tree(self, tree: Any, name: str) -> bool:
        ...

    def traverse_by(self, fixers: Dict[int, List[Any]], traversal: Any) -> None:
        ...

    def processed_file(self, new_text: str, filename: str, old_text: str = None, write: bool = False, encoding: str = None) -> None:
        ...

    def write_file(self, new_text: str, filename: str, old_text: str, encoding: str = None) -> None:
        ...

    def summarize(self) -> None:
        ...

    def parse_block(self, block: List[str], lineno: int, indent: str) -> Any:
        ...

    def wrap_toks(self, block: List[str], lineno: int, indent: str) -> Any:
        ...

    def gen_lines(self, block: List[str], indent: str) -> Any:
        ...

class MultiprocessingUnsupported(Exception):
    pass

class MultiprocessRefactoringTool(RefactoringTool):

    def __init__(self, *args, **kwargs) -> None:
        ...

    def refactor(self, items: List[str], write: bool = False, doctests_only: bool = False, num_processes: int = 1) -> None:
        ...

    def _child(self) -> None:
        ...

    def refactor_file(self, *args, **kwargs) -> None:
        ...
