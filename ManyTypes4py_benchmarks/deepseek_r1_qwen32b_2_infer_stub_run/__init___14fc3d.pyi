"""
Stub file for jedi.api module
"""

from typing import (
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    Dict,
    AnyStr,
    Iterable,
    Set,
    Type,
    TypeVar,
    overload,
)
from pathlib import Path
from jedi.api.classes import (
    Name,
    Completion,
    Signature,
    Refactoring,
    SyntaxError,
)
from jedi.api.keywords import KeywordName
from jedi.api.project import Project
from jedi.api.environment import InterpreterEnvironment
from jedi.api.refactoring import RefactoringError

T = TypeVar("T")

class Script:
    """
    Base class for Jedi API
    """
    def __init__(self, code: str = None, *, path: Union[str, Path, None] = None, environment: Optional[InterpreterEnvironment] = None, project: Optional[Project] = None) -> None:
        ...

    @validate_line_column
    def complete(self, line: int = None, column: int = None, *, fuzzy: bool = False) -> List[Completion]:
        ...

    @validate_line_column
    def infer(self, line: int = None, column: int = None, *, only_stubs: bool = False, prefer_stubs: bool = False) -> List[Name]:
        ...

    @validate_line_column
    def goto(self, line: int = None, column: int = None, *, follow_imports: bool = False, follow_builtin_imports: bool = False, only_stubs: bool = False, prefer_stubs: bool = False) -> List[Name]:
        ...

    def search(self, string: str, *, all_scopes: bool = False) -> Generator[Name, None, None]:
        ...

    def complete_search(self, string: str, **kwargs: Any) -> List[Completion]:
        ...

    @validate_line_column
    def help(self, line: int = None, column: int = None) -> List[Name]:
        ...

    @validate_line_column
    def get_references(self, line: int = None, column: int = None, **kwargs: Any) -> List[Name]:
        ...

    @validate_line_column
    def get_signatures(self, line: int = None, column: int = None) -> List[Signature]:
        ...

    @validate_line_column
    def get_context(self, line: int = None, column: int = None) -> Name:
        ...

    def get_names(self, **kwargs: Any) -> List[Name]:
        ...

    def get_syntax_errors(self) -> List[SyntaxError]:
        ...

    def rename(self, line: int = None, column: int = None, *, new_name: str) -> Refactoring:
        ...

    @validate_line_column
    def extract_variable(self, line: int, column: int, *, new_name: str, until_line: int = None, until_column: int = None) -> Refactoring:
        ...

    @validate_line_column
    def extract_function(self, line: int, column: int, *, new_name: str, until_line: int = None, until_column: int = None) -> Refactoring:
        ...

    def inline(self, line: int = None, column: int = None) -> Refactoring:
        ...

class Interpreter(Script):
    """
    Jedi's API for Python REPLs
    """
    def __init__(self, code: str, namespaces: List[Dict[str, Any]], *, project: Optional[Project] = None, **kwds: Any) -> None:
        ...

def preload_module(*modules: str) -> None:
    ...

def set_debug_function(func_cb: Callable[[str], None] = debug.print_to_stdout, warnings: bool = True, notices: bool = True, speed: bool = True) -> None:
    ...