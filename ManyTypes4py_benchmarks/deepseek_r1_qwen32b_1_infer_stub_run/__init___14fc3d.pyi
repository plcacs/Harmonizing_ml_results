"""
The API basically only provides one class. You can create a :class:`Script` and
use its methods.

Additionally you can add a debug function with :func:`set_debug_function`.
Alternatively, if you don't need a custom function and are happy with printing
debug messages to stdout, simply call :func:`set_debug_function` without
arguments.
"""

import sys
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    Any,
    AnyStr,
    overload,
)
from jedi.api.classes import (
    Completion,
    Name,
    Signature,
    SyntaxError,
    Refactoring,
)
from jedi.api.environment import InterpreterEnvironment
from jedi.api.project import Project

class Script:
    """
    A Script is the base for completions, goto or whatever you want to do with
    Jedi. The counter part of this class is :class:`Interpreter`, which works
    with actual dictionaries and can work with a REPL. This class
    should be used when a user edits code in an editor.
    """

    def __init__(self, code: Optional[str] = None, *, path: Optional[Union[str, Path]] = None, environment: Optional[InterpreterEnvironment] = None, project: Optional[Project] = None) -> None:
        ...

    def complete(self, line: int = ..., column: int = ..., *, fuzzy: bool = False) -> List[Completion]:
        """
        Completes objects under the cursor.
        """
        ...

    def infer(self, line: int = ..., column: int = ..., *, only_stubs: bool = False, prefer_stubs: bool = False) -> List[Name]:
        """
        Return the definitions of under the cursor.
        """
        ...

    def goto(self, line: int = ..., column: int = ..., *, follow_imports: bool = False, follow_builtin_imports: bool = False, only_stubs: bool = False, prefer_stubs: bool = False) -> List[Name]:
        """
        Goes to the name that defined the object under the cursor.
        """
        ...

    def search(self, string: str, *, all_scopes: bool = False) -> Generator[Name, None, None]:
        """
        Searches a name in the current file.
        """
        ...

    def complete_search(self, string: str, **kwargs: Any) -> Generator[Completion, None, None]:
        """
        Like :meth:`.Script.search`, but completes that string.
        """
        ...

    def help(self, line: int = ..., column: int = ...) -> List[Name]:
        """
        Used to display a help window to users.
        """
        ...

    def get_references(self, line: int = ..., column: int = ..., **kwargs: Any) -> List[Name]:
        """
        Lists all references of a variable in a project.
        """
        ...

    def get_signatures(self, line: int = ..., column: int = ...) -> List[Signature]:
        """
        Return the function object of the call under the cursor.
        """
        ...

    def get_context(self, line: int = ..., column: int = ...) -> Name:
        """
        Returns the scope context under the cursor.
        """
        ...

    def get_names(self, **kwargs: Any) -> List[Name]:
        """
        Returns names defined in the current file.
        """
        ...

    def get_syntax_errors(self) -> List[SyntaxError]:
        """
        Lists all syntax errors in the current file.
        """
        ...

    def rename(self, line: int = ..., column: int = ..., *, new_name: str) -> Refactoring:
        """
        Renames all references of the variable under the cursor.
        """
        ...

    def extract_variable(self, line: int, column: int, *, new_name: str, until_line: Optional[int] = None, until_column: Optional[int] = None) -> Refactoring:
        """
        Moves an expression to a new statement.
        """
        ...

    def extract_function(self, line: int, column: int, *, new_name: str, until_line: Optional[int] = None, until_column: Optional[int] = None) -> Refactoring:
        """
        Moves an expression to a new function.
        """
        ...

    def inline(self, line: int = ..., column: int = ...) -> Refactoring:
        """
        Inlines a variable under the cursor.
        """
        ...

class Interpreter(Script):
    """
    Jedi's API for Python REPLs.
    """

    def __init__(self, code: str, namespaces: List[Dict[str, Any]], *, project: Optional[Project] = None, **kwds: Any) -> None:
        ...

def preload_module(*modules: str) -> None:
    """
    Preloading modules tells Jedi to load a module now, instead of lazy parsing
    of modules. This can be useful for IDEs, to control which modules to load
    on startup.
    """
    ...

def set_debug_function(func_cb: Optional[Callable[[str], None]] = debug.print_to_stdout, warnings: bool = True, notices: bool = True, speed: bool = True) -> None:
    """
    Define a callback debug function to get all the debug messages.
    """
    ...