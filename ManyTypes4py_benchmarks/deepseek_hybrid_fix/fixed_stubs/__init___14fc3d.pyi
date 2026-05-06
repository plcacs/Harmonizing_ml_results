import sys
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

from jedi.api import classes
from jedi.api.environment import InterpreterEnvironment
from jedi.api.project import Project
from jedi.api.refactoring import Refactoring

# Re-export commonly used types
_Environment = Any  # jedi.api.environment.Environment
_InferenceState = Any  # jedi.inference.InferenceState
_ModuleNode = Any  # parso.python.tree.Module
_ModuleContext = Any  # jedi.inference.value.ModuleValue.as_context()
_Name = Any  # jedi.inference.base_value.BaseName
_Leaf = Any  # parso.python.tree.Leaf
_TreeName = Any  # parso.python.tree.Name
_Grammar = Any  # parso.grammar.Grammar

def preload_module(*modules: str) -> None:
    ...

def set_debug_function(
    func_cb: Callable[..., Any] = ...,
    warnings: bool = ...,
    notices: bool = ...,
    speed: bool = ...,
) -> None:
    ...

class Script:
    _orig_path: Optional[Union[str, Path]]
    path: Optional[Path]
    _inference_state: _InferenceState
    _module_node: _ModuleNode
    _code_lines: List[str]
    _code: Union[str, bytes]

    def __init__(
        self,
        code: Optional[Union[str, bytes]] = ...,
        *,
        path: Optional[Union[str, Path]] = ...,
        environment: Optional[_Environment] = ...,
        project: Optional[Project] = ...,
    ) -> None:
        ...

    def _get_module(self) -> Any:  # jedi.inference.value.ModuleValue
        ...

    def _get_module_context(self) -> _ModuleContext:
        ...

    def __repr__(self) -> str:
        ...

    @overload
    def complete(self, *, fuzzy: bool = ...) -> List[classes.Completion]:
        ...

    @overload
    def complete(
        self, line: int, column: int, *, fuzzy: bool = ...
    ) -> List[classes.Completion]:
        ...

    def complete(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        fuzzy: bool = ...,
    ) -> List[classes.Completion]:
        ...

    @overload
    def infer(
        self,
        *,
        only_stubs: bool = ...,
        prefer_stubs: bool = ...,
    ) -> List[classes.Name]:
        ...

    @overload
    def infer(
        self,
        line: int,
        column: int,
        *,
        only_stubs: bool = ...,
        prefer_stubs: bool = ...,
    ) -> List[classes.Name]:
        ...

    def infer(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        only_stubs: bool = ...,
        prefer_stubs: bool = ...,
    ) -> List[classes.Name]:
        ...

    @overload
    def goto(
        self,
        *,
        follow_imports: bool = ...,
        follow_builtin_imports: bool = ...,
        only_stubs: bool = ...,
        prefer_stubs: bool = ...,
    ) -> List[classes.Name]:
        ...

    @overload
    def goto(
        self,
        line: int,
        column: int,
        *,
        follow_imports: bool = ...,
        follow_builtin_imports: bool = ...,
        only_stubs: bool = ...,
        prefer_stubs: bool = ...,
    ) -> List[classes.Name]:
        ...

    def goto(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        follow_imports: bool = ...,
        follow_builtin_imports: bool = ...,
        only_stubs: bool = ...,
        prefer_stubs: bool = ...,
    ) -> List[classes.Name]:
        ...

    def search(
        self, string: str, *, all_scopes: bool = ...
    ) -> Iterator[classes.Name]:
        ...

    def _search_func(
        self,
        string: str,
        all_scopes: bool = ...,
        complete: bool = ...,
        fuzzy: bool = ...,
    ) -> List[Union[classes.Completion, classes.Name]]:
        ...

    def complete_search(
        self,
        string: str,
        *,
        all_scopes: bool = ...,
        fuzzy: bool = ...,
    ) -> Iterator[classes.Completion]:
        ...

    @overload
    def help(self) -> List[classes.Name]:
        ...

    @overload
    def help(self, line: int, column: int) -> List[classes.Name]:
        ...

    def help(
        self, line: Optional[int] = ..., column: Optional[int] = ...
    ) -> List[classes.Name]:
        ...

    @overload
    def get_references(
        self,
        *,
        include_builtins: bool = ...,
        scope: str = ...,
    ) -> List[classes.Name]:
        ...

    @overload
    def get_references(
        self,
        line: int,
        column: int,
        *,
        include_builtins: bool = ...,
        scope: str = ...,
    ) -> List[classes.Name]:
        ...

    def get_references(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        include_builtins: bool = ...,
        scope: str = ...,
    ) -> List[classes.Name]:
        ...

    @overload
    def get_signatures(self) -> List[classes.Signature]:
        ...

    @overload
    def get_signatures(
        self, line: int, column: int
    ) -> List[classes.Signature]:
        ...

    def get_signatures(
        self, line: Optional[int] = ..., column: Optional[int] = ...
    ) -> List[classes.Signature]:
        ...

    @overload
    def get_context(self) -> classes.Name:
        ...

    @overload
    def get_context(self, line: int, column: int) -> classes.Name:
        ...

    def get_context(
        self, line: Optional[int] = ..., column: Optional[int] = ...
    ) -> classes.Name:
        ...

    def _analysis(self) -> List[Any]:  # jedi.api.errors.SyntaxError
        ...

    def get_names(
        self,
        *,
        all_scopes: bool = ...,
        definitions: bool = ...,
        references: bool = ...,
    ) -> List[classes.Name]:
        ...

    def get_syntax_errors(self) -> List[Any]:  # jedi.api.errors.SyntaxError
        ...

    def _names(
        self,
        all_scopes: bool = ...,
        definitions: bool = ...,
        references: bool = ...,
    ) -> List[_Name]:
        ...

    @overload
    def rename(self, *, new_name: str) -> Refactoring:
        ...

    @overload
    def rename(
        self, line: int, column: int, *, new_name: str
    ) -> Refactoring:
        ...

    def rename(
        self,
        line: Optional[int] = ...,
        column: Optional[int] = ...,
        *,
        new_name: str,
    ) -> Refactoring:
        ...

    def extract_variable(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: Optional[int] = ...,
        until_column: Optional[int] = ...,
    ) -> Refactoring:
        ...

    def extract_function(
        self,
        line: int,
        column: int,
        *,
        new_name: str,
        until_line: Optional[int] = ...,
        until_column: Optional[int] = ...,
    ) -> Refactoring:
        ...

    @overload
    def inline(self) -> Refactoring:
        ...

    @overload
    def inline(self, line: int, column: int) -> Refactoring:
        ...

    def inline(
        self, line: Optional[int] = ..., column: Optional[int] = ...
    ) -> Refactoring:
        ...

class Interpreter(Script):
    _allow_descriptor_getattr_default: bool
    namespaces: List[Dict[str, Any]]

    def __init__(
        self,
        code: str,
        namespaces: List[Dict[str, Any]],
        *,
        project: Optional[Project] = ...,
        environment: Optional[InterpreterEnvironment] = ...,
        path: Optional[Union[str, Path]] = ...,
    ) -> None:
        ...

    def _get_module_context(self) -> Any:  # jedi.api.interpreter.MixedModuleContext
        ...