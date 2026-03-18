from typing import Any, Callable, Mapping, Optional, Type, TypeVar, Union

F = TypeVar("F", bound=Callable[..., Any])
U = TypeVar("U")

cache_readonly: Any

def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: Optional[str] = ...,
    klass: Optional[Type[Warning]] = ...,
    stacklevel: int = ...,
    msg: Optional[str] = ...,
) -> Callable[..., Any]: ...

def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: Optional[str],
    mapping: Optional[Union[Mapping[Any, Any], Callable[[Any], Any]]] = ...,
    stacklevel: int = ...,
) -> Callable[[F], F]: ...

def _format_argument_list(allow_args: list[str]) -> str: ...

def future_version_msg(version: Optional[str]) -> str: ...

def deprecate_nonkeyword_arguments(
    version: Optional[str],
    allowed_args: Optional[list[str]] = ...,
    name: Optional[str] = ...,
) -> Callable[[F], F]: ...

def doc(*docstrings: Any, **params: Any) -> Callable[[U], U]: ...

class Substitution:
    params: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, func: F) -> F: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...

class Appender:
    def __init__(self, addendum: str, join: str = ..., indents: int = ...) -> None: ...
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]: ...

def indent(text: str, indents: int = ...) -> str: ...

__all__: list[str]

def set_module(module: Optional[str]) -> Callable[[U], U]: ...