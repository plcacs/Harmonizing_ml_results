from typing import Any, Callable, Mapping, Optional, Type, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
D = TypeVar("D", bound=Callable[..., Any])
T = TypeVar("T")

cache_readonly: Any = ...

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
    mapping: Optional[Mapping[Any, Any] | Callable[[Any], Any]] = ...,
    stacklevel: int = ...,
) -> Callable[[F], F]: ...

def deprecate_nonkeyword_arguments(
    version: Optional[str],
    allowed_args: Optional[list[str]] = ...,
    name: Optional[str] = ...,
) -> Callable[[F], F]: ...

def doc(*docstrings: Any, **params: Any) -> Callable[[D], D]: ...

class Substitution:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...

class Appender:
    def __init__(self, addendum: str, join: str = ..., indents: int = ...) -> None: ...
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]: ...

def indent(text: Any, indents: int = ...) -> str: ...

def future_version_msg(version: Optional[str]) -> str: ...

def set_module(module: Optional[str]) -> Callable[[T], T]: ...

__all__: list[str] = ...