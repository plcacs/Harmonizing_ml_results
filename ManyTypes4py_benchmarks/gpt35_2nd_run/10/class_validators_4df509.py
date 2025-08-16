from typing import Any, Callable, Union, TypeVar, overload
from warnings import warn

_ALLOW_REUSE_WARNING_MESSAGE: str = '`allow_reuse` is deprecated and will be ignored; it should no longer be necessary'

def validator(__field: str, *fields: str, pre: bool = False, each_item: bool = False, always: bool = False, check_fields: bool = None, allow_reuse: bool = False) -> Callable:
    ...

@overload
def root_validator(*, skip_on_failure: bool, allow_reuse: bool = ...) -> None:
    ...

@overload
def root_validator(*, pre: bool, allow_reuse: bool = ...) -> None:
    ...

@overload
def root_validator(*, pre: bool, skip_on_failure: bool, allow_reuse: bool = ...) -> None:
    ...

def root_validator(*__args, pre: bool = False, skip_on_failure: bool = False, allow_reuse: bool = False) -> Callable:
    ...
