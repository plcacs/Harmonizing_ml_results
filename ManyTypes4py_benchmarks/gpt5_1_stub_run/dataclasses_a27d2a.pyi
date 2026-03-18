from typing import Any, Callable, ContextManager, Generator, List, Optional, Type, TypeVar, overload

__all__: List[str] = ...

_T = TypeVar("_T")


@overload
def dataclass(
    _cls: None = ...,
    *,
    init: bool = ...,
    repr: bool = ...,
    eq: bool = ...,
    order: bool = ...,
    unsafe_hash: bool = ...,
    frozen: bool = ...,
    config: Any = ...,
    validate_on_init: Optional[bool] = ...,
    use_proxy: Optional[bool] = ...,
    kw_only: bool = ...,
) -> Callable[[Type[_T]], Any]: ...
@overload
def dataclass(
    _cls: Type[_T],
    *,
    init: bool = ...,
    repr: bool = ...,
    eq: bool = ...,
    order: bool = ...,
    unsafe_hash: bool = ...,
    frozen: bool = ...,
    config: Any = ...,
    validate_on_init: Optional[bool] = ...,
    use_proxy: Optional[bool] = ...,
    kw_only: bool = ...,
) -> Any: ...


def set_validation(cls: Any, value: Any) -> ContextManager[Any]: ...
def create_pydantic_model_from_dataclass(dc_cls: Any, config: Any = ..., dc_cls_doc: Optional[str] = ...) -> Any: ...
def is_builtin_dataclass(_cls: Any) -> bool: ...
def make_dataclass_validator(dc_cls: Any, config: Any) -> Generator[Any, None, None]: ...


class DataclassProxy:
    __dataclass__: Any
    def __init__(self, dc_cls: Any) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, __name: str, __value: Any) -> None: ...
    def __instancecheck__(self, instance: Any) -> bool: ...
    def __copy__(self) -> "DataclassProxy": ...
    def __deepcopy__(self, memo: Any) -> "DataclassProxy": ...