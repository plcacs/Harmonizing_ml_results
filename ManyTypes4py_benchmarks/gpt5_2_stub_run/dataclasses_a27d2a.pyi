from typing import Any, ContextManager, Generator, List, overload
from typing_extensions import dataclass_transform

__all__: List[str] = ...

@overload
@dataclass_transform
def dataclass(
    *,
    init: bool = ...,
    repr: bool = ...,
    eq: bool = ...,
    order: bool = ...,
    unsafe_hash: bool = ...,
    frozen: bool = ...,
    config: Any = ...,
    validate_on_init: Any = ...,
    use_proxy: Any = ...,
    kw_only: bool = ...,
) -> Any: ...
@overload
@dataclass_transform
def dataclass(
    _cls: Any,
    *,
    init: bool = ...,
    repr: bool = ...,
    eq: bool = ...,
    order: bool = ...,
    unsafe_hash: bool = ...,
    frozen: bool = ...,
    config: Any = ...,
    validate_on_init: Any = ...,
    use_proxy: Any = ...,
    kw_only: bool = ...,
) -> Any: ...
@dataclass_transform
def dataclass(
    _cls: Any = ...,
    *,
    init: bool = ...,
    repr: bool = ...,
    eq: bool = ...,
    order: bool = ...,
    unsafe_hash: bool = ...,
    frozen: bool = ...,
    config: Any = ...,
    validate_on_init: Any = ...,
    use_proxy: Any = ...,
    kw_only: bool = ...,
) -> Any: ...

def set_validation(cls: Any, value: Any) -> ContextManager[Any]: ...

class DataclassProxy:
    def __init__(self, dc_cls: Any) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, __name: str, __value: Any) -> None: ...
    def __instancecheck__(self, instance: Any) -> bool: ...
    def __copy__(self) -> "DataclassProxy": ...
    def __deepcopy__(self, memo: Any) -> "DataclassProxy": ...

def _add_pydantic_validation_attributes(dc_cls: Any, config: Any, validate_on_init: Any, dc_cls_doc: Any) -> None: ...
def _get_validators(cls: Any) -> Generator[Any, None, None]: ...
def _validate_dataclass(cls: Any, v: Any) -> Any: ...
def create_pydantic_model_from_dataclass(dc_cls: Any, config: Any = ..., dc_cls_doc: Any = ...) -> Any: ...
def _is_field_cached_property(obj: Any, k: Any) -> bool: ...
def _dataclass_validate_values(self: Any) -> None: ...
def _dataclass_validate_assignment_setattr(self: Any, name: str, value: Any) -> None: ...
def is_builtin_dataclass(_cls: Any) -> bool: ...
def make_dataclass_validator(dc_cls: Any, config: Any) -> Generator[Any, None, None]: ...