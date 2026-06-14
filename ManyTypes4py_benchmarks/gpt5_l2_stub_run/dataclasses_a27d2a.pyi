from typing import Any, Callable, ContextManager, Dict, Generator, List, Optional, Type, TypeVar, Union, overload
import dataclasses
import sys
from typing_extensions import dataclass_transform
from pydantic.v1.config import BaseConfig, ConfigDict
from pydantic.v1.fields import Field
from pydantic.v1.main import BaseModel

__all__: List[str] = ['dataclass', 'set_validation', 'create_pydantic_model_from_dataclass', 'is_builtin_dataclass', 'make_dataclass_validator']

_T = TypeVar('_T')

if sys.version_info >= (3, 10):

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(
        *,
        init: bool = ...,
        repr: bool = ...,
        eq: bool = ...,
        order: bool = ...,
        unsafe_hash: bool = ...,
        frozen: bool = ...,
        config: Optional[Union[ConfigDict, Type[object]]] = ...,
        validate_on_init: Optional[bool] = ...,
        use_proxy: Optional[bool] = ...,
        kw_only: bool = ...,
    ) -> Callable[[Type[_T]], Type[_T]]: ...
    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
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
        config: Optional[Union[ConfigDict, Type[object]]] = ...,
        validate_on_init: Optional[bool] = ...,
        use_proxy: Optional[bool] = ...,
        kw_only: bool = ...,
    ) -> Type[_T]: ...
    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    def dataclass(
        _cls: Optional[Type[_T]] = ...,
        *,
        init: bool = ...,
        repr: bool = ...,
        eq: bool = ...,
        order: bool = ...,
        unsafe_hash: bool = ...,
        frozen: bool = ...,
        config: Optional[Union[ConfigDict, Type[object]]] = ...,
        validate_on_init: Optional[bool] = ...,
        use_proxy: Optional[bool] = ...,
        kw_only: bool = ...,
    ) -> Any: ...
else:

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(
        *,
        init: bool = ...,
        repr: bool = ...,
        eq: bool = ...,
        order: bool = ...,
        unsafe_hash: bool = ...,
        frozen: bool = ...,
        config: Optional[Union[ConfigDict, Type[object]]] = ...,
        validate_on_init: Optional[bool] = ...,
        use_proxy: Optional[bool] = ...,
    ) -> Callable[[Type[_T]], Type[_T]]: ...
    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
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
        config: Optional[Union[ConfigDict, Type[object]]] = ...,
        validate_on_init: Optional[bool] = ...,
        use_proxy: Optional[bool] = ...,
    ) -> Type[_T]: ...
    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    def dataclass(
        _cls: Optional[Type[_T]] = ...,
        *,
        init: bool = ...,
        repr: bool = ...,
        eq: bool = ...,
        order: bool = ...,
        unsafe_hash: bool = ...,
        frozen: bool = ...,
        config: Optional[Union[ConfigDict, Type[object]]] = ...,
        validate_on_init: Optional[bool] = ...,
        use_proxy: Optional[bool] = ...,
    ) -> Any: ...

def set_validation(cls: Type[Any], value: bool) -> ContextManager[Type[Any]]: ...

class DataclassProxy:
    def __init__(self, dc_cls: Type[Any]) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, __name: str, __value: Any) -> None: ...
    def __instancecheck__(self, instance: object) -> bool: ...
    def __copy__(self) -> "DataclassProxy": ...
    def __deepcopy__(self, memo: Dict[int, Any]) -> "DataclassProxy": ...

def _add_pydantic_validation_attributes(
    dc_cls: Type[Any],
    config: Type[BaseConfig],
    validate_on_init: bool,
    dc_cls_doc: Optional[str],
) -> None: ...

def _get_validators(cls: Type[Any]) -> Generator[Callable[..., Any], None, None]: ...

def _validate_dataclass(cls: Type[Any], v: Any) -> Any: ...

def create_pydantic_model_from_dataclass(
    dc_cls: Type[Any],
    config: Type[BaseConfig] = ...,
    dc_cls_doc: Optional[str] = ...,
) -> Type[BaseModel]: ...

def _is_field_cached_property(obj: object, k: str) -> bool: ...

def _dataclass_validate_values(self: Any) -> None: ...

def _dataclass_validate_assignment_setattr(self: Any, name: str, value: Any) -> None: ...

def is_builtin_dataclass(_cls: Type[Any]) -> bool: ...

def make_dataclass_validator(
    dc_cls: Type[Any],
    config: Optional[Union[ConfigDict, Type[object]]],
) -> Generator[Callable[..., Any], None, None]: ...