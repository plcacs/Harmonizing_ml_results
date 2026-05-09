import dataclasses
import copy
from typing import Any, Callable, Generator, Optional, Type, TypeVar, Union, overload
from typing_extensions import dataclass_transform
from pydantic.v1.config import BaseConfig
from pydantic.v1.main import BaseModel

_T = TypeVar('_T')
DataclassT = TypeVar('DataclassT', bound='Dataclass')

@dataclass_transform(field_specifiers=(dataclasses.field, Any))
@overload
def dataclass(
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    config: Optional[Union[BaseConfig, dict]] = None,
    validate_on_init: Optional[bool] = None,
    use_proxy: Optional[bool] = None,
    kw_only: bool = ...,
) -> Callable[[Type[Any]], Type[Any]]: ...

@dataclass_transform(field_specifiers=(dataclasses.field, Any))
@overload
def dataclass(
    _cls: Type[Any],
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    config: Optional[Union[BaseConfig, dict]] = None,
    validate_on_init: Optional[bool] = None,
    use_proxy: Optional[bool] = None,
    kw_only: bool = ...,
) -> Type[Any]: ...

@dataclass_transform(field_specifiers=(dataclasses.field, Any))
def dataclass(
    _cls: Optional[Type[Any]] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    config: Optional[Union[BaseConfig, dict]] = None,
    validate_on_init: Optional[bool] = None,
    use_proxy: Optional[bool] = None,
    kw_only: bool = False,
) -> Union[Callable[[Type[Any]], Type[Any]], Type[Any]]: ...

def set_validation(cls: Type[Any], value: bool) -> Generator[Type[Any], None, None]: ...

class DataclassProxy:
    __slots__: tuple[str, ...]
    __dataclass__: Any

    def __init__(self, dc_cls: Type[Any]) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, __name: str, __value: Any) -> None: ...
    def __instancecheck__(self, instance: Any) -> bool: ...
    def __copy__(self) -> 'DataclassProxy': ...
    def __deepcopy__(self, memo: dict[Any, Any]) -> 'DataclassProxy': ...

def create_pydantic_model_from_dataclass(
    dc_cls: Type[Any],
    config: Union[BaseConfig, dict] = ...,
    dc_cls_doc: Optional[str] = None,
) -> Type[BaseModel]: ...

def is_builtin_dataclass(_cls: Type[Any]) -> bool: ...

def make_dataclass_validator(
    dc_cls: Type[Any],
    config: Union[BaseConfig, dict],
) -> Generator[Callable, None, None]: ...

class Dataclass:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @classmethod
    def __get_validators__(cls) -> Generator[Callable, None, None]: ...
    @classmethod
    def __validate__(cls, v: Any) -> Any: ...