"""
Stub file for pydantic/v1/dataclasses.py
"""

from __future__ import annotations
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    NoReturn,
    Optional,
    overload,
    Tuple,
    Type,
    TypeVar,
    Union,
    final,
    runtime_checkable,
)
from typing_extensions import (
    Concatenate,
    ParamSpec,
    Self,
    TypeAlias,
    TypeGuard,
    dataclass_transform,
)
from pydantic.v1 import BaseModel, ConfigDict, Extra, ValidationError
from pydantic.v1.utils import ClassAttribute

_T = TypeVar('_T')
DataclassT = TypeVar('DataclassT', bound='Dataclass')
DataclassClassOrWrapper = Union[Type['Dataclass'], 'DataclassProxy']

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
@overload
def dataclass(*, init: bool = True, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: Optional[ConfigDict] = None, validate_on_init: Optional[bool] = None, use_proxy: Optional[bool] = None, kw_only: bool = False) -> Callable[[Type[DataclassT]], Type[DataclassT]]:
    ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
@overload
def dataclass(_cls: Type[DataclassT], *, init: bool = True, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: Optional[ConfigDict] = None, validate_on_init: Optional[bool] = None, use_proxy: Optional[bool] = None, kw_only: bool = False) -> Type[DataclassT]:
    ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
def dataclass(_cls: Optional[Type[DataclassT]] = None, *, init: bool = True, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: Optional[ConfigDict] = None, validate_on_init: Optional[bool] = None, use_proxy: Optional[bool] = None, kw_only: bool = False) -> Union[Type[DataclassT], Callable[[Type[DataclassT]], Type[DataclassT]]]:
    ...

class DataclassProxy:
    __slots__: ClassVar[List[str]] = ['__dataclass__']

    def __init__(self, dc_cls: Type[DataclassT]) -> None:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> DataclassT:
        ...

    def __getattr__(self, name: str) -> Any:
        ...

    def __setattr__(self, __name: str, __value: Any) -> None:
        ...

    def __instancecheck__(self, instance: Any) -> bool:
        ...

    def __copy__(self) -> DataclassProxy:
        ...

    def __deepcopy__(self, memo: Dict[int, Any]) -> DataclassProxy:
        ...

@contextmanager
def set_validation(cls: Type[DataclassT], value: bool) -> Generator[Type[DataclassT], None, None]:
    ...

def _add_pydantic_validation_attributes(dc_cls: Type[DataclassT], config: ConfigDict, validate_on_init: bool, dc_cls_doc: str) -> None:
    ...

def create_pydantic_model_from_dataclass(dc_cls: Type[DataclassT], config: ConfigDict = ..., dc_cls_doc: Optional[str] = None) -> Type[BaseModel]:
    ...

def is_builtin_dataclass(_cls: Type[Any]) -> bool:
    ...

def make_dataclass_validator(dc_cls: Type[DataclassT], config: ConfigDict) -> Iterable[Callable[..., Any]]:
    ...

__all__: List[str] = ['dataclass', 'set_validation', 'create_pydantic_model_from_dataclass', 'is_builtin_dataclass', 'make_dataclass_validator']