"""
Stub file for dataclasses_a27d2a module
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Generator, Optional, Type, TypeVar, Union, overload
from typing_extensions import dataclass_transform
from pydantic.v1.main import BaseModel

if TYPE_CHECKING:
    DataclassT = TypeVar('DataclassT', bound='Dataclass')
    DataclassClassOrWrapper = Union[Type['Dataclass'], 'DataclassProxy']

    class Dataclass:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            ...
        
        @classmethod
        def __get_validators__(cls) -> Generator[Callable[..., Any], None, None]:
            ...
        
        @classmethod
        def __validate__(cls, v: Any) -> Any:
            ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
@overload
def dataclass(*, init: bool = True, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: Optional[BaseConfig] = None, validate_on_init: Optional[bool] = None, use_proxy: Optional[bool] = None, kw_only: bool = ...) -> Callable[[Type[Any]], Type[Any]]:
    ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
@overload
def dataclass(_cls: Type[Any], *, init: bool = True, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: Optional[BaseConfig] = None, validate_on_init: Optional[bool] = None, use_proxy: Optional[bool] = None, kw_only: bool = ...) -> Type[Any]:
    ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
def dataclass(_cls: Optional[Type[Any]] = None, *, init: bool = True, repr: bool = True, eq: bool = True, order: bool = False, unsafe_hash: bool = False, frozen: bool = False, config: Optional[BaseConfig] = None, validate_on_init: Optional[bool] = None, use_proxy: Optional[bool] = None, kw_only: bool = False) -> Union[Type[Any], Callable[[Type[Any]], Type[Any]]]:
    ...

@contextmanager
def set_validation(cls: Type[Any], value: bool) -> Generator[Type[Any], None, None]:
    ...

class DataclassProxy:
    __slots__: ClassVar[Tuple[str]] = ('__dataclass__',)
    
    def __init__(self, dc_cls: Type[Any]) -> None:
        ...
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
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

def create_pydantic_model_from_dataclass(dc_cls: Type[Any], config: BaseConfig = ..., dc_cls_doc: Optional[str] = None) -> Type[BaseModel]:
    ...

def is_builtin_dataclass(_cls: Type[Any]) -> bool:
    ...

def make_dataclass_validator(dc_cls: Type[Any], config: BaseConfig) -> Generator[CallableGenerator, None, None]:
    ...