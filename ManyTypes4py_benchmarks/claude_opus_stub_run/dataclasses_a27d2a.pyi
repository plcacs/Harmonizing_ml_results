import copy
import dataclasses
import sys
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import dataclass_transform

from pydantic.v1.class_validators import gather_all_validators as gather_all_validators
from pydantic.v1.config import BaseConfig, ConfigDict, Extra, get_config
from pydantic.v1.error_wrappers import ValidationError
from pydantic.v1.errors import DataclassTypeError
from pydantic.v1.fields import Field, FieldInfo, Required, Undefined
from pydantic.v1.main import BaseModel, create_model, validate_model
from pydantic.v1.utils import ClassAttribute

if TYPE_CHECKING:
    from pydantic.v1.typing import CallableGenerator, NoArgAnyCallable

    DataclassT = TypeVar('DataclassT', bound='Dataclass')
    DataclassClassOrWrapper = Union[Type['Dataclass'], 'DataclassProxy']

    class Dataclass:
        __dataclass_fields__: ClassVar[Dict[str, Any]]
        __dataclass_params__: ClassVar[Any]
        __post_init__: ClassVar[Callable[..., None]]
        __pydantic_run_validation__: ClassVar[bool]
        __post_init_post_parse__: ClassVar[Callable[..., None]]
        __pydantic_initialised__: ClassVar[bool]
        __pydantic_model__: ClassVar[Type[BaseModel]]
        __pydantic_validate_values__: ClassVar[Callable[..., None]]
        __pydantic_has_field_info_default__: ClassVar[bool]

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        @classmethod
        def __get_validators__(cls: Type['DataclassT']) -> 'CallableGenerator': ...
        @classmethod
        def __validate__(cls: Type['DataclassT'], v: Any) -> 'DataclassT': ...

__all__ = [
    'dataclass',
    'set_validation',
    'create_pydantic_model_from_dataclass',
    'is_builtin_dataclass',
    'make_dataclass_validator',
]

_T = TypeVar('_T')

if sys.version_info >= (3, 10):
    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        config: Union[ConfigDict, Type[object], None] = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
        kw_only: bool = ...,
    ) -> Callable[[Type[Any]], Union[Type['Dataclass'], 'DataclassProxy']]: ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
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
        config: Union[ConfigDict, Type[object], None] = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
        kw_only: bool = ...,
    ) -> Union[Type['Dataclass'], 'DataclassProxy']: ...
else:
    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        config: Union[ConfigDict, Type[object], None] = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
    ) -> Callable[[Type[Any]], Union[Type['Dataclass'], 'DataclassProxy']]: ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
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
        config: Union[ConfigDict, Type[object], None] = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
    ) -> Union[Type['Dataclass'], 'DataclassProxy']: ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
def dataclass(
    _cls: Optional[Type[Any]] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    config: Union[ConfigDict, Type[object], None] = None,
    validate_on_init: Optional[bool] = None,
    use_proxy: Optional[bool] = None,
    kw_only: bool = False,
) -> Union[Type['Dataclass'], 'DataclassProxy', Callable[[Type[Any]], Union[Type['Dataclass'], 'DataclassProxy']]]: ...

@contextmanager
def set_validation(cls: Type[Any], value: bool) -> Generator[Type[Any], None, None]: ...

class DataclassProxy:
    __slots__: tuple[str, ...]
    __dataclass__: Type[Any]

    def __init__(self, dc_cls: Type[Any]) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, __name: str, __value: Any) -> None: ...
    def __instancecheck__(self, instance: Any) -> bool: ...
    def __copy__(self) -> DataclassProxy: ...
    def __deepcopy__(self, memo: Dict[int, Any]) -> DataclassProxy: ...

def _add_pydantic_validation_attributes(
    dc_cls: Type[Any],
    config: Type[BaseConfig],
    validate_on_init: bool,
    dc_cls_doc: str,
) -> None: ...

def _get_validators(cls: Type[Any]) -> Generator[Any, None, None]: ...

def _validate_dataclass(cls: Type[Any], v: Any) -> Any: ...

def create_pydantic_model_from_dataclass(
    dc_cls: Type[Any],
    config: Union[Type[BaseConfig], Type[object]] = ...,
    dc_cls_doc: Optional[str] = None,
) -> Type['BaseModel']: ...

def _is_field_cached_property(obj: Any, k: str) -> bool: ...

def _dataclass_validate_values(self: Any) -> None: ...

def _dataclass_validate_assignment_setattr(self: Any, name: str, value: Any) -> None: ...

def is_builtin_dataclass(_cls: Type[Any]) -> bool: ...

def make_dataclass_validator(
    dc_cls: Type[Any],
    config: Type[Any],
) -> Generator[Any, None, None]: ...