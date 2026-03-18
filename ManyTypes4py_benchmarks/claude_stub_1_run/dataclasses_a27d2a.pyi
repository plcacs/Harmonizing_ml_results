```pyi
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Generator, Optional, Type, TypeVar, Union, overload
from typing_extensions import dataclass_transform
from pydantic.v1.config import BaseConfig, ConfigDict, Extra
from pydantic.v1.error_wrappers import ValidationError
from pydantic.v1.fields import FieldInfo

if TYPE_CHECKING:
    from pydantic.v1.main import BaseModel
    from pydantic.v1.typing import CallableGenerator, NoArgAnyCallable
    DataclassT = TypeVar('DataclassT', bound='Dataclass')
    DataclassClassOrWrapper = Union[Type['Dataclass'], 'DataclassProxy']

    class Dataclass:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        @classmethod
        def __get_validators__(cls) -> Any: ...
        @classmethod
        def __validate__(cls, v: Any) -> Any: ...

__all__: list[str]

_T = TypeVar('_T')

if sys.version_info >= (3, 10):
    @dataclass_transform(field_specifiers=(Any, Any))
    @overload
    def dataclass(
        *,
        init: bool = ...,
        repr: bool = ...,
        eq: bool = ...,
        order: bool = ...,
        unsafe_hash: bool = ...,
        frozen: bool = ...,
        config: Optional[Any] = ...,
        validate_on_init: Optional[bool] = ...,
        use_proxy: Optional[bool] = ...,
        kw_only: Any = ...,
    ) -> Callable[[Type[_T]], Any]: ...
    @dataclass_transform(field_specifiers=(Any, Any))
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
        config: Optional[Any] = ...,
        validate_on_init: Optional[bool] = ...,
        use_proxy: Optional[bool] = ...,
        kw_only: Any = ...,
    ) -> Any: ...
else:
    @dataclass_transform(field_specifiers=(Any, Any))
    @overload
    def dataclass(
        *,
        init: bool = ...,
        repr: bool = ...,
        eq: bool = ...,
        order: bool = ...,
        unsafe_hash: bool = ...,
        frozen: bool = ...,
        config: Optional[Any] = ...,
        validate_on_init: Optional[bool] = ...,
        use_proxy: Optional[bool] = ...,
    ) -> Callable[[Type[_T]], Any]: ...
    @dataclass_transform(field_specifiers=(Any, Any))
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
        config: Optional[Any] = ...,
        validate_on_init: Optional[bool] = ...,
        use_proxy: Optional[bool] = ...,
    ) -> Any: ...

@dataclass_transform(field_specifiers=(Any, Any))
def dataclass(
    _cls: Optional[Type[_T]] = ...,
    *,
    init: bool = ...,
    repr: bool = ...,
    eq: bool = ...,
    order: bool = ...,
    unsafe_hash: bool = ...,
    frozen: bool = ...,
    config: Optional[Any] = ...,
    validate_on_init: Optional[bool] = ...,
    use_proxy: Optional[bool] = ...,
    kw_only: bool = ...,
) -> Any: ...

@contextmanager
def set_validation(cls: Any, value: bool) -> Generator[Any, None, None]: ...

class DataclassProxy:
    __slots__: tuple[str, ...]
    def __init__(self, dc_cls: Any) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, __name: str, __value: Any) -> None: ...
    def __instancecheck__(self, instance: Any) -> bool: ...
    def __copy__(self) -> DataclassProxy: ...
    def __deepcopy__(self, memo: Dict[int, Any]) -> DataclassProxy: ...

def create_pydantic_model_from_dataclass(
    dc_cls: Any,
    config: Type[BaseConfig] = ...,
    dc_cls_doc: Optional[str] = ...,
) -> Type[BaseModel]: ...

def is_builtin_dataclass(_cls: Any) -> bool: ...

def make_dataclass_validator(dc_cls: Any, config: Any) -> Generator[Any, None, None]: ...
```