from typing import Any, Callable, ClassVar, ContextManager, Dict, Generic, Optional, Type, TypeVar, Union, overload
import dataclasses
from typing_extensions import dataclass_transform
from pydantic.v1.config import BaseConfig, ConfigDict
from pydantic.v1.fields import Field
from pydantic.v1.main import BaseModel
from pydantic.v1.typing import CallableGenerator

__all__: list[str]

T = TypeVar("T")
C = TypeVar("C", bound=type)


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
    config: Optional[Union[Type[BaseConfig], ConfigDict]] = ...,
    validate_on_init: Optional[bool] = ...,
    use_proxy: Optional[bool] = ...,
    kw_only: bool = ...,
) -> Callable[[C], C]: ...
@dataclass_transform(field_specifiers=(dataclasses.field, Field))
@overload
def dataclass(
    _cls: C,
    *,
    init: bool = ...,
    repr: bool = ...,
    eq: bool = ...,
    order: bool = ...,
    unsafe_hash: bool = ...,
    frozen: bool = ...,
    config: Optional[Union[Type[BaseConfig], ConfigDict]] = ...,
    validate_on_init: Optional[bool] = ...,
    use_proxy: Optional[bool] = ...,
    kw_only: bool = ...,
) -> C: ...
def dataclass(
    _cls: Optional[C] = ...,
    *,
    init: bool = ...,
    repr: bool = ...,
    eq: bool = ...,
    order: bool = ...,
    unsafe_hash: bool = ...,
    frozen: bool = ...,
    config: Optional[Union[Type[BaseConfig], ConfigDict]] = ...,
    validate_on_init: Optional[bool] = ...,
    use_proxy: Optional[bool] = ...,
    kw_only: bool = ...,
) -> Union[C, Callable[[C], C]]: ...


def set_validation(cls: type[Any], value: bool) -> ContextManager[type[Any]]: ...


class DataclassProxy(Generic[T]):
    __slots__: ClassVar[tuple[str, ...]]
    __dataclass__: type[T]

    def __init__(self, dc_cls: type[T]) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> T: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, __name: str, __value: Any) -> None: ...
    def __instancecheck__(self, instance: object) -> bool: ...
    def __copy__(self) -> "DataclassProxy[T]": ...
    def __deepcopy__(self, memo: Dict[int, Any]) -> "DataclassProxy[T]": ...


def create_pydantic_model_from_dataclass(
    dc_cls: type[Any],
    config: Union[Type[BaseConfig], ConfigDict] = BaseConfig,
    dc_cls_doc: Optional[str] = ...,
) -> Type[BaseModel]: ...


def is_builtin_dataclass(_cls: type[Any]) -> bool: ...


def make_dataclass_validator(
    dc_cls: type[Any],
    config: Union[Type[BaseConfig], ConfigDict],
) -> CallableGenerator: ...


__all__ = ['dataclass', 'set_validation', 'create_pydantic_model_from_dataclass', 'is_builtin_dataclass', 'make_dataclass_validator']