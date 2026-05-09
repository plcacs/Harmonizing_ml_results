from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Literal,
    NoReturn,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import (
    dataclass_transform,
    final,
    get_type_hints,
    runtime_checkable,
)
from pydantic.v1.main import BaseModel
from pydantic.v1.typing import NoArgAnyCallable

_T = TypeVar('_T')
DataclassT = TypeVar('DataclassT', bound='Dataclass')
DataclassClassOrWrapper = Union[Type[Dataclass], DataclassProxy]

@runtime_checkable
class Dataclass(Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @classmethod
    def __get_validators__(cls) -> Iterable[Callable[..., Any]]:
        ...

    @classmethod
    def __validate__(cls, v: Any) -> Any:
        ...

class DataclassProxy:
    __slots__: ClassVar[Tuple[str]] = ('__dataclass__',)
    __dataclass__: Type[Any]

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

@contextmanager
def set_validation(cls: Type[Any], value: bool) -> Generator[Type[Any], None, None]:
    ...

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
    kw_only: Union[bool, Literal[Ellipsis]] = ...,
) -> Callable[[Type[Any]], Type[Any]]:
    ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
@overload
def dataclass(
    _cls: Type[Any],
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
    kw_only: Union[bool, Literal[Ellipsis]] = ...,
) -> Type[Any]:
    ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
def dataclass(...) -> Any:
    ...

def create_pydantic_model_from_dataclass(
    dc_cls: Type[Any],
    config: Union[Type[BaseConfig], ConfigDict] = ...,
    dc_cls_doc: Optional[str] = ...,
) -> Type[BaseModel]:
    ...

def is_builtin_dataclass(_cls: Type[Any]) -> bool:
    ...

def make_dataclass_validator(
    dc_cls: Type[Any], config: Union[Type[BaseConfig], ConfigDict] = ...
) -> Generator[Callable[..., Any], None, None]:
    ...