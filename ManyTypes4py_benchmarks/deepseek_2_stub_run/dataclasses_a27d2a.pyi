```python
import sys
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
    contextmanager,
)
from typing_extensions import dataclass_transform
from pydantic.v1.config import BaseConfig, ConfigDict, Extra
from pydantic.v1.error_wrappers import ValidationError
from pydantic.v1.errors import DataclassTypeError
from pydantic.v1.fields import Field, FieldInfo
from pydantic.v1.main import BaseModel

if TYPE_CHECKING:
    from pydantic.v1.typing import CallableGenerator, NoArgAnyCallable

    DataclassT = TypeVar("DataclassT", bound="Dataclass")
    DataclassClassOrWrapper = Union[Type["Dataclass"], "DataclassProxy"]

    class Dataclass:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        @classmethod
        def __get_validators__(cls) -> Generator[Any, None, None]: ...
        @classmethod
        def __validate__(cls, v: Any) -> Any: ...

_T = TypeVar("_T")

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
        config: Any = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
        kw_only: bool = ...,
    ) -> Callable[[Type[_T]], Type[_T]]: ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(
        _cls: Type[_T],
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        config: Any = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
        kw_only: bool = ...,
    ) -> Type[_T]: ...
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
        config: Any = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
    ) -> Callable[[Type[_T]], Type[_T]]: ...

    @dataclass_transform(field_specifiers=(dataclasses.field, Field))
    @overload
    def dataclass(
        _cls: Type[_T],
        *,
        init: bool = True,
        repr: bool = True,
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        config: Any = None,
        validate_on_init: Optional[bool] = None,
        use_proxy: Optional[bool] = None,
    ) -> Type[_T]: ...

@dataclass_transform(field_specifiers=(dataclasses.field, Field))
def dataclass(
    _cls: Optional[Type[_T]] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    config: Any = None,
    validate_on_init: Optional[bool] = None,
    use_proxy: Optional[bool] = None,
    kw_only: bool = False,
) -> Union[Type[_T], Callable[[Type[_T]], Type[_T]]]: ...

@contextmanager
def set_validation(cls: Type[Any], value: bool) -> Generator[Type[Any], None, None]: ...

class DataclassProxy:
    __slots__: ClassVar[tuple] = ("__dataclass__",)
    __dataclass__: Type[Any]
    def __init__(self, dc_cls: Type[Any]) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, __name: str, __value: Any) -> None: ...
    def __instancecheck__(self, instance: Any) -> bool: ...
    def __copy__(self) -> "DataclassProxy": ...
    def __deepcopy__(self, memo: Dict[Any, Any]) -> "DataclassProxy": ...

def create_pydantic_model_from_dataclass(
    dc_cls: Type[Any],
    config: Any = ...,
    dc_cls_doc: Optional[str] = None,
) -> Type[BaseModel]: ...

def is_builtin_dataclass(_cls: Type[Any]) -> bool: ...

def make_dataclass_validator(
    dc_cls: Type[Any], config: Any
) -> Generator[Any, None, None]: ...

__all__: list[str] = ...
```