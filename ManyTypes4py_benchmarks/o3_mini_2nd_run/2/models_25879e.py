import abc
import typing
from datetime import datetime
from typing import (
    Any,
    Callable,
    ClassVar,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from mode.utils.objects import cached_property
from faust.exceptions import ValidationError
from .codecs import CodecArg

__all__ = [
    'CoercionHandler',
    'FieldDescriptorT',
    'FieldMap',
    'IsInstanceArgT',
    'ModelArg',
    'ModelOptions',
    'ModelT',
]

FieldMap = Mapping[str, 'FieldDescriptorT[Any]']
T = TypeVar('T')

try:

    @typing.no_type_check
    class _InitSubclassCheck(metaclass=abc.ABCMeta):
        def __init_subclass__(cls: Type[Any], *args: Any, ident: int = 808, **kwargs: Any) -> None:
            cls.ident = ident
            super().__init_subclass__(*args, **kwargs)

    @typing.no_type_check
    class _UsingKwargsInNew(_InitSubclassCheck, ident=909):
        ...
except TypeError:
    abc_compatible_with_init_subclass = False
else:
    abc_compatible_with_init_subclass = True

ModelArg = Union[Type['ModelT'], Type[bytes], Type[str]]
IsInstanceArgT = Union[Type[Any], Tuple[Type[Any], ...]]
CoercionHandler = Callable[[Any], Any]
CoercionMapping = MutableMapping[IsInstanceArgT, CoercionHandler]


class ModelOptions(abc.ABC):
    serializer: Any = None
    include_metadata: bool = True
    polymorphic_fields: bool = False
    allow_blessed_key: bool = False
    isodates: bool = False
    decimals: bool = False
    validation: bool = False
    coerce: bool = False
    coercions: CoercionMapping = cast(CoercionMapping, None)
    date_parser: Any = None
    fields: Mapping[str, Type[Any]] = cast(Mapping[str, Type[Any]], None)
    fieldset: FrozenSet[str] = cast(FrozenSet[str], None)
    descriptors: FieldMap = cast(FieldMap, None)
    fieldpos: Mapping[int, str] = cast(Mapping[int, str], None)
    optionalset: FrozenSet[str] = cast(FrozenSet[str], None)
    defaults: Mapping[str, Any] = cast(Mapping[str, Any], None)
    tagged_fields: FrozenSet[str] = cast(FrozenSet[str], None)
    personal_fields: FrozenSet[str] = cast(FrozenSet[str], None)
    sensitive_fields: FrozenSet[str] = cast(FrozenSet[str], None)
    secret_fields: FrozenSet[str] = cast(FrozenSet[str], None)
    has_tagged_fields: bool = False
    has_personal_fields: bool = False
    has_sensitive_fields: bool = False
    has_secret_fields: bool = False
    namespace: Any = None

    def clone_defaults(self) -> 'ModelOptions':
        new_options: ModelOptions = type(self)()
        new_options.serializer = self.serializer
        new_options.namespace = self.namespace
        new_options.include_metadata = self.include_metadata
        new_options.polymorphic_fields = self.polymorphic_fields
        new_options.allow_blessed_key = self.allow_blessed_key
        new_options.isodates = self.isodates
        new_options.decimals = self.decimals
        new_options.coerce = self.coerce
        new_options.coercions = dict(self.coercions) if self.coercions is not None else {}  # type: ignore
        return new_options


base: Type = abc.ABC if abc_compatible_with_init_subclass else object


class ModelT(base):
    __is_model__: ClassVar[bool] = True
    __evaluated_fields__: ClassVar[Set[str]] = cast(Set[str], None)

    @classmethod
    @abc.abstractmethod
    def from_data(cls: Type['ModelT'], data: Any, *, preferred_type: Optional[Any] = None) -> 'ModelT':
        ...

    @classmethod
    @abc.abstractmethod
    def loads(
        cls: Type['ModelT'],
        s: Union[str, bytes],
        *,
        default_serializer: Optional[Any] = None,
        serializer: Optional[Any] = None
    ) -> 'ModelT':
        ...

    @abc.abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def dumps(self, *, serializer: Optional[Any] = None) -> Union[str, bytes]:
        ...

    @abc.abstractmethod
    def derive(self, *objects: Any, **fields: Any) -> 'ModelT':
        ...

    @abc.abstractmethod
    def to_representation(self) -> Any:
        ...

    @abc.abstractmethod
    def is_valid(self) -> bool:
        ...

    @abc.abstractmethod
    def validate(self) -> None:
        ...

    @abc.abstractmethod
    def validate_or_raise(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def validation_errors(self) -> Any:
        ...


class FieldDescriptorT(Generic[T]):
    required: bool = True
    default: Any = None

    @abc.abstractmethod
    def __init__(
        self,
        *,
        field: Optional[Any] = None,
        input_name: Optional[Any] = None,
        output_name: Optional[Any] = None,
        type: Optional[Any] = None,
        model: Optional[Any] = None,
        required: bool = True,
        default: Any = None,
        parent: Optional[Any] = None,
        exclude: Optional[Any] = None,
        date_parser: Optional[Callable[[Any], datetime]] = None,
        **kwargs: Any,
    ) -> None:
        self.date_parser: Optional[Callable[[Any], datetime]] = cast(Optional[Callable[[Any], datetime]], date_parser)

    @abc.abstractmethod
    def on_model_attached(self) -> None:
        ...

    @abc.abstractmethod
    def clone(self, **kwargs: Any) -> 'FieldDescriptorT[T]':
        ...

    @abc.abstractmethod
    def as_dict(self) -> Mapping[str, Any]:
        ...

    @abc.abstractmethod
    def validate_all(self, value: Any) -> None:
        ...

    @abc.abstractmethod
    def validate(self, value: Any) -> None:
        ...

    @abc.abstractmethod
    def to_python(self, value: Any) -> T:
        ...

    @abc.abstractmethod
    def prepare_value(self, value: Any) -> Any:
        ...

    @abc.abstractmethod
    def should_coerce(self, value: Any) -> bool:
        ...

    @abc.abstractmethod
    def getattr(self, obj: Any) -> Any:
        ...

    @abc.abstractmethod
    def validation_error(self, reason: Any) -> ValidationError:
        ...

    @property
    @abc.abstractmethod
    def ident(self) -> int:
        ...

    @cached_property
    @abc.abstractmethod
    def related_models(self) -> FrozenSet[Type[Any]]:
        ...

    @cached_property
    @abc.abstractmethod
    def lazy_coercion(self) -> Callable[[Any], Any]:
        ...


ModelArg = Union[Type[ModelT], Type[bytes], Type[str]]  
