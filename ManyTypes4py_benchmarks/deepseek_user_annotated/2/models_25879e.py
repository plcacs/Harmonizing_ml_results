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

FieldMap = Mapping[str, 'FieldDescriptorT']

T = TypeVar('T')

try:
    @typing.no_type_check
    class _InitSubclassCheck(metaclass=abc.ABCMeta):
        ident: int

        def __init_subclass__(self,
                              *args: Any,
                              ident: int = 808,
                              **kwargs: Any) -> None:
            self.ident = ident
            super().__init__(*args, **kwargs)

    @typing.no_type_check
    class _UsingKwargsInNew(_InitSubclassCheck, ident=909):
        pass
except TypeError:
    abc_compatible_with_init_subclass = False
else:
    abc_compatible_with_init_subclass = True

ModelArg = Union[Type['ModelT'], Type[bytes], Type[str]]
IsInstanceArgT = Union[Type[Any], Tuple[Type[Any], ...]]
CoercionHandler = Callable[[Any], Any]
CoercionMapping = MutableMapping[IsInstanceArgT, CoercionHandler]


class ModelOptions(abc.ABC):
    serializer: Optional[CodecArg] = None
    namespace: str = ""
    include_metadata: bool = True
    polymorphic_fields: bool = False
    allow_blessed_key: bool = False
    isodates: bool = False
    decimals: bool = False
    validation: bool = False
    coerce: bool = False
    coercions: CoercionMapping = cast(CoercionMapping, {})

    date_parser: Optional[Callable[[Any], datetime]] = None

    fields: Mapping[str, Type[Any]] = cast(Mapping[str, Type[Any]], {})
    fieldset: FrozenSet[str] = cast(FrozenSet[str], frozenset())
    descriptors: FieldMap = cast(FieldMap, {})
    fieldpos: Mapping[int, str] = cast(Mapping[int, str], {})
    optionalset: FrozenSet[str] = cast(FrozenSet[str], frozenset())
    defaults: Mapping[str, Any] = cast(Mapping[str, Any], {})
    tagged_fields: FrozenSet[str] = cast(FrozenSet[str], frozenset())
    personal_fields: FrozenSet[str] = cast(FrozenSet[str], frozenset())
    sensitive_fields: FrozenSet[str] = cast(FrozenSet[str], frozenset())
    secret_fields: FrozenSet[str] = cast(FrozenSet[str], frozenset())

    has_tagged_fields: bool = False
    has_personal_fields: bool = False
    has_sensitive_fields: bool = False
    has_secret_fields: bool = False

    def clone_defaults(self) -> 'ModelOptions':
        new_options = type(self)()
        new_options.serializer = self.serializer
        new_options.namespace = self.namespace
        new_options.include_metadata = self.include_metadata
        new_options.polymorphic_fields = self.polymorphic_fields
        new_options.allow_blessed_key = self.allow_blessed_key
        new_options.isodates = self.isodates
        new_options.decimals = self.decimals
        new_options.coerce = self.coerce
        new_options.coercions = dict(self.coercions)
        return new_options


base = abc.ABC if abc_compatible_with_init_subclass else object


class ModelT(base):
    __is_model__: ClassVar[bool] = True
    __evaluated_fields__: Set[str] = cast(Set[str], set())

    _options: ClassVar[ModelOptions]

    @classmethod
    @abc.abstractmethod
    def from_data(cls, data: Any, *, preferred_type: Type['ModelT'] = None) -> 'ModelT':
        ...

    @classmethod
    @abc.abstractmethod
    def loads(cls, s: bytes, *, default_serializer: CodecArg = None, serializer: CodecArg = None) -> 'ModelT':
        ...

    @abc.abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def dumps(self, *, serializer: CodecArg = None) -> bytes:
        ...

    @abc.abstractmethod
    def derive(self, *objects: 'ModelT', **fields: Any) -> 'ModelT':
        ...

    @abc.abstractmethod
    def to_representation(self) -> Any:
        ...

    @abc.abstractmethod
    def is_valid(self) -> bool:
        ...

    @abc.abstractmethod
    def validate(self) -> List[ValidationError]:
        ...

    @abc.abstractmethod
    def validate_or_raise(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def validation_errors(self) -> List[ValidationError]:
        ...


class FieldDescriptorT(Generic[T]):
    field: str
    input_name: str
    output_name: str
    type: Type[T]
    model: Type[ModelT]
    required: bool = True
    default: Optional[T] = None
    parent: Optional['FieldDescriptorT[Any]']
    exclude: bool

    @abc.abstractmethod
    def __init__(self, *, field: str = None, input_name: str = None, output_name: str = None, type: Type[T] = None, model: Type[ModelT] = None, required: bool = True, default: T = None, parent: 'FieldDescriptorT[Any]' = None, exclude: bool = None, date_parser: Callable[[Any], datetime] = None, **kwargs: Any) -> None:
        self.date_parser: Callable[[Any], datetime] = cast(Callable[[Any], datetime], date_parser)

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
    def validate_all(self, value: Any) -> Iterable[ValidationError]:
        ...

    @abc.abstractmethod
    def validate(self, value: T) -> Iterable[ValidationError]:
        ...

    @abc.abstractmethod
    def to_python(self, value: Any) -> Optional[T]:
        ...

    @abc.abstractmethod
    def prepare_value(self, value: Any) -> Optional[T]:
        ...

    @abc.abstractmethod
    def should_coerce(self, value: Any) -> bool:
        ...

    @abc.abstractmethod
    def getattr(self, obj: ModelT) -> T:
        ...

    @abc.abstractmethod
    def validation_error(self, reason: str) -> ValidationError:
        ...

    @property
    @abc.abstractmethod
    def ident(self) -> str:
        ...

    @cached_property
    @abc.abstractmethod
    def related_models(self) -> Set[Type[ModelT]]:
        ...

    @cached_property
    @abc.abstractmethod
    def lazy_coercion(self) -> bool:
        ...


ModelArg = Union[Type[ModelT], Type[bytes], Type[str]]
