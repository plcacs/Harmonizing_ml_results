import abc
import typing
from datetime import datetime
from typing import Any, Callable, ClassVar, FrozenSet, Generic, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple, Type, TypeVar, Union, cast
from mode.utils.objects import cached_property
from faust.exceptions import ValidationError
from .codecs import CodecArg

__all__ = ['CoercionHandler', 'FieldDescriptorT', 'FieldMap', 'IsInstanceArgT', 'ModelArg', 'ModelOptions', 'ModelT']

FieldMap = Mapping[str, 'FieldDescriptorT']
T = TypeVar('T')

try:
    @typing.no_type_check
    class _InitSubclassCheck(metaclass=abc.ABCMeta):
        ident: int

        def __init_subclass__(self, *args: Any, ident: int = 808, **kwargs: Any) -> None:
            self.ident = ident
            super().__init__(*args, **kwargs)

    @typing.no_type_check
    class _UsingKwargsInNew(_InitSubclassCheck, ident=909):
        ...
except TypeError:
    abc_compatible_with_init_subclass = False
else:
    abc_compatible_with_init_subclass = True

ModelArg = Union[Type['ModelT'], Type[bytes], Type[str]]
IsInstanceArgT = Union[Type, Tuple[Type, ...]]
CoercionHandler = Callable[[Any], Any]
CoercionMapping = MutableMapping[IsInstanceArgT, CoercionHandler]

class ModelOptions(abc.ABC):
    serializer: Optional[Any] = None
    include_metadata: bool = True
    polymorphic_fields: bool = False
    allow_blessed_key: bool = False
    isodates: bool = False
    decimals: bool = False
    validation: bool = False
    coerce: bool = False
    coercions: CoercionMapping = cast(CoercionMapping, None)
    date_parser: Optional[Callable[[Any], datetime]] = None
    fields: Mapping[str, Type] = cast(Mapping[str, Type], None)
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
    namespace: Any

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
    __is_model__: bool = True
    __evaluated_fields__: Set[str] = cast(Set[str], None)

    @classmethod
    @abc.abstractmethod
    def from_data(cls, data: Any, *, preferred_type: Optional[Type] = None) -> 'ModelT':
        ...

    @classmethod
    @abc.abstractmethod
    def loads(cls, s: Any, *, default_serializer: Any = None, serializer: Any = None) -> 'ModelT':
        ...

    @abc.abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def dumps(self, *, serializer: Any = None) -> bytes:
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
    required: bool = True
    default: Any = None
    date_parser: Optional[Callable[[Any], datetime]]
    field: Any
    input_name: Optional[str]
    output_name: Optional[str]
    type: Any
    model: Any
    parent: Any
    exclude: Any

    @abc.abstractmethod
    def __init__(self, *, field: Any = None, input_name: Optional[str] = None, output_name: Optional[str] = None, type: Any = None, model: Any = None, required: bool = True, default: Any = None, parent: Any = None, exclude: Any = None, date_parser: Optional[Callable[[Any], datetime]] = None, **kwargs: Any) -> None:
        self.date_parser = cast(Callable[[Any], datetime], date_parser)

    @abc.abstractmethod
    def on_model_attached(self) -> None:
        ...

    @abc.abstractmethod
    def clone(self, **kwargs: Any) -> 'FieldDescriptorT':
        ...

    @abc.abstractmethod
    def as_dict(self) -> Mapping[str, Any]:
        ...

    @abc.abstractmethod
    def validate_all(self, value: Any) -> Iterable[ValidationError]:
        ...

    @abc.abstractmethod
    def validate(self, value: Any) -> Optional[ValidationError]:
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
