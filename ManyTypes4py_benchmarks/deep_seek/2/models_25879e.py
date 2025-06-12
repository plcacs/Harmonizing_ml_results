import abc
import typing
from datetime import datetime
from typing import Any, Callable, ClassVar, FrozenSet, Generic, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple, Type, TypeVar, Union, cast
from mode.utils.objects import cached_property
from faust.exceptions import ValidationError
from .codecs import CodecArg
from typing_extensions import Protocol

__all__ = ['CoercionHandler', 'FieldDescriptorT', 'FieldMap', 'IsInstanceArgT', 'ModelArg', 'ModelOptions', 'ModelT']

FieldMap = Mapping[str, 'FieldDescriptorT[Any]']
T = TypeVar('T')
T_ModelT = TypeVar('T_ModelT', bound='ModelT')

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
    abc_compatible_with_init_subclass: bool = False
else:
    abc_compatible_with_init_subclass: bool = True

ModelArg = Union[Type['ModelT'], Type[bytes], Type[str]]
IsInstanceArgT = Union[Type[Any], Tuple[Type[Any], ...]]
CoercionHandler = Callable[[Any], Any]
CoercionMapping = MutableMapping[IsInstanceArgT, CoercionHandler]

class ModelOptions(abc.ABC):
    serializer: Optional[CodecArg]
    include_metadata: bool
    polymorphic_fields: bool
    allow_blessed_key: bool
    isodates: bool
    decimals: bool
    validation: bool
    coerce: bool
    coercions: CoercionMapping
    date_parser: Optional[Callable[[Any], datetime]]
    fields: Mapping[str, Type[Any]]
    fieldset: FrozenSet[str]
    descriptors: FieldMap
    fieldpos: Mapping[int, str]
    optionalset: FrozenSet[str]
    defaults: Mapping[str, Any]
    tagged_fields: FrozenSet[str]
    personal_fields: FrozenSet[str]
    sensitive_fields: FrozenSet[str]
    secret_fields: FrozenSet[str]
    has_tagged_fields: bool
    has_personal_fields: bool
    has_sensitive_fields: bool
    has_secret_fields: bool
    namespace: str

    def __init__(self) -> None:
        self.serializer = None
        self.include_metadata = True
        self.polymorphic_fields = False
        self.allow_blessed_key = False
        self.isodates = False
        self.decimals = False
        self.validation = False
        self.coerce = False
        self.coercions = cast(CoercionMapping, None)
        self.date_parser = None
        self.fields = cast(Mapping[str, Type[Any]], None)
        self.fieldset = cast(FrozenSet[str], None)
        self.descriptors = cast(FieldMap, None)
        self.fieldpos = cast(Mapping[int, str], None)
        self.optionalset = cast(FrozenSet[str], None)
        self.defaults = cast(Mapping[str, Any], None)
        self.tagged_fields = cast(FrozenSet[str], None)
        self.personal_fields = cast(FrozenSet[str], None)
        self.sensitive_fields = cast(FrozenSet[str], None)
        self.secret_fields = cast(FrozenSet[str], None)
        self.has_tagged_fields = False
        self.has_personal_fields = False
        self.has_sensitive_fields = False
        self.has_secret_fields = False
        self.namespace = ""

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
        new_options.coercions = dict(self.coercions) if self.coercions else {}
        return new_options

base = abc.ABC if abc_compatible_with_init_subclass else object

class ModelT(base):
    __is_model__: ClassVar[bool] = True
    __evaluated_fields__: Set[str]
    Options: ClassVar[Type[ModelOptions]]

    @classmethod
    @abc.abstractmethod
    def from_data(cls: Type[T_ModelT], data: Any, *, preferred_type: Optional[Type[Any]] = None) -> T_ModelT:
        ...

    @classmethod
    @abc.abstractmethod
    def loads(cls: Type[T_ModelT], s: Union[bytes, str], *, default_serializer: Optional[CodecArg] = None, serializer: Optional[CodecArg] = None) -> T_ModelT:
        ...

    @abc.abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def dumps(self, *, serializer: Optional[CodecArg] = None) -> Union[bytes, str]:
        ...

    @abc.abstractmethod
    def derive(self: T_ModelT, *objects: Any, **fields: Any) -> T_ModelT:
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
    required: bool
    default: Optional[T]
    field: Any
    input_name: Optional[str]
    output_name: Optional[str]
    type: Optional[Type[Any]]
    model: Optional[Type[ModelT]]
    parent: Optional[Any]
    exclude: Optional[bool]
    date_parser: Callable[[Any], datetime]

    @abc.abstractmethod
    def __init__(self, *, field: Any = None, input_name: Optional[str] = None, output_name: Optional[str] = None, type: Optional[Type[Any]] = None, model: Optional[Type[ModelT]] = None, required: bool = True, default: Optional[T] = None, parent: Optional[Any] = None, exclude: Optional[bool] = None, date_parser: Optional[Callable[[Any], datetime]] = None, **kwargs: Any) -> None:
        self.date_parser = cast(Callable[[Any], datetime], date_parser)

    @abc.abstractmethod
    def on_model_attached(self) -> None:
        ...

    @abc.abstractmethod
    def clone(self: 'FieldDescriptorT[T]', **kwargs: Any) -> 'FieldDescriptorT[T]':
        ...

    @abc.abstractmethod
    def as_dict(self) -> Mapping[str, Any]:
        ...

    @abc.abstractmethod
    def validate_all(self, value: T) -> T:
        ...

    @abc.abstractmethod
    def validate(self, value: T) -> T:
        ...

    @abc.abstractmethod
    def to_python(self, value: Any) -> T:
        ...

    @abc.abstractmethod
    def prepare_value(self, value: Any) -> T:
        ...

    @abc.abstractmethod
    def should_coerce(self, value: Any) -> bool:
        ...

    @abc.abstractmethod
    def getattr(self, obj: Any) -> T:
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
    def lazy_coercion(self) -> Optional[Callable[[Any], Any]]:
        ...

ModelArg = Union[Type[ModelT], Type[bytes], Type[str]]
