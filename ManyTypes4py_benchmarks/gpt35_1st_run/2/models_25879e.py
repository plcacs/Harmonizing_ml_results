from typing import Any, Callable, ClassVar, FrozenSet, Generic, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple, Type, TypeVar, Union
from mode.utils.objects import cached_property
from faust.exceptions import ValidationError

FieldMap: Mapping[str, 'FieldDescriptorT']
T = TypeVar('T')
CoercionMapping: MutableMapping[Union[Type, Tuple[Type, ...]], Callable[[Any], Any]]

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
    date_parser: Optional[Any] = None
    fields: Optional[Mapping[str, Type]] = cast(Mapping[str, Type], None)
    fieldset: Optional[FrozenSet[str]] = cast(FrozenSet[str], None)
    descriptors: Optional[FieldMap] = cast(FieldMap, None)
    fieldpos: Optional[Mapping[int, str]] = cast(Mapping[int, str], None)
    optionalset: Optional[FrozenSet[str]] = cast(FrozenSet[str], None)
    defaults: Optional[Mapping[str, Any]] = cast(Mapping[str, Any], None)
    tagged_fields: Optional[FrozenSet[str]] = cast(FrozenSet[str], None)
    personal_fields: Optional[FrozenSet[str]] = cast(FrozenSet[str], None)
    sensitive_fields: Optional[FrozenSet[str]] = cast(FrozenSet[str], None)
    secret_fields: Optional[FrozenSet[str]] = cast(FrozenSet[str], None)
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

base: Type = abc.ABC if abc_compatible_with_init_subclass else object

class ModelT(base):
    __is_model__: bool = True
    __evaluated_fields__: Set[str] = cast(Set[str], None)

    @classmethod
    @abc.abstractmethod
    def from_data(cls, data: Any, *, preferred_type: Optional[Any] = None) -> Any:
        ...

    @classmethod
    @abc.abstractmethod
    def loads(cls, s: Any, *, default_serializer: Optional[Any] = None, serializer: Optional[Any] = None) -> Any:
        ...

    @abc.abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def dumps(self, *, serializer: Optional[Any] = None) -> Any:
        ...

    @abc.abstractmethod
    def derive(self, *objects: Any, **fields: Any) -> Any:
        ...

    @abc.abstractmethod
    def to_representation(self) -> Any:
        ...

    @abc.abstractmethod
    def is_valid(self) -> Any:
        ...

    @abc.abstractmethod
    def validate(self) -> Any:
        ...

    @abc.abstractmethod
    def validate_or_raise(self) -> Any:
        ...

    @property
    @abc.abstractmethod
    def validation_errors(self) -> Any:
        ...

class FieldDescriptorT(Generic[T]):
    required: bool = True
    default: Optional[Any] = None

    @abc.abstractmethod
    def __init__(self, *, field: Optional[Any] = None, input_name: Optional[Any] = None, output_name: Optional[Any] = None, type: Optional[Any] = None, model: Optional[Any] = None, required: bool = True, default: Optional[Any] = None, parent: Optional[Any] = None, exclude: Optional[Any] = None, date_parser: Optional[Any] = None, **kwargs: Any) -> None:
        self.date_parser = cast(Callable[[Any], datetime], date_parser)

    @abc.abstractmethod
    def on_model_attached(self) -> Any:
        ...

    @abc.abstractmethod
    def clone(self, **kwargs: Any) -> Any:
        ...

    @abc.abstractmethod
    def as_dict(self) -> Any:
        ...

    @abc.abstractmethod
    def validate_all(self, value: Any) -> Any:
        ...

    @abc.abstractmethod
    def validate(self, value: Any) -> Any:
        ...

    @abc.abstractmethod
    def to_python(self, value: Any) -> Any:
        ...

    @abc.abstractmethod
    def prepare_value(self, value: Any) -> Any:
        ...

    @abc.abstractmethod
    def should_coerce(self, value: Any) -> Any:
        ...

    @abc.abstractmethod
    def getattr(self, obj: Any) -> Any:
        ...

    @abc.abstractmethod
    def validation_error(self, reason: Any) -> Any:
        ...

    @property
    @abc.abstractmethod
    def ident(self) -> Any:
        ...

    @cached_property
    @abc.abstractmethod
    def related_models(self) -> Any:
        ...

    @cached_property
    @abc.abstractmethod
    def lazy_coercion(self) -> Any:
        ...

ModelArg: Union[Type[ModelT], Type[bytes], Type[str]]
