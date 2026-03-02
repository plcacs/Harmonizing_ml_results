import abc
import typing
from datetime import datetime
from typing import Any, Callable, ClassVar, FrozenSet, Generic, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple, Type, TypeVar, Union, cast
from mode.utils.objects import cached_property
from faust.exceptions import ValidationError
from .codecs import CodecArg

__all__ = ['CoercionHandler', 'FieldDescriptorT', 'FieldMap', 'IsInstanceArgT', 'ModelArg', 'ModelOptions', 'ModelT']

FieldMap = Mapping[str, 'FieldDescriptorT[Any]']
T = TypeVar('T')
try:

    @typing.no_type_check
    class _InitSubclassCheck(metaclass=abc.ABCMeta):
        def __init_subclass__(self, *args, ident: int = 808, **kwargs):
            self.ident = ident
            super().__init__(*args, **kwargs)

    @typing.no_type_check
    class _UsingKwargsInNew(_InitSubclassCheck, ident=909):
        ...

except TypeError:
    abc_compatible_with_init_subclass = False
else:
    abc_compatible_with_init_subclass = True

ModelArg = Union[Type['ModelT[Any]'], Type[bytes], Type[str]]
IsInstanceArgT = Union[Type, Tuple[Type, ...]]
CoercionHandler = Callable[[Any], Any]
CoercionMapping = MutableMapping[IsInstanceArgT, CoercionHandler]

class ModelOptions(abc.ABC):
    serializer: Optional[Callable[[Any], Any]] = None
    include_metadata: bool = True
    polymorphic_fields: bool = False
    allow_blessed_key: bool = False
    isodates: bool = False
    decimals: bool = False
    validation: bool = False
    coerce: bool = False
    coercions: cast(CoercionMapping, None)
    date_parser: Optional[Callable[[Any], datetime]] = None
    fields: cast(Mapping[str, Type], None)
    fieldset: cast(FrozenSet[str], None)
    descriptors: cast(FieldMap, None)
    fieldpos: cast(Mapping[int, str], None)
    optionalset: cast(FrozenSet[str], None)
    defaults: cast(Mapping[str, Any], None)
    tagged_fields: cast(FrozenSet[str], None)
    personal_fields: cast(FrozenSet[str], None)
    sensitive_fields: cast(FrozenSet[str], None)
    secret_fields: cast(FrozenSet[str], None)
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

class ModelT(base, Generic[T]):
    __is_model__ = True
    __evaluated_fields__ = cast(Set[str], None)

    @classmethod
    @abc.abstractmethod
    def from_data(cls, data: Any, *, preferred_type: Optional[Type] = None) -> 'ModelT[T]':
        ...

    @classmethod
    @abc.abstractmethod
    def loads(cls, s: Any, *, default_serializer: Optional[Callable[[Any], Any]] = None, serializer: Optional[Callable[[Any], Any]] = None) -> 'ModelT[T]':
        ...

    @abc.abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @abc.abstractmethod
    def dumps(self, *, serializer: Optional[Callable[[Any], Any]] = None) -> Any:
        ...

    @abc.abstractmethod
    def derive(self, *objects: 'ModelT[T]', **fields: Any) -> 'ModelT[T]':
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
    def __init__(self, *, field: Optional['FieldDescriptorT[T]'] = None, input_name: Optional[str] = None, output_name: Optional[str] = None, type: Optional[Type] = None, model: Optional[Type['ModelT[T]']] = None, required: bool = True, default: Any = None, parent: Optional['ModelT[T]'] = None, exclude: Optional[Iterable[str]] = None, date_parser: Optional[Callable[[Any], datetime]] = None, **kwargs: Any) -> None:
        self.date_parser = cast(Callable[[Any], datetime], date_parser)

    @abc.abstractmethod
    def on_model_attached(self) -> None:
        ...

    @abc.abstractmethod
    def clone(self, **kwargs: Any) -> 'FieldDescriptorT[T]':
        ...

    @abc.abstractmethod
    def as_dict(self) -> Any:
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
    def prepare_value(self, value: Any) -> T:
        ...

    @abc.abstractmethod
    def should_coerce(self, value: Any) -> bool:
        ...

    @abc.abstractmethod
    def getattr(self, obj: 'ModelT[T]') -> T:
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
