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

        def __init_subclass__(self, *args, ident=808, **kwargs):
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
    serializer = None
    include_metadata = True
    polymorphic_fields = False
    allow_blessed_key = False
    isodates = False
    decimals = False
    validation = False
    coerce = False
    coercions = cast(CoercionMapping, None)
    date_parser = None
    fields = cast(Mapping[str, Type], None)
    fieldset = cast(FrozenSet[str], None)
    descriptors = cast(FieldMap, None)
    fieldpos = cast(Mapping[int, str], None)
    optionalset = cast(FrozenSet[str], None)
    defaults = cast(Mapping[str, Any], None)
    tagged_fields = cast(FrozenSet[str], None)
    personal_fields = cast(FrozenSet[str], None)
    sensitive_fields = cast(FrozenSet[str], None)
    secret_fields = cast(FrozenSet[str], None)
    has_tagged_fields = False
    has_personal_fields = False
    has_sensitive_fields = False
    has_secret_fields = False

    def clone_defaults(self):
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
    __is_model__ = True
    __evaluated_fields__ = cast(Set[str], None)

    @classmethod
    @abc.abstractmethod
    def from_data(cls, data, *, preferred_type=None):
        ...

    @classmethod
    @abc.abstractmethod
    def loads(cls, s, *, default_serializer=None, serializer=None):
        ...

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def dumps(self, *, serializer=None):
        ...

    @abc.abstractmethod
    def derive(self, *objects, **fields):
        ...

    @abc.abstractmethod
    def to_representation(self):
        ...

    @abc.abstractmethod
    def is_valid(self):
        ...

    @abc.abstractmethod
    def validate(self):
        ...

    @abc.abstractmethod
    def validate_or_raise(self):
        ...

    @property
    @abc.abstractmethod
    def validation_errors(self):
        ...

class FieldDescriptorT(Generic[T]):
    required = True
    default = None

    @abc.abstractmethod
    def __init__(self, *, field=None, input_name=None, output_name=None, type=None, model=None, required=True, default=None, parent=None, exclude=None, date_parser=None, **kwargs):
        self.date_parser = cast(Callable[[Any], datetime], date_parser)

    @abc.abstractmethod
    def on_model_attached(self):
        ...

    @abc.abstractmethod
    def clone(self, **kwargs):
        ...

    @abc.abstractmethod
    def as_dict(self):
        ...

    @abc.abstractmethod
    def validate_all(self, value):
        ...

    @abc.abstractmethod
    def validate(self, value):
        ...

    @abc.abstractmethod
    def to_python(self, value):
        ...

    @abc.abstractmethod
    def prepare_value(self, value):
        ...

    @abc.abstractmethod
    def should_coerce(self, value):
        ...

    @abc.abstractmethod
    def getattr(self, obj):
        ...

    @abc.abstractmethod
    def validation_error(self, reason):
        ...

    @property
    @abc.abstractmethod
    def ident(self):
        ...

    @cached_property
    @abc.abstractmethod
    def related_models(self):
        ...

    @cached_property
    @abc.abstractmethod
    def lazy_coercion(self):
        ...
ModelArg = Union[Type[ModelT], Type[bytes], Type[str]]