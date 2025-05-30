"""Model descriptions.

The model describes the components of a data structure, kind of like a struct
in C, but there's no limitation of what type of data structure the model is,
or what it's used for.

A record (faust.models.record) is a model type that serialize into
dictionaries, so the model describe the fields, and their types:

.. sourcecode:: pycon

    >>> class Point(Record):
    ...    x: int
    ...    y: int

    >>> p = Point(10, 3)
    >>> assert p.x == 10
    >>> assert p.y == 3
    >>> p
    <Point: x=10, y=3>
    >>> payload = p.dumps(serializer='json')
    '{"x": 10, "y": 3, "__faust": {"ns": "__main__.Point"}}'
    >>> p2 = Record.loads(payload)
    >>> p2
    <Point: x=10, y=3>

Models are mainly used for describing the data in messages: both keys and
values can be described as models.
"""
import abc
import warnings

from datetime import datetime
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from mode.utils.objects import canoname

from faust.exceptions import ValidationError
from faust.serializers.codecs import CodecArg, dumps, loads
from faust.types.models import (
    CoercionMapping,
    FieldDescriptorT,
    FieldMap,
    ModelOptions,
    ModelT,
)

__all__ = ['Model', 'maybe_model', 'registry']

# NOTES:
# - Records are described in the same notation as named tuples in Python 3.6.
#   To accomplish this ``__init_subclass__`` defined in :pep:`487` is used.
#
#   When accessed on the Record class, the attributes are actually field
#   descriptors that return information about the field:
#       >>> Point.x
#       <FieldDescriptor: Point.x: int>
#
#   This field descriptor holds information about the name of the field, the
#   value type of the field, and also what Record subclass it belongs to.
#
# - Sometimes field descriptions are passed around as arguments to functions.
#
# - A stream of deposits may be joined with a stream of orders if
#   both have an ``account`` field.  Field descriptors are used to
#   specify the field.
#
# - order_instance.account is data
#   (it holds the string account for this particular order).
#
# - order_instance.__class__.account is the field descriptor for the field,
#   it's not data but metadata that enables introspection, and it can be
#   passed around to describe a field we want to extract or similar.
#
# - FieldDescriptors are Python descriptors: In Python object
#   attributes can override what happens when they are get/set/deleted:
#
#       class MyDescriptor:
#
#           def __get__(self, instance: Any, cls: Type[Any]) -> Any:
#               if instance is None:
#                   print('ACCESS ON CLASS ATTRIBUTE')
#                   return self
#               print('ACCESS ON INSTANCE')
#               return 42
#
#       class Example:
#           foo = MyDescriptor()
#
#   The above descriptor overrides __get__, which is called when the attribute
#   is accessed (a descriptor may also override __set__ and __del__).
#
#
#   You can see the difference in what happens when you access the attribute
#   on the class, vs. the instance:
#
#       >>> Example.foo
#       ACCESS ON CLASS ATTRIBUTE
#       <__main__.MyDescriptor at 0x1049caac8>
#
#       >>> x = Example()
#       >>> x.foo
#       ACCESS ON INSTANCE
#       42


E_ABSTRACT_INSTANCE: str = '''
Cannot instantiate abstract model.

If this model is used as the field of another model,
and you meant to define a polymorphic relationship: make sure
your abstract model class has the `polymorphic_fields` option enabled:

    class {name}(faust.Record, abstract=True, polymorphic_fields=True):
        ...
'''

#: Global map of namespace -> Model, used to find model classes by name.
#: Every single model defined is added here automatically when a model
#: class is defined.
registry: MutableMapping[str, Type[ModelT]] = {}


def maybe_model(arg: Any) -> Any:
    """Convert argument to model if possible."""
    try:
        model: Type[ModelT] = registry[arg['__faust']['ns']]
    except (KeyError, TypeError):
        return arg
    else:
        return model.from_data(arg)


MT = TypeVar('MT', bound='Model')


class Model(ModelT):
    """Meta description model for serialization."""

    #: Set to True if this is an abstract base class.
    __is_abstract__: ClassVar[bool] = True

    __validation_errors__: Optional[List[ValidationError]] = None

    _pending_finalizers: ClassVar[Optional[List[Callable[[], None]]]] = None

    #: Serialized data may contain a "blessed key" that mandates
    #: how the data should be deserialized.  This probably only
    #: applies to records, but we need to support it at Model level.
    #: The blessed key has a dictionary value with a ``ns`` key:
    #:   data = {.., '__faust': {'ns': 'examples.simple.Withdrawal'}}
    #: When ``Model._maybe_reconstruct` sees this key it will look
    #: up that namespace in the :data:`registry`, and if it exists
    #: select it as the target model to use for serialization.
    #:
    #: Is this similar to how unsafe deserialization in pickle/yaml/etc.
    #: works?  No! pickle/pyyaml allow for arbitrary types to be
    #: deserialized (and worse in pickle's case), whereas the blessed
    #: key can only deserialize to a hardcoded list of types that are
    #: already under the remote control of messages anyway.
    #: For example it's not possible to perform remote code execution
    #: by providing a blessed key namespace of "os.system", simply
    #: because os.system is not in the registry of allowed types.
    _blessed_key: str = '__faust'

    @classmethod
    def _maybe_namespace(
        cls,
        data: Any,
        *,
        preferred_type: Optional[Type[ModelT]] = None,
        fast_types: Tuple[Type[Any], ...] = (bytes, str),
        isinstance_func: Optional[Callable[[Any, Tuple[Type[Any], ...]], bool]] = None,
    ) -> Optional[Type[ModelT]]:
        if isinstance_func is None:
            isinstance_func = isinstance
        # The serialized data may contain a ``__faust`` blessed key
        # holding the name of the model it should be deserialized as.
        # So even if value_type=MyModel, the data may mandata that it
        # should be deserialized using "foo.bar.baz" instead.

        # This is how we deal with Kafka's lack of message headers,
        # as needed by the RPC mechanism, without wrapping all data.
        if data is None or isinstance_func(data, fast_types):
            return None
        try:
            ns: str = data[cls._blessed_key]['ns']
        except (KeyError, TypeError):
            pass
        else:
            # we only allow blessed keys when type=None, or type=Model
            type_is_abstract: bool = (
                preferred_type is None
                or preferred_type is ModelT
                or preferred_type is Model
            )
            try:
                model: Type[ModelT] = registry[ns]
            except KeyError:
                if type_is_abstract:
                    raise
                return None
            else:
                if (
                    type_is_abstract
                    or model._options.allow_blessed_key
                    or model._options.polymorphic_fields
                ):
                    return model
        return None

    @classmethod
    def _maybe_reconstruct(cls, data: Any) -> Any:
        model: Optional[Type[ModelT]] = cls._maybe_namespace(data)
        return model.from_data(data) if model else data

    @classmethod
    def _from_data_field(cls, data: Any) -> Optional['Model']:
        if data is not None:
            if cls.__is_abstract__:
                return cls._maybe_reconstruct(data)
            return cls.from_data(data, preferred_type=cls)
        return None

    @classmethod
    def loads(
        cls: Type[MT],
        s: bytes,
        *,
        default_serializer: Optional[CodecArg] = None,  # XXX use serializer
        serializer: Optional[CodecArg] = None
    ) -> MT:
        """Deserialize model object from bytes.

        Keyword Arguments:
            serializer (CodecArg): Default serializer to use
                if no custom serializer was set for this model subclass.
        """
        if default_serializer is not None:
            warnings.warn(
                DeprecationWarning(
                    'default_serializer deprecated, use: serializer'
                )
            )
        ser: Optional[CodecArg] = cls._options.serializer or serializer or default_serializer
        data: Any = loads(ser, s)
        return cls.from_data(data)

    def __init_subclass__(
        cls: Type[MT],
        serializer: Optional[str] = None,
        namespace: Optional[str] = None,
        include_metadata: Optional[bool] = None,
        isodates: Optional[bool] = None,
        abstract: bool = False,
        allow_blessed_key: Optional[bool] = None,
        decimals: Optional[bool] = None,
        coerce: Optional[bool] = None,
        coercions: Optional[CoercionMapping] = None,
        polymorphic_fields: Optional[bool] = None,
        validation: Optional[bool] = None,
        date_parser: Optional[Callable[[Any], datetime]] = None,
        lazy_creation: bool = False,
        **kwargs: Any
    ) -> None:
        # Python 3.6 added the new __init_subclass__ function that
        # makes it possible to initialize subclasses without using
        # metaclasses (:pep:`487`).
        super().__init_subclass__(**kwargs)

        # mypy does not recognize `__init_subclass__` as a classmethod
        # and thinks we're mutating a ClassVar when setting:
        #   cls.__is_abstract__ = False
        # To fix this we simply delegate to a _init_subclass classmethod.
        finalizer: Callable[[], None] = partial(
            cls._init_subclass,
            serializer,
            namespace,
            include_metadata,
            isodates,
            abstract,
            allow_blessed_key,
            decimals,
            coerce,
            coercions,
            polymorphic_fields,
            validation,
            date_parser,
        )
        if lazy_creation:
            if cls._pending_finalizers is None:
                cls._pending_finalizers = []
            cls._pending_finalizers.append(finalizer)
        else:
            cls._pending_finalizers = None
            finalizer()

    @classmethod
    def make_final(cls) -> None:
        pending: Optional[List[Callable[[], None]]] = cls._pending_finalizers
        cls._pending_finalizers = None
        if pending:
            for finalizer in pending:
                finalizer()

    @classmethod
    def _init_subclass(
        cls,
        serializer: Optional[str],
        namespace: Optional[str],
        include_metadata: Optional[bool],
        isodates: Optional[bool],
        abstract: bool,
        allow_blessed_key: Optional[bool],
        decimals: Optional[bool],
        coerce: Optional[bool],
        coercions: Optional[CoercionMapping],
        polymorphic_fields: Optional[bool],
        validation: Optional[bool],
        date_parser: Optional[Callable[[Any], datetime]],
    ) -> None:
        # Can set serializer/namespace/etc. using:
        #    class X(Record, serializer='json', namespace='com.vandelay.X'):
        #        ...
        try:
            custom_options = cls.Options  # type: ignore
        except AttributeError:
            custom_options = None
        else:
            delattr(cls, 'Options')
        options: Optional[ModelOptions] = getattr(cls, '_options', None)
        if options is None:
            options = ModelOptions()
            options.coercions = {}
            options.defaults = {}
        else:
            options = options.clone_defaults()
        if custom_options:
            options.__dict__.update(custom_options.__dict__)
        if coerce is not None:
            options.coerce = coerce
        if coercions is not None:
            options.coercions.update(coercions)
        if serializer is not None:
            options.serializer = serializer
        if include_metadata is not None:
            options.include_metadata = include_metadata
        if isodates is not None:
            options.isodates = isodates
        if decimals is not None:
            options.decimals = decimals
        if allow_blessed_key is not None:
            options.allow_blessed_key = allow_blessed_key
        if polymorphic_fields is not None:
            options.polymorphic_fields = polymorphic_fields
        if validation is not None:
            options.validation = validation
            options.coerce = True  # validation implies coerce
        if date_parser is not None:
            options.date_parser = date_parser

        options.namespace = namespace or canoname(cls)

        if abstract:
            # Custom base classes can set this to skip class initialization.
            cls.__is_abstract__ = True
            cls._options = options
            cls.__init__ = cls.__abstract_init__  # type: ignore
            return
        cls.__is_abstract__ = False

        # Add introspection capabilities
        cls._contribute_to_options(options)
        # Add FieldDescriptors for every field.
        options.descriptors = cls._contribute_field_descriptors(
            cls, options
        )

        # Store options on new subclass.
        cls._options = options

        cls._contribute_methods()

        # Register in the global registry, so we can look up
        # models by namespace.
        registry[options.namespace] = cls

        codegens: List[Tuple[str, Callable[[], Callable[..., Any]], str]] = [
            ('__init__', cls._BUILD_init, '_model_init'),
            ('__hash__', cls._BUILD_hash, '_model_hash'),
            ('__eq__', cls._BUILD_eq, '_model_eq'),
            ('__ne__', cls._BUILD_ne, '_model_ne'),
            ('__gt__', cls._BUILD_gt, '_model_gt'),
            ('__ge__', cls._BUILD_ge, '_model_ge'),
            ('__lt__', cls._BUILD_lt, '_model_lt'),
            ('__le__', cls._BUILD_le, '_model_le'),
        ]

        for meth_name, meth_gen, attr_name in codegens:
            # self._model_init = cls._BUILD_init()
            # if '__init__' not in cls.__dict__:
            #     cls.__init__ = self._model_init
            meth: Callable[..., Any] = meth_gen()
            setattr(cls, attr_name, meth)
            if meth_name not in cls.__dict__:
                setattr(cls, meth_name, meth)

    def __abstract_init__(self) -> None:
        raise NotImplementedError(
            E_ABSTRACT_INSTANCE.format(
                name=type(self).__name__,
            )
        )

    @classmethod
    @abc.abstractmethod
    def _contribute_to_options(
        cls, options: ModelOptions
    ) -> None:  # pragma: no cover
        ...

    @classmethod
    def _contribute_methods(cls) -> None:  # pragma: no cover
        ...

    @classmethod
    @abc.abstractmethod
    def _contribute_field_descriptors(
        cls,
        target: Type[Any],
        options: ModelOptions,
        parent: Optional[FieldDescriptorT] = None,
    ) -> FieldMap:  # pragma: no cover
        ...

    @classmethod
    @abc.abstractmethod
    def _BUILD_init(cls) -> Callable[[], None]:  # pragma: no cover
        ...

    @classmethod
    @abc.abstractmethod
    def _BUILD_hash(cls) -> Callable[[], int]:  # pragma: no cover
        ...

    @classmethod
    @abc.abstractmethod
    def _BUILD_eq(cls) -> Callable[[Any], bool]:  # pragma: no cover
        ...

    @classmethod
    @abc.abstractmethod
    def _BUILD_ne(cls) -> Callable[[Any], bool]:  # pragma: no cover
        ...

    @classmethod
    @abc.abstractmethod
    def _BUILD_gt(cls) -> Callable[[Any], bool]:  # pragma: no cover
        ...

    @classmethod
    @abc.abstractmethod
    def _BUILD_ge(cls) -> Callable[[Any], bool]:  # pragma: no cover
        ...

    @classmethod
    @abc.abstractmethod
    def _BUILD_lt(cls) -> Callable[[Any], bool]:  # pragma: no cover
        ...

    @classmethod
    @abc.abstractmethod
    def _BUILD_le(cls) -> Callable[[Any], bool]:  # pragma: no cover
        ...

    @abc.abstractmethod
    def to_representation(self) -> Any:  # pragma: no cover
        """Convert object to JSON serializable object."""

    @abc.abstractmethod
    def _humanize(self) -> str:  # pragma: no cover
        """Return string representation of object for debugging purposes."""
        ...

    def is_valid(self) -> bool:
        return True if not self.validate() else False

    def validate(self) -> List[ValidationError]:
        errors: Optional[List[ValidationError]] = self.__validation_errors__
        if errors is None:
            errors = self.__validation_errors__ = list(self._itervalidate())
        return errors

    def validate_or_raise(self) -> None:
        errors: List[ValidationError] = self.validate()
        if errors:
            raise errors[0]

    def _itervalidate(self) -> Iterable[ValidationError]:
        for name, descr in self._options.descriptors.items():
            value: Any = getattr(self, name)
            yield from descr.validate_all(value)

    @property
    def validation_errors(self) -> List[ValidationError]:
        return self.validate()

    def derive(
        self: MT, *objects: MT, **fields: Any
    ) -> MT:
        """Derive new model with certain fields changed."""
        return self._derive(*objects, **fields)

    @abc.abstractmethod
    def _derive(
        self: MT, *objects: MT, **fields: Any
    ) -> MT:  # pragma: no cover
        raise NotImplementedError()

    def dumps(self, *, serializer: Optional[CodecArg] = None) -> bytes:
        """Serialize object to the target serialization format."""
        return dumps(serializer or self._options.serializer, self.to_representation())

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self._humanize()}>'
