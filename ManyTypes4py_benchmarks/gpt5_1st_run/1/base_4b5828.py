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
from typing import Any, Callable, ClassVar, Dict, Iterable, List, MutableMapping, Optional, Tuple, Type
from mode.utils.objects import canoname
from faust.exceptions import ValidationError
from faust.serializers.codecs import CodecArg, dumps, loads
from faust.types.models import CoercionMapping, FieldDescriptorT, FieldMap, ModelOptions, ModelT

__all__: List[str] = ['Model', 'maybe_model', 'registry']

E_ABSTRACT_INSTANCE: str = '\nCannot instantiate abstract model.\n\nIf this model is used as the field of another model,\nand you meant to define a polymorphic relationship: make sure\nyour abstract model class has the `polymorphic_fields` option enabled:\n\n    class {name}(faust.Record, abstract=True, polymorphic_fields=True):\n        ...\n'

registry: Dict[str, Type['Model']] = {}


def maybe_model(arg: Any) -> Any:
    """Convert argument to model if possible."""
    try:
        model = registry[arg['__faust']['ns']]
    except (KeyError, TypeError):
        return arg
    else:
        return model.from_data(arg)


class Model(ModelT):
    """Meta description model for serialization."""

    __is_abstract__: ClassVar[bool] = True
    __validation_errors__: Optional[List[ValidationError]] = None
    _pending_finalizers: Optional[List[Callable[[], None]]] = None
    _blessed_key: ClassVar[str] = '__faust'

    @classmethod
    def _maybe_namespace(
        cls,
        data: Any,
        *,
        preferred_type: Optional[Type[Any]] = None,
        fast_types: Tuple[Type[Any], ...] = (bytes, str),
        isinstance: Callable[[Any, Any], bool] = isinstance,
    ) -> Optional[Type['Model']]:
        if data is None or isinstance(data, fast_types):
            return None
        try:
            ns = data[cls._blessed_key]['ns']
        except (KeyError, TypeError):
            pass
        else:
            type_is_abstract = preferred_type is None or preferred_type is ModelT or preferred_type is Model
            try:
                model = registry[ns]
            except KeyError:
                if type_is_abstract:
                    raise
                return None
            else:
                if type_is_abstract or model._options.allow_blessed_key or model._options.polymorphic_fields:
                    return model
        return None

    @classmethod
    def _maybe_reconstruct(cls, data: Any) -> Any:
        model = cls._maybe_namespace(data)
        return model.from_data(data) if model else data

    @classmethod
    def _from_data_field(cls, data: Any) -> Any:
        if data is not None:
            if cls.__is_abstract__:
                return cls._maybe_reconstruct(data)
            return cls.from_data(data, preferred_type=cls)
        return None

    @classmethod
    def loads(
        cls,
        s: bytes,
        *,
        default_serializer: Optional[CodecArg] = None,
        serializer: Optional[CodecArg] = None,
    ) -> 'Model':
        """Deserialize model object from bytes.

        Keyword Arguments:
            serializer (CodecArg): Default serializer to use
                if no custom serializer was set for this model subclass.
        """
        if default_serializer is not None:
            warnings.warn(DeprecationWarning('default_serializer deprecated, use: serializer'))
        ser = cls._options.serializer or serializer or default_serializer
        data = loads(ser, s)
        return cls.from_data(data)

    def __init_subclass__(
        self: Type['Model'],
        serializer: Optional[CodecArg] = None,
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
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        finalizer: Callable[[], None] = partial(
            self._init_subclass,
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
            self._pending_finalizers = [finalizer]
        else:
            self._pending_finalizers = None
            finalizer()

    @classmethod
    def make_final(cls) -> None:
        pending, cls._pending_finalizers = (cls._pending_finalizers, None)
        if pending:
            for finalizer in pending:
                finalizer()

    @classmethod
    def _init_subclass(
        cls,
        serializer: Optional[CodecArg] = None,
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
    ) -> None:
        try:
            custom_options = cls.Options  # type: ignore[attr-defined]
        except AttributeError:
            custom_options = None
        else:
            delattr(cls, 'Options')
        options = getattr(cls, '_options', None)
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
            options.coerce = True
        if date_parser is not None:
            options.date_parser = date_parser
        options.namespace = namespace or canoname(cls)
        if abstract:
            cls.__is_abstract__ = True
            cls._options = options
            cls.__init__ = cls.__abstract_init__  # type: ignore[assignment]
            return
        cls.__is_abstract__ = False
        cls._contribute_to_options(options)
        options.descriptors = cls._contribute_field_descriptors(cls, options)
        cls._options = options
        cls._contribute_methods()
        registry[options.namespace] = cls
        codegens = [
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
            meth = meth_gen()
            setattr(cls, attr_name, meth)
            if meth_name not in cls.__dict__:
                setattr(cls, meth_name, meth)

    def __abstract_init__(self) -> None:
        raise NotImplementedError(E_ABSTRACT_INSTANCE.format(name=type(self).__name__))

    @classmethod
    @abc.abstractmethod
    def _contribute_to_options(cls, options: ModelOptions) -> None:
        ...

    @classmethod
    def _contribute_methods(cls) -> None:
        ...

    @classmethod
    @abc.abstractmethod
    def _contribute_field_descriptors(
        cls,
        target: Type['Model'],
        options: ModelOptions,
        parent: Optional[Type['Model']] = None,
    ) -> FieldMap:
        ...

    @classmethod
    @abc.abstractmethod
    def _BUILD_init(cls) -> Callable[..., Any]:
        ...

    @classmethod
    @abc.abstractmethod
    def _BUILD_hash(cls) -> Callable[..., Any]:
        ...

    @classmethod
    @abc.abstractmethod
    def _BUILD_eq(cls) -> Callable[..., Any]:
        ...

    @abc.abstractmethod
    def to_representation(self) -> Any:
        """Convert object to JSON serializable object."""

    @abc.abstractmethod
    def _humanize(self) -> str:
        """Return string representation of object for debugging purposes."""
        ...

    def is_valid(self) -> bool:
        return True if not self.validate() else False

    def validate(self) -> List[ValidationError]:
        errors = self.__validation_errors__
        if errors is None:
            errors = self.__validation_errors__ = list(self._itervalidate())
        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise errors[0]

    def _itervalidate(self) -> Iterable[ValidationError]:
        for name, descr in self._options.descriptors.items():
            yield from descr.validate_all(getattr(self, name))

    @property
    def validation_errors(self) -> List[ValidationError]:
        return self.validate()

    def derive(self, *objects: Any, **fields: Any) -> 'Model':
        """Derive new model with certain fields changed."""
        return self._derive(*objects, **fields)

    @abc.abstractmethod
    def _derive(self, *objects: Any, **fields: Any) -> 'Model':
        raise NotImplementedError()

    def dumps(self, *, serializer: Optional[CodecArg] = None) -> bytes:
        """Serialize object to the target serialization format."""
        return dumps(serializer or self._options.serializer, self.to_representation())

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self._humanize()}>'