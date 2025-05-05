"""Record - Dictionary Model."""
from datetime import datetime
from decimal import Decimal
from itertools import chain
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, MutableMapping, Optional, Set, Tuple, Type, cast
from mode.utils.compat import OrderedDict
from mode.utils.objects import annotations, is_optional, remove_optional
from mode.utils.text import pluralize
from faust.types.models import (
    CoercionMapping,
    FieldDescriptorT,
    FieldMap,
    IsInstanceArgT,
    ModelOptions,
    ModelT,
)
from faust.utils import codegen
from .base import Model
from .fields import FieldDescriptor, field_for_type
from .tags import Tag

__all__ = ['Record']

DATE_TYPES = (datetime,)
DECIMAL_TYPES = (Decimal,)
ALIAS_FIELD_TYPES: Dict[Type, Type] = {
    dict: Dict,
    tuple: Tuple,
    list: List,
    set: Set,
    frozenset: FrozenSet,
}
E_NON_DEFAULT_FOLLOWS_DEFAULT: str = '\nNon-default {cls_name} field {field_name} cannot\nfollow default {fields} {default_names}\n'
_ReconFun = Callable[..., Any]


class Record(Model, abstract=True):
    """Describes a model type that is a record (Mapping).

    Examples:
        >>> class LogEvent(Record, serializer='json'):
        ...     severity: str
        ...     message: str
        ...     timestamp: float
        ...     optional_field: str = 'default value'

        >>> event = LogEvent(
        ...     severity='error',
        ...     message='Broken pact',
        ...     timestamp=666.0,
        ... )

        >>> event.severity
        'error'

        >>> serialized = event.dumps()
        '{"severity": "error", "message": "Broken pact", "timestamp": 666.0}'

        >>> restored = LogEvent.loads(serialized)
        <LogEvent: severity='error', message='Broken pact', timestamp=666.0>

        >>> # You can also subclass a Record to create a new record
        >>> # with additional fields
        >>> class RemoteLogEvent(LogEvent):
        ...     url: str

        >>> # You can also refer to record fields and pass them around:
        >>> LogEvent.severity
        >>> <FieldDescriptor: LogEvent.severity (str)>
    """

    def __init_subclass__(
        cls,
        serializer: Optional[Any] = None,
        namespace: Optional[Any] = None,
        include_metadata: Optional[Any] = None,
        isodates: Optional[Any] = None,
        abstract: bool = False,
        allow_blessed_key: Optional[Any] = None,
        decimals: Optional[Any] = None,
        coerce: Optional[Any] = None,
        coercions: Optional[CoercionMapping] = None,
        polymorphic_fields: Optional[Any] = None,
        validation: Optional[Any] = None,
        date_parser: Optional[Any] = None,
        lazy_creation: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(
            serializer=serializer,
            namespace=namespace,
            include_metadata=include_metadata,
            isodates=isodates,
            abstract=abstract,
            allow_blessed_key=allow_blessed_key,
            decimals=decimals,
            coerce=coerce,
            coercions=coercions,
            polymorphic_fields=polymorphic_fields,
            validation=validation,
            date_parser=date_parser,
            lazy_creation=lazy_creation,
            **kwargs,
        )

    @classmethod
    def _contribute_to_options(cls, options: ModelOptions) -> None:
        fields, defaults = annotations(
            cls,
            stop=Record,
            skip_classvar=True,
            alias_types=ALIAS_FIELD_TYPES,
            localns={cls.__name__: cls},
        )
        options.fields = cast(Mapping[str, Any], fields)
        options.fieldset = frozenset(fields)
        options.fieldpos = {i: k for i, k in enumerate(fields.keys())}
        options.defaults = {
            k: v.default if isinstance(v, FieldDescriptor) else v
            for k, v in defaults.items()
            if k in fields and not (isinstance(v, FieldDescriptor) and v.required)
        }
        local_defaults: List[str] = []
        for attr_name in cls.__annotations__:
            if attr_name in cls.__dict__:
                default_value = cls.__dict__[attr_name]
                if isinstance(default_value, FieldDescriptorT):
                    if not default_value.required:
                        local_defaults.append(attr_name)
                else:
                    local_defaults.append(attr_name)
            elif local_defaults:
                raise TypeError(
                    E_NON_DEFAULT_FOLLOWS_DEFAULT.format(
                        cls_name=cls.__name__,
                        field_name=attr_name,
                        fields=pluralize(len(local_defaults), 'field'),
                        default_names=', '.join(local_defaults),
                    )
                )
        for field, typ in fields.items():
            if is_optional(typ):
                options.defaults.setdefault(field, None)
        options.optionalset = frozenset(options.defaults)

    @classmethod
    def _contribute_methods(cls) -> None:
        if not getattr(cls.asdict, 'faust_generated', False):
            raise RuntimeError('Not allowed to override Record.asdict()')
        cls.asdict = cls._BUILD_asdict()
        cls.asdict.faust_generated = True
        cls._input_translate_fields = cls._BUILD_input_translate_fields()

    @classmethod
    def _contribute_field_descriptors(
        cls, target: Type['Record'], options: ModelOptions, parent: Optional[Type['Record']] = None
    ) -> Dict[str, FieldDescriptorT]:
        fields = options.fields
        defaults = options.defaults
        date_parser = options.date_parser
        coerce = options.coerce
        index: Dict[str, FieldDescriptorT] = {}
        secret_fields: Set[str] = set()
        sensitive_fields: Set[str] = set()
        personal_fields: Set[str] = set()
        tagged_fields: Set[str] = set()

        def add_to_tagged_indices(field: str, tag: Tag) -> None:
            if tag.is_secret:
                options.has_secret_fields = True
                secret_fields.add(field)
            if tag.is_sensitive:
                options.has_sensitive_fields = True
                sensitive_fields.add(field)
            if tag.is_personal:
                options.has_personal_fields = True
                personal_fields.add(field)
            options.has_tagged_fields = True
            tagged_fields.add(field)

        def add_related_to_tagged_indices(field: str, related_model: Optional[Type['Record']] = None) -> None:
            if related_model is None:
                return
            try:
                related_options = related_model._options
            except AttributeError:
                return
            if related_options.has_secret_fields:
                options.has_secret_fields = True
                secret_fields.add(field)
            if related_options.has_sensitive_fields:
                options.has_sensitive_fields = True
                sensitive_fields.add(field)
            if related_options.has_personal_fields:
                options.has_personal_fields = True
                personal_fields.add(field)
            if related_options.has_tagged_fields:
                options.has_tagged_fields = True
                tagged_fields.add(field)

        for field, typ in fields.items():
            try:
                default, needed = (defaults[field], False)
            except KeyError:
                default, needed = (None, True)
            descr = getattr(target, field, None)
            if is_optional(typ):
                target_type = remove_optional(typ)
            else:
                target_type = typ
            if descr is None or not isinstance(descr, FieldDescriptorT):
                DescriptorType, tag = field_for_type(target_type)
                if tag:
                    add_to_tagged_indices(field, tag)
                descr = DescriptorType(
                    field=field,
                    type=typ,
                    model=cls,
                    required=needed,
                    default=default,
                    parent=parent,
                    coerce=coerce,
                    model_coercions=options.coercions,
                    date_parser=date_parser,
                    tag=tag,
                )
            else:
                descr = descr.clone(
                    field=field,
                    type=typ,
                    model=cls,
                    required=needed,
                    default=default,
                    parent=parent,
                    coerce=coerce,
                    model_coercions=options.coercions,
                )
            descr.on_model_attached()
            for related_model in descr.related_models:
                add_related_to_tagged_indices(field, related_model)
            setattr(target, field, descr)
            index[field] = descr
        options.secret_fields = frozenset(secret_fields)
        options.sensitive_fields = frozenset(sensitive_fields)
        options.personal_fields = frozenset(personal_fields)
        options.tagged_fields = frozenset(tagged_fields)
        return index

    @classmethod
    def from_data(
        cls, data: Any, *, preferred_type: Optional[Type[Any]] = None
    ) -> 'Record':
        """Create model object from Python dictionary."""
        if hasattr(data, '__is_model__'):
            return cast(Record, data)
        else:
            self_cls = cls._maybe_namespace(data, preferred_type=preferred_type)
        cls._input_translate_fields(data)
        return (self_cls or cls)(**data, __strict__=False)

    def __init__(
        self, *args: Any, __strict: bool = True, __faust: Optional[Any] = None, **kwargs: Any
    ) -> None:
        ...

    @classmethod
    def _BUILD_input_translate_fields(cls) -> Callable[..., None]:
        translate: List[str] = [
            f'data[{field!r}] = data.pop({d.input_name!r}, None)'
            for field, d in cls._options.descriptors.items()
            if d.field != d.input_name
        ]
        return cast(
            Callable[..., None],
            classmethod(
                codegen.Function(
                    '_input_translate_fields',
                    ['cls', 'data'],
                    translate if translate else ['pass'],
                    globals=globals(),
                    locals=locals(),
                )
            ),
        )

    @classmethod
    def _BUILD_init(cls) -> Callable[..., None]:
        options = cls._options
        field_positions = options.fieldpos
        optional = options.optionalset
        needs_validation = options.validation
        descriptors = options.descriptors
        has_post_init = hasattr(cls, '__post_init__')
        closures: Dict[str, str] = {
            '__defaults__': 'Model._options.defaults',
            '__descr__': 'Model._options.descriptors',
        }
        kwonlyargs: List[str] = ['*', '__strict__=True', '__faust=None', '**kwargs']
        optional_fields: OrderedDict[str, bool] = OrderedDict()
        required_fields: OrderedDict[str, bool] = OrderedDict()

        def generate_setter(field: str, getval: str) -> str:
            """Generate code that sets attribute for field in class.

            Arguments:
                field: Name of field.
                getval: Source code that initializes value for field,
                    can be the field name itself for no initialization
                    or for example: ``f"self._prepare_value({field})"``
            """
            if field in optional:
                optional_fields[field] = True
                default_var = f'_default_{field}_'
                closures[default_var] = f'__defaults__["{field}"]'
                return f'    self.{field} = {getval} if {field} is not None else {default_var}'
            else:
                required_fields[field] = True
                return f'    self.{field} = {getval}'

        def generate_prepare_value(field: str) -> str:
            descriptor = descriptors[field]
            if descriptor.lazy_coercion:
                return field
            else:
                init_field_var = f'_init_{field}_'
                closures[init_field_var] = f'__descr__["{field}"].to_python'
                return f'{init_field_var}({field})'

        preamble: List[str] = ['self.__evaluated_fields__ = set()']
        data_setters: List[str] = ['if __strict__:'] + [
            generate_setter(field, field) for field in field_positions.values()
        ]
        data_rest: List[str] = [
            '    if kwargs:',
            '        from mode.utils.text import pluralize',
            '        message = "{} got unexpected {}: {}".format(',
            '            self.__class__.__name__,',
            '            pluralize(kwargs.__len__(), "argument"),',
            '            ", ".join(map(str, sorted(kwargs))))',
            '        raise TypeError(message)',
        ]
        init_setters: List[str] = ['else:']
        if field_positions:
            init_setters.extend(
                generate_setter(field, generate_prepare_value(field))
                for field in field_positions.values()
            )
        init_setters.append('    self.__dict__.update(kwargs)')
        postamble: List[str] = []
        if has_post_init:
            postamble.append('self.__post_init__()')
        if needs_validation:
            postamble.append('self.validate_or_raise()')
        signature: List[str] = list(
            chain(
                ['self'],
                [f'{field}' for field in required_fields],
                [f'{field}=None' for field in optional_fields],
                kwonlyargs,
            )
        )
        sourcecode: str = codegen.build_closure_source(
            name='__init__',
            args=signature,
            body=list(chain(preamble, data_setters, data_rest, init_setters, postamble)),
            closures=closures,
            outer_args=['Model'],
        )
        return codegen.build_closure('__outer__', sourcecode, cls, globals={}, locals={})

    @classmethod
    def _BUILD_hash(cls) -> Callable[['Record'], int]:
        return codegen.HashMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_eq(cls) -> Callable[['Record', Any], bool]:
        return codegen.EqMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_ne(cls) -> Callable[['Record', Any], bool]:
        return codegen.NeMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_gt(cls) -> Callable[['Record', Any], bool]:
        return codegen.GtMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_ge(cls) -> Callable[['Record', Any], bool]:
        return codegen.GeMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_lt(cls) -> Callable[['Record', Any], bool]:
        return codegen.LtMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_le(cls) -> Callable[['Record', Any], bool]:
        return codegen.LeMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_asdict(cls) -> Callable[['Record'], Dict[str, Any]]:
        preamble: List[str] = ['return self._prepare_dict({']
        fields: List[str] = [
            f'  {d.output_name!r}: {cls._BUILD_asdict_field(name, d)},'
            for name, d in cls._options.descriptors.items()
            if not d.exclude
        ]
        postamble: List[str] = ['})']
        return codegen.Method(
            '_asdict',
            [],
            preamble + fields + postamble,
            globals=globals(),
            locals=locals(),
        )

    def _prepare_dict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    @classmethod
    def _BUILD_asdict_field(cls, name: str, field: FieldDescriptorT) -> str:
        return f'self.{name}'

    def _derive(self, *objects: 'Record', **fields: Any) -> 'Record':
        data: Dict[str, Any] = self.asdict()
        for obj in objects:
            data.update(cast(Record, obj).asdict())
        return type(self)(**{**data, **fields})

    def to_representation(self) -> Dict[str, Any]:
        """Convert model to its Python generic counterpart.

        Records will be converted to dictionary.
        """
        payload: Dict[str, Any] = self.asdict()
        options: ModelOptions = self._options
        if options.include_metadata:
            payload['__faust'] = {'ns': options.namespace}
        return payload

    def asdict(self) -> Dict[str, Any]:
        """Convert record to Python dictionary."""
        ...
    asdict.faust_generated = True

    def _humanize(self) -> str:
        attrs: Dict[str, Any] = self.__dict__
        defaults_items = self._options.defaults.items()
        fields: Dict[str, Any] = {
            **{k: v for k, v in attrs.items() if not k.startswith('__')},
            **{k: v for k, v in defaults_items if k not in attrs},
        }
        return _kvrepr(fields)

    def __json__(self) -> Dict[str, Any]:
        return self.to_representation()

    def __eq__(self, other: Any) -> bool:
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        return NotImplemented


def _kvrepr(d: Dict[str, Any], *, sep: str = ', ') -> str:
    """Represent dict as `k='v'` pairs separated by comma."""
    return sep.join((f'{k}={v!r}' for k, v in d.items()))
