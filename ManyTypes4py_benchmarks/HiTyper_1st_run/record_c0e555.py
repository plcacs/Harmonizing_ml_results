"""Record - Dictionary Model."""
from datetime import datetime
from decimal import Decimal
from itertools import chain
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, MutableMapping, Set, Tuple, Type, cast
from mode.utils.compat import OrderedDict
from mode.utils.objects import annotations, is_optional, remove_optional
from mode.utils.text import pluralize
from faust.types.models import CoercionMapping, FieldDescriptorT, FieldMap, IsInstanceArgT, ModelOptions, ModelT
from faust.utils import codegen
from .base import Model
from .fields import FieldDescriptor, field_for_type
from .tags import Tag
__all__ = ['Record']
DATE_TYPES = (datetime,)
DECIMAL_TYPES = (Decimal,)
ALIAS_FIELD_TYPES = {dict: Dict, tuple: Tuple, list: List, set: Set, frozenset: FrozenSet}
E_NON_DEFAULT_FOLLOWS_DEFAULT = '\nNon-default {cls_name} field {field_name} cannot\nfollow default {fields} {default_names}\n'
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

    def __init_subclass__(cls: Union[bool, str, None, typing.Sequence[str]], serializer: Union[None, bool, str, typing.Callable[typing.Any, datetime.datetime.datetime]]=None, namespace: Union[None, bool, str, typing.Callable[typing.Any, datetime.datetime.datetime]]=None, include_metadata: Union[None, bool, str, typing.Callable[typing.Any, datetime.datetime.datetime]]=None, isodates: Union[None, bool, str, typing.Callable[typing.Any, datetime.datetime.datetime]]=None, abstract: bool=False, allow_blessed_key: Union[None, bool, str, typing.Callable[typing.Any, datetime.datetime.datetime]]=None, decimals: Union[None, bool, str, typing.Callable[typing.Any, datetime.datetime.datetime]]=None, coerce: Union[None, bool, str, typing.Callable[typing.Any, datetime.datetime.datetime]]=None, coercions: Union[None, bool, str, typing.Callable[typing.Any, datetime.datetime.datetime]]=None, polymorphic_fields: Union[None, bool, str, typing.Callable[typing.Any, datetime.datetime.datetime]]=None, validation: Union[None, bool, str, typing.Callable[typing.Any, datetime.datetime.datetime]]=None, date_parser: Union[None, bool, str, typing.Callable[typing.Any, datetime.datetime.datetime]]=None, lazy_creation: bool=False, **kwargs) -> None:
        super().__init_subclass__(serializer=serializer, namespace=namespace, include_metadata=include_metadata, isodates=isodates, abstract=abstract, allow_blessed_key=allow_blessed_key, decimals=decimals, coerce=coerce, coercions=coercions, polymorphic_fields=polymorphic_fields, validation=validation, date_parser=date_parser, lazy_creation=lazy_creation, **kwargs)

    @classmethod
    def _contribute_to_options(cls: Union[dict, faustypes.models.ModelOptions, typing.Type], options: Union[dict[str, typing.Any], None, typing.Mapping, dict]) -> None:
        fields, defaults = annotations(cls, stop=Record, skip_classvar=True, alias_types=ALIAS_FIELD_TYPES, localns={cls.__name__: cls})
        options.fields = cast(Mapping, fields)
        options.fieldset = frozenset(fields)
        options.fieldpos = {i: k for i, k in enumerate(fields.keys())}
        options.defaults = {k: v.default if isinstance(v, FieldDescriptor) else v for k, v in defaults.items() if k in fields and (not (isinstance(v, FieldDescriptor) and v.required))}
        local_defaults = []
        for attr_name in cls.__annotations__:
            if attr_name in cls.__dict__:
                default_value = cls.__dict__[attr_name]
                if isinstance(default_value, FieldDescriptorT):
                    if not default_value.required:
                        local_defaults.append(attr_name)
                else:
                    local_defaults.append(attr_name)
            elif local_defaults:
                raise TypeError(E_NON_DEFAULT_FOLLOWS_DEFAULT.format(cls_name=cls.__name__, field_name=attr_name, fields=pluralize(len(local_defaults), 'field'), default_names=', '.join(local_defaults)))
        for field, typ in fields.items():
            if is_optional(typ):
                options.defaults.setdefault(field, None)
        options.optionalset = frozenset(options.defaults)

    @classmethod
    def _contribute_methods(cls: Union[typing.Type, bool]) -> None:
        if not getattr(cls.asdict, 'faust_generated', False):
            raise RuntimeError('Not allowed to override Record.asdict()')
        cls.asdict = cls._BUILD_asdict()
        cls.asdict.faust_generated = True
        cls._input_translate_fields = cls._BUILD_input_translate_fields()

    @classmethod
    def _contribute_field_descriptors(cls: Union[faustypes.models.FieldDescriptorT, faustypes.models.ModelOptions, typing.Type], target: Union[typing.Type, faustypes.models.FieldDescriptorT], options: Union[faustypes.models.FieldDescriptorT, faustypes.models.ModelOptions, typing.Type], parent: Union[None, faustypes.models.FieldDescriptorT, faustypes.models.ModelOptions, typing.Type]=None) -> dict:
        fields = options.fields
        defaults = options.defaults
        date_parser = options.date_parser
        coerce = options.coerce
        index = {}
        secret_fields = set()
        sensitive_fields = set()
        personal_fields = set()
        tagged_fields = set()

        def add_to_tagged_indices(field: Any, tag: Any) -> None:
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

        def add_related_to_tagged_indices(field: Any, related_model: None=None) -> None:
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
                descr = DescriptorType(field=field, type=typ, model=cls, required=needed, default=default, parent=parent, coerce=coerce, model_coercions=options.coercions, date_parser=date_parser, tag=tag)
            else:
                descr = descr.clone(field=field, type=typ, model=cls, required=needed, default=default, parent=parent, coerce=coerce, model_coercions=options.coercions)
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
    def from_data(cls: Union[typing.Type, bool], data: Union[T, ModelT, typing.Type], *, preferred_type: Union[None, typing.Type, ModelT, T]=None) -> Union[dict, collections.abc.AsyncIterator, zerver.models.Stream]:
        """Create model object from Python dictionary."""
        if hasattr(data, '__is_model__'):
            return cast(Record, data)
        else:
            self_cls = cls._maybe_namespace(data, preferred_type=preferred_type)
        cls._input_translate_fields(data)
        return (self_cls or cls)(**data, __strict__=False)

    def __init__(self, *args, __strict__: bool=True, __faust: Union[None, bool, list[str]]=None, **kwargs) -> None:
        ...

    @classmethod
    def _BUILD_input_translate_fields(cls: Union[typing.Type, dict[str, typing.Any], bool]):
        translate = [f'data[{field!r}] = data.pop({d.input_name!r}, None)' for field, d in cls._options.descriptors.items() if d.field != d.input_name]
        return cast(Callable, classmethod(codegen.Function('_input_translate_fields', ['cls', 'data'], translate if translate else ['pass'], globals=globals(), locals=locals())))

    @classmethod
    def _BUILD_init(cls: Union[typing.Type, str]):
        options = cls._options
        field_positions = options.fieldpos
        optional = options.optionalset
        needs_validation = options.validation
        descriptors = options.descriptors
        has_post_init = hasattr(cls, '__post_init__')
        closures = {'__defaults__': 'Model._options.defaults', '__descr__': 'Model._options.descriptors'}
        kwonlyargs = ['*', '__strict__=True', '__faust=None', '**kwargs']
        optional_fields = OrderedDict()
        required_fields = OrderedDict()

        def generate_setter(field: Any, getval: Any) -> typing.Text:
            """Generate code that sets attribute for field in class.

            Arguments:
                field: Name of field.
                getval: Source code that initializes value for field,
                    can be the field name itself for no initialization
                    or for example: ``f"self._prepare_value({field})"``.
                out: Destination list where new source code lines are added.
                """
            if field in optional:
                optional_fields[field] = True
                default_var = f'_default_{field}_'
                closures[default_var] = f'__defaults__["{field}"]'
                return f'    self.{field} = {getval} if {field} is not None else {default_var}'
            else:
                required_fields[field] = True
                return f'    self.{field} = {getval}'

        def generate_prepare_value(field: Any) -> typing.Text:
            descriptor = descriptors[field]
            if descriptor.lazy_coercion:
                return field
            else:
                init_field_var = f'_init_{field}_'
                closures[init_field_var] = f'__descr__["{field}"].to_python'
                return f'{init_field_var}({field})'
        preamble = ['self.__evaluated_fields__ = set()']
        data_setters = ['if __strict__:'] + [generate_setter(field, field) for field in field_positions.values()]
        data_rest = ['    if kwargs:', '        from mode.utils.text import pluralize', '        message = "{} got unexpected {}: {}".format(', '            self.__class__.__name__,', '            pluralize(kwargs.__len__(), "argument"),', '            ", ".join(map(str, sorted(kwargs))))', '        raise TypeError(message)']
        init_setters = ['else:']
        if field_positions:
            init_setters.extend((generate_setter(field, generate_prepare_value(field)) for field in field_positions.values()))
        init_setters.append('    self.__dict__.update(kwargs)')
        postamble = []
        if has_post_init:
            postamble.append('self.__post_init__()')
        if needs_validation:
            postamble.append('self.validate_or_raise()')
        signature = list(chain(['self'], [f'{field}' for field in required_fields], [f'{field}=None' for field in optional_fields], kwonlyargs))
        sourcecode = codegen.build_closure_source(name='__init__', args=signature, body=list(chain(preamble, data_setters, data_rest, init_setters, postamble)), closures=closures, outer_args=['Model'])
        return codegen.build_closure('__outer__', sourcecode, cls, globals={}, locals={})

    @classmethod
    def _BUILD_hash(cls: Union[typing.Type, str]):
        return codegen.HashMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_eq(cls: Union[typing.Type, str]):
        return codegen.EqMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_ne(cls: typing.Type):
        return codegen.NeMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_gt(cls: typing.Type):
        return codegen.GtMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_ge(cls: Union[typing.Type, str]):
        return codegen.GeMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_lt(cls: Union[typing.Type, str]):
        return codegen.LtMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_le(cls: typing.Type):
        return codegen.LeMethod(list(cls._options.fields), globals=globals(), locals=locals())

    @classmethod
    def _BUILD_asdict(cls: Union[str, typing.Type]):
        preamble = ['return self._prepare_dict({']
        fields = [f'  {d.output_name!r}: {cls._BUILD_asdict_field(name, d)},' for name, d in cls._options.descriptors.items() if not d.exclude]
        postamble = ['})']
        return codegen.Method('_asdict', [], preamble + fields + postamble, globals=globals(), locals=locals())

    def _prepare_dict(self, payload: Union[dict[str, typing.Any], bytes, mypy.types.Any]) -> Union[dict[str, typing.Any], bytes, mypy.types.Any]:
        return payload

    @classmethod
    def _BUILD_asdict_field(cls: Union[str, faustypes.models.FieldDescriptorT, typing.Type], name: Union[str, faustypes.models.FieldDescriptorT, typing.Type], field: Union[str, faustypes.models.FieldDescriptorT, typing.Type]) -> typing.Text:
        return f'self.{name}'

    def _derive(self, *objects, **fields) -> Union[dict, utils.SinkType]:
        data = self.asdict()
        for obj in objects:
            data.update(cast(Record, obj).asdict())
        return type(self)(**{**data, **fields})

    def to_representation(self):
        """Convert model to its Python generic counterpart.

        Records will be converted to dictionary.
        """
        payload = self.asdict()
        options = self._options
        if options.include_metadata:
            payload['__faust'] = {'ns': options.namespace}
        return payload

    def asdict(self) -> None:
        """Convert record to Python dictionary."""
        ...
    asdict.faust_generated = True

    def _humanize(self) -> Union[str, list[str], typing.Mapping]:
        attrs, defaults = (self.__dict__, self._options.defaults.items())
        fields = {**{k: v for k, v in attrs.items() if not k.startswith('__')}, **{k: v for k, v in defaults if k not in attrs}}
        return _kvrepr(fields)

    def __json__(self) -> str:
        return self.to_representation()

    def __eq__(self, other: Union[typing.Iterable[T], datetime.timedelta]):
        return NotImplemented

    def __ne__(self, other: Union[typing.Iterable[T], T]):
        return NotImplemented

    def __lt__(self, other: T):
        return NotImplemented

    def __le__(self, other: Union[T, typing.Iterable[typing.Any]]):
        return NotImplemented

    def __gt__(self, other: T):
        return NotImplemented

    def __ge__(self, other: T):
        return NotImplemented

def _kvrepr(d: Any, *, sep: typing.Text=', ') -> str:
    """Represent dict as `k='v'` pairs separated by comma."""
    return sep.join((f'{k}={v!r}' for k, v in d.items()))