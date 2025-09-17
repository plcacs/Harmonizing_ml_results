import copy
import re
from collections import Counter as CollectionCounter, defaultdict, deque
from collections.abc import Callable, Hashable as CollectionsHashable, Iterable as CollectionsIterable
from typing import TYPE_CHECKING, Any, Counter, DefaultDict, Deque, Dict, ForwardRef, FrozenSet, Generator, Iterable, Iterator, List, Mapping, Optional, Pattern, Sequence, Set, Tuple, Type, TypeVar, Union
from typing_extensions import Annotated, Final
from pydantic.v1 import errors as errors_
from pydantic.v1.class_validators import Validator, make_generic_validator, prep_validators
from pydantic.v1.error_wrappers import ErrorWrapper
from pydantic.v1.errors import ConfigError, InvalidDiscriminator, MissingDiscriminator, NoneIsNotAllowedError
from pydantic.v1.types import Json, JsonWrapper
from pydantic.v1.typing import NoArgAnyCallable, convert_generics, display_as_type, get_args, get_origin, is_finalvar, is_literal_type, is_new_type, is_none_type, is_typeddict, is_typeddict_special, is_union, new_type_supertype
from pydantic.v1.utils import PyObjectStr, Representation, ValueItems, get_discriminator_alias_and_values, get_unique_discriminator_alias, lenient_isinstance, lenient_issubclass, sequence_like, smart_deepcopy
from pydantic.v1.validators import constant_validator, dict_validator, find_validators, validate_json

Required: Any = Ellipsis
T = TypeVar('T')

class UndefinedType:
    def __repr__(self) -> str:
        return 'PydanticUndefined'

    def __copy__(self) -> "UndefinedType":
        return self

    def __reduce__(self) -> str:
        return 'Undefined'

    def __deepcopy__(self, _: Any) -> "UndefinedType":
        return self

Undefined = UndefinedType()

if TYPE_CHECKING:
    from pydantic.v1.class_validators import ValidatorsList
    from pydantic.v1.config import BaseConfig
    from pydantic.v1.error_wrappers import ErrorList
    from pydantic.v1.types import ModelOrDc
    from pydantic.v1.typing import AbstractSetIntStr, MappingIntStrAny, ReprArgs
    ValidateReturn = Tuple[Optional[Any], Optional[Any]]
    LocStr = Union[Tuple[Union[int, str], ...], str]
    BoolUndefined = Union[bool, UndefinedType]

class FieldInfo(Representation):
    """
    Captures extra information about a field.
    """
    __slots__ = (
        'default', 'default_factory', 'alias', 'alias_priority', 'title', 'description', 'exclude', 'include', 'const',
        'gt', 'ge', 'lt', 'le', 'multiple_of', 'allow_inf_nan', 'max_digits', 'decimal_places', 'min_items', 'max_items',
        'unique_items', 'min_length', 'max_length', 'allow_mutation', 'repr', 'regex', 'discriminator', 'extra'
    )
    __field_constraints__ = {
        'min_length': None, 'max_length': None, 'regex': None, 'gt': None, 'lt': None, 'ge': None, 'le': None,
        'multiple_of': None, 'allow_inf_nan': None, 'max_digits': None, 'decimal_places': None, 'min_items': None,
        'max_items': None, 'unique_items': None, 'allow_mutation': True
    }

    def __init__(self, default: Any = Undefined, **kwargs: Any) -> None:
        self.default = default
        self.default_factory = kwargs.pop('default_factory', None)
        self.alias = kwargs.pop('alias', None)
        self.alias_priority = kwargs.pop('alias_priority', 2 if self.alias is not None else None)
        self.title = kwargs.pop('title', None)
        self.description = kwargs.pop('description', None)
        self.exclude = kwargs.pop('exclude', None)
        self.include = kwargs.pop('include', None)
        self.const = kwargs.pop('const', None)
        self.gt = kwargs.pop('gt', None)
        self.ge = kwargs.pop('ge', None)
        self.lt = kwargs.pop('lt', None)
        self.le = kwargs.pop('le', None)
        self.multiple_of = kwargs.pop('multiple_of', None)
        self.allow_inf_nan = kwargs.pop('allow_inf_nan', None)
        self.max_digits = kwargs.pop('max_digits', None)
        self.decimal_places = kwargs.pop('decimal_places', None)
        self.min_items = kwargs.pop('min_items', None)
        self.max_items = kwargs.pop('max_items', None)
        self.unique_items = kwargs.pop('unique_items', None)
        self.min_length = kwargs.pop('min_length', None)
        self.max_length = kwargs.pop('max_length', None)
        self.allow_mutation = kwargs.pop('allow_mutation', True)
        self.regex = kwargs.pop('regex', None)
        self.discriminator = kwargs.pop('discriminator', None)
        self.repr = kwargs.pop('repr', True)
        self.extra = kwargs

    def __repr_args__(self) -> List[Tuple[str, Any]]:
        field_defaults_to_hide = {'repr': True, **self.__field_constraints__}
        attrs = ((s, getattr(self, s)) for s in self.__slots__)
        return [(a, v) for a, v in attrs if v != field_defaults_to_hide.get(a, None)]

    def get_constraints(self) -> Set[str]:
        """
        Gets the constraints set on the field by comparing the constraint value with its default value

        :return: the constraints set on field_info
        """
        return {attr for attr, default in self.__field_constraints__.items() if getattr(self, attr) != default}

    def update_from_config(self, from_config: Dict[str, Any]) -> None:
        """
        Update this FieldInfo based on a dict from get_field_info, only fields which have not been set are updated.
        """
        for attr_name, value in from_config.items():
            try:
                current_value = getattr(self, attr_name)
            except AttributeError:
                self.extra.setdefault(attr_name, value)
            else:
                if current_value is self.__field_constraints__.get(attr_name, None):
                    setattr(self, attr_name, value)
                elif attr_name == 'exclude':
                    self.exclude = ValueItems.merge(value, current_value)
                elif attr_name == 'include':
                    self.include = ValueItems.merge(value, current_value, intersect=True)

    def _validate(self) -> None:
        if self.default is not Undefined and self.default_factory is not None:
            raise ValueError('cannot specify both default and default_factory')

def Field(
    default: Any = Undefined,
    *,
    default_factory: Optional[Callable[[], Any]] = None,
    alias: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    exclude: Any = None,
    include: Any = None,
    const: Any = None,
    gt: Any = None,
    ge: Any = None,
    lt: Any = None,
    le: Any = None,
    multiple_of: Any = None,
    allow_inf_nan: Any = None,
    max_digits: Any = None,
    decimal_places: Any = None,
    min_items: Any = None,
    max_items: Any = None,
    unique_items: Any = None,
    min_length: Any = None,
    max_length: Any = None,
    allow_mutation: bool = True,
    regex: Any = None,
    discriminator: Any = None,
    repr: bool = True,
    **extra: Any,
) -> FieldInfo:
    field_info = FieldInfo(
        default,
        default_factory=default_factory,
        alias=alias,
        title=title,
        description=description,
        exclude=exclude,
        include=include,
        const=const,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        allow_inf_nan=allow_inf_nan,
        max_digits=max_digits,
        decimal_places=decimal_places,
        min_items=min_items,
        max_items=max_items,
        unique_items=unique_items,
        min_length=min_length,
        max_length=max_length,
        allow_mutation=allow_mutation,
        regex=regex,
        discriminator=discriminator,
        repr=repr,
        **extra
    )
    field_info._validate()
    return field_info

SHAPE_SINGLETON: int = 1
SHAPE_LIST: int = 2
SHAPE_SET: int = 3
SHAPE_MAPPING: int = 4
SHAPE_TUPLE: int = 5
SHAPE_TUPLE_ELLIPSIS: int = 6
SHAPE_SEQUENCE: int = 7
SHAPE_FROZENSET: int = 8
SHAPE_ITERABLE: int = 9
SHAPE_GENERIC: int = 10
SHAPE_DEQUE: int = 11
SHAPE_DICT: int = 12
SHAPE_DEFAULTDICT: int = 13
SHAPE_COUNTER: int = 14
SHAPE_NAME_LOOKUP: Dict[int, str] = {
    SHAPE_LIST: 'List[{}]',
    SHAPE_SET: 'Set[{}]',
    SHAPE_TUPLE_ELLIPSIS: 'Tuple[{}, ...]',
    SHAPE_SEQUENCE: 'Sequence[{}]',
    SHAPE_FROZENSET: 'FrozenSet[{}]',
    SHAPE_ITERABLE: 'Iterable[{}]',
    SHAPE_DEQUE: 'Deque[{}]',
    SHAPE_DICT: 'Dict[{}]',
    SHAPE_DEFAULTDICT: 'DefaultDict[{}]',
    SHAPE_COUNTER: 'Counter[{}]'
}
MAPPING_LIKE_SHAPES: Set[int] = {SHAPE_DEFAULTDICT, SHAPE_DICT, SHAPE_MAPPING, SHAPE_COUNTER}

class ModelField(Representation):
    __slots__ = (
        'type_', 'outer_type_', 'annotation', 'sub_fields', 'sub_fields_mapping', 'key_field', 'validators',
        'pre_validators', 'post_validators', 'default', 'default_factory', 'required', 'final', 'model_config',
        'name', 'alias', 'has_alias', 'field_info', 'discriminator_key', 'discriminator_alias', 'validate_always',
        'allow_none', 'shape', 'class_validators', 'parse_json'
    )

    def __init__(
        self,
        *,
        name: str,
        type_: Any,
        class_validators: Optional[Dict[str, Validator]] = None,
        model_config: Any,
        default: Any = None,
        default_factory: Optional[Callable[[], Any]] = None,
        required: Any = Undefined,
        final: bool = False,
        alias: Optional[str] = None,
        field_info: Optional[FieldInfo] = None,
    ) -> None:
        self.name: str = name
        self.has_alias: bool = alias is not None
        self.alias: str = alias if alias is not None else name
        self.annotation: Any = type_
        self.type_ = convert_generics(type_)
        self.outer_type_: Any = type_
        self.class_validators: Dict[str, Validator] = class_validators or {}
        self.default: Any = default
        self.default_factory: Optional[Callable[[], Any]] = default_factory
        self.required: Any = required
        self.final: bool = final
        self.model_config: Any = model_config
        self.field_info: FieldInfo = field_info or FieldInfo(default)
        self.discriminator_key: Any = self.field_info.discriminator
        self.discriminator_alias: Any = self.discriminator_key
        self.allow_none: bool = False
        self.validate_always: bool = False
        self.sub_fields: Optional[List["ModelField"]] = None
        self.sub_fields_mapping: Optional[Dict[Any, "ModelField"]] = None
        self.key_field: Optional["ModelField"] = None
        self.validators: List[Callable[..., Any]] = []
        self.pre_validators: Optional[List[Callable[..., Any]]] = None
        self.post_validators: Optional[List[Callable[..., Any]]] = None
        self.parse_json: bool = False
        self.shape: int = SHAPE_SINGLETON
        self.model_config.prepare_field(self)
        self.prepare()

    def get_default(self) -> Any:
        return smart_deepcopy(self.default) if self.default_factory is None else self.default_factory()

    @staticmethod
    def _get_field_info(
        field_name: str, annotation: Any, value: Any, config: Any
    ) -> Tuple[FieldInfo, Any]:
        field_info_from_config: Dict[str, Any] = config.get_field_info(field_name)
        field_info: Optional[FieldInfo] = None
        if get_origin(annotation) is Annotated:
            field_infos: List[FieldInfo] = [arg for arg in get_args(annotation)[1:] if isinstance(arg, FieldInfo)]
            if len(field_infos) > 1:
                raise ValueError(f'cannot specify multiple `Annotated` `Field`s for {field_name!r}')
            field_info = next(iter(field_infos), None)
            if field_info is not None:
                field_info = copy.copy(field_info)
                field_info.update_from_config(field_info_from_config)
                if field_info.default not in (Undefined, Required):
                    raise ValueError(f'`Field` default cannot be set in `Annotated` for {field_name!r}')
                if value is not Undefined and value is not Required:
                    field_info.default = value
        if isinstance(value, FieldInfo):
            if field_info is not None:
                raise ValueError(f'cannot specify `Annotated` and value `Field`s together for {field_name!r}')
            field_info = value
            field_info.update_from_config(field_info_from_config)
        elif field_info is None:
            field_info = FieldInfo(value, **field_info_from_config)
        value = None if field_info.default_factory is not None else field_info.default
        field_info._validate()
        return (field_info, value)

    @classmethod
    def infer(
        cls,
        *,
        name: str,
        value: Any,
        annotation: Any,
        class_validators: Optional[Dict[str, Validator]],
        config: Any
    ) -> "ModelField":
        from pydantic.v1.schema import get_annotation_from_field_info  # type: ignore
        field_info, value = cls._get_field_info(name, annotation, value, config)
        required: Any = Undefined
        if value is Required:
            required = True
            value = None
        elif value is not Undefined:
            required = False
        annotation = get_annotation_from_field_info(annotation, field_info, name, config.validate_assignment)
        return cls(
            name=name,
            type_=annotation,
            alias=field_info.alias,
            class_validators=class_validators,
            default=value,
            default_factory=field_info.default_factory,
            required=required,
            model_config=config,
            field_info=field_info,
        )

    def set_config(self, config: Any) -> None:
        self.model_config = config
        info_from_config: Dict[str, Any] = config.get_field_info(self.name)
        config.prepare_field(self)
        new_alias: Optional[str] = info_from_config.get('alias')
        new_alias_priority: int = info_from_config.get('alias_priority') or 0
        if new_alias and new_alias_priority >= (self.field_info.alias_priority or 0):
            self.field_info.alias = new_alias
            self.field_info.alias_priority = new_alias_priority
            self.alias = new_alias
        new_exclude: Any = info_from_config.get('exclude')
        if new_exclude is not None:
            self.field_info.exclude = ValueItems.merge(self.field_info.exclude, new_exclude)
        new_include: Any = info_from_config.get('include')
        if new_include is not None:
            self.field_info.include = ValueItems.merge(self.field_info.include, new_include, intersect=True)

    @property
    def alt_alias(self) -> bool:
        return self.name != self.alias

    def prepare(self) -> None:
        """
        Prepare the field by inspecting self.default, self.type_ etc.
        Note: this method is **not** idempotent.
        """
        self._set_default_and_type()
        if self.type_.__class__ is ForwardRef or self.type_.__class__ is DeferredType:
            return
        self._type_analysis()
        if self.required is Undefined:
            self.required = True
        if self.default is Undefined and self.default_factory is None:
            self.default = None
        self.populate_validators()

    def _set_default_and_type(self) -> None:
        """
        Set the default value, infer the type if needed and check if `None` value is valid.
        """
        if self.default_factory is not None:
            if self.type_ is Undefined:
                raise errors_.ConfigError(f'you need to set the type of field {self.name!r} when using `default_factory`')
            return
        default_value: Any = self.get_default()
        if default_value is not None and self.type_ is Undefined:
            self.type_ = default_value.__class__
            self.outer_type_ = self.type_
            self.annotation = self.type_
        if self.type_ is Undefined:
            raise errors_.ConfigError(f'unable to infer type for attribute "{self.name}"')
        if self.required is False and default_value is None:
            self.allow_none = True

    def _type_analysis(self) -> None:
        if lenient_issubclass(self.type_, JsonWrapper):
            self.type_ = self.type_.inner_type
            self.parse_json = True
        elif lenient_issubclass(self.type_, Json):
            self.type_ = Any
            self.parse_json = True
        elif isinstance(self.type_, TypeVar):
            if self.type_.__bound__:
                self.type_ = self.type_.__bound__
            elif self.type_.__constraints__:
                self.type_ = Union[self.type_.__constraints__]
            else:
                self.type_ = Any
        elif is_new_type(self.type_):
            self.type_ = new_type_supertype(self.type_)
        if self.type_ is Any or self.type_ is object:
            if self.required is Undefined:
                self.required = False
            self.allow_none = True
            return
        elif self.type_ is Pattern or self.type_ is re.Pattern:
            return
        elif is_literal_type(self.type_):
            return
        elif is_typeddict(self.type_):
            return
        if is_finalvar(self.type_):
            self.final = True
            if self.type_ is Final:
                self.type_ = Any
            else:
                self.type_ = get_args(self.type_)[0]
            self._type_analysis()
            return
        origin = get_origin(self.type_)
        if origin is Annotated or is_typeddict_special(origin):
            self.type_ = get_args(self.type_)[0]
            self._type_analysis()
            return
        if self.discriminator_key is not None and (not is_union(origin)):
            raise TypeError('`discriminator` can only be used with `Union` type with more than one variant')
        if origin is None or origin is CollectionsHashable:
            if isinstance(self.type_, type) and isinstance(None, self.type_):
                self.allow_none = True
            return
        elif origin is Callable:
            return
        elif is_union(origin):
            types_ = []
            for type_ in get_args(self.type_):
                if is_none_type(type_) or type_ is Any or type_ is object:
                    if self.required is Undefined:
                        self.required = False
                    self.allow_none = True
                if is_none_type(type_):
                    continue
                types_.append(type_)
            if len(types_) == 1:
                self.type_ = types_[0]
                self.outer_type_ = self.type_
                self._type_analysis()
            else:
                self.sub_fields = [self._create_sub_type(t, f'{self.name}_{display_as_type(t)}') for t in types_]
                if self.discriminator_key is not None:
                    self.prepare_discriminated_union_sub_fields()
            return
        elif issubclass(origin, Tuple):
            args = get_args(self.type_)
            if not args:
                self.type_ = Any
                self.shape = SHAPE_TUPLE_ELLIPSIS
            elif len(args) == 2 and args[1] is Ellipsis:
                self.type_ = args[0]
                self.shape = SHAPE_TUPLE_ELLIPSIS
                self.sub_fields = [self._create_sub_type(args[0], f'{self.name}_0')]
            elif args == ((),):
                self.shape = SHAPE_TUPLE
                self.type_ = Any
                self.sub_fields = []
            else:
                self.shape = SHAPE_TUPLE
                self.sub_fields = [self._create_sub_type(t, f'{self.name}_{i}') for i, t in enumerate(args)]
            return
        elif issubclass(origin, List):
            get_validators = getattr(self.type_, '__get_validators__', None)
            if get_validators:
                self.class_validators.update({
                    f'list_{i}': Validator(validator, pre=True)
                    for i, validator in enumerate(get_validators())
                })
            self.type_ = get_args(self.type_)[0]
            self.shape = SHAPE_LIST
        elif issubclass(origin, Set):
            get_validators = getattr(self.type_, '__get_validators__', None)
            if get_validators:
                self.class_validators.update({
                    f'set_{i}': Validator(validator, pre=True)
                    for i, validator in enumerate(get_validators())
                })
            self.type_ = get_args(self.type_)[0]
            self.shape = SHAPE_SET
        elif issubclass(origin, FrozenSet):
            get_validators = getattr(self.type_, '__get_validators__', None)
            if get_validators:
                self.class_validators.update({
                    f'frozenset_{i}': Validator(validator, pre=True)
                    for i, validator in enumerate(get_validators())
                })
            self.type_ = get_args(self.type_)[0]
            self.shape = SHAPE_FROZENSET
        elif issubclass(origin, Deque):
            self.type_ = get_args(self.type_)[0]
            self.shape = SHAPE_DEQUE
        elif issubclass(origin, Sequence):
            self.type_ = get_args(self.type_)[0]
            self.shape = SHAPE_SEQUENCE
        elif origin is dict or origin is Dict:
            self.key_field = self._create_sub_type(get_args(self.type_)[0], 'key_' + self.name, for_keys=True)
            self.type_ = get_args(self.type_)[1]
            self.shape = SHAPE_DICT
        elif issubclass(origin, DefaultDict):
            self.key_field = self._create_sub_type(get_args(self.type_)[0], 'key_' + self.name, for_keys=True)
            self.type_ = get_args(self.type_)[1]
            self.shape = SHAPE_DEFAULTDICT
        elif issubclass(origin, Counter):
            self.key_field = self._create_sub_type(get_args(self.type_)[0], 'key_' + self.name, for_keys=True)
            self.type_ = int
            self.shape = SHAPE_COUNTER
        elif issubclass(origin, Mapping):
            self.key_field = self._create_sub_type(get_args(self.type_)[0], 'key_' + self.name, for_keys=True)
            self.type_ = get_args(self.type_)[1]
            self.shape = SHAPE_MAPPING
        elif origin in {Iterable, CollectionsIterable}:
            self.type_ = get_args(self.type_)[0]
            self.shape = SHAPE_ITERABLE
            self.sub_fields = [self._create_sub_type(self.type_, f'{self.name}_type')]
        elif issubclass(origin, Type):
            return
        elif hasattr(origin, '__get_validators__') or self.model_config.arbitrary_types_allowed:
            self.shape = SHAPE_GENERIC
            self.sub_fields = [self._create_sub_type(t, f'{self.name}_{i}') for i, t in enumerate(get_args(self.type_))]
            self.type_ = origin
            return
        else:
            raise TypeError(f'Fields of type "{origin}" are not supported.')
        self.sub_fields = [self._create_sub_type(self.type_, '_' + self.name)]

    def prepare_discriminated_union_sub_fields(self) -> None:
        """
        Prepare the mapping <discriminator key> -> <ModelField> and update `sub_fields`
        """
        assert self.discriminator_key is not None
        if self.type_.__class__ is DeferredType:
            return
        assert self.sub_fields is not None
        sub_fields_mapping: Dict[Any, ModelField] = {}
        all_aliases: Set[Any] = set()
        for sub_field in self.sub_fields:
            t = sub_field.type_
            if t.__class__ is ForwardRef:
                return
            alias, discriminator_values = get_discriminator_alias_and_values(t, self.discriminator_key)
            all_aliases.add(alias)
            for discriminator_value in discriminator_values:
                sub_fields_mapping[discriminator_value] = sub_field
        self.sub_fields_mapping = sub_fields_mapping
        self.discriminator_alias = get_unique_discriminator_alias(all_aliases, self.discriminator_key)

    def _create_sub_type(self, type_: Any, name: str, *, for_keys: bool = False) -> "ModelField":
        if for_keys:
            class_validators: Optional[Dict[str, Validator]] = None
        else:
            class_validators = {
                k: Validator(
                    func=v.func,
                    pre=v.pre,
                    each_item=False,
                    always=v.always,
                    check_fields=v.check_fields,
                    skip_on_failure=v.skip_on_failure
                )
                for k, v in self.class_validators.items() if v.each_item
            }
        field_info, _ = self._get_field_info(name, type_, None, self.model_config)
        return self.__class__(
            type_=type_,
            name=name,
            class_validators=class_validators,
            model_config=self.model_config,
            field_info=field_info
        )

    def populate_validators(self) -> None:
        """
        Prepare self.pre_validators, self.validators, and self.post_validators.
        """
        self.validate_always = getattr(self.type_, 'validate_always', False) or any((v.always for v in self.class_validators.values()))
        let_class_validators = self.class_validators.values()
        if not self.sub_fields or self.shape == SHAPE_GENERIC:
            get_validators = getattr(self.type_, '__get_validators__', None)
            v_funcs = (
                *[v.func for v in let_class_validators if v.each_item and v.pre],
                *(get_validators() if get_validators else list(find_validators(self.type_, self.model_config))),
                *[v.func for v in let_class_validators if v.each_item and (not v.pre)]
            )
            self.validators = prep_validators(v_funcs)
        self.pre_validators = []
        self.post_validators = []
        if self.field_info and self.field_info.const:
            self.post_validators.append(make_generic_validator(constant_validator))
        if let_class_validators:
            self.pre_validators += prep_validators((v.func for v in let_class_validators if not v.each_item and v.pre))
            self.post_validators += prep_validators((v.func for v in let_class_validators if not v.each_item and (not v.pre)))
        if self.parse_json:
            self.pre_validators.append(make_generic_validator(validate_json))
        self.pre_validators = self.pre_validators or None
        self.post_validators = self.post_validators or None

    def validate(
        self,
        v: Any,
        values: Dict[str, Any],
        *,
        loc: Union[Tuple[Any, ...], str],
        cls: Optional[Type[Any]] = None
    ) -> Tuple[Any, Optional[Union[ErrorWrapper, List[ErrorWrapper]]]]:
        assert self.type_.__class__ is not DeferredType
        if self.type_.__class__ is ForwardRef:
            assert cls is not None
            raise ConfigError(f'field "{self.name}" not yet prepared so type is still a ForwardRef, you might need to call {cls.__name__}.update_forward_refs().')
        if self.pre_validators:
            v, errors_found = self._apply_validators(v, values, loc, cls, self.pre_validators)
            if errors_found:
                return (v, errors_found)
        if v is None:
            if is_none_type(self.type_):
                pass
            elif self.allow_none:
                if self.post_validators:
                    return self._apply_validators(v, values, loc, cls, self.post_validators)
                else:
                    return (None, None)
            else:
                return (v, ErrorWrapper(NoneIsNotAllowedError(), loc))
        if self.shape == SHAPE_SINGLETON:
            v, errors_found = self._validate_singleton(v, values, loc, cls)
        elif self.shape in MAPPING_LIKE_SHAPES:
            v, errors_found = self._validate_mapping_like(v, values, loc, cls)
        elif self.shape == SHAPE_TUPLE:
            v, errors_found = self._validate_tuple(v, values, loc, cls)
        elif self.shape == SHAPE_ITERABLE:
            v, errors_found = self._validate_iterable(v, values, loc, cls)
        elif self.shape == SHAPE_GENERIC:
            v, errors_found = self._apply_validators(v, values, loc, cls, self.validators)
        else:
            v, errors_found = self._validate_sequence_like(v, values, loc, cls)
        if not errors_found and self.post_validators:
            v, errors_found = self._apply_validators(v, values, loc, cls, self.post_validators)
        return (v, errors_found)

    def _validate_sequence_like(
        self, v: Any, values: Dict[str, Any], loc: Union[Tuple[Any, ...], str], cls: Optional[Type[Any]]
    ) -> Tuple[Any, Optional[Union[ErrorWrapper, List[ErrorWrapper]]]]:
        if not sequence_like(v):
            if self.shape == SHAPE_LIST:
                e = errors_.ListError()
            elif self.shape in (SHAPE_TUPLE, SHAPE_TUPLE_ELLIPSIS):
                e = errors_.TupleError()
            elif self.shape == SHAPE_SET:
                e = errors_.SetError()
            elif self.shape == SHAPE_FROZENSET:
                e = errors_.FrozenSetError()
            else:
                e = errors_.SequenceError()
            return (v, ErrorWrapper(e, loc))
        loc_tuple: Tuple[Any, ...] = loc if isinstance(loc, tuple) else (loc,)
        result: List[Any] = []
        errors_list: List[ErrorWrapper] = []
        for i, v_ in enumerate(v):
            v_loc = (*loc_tuple, i)
            r, ee = self._validate_singleton(v_, values, v_loc, cls)
            if ee:
                errors_list.append(ee)
            else:
                result.append(r)
        if errors_list:
            return (v, errors_list)
        converted: Any = result
        if self.shape == SHAPE_SET:
            converted = set(result)
        elif self.shape == SHAPE_FROZENSET:
            converted = frozenset(result)
        elif self.shape == SHAPE_TUPLE_ELLIPSIS:
            converted = tuple(result)
        elif self.shape == SHAPE_DEQUE:
            converted = deque(result, maxlen=getattr(v, 'maxlen', None))
        elif self.shape == SHAPE_SEQUENCE:
            if isinstance(v, tuple):
                converted = tuple(result)
            elif isinstance(v, set):
                converted = set(result)
            elif isinstance(v, Generator):
                converted = iter(result)
            elif isinstance(v, deque):
                converted = deque(result, maxlen=getattr(v, 'maxlen', None))
        return (converted, None)

    def _validate_iterable(
        self, v: Any, values: Dict[str, Any], loc: Union[Tuple[Any, ...], str], cls: Optional[Type[Any]]
    ) -> Tuple[Iterable[Any], None]:
        try:
            iterable = iter(v)
        except TypeError:
            return (v, ErrorWrapper(errors_.IterableError(), loc))
        return (iterable, None)

    def _validate_tuple(
        self, v: Any, values: Dict[str, Any], loc: Union[Tuple[Any, ...], str], cls: Optional[Type[Any]]
    ) -> Tuple[Any, Optional[Union[ErrorWrapper, List[ErrorWrapper]]]]:
        e: Optional[Any] = None
        if not sequence_like(v):
            e = errors_.TupleError()
        else:
            actual_length, expected_length = (len(v), len(self.sub_fields) if self.sub_fields is not None else 0)
            if actual_length != expected_length:
                e = errors_.TupleLengthError(actual_length=actual_length, expected_length=expected_length)
        if e:
            return (v, ErrorWrapper(e, loc))
        loc_tuple: Tuple[Any, ...] = loc if isinstance(loc, tuple) else (loc,)
        result: List[Any] = []
        errors_list: List[ErrorWrapper] = []
        if self.sub_fields is None:
            return (v, None)
        for i, (v_, field) in enumerate(zip(v, self.sub_fields)):
            v_loc = (*loc_tuple, i)
            r, ee = field.validate(v_, values, loc=v_loc, cls=cls)
            if ee:
                errors_list.append(ee)
            else:
                result.append(r)
        if errors_list:
            return (v, errors_list)
        else:
            return (tuple(result), None)

    def _validate_mapping_like(
        self, v: Any, values: Dict[str, Any], loc: Union[Tuple[Any, ...], str], cls: Optional[Type[Any]]
    ) -> Tuple[Any, Optional[Union[ErrorWrapper, List[ErrorWrapper]]]]:
        try:
            v_iter = dict_validator(v)
        except TypeError as exc:
            return (v, ErrorWrapper(exc, loc))
        loc_tuple: Tuple[Any, ...] = loc if isinstance(loc, tuple) else (loc,)
        result: Dict[Any, Any] = {}
        errors_list: List[ErrorWrapper] = []
        for k, v_ in v_iter.items():
            v_loc = (*loc_tuple, '__key__')
            key_result, key_errors = self.key_field.validate(k, values, loc=v_loc, cls=cls)  # type: ignore
            if key_errors:
                errors_list.append(key_errors)
                continue
            v_loc = (*loc_tuple, k)
            value_result, value_errors = self._validate_singleton(v_, values, v_loc, cls)
            if value_errors:
                errors_list.append(value_errors)
                continue
            result[key_result] = value_result
        if errors_list:
            return (v, errors_list)
        elif self.shape == SHAPE_DICT:
            return (result, None)
        elif self.shape == SHAPE_DEFAULTDICT:
            return (defaultdict(self.type_, result), None)
        elif self.shape == SHAPE_COUNTER:
            return (CollectionCounter(result), None)
        else:
            return (self._get_mapping_value(v, result), None)

    def _get_mapping_value(self, original: Any, converted: Dict[Any, Any]) -> Any:
        original_cls = original.__class__
        if original_cls == dict or original_cls == Dict:
            return converted
        elif original_cls in {defaultdict, DefaultDict}:
            return defaultdict(self.type_, converted)
        else:
            try:
                return original_cls(converted)
            except TypeError:
                raise RuntimeError(f'Could not convert dictionary to {original_cls.__name__!r}') from None

    def _validate_singleton(
        self, v: Any, values: Dict[str, Any], loc: Union[Tuple[Any, ...], str], cls: Optional[Type[Any]]
    ) -> Tuple[Any, Optional[Union[ErrorWrapper, List[ErrorWrapper]]]]:
        if self.sub_fields:
            if self.discriminator_key is not None:
                return self._validate_discriminated_union(v, values, loc, cls)
            errors_list: List[ErrorWrapper] = []
            if self.model_config.smart_union and is_union(get_origin(self.type_)):
                for field in self.sub_fields:
                    if v.__class__ is field.outer_type_:
                        return (v, None)
                for field in self.sub_fields:
                    try:
                        if isinstance(v, field.outer_type_):
                            return (v, None)
                    except TypeError:
                        if lenient_isinstance(v, get_origin(field.outer_type_)):
                            value, error = field.validate(v, values, loc=loc, cls=cls)
                            if not error:
                                return (value, None)
            for field in self.sub_fields:
                value, error = field.validate(v, values, loc=loc, cls=cls)
                if error:
                    errors_list.append(error)
                else:
                    return (value, None)
            return (v, errors_list)
        else:
            return self._apply_validators(v, values, loc, cls, self.validators)

    def _validate_discriminated_union(
        self, v: Any, values: Dict[str, Any], loc: Union[Tuple[Any, ...], str], cls: Optional[Type[Any]]
    ) -> Tuple[Any, Optional[Union[ErrorWrapper, List[ErrorWrapper]]]]:
        assert self.discriminator_key is not None
        assert self.discriminator_alias is not None
        try:
            try:
                discriminator_value = v[self.discriminator_alias]
            except KeyError:
                if self.model_config.allow_population_by_field_name:
                    discriminator_value = v[self.discriminator_key]
                else:
                    raise
        except KeyError:
            return (v, ErrorWrapper(MissingDiscriminator(discriminator_key=self.discriminator_key), loc))
        except TypeError:
            try:
                discriminator_value = getattr(v, self.discriminator_key)
            except (AttributeError, TypeError):
                return (v, ErrorWrapper(MissingDiscriminator(discriminator_key=self.discriminator_key), loc))
        if self.sub_fields_mapping is None:
            assert cls is not None
            raise ConfigError(f'field "{self.name}" not yet prepared so type is still a ForwardRef, you might need to call {cls.__name__}.update_forward_refs().')
        try:
            sub_field = self.sub_fields_mapping[discriminator_value]
        except (KeyError, TypeError):
            assert self.sub_fields_mapping is not None
            return (v, ErrorWrapper(InvalidDiscriminator(discriminator_key=self.discriminator_key, discriminator_value=discriminator_value, allowed_values=list(self.sub_fields_mapping)), loc))
        else:
            loc_tuple: Tuple[Any, ...] = loc if isinstance(loc, tuple) else (loc,)
            return sub_field.validate(v, values, loc=(*loc_tuple, display_as_type(sub_field.type_)), cls=cls)

    def _apply_validators(
        self,
        v: Any,
        values: Dict[str, Any],
        loc: Union[Tuple[Any, ...], str],
        cls: Optional[Type[Any]],
        validators: List[Callable[..., Any]]
    ) -> Tuple[Any, Optional[Union[ErrorWrapper, List[ErrorWrapper]]]]:
        for validator in validators:
            try:
                v = validator(cls, v, values, self, self.model_config)
            except (ValueError, TypeError, AssertionError) as exc:
                return (v, ErrorWrapper(exc, loc))
        return (v, None)

    def is_complex(self) -> bool:
        from pydantic.v1.main import BaseModel  # type: ignore
        return self.shape != SHAPE_SINGLETON or hasattr(self.type_, '__pydantic_model__') or lenient_issubclass(self.type_, (BaseModel, list, set, frozenset, dict))

    def _type_display(self) -> str:
        t: str = display_as_type(self.type_)
        if self.shape in MAPPING_LIKE_SHAPES:
            t = f'Mapping[{display_as_type(self.key_field.type_)} , {t}]'  # type: ignore
        elif self.shape == SHAPE_TUPLE:
            t = 'Tuple[{}]'.format(', '.join((display_as_type(f.type_) for f in self.sub_fields)))  # type: ignore
        elif self.shape == SHAPE_GENERIC:
            assert self.sub_fields is not None
            t = '{}[{}]'.format(display_as_type(self.type_), ', '.join((display_as_type(f.type_) for f in self.sub_fields)))
        elif self.shape != SHAPE_SINGLETON:
            t = SHAPE_NAME_LOOKUP[self.shape].format(t)
        if self.allow_none and (self.shape != SHAPE_SINGLETON or not self.sub_fields):
            t = f'Optional[{t}]'
        return PyObjectStr(t)

    def __repr_args__(self) -> List[Tuple[str, Any]]:
        args: List[Tuple[str, Any]] = [('name', self.name), ('type', self._type_display()), ('required', self.required)]
        if not self.required:
            if self.default_factory is not None:
                args.append(('default_factory', f'<function {self.default_factory.__name__}>'))
            else:
                args.append(('default', self.default))
        if self.alt_alias:
            args.append(('alias', self.alias))
        return args

class ModelPrivateAttr(Representation):
    __slots__ = ('default', 'default_factory')

    def __init__(self, default: Any = Undefined, *, default_factory: Optional[Callable[[], Any]] = None) -> None:
        self.default = default
        self.default_factory = default_factory

    def get_default(self) -> Any:
        return smart_deepcopy(self.default) if self.default_factory is None else self.default_factory()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and (self.default, self.default_factory) == (other.default, other.default_factory)

def PrivateAttr(default: Any = Undefined, *, default_factory: Optional[Callable[[], Any]] = None) -> ModelPrivateAttr:
    if default is not Undefined and default_factory is not None:
        raise ValueError('cannot specify both default and default_factory')
    return ModelPrivateAttr(default, default_factory=default_factory)

class DeferredType:
    """
    Used to postpone field preparation, while creating recursive generic models.
    """
    pass

def is_finalvar_with_default_val(type_: Any, val: Any) -> bool:
    return is_finalvar(type_) and val is not Undefined and (not isinstance(val, FieldInfo))