import copy
import re
from collections import Counter as CollectionCounter, defaultdict, deque
from collections.abc import Callable, Hashable as CollectionsHashable, Iterable as CollectionsIterable
from typing import (
    TYPE_CHECKING,
    Any,
    Counter,
    DefaultDict,
    Deque,
    Dict,
    ForwardRef,
    FrozenSet,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from typing_extensions import Annotated, Final

from pydantic.v1 import errors as errors_
from pydantic.v1.class_validators import Validator, make_generic_validator, prep_validators
from pydantic.v1.error_wrappers import ErrorWrapper
from pydantic.v1.errors import ConfigError, InvalidDiscriminator, MissingDiscriminator, NoneIsNotAllowedError
from pydantic.v1.types import Json, JsonWrapper
from pydantic.v1.typing import (
    NoArgAnyCallable,
    convert_generics,
    display_as_type,
    get_args,
    get_origin,
    is_finalvar,
    is_literal_type,
    is_new_type,
    is_none_type,
    is_typeddict,
    is_typeddict_special,
    is_union,
    new_type_supertype,
)
from pydantic.v1.utils import (
    PyObjectStr,
    Representation,
    ValueItems,
    get_discriminator_alias_and_values,
    get_unique_discriminator_alias,
    lenient_isinstance,
    lenient_issubclass,
    sequence_like,
    smart_deepcopy,
)
from pydantic.v1.validators import constant_validator, dict_validator, find_validators, validate_json

Required: Any = Ellipsis

T = TypeVar('T')

if TYPE_CHECKING:
    from pydantic.v1.class_validators import ValidatorsList
    from pydantic.v1.config import BaseConfig
    from pydantic.v1.error_wrappers import ErrorList
    from pydantic.v1.types import ModelOrDc
    from pydantic.v1.typing import AbstractSetIntStr, MappingIntStrAny, ReprArgs

    ValidateReturn = Tuple[Optional[Any], Optional[ErrorList]]
    LocStr = Union[Tuple[Union[int, str], ...], str]
    BoolUndefined = Union[bool, 'UndefinedType']


class UndefinedType:
    def __repr__(self) -> str:
        return 'PydanticUndefined'

    def __copy__(self: T) -> T:
        return self

    def __reduce__(self) -> str:
        return 'Undefined'

    def __deepcopy__(self: T, _: Any) -> T:
        return self


Undefined: UndefinedType = UndefinedType()


class FieldInfo(Representation):
    __slots__ = (
        'default',
        'default_factory',
        'alias',
        'alias_priority',
        'title',
        'description',
        'exclude',
        'include',
        'const',
        'gt',
        'ge',
        'lt',
        'le',
        'multiple_of',
        'allow_inf_nan',
        'max_digits',
        'decimal_places',
        'min_items',
        'max_items',
        'unique_items',
        'min_length',
        'max_length',
        'allow_mutation',
        'repr',
        'regex',
        'discriminator',
        'extra',
    )

    __field_constraints__: Dict[str, Any] = {
        'min_length': None,
        'max_length': None,
        'regex': None,
        'gt': None,
        'lt': None,
        'ge': None,
        'le': None,
        'multiple_of': None,
        'allow_inf_nan': None,
        'max_digits': None,
        'decimal_places': None,
        'min_items': None,
        'max_items': None,
        'unique_items': None,
        'allow_mutation': True,
    }

    def __init__(self, default: Any = Undefined, **kwargs: Any) -> None:
        self.default: Any = default
        self.default_factory: Optional[NoArgAnyCallable] = kwargs.pop('default_factory', None)
        self.alias: Optional[str] = kwargs.pop('alias', None)
        self.alias_priority: Optional[int] = kwargs.pop('alias_priority', 2 if self.alias is not None else None)
        self.title: Optional[str] = kwargs.pop('title', None)
        self.description: Optional[str] = kwargs.pop('description', None)
        self.exclude: Any = kwargs.pop('exclude', None)
        self.include: Any = kwargs.pop('include', None)
        self.const: Optional[bool] = kwargs.pop('const', None)
        self.gt: Optional[float] = kwargs.pop('gt', None)
        self.ge: Optional[float] = kwargs.pop('ge', None)
        self.lt: Optional[float] = kwargs.pop('lt', None)
        self.le: Optional[float] = kwargs.pop('le', None)
        self.multiple_of: Optional[float] = kwargs.pop('multiple_of', None)
        self.allow_inf_nan: Optional[bool] = kwargs.pop('allow_inf_nan', None)
        self.max_digits: Optional[int] = kwargs.pop('max_digits', None)
        self.decimal_places: Optional[int] = kwargs.pop('decimal_places', None)
        self.min_items: Optional[int] = kwargs.pop('min_items', None)
        self.max_items: Optional[int] = kwargs.pop('max_items', None)
        self.unique_items: Optional[bool] = kwargs.pop('unique_items', None)
        self.min_length: Optional[int] = kwargs.pop('min_length', None)
        self.max_length: Optional[int] = kwargs.pop('max_length', None)
        self.allow_mutation: bool = kwargs.pop('allow_mutation', True)
        self.regex: Optional[str] = kwargs.pop('regex', None)
        self.discriminator: Optional[str] = kwargs.pop('discriminator', None)
        self.repr: bool = kwargs.pop('repr', True)
        self.extra: Dict[str, Any] = kwargs

    def __repr_args__(self) -> 'ReprArgs':
        field_defaults_to_hide: Dict[str, Any] = {
            'repr': True,
            **self.__field_constraints__,
        }
        attrs = ((s, getattr(self, s)) for s in self.__slots__)
        return [(a, v) for a, v in attrs if v != field_defaults_to_hide.get(a, None)]

    def get_constraints(self) -> Set[str]:
        return {attr for attr, default in self.__field_constraints__.items() if getattr(self, attr) != default}

    def update_from_config(self, from_config: Dict[str, Any]) -> None:
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
    default_factory: Optional[NoArgAnyCallable] = None,
    alias: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    exclude: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny', Any]] = None,
    include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny', Any]] = None,
    const: Optional[bool] = None,
    gt: Optional[float] = None,
    ge: Optional[float] = None,
    lt: Optional[float] = None,
    le: Optional[float] = None,
    multiple_of: Optional[float] = None,
    allow_inf_nan: Optional[bool] = None,
    max_digits: Optional[int] = None,
    decimal_places: Optional[int] = None,
    min_items: Optional[int] = None,
    max_items: Optional[int] = None,
    unique_items: Optional[bool] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_mutation: bool = True,
    regex: Optional[str] = None,
    discriminator: Optional[str] = None,
    repr: bool = True,
    **extra: Any,
) -> Any:
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
        **extra,
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
    SHAPE_COUNTER: 'Counter[{}]',
}

MAPPING_LIKE_SHAPES: Set[int] = {SHAPE_DEFAULTDICT, SHAPE_DICT, SHAPE_MAPPING, SHAPE_COUNTER}


class ModelField(Representation):
    __slots__ = (
        'type_',
        'outer_type_',
        'annotation',
        'sub_fields',
        'sub_fields_mapping',
        'key_field',
        'validators',
        'pre_validators',
        'post_validators',
        'default',
        'default_factory',
        'required',
        'final',
        'model_config',
        'name',
        'alias',
        'has_alias',
        'field_info',
        'discriminator_key',
        'discriminator_alias',
        'validate_always',
        'allow_none',
        'shape',
        'class_validators',
        'parse_json',
    )

    def __init__(
        self,
        *,
        name: str,
        type_: Type[Any],
        class_validators: Optional[Dict[str, Validator]],
        model_config: Type['BaseConfig'],
        default: Any = None,
        default_factory: Optional[NoArgAnyCallable] = None,
        required: 'BoolUndefined' = Undefined,
        final: bool = False,
        alias: Optional[str] = None,
        field_info: Optional[FieldInfo] = None,
    ) -> None:
        self.name: str = name
        self.has_alias: bool = alias is not None
        self.alias: str = alias if alias is not None else name
        self.annotation: Any = type_
        self.type_: Any = convert_generics(type_)
        self.outer_type_: Any = type_
        self.class_validators: Dict[str, Validator] = class_validators or {}
        self.default: Any = default
        self.default_factory: Optional[NoArgAnyCallable] = default_factory
        self.required: 'BoolUndefined' = required
        self.final: bool = final
        self.model_config: Type['BaseConfig'] = model_config
        self.field_info: FieldInfo = field_info or FieldInfo(default)
        self.discriminator_key: Optional[str] = self.field_info.discriminator
        self.discriminator_alias: Optional[str] = self.discriminator_key

        self.allow_none: bool = False
        self.validate_always: bool = False
        self.sub_fields: Optional[List[ModelField]] = None
        self.sub_fields_mapping: Optional[Dict[str, 'ModelField']] = None
        self.key_field: Optional[ModelField] = None
        self.validators: 'ValidatorsList' = []
        self.pre_validators: Optional['ValidatorsList'] = None
        self.post_validators: Optional['ValidatorsList'] = None
        self.parse_json: bool = False
        self.shape: int = SHAPE_SINGLETON
        self.model_config.prepare_field(self)
        self.prepare()

    def get_default(self) -> Any:
        return smart_deepcopy(self.default) if self.default_factory is None else self.default_factory()

    @staticmethod
    def _get_field_info(
        field_name: str, annotation: Any, value: Any, config: Type['BaseConfig']
    ) -> Tuple[FieldInfo, Any]:
        field_info_from_config = config.get_field_info(field_name)
        field_info = None

        if get_origin(annotation) is Annotated:
            field_infos = [arg for arg in get_args(annotation)[1:] if isinstance(arg, FieldInfo)]
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
        return field_info, value

    @classmethod
    def infer(
        cls,
        *,
        name: str,
        value: Any,
        annotation: Any,
        class_validators: Optional[Dict[str, Validator]],
        config: Type['BaseConfig'],
    ) -> 'ModelField':
        from pydantic.v1.schema import get_annotation_from_field_info

        field_info, value = cls._get_field_info(name, annotation, value, config)
        required: 'BoolUndefined' = Undefined
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

    def set_config(self, config: Type['BaseConfig']) -> None:
        self.model_config = config
        info_from_config = config.get_field_info(self.name)
        config.prepare_field(self)
        new_alias = info_from_config.get('alias')
        new_alias_priority = info_from_config.get('alias_priority') or 0
        if new_alias and new_alias_priority >= (self.field_info.alias_priority or 0):
            self.field_info.alias = new_alias
            self.field_info.alias_priority = new_alias_priority
            self.alias = new_alias
        new_exclude = info_from_config.get('exclude')
        if new_exclude is not None:
            self.field_info.exclude = ValueItems.merge(self.field_info.exclude, new_exclude)
        new_include = info_from_config.get('include')
        if new_include is not None:
            self.field_info.include = ValueItems.merge(self.field_info.include, new_include, intersect=True)

    @property
    def alt_alias(self) -> bool:
        return self.name != self.alias

    def prepare(self) -> None:
        self._set_default_and_type()
        if self.type_.__class__ is ForwardRef or self.type_.__class__ is DeferredType:
            return

        self._type_analysis()
        if self.required is Undefined:
            self.required = True
        if self.default is Undefined and self.default