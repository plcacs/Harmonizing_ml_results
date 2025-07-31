import re
import warnings
from collections import defaultdict
from dataclasses import is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    FrozenSet,
    Generic,
    Iterable,
    List,
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
from uuid import UUID
from typing_extensions import Annotated, Literal

from pydantic.v1.fields import (
    MAPPING_LIKE_SHAPES,
    SHAPE_DEQUE,
    SHAPE_FROZENSET,
    SHAPE_GENERIC,
    SHAPE_ITERABLE,
    SHAPE_LIST,
    SHAPE_SEQUENCE,
    SHAPE_SET,
    SHAPE_SINGLETON,
    SHAPE_TUPLE,
    SHAPE_TUPLE_ELLIPSIS,
    FieldInfo,
    ModelField,
)
from pydantic.v1.json import pydantic_encoder
from pydantic.v1.networks import AnyUrl, EmailStr
from pydantic.v1.types import (
    ConstrainedDecimal,
    ConstrainedFloat,
    ConstrainedFrozenSet,
    ConstrainedInt,
    ConstrainedList,
    ConstrainedSet,
    ConstrainedStr,
    SecretBytes,
    SecretStr,
    StrictBytes,
    StrictStr,
    conbytes,
    condecimal,
    confloat,
    confrozenset,
    conint,
    conlist,
    conset,
    constr,
)
from pydantic.v1.typing import (
    all_literal_values,
    get_args,
    get_origin,
    get_sub_types,
    is_callable_type,
    is_literal_type,
    is_namedtuple,
    is_none_type,
    is_union,
)
from pydantic.v1.utils import ROOT_KEY, get_model, lenient_issubclass

if False:  # TYPE_CHECKING
    from pydantic.v1.dataclasses import Dataclass
    from pydantic.v1.main import BaseModel

default_prefix: str = '#/definitions/'
default_ref_template: str = '#/definitions/{model}'
TypeModelOrEnum = Union[Type['BaseModel'], Type[Enum]]
TypeModelSet = Set[TypeModelOrEnum]


def _apply_modify_schema(modify_schema: Callable, field: ModelField, field_schema: Dict[str, Any]) -> None:
    from inspect import signature

    sig = signature(modify_schema)
    args = set(sig.parameters.keys())
    if 'field' in args or 'kwargs' in args:
        modify_schema(field_schema, field=field)
    else:
        modify_schema(field_schema)


def schema(
    models: Iterable[Any],
    *,
    by_alias: bool = True,
    title: Optional[str] = None,
    description: Optional[str] = None,
    ref_prefix: Optional[str] = None,
    ref_template: str = default_ref_template,
) -> Dict[str, Any]:
    """
    Process a list of models and generate a single JSON Schema with all of them defined in the ``definitions``
    top-level JSON key, including their sub-models.
    """
    clean_models = [get_model(model) for model in models]
    flat_models = get_flat_models_from_models(clean_models)
    model_name_map = get_model_name_map(flat_models)
    definitions: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    if title:
        output_schema['title'] = title
    if description:
        output_schema['description'] = description
    for model in clean_models:
        m_schema, m_definitions, m_nested_models = model_process_schema(
            model, by_alias=by_alias, model_name_map=model_name_map, ref_prefix=ref_prefix, ref_template=ref_template
        )
        definitions.update(m_definitions)
        model_name = model_name_map[model]
        definitions[model_name] = m_schema
    if definitions:
        output_schema['definitions'] = definitions
    return output_schema


def model_schema(
    model: Any,
    *,
    by_alias: bool = True,
    ref_prefix: Optional[str] = None,
    ref_template: str = default_ref_template,
) -> Dict[str, Any]:
    """
    Generate a JSON Schema for one model. With all the sub-models defined in the ``definitions`` top-level
    JSON key.
    """
    model = get_model(model)
    flat_models = get_flat_models_from_model(model)
    model_name_map = get_model_name_map(flat_models)
    model_name = model_name_map[model]
    m_schema, m_definitions, nested_models = model_process_schema(
        model, by_alias=by_alias, model_name_map=model_name_map, ref_prefix=ref_prefix, ref_template=ref_template
    )
    if model_name in nested_models:
        m_definitions[model_name] = m_schema
        m_schema = get_schema_ref(model_name, ref_prefix, ref_template, False)
    if m_definitions:
        m_schema.update({'definitions': m_definitions})
    return m_schema


def get_field_info_schema(field: ModelField, schema_overrides: bool = False) -> Tuple[Dict[str, Any], bool]:
    schema_: Dict[str, Any] = {}
    if field.field_info.title or not lenient_issubclass(field.type_, Enum):
        schema_['title'] = field.field_info.title or field.alias.title().replace('_', ' ')
    if field.field_info.title:
        schema_overrides = True
    if field.field_info.description:
        schema_['description'] = field.field_info.description
        schema_overrides = True
    if not field.required and field.default is not None and (not is_callable_type(field.outer_type_)):
        schema_['default'] = encode_default(field.default)
        schema_overrides = True
    return (schema_, schema_overrides)


def field_schema(
    field: ModelField,
    *,
    by_alias: bool = True,
    model_name_map: Dict[Any, str],
    ref_prefix: Optional[str] = None,
    ref_template: str = default_ref_template,
    known_models: Optional[Set[Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Set[Any]]:
    """
    Process a Pydantic field and return a tuple with a JSON Schema for it as the first item.
    Also return a dictionary of definitions with models as keys and their schemas as values.
    """
    s, schema_overrides = get_field_info_schema(field)
    validation_schema = get_field_schema_validations(field)
    if validation_schema:
        s.update(validation_schema)
        schema_overrides = True
    f_schema, f_definitions, f_nested_models = field_type_schema(
        field,
        by_alias=by_alias,
        model_name_map=model_name_map,
        schema_overrides=schema_overrides,
        ref_prefix=ref_prefix,
        ref_template=ref_template,
        known_models=known_models or set(),
    )
    if '$ref' in f_schema:
        return (f_schema, f_definitions, f_nested_models)
    else:
        s.update(f_schema)
        return (s, f_definitions, f_nested_models)


numeric_types = (int, float, Decimal)
_str_types_attrs: Tuple[Tuple[str, Any, str], ...] = (
    ('max_length', numeric_types, 'maxLength'),
    ('min_length', numeric_types, 'minLength'),
    ('regex', str, 'pattern'),
)
_numeric_types_attrs: Tuple[Tuple[str, Any, str], ...] = (
    ('gt', numeric_types, 'exclusiveMinimum'),
    ('lt', numeric_types, 'exclusiveMaximum'),
    ('ge', numeric_types, 'minimum'),
    ('le', numeric_types, 'maximum'),
    ('multiple_of', numeric_types, 'multipleOf'),
)


def get_field_schema_validations(field: ModelField) -> Dict[str, Any]:
    """
    Get the JSON Schema validation keywords for a ``field`` with an annotation of
    a Pydantic ``FieldInfo`` with validation arguments.
    """
    f_schema: Dict[str, Any] = {}
    if lenient_issubclass(field.type_, Enum):
        if field.field_info.extra:
            f_schema.update(field.field_info.extra)
        return f_schema
    if lenient_issubclass(field.type_, (str, bytes)):
        for attr_name, t, keyword in _str_types_attrs:
            attr = getattr(field.field_info, attr_name, None)
            if isinstance(attr, t):
                f_schema[keyword] = attr
    if lenient_issubclass(field.type_, numeric_types) and (not issubclass(field.type_, bool)):
        for attr_name, t, keyword in _numeric_types_attrs:
            attr = getattr(field.field_info, attr_name, None)
            if isinstance(attr, t):
                f_schema[keyword] = attr
    if field.field_info is not None and field.field_info.const:
        f_schema['const'] = field.default
    if field.field_info.extra:
        f_schema.update(field.field_info.extra)
    modify_schema = getattr(field.outer_type_, '__modify_schema__', None)
    if modify_schema:
        _apply_modify_schema(modify_schema, field, f_schema)
    return f_schema


def get_model_name_map(unique_models: Set[Any]) -> Dict[Any, str]:
    """
    Process a set of models and generate unique names for them to be used as keys in the JSON Schema
    definitions.
    """
    name_model_map: Dict[str, Any] = {}
    conflicting_names: Set[str] = set()
    for model in unique_models:
        model_name = normalize_name(model.__name__)
        if model_name in conflicting_names:
            model_name = get_long_model_name(model)
            name_model_map[model_name] = model
        elif model_name in name_model_map:
            conflicting_names.add(model_name)
            conflicting_model = name_model_map.pop(model_name)
            name_model_map[get_long_model_name(conflicting_model)] = conflicting_model
            name_model_map[get_long_model_name(model)] = model
        else:
            name_model_map[model_name] = model
    return {v: k for k, v in name_model_map.items()}


def get_flat_models_from_model(model: Any, known_models: Optional[Set[Any]] = None) -> Set[Any]:
    """
    Take a single ``model`` and generate a set with itself and all the sub-models in the tree.
    """
    known_models = known_models or set()
    flat_models: Set[Any] = set()
    flat_models.add(model)
    known_models |= flat_models
    fields = cast(Sequence[ModelField], model.__fields__.values())
    flat_models |= get_flat_models_from_fields(fields, known_models=known_models)
    return flat_models


def get_flat_models_from_field(field: ModelField, known_models: Set[Any]) -> Set[Any]:
    """
    Take a single Pydantic ``ModelField`` and generate a set with its model and all the sub-models in the tree.
    """
    from pydantic.v1.main import BaseModel

    flat_models: Set[Any] = set()
    field_type = field.type_
    if lenient_issubclass(getattr(field_type, '__pydantic_model__', None), BaseModel):
        field_type = field_type.__pydantic_model__
    if field.sub_fields and (not lenient_issubclass(field_type, BaseModel)):
        flat_models |= get_flat_models_from_fields(field.sub_fields, known_models=known_models)
    elif lenient_issubclass(field_type, BaseModel) and field_type not in known_models:
        flat_models |= get_flat_models_from_model(field_type, known_models=known_models)
    elif lenient_issubclass(field_type, Enum):
        flat_models.add(field_type)
    return flat_models


def get_flat_models_from_fields(fields: Iterable[ModelField], known_models: Set[Any]) -> Set[Any]:
    """
    Take a list of Pydantic  ``ModelField``s and generate a set with their models and all the sub-models in the tree.
    """
    flat_models: Set[Any] = set()
    for field in fields:
        flat_models |= get_flat_models_from_field(field, known_models=known_models)
    return flat_models


def get_flat_models_from_models(models: Iterable[Any]) -> Set[Any]:
    """
    Take a list of ``models`` and generate a set with them and all their sub-models in their trees.
    """
    flat_models: Set[Any] = set()
    for model in models:
        flat_models |= get_flat_models_from_model(model)
    return flat_models


def get_long_model_name(model: Any) -> str:
    return f'{model.__module__}__{model.__qualname__}'.replace('.', '__')


def field_type_schema(
    field: ModelField,
    *,
    by_alias: bool,
    model_name_map: Dict[Any, str],
    ref_template: str,
    schema_overrides: bool = False,
    ref_prefix: Optional[str] = None,
    known_models: Set[Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Set[Any]]:
    """
    Generate the schema for the field's type.
    """
    from pydantic.v1.main import BaseModel

    definitions: Dict[str, Any] = {}
    nested_models: Set[Any] = set()
    if field.shape in {
        SHAPE_LIST,
        SHAPE_TUPLE_ELLIPSIS,
        SHAPE_SEQUENCE,
        SHAPE_SET,
        SHAPE_FROZENSET,
        SHAPE_ITERABLE,
        SHAPE_DEQUE,
    }:
        items_schema, f_definitions, f_nested_models = field_singleton_schema(
            field,
            by_alias=by_alias,
            model_name_map=model_name_map,
            ref_prefix=ref_prefix,
            ref_template=ref_template,
            known_models=known_models,
        )
        definitions.update(f_definitions)
        nested_models.update(f_nested_models)
        f_schema: Dict[str, Any] = {'type': 'array', 'items': items_schema}
        if field.shape in {SHAPE_SET, SHAPE_FROZENSET}:
            f_schema['uniqueItems'] = True
    elif field.shape in MAPPING_LIKE_SHAPES:
        f_schema = {'type': 'object'}
        key_field: ModelField = cast(ModelField, field.key_field)
        regex = getattr(key_field.type_, 'regex', None)
        items_schema, f_definitions, f_nested_models = field_singleton_schema(
            field,
            by_alias=by_alias,
            model_name_map=model_name_map,
            ref_prefix=ref_prefix,
            ref_template=ref_template,
            known_models=known_models,
        )
        definitions.update(f_definitions)
        nested_models.update(f_nested_models)
        if regex:
            f_schema['patternProperties'] = {ConstrainedStr._get_pattern(regex): items_schema}
        if items_schema:
            f_schema['additionalProperties'] = items_schema
    elif field.shape == SHAPE_TUPLE or (field.shape == SHAPE_GENERIC and (not issubclass(field.type_, BaseModel))):
        sub_schema = []
        sub_fields = cast(List[ModelField], field.sub_fields)
        for sf in sub_fields:
            sf_schema, sf_definitions, sf_nested_models = field_type_schema(
                sf,
                by_alias=by_alias,
                model_name_map=model_name_map,
                ref_prefix=ref_prefix,
                ref_template=ref_template,
                known_models=known_models,
            )
            definitions.update(sf_definitions)
            nested_models.update(sf_nested_models)
            sub_schema.append(sf_schema)
        sub_fields_len = len(sub_fields)
        if field.shape == SHAPE_GENERIC:
            all_of_schemas = sub_schema[0] if sub_fields_len == 1 else {'type': 'array', 'items': sub_schema}
            f_schema = {'allOf': [all_of_schemas]}
        else:
            f_schema = {'type': 'array', 'minItems': sub_fields_len, 'maxItems': sub_fields_len}
            if sub_fields_len >= 1:
                f_schema['items'] = sub_schema
    else:
        # SHAPE_SINGLETON or SHAPE_GENERIC with BaseModel
        assert field.shape in {SHAPE_SINGLETON, SHAPE_GENERIC}, field.shape
        f_schema, f_definitions, f_nested_models = field_singleton_schema(
            field,
            by_alias=by_alias,
            model_name_map=model_name_map,
            schema_overrides=schema_overrides,
            ref_prefix=ref_prefix,
            ref_template=ref_template,
            known_models=known_models,
        )
        definitions.update(f_definitions)
        nested_models.update(f_nested_models)
    if field.type_ != field.outer_type_:
        if field.shape == SHAPE_GENERIC:
            field_type = field.type_
        else:
            field_type = field.outer_type_
        modify_schema = getattr(field_type, '__modify_schema__', None)
        if modify_schema:
            _apply_modify_schema(modify_schema, field, f_schema)
    return (f_schema, definitions, nested_models)


def model_process_schema(
    model: Any,
    *,
    by_alias: bool = True,
    model_name_map: Dict[Any, str],
    ref_prefix: Optional[str] = None,
    ref_template: str = default_ref_template,
    known_models: Optional[Set[Any]] = None,
    field: Optional[ModelField] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Set[Any]]:
    """
    Generate the schema for the model.
    """
    from inspect import getdoc, signature

    known_models = known_models or set()
    if lenient_issubclass(model, Enum):
        model = cast(Type[Enum], model)
        s = enum_process_schema(model, field=field)
        return (s, {}, set())
    model = cast(Type[Any], model)
    s: Dict[str, Any] = {'title': model.__config__.title or model.__name__}
    doc = getdoc(model)
    if doc:
        s['description'] = doc
    known_models.add(model)
    m_schema, m_definitions, nested_models = model_type_schema(
        model,
        by_alias=by_alias,
        model_name_map=model_name_map,
        ref_template=ref_template,
        ref_prefix=ref_prefix,
        known_models=known_models,
    )
    s.update(m_schema)
    schema_extra = model.__config__.schema_extra
    if callable(schema_extra):
        if len(signature(schema_extra).parameters) == 1:
            schema_extra(s)
        else:
            schema_extra(s, model)
    else:
        s.update(schema_extra)
    return (s, m_definitions, nested_models)


def model_type_schema(
    model: Any,
    *,
    by_alias: bool,
    model_name_map: Dict[Any, str],
    ref_template: str,
    ref_prefix: Optional[str] = None,
    known_models: Set[Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Set[Any]]:
    """
    Generate the schema for the model's type.
    """
    properties: Dict[str, Any] = {}
    required: List[str] = []
    definitions: Dict[str, Any] = {}
    nested_models: Set[Any] = set()
    for k, f in model.__fields__.items():
        try:
            f_schema, f_definitions, f_nested_models = field_schema(
                f,
                by_alias=by_alias,
                model_name_map=model_name_map,
                ref_prefix=ref_prefix,
                ref_template=ref_template,
                known_models=known_models,
            )
        except SkipField as skip:
            warnings.warn(skip.message, UserWarning)
            continue
        definitions.update(f_definitions)
        nested_models.update(f_nested_models)
        if by_alias:
            properties[f.alias] = f_schema
            if f.required:
                required.append(f.alias)
        else:
            properties[k] = f_schema
            if f.required:
                required.append(k)
    if ROOT_KEY in properties:
        out_schema: Dict[str, Any] = properties[ROOT_KEY]
        out_schema['title'] = model.__config__.title or model.__name__
    else:
        out_schema = {'type': 'object', 'properties': properties}
        if required:
            out_schema['required'] = required
    if model.__config__.extra == 'forbid':
        out_schema['additionalProperties'] = False
    return (out_schema, definitions, nested_models)


def enum_process_schema(enum: Type[Enum], *, field: Optional[ModelField] = None) -> Dict[str, Any]:
    """
    Generate the schema for an Enum.
    """
    import inspect

    schema_: Dict[str, Any] = {
        'title': enum.__name__,
        'description': inspect.cleandoc(enum.__doc__ or 'An enumeration.'),
        'enum': [item.value for item in cast(Iterable[Enum], enum)],
    }
    add_field_type_to_schema(enum, schema_)
    modify_schema = getattr(enum, '__modify_schema__', None)
    if modify_schema:
        _apply_modify_schema(modify_schema, field, schema_)
    return schema_


def field_singleton_sub_fields_schema(
    field: ModelField,
    *,
    by_alias: bool,
    model_name_map: Dict[Any, str],
    ref_template: str,
    schema_overrides: bool = False,
    ref_prefix: Optional[str] = None,
    known_models: Set[Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Set[Any]]:
    """
    Generate schema for a type with parameters (sub-fields).
    """
    sub_fields = cast(List[ModelField], field.sub_fields)
    definitions: Dict[str, Any] = {}
    nested_models: Set[Any] = set()
    if len(sub_fields) == 1:
        return field_type_schema(
            sub_fields[0],
            by_alias=by_alias,
            model_name_map=model_name_map,
            schema_overrides=schema_overrides,
            ref_prefix=ref_prefix,
            ref_template=ref_template,
            known_models=known_models,
        )
    else:
        s: Dict[str, Any] = {}
        field_has_discriminator: bool = field.discriminator_key is not None
        if field_has_discriminator:
            assert field.sub_fields_mapping is not None
            discriminator_models_refs: Dict[Any, Any] = {}
            for discriminator_value, sub_field in field.sub_fields_mapping.items():
                if isinstance(discriminator_value, Enum):
                    discriminator_value = str(discriminator_value.value)
                if is_union(get_origin(sub_field.type_)):
                    sub_models = get_sub_types(sub_field.type_)
                    discriminator_models_refs[discriminator_value] = {
                        model_name_map[sub_model]: get_schema_ref(model_name_map[sub_model], ref_prefix, ref_template, False)
                        for sub_model in sub_models
                    }
                else:
                    sub_field_type = sub_field.type_
                    if hasattr(sub_field_type, '__pydantic_model__'):
                        sub_field_type = sub_field_type.__pydantic_model__
                    discriminator_model_name = model_name_map[sub_field_type]
                    discriminator_model_ref = get_schema_ref(discriminator_model_name, ref_prefix, ref_template, False)
                    discriminator_models_refs[discriminator_value] = discriminator_model_ref['$ref']
            s['discriminator'] = {
                'propertyName': field.discriminator_alias if by_alias else field.discriminator_key,
                'mapping': discriminator_models_refs,
            }
        sub_field_schemas: List[Dict[str, Any]] = []
        for sf in sub_fields:
            sub_schema, sub_definitions, sub_nested_models = field_type_schema(
                sf,
                by_alias=by_alias,
                model_name_map=model_name_map,
                schema_overrides=schema_overrides,
                ref_prefix=ref_prefix,
                ref_template=ref_template,
                known_models=known_models,
            )
            definitions.update(sub_definitions)
            if schema_overrides and 'allOf' in sub_schema:
                sub_schema = sub_schema['allOf'][0]
            if set(sub_schema.keys()) == {'discriminator', 'oneOf'}:
                sub_schema.pop('discriminator')
            sub_field_schemas.append(sub_schema)
            nested_models.update(sub_nested_models)
        s['oneOf' if field_has_discriminator else 'anyOf'] = sub_field_schemas
        return (s, definitions, nested_models)


field_class_to_schema: Tuple[Tuple[Any, Dict[str, Any]], ...] = (
    (Path, {'type': 'string', 'format': 'path'}),
    (datetime, {'type': 'string', 'format': 'date-time'}),
    (date, {'type': 'string', 'format': 'date'}),
    (time, {'type': 'string', 'format': 'time'}),
    (timedelta, {'type': 'number', 'format': 'time-delta'}),
    (IPv4Network, {'type': 'string', 'format': 'ipv4network'}),
    (IPv6Network, {'type': 'string', 'format': 'ipv6network'}),
    (IPv4Interface, {'type': 'string', 'format': 'ipv4interface'}),
    (IPv6Interface, {'type': 'string', 'format': 'ipv6interface'}),
    (IPv4Address, {'type': 'string', 'format': 'ipv4'}),
    (IPv6Address, {'type': 'string', 'format': 'ipv6'}),
    (Pattern, {'type': 'string', 'format': 'regex'}),
    (str, {'type': 'string'}),
    (bytes, {'type': 'string', 'format': 'binary'}),
    (bool, {'type': 'boolean'}),
    (int, {'type': 'integer'}),
    (float, {'type': 'number'}),
    (Decimal, {'type': 'number'}),
    (UUID, {'type': 'string', 'format': 'uuid'}),
    (dict, {'type': 'object'}),
    (list, {'type': 'array', 'items': {}}),
    (tuple, {'type': 'array', 'items': {}}),
    (set, {'type': 'array', 'items': {}, 'uniqueItems': True}),
    (frozenset, {'type': 'array', 'items': {}, 'uniqueItems': True}),
)
json_scheme: Dict[str, str] = {'type': 'string', 'format': 'json-string'}


def add_field_type_to_schema(field_type: Any, schema_: Dict[str, Any]) -> None:
    """
    Update the given `schema` with the type-specific metadata for the given `field_type`.
    """
    for type_, t_schema in field_class_to_schema:
        if lenient_issubclass(field_type, type_) or (field_type is type_ is Pattern):
            schema_.update(t_schema)
            break


def get_schema_ref(name: str, ref_prefix: Optional[str], ref_template: str, schema_overrides: bool) -> Dict[str, Any]:
    if ref_prefix:
        schema_ref: Dict[str, Any] = {'$ref': ref_prefix + name}
    else:
        schema_ref = {'$ref': ref_template.format(model=name)}
    return {'allOf': [schema_ref]} if schema_overrides else schema_ref


def field_singleton_schema(
    field: ModelField,
    *,
    by_alias: bool,
    model_name_map: Dict[Any, str],
    ref_template: str,
    schema_overrides: bool = False,
    ref_prefix: Optional[str] = None,
    known_models: Set[Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Set[Any]]:
    """
    Generate the schema for a single field.
    """
    from pydantic.v1.main import BaseModel

    definitions: Dict[str, Any] = {}
    nested_models: Set[Any] = set()
    field_type = field.type_
    if field.sub_fields and (field.field_info and field.field_info.const or not lenient_issubclass(field_type, BaseModel)):
        return field_singleton_sub_fields_schema(
            field,
            by_alias=by_alias,
            model_name_map=model_name_map,
            schema_overrides=schema_overrides,
            ref_prefix=ref_prefix,
            ref_template=ref_template,
            known_models=known_models,
        )
    if field_type is Any or field_type is object or field_type.__class__ == TypeVar or (get_origin(field_type) is type):
        return ({}, definitions, nested_models)
    if is_none_type(field_type):
        return ({'type': 'null'}, definitions, nested_models)
    if is_callable_type(field_type):
        raise SkipField(f'Callable {field.name} was excluded from schema since JSON schema has no equivalent type.')
    f_schema: Dict[str, Any] = {}
    if field.field_info is not None and field.field_info.const:
        f_schema['const'] = field.default
    if is_literal_type(field_type):
        values = tuple((x.value if isinstance(x, Enum) else x for x in all_literal_values(field_type)))
        if len({v.__class__ for v in values}) > 1:
            return field_schema(
                multitypes_literal_field_for_schema(values, field),
                by_alias=by_alias,
                model_name_map=model_name_map,
                ref_prefix=ref_prefix,
                ref_template=ref_template,
                known_models=known_models,
            )
        field_type = values[0].__class__
        f_schema['enum'] = list(values)
        add_field_type_to_schema(field_type, f_schema)
    elif lenient_issubclass(field_type, Enum):
        enum_name = model_name_map[field_type]
        f_schema, schema_overrides = get_field_info_schema(field, schema_overrides)
        f_schema.update(get_schema_ref(enum_name, ref_prefix, ref_template, schema_overrides))
        definitions[enum_name] = enum_process_schema(field_type, field=field)
    elif is_namedtuple(field_type):
        sub_schema, *_ = model_process_schema(
            field_type.__pydantic_model__,
            by_alias=by_alias,
            model_name_map=model_name_map,
            ref_prefix=ref_prefix,
            ref_template=ref_template,
            known_models=known_models,
            field=field,
        )
        items_schemas = list(sub_schema['properties'].values())
        f_schema.update(
            {
                'type': 'array',
                'items': items_schemas,
                'minItems': len(items_schemas),
                'maxItems': len(items_schemas),
            }
        )
    elif not hasattr(field_type, '__pydantic_model__'):
        add_field_type_to_schema(field_type, f_schema)
        modify_schema = getattr(field_type, '__modify_schema__', None)
        if modify_schema:
            _apply_modify_schema(modify_schema, field, f_schema)
    if f_schema:
        return (f_schema, definitions, nested_models)
    if lenient_issubclass(getattr(field_type, '__pydantic_model__', None), BaseModel):
        field_type = field_type.__pydantic_model__
    if issubclass(field_type, BaseModel):
        model_name = model_name_map[field_type]
        if field_type not in known_models:
            sub_schema, sub_definitions, sub_nested_models = model_process_schema(
                field_type,
                by_alias=by_alias,
                model_name_map=model_name_map,
                ref_prefix=ref_prefix,
                ref_template=ref_template,
                known_models=known_models,
                field=field,
            )
            definitions.update(sub_definitions)
            definitions[model_name] = sub_schema
            nested_models.update(sub_nested_models)
        else:
            nested_models.add(model_name)
        schema_ref = get_schema_ref(model_name, ref_prefix, ref_template, schema_overrides)
        return (schema_ref, definitions, nested_models)
    args = get_args(field_type)
    if args is not None and (not args) and (Generic in field_type.__bases__):
        return (f_schema, definitions, nested_models)
    raise ValueError(f'Value not declarable with JSON Schema, field: {field}')


def multitypes_literal_field_for_schema(values: Iterable[Any], field: ModelField) -> ModelField:
    """
    To support `Literal` with values of different types, we split it into multiple `Literal` with same type.
    """
    literal_distinct_types: Dict[Any, List[Any]] = defaultdict(list)
    for v in values:
        literal_distinct_types[v.__class__].append(v)
    distinct_literals = (Literal[tuple(same_type_values)] for same_type_values in literal_distinct_types.values())
    return ModelField(
        name=field.name,
        type_=Union[tuple(distinct_literals)],
        class_validators=field.class_validators,
        model_config=field.model_config,
        default=field.default,
        required=field.required,
        alias=field.alias,
        field_info=field.field_info,
    )


def encode_default(dft: Any) -> Any:
    from pydantic.v1.main import BaseModel

    if isinstance(dft, BaseModel) or is_dataclass(dft):
        dft = cast('dict[str, Any]', pydantic_encoder(dft))
    if isinstance(dft, dict):
        return {encode_default(k): encode_default(v) for k, v in dft.items()}
    elif isinstance(dft, Enum):
        return dft.value
    elif isinstance(dft, (int, float, str)):
        return dft
    elif isinstance(dft, (list, tuple)):
        t = dft.__class__
        seq_args = (encode_default(v) for v in dft)
        return t(seq_args) if not is_namedtuple(t) else t(*seq_args)
    elif dft is None:
        return None
    else:
        return pydantic_encoder(dft)


_map_types_constraint: Dict[Type[Any], Callable[..., Any]] = {int: conint, float: confloat, Decimal: condecimal}


def get_annotation_from_field_info(annotation: Any, field_info: FieldInfo, field_name: str, validate_assignment: bool = False) -> Any:
    """
    Get an annotation with validation implemented for numbers and strings based on the field_info.
    """
    constraints = field_info.get_constraints()
    used_constraints: Set[str] = set()
    if constraints:
        annotation, used_constraints = get_annotation_with_constraints(annotation, field_info)
    if validate_assignment:
        used_constraints.add('allow_mutation')
    unused_constraints = constraints - used_constraints
    if unused_constraints:
        raise ValueError(
            f'On field "{field_name}" the following field constraints are set but not enforced: {", ".join(unused_constraints)}. \nFor more details see https://docs.pydantic.dev/usage/schema/#unenforced-field-constraints'
        )
    return annotation


def get_annotation_with_constraints(annotation: Any, field_info: FieldInfo) -> Tuple[Any, Set[str]]:
    """
    Get an annotation with used constraints implemented for numbers and strings based on the field_info.
    """
    used_constraints: Set[str] = set()

    def go(type_: Any) -> Any:
        if is_literal_type(type_) or isinstance(type_, ForwardRef) or lenient_issubclass(type_, (ConstrainedList, ConstrainedSet, ConstrainedFrozenSet)):
            return type_
        origin = get_origin(type_)
        if origin is not None:
            args = get_args(type_)
            if any((isinstance(a, ForwardRef) for a in args)):
                return type_
            if origin is Annotated:
                return go(args[0])
            if is_union(origin):
                return Union[tuple((go(a) for a in args))]
            if issubclass(origin, List) and (field_info.min_items is not None or field_info.max_items is not None or field_info.unique_items is not None):
                used_constraints.update({'min_items', 'max_items', 'unique_items'})
                return conlist(go(args[0]), min_items=field_info.min_items, max_items=field_info.max_items, unique_items=field_info.unique_items)
            if issubclass(origin, Set) and (field_info.min_items is not None or field_info.max_items is not None):
                used_constraints.update({'min_items', 'max_items'})
                return conset(go(args[0]), min_items=field_info.min_items, max_items=field_info.max_items)
            if issubclass(origin, FrozenSet) and (field_info.min_items is not None or field_info.max_items is not None):
                used_constraints.update({'min_items', 'max_items'})
                return confrozenset(go(args[0]), min_items=field_info.min_items, max_items=field_info.max_items)
            for t in (Tuple, List, Set, FrozenSet, Sequence):
                if issubclass(origin, t):
                    return t[tuple((go(a) for a in args))]
            if issubclass(origin, Dict):
                return Dict[args[0], go(args[1])]
        attrs = None
        constraint_func: Optional[Callable[..., Any]] = None
        if isinstance(type_, type):
            if issubclass(type_, (SecretStr, SecretBytes)):
                attrs = ('max_length', 'min_length')

                def constraint_func(**kw: Any) -> Any:
                    return type(type_.__name__, (type_,), kw)

            elif issubclass(type_, str) and (not issubclass(type_, (EmailStr, AnyUrl))):
                attrs = ('max_length', 'min_length', 'regex')
                if issubclass(type_, StrictStr):

                    def constraint_func(**kw: Any) -> Any:
                        return type(type_.__name__, (type_,), kw)
                else:
                    constraint_func = constr
            elif issubclass(type_, bytes):
                attrs = ('max_length', 'min_length', 'regex')
                if issubclass(type_, StrictBytes):

                    def constraint_func(**kw: Any) -> Any:
                        return type(type_.__name__, (type_,), kw)
                else:
                    constraint_func = conbytes
            elif issubclass(type_, numeric_types) and (not issubclass(type_, (ConstrainedInt, ConstrainedFloat, ConstrainedDecimal, ConstrainedList, ConstrainedSet, ConstrainedFrozenSet, bool))):
                attrs = ('gt', 'lt', 'ge', 'le', 'multiple_of')
                if issubclass(type_, float):
                    attrs += ('allow_inf_nan',)
                if issubclass(type_, Decimal):
                    attrs += ('max_digits', 'decimal_places')
                numeric_type = next((t for t in numeric_types if issubclass(type_, t)))
                constraint_func = _map_types_constraint[numeric_type]
        if attrs:
            used_constraints.update(set(attrs))
            kwargs = {attr_name: getattr(field_info, attr_name) for attr_name in attrs if getattr(field_info, attr_name) is not None}
            if kwargs:
                constraint_func = cast(Callable[..., Any], constraint_func)
                return constraint_func(**kwargs)
        return type_

    return (go(annotation), used_constraints)


def normalize_name(name: str) -> str:
    """
    Normalizes the given name.
    """
    return re.sub('[^a-zA-Z0-9.\\-_]', '_', name)


class SkipField(Exception):
    """
    Utility exception used to exclude fields from schema.
    """

    def __init__(self, message: str) -> None:
        self.message = message
