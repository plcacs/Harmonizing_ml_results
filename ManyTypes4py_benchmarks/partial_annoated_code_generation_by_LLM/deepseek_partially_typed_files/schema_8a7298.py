import re
import warnings
from collections import defaultdict
from dataclasses import is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, ForwardRef, FrozenSet, Generic, Iterable, List, Optional, Pattern, Sequence, Set, Tuple, Type, TypeVar, Union, cast
from uuid import UUID
from typing_extensions import Annotated, Literal
from pydantic.v1.fields import MAPPING_LIKE_SHAPES, SHAPE_DEQUE, SHAPE_FROZENSET, SHAPE_GENERIC, SHAPE_ITERABLE, SHAPE_LIST, SHAPE_SEQUENCE, SHAPE_SET, SHAPE_SINGLETON, SHAPE_TUPLE, SHAPE_TUPLE_ELLIPSIS, FieldInfo, ModelField
from pydantic.v1.json import pydantic_encoder
from pydantic.v1.networks import AnyUrl, EmailStr
from pydantic.v1.types import ConstrainedDecimal, ConstrainedFloat, ConstrainedFrozenSet, ConstrainedInt, ConstrainedList, ConstrainedSet, ConstrainedStr, SecretBytes, SecretStr, StrictBytes, StrictStr, conbytes, condecimal, confloat, confrozenset, conint, conlist, conset, constr
from pydantic.v1.typing import all_literal_values, get_args, get_origin, get_sub_types, is_callable_type, is_literal_type, is_namedtuple, is_none_type, is_union
from pydantic.v1.utils import ROOT_KEY, get_model, lenient_issubclass

if TYPE_CHECKING:
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

def schema(models: List[Type['BaseModel']], *, by_alias: bool = True, title: Optional[str] = None, description: Optional[str] = None, ref_prefix: Optional[str] = None, ref_template: str = default_ref_template) -> Dict[str, Any]:
    """
    Process a list of models and generate a single JSON Schema with all of them defined in the ``definitions``
    top-level JSON key, including their sub-models.

    :param models: a list of models to include in the generated JSON Schema
    :param by_alias: generate the schemas using the aliases defined, if any
    :param title: title for the generated schema that includes the definitions
    :param description: description for the generated schema
    :param ref_prefix: the JSON Pointer prefix for schema references with ``$ref``, if None, will be set to the
      default of ``#/definitions/``. Update it if you want the schemas to reference the definitions somewhere
      else, e.g. for OpenAPI use ``#/components/schemas/``. The resulting generated schemas will still be at the
      top-level key ``definitions``, so you can extract them from there. But all the references will have the set
      prefix.
    :param ref_template: Use a ``string.format()`` template for ``$ref`` instead of a prefix. This can be useful
      for references that cannot be represented by ``ref_prefix`` such as a definition stored in another file. For
      a sibling json file in a ``/schemas`` directory use ``"/schemas/${model}.json#"``.
    :return: dict with the JSON Schema with a ``definitions`` top-level key including the schema definitions for
      the models and sub-models passed in ``models``.
    """
    clean_models: List[Type['BaseModel']] = [get_model(model) for model in models]
    flat_models: TypeModelSet = get_flat_models_from_models(clean_models)
    model_name_map: Dict[TypeModelOrEnum, str] = get_model_name_map(flat_models)
    definitions: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    if title:
        output_schema['title'] = title
    if description:
        output_schema['description'] = description
    for model in clean_models:
        (m_schema, m_definitions, m_nested_models) = model_process_schema(model, by_alias=by_alias, model_name_map=model_name_map, ref_prefix=ref_prefix, ref_template=ref_template)
        definitions.update(m_definitions)
        model_name: str = model_name_map[model]
        definitions[model_name] = m_schema
    if definitions:
        output_schema['definitions'] = definitions
    return output_schema

def model_schema(model: Type['BaseModel'], by_alias: bool = True, ref_prefix: Optional[str] = None, ref_template: str = default_ref_template) -> Dict[str, Any]:
    """
    Generate a JSON Schema for one model. With all the sub-models defined in the ``definitions`` top-level
    JSON key.

    :param model: a Pydantic model (a class that inherits from BaseModel)
    :param by_alias: generate the schemas using the aliases defined, if any
    :param ref_prefix: the JSON Pointer prefix for schema references with ``$ref``, if None, will be set to the
      default of ``#/definitions/``. Update it if you want the schemas to reference the definitions somewhere
      else, e.g. for OpenAPI use ``#/components/schemas/``. The resulting generated schemas will still be at the
      top-level key ``definitions``, so you can extract them from there. But all the references will have the set
      prefix.
    :param ref_template: Use a ``string.format()`` template for ``$ref`` instead of a prefix. This can be useful for
      references that cannot be represented by ``ref_prefix`` such as a definition stored in another file. For a
      sibling json file in a ``/schemas`` directory use ``"/schemas/${model}.json#"``.
    :return: dict with the JSON Schema for the passed ``model``
    """
    model = get_model(model)
    flat_models: TypeModelSet = get_flat_models_from_model(model)
    model_name_map: Dict[TypeModelOrEnum, str] = get_model_name_map(flat_models)
    model_name: str = model_name_map[model]
    (m_schema, m_definitions, nested_models) = model_process_schema(model, by_alias=by_alias, model_name_map=model_name_map, ref_prefix=ref_prefix, ref_template=ref_template)
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

def field_schema(field: ModelField, *, by_alias: bool = True, model_name_map: Dict[TypeModelOrEnum, str], ref_prefix: Optional[str] = None, ref_template: str = default_ref_template, known_models: Optional[TypeModelSet] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    """
    Process a Pydantic field and return a tuple with a JSON Schema for it as the first item.
    Also return a dictionary of definitions with models as keys and their schemas as values. If the passed field
    is a model and has sub-models, and those sub-models don't have overrides (as ``title``, ``default``, etc), they
    will be included in the definitions and referenced in the schema instead of included recursively.

    :param field: a Pydantic ``ModelField``
    :param by_alias: use the defined alias (if any) in the returned schema
    :param model_name_map: used to generate the JSON Schema references to other models included in the definitions
    :param ref_prefix: the JSON Pointer prefix to use for references to other schemas, if None, the default of
      #/definitions/ will be used
    :param ref_template: Use a ``string.format()`` template for ``$ref`` instead of a prefix. This can be useful for
      references that cannot be represented by ``ref_prefix`` such as a definition stored in another file. For a
      sibling json file in a ``/schemas`` directory use ``"/schemas/${model}.json#"``.
    :param known_models: used to solve circular references
    :return: tuple of the schema for this field and additional definitions
    """
    (s, schema_overrides) = get_field_info_schema(field)
    validation_schema: Dict[str, Any] = get_field_schema_validations(field)
    if validation_schema:
        s.update(validation_schema)
        schema_overrides = True
    (f_schema, f_definitions, f_nested_models) = field_type_schema(field, by_alias=by_alias, model_name_map=model_name_map, schema_overrides=schema_overrides, ref_prefix=ref_prefix, ref_template=ref_template, known_models=known_models or set())
    if '$ref' in f_schema:
        return (f_schema, f_definitions, f_nested_models)
    else:
        s.update(f_schema)
        return (s, f_definitions, f_nested_models)

numeric_types: Tuple[Type, ...] = (int, float, Decimal)
_str_types_attrs: Tuple[Tuple[str, Tuple[Type, ...], str], ...] = (('max_length', numeric_types, 'maxLength'), ('min_length', numeric_types, 'minLength'), ('regex', (str,), 'pattern'))
_numeric_types_attrs: Tuple[Tuple[str, Tuple[Type, ...], str], ...] = (('gt', numeric_types, 'exclusiveMinimum'), ('lt', numeric_types, 'exclusiveMaximum'), ('ge', numeric_types, 'minimum'), ('le', numeric_types, 'maximum'), ('multiple_of', numeric_types, 'multipleOf'))

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
        for (attr_name, t, keyword) in _str_types_attrs:
            attr = getattr(field.field_info, attr_name, None)
            if isinstance(attr, t):
                f_schema[keyword] = attr
    if lenient_issubclass(field.type_, numeric_types) and (not issubclass(field.type_, bool)):
        for (attr_name, t, keyword) in _numeric_types_attrs:
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

def get_model_name_map(unique_models: TypeModelSet) -> Dict[TypeModelOrEnum, str]:
    """
    Process a set of models and generate unique names for them to be used as keys in the JSON Schema
    definitions. By default the names are the same as the class name. But if two models in different Python
    modules have the same name (e.g. "users.Model" and "items.Model"), the generated names will be
    based on the Python module path for those conflicting models to prevent name collisions.

    :param unique_models: a Python set of models
    :return: dict mapping models to names
    """
    name_model_map: Dict[str, TypeModelOrEnum] = {}
    conflicting_names: Set[str] = set()
    for model in unique_models:
        model_name: str = normalize_name(model.__name__)
        if model_name in conflicting_names:
            model_name = get_long_model_name(model)
            name_model_map[model_name] = model
        elif model_name in name_model_map:
            conflicting_names.add(model_name)
            conflicting_model: TypeModelOrEnum = name_model_map.pop(model_name)
            name_model_map[get_long_model_name(conflicting_model)] = conflicting_model
            name_model_map[get_long_model_name(model)] = model
        else:
            name_model_map[model_name] = model
    return {v: k for (k, v) in name_model_map.items()}

def get_flat_models_from_model(model: Type['BaseModel'], known_models: Optional[TypeModelSet] = None) -> TypeModelSet:
    """
    Take a single ``model`` and generate a set with itself and all the sub-models in the tree. I.e. if you pass
    model ``Foo`` (subclass of Pydantic ``BaseModel``) as ``model``, and it has a field of type ``Bar`` (also
    subclass of ``BaseModel``) and that model ``Bar`` has a field of type ``Baz`` (also subclass of ``BaseModel``),
    the return value will be ``set([Foo, Bar, Baz])``.

    :param model: a Pydantic ``BaseModel`` subclass
    :param known_models: used to solve circular references
    :return: a set with the initial model and all its sub-models
    """
    known_models = known_models or set()
    flat_models: TypeModelSet = set()
    flat_models.add(model)
    known_models |= flat_models
    fields: Sequence[ModelField] = cast(Sequence[ModelField], model.__fields__.values())
    flat_models |= get_flat_models_from_fields(fields, known_models=known_models)
    return flat_models

def get_flat_models_from_field(field: ModelField, known_models: TypeModelSet) -> TypeModelSet:
    """
    Take a single Pydantic ``ModelField`` (from a model) that could have been declared as a subclass of BaseModel
    (so, it could be a submodel), and generate a set with its model and all the sub-models in the tree.
    I.e. if you pass a field that was declared to be of type ``Foo`` (subclass of BaseModel) as ``field``, and that
    model ``Foo`` has a field of type ``Bar`` (also subclass of ``BaseModel``) and that model ``Bar`` has a field of
    type ``Baz`` (also subclass of ``BaseModel``), the return value will be ``set([Foo, Bar, Baz])``.

    :param field: a Pydantic ``ModelField``
    :param known_models: used to solve circular references
    :return: a set with the model used in the declaration for this field, if any, and all its sub-models
    """
    from pydantic.v1.main import BaseModel
    flat_models: TypeModelSet = set()
    field_type: Any = field.type_
    if lenient_issubclass(getattr(field_type, '__pydantic_model__', None), BaseModel):
        field_type = field_type.__pydantic_model__
    if field.sub_fields and (not lenient_issubclass(field_type, BaseModel)):
        flat_models |= get_flat_models_from_fields(field.sub_fields, known_models=known_models)
    elif lenient_issubclass(field_type, BaseModel) and field_type not in known_models:
        flat_models |= get_flat_models_from_model(field_type, known_models=known_models)
    elif lenient_issubclass(field_type, Enum):
        flat_models.add(field_type)
    return flat_models

def get_flat_models_from_fields(fields: Sequence[ModelField], known_models: TypeModelSet) -> TypeModelSet:
    """
    Take a list of Pydantic  ``ModelField``s (from a model) that could have been declared as subclasses of ``BaseModel``
    (so, any of them could be a submodel), and generate a set with their models and all the sub-models in the tree.
    I.e. if you pass a the fields of a model ``Foo`` (subclass of ``BaseModel``) as ``fields``, and on of them has a
    field of type