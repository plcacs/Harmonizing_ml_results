from __future__ import annotations as _annotations
import collections.abc
import dataclasses
import datetime
import inspect
import os
import pathlib
import re
import sys
import typing
import warnings
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import copy
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from functools import partial
from inspect import Parameter, _ParameterKind, signature
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from itertools import chain
from operator import attrgetter
from types import FunctionType, GenericAlias, LambdaType, MethodType
from typing import TYPE_CHECKING, Any, Callable, Final, ForwardRef, Literal, TypeVar, Union, cast, overload
from uuid import UUID
from warnings import warn
import typing_extensions
from pydantic_core import CoreSchema, MultiHostUrl, PydanticCustomError, PydanticSerializationUnexpectedValue, PydanticUndefined, Url, core_schema, to_jsonable_python
from typing_extensions import TypeAliasType, TypedDict, get_args, get_origin, is_typeddict
from ..aliases import AliasChoices, AliasGenerator, AliasPath
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from ..config import ConfigDict, JsonDict, JsonEncoder, JsonSchemaExtraCallable
from ..errors import PydanticSchemaGenerationError, PydanticUndefinedAnnotation, PydanticUserError
from ..functional_validators import AfterValidator, BeforeValidator, FieldValidatorModes, PlainValidator, WrapValidator
from ..json_schema import JsonSchemaValue
from ..version import version_short
from ..warnings import PydanticDeprecatedSince20
from . import _decorators, _discriminated_union, _known_annotated_metadata, _repr, _typing_extra
from ._config import ConfigWrapper, ConfigWrapperStack
from ._core_metadata import CoreMetadata, update_core_metadata
from ._core_utils import get_ref, get_type_ref, is_list_like_schema_with_items_schema, validate_core_schema
from ._decorators import Decorator, DecoratorInfos, FieldSerializerDecoratorInfo, FieldValidatorDecoratorInfo, ModelSerializerDecoratorInfo, ModelValidatorDecoratorInfo, RootValidatorDecoratorInfo, ValidatorDecoratorInfo, get_attribute_from_bases, inspect_field_serializer, inspect_model_serializer, inspect_validator
from ._docs_extraction import extract_docstrings_from_cls
from ._fields import collect_dataclass_fields, rebuild_model_fields, takes_validated_data_argument
from ._forward_ref import PydanticRecursiveRef
from ._generics import get_standard_typevars_map, replace_types
from ._import_utils import import_cached_base_model, import_cached_field_info
from ._mock_val_ser import MockCoreSchema
from ._namespace_utils import NamespacesTuple, NsResolver
from ._schema_gather import MissingDefinitionError, gather_schemas_for_cleaning
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._utils import lenient_issubclass, smart_deepcopy

if TYPE_CHECKING:
    from ..fields import ComputedFieldInfo, FieldInfo
    from ..main import BaseModel
    from ..types import Discriminator
    from ._dataclasses import StandardDataclass
    from ._schema_generation_shared import GetJsonSchemaFunction

_SUPPORTS_TYPEDDICT = sys.version_info >= (3, 12)
FieldDecoratorInfo = Union[ValidatorDecoratorInfo, FieldValidatorDecoratorInfo, FieldSerializerDecoratorInfo]
FieldDecoratorInfoType = TypeVar('FieldDecoratorInfoType', bound=FieldDecoratorInfo)
AnyFieldDecorator = Union[Decorator[ValidatorDecoratorInfo], Decorator[FieldValidatorDecoratorInfo], Decorator[FieldSerializerDecoratorInfo]]
ModifyCoreSchemaWrapHandler = GetCoreSchemaHandler
GetCoreSchemaFunction = Callable[[Any, ModifyCoreSchemaWrapHandler], core_schema.CoreSchema]
TUPLE_TYPES = [typing.Tuple, tuple]
LIST_TYPES = [typing.List, list, collections.abc.MutableSequence]
SET_TYPES = [typing.Set, set, collections.abc.MutableSet]
FROZEN_SET_TYPES = [typing.FrozenSet, frozenset, collections.abc.Set]
DICT_TYPES = [typing.Dict, dict]
IP_TYPES = [IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network]
SEQUENCE_TYPES = [typing.Sequence, collections.abc.Sequence]
ITERABLE_TYPES = [typing.Iterable, collections.abc.Iterable, typing.Generator, collections.abc.Generator]
TYPE_TYPES = [typing.Type, type]
PATTERN_TYPES = [typing.Pattern, re.Pattern]
PATH_TYPES = [os.PathLike, pathlib.Path, pathlib.PurePath, pathlib.PosixPath, pathlib.PurePosixPath, pathlib.PureWindowsPath]
MAPPING_TYPES = [typing.Mapping, typing.MutableMapping, collections.abc.Mapping, collections.abc.MutableMapping, collections.OrderedDict, typing_extensions.OrderedDict, typing.DefaultDict, collections.defaultdict]
COUNTER_TYPES = [collections.Counter, typing.Counter]
DEQUE_TYPES = [collections.deque, typing.Deque]
ValidateCallSupportedTypes = Union[LambdaType, FunctionType, MethodType, partial]
VALIDATE_CALL_SUPPORTED_TYPES = get_args(ValidateCallSupportedTypes)
_mode_to_validator = {'before': BeforeValidator, 'after': AfterValidator, 'plain': PlainValidator, 'wrap': WrapValidator}

def check_validator_fields_against_field_name(info: FieldValidatorDecoratorInfo, field: str) -> bool:
    """Check if field name is in validator fields.

    Args:
        info: The field info.
        field: The field name to check.

    Returns:
        `True` if field name is in validator fields, `False` otherwise.
    """
    fields = info.fields
    return '*' in fields or field in fields

def check_decorator_fields_exist(decorators: Iterable[Decorator[FieldDecoratorInfo]], fields: Iterable[str]) -> None:
    """Check if the defined fields in decorators exist in `fields` param.

    It ignores the check for a decorator if the decorator has `*` as field or `check_fields=False`.

    Args:
        decorators: An iterable of decorators.
        fields: An iterable of fields name.

    Raises:
        PydanticUserError: If one of the field names does not exist in `fields` param.
    """
    fields_set = set(fields)
    for dec in decorators:
        if '*' in dec.info.fields:
            continue
        if dec.info.check_fields is False:
            continue
        for field in dec.info.fields:
            if field not in fields_set:
                raise PydanticUserError(f"Decorators defined with incorrect fields: {dec.cls_ref}.{dec.cls_var_name} (use check_fields=False if you're inheriting from the model and intended this)", code='decorator-missing-field')

def filter_field_decorator_info_by_field(validator_functions: list[Decorator[FieldValidatorDecoratorInfo]], field: str) -> list[Decorator[FieldValidatorDecoratorInfo]]:
    return [dec for dec in validator_functions if check_validator_fields_against_field_name(dec.info, field)]

def apply_each_item_validators(schema: CoreSchema, each_item_validators: list[Decorator[ValidatorDecoratorInfo]], field_name: str) -> CoreSchema:
    if not each_item_validators:
        return schema
    if schema['type'] == 'nullable':
        schema['schema'] = apply_each_item_validators(schema['schema'], each_item_validators, field_name)
        return schema
    elif schema['type'] == 'tuple':
        if (variadic_item_index := schema.get('variadic_item_index')) is not None:
            schema['items_schema'][variadic_item_index] = apply_validators(schema['items_schema'][variadic_item_index], each_item_validators, field_name)
    elif is_list_like_schema_with_items_schema(schema):
        inner_schema = schema.get('items_schema', core_schema.any_schema())
        schema['items_schema'] = apply_validators(inner_schema, each_item_validators, field_name)
    elif schema['type'] == 'dict':
        inner_schema = schema.get('values_schema', core_schema.any_schema())
        schema['values_schema'] = apply_validators(inner_schema, each_item_validators, field_name)
    else:
        raise TypeError(f'`@validator(..., each_item=True)` cannot be applied to fields with a schema of {schema['type']}')
    return schema

def _extract_json_schema_info_from_field_info(info: FieldInfo) -> tuple[dict[str, Any] | None, JsonSchemaExtraCallable | None]:
    json_schema_updates = {'title': info.title, 'description': info.description, 'deprecated': bool(info.deprecated) or info.deprecated == '' or None, 'examples': to_jsonable_python(info.examples)}
    json_schema_updates = {k: v for k, v in json_schema_updates.items() if v is not None}
    return (json_schema_updates or None, info.json_schema_extra)

JsonEncoders = dict[type[Any], JsonEncoder]

def _add_custom_serialization_from_json_encoders(json_encoders: JsonEncoders | None, tp: type[Any], schema: CoreSchema) -> CoreSchema:
    """Iterate over the json_encoders and add the first matching encoder to the schema.

    Args:
        json_encoders: A dictionary of types and their encoder functions.
        tp: The type to check for a matching encoder.
        schema: The schema to add the encoder to.
    """
    if not json_encoders:
        return schema
    if 'serialization' in schema:
        return schema
    for base in (tp, *getattr(tp, '__mro__', tp.__class__.__mro__)[:-1]:
        encoder = json_encoders.get(base)
        if encoder is None:
            continue
        warnings.warn(f'`json_encoders` is deprecated. See https://docs.pydantic.dev/{version_short()}/concepts/serialization/#custom-serializers for alternatives', PydanticDeprecatedSince20)
        schema['serialization'] = core_schema.plain_serializer_function_ser_schema(encoder, when_used='json')
        return schema
    return schema

def _get_first_non_null(a: Any, b: Any) -> Any:
    """Return the first argument if it is not None, otherwise return the second argument.

    Use case: serialization_alias (argument a) and alias (argument b) are both defined, and serialization_alias is ''.
    This function will return serialization_alias, which is the first argument, even though it is an empty string.
    """
    return a if a is not None else b

class InvalidSchemaError(Exception):
    """The core schema is invalid."""

class GenerateSchema:
    """Generate core schema for a Pydantic model, dataclass and types like `str`, `datetime`, ... ."""
    __slots__ = ('_config_wrapper_stack', '_ns_resolver', '_typevars_map', 'field_name_stack', 'model_type_stack', 'defs')

    def __init__(self, config_wrapper: ConfigWrapper, ns_resolver: NsResolver | None = None, typevars_map: dict[TypeVar, Any] | None = None):
        self._config_wrapper_stack = ConfigWrapperStack(config_wrapper)
        self._ns_resolver = ns_resolver or NsResolver()
        self._typevars_map = typevars_map
        self.field_name_stack = _FieldNameStack()
        self.model_type_stack = _ModelTypeStack()
        self.defs = _Definitions()

    def __init_subclass__(cls):
        super().__init_subclass__()
        warnings.warn('Subclassing `GenerateSchema` is not supported. The API is highly subject to change in minor versions.', UserWarning, stacklevel=2)

    @property
    def _config_wrapper(self) -> ConfigWrapper:
        return self._config_wrapper_stack.tail

    @property
    def _types_namespace(self) -> NamespacesTuple:
        return self._ns_resolver.types_namespace

    @property
    def _arbitrary_types(self) -> bool:
        return self._config_wrapper.arbitrary_types_allowed

    def _list_schema(self, items_type: type[Any]) -> CoreSchema:
        return core_schema.list_schema(self.generate_schema(items_type))

    def _dict_schema(self, keys_type: type[Any], values_type: type[Any]) -> CoreSchema:
        return core_schema.dict_schema(self.generate_schema(keys_type), self.generate_schema(values_type))

    def _set_schema(self, items_type: type[Any]) -> CoreSchema:
        return core_schema.set_schema(self.generate_schema(items_type))

    def _frozenset_schema(self, items_type: type[Any]) -> CoreSchema:
        return core_schema.frozenset_schema(self.generate_schema(items_type))

    def _enum_schema(self, enum_type: type[Enum]) -> CoreSchema:
        cases = list(enum_type.__members__.values())
        enum_ref = get_type_ref(enum_type)
        description = None if not enum_type.__doc__ else inspect.cleandoc(enum_type.__doc__)
        if description == 'An enumeration.':
            description = None
        js_updates = {'title': enum_type.__name__, 'description': description}
        js_updates = {k: v for k, v in js_updates.items() if v is not None}
        sub_type = None
        if issubclass(enum_type, int):
            sub_type = 'int'
            value_ser_type = core_schema.simple_ser_schema('int')
        elif issubclass(enum_type, str):
            sub_type = 'str'
            value_ser_type = core_schema.simple_ser_schema('str')
        elif issubclass(enum_type, float):
            sub_type = 'float'
            value_ser_type = core_schema.simple_ser_schema('float')
        else:
            value_ser_type = core_schema.plain_serializer_function_ser_schema(lambda x: x)
        if cases:

            def get_json_schema(schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
                json_schema = handler(schema)
                original_schema = handler.resolve_ref_schema(json_schema)
                original_schema.update(js_updates)
                return json_schema
            default_missing = getattr(enum_type._missing_, '__func__', None) is Enum._missing_.__func__
            enum_schema = core_schema.enum_schema(enum_type, cases, sub_type=sub_type, missing=None if default_missing else enum_type._missing_, ref=enum_ref, metadata={'pydantic_js_functions': [get_json_schema]})
            if self._config_wrapper.use_enum_values:
                enum_schema = core_schema.no_info_after_validator_function(attrgetter('value'), enum_schema, serialization=value_ser_type)
            return enum_schema
        else:

            def get_json_schema_no_cases(_: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
                json_schema = handler(core_schema.enum_schema(enum_type, cases, sub_type=sub_type, ref=enum_ref))
                original_schema = handler.resolve_ref_schema(json_schema)
                original_schema.update(js_updates)
                return json_schema
            return core_schema.is_instance_schema(enum_type, metadata={'pydantic_js_functions': [get_json_schema_no_cases]})

    def _ip_schema(self, tp: type[Any]) -> CoreSchema:
        from ._validators import IP_VALIDATOR_LOOKUP, IpType
        ip_type_json_schema_format = {IPv4Address: 'ipv4', IPv4Network: 'ipv4network', IPv4Interface: 'ipv4interface', IPv6Address: 'ipv6', IPv6Network: 'ipv6network', IPv6Interface: 'ipv6interface'}

        def ser_ip(ip: Any, info: Any) -> Any:
            if not isinstance(ip, (tp, str)):
                raise PydanticSerializationUnexpectedValue(f"Expected `{tp}` but got `{type(ip)}` with value `'{ip}'` - serialized value may not be as expected.")
            if info.mode == 'python':
                return ip
            return str(ip)
        return core_schema.lax_or_strict_schema(lax_schema=core_schema.no_info_plain_validator_function(IP_VALIDATOR_LOOKUP[tp]), strict_schema=core_schema.json_or_python_schema(json_schema=core_schema.no_info_after_validator_function(tp, core_schema.str_schema()), python_schema=core_schema.is_instance_schema(tp)), serialization=core_schema.plain_serializer_function_ser_schema(ser_ip, info_arg=True, when_used='always'), metadata={'pydantic_js_functions': [lambda _1, _2: {'type': 'string', 'format': ip_type_json_schema_format[tp]}]})

    def _path_schema(self, tp: type[Any], path_type: type[Any]) -> CoreSchema:
        if tp is os.PathLike and (path_type not in {str, bytes} and (not _typing_extra.is_any(path_type))):
            raise PydanticUserError('`os.PathLike` can only be used with `str`, `bytes` or `Any`', code='schema-for-unknown-type')
        path_constructor = pathlib.PurePath if tp is os.PathLike else tp
        constrained_schema = core_schema