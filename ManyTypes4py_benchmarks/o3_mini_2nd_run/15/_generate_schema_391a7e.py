#!/usr/bin/env python3
"""Convert python types to pydantic-core schema."""
from __future__ import annotations

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
from typing import TYPE_CHECKING, Any, Callable, Dict, Final, ForwardRef, Iterable as TypingIterable, List, Literal, Optional, Tuple, Type, TypeVar, Union, cast, overload
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

_SUPPORTS_TYPEDDICT: bool = sys.version_info >= (3, 12)
FieldDecoratorInfo = Union[ValidatorDecoratorInfo, FieldValidatorDecoratorInfo, FieldSerializerDecoratorInfo]
FieldDecoratorInfoType = TypeVar('FieldDecoratorInfoType', bound=FieldDecoratorInfo)
AnyFieldDecorator = Union[Decorator[ValidatorDecoratorInfo], Decorator[FieldValidatorDecoratorInfo], Decorator[FieldSerializerDecoratorInfo]]
ModifyCoreSchemaWrapHandler = GetCoreSchemaHandler
GetCoreSchemaFunction = Callable[[Any, ModifyCoreSchemaWrapHandler], CoreSchema]
TUPLE_TYPES: List[Any] = [typing.Tuple, tuple]
LIST_TYPES: List[Any] = [typing.List, list, collections.abc.MutableSequence]
SET_TYPES: List[Any] = [typing.Set, set, collections.abc.MutableSet]
FROZEN_SET_TYPES: List[Any] = [typing.FrozenSet, frozenset, collections.abc.Set]
DICT_TYPES: List[Any] = [typing.Dict, dict]
IP_TYPES: List[Any] = [IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network]
SEQUENCE_TYPES: List[Any] = [typing.Sequence, collections.abc.Sequence]
ITERABLE_TYPES: List[Any] = [typing.Iterable, collections.abc.Iterable, typing.Generator, collections.abc.Generator]
TYPE_TYPES: List[Any] = [typing.Type, type]
PATTERN_TYPES: List[Any] = [typing.Pattern, re.Pattern]
PATH_TYPES: List[Any] = [
    os.PathLike,
    pathlib.Path,
    pathlib.PurePath,
    pathlib.PosixPath,
    pathlib.PurePosixPath,
    pathlib.PureWindowsPath,
]
MAPPING_TYPES: List[Any] = [
    typing.Mapping,
    typing.MutableMapping,
    collections.abc.Mapping,
    collections.abc.MutableMapping,
    collections.OrderedDict,
    typing_extensions.OrderedDict,
    typing.DefaultDict,
    collections.defaultdict,
]
COUNTER_TYPES: List[Any] = [collections.Counter, typing.Counter]
DEQUE_TYPES: List[Any] = [collections.deque, typing.Deque]
ValidateCallSupportedTypes = Union[LambdaType, FunctionType, MethodType, partial]
VALIDATE_CALL_SUPPORTED_TYPES: Tuple[Any, ...] = get_args(ValidateCallSupportedTypes)
_mode_to_validator: Dict[str, Any] = {'before': BeforeValidator, 'after': AfterValidator, 'plain': PlainValidator, 'wrap': WrapValidator}


def check_validator_fields_against_field(info: Any, field: str) -> bool:
    """Check if field name is in validator fields.

    Args:
        info: The field info.
        field: The field name to check.

    Returns:
        True if field name is in validator fields, False otherwise.
    """
    fields = info.fields
    return '*' in fields or field in fields


def check_decorator_fields_exist(decorators: TypingIterable[Any], fields: TypingIterable[str]) -> None:
    """Check if the defined fields in decorators exist in `fields` param.

    It ignores the check for a decorator if the decorator has `*` as field or `check_fields=False`.

    Args:
        decorators: An iterable of decorators.
        fields: An iterable of field names.

    Raises:
        PydanticUserError: If one of the field names does not exist in `fields` param.
    """
    fields_set: set[str] = set(fields)
    for dec in decorators:
        if '*' in dec.info.fields:
            continue
        if dec.info.check_fields is False:
            continue
        for field in dec.info.fields:
            if field not in fields_set:
                raise PydanticUserError(
                    f"Decorators defined with incorrect fields: {dec.cls_ref}.{dec.cls_var_name} (use check_fields=False if you're inheriting from the model and intended this)",
                    code='decorator-missing-field',
                )


def filter_field_decorator_info_by_field(validator_functions: List[Any], field: str) -> List[Any]:
    return [dec for dec in validator_functions if check_validator_fields_against_field(dec.info, field)]


def apply_each_item_validators(schema: Dict[str, Any], each_item_validators: List[Any], field_name: str) -> Dict[str, Any]:
    if not each_item_validators:
        return schema
    if schema['type'] == 'nullable':
        schema['schema'] = apply_each_item_validators(schema['schema'], each_item_validators, field_name)
        return schema
    elif schema['type'] == 'tuple':
        variadic_item_index: Optional[int] = schema.get('variadic_item_index')
        if variadic_item_index is not None:
            schema['items_schema'][variadic_item_index] = apply_validators(
                schema['items_schema'][variadic_item_index], each_item_validators, field_name
            )
    elif is_list_like_schema_with_items_schema(schema):
        inner_schema: Dict[str, Any] = schema.get('items_schema', core_schema.any_schema())
        schema['items_schema'] = apply_validators(inner_schema, each_item_validators, field_name)
    elif schema['type'] == 'dict':
        inner_schema = schema.get('values_schema', core_schema.any_schema())
        schema['values_schema'] = apply_validators(inner_schema, each_item_validators, field_name)
    else:
        raise TypeError(f'`@validator(..., each_item=True)` cannot be applied to fields with a schema of {schema["type"]}')
    return schema


def _extract_json_schema_info_from_field_info(info: Any) -> Tuple[Optional[Dict[str, Any]], Any]:
    json_schema_updates: Dict[str, Any] = {
        'title': info.title,
        'description': info.description,
        'deprecated': bool(info.deprecated) or info.deprecated == '' or None,
        'examples': to_jsonable_python(info.examples),
    }
    json_schema_updates = {k: v for k, v in json_schema_updates.items() if v is not None}
    return (json_schema_updates or None, info.json_schema_extra)


JsonEncoders: Dict[Type[Any], JsonEncoder] = {}


def _add_custom_serialization_from_json_encoders(
    json_encoders: Optional[Dict[Type[Any], JsonEncoder]],
    tp: Any,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
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
    for base in (tp, *getattr(tp, '__mro__', tp.__class__.__mro__)[:-1]):
        encoder = json_encoders.get(base)
        if encoder is None:
            continue
        warnings.warn(
            f'`json_encoders` is deprecated. See https://docs.pydantic.dev/{version_short()}/concepts/serialization/#custom-serializers for alternatives',
            PydanticDeprecatedSince20,
        )
        schema['serialization'] = core_schema.plain_serializer_function_ser_schema(encoder, when_used='json')
        return schema
    return schema


def _get_first_non_null(a: Any, b: Any) -> Any:
    """Return the first argument if it is not None, otherwise return the second argument."""
    return a if a is not None else b


class InvalidSchemaError(Exception):
    """The core schema is invalid."""


class GenerateSchema:
    """Generate core schema for a Pydantic model, dataclass and types like `str`, `datetime`, ... ."""

    __slots__ = ('_config_wrapper_stack', '_ns_resolver', '_typevars_map', 'field_name_stack', 'model_type_stack', 'defs')

    def __init__(self, config_wrapper: ConfigDict, ns_resolver: Optional[NsResolver] = None, typevars_map: Optional[Any] = None) -> None:
        self._config_wrapper_stack: ConfigWrapperStack = ConfigWrapperStack(config_wrapper)
        self._ns_resolver: NsResolver = ns_resolver or NsResolver()
        self._typevars_map: Optional[Any] = typevars_map
        self.field_name_stack: _FieldNameStack = _FieldNameStack()
        self.model_type_stack: _ModelTypeStack = _ModelTypeStack()
        self.defs: _Definitions = _Definitions()

    def __init_subclass__(cls: Type[Any]) -> None:
        super().__init_subclass__()
        warnings.warn(
            'Subclassing `GenerateSchema` is not supported. The API is highly subject to change in minor versions.',
            UserWarning,
            stacklevel=2,
        )

    @property
    def _config_wrapper(self) -> ConfigWrapper:
        return self._config_wrapper_stack.tail

    @property
    def _types_namespace(self) -> Any:
        return self._ns_resolver.types_namespace

    @property
    def _arbitrary_types(self) -> bool:
        return self._config_wrapper.arbitrary_types_allowed

    def _list_schema(self, items_type: Any) -> Dict[str, Any]:
        return core_schema.list_schema(self.generate_schema(items_type))

    def _dict_schema(self, keys_type: Any, values_type: Any) -> Dict[str, Any]:
        return core_schema.dict_schema(self.generate_schema(keys_type), self.generate_schema(values_type))

    def _set_schema(self, items_type: Any) -> Dict[str, Any]:
        return core_schema.set_schema(self.generate_schema(items_type))

    def _frozenset_schema(self, items_type: Any) -> Dict[str, Any]:
        return core_schema.frozenset_schema(self.generate_schema(items_type))

    def _enum_schema(self, enum_type: Type[Enum]) -> Dict[str, Any]:
        cases: List[Any] = list(enum_type.__members__.values())
        enum_ref: Any = get_type_ref(enum_type)
        description: Optional[str] = None if not enum_type.__doc__ else inspect.cleandoc(enum_type.__doc__)
        if description == 'An enumeration.':
            description = None
        js_updates: Dict[str, Any] = {'title': enum_type.__name__, 'description': description}
        js_updates = {k: v for k, v in js_updates.items() if v is not None}
        sub_type: Optional[str] = None
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
            def get_json_schema(schema: Dict[str, Any], handler: Any) -> Dict[str, Any]:
                json_schema: Dict[str, Any] = handler(schema)
                original_schema: Dict[str, Any] = handler.resolve_ref_schema(json_schema)
                original_schema.update(js_updates)
                return json_schema
            default_missing: bool = getattr(enum_type._missing_, '__func__', None) is Enum._missing_.__func__
            enum_schema: Dict[str, Any] = core_schema.enum_schema(
                enum_type, cases, sub_type=sub_type, missing=None if default_missing else enum_type._missing_, ref=enum_ref, metadata={'pydantic_js_functions': [get_json_schema]},
            )
            if self._config_wrapper.use_enum_values:
                enum_schema = core_schema.no_info_after_validator_function(attrgetter('value'), enum_schema, serialization=value_ser_type)
            return enum_schema
        else:
            def get_json_schema_no_cases(_: Any, handler: Any) -> Dict[str, Any]:
                json_schema = handler(core_schema.enum_schema(enum_type, cases, sub_type=sub_type, ref=enum_ref))
                original_schema = handler.resolve_ref_schema(json_schema)
                original_schema.update(js_updates)
                return json_schema
            return core_schema.is_instance_schema(enum_type, metadata={'pydantic_js_functions': [get_json_schema_no_cases]})

    def _ip_schema(self, tp: Any) -> Dict[str, Any]:
        from ._validators import IP_VALIDATOR_LOOKUP, IpType
        ip_type_json_schema_format: Dict[Any, str] = {
            IPv4Address: 'ipv4',
            IPv4Network: 'ipv4network',
            IPv4Interface: 'ipv4interface',
            IPv6Address: 'ipv6',
            IPv6Network: 'ipv6network',
            IPv6Interface: 'ipv6interface',
        }

        def ser_ip(ip: Any, info: Any) -> Any:
            if not isinstance(ip, (tp, str)):
                raise PydanticSerializationUnexpectedValue(
                    f"Expected `{tp}` but got `{type(ip)}` with value `'{ip}'` - serialized value may not be as expected."
                )
            if info.mode == 'python':
                return ip
            return str(ip)
        return core_schema.lax_or_strict_schema(
            lax_schema=core_schema.no_info_plain_validator_function(IP_VALIDATOR_LOOKUP[tp]),
            strict_schema=core_schema.json_or_python_schema(
                json_schema=core_schema.no_info_after_validator_function(tp, core_schema.str_schema()),
                python_schema=core_schema.is_instance_schema(tp),
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(ser_ip, info_arg=True, when_used='always'),
            metadata={'pydantic_js_functions': [lambda _1, _2: {'type': 'string', 'format': ip_type_json_schema_format[tp]}]},
        )

    def _path_schema(self, tp: Any, path_type: Any) -> Dict[str, Any]:
        if tp is os.PathLike and (path_type not in {str, bytes} and (not _typing_extra.is_any(path_type))):
            raise PydanticUserError('`os.PathLike` can only be used with `str`, `bytes` or `Any`', code='schema-for-unknown-type')
        path_constructor: Any = pathlib.PurePath if tp is os.PathLike else tp
        constrained_schema: Dict[str, Any] = core_schema.bytes_schema() if path_type is bytes else core_schema.str_schema()

        def path_validator(input_value: Any) -> Any:
            try:
                if path_type is bytes:
                    if isinstance(input_value, bytes):
                        try:
                            input_value = input_value.decode()
                        except UnicodeDecodeError as e:
                            raise PydanticCustomError('bytes_type', 'Input must be valid bytes') from e
                    else:
                        raise PydanticCustomError('bytes_type', 'Input must be bytes')
                elif not isinstance(input_value, str):
                    raise PydanticCustomError('path_type', 'Input is not a valid path')
                return path_constructor(input_value)
            except TypeError as e:
                raise PydanticCustomError('path_type', 'Input is not a valid path') from e

        def ser_path(path: Any, info: Any) -> Any:
            if not isinstance(path, (tp, str)):
                raise PydanticSerializationUnexpectedValue(
                    f"Expected `{tp}` but got `{type(path)}` with value `'{path}'` - serialized value may not be as expected."
                )
            if info.mode == 'python':
                return path
            return str(path)
        instance_schema: Dict[str, Any] = core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(path_validator, constrained_schema),
            python_schema=core_schema.is_instance_schema(tp),
        )
        schema: Dict[str, Any] = core_schema.lax_or_strict_schema(
            lax_schema=core_schema.union_schema(
                [instance_schema, core_schema.no_info_after_validator_function(path_validator, constrained_schema)],
                custom_error_type='path_type',
                custom_error_message=f'Input is not a valid path for {tp}',
                strict=True,
            ),
            strict_schema=instance_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(ser_path, info_arg=True, when_used='always'),
            metadata={'pydantic_js_functions': [lambda source, handler: {**handler(source), 'format': 'path'}]},
        )
        return schema

    def _deque_schema(self, items_type: Any) -> Dict[str, Any]:
        from ._serializers import serialize_sequence_via_list
        from ._validators import deque_validator
        item_type_schema: Dict[str, Any] = self.generate_schema(items_type)
        list_schema: Dict[str, Any] = core_schema.list_schema(item_type_schema, strict=False)
        check_instance: Dict[str, Any] = core_schema.json_or_python_schema(
            json_schema=list_schema,
            python_schema=core_schema.is_instance_schema(collections.deque, cls_repr='Deque'),
        )
        lax_schema: Dict[str, Any] = core_schema.no_info_wrap_validator_function(deque_validator, list_schema)
        return core_schema.lax_or_strict_schema(
            lax_schema=lax_schema,
            strict_schema=core_schema.chain_schema([check_instance, lax_schema]),
            serialization=core_schema.wrap_serializer_function_ser_schema(serialize_sequence_via_list, schema=item_type_schema, info_arg=True),
        )

    def _mapping_schema(self, tp: Any, keys_type: Any, values_type: Any) -> Dict[str, Any]:
        from ._validators import MAPPING_ORIGIN_MAP, defaultdict_validator, get_defaultdict_default_default_factory
        keys_schema: Dict[str, Any] = self.generate_schema(keys_type)
        values_schema: Dict[str, Any] = self.generate_schema(values_type)
        dict_schema: Dict[str, Any] = core_schema.dict_schema(keys_schema, values_schema)
        check_instance: Dict[str, Any] = core_schema.json_or_python_schema(
            json_schema=dict_schema,
            python_schema=core_schema.is_instance_schema(tp),
        )
        if tp is collections.defaultdict:
            default_default_factory: Any = get_defaultdict_default_default_factory(values_type)
            coerce_instance_wrap: Callable[..., Dict[str, Any]] = partial(core_schema.no_info_wrap_validator_function, partial(defaultdict_validator, default_default_factory=default_default_factory))
        else:
            coerce_instance_wrap = partial(core_schema.no_info_after_validator_function, MAPPING_ORIGIN_MAP[tp])
        lax_schema: Dict[str, Any] = coerce_instance_wrap(dict_schema)
        schema: Dict[str, Any] = core_schema.lax_or_strict_schema(
            lax_schema=lax_schema,
            strict_schema=core_schema.union_schema([check_instance, lax_schema]),
            serialization=core_schema.wrap_serializer_function_ser_schema(lambda v, h: h(v), schema=dict_schema, info_arg=False),
        )
        return schema

    def _fraction_schema(self) -> Dict[str, Any]:
        """Support for [`fractions.Fraction`][fractions.Fraction]."""
        from ._validators import fraction_validator
        return core_schema.lax_or_strict_schema(
            lax_schema=core_schema.no_info_plain_validator_function(fraction_validator),
            strict_schema=core_schema.json_or_python_schema(
                json_schema=core_schema.no_info_plain_validator_function(fraction_validator),
                python_schema=core_schema.is_instance_schema(Fraction),
            ),
            serialization=core_schema.to_string_ser_schema(when_used='always'),
            metadata={'pydantic_js_functions': [lambda _1, _2: {'type': 'string', 'format': 'fraction'}]},
        )

    def _arbitrary_type_schema(self, tp: Any) -> Dict[str, Any]:
        if not isinstance(tp, type):
            warn(
                f'{tp!r} is not a Python type (it may be an instance of an object), Pydantic will allow any object with no validation since we cannot even enforce that the input is an instance of the given type. To get rid of this error wrap the type with `pydantic.SkipValidation`.',
                UserWarning,
            )
            return core_schema.any_schema()
        return core_schema.is_instance_schema(tp)

    def _unknown_type_schema(self, obj: Any) -> Dict[str, Any]:
        raise PydanticSchemaGenerationError(
            f'Unable to generate pydantic-core schema for {obj!r}. Set `arbitrary_types_allowed=True` in the model_config to ignore this error or implement `__get_pydantic_core_schema__` on your type to fully support it.\n\nIf you got this error by calling handler(<some type>) within `__get_pydantic_core_schema__` then you likely need to call `handler.generate_schema(<some type>)` since we do not call `__get_pydantic_core_schema__` on `<some type>` otherwise to avoid infinite recursion.'
        )

    def _apply_discriminator_to_union(self, schema: Dict[str, Any], discriminator: Optional[Any]) -> Dict[str, Any]:
        if discriminator is None:
            return schema
        try:
            return _discriminated_union.apply_discriminator(schema, discriminator)
        except _discriminated_union.MissingDefinitionForUnionRef:
            _discriminated_union.set_discriminator_in_metadata(schema, discriminator)
            return schema

    def clean_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        schema = self.defs.finalize_schema(schema)
        schema = validate_core_schema(schema)
        return schema

    def _add_js_function(self, metadata_schema: Dict[str, Any], js_function: Callable[..., Any]) -> None:
        metadata: Dict[str, Any] = metadata_schema.get('metadata', {})
        pydantic_js_functions: List[Any] = metadata.setdefault('pydantic_js_functions', [])
        if js_function not in pydantic_js_functions:
            pydantic_js_functions.append(js_function)
        metadata_schema['metadata'] = metadata

    def generate_schema(self, obj: Any) -> Dict[str, Any]:
        """Generate core schema.

        Args:
            obj: The object to generate core schema for.

        Returns:
            The generated core schema.

        Raises:
            PydanticUndefinedAnnotation:
                If it is not possible to evaluate forward reference.
            PydanticSchemaGenerationError:
                If it is not possible to generate pydantic-core schema.
            TypeError:
                - If `alias_generator` returns a disallowed type (must be str, AliasPath or AliasChoices).
                - If V1 style validator with `each_item=True` applied on a wrong field.
            PydanticUserError:
                - If `typing.TypedDict` is used instead of `typing_extensions.TypedDict` on Python < 3.12.
                - If `__modify_schema__` method is used instead of `__get_pydantic_json_schema__`.
        """
        schema: Optional[Dict[str, Any]] = self._generate_schema_from_get_schema_method(obj, obj)
        if schema is None:
            schema = self._generate_schema_inner(obj)
        metadata_js_function: Optional[Callable[..., Any]] = _extract_get_pydantic_json_schema(obj)
        if metadata_js_function is not None:
            metadata_schema: Optional[Dict[str, Any]] = resolve_original_schema(schema, self.defs)
            if metadata_schema:
                self._add_js_function(metadata_schema, metadata_js_function)
        schema = _add_custom_serialization_from_json_encoders(self._config_wrapper.json_encoders, obj, schema)
        return schema

    def _model_schema(self, cls: Type[Any]) -> Dict[str, Any]:
        """Generate schema for a Pydantic model."""
        BaseModel_ = import_cached_base_model()
        with self.defs.get_schema_or_ref(cls) as (model_ref, maybe_schema):
            if maybe_schema is not None:
                return maybe_schema
            schema: Optional[Dict[str, Any]] = cls.__dict__.get('__pydantic_core_schema__')
            if schema is not None and (not isinstance(schema, MockCoreSchema)):
                if schema['type'] == 'definitions':
                    schema = self.defs.unpack_definitions(schema)
                ref = get_ref(schema)
                if ref:
                    return self.defs.create_definition_reference_schema(schema)
                else:
                    return schema
            config_wrapper = ConfigWrapper(cls.model_config, check=False)
            with self._config_wrapper_stack.push(config_wrapper), self._ns_resolver.push(cls):
                core_config: Dict[str, Any] = self._config_wrapper.core_config(title=cls.__name__)
                if cls.__pydantic_fields_complete__ or cls is BaseModel_:
                    fields: Dict[str, Any] = getattr(cls, '__pydantic_fields__', {})
                else:
                    try:
                        fields = rebuild_model_fields(cls, ns_resolver=self._ns_resolver, typevars_map=self._typevars_map or {})
                    except NameError as e:
                        raise PydanticUndefinedAnnotation.from_name_error(e) from e
                decorators = cls.__pydantic_decorators__
                computed_fields = decorators.computed_fields
                check_decorator_fields_exist(
                    chain(decorators.field_validators.values(), decorators.field_serializers.values(), decorators.validators.values()),
                    {*fields.keys(), *computed_fields.keys()},
                )
                model_validators = decorators.model_validators.values()
                extras_schema: Optional[Dict[str, Any]] = None
                if core_config.get('extra_fields_behavior') == 'allow':
                    assert cls.__mro__[0] is cls
                    assert cls.__mro__[-1] is object
                    for candidate_cls in cls.__mro__[:-1]:
                        extras_annotation = getattr(candidate_cls, '__annotations__', {}).get('__pydantic_extra__', None)
                        if extras_annotation is not None:
                            if isinstance(extras_annotation, str):
                                extras_annotation = _typing_extra.eval_type_backport(_typing_extra._make_forward_ref(extras_annotation, is_argument=False, is_class=True), *self._types_namespace)
                            tp = get_origin(extras_annotation)
                            if tp not in DICT_TYPES:
                                raise PydanticSchemaGenerationError('The type annotation for `__pydantic_extra__` must be `dict[str, ...]`')
                            extra_items_type = self._get_args_resolving_forward_refs(extras_annotation, required=True)[1]
                            if not _typing_extra.is_any(extra_items_type):
                                extras_schema = self.generate_schema(extra_items_type)
                                break
                generic_origin = getattr(cls, '__pydantic_generic_metadata__', {}).get('origin')
                if cls.__pydantic_root_model__:
                    root_field = self._common_field_schema('root', fields['root'], decorators)
                    inner_schema = root_field['schema']
                    inner_schema = apply_model_validators(inner_schema, model_validators, 'inner')
                    model_schema: Dict[str, Any] = core_schema.model_schema(
                        cls,
                        inner_schema,
                        generic_origin=generic_origin,
                        custom_init=getattr(cls, '__pydantic_custom_init__', None),
                        root_model=True,
                        post_init=getattr(cls, '__pydantic_post_init__', None),
                        config=core_config,
                        ref=model_ref,
                    )
                else:
                    fields_schema: Dict[str, Any] = core_schema.model_fields_schema(
                        {k: self._generate_md_field_schema(k, v, decorators) for k, v in fields.items()},
                        computed_fields=[self._computed_field_schema(d, decorators.field_serializers) for d in computed_fields.values()],
                        extras_schema=extras_schema,
                        model_name=cls.__name__,
                    )
                    inner_schema = apply_validators(fields_schema, decorators.root_validators.values(), None)
                    inner_schema = apply_model_validators(inner_schema, model_validators, 'inner')
                    model_schema = core_schema.model_schema(
                        cls,
                        inner_schema,
                        generic_origin=generic_origin,
                        custom_init=getattr(cls, '__pydantic_custom_init__', None),
                        root_model=False,
                        post_init=getattr(cls, '__pydantic_post_init__', None),
                        config=core_config,
                        ref=model_ref,
                    )
                schema = self._apply_model_serializers(model_schema, decorators.model_serializers.values())
                schema = apply_model_validators(schema, model_validators, 'outer')
                return self.defs.create_definition_reference_schema(schema)

    def _resolve_self_type(self, obj: Any) -> Any:
        obj = self.model_type_stack.get()
        if obj is None:
            raise PydanticUserError('`typing.Self` is invalid in this context', code='invalid-self-type')
        return obj

    def _generate_schema_from_get_schema_method(self, obj: Any, source: Any) -> Optional[Dict[str, Any]]:
        BaseModel_ = import_cached_base_model()
        get_schema: Optional[Callable[..., Any]] = getattr(obj, '__get_pydantic_core_schema__', None)
        is_base_model_get_schema: bool = getattr(get_schema, '__func__', None) is BaseModel_.__get_pydantic_core_schema__.__func__
        if get_schema is not None and (not is_base_model_get_schema):
            with self.defs.get_schema_or_ref(obj) as (_, maybe_schema):
                if maybe_schema is not None:
                    return maybe_schema
            if obj is source:
                ref_mode: str = 'unpack'
            else:
                ref_mode = 'to-def'
            schema: Dict[str, Any] = get_schema(source, CallbackGetCoreSchemaHandler(self._generate_schema_inner, self, ref_mode=ref_mode))
            if schema['type'] == 'definitions':
                schema = self.defs.unpack_definitions(schema)
            ref = get_ref(schema)
            if ref:
                return self.defs.create_definition_reference_schema(schema)
            return schema
        if (validators := getattr(obj, '__get_validators__', None)) is not None:
            from pydantic.v1 import BaseModel as BaseModelV1
            if issubclass(obj, BaseModelV1):
                warn(f'Mixing V1 models and V2 models (or constructs, like `TypeAdapter`) is not supported. Please upgrade `{obj.__name__}` to V2.', UserWarning)
            else:
                warn('`__get_validators__` is deprecated and will be removed, use `__get_pydantic_core_schema__` instead.', PydanticDeprecatedSince20)
            return core_schema.chain_schema([core_schema.with_info_plain_validator_function(v) for v in validators()])
        return None

    def _resolve_forward_ref(self, obj: Any) -> Any:
        try:
            obj = _typing_extra.eval_type_backport(obj, *self._types_namespace)
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        if isinstance(obj, ForwardRef):
            raise PydanticUndefinedAnnotation(obj.__forward_arg__, f'Unable to evaluate forward reference {obj}')
        if self._typevars_map:
            obj = replace_types(obj, self._typevars_map)
        return obj

    @overload
    def _get_args_resolving_forward_refs(self, obj: Any, required: bool) -> Tuple[Any, ...]:
        ...

    @overload
    def _get_args_resolving_forward_refs(self, obj: Any) -> Tuple[Any, ...]:
        ...

    def _get_args_resolving_forward_refs(self, obj: Any, required: bool = False) -> Tuple[Any, ...]:
        args = get_args(obj)
        if args:
            if isinstance(obj, GenericAlias):
                args = (_typing_extra._make_forward_ref(a) if isinstance(a, str) else a for a in args)
            args = tuple((self._resolve_forward_ref(a) if isinstance(a, ForwardRef) else a for a in args))
        elif required:
            raise TypeError(f'Expected {obj} to have generic parameters but it had none')
        return args

    def _get_first_arg_or_any(self, obj: Any) -> Any:
        args = self._get_args_resolving_forward_refs(obj)
        if not args:
            return Any
        return args[0]

    def _get_first_two_args_or_any(self, obj: Any) -> Tuple[Any, Any]:
        args = self._get_args_resolving_forward_refs(obj)
        if not args:
            return (Any, Any)
        if len(args) < 2:
            origin = get_origin(obj)
            raise TypeError(f'Expected two type arguments for {origin}, got 1')
        return (args[0], args[1])

    def _generate_schema_inner(self, obj: Any) -> Dict[str, Any]:
        if _typing_extra.is_self(obj):
            obj = self._resolve_self_type(obj)
        if _typing_extra.is_annotated(obj):
            return self._annotated_schema(obj)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            obj = ForwardRef(obj)
        if isinstance(obj, ForwardRef):
            return self.generate_schema(self._resolve_forward_ref(obj))
        BaseModel = import_cached_base_model()
        if lenient_issubclass(obj, BaseModel):
            with self.model_type_stack.push(obj):
                return self._model_schema(obj)
        if isinstance(obj, PydanticRecursiveRef):
            return core_schema.definition_reference_schema(schema_ref=obj.type_ref)
        return self.match_type(obj)

    def match_type(self, obj: Any) -> Dict[str, Any]:
        """Main mapping of types to schemas."""
        if obj is str:
            return core_schema.str_schema()
        elif obj is bytes:
            return core_schema.bytes_schema()
        elif obj is int:
            return core_schema.int_schema()
        elif obj is float:
            return core_schema.float_schema()
        elif obj is bool:
            return core_schema.bool_schema()
        elif obj is complex:
            return core_schema.complex_schema()
        elif _typing_extra.is_any(obj) or obj is object:
            return core_schema.any_schema()
        elif obj is datetime.date:
            return core_schema.date_schema()
        elif obj is datetime.datetime:
            return core_schema.datetime_schema()
        elif obj is datetime.time:
            return core_schema.time_schema()
        elif obj is datetime.timedelta:
            return core_schema.timedelta_schema()
        elif obj is Decimal:
            return core_schema.decimal_schema()
        elif obj is UUID:
            return core_schema.uuid_schema()
        elif obj is Url:
            return core_schema.url_schema()
        elif obj is Fraction:
            return self._fraction_schema()
        elif obj is MultiHostUrl:
            return core_schema.multi_host_url_schema()
        elif obj is None or obj is _typing_extra.NoneType:
            return core_schema.none_schema()
        elif obj in IP_TYPES:
            return self._ip_schema(obj)
        elif obj in TUPLE_TYPES:
            return self._tuple_schema(obj)
        elif obj in LIST_TYPES:
            return self._list_schema(Any)
        elif obj in SET_TYPES:
            return self._set_schema(Any)
        elif obj in FROZEN_SET_TYPES:
            return self._frozenset_schema(Any)
        elif obj in SEQUENCE_TYPES:
            return self._sequence_schema(Any)
        elif obj in ITERABLE_TYPES:
            return self._iterable_schema(obj)
        elif obj in DICT_TYPES:
            return self._dict_schema(Any, Any)
        elif obj in PATH_TYPES:
            return self._path_schema(obj, Any)
        elif obj in DEQUE_TYPES:
            return self._deque_schema(Any)
        elif obj in MAPPING_TYPES:
            return self._mapping_schema(obj, Any, Any)
        elif obj in COUNTER_TYPES:
            return self._mapping_schema(obj, Any, int)
        elif _typing_extra.is_type_alias_type(obj):
            return self._type_alias_type_schema(obj)
        elif obj is type:
            return self._type_schema()
        elif _typing_extra.is_callable(obj):
            return core_schema.callable_schema()
        elif _typing_extra.is_literal(obj):
            return self._literal_schema(obj)
        elif is_typeddict(obj):
            return self._typed_dict_schema(obj, None)
        elif _typing_extra.is_namedtuple(obj):
            return self._namedtuple_schema(obj, None)
        elif _typing_extra.is_new_type(obj):
            return self.generate_schema(obj.__supertype__)
        elif obj in PATTERN_TYPES:
            return self._pattern_schema(obj)
        elif _typing_extra.is_hashable(obj):
            return self._hashable_schema()
        elif isinstance(obj, typing.TypeVar):
            return self._unsubstituted_typevar_schema(obj)
        elif _typing_extra.is_finalvar(obj):
            if obj is Final:
                return core_schema.any_schema()
            return self.generate_schema(self._get_first_arg_or_any(obj))
        elif isinstance(obj, VALIDATE_CALL_SUPPORTED_TYPES):
            return self._call_schema(obj)
        elif inspect.isclass(obj) and issubclass(obj, Enum):
            return self._enum_schema(obj)
        elif _typing_extra.is_zoneinfo_type(obj):
            return self._zoneinfo_schema()
        if dataclasses.is_dataclass(obj):
            return self._dataclass_schema(obj, None)
        origin = get_origin(obj)
        if origin is not None:
            return self._match_generic_type(obj, origin)
        if self._arbitrary_types:
            return self._arbitrary_type_schema(obj)
        return self._unknown_type_schema(obj)

    def _match_generic_type(self, obj: Any, origin: Any) -> Dict[str, Any]:
        if dataclasses.is_dataclass(origin):
            return self._dataclass_schema(obj, origin)
        if _typing_extra.is_namedtuple(origin):
            return self._namedtuple_schema(obj, origin)
        schema: Optional[Dict[str, Any]] = self._generate_schema_from_get_schema_method(origin, obj)
        if schema is not None:
            return schema
        if _typing_extra.is_type_alias_type(origin):
            return self._type_alias_type_schema(obj)
        elif _typing_extra.origin_is_union(origin):
            return self._union_schema(obj)
        elif origin in TUPLE_TYPES:
            return self._tuple_schema(obj)
        elif origin in LIST_TYPES:
            return self._list_schema(self._get_first_arg_or_any(obj))
        elif origin in SET_TYPES:
            return self._set_schema(self._get_first_arg_or_any(obj))
        elif origin in FROZEN_SET_TYPES:
            return self._frozenset_schema(self._get_first_arg_or_any(obj))
        elif origin in DICT_TYPES:
            return self._dict_schema(*self._get_first_two_args_or_any(obj))
        elif origin in PATH_TYPES:
            return self._path_schema(origin, self._get_first_arg_or_any(obj))
        elif origin in DEQUE_TYPES:
            return self._deque_schema(self._get_first_arg_or_any(obj))
        elif origin in MAPPING_TYPES:
            return self._mapping_schema(origin, *self._get_first_two_args_or_any(obj))
        elif origin in COUNTER_TYPES:
            return self._mapping_schema(origin, self._get_first_arg_or_any(obj), int)
        elif is_typeddict(origin):
            return self._typed_dict_schema(obj, origin)
        elif origin in TYPE_TYPES:
            return self._subclass_schema(obj)
        elif origin in SEQUENCE_TYPES:
            return self._sequence_schema(self._get_first_arg_or_any(obj))
        elif origin in ITERABLE_TYPES:
            return self._iterable_schema(obj)
        elif origin in PATTERN_TYPES:
            return self._pattern_schema(obj)
        if self._arbitrary_types:
            return self._arbitrary_type_schema(origin)
        return self._unknown_type_schema(obj)

    def _generate_td_field_schema(self, name: str, field_info: Any, decorators: Any, *, required: bool = True) -> Dict[str, Any]:
        """Prepare a TypedDictField to represent a model or typeddict field."""
        common_field: Dict[str, Any] = self._common_field_schema(name, field_info, decorators)
        return core_schema.typed_dict_field(
            common_field['schema'],
            required=False if not field_info.is_required() else required,
            serialization_exclude=common_field['serialization_exclude'],
            validation_alias=common_field['validation_alias'],
            serialization_alias=common_field['serialization_alias'],
            metadata=common_field['metadata'],
        )

    def _generate_md_field_schema(self, name: str, field_info: Any, decorators: Any) -> Dict[str, Any]:
        """Prepare a ModelField to represent a model field."""
        common_field: Dict[str, Any] = self._common_field_schema(name, field_info, decorators)
        return core_schema.model_field(
            common_field['schema'],
            serialization_exclude=common_field['serialization_exclude'],
            validation_alias=common_field['validation_alias'],
            serialization_alias=common_field['serialization_alias'],
            frozen=common_field['frozen'],
            metadata=common_field['metadata'],
        )

    def _generate_dc_field_schema(self, name: str, field_info: Any, decorators: Any) -> Dict[str, Any]:
        """Prepare a DataclassField to represent the parameter/field, of a dataclass."""
        common_field: Dict[str, Any] = self._common_field_schema(name, field_info, decorators)
        return core_schema.dataclass_field(
            name,
            common_field['schema'],
            init=field_info.init,
            init_only=field_info.init_var or None,
            kw_only=None if field_info.kw_only else False,
            serialization_exclude=common_field['serialization_exclude'],
            validation_alias=common_field['validation_alias'],
            serialization_alias=common_field['serialization_alias'],
            frozen=common_field['frozen'],
            metadata=common_field['metadata'],
        )

    @staticmethod
    def _apply_alias_generator_to_field_info(alias_generator: Union[AliasGenerator, Callable[[str], str]], field_info: Any, field_name: str) -> None:
        """Apply an alias_generator to aliases on a FieldInfo instance if appropriate."""
        if field_info.alias_priority is None or field_info.alias_priority <= 1 or field_info.alias is None or (field_info.validation_alias is None) or (field_info.serialization_alias is None):
            alias: Optional[str] = None
            validation_alias: Optional[str] = None
            serialization_alias: Optional[str] = None
            if isinstance(alias_generator, AliasGenerator):
                alias, validation_alias, serialization_alias = alias_generator.generate_aliases(field_name)
            elif isinstance(alias_generator, Callable):
                alias = alias_generator(field_name)
                if not isinstance(alias, str):
                    raise TypeError(f'alias_generator {alias_generator} must return str, not {alias.__class__}')
            if field_info.alias_priority is None or field_info.alias_priority <= 1:
                field_info.alias_priority = 1
            if field_info.alias_priority == 1:
                field_info.serialization_alias = _get_first_non_null(serialization_alias, alias)
                field_info.validation_alias = _get_first_non_null(validation_alias, alias)
                field_info.alias = alias
            if field_info.alias is None:
                field_info.alias = alias
            if field_info.serialization_alias is None:
                field_info.serialization_alias = _get_first_non_null(serialization_alias, alias)
            if field_info.validation_alias is None:
                field_info.validation_alias = _get_first_non_null(validation_alias, alias)

    @staticmethod
    def _apply_alias_generator_to_computed_field_info(alias_generator: Union[AliasGenerator, Callable[[str], str]], computed_field_info: Any, computed_field_name: str) -> None:
        """Apply an alias_generator to alias on a ComputedFieldInfo instance if appropriate."""
        if computed_field_info.alias_priority is None or computed_field_info.alias_priority <= 1 or computed_field_info.alias is None:
            alias: Optional[str] = None
            validation_alias: Optional[str] = None
            serialization_alias: Optional[str] = None
            if isinstance(alias_generator, AliasGenerator):
                alias, validation_alias, serialization_alias = alias_generator.generate_aliases(computed_field_name)
            elif isinstance(alias_generator, Callable):
                alias = alias_generator(computed_field_name)
                if not isinstance(alias, str):
                    raise TypeError(f'alias_generator {alias_generator} must return str, not {alias.__class__}')
            if computed_field_info.alias_priority is None or computed_field_info.alias_priority <= 1:
                computed_field_info.alias_priority = 1
            if computed_field_info.alias_priority == 1:
                computed_field_info.alias = _get_first_non_null(serialization_alias, alias)

    @staticmethod
    def _apply_field_title_generator_to_field_info(config_wrapper: ConfigWrapper, field_info: Any, field_name: str) -> None:
        """Apply a field_title_generator on a FieldInfo or ComputedFieldInfo instance if appropriate.
        Args:
            config_wrapper: The config of the model
            field_info: The FieldInfo or ComputedField instance to which the title_generator is (maybe) applied.
            field_name: The name of the field from which to generate the title.
        """
        field_title_generator: Optional[Callable[[str, Any], str]] = field_info.field_title_generator or config_wrapper.field_title_generator
        if field_title_generator is None:
            return
        if field_info.title is None:
            title: Any = field_title_generator(field_name, field_info)
            if not isinstance(title, str):
                raise TypeError(f'field_title_generator {field_title_generator} must return str, not {title.__class__}')
            field_info.title = title

    def _common_field_schema(self, name: str, field_info: Any, decorators: Any) -> Dict[str, Any]:
        source_type, annotations = (field_info.annotation, field_info.metadata)

        def set_discriminator(schema: Dict[str, Any]) -> Dict[str, Any]:
            schema = self._apply_discriminator_to_union(schema, field_info.discriminator)
            return schema

        validators_from_decorators: List[Any] = []
        for decorator in filter_field_decorator_info_by_field(decorators.field_validators.values(), name):
            validators_from_decorators.append(_mode_to_validator[decorator.info.mode]._from_decorator(decorator))
        with self.field_name_stack.push(name):
            if field_info.discriminator is not None:
                schema: Dict[str, Any] = self._apply_annotations(source_type, annotations + validators_from_decorators, transform_inner_schema=set_discriminator)
            else:
                schema = self._apply_annotations(source_type, annotations + validators_from_decorators)
        this_field_validators = filter_field_decorator_info_by_field(decorators.validators.values(), name)
        if _validators_require_validate_default(this_field_validators):
            field_info.validate_default = True
        each_item_validators = [v for v in this_field_validators if v.info.each_item is True]
        this_field_validators = [v for v in this_field_validators if v not in each_item_validators]
        schema = apply_each_item_validators(schema, each_item_validators, name)
        schema = apply_validators(schema, this_field_validators, name)
        if not field_info.is_required():
            schema = wrap_default(field_info, schema)
        schema = self._apply_field_serializers(schema, filter_field_decorator_info_by_field(decorators.field_serializers.values(), name))
        self._apply_field_title_generator_to_field_info(self._config_wrapper, field_info, name)
        pydantic_js_updates, pydantic_js_extra = _extract_json_schema_info_from_field_info(field_info)
        core_metadata: Dict[str, Any] = {}
        update_core_metadata(core_metadata, pydantic_js_updates=pydantic_js_updates, pydantic_js_extra=pydantic_js_extra)
        alias_generator = self._config_wrapper.alias_generator
        if alias_generator is not None:
            self._apply_alias_generator_to_field_info(alias_generator, field_info, name)
        if isinstance(field_info.validation_alias, (AliasChoices, AliasPath)):
            validation_alias = field_info.validation_alias.convert_to_aliases()
        else:
            validation_alias = field_info.validation_alias
        return _common_field(
            schema, serialization_exclude=True if field_info.exclude else None,
            validation_alias=validation_alias,
            serialization_alias=field_info.serialization_alias,
            frozen=field_info.frozen,
            metadata=core_metadata,
        )

    def _union_schema(self, union_type: Any) -> Dict[str, Any]:
        """Generate schema for a Union."""
        args = self._get_args_resolving_forward_refs(union_type, required=True)
        choices: List[Dict[str, Any]] = []
        nullable: bool = False
        for arg in args:
            if arg is None or arg is _typing_extra.NoneType:
                nullable = True
            else:
                choices.append(self.generate_schema(arg))
        if len(choices) == 1:
            s: Dict[str, Any] = choices[0]
        else:
            choices_with_tags: List[Union[Dict[str, Any], Tuple[Dict[str, Any], Any]]] = []
            for choice in choices:
                tag = cast(CoreMetadata, choice.get('metadata', {})).get('pydantic_internal_union_tag_key')
                if tag is not None:
                    choices_with_tags.append((choice, tag))
                else:
                    choices_with_tags.append(choice)
            s = core_schema.union_schema(choices_with_tags)
        if nullable:
            s = core_schema.nullable_schema(s)
        return s

    def _type_alias_type_schema(self, obj: Any) -> Dict[str, Any]:
        with self.defs.get_schema_or_ref(obj) as (ref, maybe_schema):
            if maybe_schema is not None:
                return maybe_schema
            origin = get_origin(obj) or obj
            typevars_map = get_standard_typevars_map(obj)
            with self._ns_resolver.push(origin):
                try:
                    annotation = _typing_extra.eval_type(origin.__value__, *self._types_namespace)
                except NameError as e:
                    raise PydanticUndefinedAnnotation.from_name_error(e) from e
                annotation = replace_types(annotation, typevars_map)
                schema = self.generate_schema(annotation)
                assert schema['type'] != 'definitions'
                schema['ref'] = ref
            return self.defs.create_definition_reference_schema(schema)

    def _literal_schema(self, literal_type: Any) -> Dict[str, Any]:
        """Generate schema for a Literal."""
        expected = _typing_extra.literal_values(literal_type)
        assert expected, f'literal "expected" cannot be empty, obj={literal_type}'
        schema = core_schema.literal_schema(expected)
        if self._config_wrapper.use_enum_values and any((isinstance(v, Enum) for v in expected)):
            schema = core_schema.no_info_after_validator_function(lambda v: v.value if isinstance(v, Enum) else v, schema)
        return schema

    def _typed_dict_schema(self, typed_dict_cls: Any, origin: Optional[Any]) -> Dict[str, Any]:
        """Generate schema for a TypedDict."""
        FieldInfo = import_cached_field_info()
        with self.model_type_stack.push(typed_dict_cls), self.defs.get_schema_or_ref(typed_dict_cls) as (typed_dict_ref, maybe_schema):
            if maybe_schema is not None:
                return maybe_schema
            typevars_map = get_standard_typevars_map(typed_dict_cls)
            if origin is not None:
                typed_dict_cls = origin
            if not _SUPPORTS_TYPEDDICT and type(typed_dict_cls).__module__ == 'typing':
                raise PydanticUserError('Please use `typing_extensions.TypedDict` instead of `typing.TypedDict` on Python < 3.12.', code='typed-dict-version')
            try:
                config = get_attribute_from_bases(typed_dict_cls, '__pydantic_config__')
            except AttributeError:
                config = None
            with self._config_wrapper_stack.push(config):
                core_config: Dict[str, Any] = self._config_wrapper.core_config(title=typed_dict_cls.__name__)
                required_keys = typed_dict_cls.__required_keys__
                fields: Dict[str, Any] = {}
                decorators = DecoratorInfos.build(typed_dict_cls)
                if self._config_wrapper.use_attribute_docstrings:
                    field_docstrings = extract_docstrings_from_cls(typed_dict_cls, use_inspect=True)
                else:
                    field_docstrings = None
                try:
                    annotations = _typing_extra.get_cls_type_hints(typed_dict_cls, ns_resolver=self._ns_resolver)
                except NameError as e:
                    raise PydanticUndefinedAnnotation.from_name_error(e) from e
                for field_name, annotation in annotations.items():
                    annotation = replace_types(annotation, typevars_map)
                    required = field_name in required_keys
                    if _typing_extra.is_required(annotation):
                        required = True
                        annotation = self._get_args_resolving_forward_refs(annotation, required=True)[0]
                    elif _typing_extra.is_not_required(annotation):
                        required = False
                        annotation = self._get_args_resolving_forward_refs(annotation, required=True)[0]
                    field_info = FieldInfo.from_annotation(annotation)
                    if field_docstrings is not None and field_info.description is None and (field_name in field_docstrings):
                        field_info.description = field_docstrings[field_name]
                    self._apply_field_title_generator_to_field_info(self._config_wrapper, field_info, field_name)
                    fields[field_name] = self._generate_td_field_schema(field_name, field_info, decorators, required=required)
                td_schema: Dict[str, Any] = core_schema.typed_dict_schema(
                    fields, cls=typed_dict_cls,
                    computed_fields=[self._computed_field_schema(d, decorators.field_serializers) for d in decorators.computed_fields.values()],
                    ref=typed_dict_ref,
                    config=core_config,
                )
                schema = self._apply_model_serializers(td_schema, decorators.model_serializers.values())
                schema = apply_model_validators(schema, decorators.model_validators.values(), 'all')
                return self.defs.create_definition_reference_schema(schema)

    def _namedtuple_schema(self, namedtuple_cls: Any, origin: Optional[Any]) -> Dict[str, Any]:
        """Generate schema for a NamedTuple."""
        with self.model_type_stack.push(namedtuple_cls), self.defs.get_schema_or_ref(namedtuple_cls) as (namedtuple_ref, maybe_schema):
            if maybe_schema is not None:
                return maybe_schema
            typevars_map = get_standard_typevars_map(namedtuple_cls)
            if origin is not None:
                namedtuple_cls = origin
            try:
                annotations = _typing_extra.get_cls_type_hints(namedtuple_cls, ns_resolver=self._ns_resolver)
            except NameError as e:
                raise PydanticUndefinedAnnotation.from_name_error(e) from e
            if not annotations:
                annotations = {k: Any for k in namedtuple_cls._fields}
            if typevars_map:
                annotations = {field_name: replace_types(annotation, typevars_map) for field_name, annotation in annotations.items()}
            arguments_schema: Dict[str, Any] = core_schema.arguments_schema(
                [self._generate_parameter_schema(field_name, annotation, default=namedtuple_cls._field_defaults.get(field_name, Parameter.empty))
                 for field_name, annotation in annotations.items()],
                metadata={'pydantic_js_prefer_positional_arguments': True},
            )
            schema: Dict[str, Any] = core_schema.call_schema(arguments_schema, namedtuple_cls, ref=namedtuple_ref)
            return self.defs.create_definition_reference_schema(schema)

    def _generate_parameter_schema(self, name: str, annotation: Any, default: Any = Parameter.empty, mode: Optional[str] = None) -> Dict[str, Any]:
        """Prepare an ArgumentsParameter to represent a field in a namedtuple or function signature."""
        FieldInfo = import_cached_field_info()
        if default is Parameter.empty:
            field = FieldInfo.from_annotation(annotation)
        else:
            field = FieldInfo.from_annotated_attribute(annotation, default)
        assert field.annotation is not None, 'field.annotation should not be None when generating a schema'
        with self.field_name_stack.push(name):
            schema = self._apply_annotations(field.annotation, [field])
        if not field.is_required():
            schema = wrap_default(field, schema)
        parameter_schema: Dict[str, Any] = core_schema.arguments_parameter(name, schema)
        if mode is not None:
            parameter_schema['mode'] = mode
        if field.alias is not None:
            parameter_schema['alias'] = field.alias
        else:
            alias_generator = self._config_wrapper.alias_generator
            if isinstance(alias_generator, AliasGenerator) and alias_generator.alias is not None:
                parameter_schema['alias'] = alias_generator.alias(name)
            elif isinstance(alias_generator, Callable):
                parameter_schema['alias'] = alias_generator(name)
        return parameter_schema

    def _tuple_schema(self, tuple_type: Any) -> Dict[str, Any]:
        """Generate schema for a Tuple, e.g. `tuple[int, str]` or `tuple[int, ...]`."""
        typevars_map = get_standard_typevars_map(tuple_type)
        params = self._get_args_resolving_forward_refs(tuple_type)
        if typevars_map and params:
            params = tuple((replace_types(param, typevars_map) for param in params))
        if not params:
            if tuple_type in TUPLE_TYPES:
                return core_schema.tuple_schema([core_schema.any_schema()], variadic_item_index=0)
            else:
                return core_schema.tuple_schema([])
        elif params[-1] is Ellipsis:
            if len(params) == 2:
                return core_schema.tuple_schema([self.generate_schema(params[0])], variadic_item_index=0)
            else:
                raise ValueError('Variable tuples can only have one type')
        elif len(params) == 1 and params[0] == ():
            return core_schema.tuple_schema([])
        else:
            return core_schema.tuple_schema([self.generate_schema(param) for param in params])

    def _type_schema(self) -> Dict[str, Any]:
        return core_schema.custom_error_schema(core_schema.is_instance_schema(type), custom_error_type='is_type', custom_error_message='Input should be a type')

    def _zoneinfo_schema(self) -> Dict[str, Any]:
        """Generate schema for a zone_info.ZoneInfo object"""
        from ._validators import validate_str_is_valid_iana_tz
        metadata = {'pydantic_js_functions': [lambda _1, _2: {'type': 'string', 'format': 'zoneinfo'}]}
        return core_schema.no_info_plain_validator_function(validate_str_is_valid_iana_tz, serialization=core_schema.to_string_ser_schema(), metadata=metadata)

    def _union_is_subclass_schema(self, union_type: Any) -> Dict[str, Any]:
        """Generate schema for `type[Union[X, ...]]`."""
        args = self._get_args_resolving_forward_refs(union_type, required=True)
        return core_schema.union_schema([self.generate_schema(type[args]) for args in args])

    def _subclass_schema(self, type_: Any) -> Dict[str, Any]:
        """Generate schema for a type, e.g. `type[int]`."""
        type_param = self._get_first_arg_or_any(type_)
        type_param = _typing_extra.annotated_type(type_param) or type_param
        if _typing_extra.is_any(type_param):
            return self._type_schema()
        elif _typing_extra.is_type_alias_type(type_param):
            return self.generate_schema(type[type_param.__value__])
        elif isinstance(type_param, typing.TypeVar):
            if type_param.__bound__:
                if _typing_extra.origin_is_union(get_origin(type_param.__bound__)):
                    return self._union_is_subclass_schema(type_param.__bound__)
                return core_schema.is_subclass_schema(type_param.__bound__)
            elif type_param.__constraints__:
                return core_schema.union_schema([self.generate_schema(type[c]) for c in type_param.__constraints__])
            else:
                return self._type_schema()
        elif _typing_extra.origin_is_union(get_origin(type_param)):
            return self._union_is_subclass_schema(type_param)
        else:
            if _typing_extra.is_self(type_param):
                type_param = self._resolve_self_type(type_param)
            if _typing_extra.is_generic_alias(type_param):
                raise PydanticUserError(
                    f'Subscripting `type[]` with an already parametrized type is not supported. Instead of using type[{type_param!r}], use type[{_repr.display_as_type(get_origin(type_param))}].',
                    code=None,
                )
            if not inspect.isclass(type_param):
                if type_param is None:
                    return core_schema.is_subclass_schema(_typing_extra.NoneType)
                raise TypeError(f'Expected a class, got {type_param!r}')
            return core_schema.is_subclass_schema(type_param)

    def _sequence_schema(self, items_type: Any) -> Dict[str, Any]:
        """Generate schema for a Sequence, e.g. `Sequence[int]`."""
        from ._serializers import serialize_sequence_via_list
        item_type_schema: Dict[str, Any] = self.generate_schema(items_type)
        list_schema: Dict[str, Any] = core_schema.list_schema(item_type_schema)
        json_schema = smart_deepcopy(list_schema)
        python_schema: Dict[str, Any] = core_schema.is_instance_schema(typing.Sequence, cls_repr='Sequence')
        if not _typing_extra.is_any(items_type):
            from ._validators import sequence_validator
            python_schema = core_schema.chain_schema([python_schema, core_schema.no_info_wrap_validator_function(sequence_validator, list_schema)])
        serialization = core_schema.wrap_serializer_function_ser_schema(serialize_sequence_via_list, schema=item_type_schema, info_arg=True)
        return core_schema.json_or_python_schema(json_schema=json_schema, python_schema=python_schema, serialization=serialization)

    def _iterable_schema(self, type_: Any) -> Dict[str, Any]:
        """Generate a schema for an `Iterable`."""
        item_type = self._get_first_arg_or_any(type_)
        return core_schema.generator_schema(self.generate_schema(item_type))

    def _pattern_schema(self, pattern_type: Any) -> Dict[str, Any]:
        from . import _validators
        metadata = {'pydantic_js_functions': [lambda _1, _2: {'type': 'string', 'format': 'regex'}]}
        ser = core_schema.plain_serializer_function_ser_schema(attrgetter('pattern'), when_used='json', return_schema=core_schema.str_schema())
        if pattern_type is typing.Pattern or pattern_type is re.Pattern:
            return core_schema.no_info_plain_validator_function(_validators.pattern_either_validator, serialization=ser, metadata=metadata)
        param = self._get_args_resolving_forward_refs(pattern_type, required=True)[0]
        if param is str:
            return core_schema.no_info_plain_validator_function(_validators.pattern_str_validator, serialization=ser, metadata=metadata)
        elif param is bytes:
            return core_schema.no_info_plain_validator_function(_validators.pattern_bytes_validator, serialization=ser, metadata=metadata)
        else:
            raise PydanticSchemaGenerationError(f'Unable to generate pydantic-core schema for {pattern_type!r}.')

    def _hashable_schema(self) -> Dict[str, Any]:
        return core_schema.custom_error_schema(
            schema=core_schema.json_or_python_schema(
                json_schema=core_schema.chain_schema([core_schema.any_schema(), core_schema.is_instance_schema(collections.abc.Hashable)]),
                python_schema=core_schema.is_instance_schema(collections.abc.Hashable),
            ),
            custom_error_type='is_hashable',
            custom_error_message='Input should be hashable',
        )

    def _dataclass_schema(self, dataclass_obj: Any, origin: Optional[Any]) -> Dict[str, Any]:
        """Generate schema for a dataclass."""
        with self.model_type_stack.push(dataclass_obj), self.defs.get_schema_or_ref(dataclass_obj) as (dataclass_ref, maybe_schema):
            if maybe_schema is not None:
                return maybe_schema
            schema: Optional[Dict[str, Any]] = dataclass_obj.__dict__.get('__pydantic_core_schema__')
            if schema is not None and (not isinstance(schema, MockCoreSchema)):
                if schema['type'] == 'definitions':
                    schema = self.defs.unpack_definitions(schema)
                ref = get_ref(schema)
                if ref:
                    return self.defs.create_definition_reference_schema(schema)
                else:
                    return schema
            typevars_map = get_standard_typevars_map(dataclass_obj)
            if origin is not None:
                dataclass_obj = origin
            config = getattr(dataclass_obj, '__pydantic_config__', None)
            from ..dataclasses import is_pydantic_dataclass
            with self._ns_resolver.push(dataclass_obj), self._config_wrapper_stack.push(config):
                if is_pydantic_dataclass(dataclass_obj):
                    fields = {f_name: copy(field_info) for f_name, field_info in dataclass_obj.__pydantic_fields__.items()}
                    if typevars_map:
                        for field in fields.values():
                            field.apply_typevars_map(typevars_map, *self._types_namespace)
                else:
                    fields = collect_dataclass_fields(dataclass_obj, typevars_map=typevars_map, config_wrapper=self._config_wrapper)
                if self._config_wrapper.extra == 'allow':
                    for field_name, field in fields.items():
                        if field.init is False:
                            raise PydanticUserError(
                                f'Field {field_name} has `init=False` and dataclass has config setting `extra="allow"`. This combination is not allowed.',
                                code='dataclass-init-false-extra-allow',
                            )
                decorators = dataclass_obj.__dict__.get('__pydantic_decorators__') or DecoratorInfos.build(dataclass_obj)
                args = sorted(
                    (self._generate_dc_field_schema(k, v, decorators) for k, v in fields.items()),
                    key=lambda a: a.get('kw_only') is not False,
                )
                has_post_init: bool = hasattr(dataclass_obj, '__post_init__')
                has_slots: bool = hasattr(dataclass_obj, '__slots__')
                args_schema: Dict[str, Any] = core_schema.dataclass_args_schema(
                    dataclass_obj.__name__, args,
                    computed_fields=[self._computed_field_schema(d, decorators.field_serializers) for d in decorators.computed_fields.values()],
                    collect_init_only=has_post_init,
                )
                inner_schema = apply_validators(args_schema, decorators.root_validators.values(), None)
                model_validators = decorators.model_validators.values()
                inner_schema = apply_model_validators(inner_schema, model_validators, 'inner')
                core_config: Dict[str, Any] = self._config_wrapper.core_config(title=dataclass_obj.__name__)
                dc_schema: Dict[str, Any] = core_schema.dataclass_schema(
                    dataclass_obj, inner_schema, generic_origin=origin,
                    post_init=has_post_init, ref=dataclass_ref,
                    fields=[field.name for field in dataclasses.fields(dataclass_obj)],
                    slots=has_slots, config=core_config, frozen=self._config_wrapper_stack.tail.frozen,
                )
                schema = self._apply_model_serializers(dc_schema, decorators.model_serializers.values())
                schema = apply_model_validators(schema, model_validators, 'outer')
                return self.defs.create_definition_reference_schema(schema)

    def _call_schema(self, function: Union[LambdaType, FunctionType, MethodType, partial]) -> Dict[str, Any]:
        """Generate schema for a Callable."""
        sig = signature(function)
        globalns, localns = self._types_namespace
        type_hints: Dict[str, Any] = _typing_extra.get_function_type_hints(function, globalns=globalns, localns=localns)
        mode_lookup: Dict[int, str] = {Parameter.POSITIONAL_ONLY: 'positional_only', Parameter.POSITIONAL_OR_KEYWORD: 'positional_or_keyword', Parameter.KEYWORD_ONLY: 'keyword_only'}
        arguments_list: List[Dict[str, Any]] = []
        var_args_schema: Optional[Dict[str, Any]] = None
        var_kwargs_schema: Optional[Dict[str, Any]] = None
        var_kwargs_mode: Optional[str] = None
        for name, p in sig.parameters.items():
            if p.annotation is sig.empty:
                annotation = Any
            else:
                annotation = type_hints[name]
            parameter_mode: Optional[str] = mode_lookup.get(p.kind)
            if parameter_mode is not None:
                arg_schema = self._generate_parameter_schema(name, annotation, p.default, parameter_mode)
                arguments_list.append(arg_schema)
            elif p.kind == Parameter.VAR_POSITIONAL:
                var_args_schema = self.generate_schema(annotation)
            else:
                assert p.kind == Parameter.VAR_KEYWORD, p.kind
                unpack_type = _typing_extra.unpack_type(annotation)
                if unpack_type is not None:
                    if not is_typeddict(unpack_type):
                        raise PydanticUserError(f'Expected a `TypedDict` class, got {unpack_type.__name__!r}', code='unpack-typed-dict')
                    non_pos_only_param_names = {name for name, p in sig.parameters.items() if p.kind != Parameter.POSITIONAL_ONLY}
                    overlapping_params = non_pos_only_param_names.intersection(unpack_type.__annotations__)
                    if overlapping_params:
                        raise PydanticUserError(f'Typed dictionary {unpack_type.__name__!r} overlaps with parameter{("s" if len(overlapping_params) >= 2 else "")} {", ".join((repr(p) for p in sorted(overlapping_params)))}', code='overlapping-unpack-typed-dict')
                    var_kwargs_mode = 'unpacked-typed-dict'
                    var_kwargs_schema = self._typed_dict_schema(unpack_type, None)
                else:
                    var_kwargs_mode = 'uniform'
                    var_kwargs_schema = self.generate_schema(annotation)
        return_schema: Optional[Dict[str, Any]] = None
        config_wrapper = self._config_wrapper
        if config_wrapper.validate_return:
            return_hint = sig.return_annotation
            if return_hint is not sig.empty:
                return_schema = self.generate_schema(return_hint)
        return core_schema.call_schema(
            core_schema.arguments_schema(
                arguments_list,
                var_args_schema=var_args_schema,
                var_kwargs_mode=var_kwargs_mode,
                var_kwargs_schema=var_kwargs_schema,
                populate_by_name=config_wrapper.populate_by_name,
            ),
            function,
            return_schema=return_schema,
        )

    def _unsubstituted_typevar_schema(self, typevar: Any) -> Dict[str, Any]:
        try:
            has_default = typevar.has_default()
        except AttributeError:
            pass
        else:
            if has_default:
                return self.generate_schema(typevar.__default__)
        if (constraints := typevar.__constraints__):
            return self._union_schema(typing.Union[constraints])
        if (bound := typevar.__bound__):
            schema = self.generate_schema(bound)
            schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(lambda x, h: h(x), schema=core_schema.any_schema())
            return schema
        return core_schema.any_schema()

    def _computed_field_schema(self, d: Any, field_serializers: Any) -> Dict[str, Any]:
        try:
            return_type = _decorators.get_function_return_type(d.func, d.info.return_type, localns=self._types_namespace.locals)
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        if return_type is PydanticUndefined:
            raise PydanticUserError('Computed field is missing return type annotation or specifying `return_type` to the `@computed_field` decorator (e.g. `@computed_field(return_type=int|str)`)', code='model-field-missing-annotation')
        return_type = replace_types(return_type, self._typevars_map)
        d.info = dataclasses.replace(d.info, return_type=return_type)
        return_type_schema = self.generate_schema(return_type)
        return_type_schema = self._apply_field_serializers(return_type_schema, filter_field_decorator_info_by_field(field_serializers.values(), d.cls_var_name))
        alias_generator = self._config_wrapper.alias_generator
        if alias_generator is not None:
            self._apply_alias_generator_to_computed_field_info(alias_generator=alias_generator, computed_field_info=d.info, computed_field_name=d.cls_var_name)
        self._apply_field_title_generator_to_field_info(self._config_wrapper, d.info, d.cls_var_name)
        pydantic_js_updates, pydantic_js_extra = _extract_json_schema_info_from_field_info(d.info)
        core_metadata: Dict[str, Any] = {}
        update_core_metadata(core_metadata, pydantic_js_updates={'readOnly': True, **(pydantic_js_updates if pydantic_js_updates else {})}, pydantic_js_extra=pydantic_js_extra)
        return core_schema.computed_field(d.cls_var_name, return_schema=return_type_schema, alias=d.info.alias, metadata=core_metadata)

    def _annotated_schema(self, annotated_type: Any) -> Dict[str, Any]:
        """Generate schema for an Annotated type, e.g. `Annotated[int, Field(...)]` or `Annotated[int, Gt(0)]`."""
        FieldInfo = import_cached_field_info()
        typ, *annotations = get_args(annotated_type)
        if isinstance(typ, str):
            typ = _typing_extra._make_forward_ref(typ)
        if isinstance(typ, ForwardRef):
            typ = self._resolve_forward_ref(typ)
        typ, sub_annotations = _typing_extra.unpack_annotated(typ)
        annotations = sub_annotations + annotations
        schema = self._apply_annotations(typ, annotations)
        for annotation in annotations:
            if isinstance(annotation, FieldInfo):
                schema = wrap_default(annotation, schema)
        return schema

    def _apply_annotations(self, source_type: Any, annotations: List[Any], transform_inner_schema: Callable[[Dict[str, Any]], Dict[str, Any]] = lambda x: x) -> Dict[str, Any]:
        """Apply arguments from `Annotated` or from `FieldInfo` to a schema."""
        annotations = list(_known_annotated_metadata.expand_grouped_metadata(annotations))
        pydantic_js_annotation_functions: List[Callable[..., Any]] = []

        def inner_handler(obj: Any) -> Dict[str, Any]:
            schema = self._generate_schema_from_get_schema_method(obj, source_type)
            if schema is None:
                schema = self._generate_schema_inner(obj)
            metadata_js_function = _extract_get_pydantic_json_schema(obj)
            if metadata_js_function is not None:
                metadata_schema = resolve_original_schema(schema, self.defs)
                if metadata_schema is not None:
                    self._add_js_function(metadata_schema, metadata_js_function)
            return transform_inner_schema(schema)
        get_inner_schema: CallbackGetCoreSchemaHandler = CallbackGetCoreSchemaHandler(inner_handler, self)
        for annotation in annotations:
            if annotation is None:
                continue
            get_inner_schema = self._get_wrapped_inner_schema(get_inner_schema, annotation, pydantic_js_annotation_functions)
        schema = get_inner_schema(source_type)
        if pydantic_js_annotation_functions:
            core_metadata = schema.setdefault('metadata', {})
            update_core_metadata(core_metadata, pydantic_js_annotation_functions=pydantic_js_annotation_functions)
        return _add_custom_serialization_from_json_encoders(self._config_wrapper.json_encoders, source_type, schema)

    def _apply_single_annotation(self, schema: Dict[str, Any], metadata: Any) -> Dict[str, Any]:
        FieldInfo = import_cached_field_info()
        if isinstance(metadata, FieldInfo):
            for field_metadata in metadata.metadata:
                schema = self._apply_single_annotation(schema, field_metadata)
            if metadata.discriminator is not None:
                schema = self._apply_discriminator_to_union(schema, metadata.discriminator)
            return schema
        if schema['type'] == 'nullable':
            inner = schema.get('schema', core_schema.any_schema())
            inner = self._apply_single_annotation(inner, metadata)
            if inner:
                schema['schema'] = inner
            return schema
        original_schema = schema
        ref = schema.get('ref')
        if ref is not None:
            schema = schema.copy()
            new_ref = ref + f'_{repr(metadata)}'
            if (existing := self.defs.get_schema_from_ref(new_ref)) is not None:
                return existing
            schema['ref'] = new_ref
        elif schema['type'] == 'definition-ref':
            ref = schema['schema_ref']
            if (referenced_schema := self.defs.get_schema_from_ref(ref)) is not None:
                schema = referenced_schema.copy()
                new_ref = ref + f'_{repr(metadata)}'
                if (existing := self.defs.get_schema_from_ref(new_ref)) is not None:
                    return existing
                schema['ref'] = new_ref
        maybe_updated_schema = _known_annotated_metadata.apply_known_metadata(metadata, schema)
        if maybe_updated_schema is not None:
            return maybe_updated_schema
        return original_schema

    def _apply_single_annotation_json_schema(self, schema: Dict[str, Any], metadata: Any) -> Dict[str, Any]:
        FieldInfo = import_cached_field_info()
        if isinstance(metadata, FieldInfo):
            for field_metadata in metadata.metadata:
                schema = self._apply_single_annotation_json_schema(schema, field_metadata)
            pydantic_js_updates, pydantic_js_extra = _extract_json_schema_info_from_field_info(metadata)
            core_metadata = schema.setdefault('metadata', {})
            update_core_metadata(core_metadata, pydantic_js_updates=pydantic_js_updates, pydantic_js_extra=pydantic_js_extra)
        return schema

    def _get_wrapped_inner_schema(self, get_inner_schema: CallbackGetCoreSchemaHandler, annotation: Any, pydantic_js_annotation_functions: List[Callable[..., Any]]) -> CallbackGetCoreSchemaHandler:
        annotation_get_schema = getattr(annotation, '__get_pydantic_core_schema__', None)

        def new_handler(source: Any) -> Dict[str, Any]:
            if annotation_get_schema is not None:
                schema = annotation_get_schema(source, get_inner_schema)
            else:
                schema = get_inner_schema(source)
                schema = self._apply_single_annotation(schema, annotation)
                schema = self._apply_single_annotation_json_schema(schema, annotation)
            metadata_js_function = _extract_get_pydantic_json_schema(annotation)
            if metadata_js_function is not None:
                pydantic_js_annotation_functions.append(metadata_js_function)
            return schema
        return CallbackGetCoreSchemaHandler(new_handler, self)

    def _apply_field_serializers(self, schema: Dict[str, Any], serializers: List[Any]) -> Dict[str, Any]:
        """Apply field serializers to a schema."""
        if serializers:
            schema = copy(schema)
            if schema['type'] == 'definitions':
                inner_schema = schema['schema']
                schema['schema'] = self._apply_field_serializers(inner_schema, serializers)
                return schema
            elif 'ref' in schema:
                schema = self.defs.create_definition_reference_schema(schema)
            serializer = serializers[-1]
            is_field_serializer, info_arg = inspect_field_serializer(serializer.func, serializer.info.mode)
            try:
                return_type = _decorators.get_function_return_type(serializer.func, serializer.info.return_type, localns=self._types_namespace.locals)
            except NameError as e:
                raise PydanticUndefinedAnnotation.from_name_error(e) from e
            if return_type is PydanticUndefined:
                return_schema = None
            else:
                return_schema = self.generate_schema(return_type)
            if serializer.info.mode == 'wrap':
                schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(serializer.func, is_field_serializer=is_field_serializer, info_arg=info_arg, return_schema=return_schema, when_used=serializer.info.when_used)
            else:
                assert serializer.info.mode == 'plain'
                schema['serialization'] = core_schema.plain_serializer_function_ser_schema(serializer.func, is_field_serializer=is_field_serializer, info_arg=info_arg, return_schema=return_schema, when_used=serializer.info.when_used)
        return schema

    def _apply_model_serializers(self, schema: Dict[str, Any], serializers: List[Any]) -> Dict[str, Any]:
        """Apply model serializers to a schema."""
        ref = schema.pop('ref', None)
        if serializers:
            serializer = list(serializers)[-1]
            info_arg = inspect_model_serializer(serializer.func, serializer.info.mode)
            try:
                return_type = _decorators.get_function_return_type(serializer.func, serializer.info.return_type, localns=self._types_namespace.locals)
            except NameError as e:
                raise PydanticUndefinedAnnotation.from_name_error(e) from e
            if return_type is PydanticUndefined:
                return_schema = None
            else:
                return_schema = self.generate_schema(return_type)
            if serializer.info.mode == 'wrap':
                ser_schema = core_schema.wrap_serializer_function_ser_schema(serializer.func, info_arg=info_arg, return_schema=return_schema, when_used=serializer.info.when_used)
            else:
                ser_schema = core_schema.plain_serializer_function_ser_schema(serializer.func, info_arg=info_arg, return_schema=return_schema, when_used=serializer.info.when_used)
            schema['serialization'] = ser_schema
        if ref:
            schema['ref'] = ref
        return schema


_VALIDATOR_F_MATCH: Dict[Tuple[str, str], Callable[[Any, Dict[str, Any], Optional[str]], Dict[str, Any]]] = {
    ('before', 'no-info'): lambda f, schema, _: core_schema.no_info_before_validator_function(f, schema),
    ('after', 'no-info'): lambda f, schema, _: core_schema.no_info_after_validator_function(f, schema),
    ('plain', 'no-info'): lambda f, _1, _2: core_schema.no_info_plain_validator_function(f),
    ('wrap', 'no-info'): lambda f, schema, _: core_schema.no_info_wrap_validator_function(f, schema),
    ('before', 'with-info'): lambda f, schema, field_name: core_schema.with_info_before_validator_function(f, schema, field_name=field_name),
    ('after', 'with-info'): lambda f, schema, field_name: core_schema.with_info_after_validator_function(f, schema, field_name=field_name),
    ('plain', 'with-info'): lambda f, _, field_name: core_schema.with_info_plain_validator_function(f, field_name=field_name),
    ('wrap', 'with-info'): lambda f, schema, field_name: core_schema.with_info_wrap_validator_function(f, schema, field_name=field_name),
}


def apply_validators(schema: Dict[str, Any], validators: TypingIterable[Any], field_name: Optional[str]) -> Dict[str, Any]:
    """Apply validators to a schema."""
    for validator in validators:
        info_arg = inspect_validator(validator.func, validator.info.mode)
        val_type = 'with-info' if info_arg else 'no-info'
        schema = _VALIDATOR_F_MATCH[(validator.info.mode, val_type)](validator.func, schema, field_name)
    return schema


def _validators_require_validate_default(validators: List[Any]) -> bool:
    """Loop over validators to determine if default validation is required."""
    for validator in validators:
        if validator.info.always:
            return True
    return False


def apply_model_validators(schema: Dict[str, Any], validators: TypingIterable[Any], mode: str) -> Dict[str, Any]:
    """Apply model validators to a schema."""
    ref = schema.pop('ref', None)
    for validator in validators:
        if mode == 'inner' and validator.info.mode != 'before':
            continue
        if mode == 'outer' and validator.info.mode == 'before':
            continue
        info_arg = inspect_validator(validator.func, validator.info.mode)
        if validator.info.mode == 'wrap':
            if info_arg:
                schema = core_schema.with_info_wrap_validator_function(function=validator.func, schema=schema)
            else:
                schema = core_schema.no_info_wrap_validator_function(function=validator.func, schema=schema)
        elif validator.info.mode == 'before':
            if info_arg:
                schema = core_schema.with_info_before_validator_function(function=validator.func, schema=schema)
            else:
                schema = core_schema.no_info_before_validator_function(function=validator.func, schema=schema)
        else:
            assert validator.info.mode == 'after'
            if info_arg:
                schema = core_schema.with_info_after_validator_function(function=validator.func, schema=schema)
            else:
                schema = core_schema.no_info_after_validator_function(function=validator.func, schema=schema)
    if ref:
        schema['ref'] = ref
    return schema


def wrap_default(field_info: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap schema with default schema if available."""
    if field_info.default_factory:
        return core_schema.with_default_schema(schema, default_factory=field_info.default_factory, default_factory_takes_data=takes_validated_data_argument(field_info.default_factory), validate_default=field_info.validate_default)
    elif field_info.default is not PydanticUndefined:
        return core_schema.with_default_schema(schema, default=field_info.default, validate_default=field_info.validate_default)
    else:
        return schema


def _extract_get_pydantic_json_schema(tp: Any) -> Optional[Callable[..., Any]]:
    """Extract `__get_pydantic_json_schema__` from a type, handling the deprecated `__modify_schema__`."""
    js_modify_function = getattr(tp, '__get_pydantic_json_schema__', None)
    if hasattr(tp, '__modify_schema__'):
        BaseModel = import_cached_base_model()
        has_custom_v2_modify_js_func = js_modify_function is not None and BaseModel.__get_pydantic_json_schema__.__func__ not in (js_modify_function, getattr(js_modify_function, '__func__', None))
        if not has_custom_v2_modify_js_func:
            cls_name = getattr(tp, '__name__', None)
            raise PydanticUserError(f'The `__modify_schema__` method is not supported in Pydantic v2. Use `__get_pydantic_json_schema__` instead{(f" in class `{cls_name}`" if cls_name else "")}.', code='custom-json-schema')
    if hasattr(tp, '__origin__') and (not _typing_extra.is_annotated(tp)):
        return _extract_get_pydantic_json_schema(tp.__origin__)
    if js_modify_function is None:
        return None
    return js_modify_function


class _CommonField(TypedDict):
    pass


def _common_field(
    schema: Dict[str, Any],
    *,
    validation_alias: Optional[Any] = None,
    serialization_alias: Optional[Any] = None,
    serialization_exclude: Optional[Any] = None,
    frozen: Optional[Any] = None,
    metadata: Optional[Any] = None,
) -> Dict[str, Any]:
    return {
        'schema': schema,
        'validation_alias': validation_alias,
        'serialization_alias': serialization_alias,
        'serialization_exclude': serialization_exclude,
        'frozen': frozen,
        'metadata': metadata,
    }


def resolve_original_schema(schema: Dict[str, Any], definitions: _Definitions) -> Optional[Dict[str, Any]]:
    if schema['type'] == 'definition-ref':
        return definitions.get_schema_from_ref(schema['schema_ref'])
    elif schema['type'] == 'definitions':
        return schema['schema']
    else:
        return schema


def _can_be_inlined(def_ref: Dict[str, Any]) -> bool:
    """Return whether the 'definition-ref' schema can be replaced by its definition."""
    return 'serialization' not in def_ref and (not def_ref.get('metadata'))


class _Definitions:
    """Keeps track of references and definitions."""

    def __init__(self) -> None:
        self._recursively_seen: set[Any] = set()
        self._definitions: Dict[str, Dict[str, Any]] = {}

    @contextmanager
    def get_schema_or_ref(self, tp: Any) -> Generator[Tuple[str, Optional[Dict[str, Any]]], None, None]:
        ref: str = get_type_ref(tp)
        if ref in self._recursively_seen or ref in self._definitions:
            yield (ref, core_schema.definition_reference_schema(ref))
        else:
            self._recursively_seen.add(ref)
            try:
                yield (ref, None)
            finally:
                self._recursively_seen.discard(ref)

    def get_schema_from_ref(self, ref: str) -> Optional[Dict[str, Any]]:
        return self._definitions.get(ref)

    def create_definition_reference_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        ref = schema['ref']
        self._definitions[ref] = schema
        return core_schema.definition_reference_schema(ref)

    def unpack_definitions(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        for def_schema in schema['definitions']:
            self._definitions[def_schema['ref']] = def_schema
        return schema['schema']

    def finalize_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        definitions = self._definitions
        try:
            gather_result = gather_schemas_for_cleaning(schema, definitions=definitions)
        except MissingDefinitionError as e:
            raise InvalidSchemaError from e
        remaining_defs: Dict[str, Dict[str, Any]] = {}
        for ref, inlinable_def_ref in gather_result['collected_references'].items():
            if inlinable_def_ref is not None and _can_be_inlined(inlinable_def_ref):
                inlinable_def_ref.clear()
                inlinable_def_ref.update(self._resolve_definition(ref, definitions))
            else:
                remaining_defs[ref] = self._resolve_definition(ref, definitions)
        for cs in gather_result['deferred_discriminator_schemas']:
            discriminator = cs['metadata']['pydantic_internal_union_discriminator']
            applied = _discriminated_union.apply_discriminator(cs.copy(), discriminator, remaining_defs)
            cs.clear()
            cs.update(applied)
        if remaining_defs:
            schema = core_schema.definitions_schema(schema=schema, definitions=[*remaining_defs.values()])
        return schema

    def _resolve_definition(self, ref: str, definitions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        definition = definitions[ref]
        if definition['type'] != 'definition-ref':
            return definition
        visited: set[str] = set()
        while definition['type'] == 'definition-ref':
            schema_ref = definition['schema_ref']
            if schema_ref in visited:
                raise PydanticUserError(f'{ref} contains a circular reference to itself.', code='circular-reference-schema')
            visited.add(schema_ref)
            definition = definitions[schema_ref]
        return {**definition, 'ref': ref}


class _FieldNameStack:
    __slots__ = ('_stack',)

    def __init__(self) -> None:
        self._stack: List[str] = []

    @contextmanager
    def push(self, field_name: str) -> Generator[None, None, None]:
        self._stack.append(field_name)
        try:
            yield
        finally:
            self._stack.pop()

    def get(self) -> Optional[str]:
        if self._stack:
            return self._stack[-1]
        else:
            return None


class _ModelTypeStack:
    __slots__ = ('_stack',)

    def __init__(self) -> None:
        self._stack: List[Any] = []

    @contextmanager
    def push(self, type_obj: Any) -> Generator[None, None, None]:
        self._stack.append(type_obj)
        try:
            yield
        finally:
            self._stack.pop()

    def get(self) -> Optional[Any]:
        if self._stack:
            return self._stack[-1]
        else:
            return None