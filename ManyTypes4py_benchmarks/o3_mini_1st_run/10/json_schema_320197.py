from __future__ import annotations
import dataclasses
import inspect
import math
import os
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Sequence
from copy import deepcopy
from enum import Enum
from re import Pattern
from typing import TYPE_CHECKING, Annotated, Any, Callable, Dict, List, Literal, NewType, Tuple, TypeVar, Union, cast, overload
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import TypeAlias, assert_never, deprecated, final
from pydantic.warnings import PydanticDeprecatedSince26, PydanticDeprecatedSince29
from ._internal import _config, _core_metadata, _core_utils, _decorators, _internal_dataclass, _mock_val_ser, _schema_generation_shared, _typing_extra
from .annotated_handlers import GetJsonSchemaHandler
from .config import JsonDict, JsonValue
from .errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError, PydanticUserError

if TYPE_CHECKING:
    from . import ConfigDict
    from ._internal._core_utils import CoreSchemaField, CoreSchemaOrField
    from ._internal._dataclasses import PydanticDataclass
    from ._internal._schema_generation_shared import GetJsonSchemaFunction
    from .main import BaseModel

CoreSchemaOrFieldType: TypeAlias = Literal[core_schema.CoreSchemaType, core_schema.CoreSchemaFieldType]
JsonSchemaValue: TypeAlias = Dict[str, Any]
JsonSchemaMode: TypeAlias = Literal['validation', 'serialization']
_MODE_TITLE_MAPPING: Dict[str, str] = {'validation': 'Input', 'serialization': 'Output'}
JsonSchemaWarningKind: TypeAlias = Literal['skipped-choice', 'non-serializable-default', 'skipped-discriminator']

class PydanticJsonSchemaWarning(UserWarning):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

DEFAULT_REF_TEMPLATE: str = '#/$defs/{model}'
CoreRef = NewType('CoreRef', str)
DefsRef = NewType('DefsRef', str)
JsonRef = NewType('JsonRef', str)
CoreModeRef = Tuple[CoreRef, JsonSchemaMode]
JsonSchemaKeyT = TypeVar('JsonSchemaKeyT', bound=Hashable)

@dataclasses.dataclass(**_internal_dataclass.slots_true)
class _DefinitionsRemapping:
    defs_remapping: Dict[DefsRef, DefsRef]
    json_remapping: Dict[JsonRef, JsonRef]

    @staticmethod
    def from_prioritized_choices(
        prioritized_choices: Dict[DefsRef, List[DefsRef]],
        defs_to_json: Dict[DefsRef, JsonRef],
        definitions: Dict[DefsRef, JsonSchemaValue]
    ) -> _DefinitionsRemapping:
        copied_definitions: Dict[DefsRef, JsonSchemaValue] = deepcopy(definitions)
        definitions_schema: JsonSchemaValue = {'$defs': copied_definitions}
        for _iter in range(100):
            schemas_for_alternatives: Dict[DefsRef, List[JsonSchemaValue]] = defaultdict(list)
            for defs_ref in copied_definitions:
                alternatives: List[DefsRef] = prioritized_choices[defs_ref]
                for alternative in alternatives:
                    schemas_for_alternatives[alternative].append(copied_definitions[defs_ref])
            for defs_ref in schemas_for_alternatives:
                schemas_for_alternatives[defs_ref] = _deduplicate_schemas(schemas_for_alternatives[defs_ref])
            defs_remapping: Dict[DefsRef, DefsRef] = {}
            json_remapping: Dict[JsonRef, JsonRef] = {}
            for original_defs_ref in definitions:
                alternatives: List[DefsRef] = prioritized_choices[original_defs_ref]
                remapped_defs_ref: DefsRef = next((x for x in alternatives if len(schemas_for_alternatives[x]) == 1), alternatives[0])
                defs_remapping[original_defs_ref] = remapped_defs_ref
                json_remapping[defs_to_json[original_defs_ref]] = defs_to_json[remapped_defs_ref]
            remapping = _DefinitionsRemapping(defs_remapping, json_remapping)
            new_definitions_schema: JsonSchemaValue = remapping.remap_json_schema({'$defs': copied_definitions})
            if definitions_schema == new_definitions_schema:
                return remapping
            definitions_schema = new_definitions_schema
        raise PydanticInvalidForJsonSchema('Failed to simplify the JSON schema definitions')

    def remap_defs_ref(self, ref: DefsRef) -> DefsRef:
        return self.defs_remapping.get(ref, ref)

    def remap_json_ref(self, ref: JsonRef) -> JsonRef:
        return self.json_remapping.get(ref, ref)

    def remap_json_schema(self, schema: Any) -> Any:
        if isinstance(schema, str):
            return self.remap_json_ref(JsonRef(schema))
        elif isinstance(schema, list):
            return [self.remap_json_schema(item) for item in schema]
        elif isinstance(schema, dict):
            for key, value in schema.items():
                if key == '$ref' and isinstance(value, str):
                    schema['$ref'] = self.remap_json_ref(JsonRef(value))
                elif key == '$defs':
                    schema['$defs'] = {self.remap_defs_ref(DefsRef(key)): self.remap_json_schema(value) for key, value in schema['$defs'].items()}
                else:
                    schema[key] = self.remap_json_schema(value)
        return schema

class GenerateJsonSchema:
    schema_dialect: str = 'https://json-schema.org/draft/2020-12/schema'
    ignored_warning_kinds: set[str] = {'skipped-choice'}

    def __init__(self, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE) -> None:
        self.by_alias: bool = by_alias
        self.ref_template: str = ref_template
        self.core_to_json_refs: Dict[CoreModeRef, JsonRef] = {}
        self.core_to_defs_refs: Dict[CoreModeRef, DefsRef] = {}
        self.defs_to_core_refs: Dict[DefsRef, CoreModeRef] = {}
        self.json_to_defs_refs: Dict[JsonRef, DefsRef] = {}
        self.definitions: Dict[DefsRef, JsonSchemaValue] = {}
        self._config_wrapper_stack: _config.ConfigWrapperStack = _config.ConfigWrapperStack(_config.ConfigWrapper({}))
        self._mode: JsonSchemaMode = 'validation'
        self._prioritized_defsref_choices: Dict[DefsRef, List[DefsRef]] = {}
        self._collision_counter: defaultdict[str, int] = defaultdict(int)
        self._collision_index: Dict[str, int] = {}
        self._schema_type_to_method: Dict[str, Callable[[Dict[str, Any]], JsonSchemaValue]] = self.build_schema_type_to_method()
        self._core_defs_invalid_for_json_schema: Dict[DefsRef, Exception] = {}
        self._used: bool = False

    @property
    def _config(self) -> _config.ConfigWrapper:
        return self._config_wrapper_stack.tail

    @property
    def mode(self) -> JsonSchemaMode:
        if self._config.json_schema_mode_override is not None:
            return self._config.json_schema_mode_override  # type: ignore
        else:
            return self._mode

    def build_schema_type_to_method(self) -> Dict[str, Callable[[Dict[str, Any]], JsonSchemaValue]]:
        mapping: Dict[str, Callable[[Dict[str, Any]], JsonSchemaValue]] = {}
        core_schema_types = _typing_extra.literal_values(CoreSchemaOrFieldType)
        for key in core_schema_types:
            method_name: str = f'{key.replace("-", "_")}_schema'
            try:
                mapping[key] = getattr(self, method_name)
            except AttributeError as e:
                if os.getenv('PYDANTIC_PRIVATE_ALLOW_UNHANDLED_SCHEMA_TYPES'):
                    continue
                raise TypeError(f'No method for generating JsonSchema for core_schema.type={key!r} (expected: {type(self).__name__}.{method_name})') from e
        return mapping

    def generate_definitions(self, inputs: Sequence[Tuple[Any, JsonSchemaMode, Dict[str, Any]]]) -> Tuple[Dict[Tuple[Any, JsonSchemaMode], JsonSchemaValue], Dict[DefsRef, JsonSchemaValue]]:
        if self._used:
            raise PydanticUserError(f'This JSON schema generator has already been used to generate a JSON schema. You must create a new instance of {type(self).__name__} to generate a new JSON schema.', code='json-schema-already-used')
        for _, mode, schema in inputs:
            self._mode = mode
            self.generate_inner(schema)
        definitions_remapping: _DefinitionsRemapping = self._build_definitions_remapping()
        json_schemas_map: Dict[Tuple[Any, JsonSchemaMode], JsonSchemaValue] = {}
        for key, mode, schema in inputs:
            self._mode = mode
            json_schema: JsonSchemaValue = self.generate_inner(schema)
            json_schemas_map[(key, mode)] = definitions_remapping.remap_json_schema(json_schema)
        json_schema: JsonSchemaValue = {'$defs': self.definitions}
        json_schema = definitions_remapping.remap_json_schema(json_schema)
        self._used = True
        return (json_schemas_map, self.sort(json_schema['$defs']))

    def generate(self, schema: Dict[str, Any], mode: JsonSchemaMode = 'validation') -> JsonSchemaValue:
        self._mode = mode
        if self._used:
            raise PydanticUserError(f'This JSON schema generator has already been used to generate a JSON schema. You must create a new instance of {type(self).__name__} to generate a new JSON schema.', code='json-schema-already-used')
        json_schema: JsonSchemaValue = self.generate_inner(schema)
        json_ref_counts: Counter[JsonRef] = self.get_json_ref_counts(json_schema)
        ref: Union[JsonRef, None] = cast(Union[JsonRef, None], json_schema.get('$ref'))
        while ref is not None:
            ref_json_schema: Union[JsonSchemaValue, None] = self.get_schema_from_definitions(ref)
            if json_ref_counts[ref] == 1 and ref_json_schema is not None and (len(json_schema) == 1):
                json_schema = ref_json_schema.copy()
                json_ref_counts[ref] -= 1
                ref = cast(Union[JsonRef, None], json_schema.get('$ref'))
            ref = None
        self._garbage_collect_definitions(json_schema)
        definitions_remapping: _DefinitionsRemapping = self._build_definitions_remapping()
        if self.definitions:
            json_schema['$defs'] = self.definitions
        json_schema = definitions_remapping.remap_json_schema(json_schema)
        self._used = True
        return self.sort(json_schema)

    def generate_inner(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        if 'ref' in schema:
            core_ref: CoreRef = CoreRef(schema['ref'])
            core_mode_ref: CoreModeRef = (core_ref, self.mode)
            if core_mode_ref in self.core_to_defs_refs and self.core_to_defs_refs[core_mode_ref] in self.definitions:
                return {'$ref': self.core_to_json_refs[core_mode_ref]}
        def populate_defs(core_schema: Dict[str, Any], json_schema: JsonSchemaValue) -> JsonSchemaValue:
            if 'ref' in core_schema:
                core_ref: CoreRef = CoreRef(core_schema['ref'])
                defs_ref, ref_json_schema = self.get_cache_defs_ref_schema(core_ref)
                json_ref: JsonRef = JsonRef(ref_json_schema['$ref'])
                if json_schema.get('$ref', None) != json_ref:
                    self.definitions[defs_ref] = json_schema
                    self._core_defs_invalid_for_json_schema.pop(defs_ref, None)
                json_schema = ref_json_schema
            return json_schema
        def handler_func(schema_or_field: Dict[str, Any]) -> JsonSchemaValue:
            json_schema: Union[JsonSchemaValue, None] = None
            if self.mode == 'serialization' and 'serialization' in schema_or_field:
                ser_schema: Dict[str, Any] = schema_or_field['serialization']
                json_schema = self.ser_schema(ser_schema)
                if json_schema is not None and ser_schema.get('when_used') in ('unless-none', 'json-unless-none') and (schema_or_field['type'] == 'nullable'):
                    json_schema = self.get_flattened_anyof([{'type': 'null'}, json_schema])
            if json_schema is None:
                if _core_utils.is_core_schema(schema_or_field) or _core_utils.is_core_schema_field(schema_or_field):
                    generate_for_schema_type: Callable[[Dict[str, Any]], JsonSchemaValue] = self._schema_type_to_method[schema_or_field['type']]
                    json_schema = generate_for_schema_type(schema_or_field)
                else:
                    raise TypeError(f'Unexpected schema type: schema={schema_or_field}')
            if _core_utils.is_core_schema(schema_or_field):
                json_schema = populate_defs(schema_or_field, json_schema)
            return json_schema
        current_handler: _schema_generation_shared.GenerateJsonSchemaHandler = _schema_generation_shared.GenerateJsonSchemaHandler(self, handler_func)
        metadata: _core_metadata.CoreMetadata = cast(_core_metadata.CoreMetadata, schema.get('metadata', {}))
        if (js_updates := metadata.get('pydantic_js_updates')):
            def js_updates_handler_func(schema_or_field: Dict[str, Any], current_handler=current_handler) -> JsonSchemaValue:
                json_schema = {**current_handler(schema_or_field), **js_updates}
                return json_schema
            current_handler = _schema_generation_shared.GenerateJsonSchemaHandler(self, js_updates_handler_func)
        if (js_extra := metadata.get('pydantic_js_extra')):
            def js_extra_handler_func(schema_or_field: Dict[str, Any], current_handler=current_handler) -> JsonSchemaValue:
                json_schema = current_handler(schema_or_field)
                if isinstance(js_extra, dict):
                    json_schema.update(to_jsonable_python(js_extra))
                elif callable(js_extra):
                    js_extra(json_schema)
                return json_schema
            current_handler = _schema_generation_shared.GenerateJsonSchemaHandler(self, js_extra_handler_func)
        for js_modify_function in metadata.get('pydantic_js_functions', ()):
            def new_handler_func(schema_or_field: Dict[str, Any], current_handler=current_handler, js_modify_function=js_modify_function) -> JsonSchemaValue:
                json_schema = js_modify_function(schema_or_field, current_handler)
                if _core_utils.is_core_schema(schema_or_field):
                    json_schema = populate_defs(schema_or_field, json_schema)
                original_schema = current_handler.resolve_ref_schema(json_schema)
                ref = json_schema.pop('$ref', None)
                if ref and json_schema:
                    original_schema.update(json_schema)
                return original_schema
            current_handler = _schema_generation_shared.GenerateJsonSchemaHandler(self, new_handler_func)
        for js_modify_function in metadata.get('pydantic_js_annotation_functions', ()):
            def new_handler_func(schema_or_field: Dict[str, Any], current_handler=current_handler, js_modify_function=js_modify_function) -> JsonSchemaValue:
                json_schema = js_modify_function(schema_or_field, current_handler)
                if _core_utils.is_core_schema(schema_or_field):
                    json_schema = populate_defs(schema_or_field, json_schema)
                return json_schema
            current_handler = _schema_generation_shared.GenerateJsonSchemaHandler(self, new_handler_func)
        json_schema: JsonSchemaValue = current_handler(schema)
        if _core_utils.is_core_schema(schema):
            json_schema = populate_defs(schema, json_schema)
        return json_schema

    def sort(self, value: Dict[str, Any], parent_key: Union[str, None] = None) -> Dict[str, Any]:
        sorted_dict: Dict[str, Any] = {}
        keys: Iterable[str] = value.keys()
        if parent_key not in ('properties', 'default'):
            keys = sorted(keys)
        for key in keys:
            sorted_dict[key] = self._sort_recursive(value[key], parent_key=key)
        return sorted_dict

    def _sort_recursive(self, value: Any, parent_key: Union[str, None] = None) -> Any:
        if isinstance(value, dict):
            sorted_dict: Dict[str, Any] = {}
            keys: Iterable[str] = value.keys()
            if parent_key not in ('properties', 'default'):
                keys = sorted(keys)
            for key in keys:
                sorted_dict[key] = self._sort_recursive(value[key], parent_key=key)
            return sorted_dict
        elif isinstance(value, list):
            sorted_list: List[Any] = []
            for item in value:
                sorted_list.append(self._sort_recursive(item, parent_key))
            return sorted_list
        else:
            return value

    def invalid_schema(self, schema: Dict[str, Any]) -> None:
        raise RuntimeError('Cannot generate schema for invalid_schema. This is a bug! Please report it.')

    def any_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return {}

    def none_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return {'type': 'null'}

    def bool_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return {'type': 'boolean'}

    def int_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        json_schema: JsonSchemaValue = {'type': 'integer'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.numeric)
        json_schema = {k: v for k, v in json_schema.items() if v not in {math.inf, -math.inf}}
        return json_schema

    def float_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        json_schema: JsonSchemaValue = {'type': 'number'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.numeric)
        json_schema = {k: v for k, v in json_schema.items() if v not in {math.inf, -math.inf}}
        return json_schema

    def decimal_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        json_schema: JsonSchemaValue = self.str_schema(core_schema.str_schema())
        if self.mode == 'validation':
            multiple_of = schema.get('multiple_of')
            le = schema.get('le')
            ge = schema.get('ge')
            lt = schema.get('lt')
            gt = schema.get('gt')
            json_schema = {'anyOf': [
                self.float_schema(core_schema.float_schema(
                    allow_inf_nan=schema.get('allow_inf_nan'),
                    multiple_of=None if multiple_of is None else float(multiple_of),
                    le=None if le is None else float(le),
                    ge=None if ge is None else float(ge),
                    lt=None if lt is None else float(lt),
                    gt=None if gt is None else float(gt)
                )),
                json_schema
            ]}
        return json_schema

    def str_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        json_schema: JsonSchemaValue = {'type': 'string'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.string)
        if isinstance(json_schema.get('pattern'), Pattern):
            json_schema['pattern'] = json_schema.get('pattern').pattern
        return json_schema

    def bytes_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        json_schema: JsonSchemaValue = {'type': 'string', 'format': 'base64url' if self._config.ser_json_bytes == 'base64' else 'binary'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.bytes)
        return json_schema

    def date_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return {'type': 'string', 'format': 'date'}

    def time_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return {'type': 'string', 'format': 'time'}

    def datetime_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return {'type': 'string', 'format': 'date-time'}

    def timedelta_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        if self._config.ser_json_timedelta == 'float':
            return {'type': 'number'}
        return {'type': 'string', 'format': 'duration'}

    def literal_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        expected = [to_jsonable_python(v.value if isinstance(v, Enum) else v) for v in schema['expected']]
        result: JsonSchemaValue = {}
        if len(expected) == 1:
            result['const'] = expected[0]
        else:
            result['enum'] = expected
        types = {type(e) for e in expected}
        if types == {str}:
            result['type'] = 'string'
        elif types == {int}:
            result['type'] = 'integer'
        elif types == {float}:
            result['type'] = 'number'
        elif types == {bool}:
            result['type'] = 'boolean'
        elif types == {list}:
            result['type'] = 'array'
        elif types == {type(None)}:
            result['type'] = 'null'
        return result

    def enum_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        enum_type = schema['cls']
        description: Union[str, None] = None if not enum_type.__doc__ else inspect.cleandoc(enum_type.__doc__)
        if description == 'An enumeration.':
            description = None
        result: JsonSchemaValue = {'title': enum_type.__name__, 'description': description}
        result = {k: v for k, v in result.items() if v is not None}
        expected = [to_jsonable_python(v.value) for v in schema['members']]
        result['enum'] = expected
        types = {type(e) for e in expected}
        if isinstance(enum_type, str) or types == {str}:
            result['type'] = 'string'
        elif isinstance(enum_type, int) or types == {int}:
            result['type'] = 'integer'
        elif isinstance(enum_type, float) or types == {float}:
            result['type'] = 'number'
        elif types == {bool}:
            result['type'] = 'boolean'
        elif types == {list}:
            result['type'] = 'array'
        return result

    def is_instance_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self.handle_invalid_for_json_schema(schema, f"core_schema.IsInstanceSchema ({schema['cls']})")

    def is_subclass_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return {}

    def callable_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self.handle_invalid_for_json_schema(schema, 'core_schema.CallableSchema')

    def list_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        items_schema: JsonSchemaValue = {} if 'items_schema' not in schema else self.generate_inner(schema['items_schema'])
        json_schema: JsonSchemaValue = {'type': 'array', 'items': items_schema}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    @deprecated('`tuple_positional_schema` is deprecated. Use `tuple_schema` instead.', category=None)
    @final
    def tuple_positional_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        warnings.warn('`tuple_positional_schema` is deprecated. Use `tuple_schema` instead.', PydanticDeprecatedSince26, stacklevel=2)
        return self.tuple_schema(schema)

    @deprecated('`tuple_variable_schema` is deprecated. Use `tuple_schema` instead.', category=None)
    @final
    def tuple_variable_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        warnings.warn('`tuple_variable_schema` is deprecated. Use `tuple_schema` instead.', PydanticDeprecatedSince26, stacklevel=2)
        return self.tuple_schema(schema)

    def tuple_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        json_schema: JsonSchemaValue = {'type': 'array'}
        if 'variadic_item_index' in schema:
            variadic_item_index: int = schema['variadic_item_index']
            if variadic_item_index > 0:
                json_schema['minItems'] = variadic_item_index
                json_schema['prefixItems'] = [self.generate_inner(item) for item in schema['items_schema'][:variadic_item_index]]
            if variadic_item_index + 1 == len(schema['items_schema']):
                json_schema['items'] = self.generate_inner(schema['items_schema'][variadic_item_index])
            else:
                json_schema['items'] = True
        else:
            prefixItems: List[JsonSchemaValue] = [self.generate_inner(item) for item in schema['items_schema']]
            if prefixItems:
                json_schema['prefixItems'] = prefixItems
            json_schema['minItems'] = len(prefixItems)
            json_schema['maxItems'] = len(prefixItems)
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    def set_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self._common_set_schema(schema)

    def frozenset_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self._common_set_schema(schema)

    def _common_set_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        items_schema: JsonSchemaValue = {} if 'items_schema' not in schema else self.generate_inner(schema['items_schema'])
        json_schema: JsonSchemaValue = {'type': 'array', 'uniqueItems': True, 'items': items_schema}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    def generator_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        items_schema: JsonSchemaValue = {} if 'items_schema' not in schema else self.generate_inner(schema['items_schema'])
        json_schema: JsonSchemaValue = {'type': 'array', 'items': items_schema}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    def dict_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        json_schema: JsonSchemaValue = {'type': 'object'}
        keys_schema: Dict[str, Any] = self.generate_inner(schema['keys_schema']).copy() if 'keys_schema' in schema else {}
        if '$ref' not in keys_schema:
            keys_pattern = keys_schema.pop('pattern', None)
            keys_schema.pop('title', None)
        else:
            keys_pattern = None
        values_schema: Dict[str, Any] = self.generate_inner(schema['values_schema']).copy() if 'values_schema' in schema else {}
        values_schema.pop('title', None)
        if values_schema or keys_pattern is not None:
            if keys_pattern is None:
                json_schema['additionalProperties'] = values_schema
            else:
                json_schema['patternProperties'] = {keys_pattern: values_schema}
        else:
            json_schema['additionalProperties'] = True
        if keys_schema.get('type') == 'string' and len(keys_schema) > 1 or '$ref' in keys_schema:
            keys_schema.pop('type', None)
            json_schema['propertyNames'] = keys_schema
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.object)
        return json_schema

    def function_before_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        if self.mode == 'validation' and (input_schema := schema.get('json_schema_input_schema')):
            return self.generate_inner(input_schema)
        return self.generate_inner(schema['schema'])

    def function_after_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self.generate_inner(schema['schema'])

    def function_plain_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        if self.mode == 'validation' and (input_schema := schema.get('json_schema_input_schema')):
            return self.generate_inner(input_schema)
        return self.handle_invalid_for_json_schema(schema, f"core_schema.PlainValidatorFunctionSchema ({schema['function']})")

    def function_wrap_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        if self.mode == 'validation' and (input_schema := schema.get('json_schema_input_schema')):
            return self.generate_inner(input_schema)
        return self.generate_inner(schema['schema'])

    def default_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        json_schema: JsonSchemaValue = self.generate_inner(schema['schema'])
        if 'default' not in schema:
            return json_schema
        default = schema['default']
        if self.mode == 'serialization' and (ser_schema := schema['schema'].get('serialization')) and (ser_func := ser_schema.get('function')) and (ser_schema.get('type') == 'function-plain') and (not ser_schema.get('info_arg')) and (not (default is None and ser_schema.get('when_used') in ('unless-none', 'json-unless-none'))):
            try:
                default = ser_func(default)
            except Exception:
                self.emit_warning('non-serializable-default', f'Unable to serialize value {default!r} with the plain serializer; excluding default from JSON schema')
                return json_schema
        try:
            encoded_default = self.encode_default(default)
        except pydantic_core.PydanticSerializationError:
            self.emit_warning('non-serializable-default', f'Default value {default} is not JSON serializable; excluding default from JSON schema')
            return json_schema
        json_schema['default'] = encoded_default
        return json_schema

    def nullable_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        null_schema: JsonSchemaValue = {'type': 'null'}
        inner_json_schema: JsonSchemaValue = self.generate_inner(schema['schema'])
        if inner_json_schema == null_schema:
            return null_schema
        else:
            return self.get_flattened_anyof([inner_json_schema, null_schema])

    def union_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        generated: List[JsonSchemaValue] = []
        choices = schema['choices']
        for choice in choices:
            choice_schema: Any = choice[0] if isinstance(choice, tuple) else choice
            try:
                generated.append(self.generate_inner(choice_schema))
            except PydanticOmit:
                continue
            except PydanticInvalidForJsonSchema as exc:
                self.emit_warning('skipped-choice', exc.message)
        if len(generated) == 1:
            return generated[0]
        return self.get_flattened_anyof(generated)

    def tagged_union_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        generated: Dict[str, JsonSchemaValue] = {}
        for k, v in schema['choices'].items():
            key_val: Any = k.value if isinstance(k, Enum) else k
            try:
                generated[str(key_val)] = self.generate_inner(v).copy()
            except PydanticOmit:
                continue
            except PydanticInvalidForJsonSchema as exc:
                self.emit_warning('skipped-choice', exc.message)
        one_of_choices: List[JsonSchemaValue] = _deduplicate_schemas(list(generated.values()))
        json_schema: JsonSchemaValue = {'oneOf': one_of_choices}
        openapi_discriminator: Union[str, None] = self._extract_discriminator(schema, one_of_choices)
        if openapi_discriminator is not None:
            json_schema['discriminator'] = {'propertyName': openapi_discriminator, 'mapping': {k: v.get('$ref', v) for k, v in generated.items()}}
        return json_schema

    def _extract_discriminator(self, schema: Dict[str, Any], one_of_choices: List[JsonSchemaValue]) -> Union[str, None]:
        openapi_discriminator: Union[str, None] = None
        if isinstance(schema['discriminator'], str):
            return schema['discriminator']
        if isinstance(schema['discriminator'], list):
            if len(schema['discriminator']) == 1 and isinstance(schema['discriminator'][0], str):
                return schema['discriminator'][0]
            for alias_path in schema['discriminator']:
                if not isinstance(alias_path, list):
                    break
                if len(alias_path) != 1:
                    continue
                alias = alias_path[0]
                if not isinstance(alias, str):
                    continue
                alias_is_present_on_all_choices = True
                for choice in one_of_choices:
                    try:
                        choice = self.resolve_ref_schema(choice)
                    except RuntimeError as exc:
                        self.emit_warning('skipped-discriminator', str(exc))
                        choice = {}
                    properties = choice.get('properties', {})
                    if not isinstance(properties, dict) or alias not in properties:
                        alias_is_present_on_all_choices = False
                        break
                if alias_is_present_on_all_choices:
                    openapi_discriminator = alias
                    break
        return openapi_discriminator

    def chain_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        step_index: int = 0 if self.mode == 'validation' else -1
        return self.generate_inner(schema['steps'][step_index])

    def lax_or_strict_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        use_strict = schema.get('strict', False)
        if use_strict:
            return self.generate_inner(schema['strict_schema'])
        else:
            return self.generate_inner(schema['lax_schema'])

    def json_or_python_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self.generate_inner(schema['json_schema'])

    def typed_dict_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        total: bool = schema.get('total', True)
        named_required_fields: List[Tuple[str, bool, Dict[str, Any]]] = [(name, self.field_is_required(field, total), field) for name, field in schema['fields'].items() if self.field_is_present(field)]
        if self.mode == 'serialization':
            named_required_fields.extend(self._name_required_computed_fields(schema.get('computed_fields', [])))
        cls = schema.get('cls')
        config = _get_typed_dict_config(cls)
        with self._config_wrapper_stack.push(config):
            json_schema: JsonSchemaValue = self._named_required_fields_schema(named_required_fields)
        if cls is not None:
            self._update_class_schema(json_schema, cls, config)
        else:
            extra = config.get('extra')
            if extra == 'forbid':
                json_schema['additionalProperties'] = False
            elif extra == 'allow':
                json_schema['additionalProperties'] = True
        return json_schema

    @staticmethod
    def _name_required_computed_fields(computed_fields: List[Dict[str, Any]]) -> List[Tuple[str, bool, Dict[str, Any]]]:
        return [(field['property_name'], True, field) for field in computed_fields]

    def _named_required_fields_schema(self, named_required_fields: List[Tuple[str, bool, Dict[str, Any]]]) -> JsonSchemaValue:
        properties: Dict[str, JsonSchemaValue] = {}
        required_fields: List[str] = []
        for name, required, field in named_required_fields:
            if self.by_alias:
                name = self._get_alias_name(field, name)
            try:
                field_json_schema: JsonSchemaValue = self.generate_inner(field).copy()
            except PydanticOmit:
                continue
            if 'title' not in field_json_schema and self.field_title_should_be_set(field):
                title: str = self.get_title_from_name(name)
                field_json_schema['title'] = title
            field_json_schema = self.handle_ref_overrides(field_json_schema)
            properties[name] = field_json_schema
            if required:
                required_fields.append(name)
        json_schema: JsonSchemaValue = {'type': 'object', 'properties': properties}
        if required_fields:
            json_schema['required'] = required_fields
        return json_schema

    def _get_alias_name(self, field: Dict[str, Any], name: str) -> str:
        if field['type'] == 'computed-field':
            alias = field.get('alias', name)
        elif self.mode == 'validation':
            alias = field.get('validation_alias', name)
        else:
            alias = field.get('serialization_alias', name)
        if isinstance(alias, str):
            name = alias
        elif isinstance(alias, list):
            for path in alias:
                if isinstance(path, list) and len(path) == 1 and isinstance(path[0], str):
                    name = path[0]
                    break
        else:
            assert_never(alias)
        return name

    def typed_dict_field_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self.generate_inner(schema['schema'])

    def dataclass_field_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self.generate_inner(schema['schema'])

    def model_field_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self.generate_inner(schema['schema'])

    def computed_field_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self.generate_inner(schema['return_schema'])

    def model_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        from .main import BaseModel
        cls: type[BaseModel] = cast(type[BaseModel], schema['cls'])
        config = cls.model_config
        with self._config_wrapper_stack.push(config):
            json_schema: JsonSchemaValue = self.generate_inner(schema['schema'])
        self._update_class_schema(json_schema, cls, config)
        return json_schema

    def _update_class_schema(self, json_schema: JsonSchemaValue, cls: type, config: Dict[str, Any]) -> None:
        from .main import BaseModel
        from .root_model import RootModel
        if (config_title := config.get('title')) is not None:
            json_schema.setdefault('title', config_title)
        elif (model_title_generator := config.get('model_title_generator')):
            title = model_title_generator(cls)
            if not isinstance(title, str):
                raise TypeError(f'model_title_generator {model_title_generator} must return str, not {title.__class__}')
            json_schema.setdefault('title', title)
        if 'title' not in json_schema:
            json_schema['title'] = cls.__name__
        docstring: Union[str, None] = None if cls is BaseModel or dataclasses.is_dataclass(cls) else cls.__doc__
        if docstring:
            json_schema.setdefault('description', inspect.cleandoc(docstring))
        elif issubclass(cls, RootModel) and (root_description := cls.__pydantic_fields__['root'].description):
            json_schema.setdefault('description', root_description)
        extra = config.get('extra')
        if 'additionalProperties' not in json_schema:
            if extra == 'allow':
                json_schema['additionalProperties'] = True
            elif extra == 'forbid':
                json_schema['additionalProperties'] = False
        json_schema_extra = config.get('json_schema_extra')
        if issubclass(cls, BaseModel) and cls.__pydantic_root_model__:
            root_json_schema_extra = cls.model_fields['root'].json_schema_extra
            if json_schema_extra and root_json_schema_extra:
                raise ValueError('"model_config[\'json_schema_extra\']" and "Field.json_schema_extra" on "RootModel.root" field must not be set simultaneously')
            if root_json_schema_extra:
                json_schema_extra = root_json_schema_extra
        if isinstance(json_schema_extra, (staticmethod, classmethod)):
            json_schema_extra = json_schema_extra.__get__(cls)
        if isinstance(json_schema_extra, dict):
            json_schema.update(json_schema_extra)
        elif callable(json_schema_extra):
            if len(inspect.signature(json_schema_extra).parameters) > 1:
                json_schema_extra(json_schema, cls)
            else:
                json_schema_extra(json_schema)
        elif json_schema_extra is not None:
            raise ValueError(f"model_config['json_schema_extra']={json_schema_extra} should be a dict, callable, or None")
        if hasattr(cls, '__deprecated__'):
            json_schema['deprecated'] = True

    def resolve_ref_schema(self, json_schema: JsonSchemaValue) -> JsonSchemaValue:
        while '$ref' in json_schema:
            ref = json_schema['$ref']
            schema_to_update = self.get_schema_from_definitions(JsonRef(ref))
            if schema_to_update is None:
                raise RuntimeError(f'Cannot update undefined schema for $ref={ref}')
            json_schema = schema_to_update
        return json_schema

    def model_fields_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        named_required_fields: List[Tuple[str, bool, Dict[str, Any]]] = [(name, self.field_is_required(field, total=True), field) for name, field in schema['fields'].items() if self.field_is_present(field)]
        if self.mode == 'serialization':
            named_required_fields.extend(self._name_required_computed_fields(schema.get('computed_fields', [])))
        json_schema: JsonSchemaValue = self._named_required_fields_schema(named_required_fields)
        extras_schema = schema.get('extras_schema', None)
        if extras_schema is not None:
            schema_to_update: JsonSchemaValue = self.resolve_ref_schema(json_schema)
            schema_to_update['additionalProperties'] = self.generate_inner(extras_schema)
        return json_schema

    def field_is_present(self, field: Dict[str, Any]) -> bool:
        if self.mode == 'serialization':
            return not field.get('serialization_exclude', False)
        elif self.mode == 'validation':
            return True
        else:
            assert_never(self.mode)
            return False

    def field_is_required(self, field: Dict[str, Any], total: bool) -> bool:
        if self.mode == 'serialization' and self._config.json_schema_serialization_defaults_required:
            return not field.get('serialization_exclude', False)
        elif field['type'] == 'typed-dict-field':
            return field.get('required', total)
        else:
            return field['schema']['type'] != 'default'

    def dataclass_args_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        named_required_fields: List[Tuple[str, bool, Dict[str, Any]]] = [(field['name'], self.field_is_required(field, total=True), field) for field in schema['fields'] if self.field_is_present(field)]
        if self.mode == 'serialization':
            named_required_fields.extend(self._name_required_computed_fields(schema.get('computed_fields', [])))
        return self._named_required_fields_schema(named_required_fields)

    def dataclass_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        from ._internal._dataclasses import is_builtin_dataclass
        cls = schema['cls']
        config = getattr(cls, '__pydantic_config__', cast('ConfigDict', {}))
        with self._config_wrapper_stack.push(config):
            json_schema: JsonSchemaValue = self.generate_inner(schema['schema']).copy()
        self._update_class_schema(json_schema, cls, config)
        if is_builtin_dataclass(cls):
            description = None
        else:
            description = None if cls.__doc__ is None else inspect.cleandoc(cls.__doc__)
        if description:
            json_schema['description'] = description
        return json_schema

    def arguments_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        prefer_positional = schema.get('metadata', {}).get('pydantic_js_prefer_positional_arguments')
        arguments = schema['arguments_schema']
        kw_only_arguments = [a for a in arguments if a.get('mode') == 'keyword_only']
        kw_or_p_arguments = [a for a in arguments if a.get('mode') in {'positional_or_keyword', None}]
        p_only_arguments = [a for a in arguments if a.get('mode') == 'positional_only']
        var_args_schema = schema.get('var_args_schema')
        var_kwargs_schema = schema.get('var_kwargs_schema')
        if prefer_positional:
            positional_possible = not kw_only_arguments and (not var_kwargs_schema)
            if positional_possible:
                return self.p_arguments_schema(p_only_arguments + kw_or_p_arguments, var_args_schema)
        keyword_possible = not p_only_arguments and (not var_args_schema)
        if keyword_possible:
            return self.kw_arguments_schema(kw_or_p_arguments + kw_only_arguments, var_kwargs_schema)
        if not prefer_positional:
            positional_possible = not kw_only_arguments and (not var_kwargs_schema)
            if positional_possible:
                return self.p_arguments_schema(p_only_arguments + kw_or_p_arguments, var_args_schema)
        raise PydanticInvalidForJsonSchema('Unable to generate JSON schema for arguments validator with positional-only and keyword-only arguments')

    def kw_arguments_schema(self, arguments: List[Dict[str, Any]], var_kwargs_schema: Any) -> JsonSchemaValue:
        properties: Dict[str, JsonSchemaValue] = {}
        required: List[str] = []
        for argument in arguments:
            name: str = self.get_argument_name(argument)
            argument_schema: JsonSchemaValue = self.generate_inner(argument['schema']).copy()
            argument_schema['title'] = self.get_title_from_name(name)
            properties[name] = argument_schema
            if argument['schema']['type'] != 'default':
                required.append(name)
        json_schema: JsonSchemaValue = {'type': 'object', 'properties': properties}
        if required:
            json_schema['required'] = required
        if var_kwargs_schema:
            additional_properties_schema = self.generate_inner(var_kwargs_schema)
            if additional_properties_schema:
                json_schema['additionalProperties'] = additional_properties_schema
        else:
            json_schema['additionalProperties'] = False
        return json_schema

    def p_arguments_schema(self, arguments: List[Dict[str, Any]], var_args_schema: Any) -> JsonSchemaValue:
        prefix_items: List[JsonSchemaValue] = []
        min_items: int = 0
        for argument in arguments:
            name: str = self.get_argument_name(argument)
            argument_schema: JsonSchemaValue = self.generate_inner(argument['schema']).copy()
            argument_schema['title'] = self.get_title_from_name(name)
            prefix_items.append(argument_schema)
            if argument['schema']['type'] != 'default':
                min_items += 1
        json_schema: JsonSchemaValue = {'type': 'array'}
        if prefix_items:
            json_schema['prefixItems'] = prefix_items
        if min_items:
            json_schema['minItems'] = min_items
        if var_args_schema:
            items_schema = self.generate_inner(var_args_schema)
            if items_schema:
                json_schema['items'] = items_schema
        else:
            json_schema['maxItems'] = len(prefix_items)
        return json_schema

    def get_argument_name(self, argument: Dict[str, Any]) -> str:
        name: str = argument['name']
        if self.by_alias:
            alias = argument.get('alias')
            if isinstance(alias, str):
                name = alias
        return name

    def call_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self.generate_inner(schema['arguments_schema'])

    def custom_error_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return self.generate_inner(schema['schema'])

    def json_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        content_core_schema: Dict[str, Any] = schema.get('schema') or core_schema.any_schema()
        content_json_schema: JsonSchemaValue = self.generate_inner(content_core_schema)
        if self.mode == 'validation':
            return {'type': 'string', 'contentMediaType': 'application/json', 'contentSchema': content_json_schema}
        else:
            return content_json_schema

    def url_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        json_schema: JsonSchemaValue = {'type': 'string', 'format': 'uri', 'minLength': 1}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.string)
        return json_schema

    def multi_host_url_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        json_schema: JsonSchemaValue = {'type': 'string', 'format': 'multi-host-uri', 'minLength': 1}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.string)
        return json_schema

    def uuid_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return {'type': 'string', 'format': 'uuid'}

    def definitions_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        for definition in schema['definitions']:
            try:
                self.generate_inner(definition)
            except PydanticInvalidForJsonSchema as e:
                core_ref = CoreRef(definition['ref'])
                self._core_defs_invalid_for_json_schema[self.get_defs_ref((core_ref, self.mode))] = e
                continue
        return self.generate_inner(schema['schema'])

    def definition_ref_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        core_ref: CoreRef = CoreRef(schema['schema_ref'])
        _, ref_json_schema = self.get_cache_defs_ref_schema(core_ref)
        return ref_json_schema

    def ser_schema(self, schema: Dict[str, Any]) -> Union[JsonSchemaValue, None]:
        schema_type: str = schema['type']
        if schema_type in ('function-plain', 'function-wrap'):
            return_schema = schema.get('return_schema')
            if return_schema is not None:
                return self.generate_inner(return_schema)
        elif schema['type'] in ('format', 'to-string'):
            return self.str_schema(core_schema.str_schema())
        elif schema['type'] == 'model':
            return self.generate_inner(schema['schema'])
        return None

    def complex_schema(self, schema: Dict[str, Any]) -> JsonSchemaValue:
        return {'type': 'string'}

    def get_title_from_name(self, name: str) -> str:
        return name.title().replace('_', ' ').strip()

    def field_title_should_be_set(self, schema: Dict[str, Any]) -> bool:
        if _core_utils.is_core_schema_field(schema):
            if schema['type'] == 'computed-field':
                field_schema = schema['return_schema']
            else:
                field_schema = schema['schema']
            return self.field_title_should_be_set(field_schema)
        elif _core_utils.is_core_schema(schema):
            if schema.get('ref'):
                return False
            if schema['type'] in {'default', 'nullable', 'definitions'}:
                return self.field_title_should_be_set(schema['schema'])
            if _core_utils.is_function_with_inner_schema(schema):
                return self.field_title_should_be_set(schema['schema'])
            if schema['type'] == 'definition-ref':
                return False
            return True
        else:
            raise PydanticInvalidForJsonSchema(f'Unexpected schema type: schema={schema}')

    def normalize_name(self, name: str) -> str:
        return re.sub('[^a-zA-Z0-9.\\-_]', '_', name).replace('.', '__')

    def get_defs_ref(self, core_mode_ref: CoreModeRef) -> DefsRef:
        core_ref, mode = core_mode_ref
        components = re.split('([\\][,])', core_ref)
        components = [x.rsplit(':', 1)[0] for x in components]
        core_ref_no_id = ''.join(components)
        components = [re.sub('(?:[^.[\\]]+\\.)+((?:[^.[\\]]+))', '\\1', x) for x in components]
        short_ref = ''.join(components)
        mode_title = _MODE_TITLE_MAPPING[mode]
        name = DefsRef(self.normalize_name(short_ref))
        name_mode = DefsRef(self.normalize_name(short_ref) + f'-{mode_title}')
        module_qualname = DefsRef(self.normalize_name(core_ref_no_id))
        module_qualname_mode = DefsRef(self.normalize_name(core_ref_no_id) + f'-{mode_title}')
        module_qualname_id = DefsRef(self.normalize_name(core_ref))
        occurrence_index = self._collision_index.get(module_qualname_id)
        if occurrence_index is None:
            self._collision_counter[module_qualname] += 1
            occurrence_index = self._collision_index[module_qualname_id] = self._collision_counter[module_qualname]
        module_qualname_occurrence = DefsRef(f'{module_qualname}__{occurrence_index}')
        module_qualname_occurrence_mode = DefsRef(f'{module_qualname_mode}__{occurrence_index}')
        self._prioritized_defsref_choices[module_qualname_occurrence_mode] = [name, name_mode, module_qualname, module_qualname_mode, module_qualname_occurrence, module_qualname_occurrence_mode]
        return module_qualname_occurrence_mode

    def get_cache_defs_ref_schema(self, core_ref: CoreRef) -> Tuple[DefsRef, JsonSchemaValue]:
        core_mode_ref: CoreModeRef = (core_ref, self.mode)
        maybe_defs_ref = self.core_to_defs_refs.get(core_mode_ref)
        if maybe_defs_ref is not None:
            json_ref = self.core_to_json_refs[core_mode_ref]
            return (maybe_defs_ref, {'$ref': json_ref})
        defs_ref: DefsRef = self.get_defs_ref(core_mode_ref)
        self.core_to_defs_refs[core_mode_ref] = defs_ref
        self.defs_to_core_refs[defs_ref] = core_mode_ref
        json_ref: JsonRef = JsonRef(self.ref_template.format(model=defs_ref))
        self.core_to_json_refs[core_mode_ref] = json_ref
        self.json_to_defs_refs[json_ref] = defs_ref
        ref_json_schema: JsonSchemaValue = {'$ref': json_ref}
        return (defs_ref, ref_json_schema)

    def handle_ref_overrides(self, json_schema: JsonSchemaValue) -> JsonSchemaValue:
        if '$ref' in json_schema:
            json_schema = json_schema.copy()
            referenced_json_schema = self.get_schema_from_definitions(JsonRef(json_schema['$ref']))
            if referenced_json_schema is None:
                return json_schema
            for k, v in list(json_schema.items()):
                if k == '$ref':
                    continue
                if k in referenced_json_schema and referenced_json_schema[k] == v:
                    del json_schema[k]
        return json_schema

    def get_schema_from_definitions(self, json_ref: JsonRef) -> Union[JsonSchemaValue, None]:
        try:
            def_ref = self.json_to_defs_refs[json_ref]
            if def_ref in self._core_defs_invalid_for_json_schema:
                raise self._core_defs_invalid_for_json_schema[def_ref]
            return self.definitions.get(def_ref, None)
        except KeyError:
            if json_ref.startswith(('http://', 'https://')):
                return None
            raise

    def encode_default(self, dft: Any) -> Any:
        from .type_adapter import TypeAdapter, _type_has_config
        config = self._config
        try:
            default = dft if _type_has_config(type(dft)) else TypeAdapter(type(dft), config=config.config_dict).dump_python(dft, mode='json')
        except PydanticSchemaGenerationError:
            raise pydantic_core.PydanticSerializationError(f'Unable to encode default value {dft}')
        return pydantic_core.to_jsonable_python(default, timedelta_mode=config.ser_json_timedelta, bytes_mode=config.ser_json_bytes)

    def update_with_validations(self, json_schema: JsonSchemaValue, core_schema: Dict[str, Any], mapping: Dict[str, str]) -> None:
        for core_key, json_schema_key in mapping.items():
            if core_key in core_schema:
                json_schema[json_schema_key] = core_schema[core_key]

    class ValidationsMapping:
        numeric: Dict[str, str] = {'multiple_of': 'multipleOf', 'le': 'maximum', 'ge': 'minimum', 'lt': 'exclusiveMaximum', 'gt': 'exclusiveMinimum'}
        bytes: Dict[str, str] = {'min_length': 'minLength', 'max_length': 'maxLength'}
        string: Dict[str, str] = {'min_length': 'minLength', 'max_length': 'maxLength', 'pattern': 'pattern'}
        array: Dict[str, str] = {'min_length': 'minItems', 'max_length': 'maxItems'}
        object: Dict[str, str] = {'min_length': 'minProperties', 'max_length': 'maxProperties'}

    def get_flattened_anyof(self, schemas: List[JsonSchemaValue]) -> JsonSchemaValue:
        members: List[JsonSchemaValue] = []
        for schema in schemas:
            if len(schema) == 1 and 'anyOf' in schema:
                members.extend(schema['anyOf'])
            else:
                members.append(schema)
        members = _deduplicate_schemas(members)
        if len(members) == 1:
            return members[0]
        return {'anyOf': members}

    def get_json_ref_counts(self, json_schema: JsonSchemaValue) -> Counter[JsonRef]:
        json_refs: Counter[JsonRef] = Counter()
        def _add_json_refs(schema: Any) -> None:
            if isinstance(schema, dict):
                if '$ref' in schema:
                    json_ref: JsonRef = JsonRef(schema['$ref'])
                    if not isinstance(json_ref, str):
                        return
                    already_visited = json_ref in json_refs
                    json_refs[json_ref] += 1
                    if already_visited:
                        return
                    try:
                        defs_ref = self.json_to_defs_refs[json_ref]
                        if defs_ref in self._core_defs_invalid_for_json_schema:
                            raise self._core_defs_invalid_for_json_schema[defs_ref]
                        _add_json_refs(self.definitions[defs_ref])
                    except KeyError:
                        if not json_ref.startswith(('http://', 'https://')):
                            raise
                for k, v in schema.items():
                    if k == 'examples' and isinstance(v, list):
                        continue
                    _add_json_refs(v)
            elif isinstance(schema, list):
                for v in schema:
                    _add_json_refs(v)
        _add_json_refs(json_schema)
        return json_refs

    def handle_invalid_for_json_schema(self, schema: Dict[str, Any], error_info: str) -> JsonSchemaValue:
        raise PydanticInvalidForJsonSchema(f'Cannot generate a JsonSchema for {error_info}')

    def emit_warning(self, kind: JsonSchemaWarningKind, detail: str) -> None:
        message: Union[str, None] = self.render_warning_message(kind, detail)
        if message is not None:
            warnings.warn(message, PydanticJsonSchemaWarning)

    def render_warning_message(self, kind: JsonSchemaWarningKind, detail: str) -> Union[str, None]:
        if kind in self.ignored_warning_kinds:
            return None
        return f'{detail} [{kind}]'

    def _build_definitions_remapping(self) -> _DefinitionsRemapping:
        defs_to_json: Dict[DefsRef, JsonRef] = {}
        for defs_refs in self._prioritized_defsref_choices.values():
            for defs_ref in defs_refs:
                json_ref: JsonRef = JsonRef(self.ref_template.format(model=defs_ref))
                defs_to_json[defs_ref] = json_ref
        return _DefinitionsRemapping.from_prioritized_choices(self._prioritized_defsref_choices, defs_to_json, self.definitions)

    def _garbage_collect_definitions(self, schema: JsonSchemaValue) -> None:
        visited_defs_refs: set[DefsRef] = set()
        unvisited_json_refs: set[JsonRef] = _get_all_json_refs(schema)
        while unvisited_json_refs:
            next_json_ref: JsonRef = unvisited_json_refs.pop()
            try:
                next_defs_ref = self.json_to_defs_refs[next_json_ref]
                if next_defs_ref in visited_defs_refs:
                    continue
                visited_defs_refs.add(next_defs_ref)
                unvisited_json_refs.update(_get_all_json_refs(self.definitions[next_defs_ref]))
            except KeyError:
                if not next_json_ref.startswith(('http://', 'https://')):
                    raise
        self.definitions = {k: v for k, v in self.definitions.items() if k in visited_defs_refs}

class WithJsonSchema:
    mode: Any = None
    def __get_pydantic_json_schema__(self, core_schema: Dict[str, Any], handler: Callable[[Dict[str, Any]], JsonSchemaValue]) -> JsonSchemaValue:
        mode = self.mode or handler.mode
        if mode != handler.mode:
            return handler(core_schema)
        if self.json_schema is None:
            raise PydanticOmit
        else:
            return self.json_schema.copy()
    def __hash__(self) -> int:
        return hash(type(self))

class Examples:
    @overload
    @deprecated('Using a dict for `examples` is deprecated since v2.9 and will be removed in v3.0. Use a list instead.')
    def __init__(self, examples: Dict[Any, Any], mode: Any = None) -> None: ...
    @overload
    def __init__(self, examples: List[Any], mode: Any = None) -> None: ...
    def __init__(self, examples: Union[Dict[Any, Any], List[Any]], mode: Any = None) -> None:
        if isinstance(examples, dict):
            warnings.warn('Using a dict for `examples` is deprecated, use a list instead.', PydanticDeprecatedSince29, stacklevel=2)
        self.examples: Union[Dict[Any, Any], List[Any]] = examples
        self.mode = mode
    def __get_pydantic_json_schema__(self, core_schema: Dict[str, Any], handler: Callable[[Dict[str, Any]], JsonSchemaValue]) -> JsonSchemaValue:
        mode = self.mode or handler.mode
        json_schema: JsonSchemaValue = handler(core_schema)
        if mode != handler.mode:
            return json_schema
        examples_val = json_schema.get('examples')
        if examples_val is None:
            json_schema['examples'] = to_jsonable_python(self.examples)
        if isinstance(examples_val, dict):
            if isinstance(self.examples, list):
                warnings.warn('Updating existing JSON Schema examples of type dict with examples of type list. Only the existing examples values will be retained. Note that dict support for examples is deprecated and will be removed in v3.0.', UserWarning)
                json_schema['examples'] = to_jsonable_python([ex for value in examples_val.values() for ex in value] + self.examples)
            else:
                json_schema['examples'] = to_jsonable_python({**examples_val, **self.examples})
        if isinstance(examples_val, list):
            if isinstance(self.examples, list):
                json_schema['examples'] = to_jsonable_python(examples_val + self.examples)
            elif isinstance(self.examples, dict):
                warnings.warn('Updating existing JSON Schema examples of type list with examples of type dict. Only the examples values will be retained. Note that dict support for examples is deprecated and will be removed in v3.0.', UserWarning)
                json_schema['examples'] = to_jsonable_python(examples_val + [ex for value in self.examples.values() for ex in value])
        return json_schema
    def __hash__(self) -> int:
        return hash(type(self))

def _deduplicate_schemas(schemas: List[JsonSchemaValue]) -> List[JsonSchemaValue]:
    return list({_make_json_hashable(schema): schema for schema in schemas}.values())

def _make_json_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted(((k, _make_json_hashable(v)) for k, v in value.items())))
    elif isinstance(value, list):
        return tuple((_make_json_hashable(v) for v in value))
    else:
        return value

@dataclasses.dataclass(**_internal_dataclass.slots_true)
class SkipJsonSchema:
    def __class_getitem__(cls, item: Any) -> Any:
        return Annotated[item, cls()]
    def __get_pydantic_json_schema__(self, core_schema: Dict[str, Any], handler: Callable[[Dict[str, Any]], JsonSchemaValue]) -> Any:
        raise PydanticOmit
    def __hash__(self) -> int:
        return hash(type(self))

def _get_typed_dict_config(cls: Any) -> Any:
    if cls is not None:
        try:
            return _decorators.get_attribute_from_bases(cls, '__pydantic_config__')
        except AttributeError:
            pass
    return {}

_HashableJsonValue = Union[int, float, str, bool, None, Tuple['_HashableJsonValue', ...], Tuple[Tuple[str, '_HashableJsonValue'], ...]]

def _get_all_json_refs(item: Any) -> set[JsonRef]:
    refs: set[JsonRef] = set()
    stack: List[Any] = [item]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key == 'examples' and isinstance(value, list):
                    continue
                if key == '$ref' and isinstance(value, str):
                    refs.add(JsonRef(value))
                elif isinstance(value, dict):
                    stack.append(value)
                elif isinstance(value, list):
                    stack.extend(value)
        elif isinstance(current, list):
            stack.extend(current)
    return refs

if TYPE_CHECKING:
    AnyType = Any
else:
    @dataclasses.dataclass(**_internal_dataclass.slots_true)
    class SkipJsonSchema:
        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, cls()]
        def __get_pydantic_json_schema__(self, core_schema: Dict[str, Any], handler: Callable[[Dict[str, Any]], JsonSchemaValue]) -> Any:
            raise PydanticOmit
        def __hash__(self) -> int:
            return hash(type(self))