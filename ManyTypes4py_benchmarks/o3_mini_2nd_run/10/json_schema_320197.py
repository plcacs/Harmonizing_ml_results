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
from typing import (
    Any,
    Callable,
    Dict,
    Iterable as TypingIterable,
    List,
    Mapping,
    Optional,
    Sequence as TypingSequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import Literal, NewType, overload, Annotated, final, deprecated, TypeAlias, assert_never
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
JsonSchemaValue: TypeAlias = dict[str, Any]
JsonSchemaMode: TypeAlias = Literal['validation', 'serialization']
_MODE_TITLE_MAPPING: Dict[JsonSchemaMode, str] = {'validation': 'Input', 'serialization': 'Output'}
JsonSchemaWarningKind: TypeAlias = Literal['skipped-choice', 'non-serializable-default', 'skipped-discriminator']

class PydanticJsonSchemaWarning(UserWarning):
    """
    This class is used to emit warnings produced during JSON schema generation.
    See the `GenerateJsonSchema.emit_warning` and `GenerateJsonSchema.render_warning_message`
    methods for more details; these can be overridden to control warning behavior.
    """

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
        prioritized_choices: Mapping[DefsRef, List[DefsRef]],
        defs_to_json: Mapping[DefsRef, JsonRef],
        definitions: JsonDict,
    ) -> _DefinitionsRemapping:
        copied_definitions: JsonDict = deepcopy(definitions)
        definitions_schema: JsonDict = {'$defs': copied_definitions}
        for _iter in range(100):
            schemas_for_alternatives: Dict[DefsRef, List[Any]] = defaultdict(list)
            for defs_ref in copied_definitions:
                alternatives: List[DefsRef] = prioritized_choices[DefsRef(defs_ref)]
                for alternative in alternatives:
                    schemas_for_alternatives[alternative].append(copied_definitions[defs_ref])
            for defs_ref in schemas_for_alternatives:
                schemas_for_alternatives[defs_ref] = _deduplicate_schemas(schemas_for_alternatives[defs_ref])
            defs_remapping: Dict[DefsRef, DefsRef] = {}
            json_remapping: Dict[JsonRef, JsonRef] = {}
            for original_defs_ref in definitions:
                alternatives: List[DefsRef] = prioritized_choices[DefsRef(original_defs_ref)]
                remapped_defs_ref: DefsRef = next((x for x in alternatives if len(schemas_for_alternatives[x]) == 1), alternatives[0])
                defs_remapping[DefsRef(original_defs_ref)] = remapped_defs_ref
                json_remapping[defs_to_json[DefsRef(original_defs_ref)]] = defs_to_json[remapped_defs_ref]
            remapping: _DefinitionsRemapping = _DefinitionsRemapping(defs_remapping, json_remapping)
            new_definitions_schema: JsonDict = remapping.remap_json_schema({'$defs': copied_definitions})
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
            for key, value in list(schema.items()):
                if key == '$ref' and isinstance(value, str):
                    schema['$ref'] = self.remap_json_ref(JsonRef(value))
                elif key == '$defs':
                    schema['$defs'] = {self.remap_defs_ref(DefsRef(k)): self.remap_json_schema(v) for k, v in schema['$defs'].items()}
                else:
                    schema[key] = self.remap_json_schema(value)
        return schema

class GenerateJsonSchema:
    schema_dialect: str = 'https://json-schema.org/draft/2020-12/schema'
    ignored_warning_kinds: set[str] = {'skipped-choice'}

    def __init__(self, *, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE) -> None:
        self.by_alias: bool = by_alias
        self.ref_template: str = ref_template
        self.core_to_json_refs: Dict[CoreModeRef, JsonRef] = {}
        self.core_to_defs_refs: Dict[CoreModeRef, DefsRef] = {}
        self.defs_to_core_refs: Dict[DefsRef, CoreModeRef] = {}
        self.json_to_defs_refs: Dict[JsonRef, DefsRef] = {}
        self.definitions: JsonDict = {}
        self._config_wrapper_stack = _config.ConfigWrapperStack(_config.ConfigWrapper({}))
        self._mode: JsonSchemaMode = 'validation'
        self._prioritized_defsref_choices: Dict[DefsRef, List[DefsRef]] = {}
        self._collision_counter: Dict[str, int] = defaultdict(int)
        self._collision_index: Dict[str, int] = {}
        self._schema_type_to_method: Dict[str, Callable[[dict[str, Any]], Any]] = self.build_schema_type_to_method()
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

    def build_schema_type_to_method(self) -> Dict[str, Callable[[dict[str, Any]], Any]]:
        mapping: Dict[str, Callable[[dict[str, Any]], Any]] = {}
        core_schema_types: List[str] = _typing_extra.literal_values(CoreSchemaOrFieldType)
        for key in core_schema_types:
            method_name: str = f'{key.replace("-", "_")}_schema'
            try:
                mapping[key] = getattr(self, method_name)
            except AttributeError as e:
                if os.getenv('PYDANTIC_PRIVATE_ALLOW_UNHANDLED_SCHEMA_TYPES'):
                    continue
                raise TypeError(
                    f'No method for generating JsonSchema for core_schema.type={key!r} (expected: {type(self).__name__}.{method_name})'
                ) from e
        return mapping

    def generate_definitions(self, inputs: TypingSequence[Tuple[JsonSchemaKeyT, JsonSchemaMode, CoreSchema]]) -> Tuple[Dict[Tuple[JsonSchemaKeyT, JsonSchemaMode], Any], JsonDict]:
        if self._used:
            raise PydanticUserError(
                f'This JSON schema generator has already been used to generate a JSON schema. You must create a new instance of {type(self).__name__} to generate a new JSON schema.',
                code='json-schema-already-used')
        for _, mode, schema in inputs:
            self._mode = mode
            self.generate_inner(schema)
        definitions_remapping: _DefinitionsRemapping = self._build_definitions_remapping()
        json_schemas_map: Dict[Tuple[JsonSchemaKeyT, JsonSchemaMode], Any] = {}
        for key, mode, schema in inputs:
            self._mode = mode
            json_schema: Any = self.generate_inner(schema)
            json_schemas_map[(key, mode)] = definitions_remapping.remap_json_schema(json_schema)
        json_schema: JsonDict = {'$defs': self.definitions}
        json_schema = definitions_remapping.remap_json_schema(json_schema)
        self._used = True
        return (json_schemas_map, self.sort(json_schema['$defs']))

    def generate(self, schema: CoreSchema, *, mode: JsonSchemaMode = 'validation') -> JsonDict:
        self._mode = mode
        if self._used:
            raise PydanticUserError(
                f'This JSON schema generator has already been used to generate a JSON schema. You must create a new instance of {type(self).__name__} to generate a new JSON schema.',
                code='json-schema-already-used')
        json_schema: JsonDict = self.generate_inner(schema)
        json_ref_counts: Counter[str] = self.get_json_ref_counts(json_schema)
        ref: Optional[JsonRef] = cast(Optional[JsonRef], json_schema.get('$ref'))
        while ref is not None:
            ref_json_schema: Optional[JsonDict] = self.get_schema_from_definitions(ref)
            if json_ref_counts[ref] == 1 and ref_json_schema is not None and (len(json_schema) == 1):
                json_schema = ref_json_schema.copy()
                json_ref_counts[ref] -= 1
                ref = cast(Optional[JsonRef], json_schema.get('$ref'))
            ref = None
        self._garbage_collect_definitions(json_schema)
        definitions_remapping: _DefinitionsRemapping = self._build_definitions_remapping()
        if self.definitions:
            json_schema['$defs'] = self.definitions
        json_schema = definitions_remapping.remap_json_schema(json_schema)
        self._used = True
        return self.sort(json_schema)

    def generate_inner(self, schema: CoreSchema) -> Any:
        if 'ref' in schema:
            core_ref: CoreRef = CoreRef(schema['ref'])
            core_mode_ref: CoreModeRef = (core_ref, self.mode)
            if core_mode_ref in self.core_to_defs_refs and self.core_to_defs_refs[core_mode_ref] in self.definitions:
                return {'$ref': self.core_to_json_refs[core_mode_ref]}

        def populate_defs(core_schema: dict[str, Any], json_schema: Any) -> Any:
            if 'ref' in core_schema:
                core_ref: CoreRef = CoreRef(core_schema['ref'])
                defs_ref, ref_json_schema = self.get_cache_defs_ref_schema(core_ref)
                json_ref: JsonRef = JsonRef(ref_json_schema['$ref'])
                if json_schema.get('$ref', None) != json_ref:
                    self.definitions[defs_ref] = json_schema
                    self._core_defs_invalid_for_json_schema.pop(defs_ref, None)
                json_schema = ref_json_schema
            return json_schema

        def handler_func(schema_or_field: dict[str, Any]) -> Any:
            json_schema: Optional[Any] = None
            if self.mode == 'serialization' and 'serialization' in schema_or_field:
                ser_schema: dict[str, Any] = schema_or_field['serialization']
                json_schema = self.ser_schema(ser_schema)
                if json_schema is not None and ser_schema.get('when_used') in ('unless-none', 'json-unless-none') and (schema_or_field['type'] == 'nullable'):
                    json_schema = self.get_flattened_anyof([{'type': 'null'}, json_schema])
            if json_schema is None:
                if _core_utils.is_core_schema(schema_or_field) or _core_utils.is_core_schema_field(schema_or_field):
                    generate_for_schema_type: Callable[[dict[str, Any]], Any] = self._schema_type_to_method[schema_or_field['type']]
                    json_schema = generate_for_schema_type(schema_or_field)
                else:
                    raise TypeError(f'Unexpected schema type: schema={schema_or_field}')
            if _core_utils.is_core_schema(schema_or_field):
                json_schema = populate_defs(schema_or_field, json_schema)
            return json_schema

        current_handler: _schema_generation_shared.GenerateJsonSchemaHandler = _schema_generation_shared.GenerateJsonSchemaHandler(self, handler_func)
        metadata: _core_metadata.CoreMetadata = cast(_core_metadata.CoreMetadata, schema.get('metadata', {}))
        if (js_updates := metadata.get('pydantic_js_updates')):
            def js_updates_handler_func(schema_or_field: dict[str, Any], current_handler=current_handler) -> Any:
                json_schema_local: Any = {**current_handler(schema_or_field), **js_updates}
                return json_schema_local
            current_handler = _schema_generation_shared.GenerateJsonSchemaHandler(self, js_updates_handler_func)
        if (js_extra := metadata.get('pydantic_js_extra')):
            def js_extra_handler_func(schema_or_field: dict[str, Any], current_handler=current_handler) -> Any:
                json_schema_local: Any = current_handler(schema_or_field)
                if isinstance(js_extra, dict):
                    json_schema_local.update(to_jsonable_python(js_extra))
                elif callable(js_extra):
                    js_extra(json_schema_local)
                return json_schema_local
            current_handler = _schema_generation_shared.GenerateJsonSchemaHandler(self, js_extra_handler_func)
        for js_modify_function in metadata.get('pydantic_js_functions', ()):
            def new_handler_func(schema_or_field: dict[str, Any], current_handler=current_handler, js_modify_function=js_modify_function) -> Any:
                json_schema_local: Any = js_modify_function(schema_or_field, current_handler)
                if _core_utils.is_core_schema(schema_or_field):
                    json_schema_local = populate_defs(schema_or_field, json_schema_local)
                original_schema: Any = current_handler.resolve_ref_schema(json_schema_local)
                ref_local: Optional[str] = json_schema_local.pop('$ref', None)
                if ref_local and json_schema_local:
                    original_schema.update(json_schema_local)
                return original_schema
            current_handler = _schema_generation_shared.GenerateJsonSchemaHandler(self, new_handler_func)
        for js_modify_function in metadata.get('pydantic_js_annotation_functions', ()):
            def new_handler_func(schema_or_field: dict[str, Any], current_handler=current_handler, js_modify_function=js_modify_function) -> Any:
                json_schema_local: Any = js_modify_function(schema_or_field, current_handler)
                if _core_utils.is_core_schema(schema_or_field):
                    json_schema_local = populate_defs(schema_or_field, json_schema_local)
                return json_schema_local
            current_handler = _schema_generation_shared.GenerateJsonSchemaHandler(self, new_handler_func)
        json_schema: Any = current_handler(schema)
        if _core_utils.is_core_schema(schema):
            json_schema = populate_defs(schema, json_schema)
        return json_schema

    def sort(self, value: Any, parent_key: Optional[str] = None) -> Any:
        sorted_dict: Dict[Any, Any] = {}
        keys: Any = value.keys()
        if parent_key not in ('properties', 'default'):
            keys = sorted(keys)
        for key in keys:
            sorted_dict[key] = self._sort_recursive(value[key], parent_key=key)
        return sorted_dict

    def _sort_recursive(self, value: Any, parent_key: Optional[str] = None) -> Any:
        if isinstance(value, dict):
            sorted_dict: Dict[Any, Any] = {}
            keys: Any = value.keys()
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

    def invalid_schema(self, schema: Any) -> Any:
        raise RuntimeError('Cannot generate schema for invalid_schema. This is a bug! Please report it.')

    def any_schema(self, schema: CoreSchema) -> JsonDict:
        return {}

    def none_schema(self, schema: CoreSchema) -> JsonDict:
        return {'type': 'null'}

    def bool_schema(self, schema: CoreSchema) -> JsonDict:
        return {'type': 'boolean'}

    def int_schema(self, schema: CoreSchema) -> JsonDict:
        json_schema: JsonDict = {'type': 'integer'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.numeric)
        json_schema = {k: v for k, v in json_schema.items() if v not in {math.inf, -math.inf}}
        return json_schema

    def float_schema(self, schema: CoreSchema) -> JsonDict:
        json_schema: JsonDict = {'type': 'number'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.numeric)
        json_schema = {k: v for k, v in json_schema.items() if v not in {math.inf, -math.inf}}
        return json_schema

    def decimal_schema(self, schema: CoreSchema) -> JsonDict:
        json_schema: JsonDict = self.str_schema(core_schema.str_schema())
        if self.mode == 'validation':
            multiple_of = schema.get('multiple_of')
            le = schema.get('le')
            ge = schema.get('ge')
            lt = schema.get('lt')
            gt = schema.get('gt')
            json_schema = {
                'anyOf': [
                    self.float_schema(core_schema.float_schema(
                        allow_inf_nan=schema.get('allow_inf_nan'),
                        multiple_of=None if multiple_of is None else float(multiple_of),
                        le=None if le is None else float(le),
                        ge=None if ge is None else float(ge),
                        lt=None if lt is None else float(lt),
                        gt=None if gt is None else float(gt))),
                    json_schema
                ]
            }
        return json_schema

    def str_schema(self, schema: CoreSchema) -> JsonDict:
        json_schema: JsonDict = {'type': 'string'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.string)
        if isinstance(json_schema.get('pattern'), Pattern):
            json_schema['pattern'] = json_schema.get('pattern').pattern
        return json_schema

    def bytes_schema(self, schema: CoreSchema) -> JsonDict:
        json_schema: JsonDict = {
            'type': 'string',
            'format': 'base64url' if self._config.ser_json_bytes == 'base64' else 'binary'
        }
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.bytes)
        return json_schema

    def date_schema(self, schema: CoreSchema) -> JsonDict:
        return {'type': 'string', 'format': 'date'}

    def time_schema(self, schema: CoreSchema) -> JsonDict:
        return {'type': 'string', 'format': 'time'}

    def datetime_schema(self, schema: CoreSchema) -> JsonDict:
        return {'type': 'string', 'format': 'date-time'}

    def timedelta_schema(self, schema: CoreSchema) -> JsonDict:
        if self._config.ser_json_timedelta == 'float':
            return {'type': 'number'}
        return {'type': 'string', 'format': 'duration'}

    def literal_schema(self, schema: CoreSchema) -> JsonDict:
        expected: List[Any] = [to_jsonable_python(v.value if isinstance(v, Enum) else v) for v in schema['expected']]
        result: JsonDict = {}
        if len(expected) == 1:
            result['const'] = expected[0]
        else:
            result['enum'] = expected
        types_set = {type(e) for e in expected}
        if types_set == {str}:
            result['type'] = 'string'
        elif types_set == {int}:
            result['type'] = 'integer'
        elif types_set == {float}:
            result['type'] = 'number'
        elif types_set == {bool}:
            result['type'] = 'boolean'
        elif types_set == {list}:
            result['type'] = 'array'
        elif types_set == {type(None)}:
            result['type'] = 'null'
        return result

    def enum_schema(self, schema: CoreSchema) -> JsonDict:
        enum_type = schema['cls']
        description: Optional[str] = None if not enum_type.__doc__ else inspect.cleandoc(enum_type.__doc__)
        if description == 'An enumeration.':
            description = None
        result: JsonDict = {'title': enum_type.__name__, 'description': description}
        result = {k: v for k, v in result.items() if v is not None}
        expected: List[Any] = [to_jsonable_python(v.value) for v in schema['members']]
        result['enum'] = expected
        types_set = {type(e) for e in expected}
        if isinstance(enum_type, str) or types_set == {str}:
            result['type'] = 'string'
        elif isinstance(enum_type, int) or types_set == {int}:
            result['type'] = 'integer'
        elif isinstance(enum_type, float) or types_set == {float}:
            result['type'] = 'number'
        elif types_set == {bool}:
            result['type'] = 'boolean'
        elif types_set == {list}:
            result['type'] = 'array'
        return result

    def is_instance_schema(self, schema: CoreSchema) -> JsonDict:
        return self.handle_invalid_for_json_schema(schema, f'core_schema.IsInstanceSchema ({schema["cls"]})')

    def is_subclass_schema(self, schema: CoreSchema) -> JsonDict:
        return {}

    def callable_schema(self, schema: CoreSchema) -> JsonDict:
        return self.handle_invalid_for_json_schema(schema, 'core_schema.CallableSchema')

    def list_schema(self, schema: CoreSchema) -> JsonDict:
        items_schema: Any = {} if 'items_schema' not in schema else self.generate_inner(schema['items_schema'])
        json_schema: JsonDict = {'type': 'array', 'items': items_schema}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    @deprecated('`tuple_positional_schema` is deprecated. Use `tuple_schema` instead.', category=None)
    @final
    def tuple_positional_schema(self, schema: CoreSchema) -> JsonDict:
        warnings.warn('`tuple_positional_schema` is deprecated. Use `tuple_schema` instead.', PydanticDeprecatedSince26, stacklevel=2)
        return self.tuple_schema(schema)

    @deprecated('`tuple_variable_schema` is deprecated. Use `tuple_schema` instead.', category=None)
    @final
    def tuple_variable_schema(self, schema: CoreSchema) -> JsonDict:
        warnings.warn('`tuple_variable_schema` is deprecated. Use `tuple_schema` instead.', PydanticDeprecatedSince26, stacklevel=2)
        return self.tuple_schema(schema)

    def tuple_schema(self, schema: CoreSchema) -> JsonDict:
        json_schema: JsonDict = {'type': 'array'}
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
            prefixItems = [self.generate_inner(item) for item in schema['items_schema']]
            if prefixItems:
                json_schema['prefixItems'] = prefixItems
            json_schema['minItems'] = len(prefixItems)
            json_schema['maxItems'] = len(prefixItems)
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    def set_schema(self, schema: CoreSchema) -> JsonDict:
        return self._common_set_schema(schema)

    def frozenset_schema(self, schema: CoreSchema) -> JsonDict:
        return self._common_set_schema(schema)

    def _common_set_schema(self, schema: CoreSchema) -> JsonDict:
        items_schema: Any = {} if 'items_schema' not in schema else self.generate_inner(schema['items_schema'])
        json_schema: JsonDict = {'type': 'array', 'uniqueItems': True, 'items': items_schema}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    def generator_schema(self, schema: CoreSchema) -> JsonDict:
        items_schema: Any = {} if 'items_schema' not in schema else self.generate_inner(schema['items_schema'])
        json_schema: JsonDict = {'type': 'array', 'items': items_schema}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    def dict_schema(self, schema: CoreSchema) -> JsonDict:
        json_schema: JsonDict = {'type': 'object'}
        keys_schema: JsonDict = self.generate_inner(schema['keys_schema']).copy() if 'keys_schema' in schema else {}
        if '$ref' not in keys_schema:
            keys_pattern = keys_schema.pop('pattern', None)
            keys_schema.pop('title', None)
        else:
            keys_pattern = None
        values_schema: JsonDict = self.generate_inner(schema['values_schema']).copy() if 'values_schema' in schema else {}
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

    def function_before_schema(self, schema: CoreSchema) -> JsonDict:
        if self.mode == 'validation' and (input_schema := schema.get('json_schema_input_schema')):
            return self.generate_inner(input_schema)
        return self.generate_inner(schema['schema'])

    def function_after_schema(self, schema: CoreSchema) -> JsonDict:
        return self.generate_inner(schema['schema'])

    def function_plain_schema(self, schema: CoreSchema) -> JsonDict:
        if self.mode == 'validation' and (input_schema := schema.get('json_schema_input_schema')):
            return self.generate_inner(input_schema)
        return self.handle_invalid_for_json_schema(schema, f'core_schema.PlainValidatorFunctionSchema ({schema["function"]})')

    def function_wrap_schema(self, schema: CoreSchema) -> JsonDict:
        if self.mode == 'validation' and (input_schema := schema.get('json_schema_input_schema')):
            return self.generate_inner(input_schema)
        return self.generate_inner(schema['schema'])

    def default_schema(self, schema: CoreSchema) -> JsonDict:
        json_schema: JsonDict = self.generate_inner(schema['schema'])
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

    def nullable_schema(self, schema: CoreSchema) -> JsonDict:
        null_schema: JsonDict = {'type': 'null'}
        inner_json_schema: Any = self.generate_inner(schema['schema'])
        if inner_json_schema == null_schema:
            return null_schema
        else:
            return self.get_flattened_anyof([inner_json_schema, null_schema])

    def union_schema(self, schema: CoreSchema) -> JsonDict:
        generated: List[Any] = []
        choices = schema['choices']
        for choice in choices:
            choice_schema = choice[0] if isinstance(choice, tuple) else choice
            try:
                generated.append(self.generate_inner(choice_schema))
            except PydanticOmit:
                continue
            except PydanticInvalidForJsonSchema as exc:
                self.emit_warning('skipped-choice', exc.message)
        if len(generated) == 1:
            return generated[0]
        return self.get_flattened_anyof(generated)

    def tagged_union_schema(self, schema: CoreSchema) -> JsonDict:
        generated: Dict[str, Any] = {}
        for k, v in schema['choices'].items():
            if isinstance(k, Enum):
                k = k.value
            try:
                generated[str(k)] = self.generate_inner(v).copy()
            except PydanticOmit:
                continue
            except PydanticInvalidForJsonSchema as exc:
                self.emit_warning('skipped-choice', exc.message)
        one_of_choices: List[JsonDict] = _deduplicate_schemas(list(generated.values()))
        json_schema: JsonDict = {'oneOf': one_of_choices}
        openapi_discriminator: Optional[str] = self._extract_discriminator(schema, one_of_choices)
        if openapi_discriminator is not None:
            json_schema['discriminator'] = {'propertyName': openapi_discriminator, 'mapping': {k: v.get('$ref', v) for k, v in generated.items()}}
        return json_schema

    def _extract_discriminator(self, schema: CoreSchema, one_of_choices: TypingIterable[JsonDict]) -> Optional[str]:
        openapi_discriminator: Optional[str] = None
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

    def chain_schema(self, schema: CoreSchema) -> JsonDict:
        step_index: int = 0 if self.mode == 'validation' else -1
        return self.generate_inner(schema['steps'][step_index])

    def lax_or_strict_schema(self, schema: CoreSchema) -> JsonDict:
        use_strict: bool = schema.get('strict', False)
        if use_strict:
            return self.generate_inner(schema['strict_schema'])
        else:
            return self.generate_inner(schema['lax_schema'])

    def json_or_python_schema(self, schema: CoreSchema) -> JsonDict:
        return self.generate_inner(schema['json_schema'])

    def typed_dict_schema(self, schema: CoreSchema) -> JsonDict:
        total: bool = schema.get('total', True)
        named_required_fields: List[Tuple[str, bool, Any]] = [
            (name, self.field_is_required(field, total), field)
            for name, field in schema['fields'].items() if self.field_is_present(field)
        ]
        if self.mode == 'serialization':
            named_required_fields.extend(self._name_required_computed_fields(schema.get('computed_fields', [])))
        cls = schema.get('cls')
        config: Dict[str, Any] = _get_typed_dict_config(cls)
        with self._config_wrapper_stack.push(config):
            json_schema: JsonDict = self._named_required_fields_schema(named_required_fields)
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
    def _name_required_computed_fields(computed_fields: list[ComputedField]) -> list[Tuple[str, bool, Any]]:
        return [(field['property_name'], True, field) for field in computed_fields]

    def _named_required_fields_schema(self, named_required_fields: list[Tuple[str, bool, Any]]) -> JsonDict:
        properties: Dict[str, Any] = {}
        required_fields: List[str] = []
        for name, required, field in named_required_fields:
            if self.by_alias:
                name = self._get_alias_name(field, name)
            try:
                field_json_schema: Any = self.generate_inner(field).copy()
            except PydanticOmit:
                continue
            if 'title' not in field_json_schema and self.field_title_should_be_set(field):
                title: str = self.get_title_from_name(name)
                field_json_schema['title'] = title
            field_json_schema = self.handle_ref_overrides(field_json_schema)
            properties[name] = field_json_schema
            if required:
                required_fields.append(name)
        json_schema: JsonDict = {'type': 'object', 'properties': properties}
        if required_fields:
            json_schema['required'] = required_fields
        return json_schema

    def _get_alias_name(self, field: dict[str, Any], name: str) -> str:
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

    def typed_dict_field_schema(self, schema: CoreSchema) -> JsonDict:
        return self.generate_inner(schema['schema'])

    def dataclass_field_schema(self, schema: CoreSchema) -> JsonDict:
        return self.generate_inner(schema['schema'])

    def model_field_schema(self, schema: CoreSchema) -> JsonDict:
        return self.generate_inner(schema['schema'])

    def computed_field_schema(self, schema: CoreSchema) -> JsonDict:
        return self.generate_inner(schema['return_schema'])

    def model_schema(self, schema: CoreSchema) -> JsonDict:
        from .main import BaseModel
        cls: type[BaseModel] = cast(type[BaseModel], schema['cls'])
        config: Dict[str, Any] = cls.model_config
        with self._config_wrapper_stack.push(config):
            json_schema: JsonDict = self.generate_inner(schema['schema'])
        self._update_class_schema(json_schema, cls, config)
        return json_schema

    def _update_class_schema(self, json_schema: JsonDict, cls: type, config: dict[str, Any]) -> None:
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
        docstring: Optional[str] = None if cls is BaseModel or dataclasses.is_dataclass(cls) else cls.__doc__
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

    def resolve_ref_schema(self, json_schema: JsonDict) -> JsonDict:
        while '$ref' in json_schema:
            ref: str = json_schema['$ref']
            schema_to_update: Optional[JsonDict] = self.get_schema_from_definitions(JsonRef(ref))
            if schema_to_update is None:
                raise RuntimeError(f'Cannot update undefined schema for $ref={ref}')
            json_schema = schema_to_update
        return json_schema

    def model_fields_schema(self, schema: CoreSchema) -> JsonDict:
        named_required_fields: List[Tuple[str, bool, Any]] = [
            (name, self.field_is_required(field, total=True), field) for name, field in schema['fields'].items() if self.field_is_present(field)
        ]
        if self.mode == 'serialization':
            named_required_fields.extend(self._name_required_computed_fields(schema.get('computed_fields', [])))
        json_schema: JsonDict = self._named_required_fields_schema(named_required_fields)
        extras_schema: Optional[Any] = schema.get('extras_schema', None)
        if extras_schema is not None:
            schema_to_update: JsonDict = self.resolve_ref_schema(json_schema)
            schema_to_update['additionalProperties'] = self.generate_inner(extras_schema)
        return json_schema

    def field_is_present(self, field: dict[str, Any]) -> bool:
        if self.mode == 'serialization':
            return not field.get('serialization_exclude', False)
        elif self.mode == 'validation':
            return True
        else:
            assert_never(self.mode)

    def field_is_required(self, field: dict[str, Any], total: bool) -> bool:
        if self.mode == 'serialization' and self._config.json_schema_serialization_defaults_required:
            return not field.get('serialization_exclude', False)
        elif field['type'] == 'typed-dict-field':
            return field.get('required', total)
        else:
            return field['schema']['type'] != 'default'

    def dataclass_args_schema(self, schema: CoreSchema) -> JsonDict:
        named_required_fields: List[Tuple[str, bool, Any]] = [
            (field['name'], self.field_is_required(field, total=True), field) for field in schema['fields'] if self.field_is_present(field)
        ]
        if self.mode == 'serialization':
            named_required_fields.extend(self._name_required_computed_fields(schema.get('computed_fields', [])))
        return self._named_required_fields_schema(named_required_fields)

    def dataclass_schema(self, schema: CoreSchema) -> JsonDict:
        from ._internal._dataclasses import is_builtin_dataclass
        cls = schema['cls']
        config: Dict[str, Any] = getattr(cls, '__pydantic_config__', cast('ConfigDict', {}))
        with self._config_wrapper_stack.push(config):
            json_schema: JsonDict = self.generate_inner(schema['schema']).copy()
        self._update_class_schema(json_schema, cls, config)
        if is_builtin_dataclass(cls):
            description = None
        else:
            description = None if cls.__doc__ is None else inspect.cleandoc(cls.__doc__)
        if description:
            json_schema['description'] = description
        return json_schema

    def arguments_schema(self, schema: CoreSchema) -> JsonDict:
        prefer_positional = schema.get('metadata', {}).get('pydantic_js_prefer_positional_arguments')
        arguments: List[dict[str, Any]] = schema['arguments_schema']
        kw_only_arguments = [a for a in arguments if a.get('mode') == 'keyword_only']
        kw_or_p_arguments = [a for a in arguments if a.get('mode') in {'positional_or_keyword', None}]
        p_only_arguments = [a for a in arguments if a.get('mode') == 'positional_only']
        var_args_schema: Optional[CoreSchema] = schema.get('var_args_schema')
        var_kwargs_schema: Optional[CoreSchema] = schema.get('var_kwargs_schema')
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

    def kw_arguments_schema(self, arguments: List[dict[str, Any]], var_kwargs_schema: Optional[CoreSchema]) -> JsonDict:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for argument in arguments:
            name: str = self.get_argument_name(argument)
            argument_schema: JsonDict = self.generate_inner(argument['schema']).copy()
            argument_schema['title'] = self.get_title_from_name(name)
            properties[name] = argument_schema
            if argument['schema']['type'] != 'default':
                required.append(name)
        json_schema: JsonDict = {'type': 'object', 'properties': properties}
        if required:
            json_schema['required'] = required
        if var_kwargs_schema:
            additional_properties_schema = self.generate_inner(var_kwargs_schema)
            if additional_properties_schema:
                json_schema['additionalProperties'] = additional_properties_schema
        else:
            json_schema['additionalProperties'] = False
        return json_schema

    def p_arguments_schema(self, arguments: List[dict[str, Any]], var_args_schema: Optional[CoreSchema]) -> JsonDict:
        prefix_items: List[Any] = []
        min_items: int = 0
        for argument in arguments:
            name: str = self.get_argument_name(argument)
            argument_schema: JsonDict = self.generate_inner(argument['schema']).copy()
            argument_schema['title'] = self.get_title_from_name(name)
            prefix_items.append(argument_schema)
            if argument['schema']['type'] != 'default':
                min_items += 1
        json_schema: JsonDict = {'type': 'array'}
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

    def get_argument_name(self, argument: dict[str, Any]) -> str:
        name: str = argument['name']
        if self.by_alias:
            alias = argument.get('alias')
            if isinstance(alias, str):
                name = alias
        return name

    def call_schema(self, schema: CoreSchema) -> JsonDict:
        return self.generate_inner(schema['arguments_schema'])

    def custom_error_schema(self, schema: CoreSchema) -> JsonDict:
        return self.generate_inner(schema['schema'])

    def json_schema(self, schema: CoreSchema) -> JsonDict:
        content_core_schema: CoreSchema = schema.get('schema') or core_schema.any_schema()
        content_json_schema: Any = self.generate_inner(content_core_schema)
        if self.mode == 'validation':
            return {'type': 'string', 'contentMediaType': 'application/json', 'contentSchema': content_json_schema}
        else:
            return content_json_schema

    def url_schema(self, schema: CoreSchema) -> JsonDict:
        json_schema: JsonDict = {'type': 'string', 'format': 'uri', 'minLength': 1}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.string)
        return json_schema

    def multi_host_url_schema(self, schema: CoreSchema) -> JsonDict:
        json_schema: JsonDict = {'type': 'string', 'format': 'multi-host-uri', 'minLength': 1}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.string)
        return json_schema

    def uuid_schema(self, schema: CoreSchema) -> JsonDict:
        return {'type': 'string', 'format': 'uuid'}

    def definitions_schema(self, schema: CoreSchema) -> JsonDict:
        for definition in schema['definitions']:
            try:
                self.generate_inner(definition)
            except PydanticInvalidForJsonSchema as e:
                core_ref: CoreRef = CoreRef(definition['ref'])
                self._core_defs_invalid_for_json_schema[self.get_defs_ref((core_ref, self.mode))] = e
                continue
        return self.generate_inner(schema['schema'])

    def definition_ref_schema(self, schema: CoreSchema) -> JsonDict:
        core_ref: CoreRef = CoreRef(schema['schema_ref'])
        _, ref_json_schema = self.get_cache_defs_ref_schema(core_ref)
        return ref_json_schema

    def ser_schema(self, schema: CoreSchema) -> Optional[JsonDict]:
        schema_type: str = schema['type']
        if schema_type in ('function-plain', 'function-wrap'):
            return_schema = schema.get('return_schema')
            if return_schema is not None:
                return self.generate_inner(return_schema)
        elif schema_type in ('format', 'to-string'):
            return self.str_schema(core_schema.str_schema())
        elif schema['type'] == 'model':
            return self.generate_inner(schema['schema'])
        return None

    def complex_schema(self, schema: CoreSchema) -> JsonDict:
        return {'type': 'string'}

    def get_title_from_name(self, name: str) -> str:
        return name.title().replace('_', ' ').strip()

    def field_title_should_be_set(self, schema: Any) -> bool:
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
        components = re.split(r'([\][,])', core_ref)
        components = [x.rsplit(':', 1)[0] for x in components]
        core_ref_no_id = ''.join(components)
        components = [re.sub(r'(?:[^.[\]]+\.)+((?:[^.[\]]+))', r'\1', x) for x in components]
        short_ref = ''.join(components)
        mode_title = _MODE_TITLE_MAPPING[mode]
        name: DefsRef = DefsRef(self.normalize_name(short_ref))
        name_mode: DefsRef = DefsRef(self.normalize_name(short_ref) + f'-{mode_title}')
        module_qualname: DefsRef = DefsRef(self.normalize_name(core_ref_no_id))
        module_qualname_mode: DefsRef = DefsRef(self.normalize_name(core_ref_no_id) + f'-{mode_title}')
        module_qualname_id: DefsRef = DefsRef(self.normalize_name(core_ref))
        occurrence_index = self._collision_index.get(module_qualname_id)
        if occurrence_index is None:
            self._collision_counter[module_qualname] += 1
            occurrence_index = self._collision_index[module_qualname_id] = self._collision_counter[module_qualname]
        module_qualname_occurrence: DefsRef = DefsRef(f'{module_qualname}__{occurrence_index}')
        module_qualname_occurrence_mode: DefsRef = DefsRef(f'{module_qualname_mode}__{occurrence_index}')
        self._prioritized_defsref_choices[module_qualname_occurrence_mode] = [
            name,
            name_mode,
            module_qualname,
            module_qualname_mode,
            module_qualname_occurrence,
            module_qualname_occurrence_mode,
        ]
        return module_qualname_occurrence_mode

    def get_cache_defs_ref_schema(self, core_ref: CoreRef) -> Tuple[DefsRef, JsonDict]:
        core_mode_ref: CoreModeRef = (core_ref, self.mode)
        maybe_defs_ref: Optional[DefsRef] = self.core_to_defs_refs.get(core_mode_ref)
        if maybe_defs_ref is not None:
            json_ref: JsonRef = self.core_to_json_refs[core_mode_ref]
            return (maybe_defs_ref, {'$ref': json_ref})
        defs_ref: DefsRef = self.get_defs_ref(core_mode_ref)
        self.core_to_defs_refs[core_mode_ref] = defs_ref
        self.defs_to_core_refs[defs_ref] = core_mode_ref
        json_ref: JsonRef = JsonRef(self.ref_template.format(model=defs_ref))
        self.core_to_json_refs[core_mode_ref] = json_ref
        self.json_to_defs_refs[json_ref] = defs_ref
        ref_json_schema: JsonDict = {'$ref': json_ref}
        return (defs_ref, ref_json_schema)

    def handle_ref_overrides(self, json_schema: JsonDict) -> JsonDict:
        if '$ref' in json_schema:
            json_schema = json_schema.copy()
            referenced_json_schema: Optional[JsonDict] = self.get_schema_from_definitions(JsonRef(json_schema['$ref']))
            if referenced_json_schema is None:
                return json_schema
            for k, v in list(json_schema.items()):
                if k == '$ref':
                    continue
                if k in referenced_json_schema and referenced_json_schema[k] == v:
                    del json_schema[k]
        return json_schema

    def get_schema_from_definitions(self, json_ref: JsonRef) -> Optional[JsonDict]:
        try:
            def_ref: DefsRef = self.json_to_defs_refs[json_ref]
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

    def update_with_validations(self, json_schema: JsonDict, core_schema: dict[str, Any], mapping: Dict[str, str]) -> None:
        for core_key, json_schema_key in mapping.items():
            if core_key in core_schema:
                json_schema[json_schema_key] = core_schema[core_key]

    class ValidationsMapping:
        numeric: Dict[str, str] = {'multiple_of': 'multipleOf', 'le': 'maximum', 'ge': 'minimum', 'lt': 'exclusiveMaximum', 'gt': 'exclusiveMinimum'}
        bytes: Dict[str, str] = {'min_length': 'minLength', 'max_length': 'maxLength'}
        string: Dict[str, str] = {'min_length': 'minLength', 'max_length': 'maxLength', 'pattern': 'pattern'}
        array: Dict[str, str] = {'min_length': 'minItems', 'max_length': 'maxItems'}
        object: Dict[str, str] = {'min_length': 'minProperties', 'max_length': 'maxProperties'}

    def get_flattened_anyof(self, schemas: TypingIterable[JsonDict]) -> JsonDict:
        members: List[JsonDict] = []
        for schema in schemas:
            if len(schema) == 1 and 'anyOf' in schema:
                members.extend(schema['anyOf'])
            else:
                members.append(schema)
        members = _deduplicate_schemas(members)
        if len(members) == 1:
            return members[0]
        return {'anyOf': members}

    def get_json_ref_counts(self, json_schema: JsonDict) -> Counter[str]:
        json_refs: Counter[str] = Counter()

        def _add_json_refs(schema: Any) -> None:
            if isinstance(schema, dict):
                if '$ref' in schema:
                    json_ref_local: JsonRef = JsonRef(schema['$ref'])
                    if not isinstance(json_ref_local, str):
                        return
                    already_visited = json_ref_local in json_refs
                    json_refs[json_ref_local] += 1
                    if already_visited:
                        return
                    try:
                        defs_ref = self.json_to_defs_refs[json_ref_local]
                        if defs_ref in self._core_defs_invalid_for_json_schema:
                            raise self._core_defs_invalid_for_json_schema[defs_ref]
                        _add_json_refs(self.definitions[defs_ref])
                    except KeyError:
                        if not json_ref_local.startswith(('http://', 'https://')):
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

    def handle_invalid_for_json_schema(self, schema: Any, error_info: str) -> Any:
        raise PydanticInvalidForJsonSchema(f'Cannot generate a JsonSchema for {error_info}')

    def emit_warning(self, kind: str, detail: str) -> None:
        message: Optional[str] = self.render_warning_message(kind, detail)
        if message is not None:
            warnings.warn(message, PydanticJsonSchemaWarning)

    def render_warning_message(self, kind: str, detail: str) -> Optional[str]:
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

    def _garbage_collect_definitions(self, schema: JsonDict) -> None:
        visited_defs_refs: set[DefsRef] = set()
        unvisited_json_refs: set[JsonRef] = _get_all_json_refs(schema)
        while unvisited_json_refs:
            next_json_ref: JsonRef = unvisited_json_refs.pop()
            try:
                next_defs_ref: DefsRef = self.json_to_defs_refs[next_json_ref]
                if next_defs_ref in visited_defs_refs:
                    continue
                visited_defs_refs.add(next_defs_ref)
                unvisited_json_refs.update(_get_all_json_refs(self.definitions[next_defs_ref]))
            except KeyError:
                if not next_json_ref.startswith(('http://', 'https://')):
                    raise
        self.definitions = {k: v for k, v in self.definitions.items() if k in visited_defs_refs}

def model_json_schema(cls: Any, *, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE, schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema, mode: JsonSchemaMode = 'validation') -> JsonDict:
    from .main import BaseModel
    schema_generator_instance: GenerateJsonSchema = schema_generator(by_alias=by_alias, ref_template=ref_template)
    if isinstance(cls.__pydantic_core_schema__, _mock_val_ser.MockCoreSchema):
        cls.__pydantic_core_schema__.rebuild()
    if cls is BaseModel:
        raise AttributeError('model_json_schema() must be called on a subclass of BaseModel, not BaseModel itself.')
    assert not isinstance(cls.__pydantic_core_schema__, _mock_val_ser.MockCoreSchema), 'this is a bug! please report it'
    return schema_generator_instance.generate(cls.__pydantic_core_schema__, mode=mode)

def models_json_schema(models: TypingSequence[Tuple[Any, str]], *, by_alias: bool = True, title: Optional[str] = None, description: Optional[str] = None, ref_template: str = DEFAULT_REF_TEMPLATE, schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema) -> Tuple[Dict[Tuple[Any, str], Any], JsonDict]:
    for cls, _ in models:
        if isinstance(cls.__pydantic_core_schema__, _mock_val_ser.MockCoreSchema):
            cls.__pydantic_core_schema__.rebuild()
    instance: GenerateJsonSchema = schema_generator(by_alias=by_alias, ref_template=ref_template)
    inputs: List[Tuple[Any, str, CoreSchema]] = [(m, mode, m.__pydantic_core_schema__) for m, mode in models]
    json_schemas_map, definitions = instance.generate_definitions(inputs)
    json_schema: JsonDict = {}
    if definitions:
        json_schema['$defs'] = definitions
    if title:
        json_schema['title'] = title
    if description:
        json_schema['description'] = description
    return (json_schemas_map, json_schema)

_HashableJsonValue = Union[int, float, str, bool, None, Tuple['_HashableJsonValue', ...], Tuple[Tuple[str, '_HashableJsonValue'], ...]]

def _deduplicate_schemas(schemas: TypingIterable[Any]) -> List[Any]:
    return list({_make_json_hashable(schema): schema for schema in schemas}.values())

def _make_json_hashable(value: Any) -> _HashableJsonValue:
    if isinstance(value, dict):
        return tuple(sorted(((k, _make_json_hashable(v)) for k, v in value.items())))
    elif isinstance(value, list):
        return tuple((_make_json_hashable(v) for v in value))
    else:
        return value

@dataclasses.dataclass(**_internal_dataclass.slots_true)
class WithJsonSchema:
    mode: Optional[Any] = None
    json_schema: Optional[JsonDict] = None

    def __get_pydantic_json_schema__(self, core_schema: Any, handler: Callable[[Any], Any]) -> Any:
        mode = self.mode or handler.mode
        if mode != handler.mode:
            return handler(core_schema)
        if self.json_schema is None:
            raise PydanticOmit
        else:
            return self.json_schema.copy()

    def __hash__(self) -> int:
        return hash(type(self.mode))

class Examples:
    @overload
    @deprecated('Using a dict for `examples` is deprecated since v2.9 and will be removed in v3.0. Use a list instead.')
    def __init__(self, examples: dict[Any, Any], mode: Optional[Any] = None) -> None:
        ...

    @overload
    def __init__(self, examples: list[Any], mode: Optional[Any] = None) -> None:
        ...

    def __init__(self, examples: Union[dict[Any, Any], list[Any]], mode: Optional[Any] = None) -> None:
        if isinstance(examples, dict):
            warnings.warn('Using a dict for `examples` is deprecated, use a list instead.', PydanticDeprecatedSince29, stacklevel=2)
        self.examples: Union[dict[Any, Any], list[Any]] = examples
        self.mode: Optional[Any] = mode

    def __get_pydantic_json_schema__(self, core_schema: Any, handler: Callable[[Any], Any]) -> Any:
        mode = self.mode or handler.mode
        json_schema: JsonDict = handler(core_schema)
        if mode != handler.mode:
            return json_schema
        examples_existing = json_schema.get('examples')
        if examples_existing is None:
            json_schema['examples'] = to_jsonable_python(self.examples)
        if isinstance(examples_existing, dict):
            if isinstance(self.examples, list):
                warnings.warn('Updating existing JSON Schema examples of type dict with examples of type list. Only the existing examples values will be retained. Note that dict support for examples is deprecated and will be removed in v3.0.', UserWarning)
                json_schema['examples'] = to_jsonable_python([ex for value in examples_existing.values() for ex in value] + self.examples)
            else:
                json_schema['examples'] = to_jsonable_python({**examples_existing, **self.examples})
        if isinstance(examples_existing, list):
            if isinstance(self.examples, list):
                json_schema['examples'] = to_jsonable_python(examples_existing + self.examples)
            elif isinstance(self.examples, dict):
                warnings.warn('Updating existing JSON Schema examples of type list with examples of type dict. Only the examples values will be retained. Note that dict support for examples is deprecated and will be removed in v3.0.', UserWarning)
                json_schema['examples'] = to_jsonable_python(examples_existing + [ex for value in self.examples.values() for ex in value])
        return json_schema

    def __hash__(self) -> int:
        return hash(type(self.mode))

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

AnyType = TypeVar('AnyType')
if TYPE_CHECKING:
    SkipJsonSchema = Annotated[AnyType, ...]
else:
    @dataclasses.dataclass(**_internal_dataclass.slots_true)
    class SkipJsonSchema:
        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, cls()]

        def __get_pydantic_json_schema__(self, core_schema: Any, handler: Callable[[Any], Any]) -> Any:
            raise PydanticOmit

        def __hash__(self) -> int:
            return hash(type(self))

def _get_typed_dict_config(cls: Optional[type]) -> dict[str, Any]:
    if cls is not None:
        try:
            return _decorators.get_attribute_from_bases(cls, '__pydantic_config__')
        except AttributeError:
            pass
    return {}