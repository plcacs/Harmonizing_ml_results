"""!!! abstract "Usage Documentation"
    [JSON Schema](../concepts/json_schema.md)

The `json_schema` module contains classes and functions to allow the way [JSON Schema](https://json-schema.org/)
is generated to be customized.

In general you shouldn't need to use this module directly; instead, you can use
[`BaseModel.model_json_schema`][pydantic.BaseModel.model_json_schema] and
[`TypeAdapter.json_schema`][pydantic.TypeAdapter.json_schema].
"""
from __future__ import annotations as _annotations
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
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NewType,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import TypeAlias, assert_never, deprecated, final
from pydantic.warnings import PydanticDeprecatedSince26, PydanticDeprecatedSince29
from ._internal import (
    _config,
    _core_metadata,
    _core_utils,
    _decorators,
    _internal_dataclass,
    _mock_val_ser,
    _schema_generation_shared,
    _typing_extra,
)
from .annotated_handlers import GetJsonSchemaHandler
from .config import JsonDict, JsonValue
from .errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError, PydanticUserError

if TYPE_CHECKING:
    from . import ConfigDict
    from ._internal._core_utils import CoreSchemaField, CoreSchemaOrField
    from ._internal._dataclasses import PydanticDataclass
    from ._internal._schema_generation_shared import GetJsonSchemaFunction
    from .main import BaseModel

CoreSchemaOrFieldType = Literal[
    core_schema.CoreSchemaType, core_schema.CoreSchemaFieldType
]
'\nA type alias for defined schema types that represents a union of\n`core_schema.CoreSchemaType` and\n`core_schema.CoreSchemaFieldType`.\n'

JsonSchemaValue = Dict[str, Any]
'\nA type alias for a JSON schema value. This is a dictionary of string keys to arbitrary JSON values.\n'

JsonSchemaMode = Literal['validation', 'serialization']
"\nA type alias that represents the mode of a JSON schema; either 'validation' or 'serialization'.\n\nFor some types, the inputs to validation differ from the outputs of serialization. For example,\ncomputed fields will only be present when serializing, and should not be provided when\nvalidating. This flag provides a way to indicate whether you want the JSON schema required\nfor validation inputs, or that will be matched by serialization outputs.\n"

_MODE_TITLE_MAPPING = {'validation': 'Input', 'serialization': 'Output'}

JsonSchemaWarningKind = Literal[
    'skipped-choice', 'non-serializable-default', 'skipped-discriminator'
]
'\nA type alias representing the kinds of warnings that can be emitted during JSON schema generation.\n\nSee [`GenerateJsonSchema.render_warning_message`][pydantic.json_schema.GenerateJsonSchema.render_warning_message]\nfor more details.\n'


class PydanticJsonSchemaWarning(UserWarning):
    """This class is used to emit warnings produced during JSON schema generation.
    See the [`GenerateJsonSchema.emit_warning`][pydantic.json_schema.GenerateJsonSchema.emit_warning] and
    [`GenerateJsonSchema.render_warning_message`][pydantic.json_schema.GenerateJsonSchema.render_warning_message]
    methods for more details; these can be overridden to control warning behavior.
    """


DEFAULT_REF_TEMPLATE = '#/$defs/{model}'
'The default format string used to generate reference names.'

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
        definitions: Dict[DefsRef, JsonSchemaValue],
    ) -> _DefinitionsRemapping:
        """
        This function should produce a remapping that replaces complex DefsRef with the simpler ones from the
        prioritized_choices such that applying the name remapping would result in an equivalent JSON schema.
        """
        copied_definitions = deepcopy(definitions)
        definitions_schema = {'$defs': copied_definitions}
        for _iter in range(100):
            schemas_for_alternatives = defaultdict(list)
            for defs_ref in copied_definitions:
                alternatives = prioritized_choices[defs_ref]
                for alternative in alternatives:
                    schemas_for_alternatives[alternative].append(
                        copied_definitions[defs_ref]
                    )
            for defs_ref in schemas_for_alternatives:
                schemas_for_alternatives[defs_ref] = _deduplicate_schemas(
                    schemas_for_alternatives[defs_ref]
                )
            defs_remapping = {}
            json_remapping = {}
            for original_defs_ref in definitions:
                alternatives = prioritized_choices[original_defs_ref]
                remapped_defs_ref = next(
                    (x for x in alternatives if len(schemas_for_alternatives[x]) == 1)
                )
                defs_remapping[original_defs_ref] = remapped_defs_ref
                json_remapping[defs_to_json[original_defs_ref]] = defs_to_json[
                    remapped_defs_ref
                ]
            remapping = _DefinitionsRemapping(defs_remapping, json_remapping)
            new_definitions_schema = remapping.remap_json_schema(
                {'$defs': copied_definitions}
            )
            if definitions_schema == new_definitions_schema:
                return remapping
            definitions_schema = new_definitions_schema
        raise PydanticInvalidForJsonSchema(
            'Failed to simplify the JSON schema definitions'
        )

    def remap_defs_ref(self, ref: DefsRef) -> DefsRef:
        return self.defs_remapping.get(ref, ref)

    def remap_json_ref(self, ref: JsonRef) -> JsonRef:
        return self.json_remapping.get(ref, ref)

    def remap_json_schema(self, schema: JsonSchemaValue) -> JsonSchemaValue:
        """
        Recursively update the JSON schema replacing all $refs
        """
        if isinstance(schema, str):
            return self.remap_json_ref(JsonRef(schema))
        elif isinstance(schema, list):
            return [self.remap_json_schema(item) for item in schema]
        elif isinstance(schema, dict):
            for key, value in schema.items():
                if key == '$ref' and isinstance(value, str):
                    schema['$ref'] = self.remap_json_ref(JsonRef(value))
                elif key == '$defs':
                    schema['$defs'] = {
                        self.remap_defs_ref(DefsRef(key)): self.remap_json_schema(value)
                        for key, value in schema['$defs'].items()
                    }
                else:
                    schema[key] = self.remap_json_schema(value)
        return schema


class GenerateJsonSchema:
    """!!! abstract "Usage Documentation"
        [Customizing the JSON Schema Generation Process](../concepts/json_schema.md#customizing-the-json-schema-generation-process)

    A class for generating JSON schemas.

    This class generates JSON schemas based on configured parameters. The default schema dialect
    is [https://json-schema.org/draft/2020-12/schema](https://json-schema.org/draft/2020-12/schema).
    The class uses `by_alias` to configure how fields with
    multiple names are handled and `ref_template` to format reference names.

    Attributes:
        schema_dialect: The JSON schema dialect used to generate the schema. See
            [Declaring a Dialect](https://json-schema.org/understanding-json-schema/reference/schema.html#id4)
            in the JSON Schema documentation for more information about dialects.
        ignored_warning_kinds: Warnings to ignore when generating the schema. `self.render_warning_message` will
            do nothing if its argument `kind` is in `ignored_warning_kinds`;
            this value can be modified on subclasses to easily control which warnings are emitted.
        by_alias: Whether to use field aliases when generating the schema.
        ref_template: The format string used when generating reference names.
        core_to_json_refs: A mapping of core refs to JSON refs.
        core_to_defs_refs: A mapping of core refs to definition refs.
        defs_to_core_refs: A mapping of definition refs to core refs.
        json_to_defs_refs: A mapping of JSON refs to definition refs.
        definitions: Definitions in the schema.

    Args:
        by_alias: Whether to use field aliases in the generated schemas.
        ref_template: The format string to use when generating reference names.

    Raises:
        JsonSchemaError: If the instance of the class is inadvertently reused after generating a schema.
    """

    schema_dialect = 'https://json-schema.org/draft/2020-12/schema'
    ignored_warning_kinds = {'skipped-choice'}

    def __init__(self, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE):
        self.by_alias = by_alias
        self.ref_template = ref_template
        self.core_to_json_refs: Dict[CoreModeRef, JsonRef] = {}
        self.core_to_defs_refs: Dict[CoreModeRef, DefsRef] = {}
        self.defs_to_core_refs: Dict[DefsRef, CoreModeRef] = {}
        self.json_to_defs_refs: Dict[JsonRef, DefsRef] = {}
        self.definitions: Dict[DefsRef, JsonSchemaValue] = {}
        self._config_wrapper_stack = _config.ConfigWrapperStack(
            _config.ConfigWrapper({})
        )
        self._mode = 'validation'
        self._prioritized_defsref_choices: Dict[DefsRef, List[DefsRef]] = {}
        self._collision_counter: Dict[DefsRef, int] = defaultdict(int)
        self._collision_index: Dict[DefsRef, int] = {}
        self._schema_type_to_method = self.build_schema_type_to_method()
        self._core_defs_invalid_for_json_schema: Dict[DefsRef, PydanticInvalidForJsonSchema] = {}
        self._used = False

    @property
    def _config(self) -> _config.ConfigWrapper:
        return self._config_wrapper_stack.tail

    @property
    def mode(self) -> JsonSchemaMode:
        if self._config.json_schema_mode_override is not None:
            return self._config.json_schema_mode_override
        else:
            return self._mode

    def build_schema_type_to_method(self) -> Dict[str, Callable[[CoreSchema], JsonSchemaValue]]:
        """Builds a dictionary mapping fields to methods for generating JSON schemas.

        Returns:
            A dictionary containing the mapping of `CoreSchemaOrFieldType` to a handler method.

        Raises:
            TypeError: If no method has been defined for generating a JSON schema for a given pydantic core schema type.
        """
        mapping = {}
        core_schema_types = _typing_extra.literal_values(CoreSchemaOrFieldType)
        for key in core_schema_types:
            method_name = f'{key.replace('-', '_')}_schema'
            try:
                mapping[key] = getattr(self, method_name)
            except AttributeError as e:
                if os.getenv('PYDANTIC_PRIVATE_ALLOW_UNHANDLED_SCHEMA_TYPES'):
                    continue
                raise TypeError(
                    f'No method for generating JsonSchema for core_schema.type={key!r} (expected: {type(self).__name__}.{method_name})'
                ) from e
        return mapping

    def generate_definitions(
        self, inputs: Sequence[Tuple[JsonSchemaKeyT, JsonSchemaMode, CoreSchema]]
    ) -> Tuple[
        Dict[Tuple[JsonSchemaKeyT, JsonSchemaMode], JsonSchemaValue],
        Dict[DefsRef, JsonSchemaValue],
    ]:
        """Generates JSON schema definitions from a list of core schemas, pairing the generated definitions with a
        mapping that links the input keys to the definition references.

        Args:
            inputs: A sequence of tuples, where:

                - The first element is a JSON schema key type.
                - The second element is the JSON mode: either 'validation' or 'serialization'.
                - The third element is a core schema.

        Returns:
            A tuple where:

                - The first element is a dictionary whose keys are tuples of JSON schema key type and JSON mode, and
                    whose values are the JSON schema corresponding to that pair of inputs. (These schemas may have
                    JsonRef references to definitions that are defined in the second returned element.)
                - The second element is a dictionary whose keys are definition references for the JSON schemas
                    from the first returned element, and whose values are the actual JSON schema definitions.

        Raises:
            PydanticUserError: Raised if the JSON schema generator has already been used to generate a JSON schema.
        """
        if self._used:
            raise PydanticUserError(
                f'This JSON schema generator has already been used to generate a JSON schema. You must create a new instance of {type(self).__name__} to generate a new JSON schema.',
                code='json-schema-already-used',
            )
        for _, mode, schema in inputs:
            self._mode = mode
            self.generate_inner(schema)
        definitions_remapping = self._build_definitions_remapping()
        json_schemas_map = {}
        for key, mode, schema in inputs:
            self._mode = mode
            json_schema = self.generate_inner(schema)
            json_schemas_map[key, mode] = definitions_remapping.remap_json_schema(
                json_schema
            )
        json_schema = {'$defs': self.definitions}
        json_schema = definitions_remapping.remap_json_schema(json_schema)
        self._used = True
        return (json_schemas_map, self.sort(json_schema['$defs']))

    def generate(self, schema: CoreSchema, mode: JsonSchemaMode = 'validation') -> JsonSchemaValue:
        """Generates a JSON schema for a specified schema in a specified mode.

        Args:
            schema: A Pydantic model.
            mode: The mode in which to generate the schema. Defaults to 'validation'.

        Returns:
            A JSON schema representing the specified schema.

        Raises:
            PydanticUserError: If the JSON schema generator has already been used to generate a JSON schema.
        """
        self._mode = mode
        if self._used:
            raise PydanticUserError(
                f'This JSON schema generator has already been used to generate a JSON schema. You must create a new instance of {type(self).__name__} to generate a new JSON schema.',
                code='json-schema-already-used',
            )
        json_schema = self.generate_inner(schema)
        json_ref_counts = self.get_json_ref_counts(json_schema)
        ref = cast(JsonRef, json_schema.get('$ref'))
        while ref is not None:
            ref_json_schema = self.get_schema_from_definitions(ref)
            if (
                json_ref_counts[ref] == 1
                and ref_json_schema is not None
                and (len(json_schema) == 1)
            ):
                json_schema = ref_json_schema.copy()
                json_ref_counts[ref] -= 1
                ref = cast(JsonRef, json_schema.get('$ref'))
            ref = None
        self._garbage_collect_definitions(json_schema)
        definitions_remapping = self._build_definitions_remapping()
        if self.definitions:
            json_schema['$defs'] = self.definitions
        json_schema = definitions_remapping.remap_json_schema(json_schema)
        self._used = True
        return self.sort(json_schema)

    def generate_inner(self, schema: CoreSchema) -> JsonSchemaValue:
        """Generates a JSON schema for a given core schema.

        Args:
            schema: The given core schema.

        Returns:
            The generated JSON schema.

        TODO: the nested function definitions here seem like bad practice, I'd like to unpack these
        in a future PR. It'd be great if we could shorten the call stack a bit for JSON schema generation,
        and I think there's potential for that here.
        """
        if 'ref' in schema:
            core_ref = CoreRef(schema['ref'])
            core_mode_ref = (core_ref, self.mode)
            if (
                core_mode_ref in self.core_to_defs_refs
                and self.core_to_defs_refs[core_mode_ref] in self.definitions
            ):
                return {'$ref': self.core_to_json_refs[core_mode_ref]}

        def populate_defs(
            core_schema: CoreSchema, json_schema: JsonSchemaValue
        ) -> JsonSchemaValue:
            if 'ref' in core_schema:
                core_ref = CoreRef(core_schema['ref'])
                defs_ref, ref_json_schema = self.get_cache_defs_ref_schema(core_ref)
                json_ref = JsonRef(ref_json_schema['$ref'])
               