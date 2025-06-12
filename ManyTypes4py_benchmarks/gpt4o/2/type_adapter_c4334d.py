"""Type adapter specification."""
from __future__ import annotations as _annotations
import sys
from collections.abc import Callable, Iterable
from dataclasses import is_dataclass
from types import FrameType
from typing import Any, Generic, Literal, TypeVar, cast, final, overload
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator, Some
from typing_extensions import ParamSpec, is_typeddict
from pydantic.errors import PydanticUserError
from pydantic.main import BaseModel, IncEx
from ._internal import _config, _generate_schema, _mock_val_ser, _namespace_utils, _repr, _typing_extra, _utils
from .config import ConfigDict
from .errors import PydanticUndefinedAnnotation
from .json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaKeyT, JsonSchemaMode, JsonSchemaValue
from .plugin._schema_validator import PluggableSchemaValidator, create_schema_validator

T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')
TypeAdapterT = TypeVar('TypeAdapterT', bound='TypeAdapter')

def _getattr_no_parents(obj: Any, attribute: str) -> Any:
    """Returns the attribute value without attempting to look up attributes from parent types."""
    if hasattr(obj, '__dict__'):
        try:
            return obj.__dict__[attribute]
        except KeyError:
            pass
    slots = getattr(obj, '__slots__', None)
    if slots is not None and attribute in slots:
        return getattr(obj, attribute)
    else:
        raise AttributeError(attribute)

def _type_has_config(type_: Any) -> bool:
    """Returns whether the type has config."""
    type_ = _typing_extra.annotated_type(type_) or type_
    try:
        return issubclass(type_, BaseModel) or is_dataclass(type_) or is_typeddict(type_)
    except TypeError:
        return False

@final
class TypeAdapter(Generic[T]):
    @overload
    def __init__(self, type: Any, *, config: ConfigDict | None = ..., _parent_depth: int = ..., module: str | None = ...) -> None:
        ...

    @overload
    def __init__(self, type: Any, *, config: ConfigDict | None = ..., _parent_depth: int = ..., module: str | None = ...) -> None:
        ...

    def __init__(self, type: Any, *, config: ConfigDict | None = None, _parent_depth: int = 2, module: str | None = None) -> None:
        if _type_has_config(type) and config is not None:
            raise PydanticUserError('Cannot use `config` when the type is a BaseModel, dataclass or TypedDict. These types can have their own config and setting the config via the `config` parameter to TypeAdapter will not override it, thus the `config` you passed to TypeAdapter becomes meaningless, which is probably not what you want.', code='type-adapter-config-unused')
        self._type = type
        self._config = config
        self._parent_depth = _parent_depth
        self.pydantic_complete = False
        parent_frame = self._fetch_parent_frame()
        if parent_frame is not None:
            globalns = parent_frame.f_globals
            localns = parent_frame.f_locals if parent_frame.f_locals is not globalns else {}
        else:
            globalns = {}
            localns = {}
        self._module_name = module or cast(str, globalns.get('__name__', ''))
        self._init_core_attrs(ns_resolver=_namespace_utils.NsResolver(namespaces_tuple=_namespace_utils.NamespacesTuple(locals=localns, globals=globalns), parent_namespace=localns), force=False)

    def _fetch_parent_frame(self) -> FrameType | None:
        frame = sys._getframe(self._parent_depth)
        if frame.f_globals.get('__name__') == 'typing':
            return frame.f_back
        return frame

    def _init_core_attrs(self, ns_resolver: _namespace_utils.NsResolver, force: bool, raise_errors: bool = False) -> bool:
        """Initialize the core schema, validator, and serializer for the type.

        Args:
            ns_resolver: The namespace resolver to use when building the core schema for the adapted type.
            force: Whether to force the construction of the core schema, validator, and serializer.
                If `force` is set to `False` and `_defer_build` is `True`, the core schema, validator, and serializer will be set to mocks.
            raise_errors: Whether to raise errors if initializing any of the core attrs fails.

        Returns:
            `True` if the core schema, validator, and serializer were successfully initialized, otherwise `False`.

        Raises:
            PydanticUndefinedAnnotation: If `PydanticUndefinedAnnotation` occurs in`__get_pydantic_core_schema__`
                and `raise_errors=True`.
        """
        if not force and self._defer_build:
            _mock_val_ser.set_type_adapter_mocks(self)
            self.pydantic_complete = False
            return False
        try:
            self.core_schema = _getattr_no_parents(self._type, '__pydantic_core_schema__')
            self.validator = _getattr_no_parents(self._type, '__pydantic_validator__')
            self.serializer = _getattr_no_parents(self._type, '__pydantic_serializer__')
            if isinstance(self.core_schema, _mock_val_ser.MockCoreSchema) or isinstance(self.validator, _mock_val_ser.MockValSer) or isinstance(self.serializer, _mock_val_ser.MockValSer):
                raise AttributeError()
        except AttributeError:
            config_wrapper = _config.ConfigWrapper(self._config)
            schema_generator = _generate_schema.GenerateSchema(config_wrapper, ns_resolver=ns_resolver)
            try:
                core_schema = schema_generator.generate_schema(self._type)
            except PydanticUndefinedAnnotation:
                if raise_errors:
                    raise
                _mock_val_ser.set_type_adapter_mocks(self)
                return False
            try:
                self.core_schema = schema_generator.clean_schema(core_schema)
            except _generate_schema.InvalidSchemaError:
                _mock_val_ser.set_type_adapter_mocks(self)
                return False
            core_config = config_wrapper.core_config(None)
            self.validator = create_schema_validator(schema=self.core_schema, schema_type=self._type, schema_type_module=self._module_name, schema_type_name=str(self._type), schema_kind='TypeAdapter', config=core_config, plugin_settings=config_wrapper.plugin_settings)
            self.serializer = SchemaSerializer(self.core_schema, core_config)
        self.pydantic_complete = True
        return True

    @property
    def _defer_build(self) -> bool:
        config = self._config if self._config is not None else self._model_config
        if config:
            return config.get('defer_build') is True
        return False

    @property
    def _model_config(self) -> ConfigDict | None:
        type_ = _typing_extra.annotated_type(self._type) or self._type
        if _utils.lenient_issubclass(type_, BaseModel):
            return type_.model_config
        return getattr(type_, '__pydantic_config__', None)

    def __repr__(self) -> str:
        return f'TypeAdapter({_repr.display_as_type(self._type)})'

    def rebuild(self, *, force: bool = False, raise_errors: bool = True, _parent_namespace_depth: int = 2, _types_namespace: dict[str, Any] | None = None) -> bool | None:
        """Try to rebuild the pydantic-core schema for the adapter's type.

        This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
        the initial attempt to build the schema, and automatic rebuilding fails.

        Args:
            force: Whether to force the rebuilding of the type adapter's schema, defaults to `False`.
            raise_errors: Whether to raise errors, defaults to `True`.
            _parent_namespace_depth: Depth at which to search for the [parent frame][frame-objects]. This
                frame is used when resolving forward annotations during schema rebuilding, by looking for
                the locals of this frame. Defaults to 2, which will result in the frame where the method
                was called.
            _types_namespace: An explicit types namespace to use, instead of using the local namespace
                from the parent frame. Defaults to `None`.

        Returns:
            Returns `None` if the schema is already "complete" and rebuilding was not required.
            If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.
        """
        if not force and self.pydantic_complete:
            return None
        if _types_namespace is not None:
            rebuild_ns = _types_namespace
        elif _parent_namespace_depth > 0:
            rebuild_ns = _typing_extra.parent_frame_namespace(parent_depth=_parent_namespace_depth, force=True) or {}
        else:
            rebuild_ns = {}
        globalns = sys._getframe(max(_parent_namespace_depth - 1, 1)).f_globals
        ns_resolver = _namespace_utils.NsResolver(namespaces_tuple=_namespace_utils.NamespacesTuple(locals=rebuild_ns, globals=globalns), parent_namespace=rebuild_ns)
        return self._init_core_attrs(ns_resolver=ns_resolver, force=True, raise_errors=raise_errors)

    def validate_python(self, object: Any, /, *, strict: bool | None = None, from_attributes: bool | None = None, context: dict[str, Any] | None = None, experimental_allow_partial: bool | str = False) -> Any:
        """Validate a Python object against the model.

        Args:
            object: The Python object to validate against the model.
            strict: Whether to strictly check types.
            from_attributes: Whether to extract data from object attributes.
            context: Additional context to pass to the validator.
            experimental_allow_partial: **Experimental** whether to enable
                [partial validation](../concepts/experimental.md#partial-validation), e.g. to process streams.
                * False / 'off': Default behavior, no partial validation.
                * True / 'on': Enable partial validation.
                * 'trailing-strings': Enable partial validation and allow trailing strings in the input.

        !!! note
            When using `TypeAdapter` with a Pydantic `dataclass`, the use of the `from_attributes`
            argument is not supported.

        Returns:
            The validated object.
        """
        return self.validator.validate_python(object, strict=strict, from_attributes=from_attributes, context=context, allow_partial=experimental_allow_partial)

    def validate_json(self, data: str | bytes, /, *, strict: bool | None = None, context: dict[str, Any] | None = None, experimental_allow_partial: bool | str = False) -> Any:
        """!!! abstract "Usage Documentation"
            [JSON Parsing](../concepts/json.md#json-parsing)

        Validate a JSON string or bytes against the model.

        Args:
            data: The JSON data to validate against the model.
            strict: Whether to strictly check types.
            context: Additional context to use during validation.
            experimental_allow_partial: **Experimental** whether to enable
                [partial validation](../concepts/experimental.md#partial-validation), e.g. to process streams.
                * False / 'off': Default behavior, no partial validation.
                * True / 'on': Enable partial validation.
                * 'trailing-strings': Enable partial validation and allow trailing strings in the input.

        Returns:
            The validated object.
        """
        return self.validator.validate_json(data, strict=strict, context=context, allow_partial=experimental_allow_partial)

    def validate_strings(self, obj: Any, /, *, strict: bool | None = None, context: dict[str, Any] | None = None, experimental_allow_partial: bool | str = False) -> Any:
        """Validate object contains string data against the model.

        Args:
            obj: The object contains string data to validate.
            strict: Whether to strictly check types.
            context: Additional context to use during validation.
            experimental_allow_partial: **Experimental** whether to enable
                [partial validation](../concepts/experimental.md#partial-validation), e.g. to process streams.
                * False / 'off': Default behavior, no partial validation.
                * True / 'on': Enable partial validation.
                * 'trailing-strings': Enable partial validation and allow trailing strings in the input.

        Returns:
            The validated object.
        """
        return self.validator.validate_strings(obj, strict=strict, context=context, allow_partial=experimental_allow_partial)

    def get_default_value(self, *, strict: bool | None = None, context: dict[str, Any] | None = None) -> Some | None:
        """Get the default value for the wrapped type.

        Args:
            strict: Whether to strictly check types.
            context: Additional context to pass to the validator.

        Returns:
            The default value wrapped in a `Some` if there is one or None if not.
        """
        return self.validator.get_default_value(strict=strict, context=context)

    def dump_python(self, instance: Any, /, *, mode: str = 'python', include: IncEx | None = None, exclude: IncEx | None = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, round_trip: bool = False, warnings: bool | str = True, fallback: Callable[[Any], Any] | None = None, serialize_as_any: bool = False, context: dict[str, Any] | None = None) -> Any:
        """Dump an instance of the adapted type to a Python object.

        Args:
            instance: The Python object to serialize.
            mode: The output format.
            include: Fields to include in the output.
            exclude: Fields to exclude from the output.
            by_alias: Whether to use alias names for field names.
            exclude_unset: Whether to exclude unset fields.
            exclude_defaults: Whether to exclude fields with default values.
            exclude_none: Whether to exclude fields with None values.
            round_trip: Whether to output the serialized data in a way that is compatible with deserialization.
            warnings: How to handle serialization errors. False/"none" ignores them, True/"warn" logs errors,
                "error" raises a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError].
            fallback: A function to call when an unknown value is encountered. If not provided,
                a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError] error is raised.
            serialize_as_any: Whether to serialize fields with duck-typing serialization behavior.
            context: Additional context to pass to the serializer.

        Returns:
            The serialized object.
        """
        return self.serializer.to_python(instance, mode=mode, by_alias=by_alias, include=include, exclude=exclude, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings, fallback=fallback, serialize_as_any=serialize_as_any, context=context)

    def dump_json(self, instance: Any, /, *, indent: int | None = None, include: IncEx | None = None, exclude: IncEx | None = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, round_trip: bool = False, warnings: bool | str = True, fallback: Callable[[Any], Any] | None = None, serialize_as_any: bool = False, context: dict[str, Any] | None = None) -> bytes:
        """!!! abstract "Usage Documentation"
            [JSON Serialization](../concepts/json.md#json-serialization)

        Serialize an instance of the adapted type to JSON.

        Args:
            instance: The instance to be serialized.
            indent: Number of spaces for JSON indentation.
            include: Fields to include.
            exclude: Fields to exclude.
            by_alias: Whether to use alias names for field names.
            exclude_unset: Whether to exclude unset fields.
            exclude_defaults: Whether to exclude fields with default values.
            exclude_none: Whether to exclude fields with a value of `None`.
            round_trip: Whether to serialize and deserialize the instance to ensure round-tripping.
            warnings: How to handle serialization errors. False/"none" ignores them, True/"warn" logs errors,
                "error" raises a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError].
            fallback: A function to call when an unknown value is encountered. If not provided,
                a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError] error is raised.
            serialize_as_any: Whether to serialize fields with duck-typing serialization behavior.
            context: Additional context to pass to the serializer.

        Returns:
            The JSON representation of the given instance as bytes.
        """
        return self.serializer.to_json(instance, indent=indent, include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings, fallback=fallback, serialize_as_any=serialize_as_any, context=context)

    def json_schema(self, *, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE, schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema, mode: JsonSchemaMode = 'validation') -> dict[str, Any]:
        """Generate a JSON schema for the adapted type.

        Args:
            by_alias: Whether to use alias names for field names.
            ref_template: The format string used for generating $ref strings.
            schema_generator: The generator class used for creating the schema.
            mode: The mode to use for schema generation.

        Returns:
            The JSON schema for the model as a dictionary.
        """
        schema_generator_instance = schema_generator(by_alias=by_alias, ref_template=ref_template)
        if isinstance(self.core_schema, _mock_val_ser.MockCoreSchema):
            self.core_schema.rebuild()
            assert not isinstance(self.core_schema, _mock_val_ser.MockCoreSchema), 'this is a bug! please report it'
        return schema_generator_instance.generate(self.core_schema, mode=mode)

    @staticmethod
    def json_schemas(inputs: Iterable[tuple[JsonSchemaKeyT, JsonSchemaMode, TypeAdapter]], /, *, by_alias: bool = True, title: str | None = None, description: str | None = None, ref_template: str = DEFAULT_REF_TEMPLATE, schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema) -> tuple[dict[tuple[JsonSchemaKeyT, JsonSchemaMode], JsonSchemaValue], dict[str, Any]]:
        """Generate a JSON schema including definitions from multiple type adapters.

        Args:
            inputs: Inputs to schema generation. The first two items will form the keys of the (first)
                output mapping; the type adapters will provide the core schemas that get converted into
                definitions in the output JSON schema.
            by_alias: Whether to use alias names.
            title: The title for the schema.
            description: The description for the schema.
            ref_template: The format string used for generating $ref strings.
            schema_generator: The generator class used for creating the schema.

        Returns:
            A tuple where:

                - The first element is a dictionary whose keys are tuples of JSON schema key type and JSON mode, and
                    whose values are the JSON schema corresponding to that pair of inputs. (These schemas may have
                    JsonRef references to definitions that are defined in the second returned element.)
                - The second element is a JSON schema containing all definitions referenced in the first returned
                    element, along with the optional title and description keys.

        """
        schema_generator_instance = schema_generator(by_alias=by_alias, ref_template=ref_template)
        inputs_ = []
        for key, mode, adapter in inputs:
            if isinstance(adapter.core_schema, _mock_val_ser.MockCoreSchema):
                adapter.core_schema.rebuild()
                assert not isinstance(adapter.core_schema, _mock_val_ser.MockCoreSchema), 'this is a bug! please report it'
            inputs_.append((key, mode, adapter.core_schema))
        json_schemas_map, definitions = schema_generator_instance.generate_definitions(inputs_)
        json_schema = {}
        if definitions:
            json_schema['$defs'] = definitions
        if title:
            json_schema['title'] = title
        if description:
            json_schema['description'] = description
        return (json_schemas_map, json_schema)
