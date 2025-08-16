from __future__ import annotations
from typing import Any, Callable, Literal, NamedTuple, Tuple
from pydantic_core import CoreConfig, CoreSchema, ValidationError
from typing_extensions import Protocol, TypeAlias

__all__: Tuple[str] = ('PydanticPluginProtocol', 'BaseValidateHandlerProtocol', 'ValidatePythonHandlerProtocol', 'ValidateJsonHandlerProtocol', 'ValidateStringsHandlerProtocol', 'NewSchemaReturns', 'SchemaTypePath', 'SchemaKind')
NewSchemaReturns: TypeAlias = 'tuple[ValidatePythonHandlerProtocol | None, ValidateJsonHandlerProtocol | None, ValidateStringsHandlerProtocol | None]'

class SchemaTypePath(NamedTuple):
    """Path defining where `schema_type` was defined, or where `TypeAdapter` was called."""
    schema_kind: Literal['BaseModel', 'TypeAdapter', 'dataclass', 'create_model', 'validate_call']

class PydanticPluginProtocol(Protocol):
    """Protocol defining the interface for Pydantic plugins."""

    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings) -> NewSchemaReturns:
        """This method is called for each plugin every time a new [`SchemaValidator`][pydantic_core.SchemaValidator]
        is created.

        It should return an event handler for each of the three validation methods, or `None` if the plugin does not
        implement that method.

        Args:
            schema: The schema to validate against.
            schema_type: The original type which the schema was created from, e.g. the model class.
            schema_type_path: Path defining where `schema_type` was defined, or where `TypeAdapter` was called.
            schema_kind: The kind of schema to validate against.
            config: The config to use for validation.
            plugin_settings: Any plugin settings.

        Returns:
            A tuple of optional event handlers for each of the three validation methods -
                `validate_python`, `validate_json`, `validate_strings`.
        """

class BaseValidateHandlerProtocol(Protocol):
    """Base class for plugin callbacks protocols.

    You shouldn't implement this protocol directly, instead use one of the subclasses with adds the correctly
    typed `on_error` method.
    """
    '`on_enter` is changed to be more specific on all subclasses'

    def on_success(self, result: Any) -> None:
        """Callback to be notified of successful validation.

        Args:
            result: The result of the validation.
        """

    def on_error(self, error: Any) -> None:
        """Callback to be notified of validation errors.

        Args:
            error: The validation error.
        """

    def on_exception(self, exception: Any) -> None:
        """Callback to be notified of validation exceptions.

        Args:
            exception: The exception raised during validation.
        """

class ValidatePythonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
    """Event handler for `SchemaValidator.validate_python`."""

    def on_enter(self, input, *, strict=None, from_attributes=None, context=None, self_instance=None) -> None:
        """Callback to be notified of validation start, and create an instance of the event handler.

        Args:
            input: The input to be validated.
            strict: Whether to validate the object in strict mode.
            from_attributes: Whether to validate objects as inputs by extracting attributes.
            context: The context to use for validation, this is passed to functional validators.
            self_instance: An instance of a model to set attributes on from validation, this is used when running
                validation from the `__init__` method of a model.
        """

class ValidateJsonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
    """Event handler for `SchemaValidator.validate_json`."""

    def on_enter(self, input, *, strict=None, context=None, self_instance=None) -> None:
        """Callback to be notified of validation start, and create an instance of the event handler.

        Args:
            input: The JSON data to be validated.
            strict: Whether to validate the object in strict mode.
            context: The context to use for validation, this is passed to functional validators.
            self_instance: An instance of a model to set attributes on from validation, this is used when running
                validation from the `__init__` method of a model.
        """

StringInput: TypeAlias = 'dict[str, StringInput]'

class ValidateStringsHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
    """Event handler for `SchemaValidator.validate_strings`."""

    def on_enter(self, input, *, strict=None, context=None) -> None:
        """Callback to be notified of validation start, and create an instance of the event handler.

        Args:
            input: The string data to be validated.
            strict: Whether to validate the object in strict mode.
            context: The context to use for validation, this is passed to functional validators.
        """
