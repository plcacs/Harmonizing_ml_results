from __future__ import annotations
from typing import Any, Callable, Literal, NamedTuple, Tuple
from pydantic_core import CoreConfig, CoreSchema, ValidationError
from typing_extensions import Protocol, TypeAlias

__all__: Tuple[str] = ('PydanticPluginProtocol', 'BaseValidateHandlerProtocol', 'ValidatePythonHandlerProtocol', 'ValidateJsonHandlerProtocol', 'ValidateStringsHandlerProtocol', 'NewSchemaReturns', 'SchemaTypePath', 'SchemaKind')
NewSchemaReturns: TypeAlias = 'tuple[ValidatePythonHandlerProtocol | None, ValidateJsonHandlerProtocol | None, ValidateStringsHandlerProtocol | None]'

class SchemaTypePath(NamedTuple):
    """Path defining where `schema_type` was defined, or where `TypeAdapter` was called."""

SchemaKind: Literal['BaseModel', 'TypeAdapter', 'dataclass', 'create_model', 'validate_call']

class PydanticPluginProtocol(Protocol):
    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings) -> NewSchemaReturns:
        ...

class BaseValidateHandlerProtocol(Protocol):
    def on_success(self, result):
        ...

    def on_error(self, error):
        ...

    def on_exception(self, exception):
        ...

class ValidatePythonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
    def on_enter(self, input, *, strict=None, from_attributes=None, context=None, self_instance=None):
        ...

class ValidateJsonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
    def on_enter(self, input, *, strict=None, context=None, self_instance=None):
        ...

StringInput: TypeAlias = 'dict[str, StringInput]'

class ValidateStringsHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
    def on_enter(self, input, *, strict=None, context=None):
        ...
