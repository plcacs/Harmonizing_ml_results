from __future__ import annotations

from typing import Any, Callable, Literal, NamedTuple, Tuple, Dict
from typing_extensions import Protocol, TypeAlias

from pydantic_core import CoreConfig, CoreSchema, ValidationError

__all__ = (
    "PydanticPluginProtocol",
    "BaseValidateHandlerProtocol",
    "ValidatePythonHandlerProtocol",
    "ValidateJsonHandlerProtocol",
    "ValidateStringsHandlerProtocol",
    "NewSchemaReturns",
    "SchemaTypePath",
    "SchemaKind",
)

NewSchemaReturns: TypeAlias = Tuple[
    ValidatePythonHandlerProtocol | None,
    ValidateJsonHandlerProtocol | None,
    ValidateStringsHandlerProtocol | None,
]

class SchemaTypePath(NamedTuple):
    module: str
    name: str

SchemaKind: TypeAlias = Literal["BaseModel", "TypeAdapter", "dataclass", "create_model", "validate_call"]

class PydanticPluginProtocol(Protocol):
    def new_schema_validator(
        self,
        schema: CoreSchema,
        schema_type: Any,
        schema_type_path: SchemaTypePath,
        schema_kind: SchemaKind,
        config: CoreConfig | None,
        plugin_settings: Dict[str, object],
    ) -> Tuple[
        ValidatePythonHandlerProtocol | None,
        ValidateJsonHandlerProtocol | None,
        ValidateStringsHandlerProtocol | None,
    ]:
        raise NotImplementedError("Pydantic plugins should implement `new_schema_validator`.")

class BaseValidateHandlerProtocol(Protocol):
    on_enter: Callable[..., None]

    def on_success(self, result: Any) -> None:
        return

    def on_error(self, error: ValidationError) -> None:
        return

    def on_exception(self, exception: Exception) -> None:
        return

class ValidatePythonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
    def on_enter(
        self,
        input: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: Dict[str, Any] | None = None,
        self_instance: Any | None = None,
    ) -> None:
        pass

class ValidateJsonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
    def on_enter(
        self,
        input: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: Dict[str, Any] | None = None,
        self_instance: Any | None = None,
    ) -> None:
        pass

StringInput: TypeAlias = dict[str, StringInput]

class ValidateStringsHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
    def on_enter(
        self,
        input: StringInput,
        *,
        strict: bool | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        pass
