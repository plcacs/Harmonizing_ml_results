from typing import Any

# === Internal dependency: pydantic ===
BaseModel: Any
TypeAdapter: Any
create_model: Any
field_validator: Any
validate_call: Any

# === Internal dependency: pydantic.dataclasses ===
def dataclass(_cls=..., *, init=..., repr=..., eq=..., order=..., unsafe_hash=..., frozen=..., config=..., validate_on_init=..., kw_only=..., slots=...): ...

# === Internal dependency: pydantic.plugin ===
class SchemaTypePath(NamedTuple): ...
class PydanticPluginProtocol(Protocol):
    ...
class ValidatePythonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
class ValidateJsonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
class ValidateStringsHandlerProtocol(BaseValidateHandlerProtocol, Protocol):

# === Internal dependency: pydantic.plugin._loader ===
_plugins = None

# === Third-party dependency: pydantic_core ===
# Used symbols: ValidationError