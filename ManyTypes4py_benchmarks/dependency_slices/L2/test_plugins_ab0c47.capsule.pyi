from typing import Any

# === Internal dependency: pydantic ===
BaseModel: Any
TypeAdapter: Any
create_model: Any
field_validator: Any
validate_call: Any

# === Internal dependency: pydantic.dataclasses ===
def dataclass(_cls: type[_T] | None = ..., *, init: Literal[False] = ..., repr: bool = ..., eq: bool = ..., order: bool = ..., unsafe_hash: bool = ..., frozen: bool | None = ..., config: ConfigDict | type[object] | None = ..., validate_on_init: bool | None = ..., kw_only: bool = ..., slots: bool = ...) -> Callable[[type[_T]], type[PydanticDataclass]] | type[PydanticDataclass]: ...

# === Internal dependency: pydantic.plugin ===
class SchemaTypePath(NamedTuple): ...
class PydanticPluginProtocol(Protocol):
    ...
class ValidatePythonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
class ValidateJsonHandlerProtocol(BaseValidateHandlerProtocol, Protocol):
class ValidateStringsHandlerProtocol(BaseValidateHandlerProtocol, Protocol):

# === Internal dependency: pydantic.plugin._loader ===
_plugins: dict[str, PydanticPluginProtocol] | None

# === Third-party dependency: pydantic_core ===
# Used symbols: ValidationError