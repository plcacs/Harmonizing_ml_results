from typing import Any

# === Internal dependency: fastapi.exceptions ===
RequestErrorModel: Type[BaseModel]

# === Internal dependency: fastapi.openapi.constants ===
REF_PREFIX: str

# === Internal dependency: fastapi.params ===
class Body(FieldInfo): ...

# === Internal dependency: fastapi.types ===
UnionType: Any
ModelNameMap: Any

# === Third-party dependency: pydantic ===
# Used symbols: BaseModel, TypeAdapter, ValidationError, create_model

# === Third-party dependency: pydantic._internal._typing_extra ===
def eval_type_lenient(value: Any, globalns: GlobalsNamespace | None = ..., localns: MappingNamespace | None = ...) -> Any: ...

# === Third-party dependency: pydantic.error_wrappers ===
# Used symbols: ErrorWrapper

# === Third-party dependency: pydantic.errors ===
# Used symbols: MissingError

# === Third-party dependency: pydantic.fields ===
class FieldInfo(_repr.Representation): ...

# === Third-party dependency: pydantic.json_schema ===
# Used symbols: JsonSchemaValue

# === Third-party dependency: pydantic.schema ===
# Used symbols: field_schema, get_flat_models_from_fields, get_model_name_map, model_process_schema

# === Third-party dependency: pydantic.utils ===
# Used symbols: lenient_issubclass

# === Third-party dependency: pydantic.version ===
VERSION: str

# === Third-party dependency: pydantic_core ===
# Used symbols: PydanticUndefined, PydanticUndefinedType

# === Third-party dependency: starlette.datastructures ===
class UploadFile: ...