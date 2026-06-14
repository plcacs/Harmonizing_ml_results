from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from fastapi.types import IncEx, ModelNameMap, UnionType

PYDANTIC_VERSION_MINOR_TUPLE: Tuple[int, ...]
PYDANTIC_V2: bool

sequence_annotation_to_type: Dict[Any, type]
sequence_types: Tuple[Any, ...]

# Conditionally available depending on Pydantic version, but we declare them
# unconditionally for the stub since they are always assigned at module level.

# --- Always available (from either branch) ---

from pydantic import ValidationError as ValidationError
from pydantic.fields import FieldInfo as FieldInfo

RequiredParam: Any
Undefined: Any
UndefinedType: Any
evaluate_forwardref: Any
Validator: Any
GetJsonSchemaHandler: Any
JsonSchemaValue: Any
CoreSchema: Any

lenient_issubclass: Callable[..., bool]

# Re-exports
Url: Any

class BaseConfig: ...

class ErrorWrapper(Exception): ...

class PydanticSchemaGenerationError(Exception): ...

def with_info_plain_validator_function(
    function: Any, *, ref: Any = ..., metadata: Any = ..., serialization: Any = ...
) -> Any: ...

@dataclass
class GenerateJsonSchema: ...

@dataclass
class ModelField:
    field_info: FieldInfo
    name: str
    mode: str

    @property
    def alias(self) -> str: ...
    @property
    def required(self) -> bool: ...
    @property
    def default(self) -> Any: ...
    @property
    def type_(self) -> Any: ...
    def __post_init__(self) -> None: ...
    def get_default(self) -> Any: ...
    def validate(
        self, value: Any, values: Dict[str, Any] = ..., *, loc: Tuple[Union[str, int], ...] = ...
    ) -> Tuple[Any, Any]: ...
    def serialize(
        self,
        value: Any,
        *,
        mode: str = ...,
        include: IncEx = ...,
        exclude: IncEx = ...,
        by_alias: bool = ...,
        exclude_unset: bool = ...,
        exclude_defaults: bool = ...,
        exclude_none: bool = ...,
    ) -> Any: ...
    def __hash__(self) -> int: ...

def get_annotation_from_field_info(
    annotation: Any, field_info: FieldInfo, field_name: str
) -> Any: ...

def _normalize_errors(errors: Sequence[Any]) -> List[Any]: ...

def _model_rebuild(model: Type[BaseModel]) -> None: ...

def _model_dump(model: BaseModel, mode: str = ..., **kwargs: Any) -> Any: ...

def _get_model_config(model: Type[BaseModel]) -> Any: ...

def get_schema_from_model_field(
    *,
    field: ModelField,
    schema_generator: Any,
    model_name_map: ModelNameMap,
    field_mapping: Any,
    separate_input_output_schemas: bool = ...,
) -> Dict[str, Any]: ...

def get_compat_model_name_map(fields: List[ModelField]) -> ModelNameMap: ...

def get_definitions(
    *,
    fields: List[ModelField],
    schema_generator: Any,
    model_name_map: ModelNameMap,
    separate_input_output_schemas: bool = ...,
) -> Tuple[Any, Any]: ...

def is_scalar_field(field: ModelField) -> bool: ...
def is_sequence_field(field: ModelField) -> bool: ...
def is_scalar_sequence_field(field: ModelField) -> bool: ...
def is_bytes_field(field: ModelField) -> bool: ...
def is_bytes_sequence_field(field: ModelField) -> bool: ...

def copy_field_info(*, field_info: FieldInfo, annotation: Any) -> FieldInfo: ...

def serialize_sequence_value(*, field: ModelField, value: Any) -> Any: ...

def get_missing_field_error(loc: Tuple[Union[str, int], ...]) -> Dict[str, Any]: ...

def create_body_model(
    *, fields: List[ModelField], model_name: str
) -> Type[BaseModel]: ...

def get_model_fields(model: Type[BaseModel]) -> List[ModelField]: ...

# --- Always available functions ---

def _regenerate_error_with_loc(
    *, errors: Sequence[Any], loc_prefix: Tuple[Union[str, int], ...]
) -> List[Dict[str, Any]]: ...

def _annotation_is_sequence(annotation: Any) -> bool: ...
def field_annotation_is_sequence(annotation: Any) -> bool: ...
def value_is_sequence(value: Any) -> bool: ...
def _annotation_is_complex(annotation: Any) -> bool: ...
def field_annotation_is_complex(annotation: Any) -> bool: ...
def field_annotation_is_scalar(annotation: Any) -> bool: ...
def field_annotation_is_scalar_sequence(annotation: Any) -> bool: ...
def is_bytes_or_nonable_bytes_annotation(annotation: Any) -> bool: ...
def is_uploadfile_or_nonable_uploadfile_annotation(annotation: Any) -> bool: ...
def is_bytes_sequence_annotation(annotation: Any) -> bool: ...
def is_uploadfile_sequence_annotation(annotation: Any) -> bool: ...
def get_cached_model_fields(model: Type[BaseModel]) -> List[ModelField]: ...