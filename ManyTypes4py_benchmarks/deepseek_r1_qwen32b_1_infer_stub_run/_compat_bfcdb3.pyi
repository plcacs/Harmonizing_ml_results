from collections import deque
from copy import copy
from dataclasses import dataclass, is_dataclass
from enum import Enum
from functools import lru_cache
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
from fastapi.exceptions import RequestErrorModel
from fastapi.types import IncEx, ModelNameMap, UnionType
from pydantic import BaseModel
from starlette.datastructures import UploadFile
from typing_extensions import Annotated, Literal, get_args, get_origin
from pydantic_core import CoreSchema, PydanticUndefined, PydanticUndefinedType, Url

PYDANTIC_VERSION_MINOR_TUPLE: tuple[int, int]
PYDANTIC_V2: bool
sequence_annotation_to_type: dict[type, type]
sequence_types: tuple[type, ...]

class BaseConfig:
    pass

class ErrorWrapper(Exception):
    pass

@dataclass
class ModelField:
    name: str
    field_info: Any
    mode: str = 'validation'
    _type_adapter: Any = ...

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
    def validate(self, value: Any, values: dict[str, Any] = {}, loc: tuple[str, ...] = ()) -> tuple[Any, list[dict[str, Any]] | None]: ...
    def serialize(
        self,
        value: Any,
        mode: str = 'json',
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        by_alias: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> Any: ...
    def __hash__(self) -> int: ...

class GenerateJsonSchema:
    pass

class PydanticSchemaGenerationError(Exception):
    pass

def get_annotation_from_field_info(annotation: Any, field_info: Any, field_name: str) -> Any: ...
def _normalize_errors(errors: list[Any]) -> list[dict[str, Any]]: ...
def _model_rebuild(model: Any) -> None: ...
def _model_dump(model: Any, mode: str = 'json', **kwargs: Any) -> Any: ...
def _get_model_config(model: Any) -> Any: ...
def get_schema_from_model_field(
    field: Any,
    schema_generator: Any,
    model_name_map: ModelNameMap,
    field_mapping: Any,
    separate_input_output_schemas: bool = True,
) -> dict[str, Any]: ...
def get_compat_model_name_map(fields: list[Any]) -> ModelNameMap: ...
def get_definitions(
    fields: list[Any],
    schema_generator: Any,
    model_name_map: ModelNameMap,
    separate_input_output_schemas: bool = True,
) -> tuple[dict[Any, Any], dict[str, Any]]: ...
def is_scalar_field(field: Any) -> bool: ...
def is_sequence_field(field: Any) -> bool: ...
def is_scalar_sequence_field(field: Any) -> bool: ...
def is_bytes_field(field: Any) -> bool: ...
def is_bytes_sequence_field(field: Any) -> bool: ...
def copy_field_info(field_info: Any, annotation: Any) -> Any: ...
def serialize_sequence_value(field: Any, value: Any) -> Any: ...
def get_missing_field_error(loc: tuple[str, ...]) -> dict[str, Any]: ...
def create_body_model(fields: list[Any], model_name: str) -> type[BaseModel]: ...
def get_model_fields(model: Any) -> list[Any]: ...
def _regenerate_error_with_loc(errors: list[Any], loc_prefix: tuple[str, ...]) -> list[dict[str, Any]]: ...
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
def get_cached_model_fields(model: Any) -> list[Any]: ...