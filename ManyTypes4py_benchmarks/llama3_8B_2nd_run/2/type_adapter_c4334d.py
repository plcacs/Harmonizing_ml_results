from __future__ import annotations
import sys
from collections.abc import Callable, Iterable
from dataclasses import is_dataclass
from types import FrameType
from typing import Any, Generic, Literal, TypeVar, cast, final, overload
from typing_extensions import ParamSpec, is_typeddict
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator, Some
from typing import TypeVar, ParamSpec, Generic
from pydantic.errors import PydanticUserError
from pydantic.main import BaseModel, IncEx
from ._internal import _config, _generate_schema, _namespace_utils, _repr, _typing_extra, _utils
from .config import ConfigDict
from .errors import PydanticUndefinedAnnotation
from .json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaKeyT, JsonSchemaMode, JsonSchemaValue
from .plugin._schema_validator import PluggableSchemaValidator, create_schema_validator

T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')
TypeAdapterT = TypeVar('TypeAdapterT', bound='TypeAdapter')

@overload
def __init__(self, type: Type[T], *, config: ConfigDict = ..., _parent_depth: int = ..., module: str = ...) -> None:
    ...

@overload
def __init__(self, type: Type[T], *, config: ConfigDict = ..., _parent_depth: int = ..., module: str = ...) -> None:
    ...

def __init__(self, type: Type[T], *, config: ConfigDict = None, _parent_depth: int = 2, module: str = None) -> None:
    ...

def _init_core_attrs(self, ns_resolver: _namespace_utils.NsResolver, force: bool, raise_errors: bool) -> bool:
    ...

def rebuild(self, *, force: bool = False, raise_errors: bool = True, _parent_namespace_depth: int = 2, _types_namespace: dict = None) -> bool | None:
    ...

def validate_python(self, obj: Any, /, *, strict: bool = None, from_attributes: bool = None, context: dict = None, experimental_allow_partial: bool = False) -> Any:
    ...

def validate_json(self, data: Any, /, *, strict: bool = None, context: dict = None, experimental_allow_partial: bool = False) -> Any:
    ...

def validate_strings(self, obj: Any, /, *, strict: bool = None, context: dict = None, experimental_allow_partial: bool = False) -> Any:
    ...

def get_default_value(self, *, strict: bool = None, context: dict = None) -> Some[Any] | None:
    ...

def dump_python(self, instance: Any, /, *, mode: str = 'python', include: Iterable[str] = ..., exclude: Iterable[str] = ..., by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, round_trip: bool = False, warnings: bool = True, fallback: Callable[[Any], Any] = ..., serialize_as_any: bool = False, context: dict = None) -> Any:
    ...

def dump_json(self, instance: Any, /, *, indent: int = None, include: Iterable[str] = ..., exclude: Iterable[str] = ..., by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, round_trip: bool = False, warnings: bool = True, fallback: Callable[[Any], Any] = ..., serialize_as_any: bool = False, context: dict = None) -> bytes:
    ...

def json_schema(self, *, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE, schema_generator: GenerateJsonSchema = GenerateJsonSchema, mode: str = 'validation') -> dict:
    ...

@overload
def json_schemas(inputs: Iterable[tuple], /, *, by_alias: bool = True, title: str = None, description: str = None, ref_template: str = DEFAULT_REF_TEMPLATE, schema_generator: GenerateJsonSchema = GenerateJsonSchema) -> tuple:
    ...

def json_schemas(inputs: Iterable[tuple], /, *, by_alias: bool = True, title: str = None, description: str = None, ref_template: str = DEFAULT_REF_TEMPLATE, schema_generator: GenerateJsonSchema = GenerateJsonSchema) -> tuple:
    ...
