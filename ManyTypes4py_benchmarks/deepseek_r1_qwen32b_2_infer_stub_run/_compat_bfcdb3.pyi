from collections import deque
from copy import copy
from dataclasses import dataclass
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
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    Any,
    Optional,
    TypeVar,
    Union,
    overload,
    Literal,
    get_origin,
    get_args,
    Annotated,
    cast,
    TypeAlias,
    runtime_checkable,
    Protocol,
    NoReturn,
    NamedTuple,
    TypedDict,
    Final,
    ClassVar,
    NewType,
    ChainMap,
    Counter,
    ItemsView,
    KeysView,
    ValuesView,
    ContextManager,
    TextIO,
    BinaryIO,
    Match,
    Pattern,
    Generator,
    Iterable,
    Iterator,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    overload,
    final,
    TypedDict,
    NotRequired,
    Required,
    TypeVarTuple,
    Unpack,
    ParamSpec,
    Concatenate,
    TypeGuard,
    Never,
    Self,
    EllipsisType,
    SupportsAbs,
    SupportsBytes,
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    SupportsRound,
    Hashable,
    Collection,
    Container,
    Sized,
   Callable,
    Any,
    Optional,
    TypeVar,
    Union,
    overload,
    Literal,
    get_origin,
    get_args,
    Annotated,
    cast,
    TypeAlias,
    runtime_checkable,
    Protocol,
    NoReturn,
    NamedTuple,
    TypedDict,
    Final,
    ClassVar,
    NewType,
    ChainMap,
    Counter,
    ItemsView,
    KeysView,
    ValuesView,
    ContextManager,
    TextIO,
    BinaryIO,
    Match,
    Pattern,
    Generator,
    Iterable,
    Iterator,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    overload,
    final,
    TypedDict,
    NotRequired,
    Required,
    TypeVarTuple,
    Unpack,
    ParamSpec,
    Concatenate,
    TypeGuard,
    Never,
    Self,
    EllipsisType,
    SupportsAbs,
    SupportsBytes,
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    SupportsRound,
    Hashable,
    Collection,
    Container,
    Sized,
   Callable,
    Any,
    Optional,
    TypeVar,
    Union,
    overload,
    Literal,
    get_origin,
    get_args,
    Annotated,
    cast,
    TypeAlias,
    runtime_checkable,
    Protocol,
    NoReturn,
    NamedTuple,
    TypedDict,
    Final,
    ClassVar,
    NewType,
    ChainMap,
    Counter,
    ItemsView,
    KeysView,
    ValuesView,
    ContextManager,
    TextIO,
    BinaryIO,
    Match,
    Pattern,
    Generator,
    Iterable,
    Iterator,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    overload,
    final,
    TypedDict,
    NotRequired,
    Required,
    TypeVarTuple,
    Unpack,
    ParamSpec,
    Concatenate,
    TypeGuard,
    Never,
    Self,
    EllipsisType,
    SupportsAbs,
    SupportsBytes,
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    SupportsRound,
    Hashable,
    Collection,
    Container,
    Sized,
)
from fastapi.exceptions import RequestErrorModel
from fastapi.types import IncEx, ModelNameMap, UnionType
from pydantic import BaseModel
from pydantic.version import VERSION as PYDANTIC_VERSION
from starlette.datastructures import UploadFile
from typing_extensions import Annotated, Literal, get_args, get_origin

PYDANTIC_VERSION_MINOR_TUPLE = tuple((int(x) for x in PYDANTIC_VERSION.split('.')[:2]))
PYDANTIC_V2 = PYDANTIC_VERSION_MINOR_TUPLE[0] == 2
sequence_annotation_to_type: Dict[Type, Type] = {
    Sequence: list,
    List: list,
    list: list,
    Tuple: tuple,
    tuple: tuple,
    Set: set,
    set: set,
    FrozenSet: frozenset,
    frozenset: frozenset,
    Deque: deque,
    deque: deque,
}
sequence_types = tuple(sequence_annotation_to_type.keys())

class BaseConfig:
    pass

class ErrorWrapper(Exception):
    pass

@dataclass
class ModelField:
    mode: str = 'validation'
    name: str
    field_info: FieldInfo

    @property
    def alias(self) -> str:
        ...

    @property
    def required(self) -> bool:
        ...

    @property
    def default(self) -> Union[Undefined, Any]:
        ...

    @property
    def type_(self) -> Any:
        ...

    def get_default(self) -> Union[Undefined, Any]:
        ...

    def validate(self, value: Any, values: Dict[str, Any] = ..., *, loc: Tuple[str, ...] = ...) -> Tuple[Any, Optional[List[Dict[str, Any]]]]:
        ...

    def serialize(self, value: Any, *, mode: str = 'json', include: Optional[Union[Set[str], Dict[str, Any]]] = ..., exclude: Optional[Union[Set[str], Dict[str, Any]]] = ..., by_alias: bool = ..., exclude_unset: bool = ..., exclude_defaults: bool = ..., exclude_none: bool = ...) -> Any:
        ...

    def __hash__(self) -> int:
        ...

def get_annotation_from_field_info(annotation: Any, field_info: FieldInfo, field_name: str) -> Any:
    ...

def _normalize_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ...

def _model_rebuild(model: Any) -> None:
    ...

def _model_dump(model: Any, mode: str = 'json', **kwargs: Any) -> Any:
    ...

def _get_model_config(model: Any) -> Any:
    ...

def get_schema_from_model_field(*, field: ModelField, schema_generator: Any, model_name_map: ModelNameMap, field_mapping: Dict[Tuple[ModelField, str], Dict[str, Any]], separate_input_output_schemas: bool = True) -> Dict[str, Any]:
    ...

def get_compat_model_name_map(fields: List[ModelField]) -> ModelNameMap:
    ...

def get_definitions(*, fields: List[ModelField], schema_generator: Any, model_name_map: ModelNameMap, separate_input_output_schemas: bool = True) -> Tuple[Dict[Tuple[ModelField, str], Dict[str, Any]], Dict[str, Any]]:
    ...

def is_scalar_field(field: ModelField) -> bool:
    ...

def is_sequence_field(field: ModelField) -> bool:
    ...

def is_scalar_sequence_field(field: ModelField) -> bool:
    ...

def is_bytes_field(field: ModelField) -> bool:
    ...

def is_bytes_sequence_field(field: ModelField) -> bool:
    ...

def copy_field_info(*, field_info: FieldInfo, annotation: Any) -> FieldInfo:
    ...

def serialize_sequence_value(*, field: ModelField, value: Any) -> Any:
    ...

def get_missing_field_error(loc: Tuple[str, ...]) -> Dict[str, Any]:
    ...

def create_body_model(*, fields: List[ModelField], model_name: str) -> Type[BaseModel]:
    ...

def get_model_fields(model: Any) -> List[ModelField]:
    ...

def _regenerate_error_with_loc(*, errors: List[Dict[str, Any]], loc_prefix: Tuple[str, ...]) -> List[Dict[str, Any]]:
    ...

def _annotation_is_sequence(annotation: Any) -> bool:
    ...

def field_annotation_is_sequence(annotation: Any) -> bool:
    ...

def value_is_sequence(value: Any) -> bool:
    ...

def _annotation_is_complex(annotation: Any) -> bool:
    ...

def field_annotation_is_complex(annotation: Any) -> bool:
    ...

def field_annotation_is_scalar(annotation: Any) -> bool:
    ...

def field_annotation_is_scalar_sequence(annotation: Any) -> bool:
    ...

def is_bytes_or_nonable_bytes_annotation(annotation: Any) -> bool:
    ...

def is_uploadfile_or_nonable_uploadfile_annotation(annotation: Any) -> bool:
    ...

def is_bytes_sequence_annotation(annotation: Any) -> bool:
    ...

def is_uploadfile_sequence_annotation(annotation: Any) -> bool:
    ...

@lru_cache(maxsize=None)
def get_cached_model_fields(model: Any) -> List[ModelField]:
    ...