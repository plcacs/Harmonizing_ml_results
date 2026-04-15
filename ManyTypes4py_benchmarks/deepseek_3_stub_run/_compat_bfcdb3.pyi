from collections import deque
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
    TypeVar,
    Union,
)
from fastapi.exceptions import RequestErrorModel
from fastapi.types import IncEx, ModelNameMap, UnionType
from pydantic import BaseModel, create_model
from pydantic.version import VERSION as PYDANTIC_VERSION
from starlette.datastructures import UploadFile
from typing_extensions import Annotated, Literal, get_args, get_origin

PYDANTIC_VERSION_MINOR_TUPLE: Tuple[int, int] = ...
PYDANTIC_V2: bool = ...
sequence_annotation_to_type: Dict[Any, Any] = ...
sequence_types: Tuple[Any, ...] = ...

if PYDANTIC_V2:
    from pydantic import PydanticSchemaGenerationError as PydanticSchemaGenerationError
    from pydantic import TypeAdapter
    from pydantic import ValidationError as ValidationError
    from pydantic._internal._schema_generation_shared import (
        GetJsonSchemaHandler as GetJsonSchemaHandler,
    )
    from pydantic._internal._typing_extra import eval_type_lenient
    from pydantic._internal._utils import lenient_issubclass as lenient_issubclass
    from pydantic.fields import FieldInfo
    from pydantic.json_schema import GenerateJsonSchema as GenerateJsonSchema
    from pydantic.json_schema import JsonSchemaValue as JsonSchemaValue
    from pydantic_core import CoreSchema as CoreSchema
    from pydantic_core import PydanticUndefined, PydanticUndefinedType
    from pydantic_core import Url as Url

    try:
        from pydantic_core.core_schema import (
            with_info_plain_validator_function as with_info_plain_validator_function,
        )
    except ImportError:
        from pydantic_core.core_schema import (
            general_plain_validator_function as with_info_plain_validator_function,
        )

    RequiredParam: Any = ...
    Undefined: Any = ...
    UndefinedType: Any = ...
    evaluate_forwardref: Any = ...
    Validator: Any = ...

    class BaseConfig:
        pass

    class ErrorWrapper(Exception):
        pass

    @dataclass
    class ModelField:
        mode: str = ...

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
            self,
            value: Any,
            values: Dict[str, Any] = ...,
            *,
            loc: Tuple[Union[str, int], ...] = ...,
        ) -> Tuple[Optional[Any], Optional[List[Dict[str, Any]]]]: ...

        def serialize(
            self,
            value: Any,
            *,
            mode: str = ...,
            include: Optional[IncEx] = ...,
            exclude: Optional[IncEx] = ...,
            by_alias: bool = ...,
            exclude_unset: bool = ...,
            exclude_defaults: bool = ...,
            exclude_none: bool = ...,
        ) -> Any: ...

        def __hash__(self) -> int: ...

    def get_annotation_from_field_info(
        annotation: Any, field_info: FieldInfo, field_name: str
    ) -> Any: ...

    def _normalize_errors(errors: List[Any]) -> List[Dict[str, Any]]: ...

    def _model_rebuild(model: Type[BaseModel]) -> None: ...

    def _model_dump(model: BaseModel, mode: str = ..., **kwargs: Any) -> Dict[str, Any]: ...

    def _get_model_config(model: Type[BaseModel]) -> Any: ...

    def get_schema_from_model_field(
        *,
        field: ModelField,
        schema_generator: GenerateJsonSchema,
        model_name_map: ModelNameMap,
        field_mapping: Dict[Tuple[ModelField, Optional[str]], JsonSchemaValue],
        separate_input_output_schemas: bool = ...,
    ) -> JsonSchemaValue: ...

    def get_compat_model_name_map(fields: List[ModelField]) -> ModelNameMap: ...

    def get_definitions(
        *,
        fields: List[ModelField],
        schema_generator: GenerateJsonSchema,
        model_name_map: ModelNameMap,
        separate_input_output_schemas: bool = ...,
    ) -> Tuple[
        Dict[Tuple[ModelField, Optional[str]], JsonSchemaValue],
        Dict[str, JsonSchemaValue],
    ]: ...

    def is_scalar_field(field: ModelField) -> bool: ...

    def is_sequence_field(field: ModelField) -> bool: ...

    def is_scalar_sequence_field(field: ModelField) -> bool: ...

    def is_bytes_field(field: ModelField) -> bool: ...

    def is_bytes_sequence_field(field: ModelField) -> bool: ...

    def copy_field_info(
        *, field_info: FieldInfo, annotation: Any
    ) -> FieldInfo: ...

    def serialize_sequence_value(*, field: ModelField, value: Any) -> Any: ...

    def get_missing_field_error(
        loc: Tuple[Union[str, int], ...]
    ) -> Dict[str, Any]: ...

    def create_body_model(
        *, fields: List[ModelField], model_name: str
    ) -> Type[BaseModel]: ...

    def get_model_fields(model: Type[BaseModel]) -> List[ModelField]: ...

else:
    from fastapi.openapi.constants import REF_PREFIX as REF_PREFIX
    from pydantic import AnyUrl as Url
    from pydantic import BaseConfig as BaseConfig
    from pydantic import ValidationError as ValidationError
    from pydantic.class_validators import Validator as Validator
    from pydantic.error_wrappers import ErrorWrapper as ErrorWrapper
    from pydantic.errors import MissingError
    from pydantic.fields import (
        SHAPE_FROZENSET,
        SHAPE_LIST,
        SHAPE_SEQUENCE,
        SHAPE_SET,
        SHAPE_SINGLETON,
        SHAPE_TUPLE,
        SHAPE_TUPLE_ELLIPSIS,
    )
    from pydantic.fields import FieldInfo as FieldInfo
    from pydantic.fields import ModelField as ModelField

    RequiredParam: Any = ...
    from pydantic.fields import Undefined as Undefined
    from pydantic.fields import UndefinedType as UndefinedType
    from pydantic.schema import (
        field_schema,
        get_flat_models_from_fields,
        get_model_name_map,
        model_process_schema,
    )
    from pydantic.schema import (
        get_annotation_from_field_info as get_annotation_from_field_info,
    )
    from pydantic.typing import evaluate_forwardref as evaluate_forwardref
    from pydantic.utils import lenient_issubclass as lenient_issubclass

    GetJsonSchemaHandler: Any = ...
    JsonSchemaValue: Type[Dict[str, Any]] = ...
    CoreSchema: Any = ...
    sequence_shapes: Set[int] = ...
    sequence_shape_to_type: Dict[int, Type[Any]] = ...

    @dataclass
    class GenerateJsonSchema:
        pass

    class PydanticSchemaGenerationError(Exception):
        pass

    def with_info_plain_validator_function(
        function: Callable[..., Any],
        *,
        ref: Optional[str] = ...,
        metadata: Optional[Any] = ...,
        serialization: Optional[Any] = ...,
    ) -> Dict[str, Any]: ...

    def get_model_definitions(
        *, flat_models: Set[Type[BaseModel]], model_name_map: ModelNameMap
    ) -> Dict[str, JsonSchemaValue]: ...

    def is_pv1_scalar_field(field: ModelField) -> bool: ...

    def is_pv1_scalar_sequence_field(field: ModelField) -> bool: ...

    def _normalize_errors(errors: List[Any]) -> List[Dict[str, Any]]: ...

    def _model_rebuild(model: Type[BaseModel]) -> None: ...

    def _model_dump(model: BaseModel, mode: str = ..., **kwargs: Any) -> Dict[str, Any]: ...

    def _get_model_config(model: Type[BaseModel]) -> Any: ...

    def get_schema_from_model_field(
        *,
        field: ModelField,
        schema_generator: GenerateJsonSchema,
        model_name_map: ModelNameMap,
        field_mapping: Dict[Any, Any],
        separate_input_output_schemas: bool = ...,
    ) -> JsonSchemaValue: ...

    def get_compat_model_name_map(fields: List[ModelField]) -> ModelNameMap: ...

    def get_definitions(
        *,
        fields: List[ModelField],
        schema_generator: GenerateJsonSchema,
        model_name_map: ModelNameMap,
        separate_input_output_schemas: bool = ...,
    ) -> Tuple[Dict[Any, Any], Dict[str, JsonSchemaValue]]: ...

    def is_scalar_field(field: ModelField) -> bool: ...

    def is_sequence_field(field: ModelField) -> bool: ...

    def is_scalar_sequence_field(field: ModelField) -> bool: ...

    def is_bytes_field(field: ModelField) -> bool: ...

    def is_bytes_sequence_field(field: ModelField) -> bool: ...

    def copy_field_info(*, field_info: FieldInfo, annotation: Any) -> FieldInfo: ...

    def serialize_sequence_value(*, field: ModelField, value: Any) -> Any: ...

    def get_missing_field_error(
        loc: Tuple[Union[str, int], ...]
    ) -> Dict[str, Any]: ...

    def create_body_model(
        *, fields: List[ModelField], model_name: str
    ) -> Type[BaseModel]: ...

    def get_model_fields(model: Type[BaseModel]) -> List[ModelField]: ...


def _regenerate_error_with_loc(
    *, errors: List[Any], loc_prefix: Tuple[Union[str, int], ...]
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


@lru_cache
def get_cached_model_fields(model: Type[BaseModel]) -> List[ModelField]: ...