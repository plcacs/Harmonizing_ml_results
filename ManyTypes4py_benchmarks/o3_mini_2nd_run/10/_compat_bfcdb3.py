from collections import deque
from copy import copy
from dataclasses import dataclass, is_dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Deque, Dict, FrozenSet, List, Mapping, Sequence, Set, Tuple, Type, Union, cast
from fastapi.exceptions import RequestErrorModel
from fastapi.types import IncEx, ModelNameMap, UnionType
from pydantic import BaseModel, create_model
from pydantic.version import VERSION as PYDANTIC_VERSION
from starlette.datastructures import UploadFile
from typing_extensions import Annotated, Literal, get_args, get_origin

PYDANTIC_VERSION_MINOR_TUPLE: Tuple[int, int] = tuple((int(x) for x in PYDANTIC_VERSION.split('.')[:2]))
PYDANTIC_V2: bool = PYDANTIC_VERSION_MINOR_TUPLE[0] == 2

sequence_annotation_to_type: Dict[Any, Any] = {
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
sequence_types: Tuple[Any, ...] = tuple(sequence_annotation_to_type.keys())

if PYDANTIC_V2:
    from pydantic import PydanticSchemaGenerationError as PydanticSchemaGenerationError
    from pydantic import TypeAdapter
    from pydantic import ValidationError as ValidationError
    from pydantic._internal._schema_generation_shared import GetJsonSchemaHandler as GetJsonSchemaHandler
    from pydantic._internal._typing_extra import eval_type_lenient
    from pydantic._internal._utils import lenient_issubclass as lenient_issubclass
    from pydantic.fields import FieldInfo
    from pydantic.json_schema import GenerateJsonSchema as GenerateJsonSchema
    from pydantic.json_schema import JsonSchemaValue as JsonSchemaValue
    from pydantic_core import CoreSchema as CoreSchema
    from pydantic_core import PydanticUndefined, PydanticUndefinedType
    from pydantic_core import Url as Url
    try:
        from pydantic_core.core_schema import with_info_plain_validator_function as with_info_plain_validator_function
    except ImportError:
        from pydantic_core.core_schema import general_plain_validator_function as with_info_plain_validator_function

    RequiredParam: Any = PydanticUndefined
    Undefined: Any = PydanticUndefined
    UndefinedType: Any = PydanticUndefinedType
    evaluate_forwardref: Callable[[Any], Any] = eval_type_lenient
    Validator = Any

    class BaseConfig:
        pass

    class ErrorWrapper(Exception):
        pass

    @dataclass
    class ModelField:
        field_info: FieldInfo
        name: str
        mode: str = 'validation'

        @property
        def alias(self) -> str:
            a = self.field_info.alias
            return a if a is not None else self.name

        @property
        def required(self) -> bool:
            return self.field_info.is_required()

        @property
        def default(self) -> Any:
            return self.get_default()

        @property
        def type_(self) -> Any:
            return self.field_info.annotation

        def __post_init__(self) -> None:
            self._type_adapter: TypeAdapter = TypeAdapter(Annotated[self.field_info.annotation, self.field_info])

        def get_default(self) -> Any:
            if self.field_info.is_required():
                return Undefined
            return self.field_info.get_default(call_default_factory=True)

        def validate(self, value: Any, values: Dict[str, Any] = {}, *, loc: Tuple[Any, ...] = ()) -> Tuple[Any, Any]:
            try:
                return (self._type_adapter.validate_python(value, from_attributes=True), None)
            except ValidationError as exc:
                return (None, _regenerate_error_with_loc(errors=exc.errors(include_url=False), loc_prefix=loc))

        def serialize(self, value: Any, *, mode: str = 'json', include: Any = None, exclude: Any = None, by_alias: bool = True, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False) -> Any:
            return self._type_adapter.dump_python(
                value,
                mode=mode,
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            )

        def __hash__(self) -> int:
            return id(self)

    def get_annotation_from_field_info(annotation: Any, field_info: Any, field_name: str) -> Any:
        return annotation

    def _normalize_errors(errors: List[Any]) -> List[Any]:
        return errors

    def _model_rebuild(model: Any) -> None:
        model.model_rebuild()

    def _model_dump(model: Any, mode: str = 'json', **kwargs: Any) -> Any:
        return model.model_dump(mode=mode, **kwargs)

    def _get_model_config(model: Any) -> Any:
        return model.model_config

    def get_schema_from_model_field(*, field: Any, schema_generator: Any, model_name_map: ModelNameMap, field_mapping: Mapping[Tuple[Any, Any], Any], separate_input_output_schemas: bool = True) -> Dict[str, Any]:
        override_mode: Any = None if separate_input_output_schemas else 'validation'
        json_schema: Dict[str, Any] = field_mapping[(field, override_mode or field.mode)]
        if '$ref' not in json_schema:
            json_schema['title'] = field.field_info.title or field.alias.title().replace('_', ' ')
        return json_schema

    def get_compat_model_name_map(fields: Sequence[Any]) -> Dict[Any, Any]:
        return {}

    def get_definitions(*, fields: Sequence[Any], schema_generator: Any, model_name_map: Any, separate_input_output_schemas: bool = True) -> Tuple[Any, Any]:
        override_mode: Any = None if separate_input_output_schemas else 'validation'
        inputs: List[Tuple[Any, Any, Any]] = [
            (field, override_mode or field.mode, field._type_adapter.core_schema) for field in fields
        ]
        field_mapping, definitions = schema_generator.generate_definitions(inputs=inputs)
        return (field_mapping, definitions)

    def is_scalar_field(field: Any) -> bool:
        from fastapi import params
        return field_annotation_is_scalar(field.field_info.annotation) and (not isinstance(field.field_info, params.Body))

    def is_sequence_field(field: Any) -> bool:
        return field_annotation_is_sequence(field.field_info.annotation)

    def is_scalar_sequence_field(field: Any) -> bool:
        return field_annotation_is_scalar_sequence(field.field_info.annotation)

    def is_bytes_field(field: Any) -> bool:
        return is_bytes_or_nonable_bytes_annotation(field.type_)

    def is_bytes_sequence_field(field: Any) -> bool:
        return is_bytes_sequence_annotation(field.type_)

    def copy_field_info(*, field_info: Any, annotation: Any) -> Any:
        cls: Any = type(field_info)
        merged_field_info: Any = cls.from_annotation(annotation)
        new_field_info: Any = copy(field_info)
        new_field_info.metadata = merged_field_info.metadata
        new_field_info.annotation = merged_field_info.annotation
        return new_field_info

    def serialize_sequence_value(*, field: Any, value: Any) -> Any:
        origin_type: Any = get_origin(field.field_info.annotation) or field.field_info.annotation
        assert issubclass(origin_type, sequence_types)  # type: ignore
        return sequence_annotation_to_type[origin_type](value)

    def get_missing_field_error(loc: Tuple[Any, ...]) -> Dict[str, Any]:
        error = ValidationError.from_exception_data(
            'Field required', [{'type': 'missing', 'loc': loc, 'input': {}}]
        ).errors(include_url=False)[0]
        error['input'] = None
        return error

    def create_body_model(*, fields: Sequence[Any], model_name: str) -> Any:
        field_params: Dict[str, Tuple[Any, Any]] = {f.name: (f.field_info.annotation, f.field_info) for f in fields}
        BodyModel: Any = create_model(model_name, **field_params)
        return BodyModel

    def get_model_fields(model: Any) -> List[Any]:
        return [ModelField(field_info=field_info, name=name) for name, field_info in model.model_fields.items()]
else:
    from fastapi.openapi.constants import REF_PREFIX as REF_PREFIX
    from pydantic import AnyUrl as Url
    from pydantic import BaseConfig as BaseConfig
    from pydantic import ValidationError as ValidationError
    from pydantic.class_validators import Validator as Validator
    from pydantic.error_wrappers import ErrorWrapper as ErrorWrapper
    from pydantic.errors import MissingError
    from pydantic.fields import SHAPE_FROZENSET, SHAPE_LIST, SHAPE_SEQUENCE, SHAPE_SET, SHAPE_SINGLETON, SHAPE_TUPLE, SHAPE_TUPLE_ELLIPSIS
    from pydantic.fields import FieldInfo as FieldInfo
    from pydantic.fields import ModelField as ModelField
    RequiredParam: Any = Ellipsis
    from pydantic.fields import Undefined as Undefined
    from pydantic.fields import UndefinedType as UndefinedType
    from pydantic.schema import field_schema, get_flat_models_from_fields, get_model_name_map, model_process_schema
    from pydantic.schema import get_annotation_from_field_info as get_annotation_from_field_info
    from pydantic.typing import evaluate_forwardref as evaluate_forwardref
    from pydantic.utils import lenient_issubclass as lenient_issubclass
    GetJsonSchemaHandler = Any
    JsonSchemaValue = Dict[str, Any]
    CoreSchema = Any
    sequence_shapes: Set[Any] = {SHAPE_LIST, SHAPE_SET, SHAPE_FROZENSET, SHAPE_TUPLE, SHAPE_SEQUENCE, SHAPE_TUPLE_ELLIPSIS}
    sequence_shape_to_type: Dict[Any, Any] = {
        SHAPE_LIST: list,
        SHAPE_SET: set,
        SHAPE_TUPLE: tuple,
        SHAPE_SEQUENCE: list,
        SHAPE_TUPLE_ELLIPSIS: list,
    }

    @dataclass
    class GenerateJsonSchema:
        pass

    class PydanticSchemaGenerationError(Exception):
        pass

    def with_info_plain_validator_function(function: Callable[..., Any], *, ref: Any = None, metadata: Any = None, serialization: Any = None) -> Dict[Any, Any]:
        return {}

    def get_model_definitions(*, flat_models: Sequence[Any], model_name_map: Any) -> Dict[Any, Any]:
        definitions: Dict[Any, Any] = {}
        for model in flat_models:
            m_schema, m_definitions, m_nested_models = model_process_schema(model, model_name_map=model_name_map, ref_prefix=REF_PREFIX)
            definitions.update(m_definitions)
            model_name: str = model_name_map[model]
            if 'description' in m_schema:
                m_schema['description'] = m_schema['description'].split('\x0c')[0]
            definitions[model_name] = m_schema
        return definitions

    def is_pv1_scalar_field(field: Any) -> bool:
        from fastapi import params
        field_info: Any = field.field_info
        if not (field.shape == SHAPE_SINGLETON and (not lenient_issubclass(field.type_, BaseModel)) and (not lenient_issubclass(field.type_, dict)) and (not field_annotation_is_sequence(field.type_)) and (not is_dataclass(field.type_)) and (not isinstance(field_info, params.Body))):
            return False
        if field.sub_fields:
            if not all((is_pv1_scalar_field(f) for f in field.sub_fields)):
                return False
        return True

    def is_pv1_scalar_sequence_field(field: Any) -> bool:
        if field.shape in sequence_shapes and (not lenient_issubclass(field.type_, BaseModel)):
            if field.sub_fields is not None:
                for sub_field in field.sub_fields:
                    if not is_pv1_scalar_field(sub_field):
                        return False
            return True
        if _annotation_is_sequence(field.type_):
            return True
        return False

    def _normalize_errors(errors: List[Any]) -> List[Any]:
        use_errors: List[Any] = []
        for error in errors:
            if isinstance(error, ErrorWrapper):
                new_errors = ValidationError(errors=[error], model=RequestErrorModel).errors()
                use_errors.extend(new_errors)
            elif isinstance(error, list):
                use_errors.extend(_normalize_errors(error))
            else:
                use_errors.append(error)
        return use_errors

    def _model_rebuild(model: Any) -> None:
        model.update_forward_refs()

    def _model_dump(model: Any, mode: str = 'json', **kwargs: Any) -> Any:
        return model.dict(**kwargs)

    def _get_model_config(model: Any) -> Any:
        return model.__config__

    def get_schema_from_model_field(*, field: Any, schema_generator: Any, model_name_map: Any, field_mapping: Any, separate_input_output_schemas: bool = True) -> Any:
        return field_schema(field, model_name_map=model_name_map, ref_prefix=REF_PREFIX)[0]

    def get_compat_model_name_map(fields: Sequence[Any]) -> Any:
        models = get_flat_models_from_fields(fields, known_models=set())
        return get_model_name_map(models)

    def get_definitions(*, fields: Sequence[Any], schema_generator: Any, model_name_map: Any, separate_input_output_schemas: bool = True) -> Tuple[Any, Any]:
        models = get_flat_models_from_fields(fields, known_models=set())
        return ({}, get_model_definitions(flat_models=models, model_name_map=model_name_map))

    def is_scalar_field(field: Any) -> bool:
        return is_pv1_scalar_field(field)

    def is_sequence_field(field: Any) -> bool:
        return field.shape in sequence_shapes or _annotation_is_sequence(field.type_)

    def is_scalar_sequence_field(field: Any) -> bool:
        return is_pv1_scalar_sequence_field(field)

    def is_bytes_field(field: Any) -> bool:
        return lenient_issubclass(field.type_, bytes)

    def is_bytes_sequence_field(field: Any) -> bool:
        return field.shape in sequence_shapes and lenient_issubclass(field.type_, bytes)

    def copy_field_info(*, field_info: Any, annotation: Any) -> Any:
        return copy(field_info)

    def serialize_sequence_value(*, field: Any, value: Any) -> Any:
        return sequence_shape_to_type[field.shape](value)

    def get_missing_field_error(loc: Tuple[Any, ...]) -> Dict[str, Any]:
        missing_field_error = ErrorWrapper(MissingError(), loc=loc)
        new_error = ValidationError([missing_field_error], RequestErrorModel)
        return new_error.errors()[0]

    def create_body_model(*, fields: Sequence[Any], model_name: str) -> Any:
        BodyModel: Any = create_model(model_name)
        for f in fields:
            BodyModel.__fields__[f.name] = f
        return BodyModel

    def get_model_fields(model: Any) -> List[Any]:
        return list(model.__fields__.values())

def _regenerate_error_with_loc(*, errors: List[Any], loc_prefix: Tuple[Any, ...]) -> List[Any]:
    updated_loc_errors: List[Any] = [{**err, 'loc': loc_prefix + err.get('loc', ())} for err in _normalize_errors(errors)]
    return updated_loc_errors

def _annotation_is_sequence(annotation: Any) -> bool:
    if lenient_issubclass(annotation, (str, bytes)):
        return False
    return lenient_issubclass(annotation, sequence_types)

def field_annotation_is_sequence(annotation: Any) -> bool:
    origin: Any = get_origin(annotation)
    if origin is Union or origin is UnionType:
        for arg in get_args(annotation):
            if field_annotation_is_sequence(arg):
                return True
        return False
    return _annotation_is_sequence(annotation) or _annotation_is_sequence(get_origin(annotation))

def value_is_sequence(value: Any) -> bool:
    return isinstance(value, sequence_types) and (not isinstance(value, (str, bytes)))

def _annotation_is_complex(annotation: Any) -> bool:
    return lenient_issubclass(annotation, (BaseModel, Mapping, UploadFile)) or _annotation_is_sequence(annotation) or is_dataclass(annotation)

def field_annotation_is_complex(annotation: Any) -> bool:
    origin: Any = get_origin(annotation)
    if origin is Union or origin is UnionType:
        return any((field_annotation_is_complex(arg) for arg in get_args(annotation)))
    return _annotation_is_complex(annotation) or _annotation_is_complex(origin) or hasattr(origin, '__pydantic_core_schema__') or hasattr(origin, '__get_pydantic_core_schema__')

def field_annotation_is_scalar(annotation: Any) -> bool:
    return annotation is Ellipsis or not field_annotation_is_complex(annotation)

def field_annotation_is_scalar_sequence(annotation: Any) -> bool:
    origin: Any = get_origin(annotation)
    if origin is Union or origin is UnionType:
        at_least_one_scalar_sequence: bool = False
        for arg in get_args(annotation):
            if field_annotation_is_scalar_sequence(arg):
                at_least_one_scalar_sequence = True
                continue
            elif not field_annotation_is_scalar(arg):
                return False
        return at_least_one_scalar_sequence
    return field_annotation_is_sequence(annotation) and all((field_annotation_is_scalar(sub_annotation) for sub_annotation in get_args(annotation)))

def is_bytes_or_nonable_bytes_annotation(annotation: Any) -> bool:
    if lenient_issubclass(annotation, bytes):
        return True
    origin: Any = get_origin(annotation)
    if origin is Union or origin is UnionType:
        for arg in get_args(annotation):
            if lenient_issubclass(arg, bytes):
                return True
    return False

def is_uploadfile_or_nonable_uploadfile_annotation(annotation: Any) -> bool:
    if lenient_issubclass(annotation, UploadFile):
        return True
    origin: Any = get_origin(annotation)
    if origin is Union or origin is UnionType:
        for arg in get_args(annotation):
            if lenient_issubclass(arg, UploadFile):
                return True
    return False

def is_bytes_sequence_annotation(annotation: Any) -> bool:
    origin: Any = get_origin(annotation)
    if origin is Union or origin is UnionType:
        at_least_one: bool = False
        for arg in get_args(annotation):
            if is_bytes_sequence_annotation(arg):
                at_least_one = True
                continue
        return at_least_one
    return field_annotation_is_sequence(annotation) and all((is_bytes_or_nonable_bytes_annotation(sub_annotation) for sub_annotation in get_args(annotation)))

def is_uploadfile_sequence_annotation(annotation: Any) -> bool:
    origin: Any = get_origin(annotation)
    if origin is Union or origin is UnionType:
        at_least_one: bool = False
        for arg in get_args(annotation):
            if is_uploadfile_sequence_annotation(arg):
                at_least_one = True
                continue
        return at_least_one
    return field_annotation_is_sequence(annotation) and all((is_uploadfile_or_nonable_uploadfile_annotation(sub_annotation) for sub_annotation in get_args(annotation)))

@lru_cache
def get_cached_model_fields(model: Any) -> List[Any]:
    return get_model_fields(model)