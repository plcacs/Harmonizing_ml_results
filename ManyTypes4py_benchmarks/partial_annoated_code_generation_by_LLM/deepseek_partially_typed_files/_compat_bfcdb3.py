from collections import deque
from copy import copy
from dataclasses import dataclass, is_dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Deque, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union
from fastapi.exceptions import RequestErrorModel
from fastapi.types import IncEx, ModelNameMap, UnionType
from pydantic import BaseModel, create_model
from pydantic.version import VERSION as PYDANTIC_VERSION
from starlette.datastructures import UploadFile
from typing_extensions import Annotated, Literal, get_args, get_origin

PYDANTIC_VERSION_MINOR_TUPLE: Tuple[int, ...] = tuple((int(x) for x in PYDANTIC_VERSION.split('.')[:2]))
PYDANTIC_V2: bool = PYDANTIC_VERSION_MINOR_TUPLE[0] == 2

sequence_annotation_to_type: Dict[Type[Any], Type[Any]] = {
    Sequence: list, List: list, list: list, Tuple: tuple, tuple: tuple, 
    Set: set, set: set, FrozenSet: frozenset, frozenset: frozenset, 
    Deque: deque, deque: deque
}
sequence_types: Tuple[Type[Any], ...] = tuple(sequence_annotation_to_type.keys())

Url: Type[Any]

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
    UndefinedType: Type[Any] = PydanticUndefinedType
    evaluate_forwardref: Callable[..., Any] = eval_type_lenient
    Validator: Type[Any] = Any

    class BaseConfig:
        pass

    class ErrorWrapper(Exception):
        pass

    @dataclass
    class ModelField:
        field_info: FieldInfo
        name: str
        mode: Literal['validation', 'serialization'] = 'validation'

        @property
        def alias(self) -> str:
            a: Optional[str] = self.field_info.alias
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
            self._type_adapter: TypeAdapter[Any] = TypeAdapter(Annotated[self.field_info.annotation, self.field_info])

        def get_default(self) -> Any:
            if self.field_info.is_required():
                return Undefined
            return self.field_info.get_default(call_default_factory=True)

        def validate(self, value: Any, values: Dict[str, Any] = {}, *, loc: Tuple[Union[int, str], ...] = ()) -> Tuple[Any, Optional[List[Dict[str, Any]]]]:
            try:
                return (self._type_adapter.validate_python(value, from_attributes=True), None)
            except ValidationError as exc:
                return (None, _regenerate_error_with_loc(errors=exc.errors(include_url=False), loc_prefix=loc))

        def serialize(self, value: Any, *, mode: Literal['json', 'python'] = 'json', include: Optional[IncEx] = None, exclude: Optional[IncEx] = None, by_alias: bool = True, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False) -> Any:
            return self._type_adapter.dump_python(value, mode=mode, include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)

        def __hash__(self) -> int:
            return id(self)

    def get_annotation_from_field_info(annotation: Any, field_info: FieldInfo, field_name: str) -> Any:
        return annotation

    def _normalize_errors(errors: Sequence[Any]) -> List[Dict[str, Any]]:
        return errors

    def _model_rebuild(model: Type[BaseModel]) -> None:
        model.model_rebuild()

    def _model_dump(model: BaseModel, mode: Literal['json', 'python'] = 'json', **kwargs: Any) -> Any:
        return model.model_dump(mode=mode, **kwargs)

    def _get_model_config(model: BaseModel) -> Any:
        return model.model_config

    def get_schema_from_model_field(*, field: 'ModelField', schema_generator: GenerateJsonSchema, model_name_map: ModelNameMap, field_mapping: Dict[Tuple['ModelField', Literal['validation', 'serialization']], JsonSchemaValue], separate_input_output_schemas: bool = True) -> Dict[str, Any]:
        override_mode: Optional[Literal['validation']] = None if separate_input_output_schemas else 'validation'
        json_schema: Dict[str, Any] = field_mapping[field, override_mode or field.mode]
        if '$ref' not in json_schema:
            json_schema['title'] = field.field_info.title or field.alias.title().replace('_', ' ')
        return json_schema

    def get_compat_model_name_map(fields: List['ModelField']) -> ModelNameMap:
        return {}

    def get_definitions(*, fields: List['ModelField'], schema_generator: GenerateJsonSchema, model_name_map: ModelNameMap, separate_input_output_schemas: bool = True) -> Tuple[Dict[Tuple['ModelField', Literal['validation', 'serialization']], JsonSchemaValue], Dict[str, Dict[str, Any]]]:
        override_mode: Optional[Literal['validation']] = None if separate_input_output_schemas else 'validation'
        inputs: List[Tuple['ModelField', Literal['validation', 'serialization'], CoreSchema]] = [(field, override_mode or field.mode, field._type_adapter.core_schema) for field in fields]
        field_mapping: Dict[Tuple['ModelField', Literal['validation', 'serialization']], JsonSchemaValue]
        definitions: Dict[str, Dict[str, Any]]
        (field_mapping, definitions) = schema_generator.generate_definitions(inputs=inputs)
        return (field_mapping, definitions)

    def is_scalar_field(field: 'ModelField') -> bool:
        from fastapi import params
        return field_annotation_is_scalar(field.field_info.annotation) and (not isinstance(field.field_info, params.Body))

    def is_sequence_field(field: 'ModelField') -> bool:
        return field_annotation_is_sequence(field.field_info.annotation)

    def is_scalar_sequence_field(field: 'ModelField') -> bool:
        return field_annotation_is_scalar_sequence(field.field_info.annotation)

    def is_bytes_field(field: 'ModelField') -> bool:
        return is_bytes_or_nonable_bytes_annotation(field.type_)

    def is_bytes_sequence_field(field: 'ModelField') -> bool:
        return is_bytes_sequence_annotation(field.type_)

    def copy_field_info(*, field_info: FieldInfo, annotation: Any) -> FieldInfo:
        cls: Type[FieldInfo] = type(field_info)
        merged_field_info: FieldInfo = cls.from_annotation(annotation)
        new_field_info: FieldInfo = copy(field_info)
        new_field_info.metadata = merged_field_info.metadata
        new_field_info.annotation = merged_field_info.annotation
        return new_field_info

    def serialize_sequence_value(*, field: 'ModelField', value: Any) -> Sequence[Any]:
        origin_type: Type[Any] = get_origin(field.field_info.annotation) or field.field_info.annotation
        assert issubclass(origin_type, sequence_types)
        return sequence_annotation_to_type[origin_type](value)

    def get_missing_field_error(loc: Tuple[str, ...]) -> Dict[str, Any]:
        error: Dict[str, Any] = ValidationError.from_exception_data('Field required', [{'type': 'missing', 'loc': loc, 'input': {}}]).errors(include_url=False)[0]
        error['input'] = None
        return error

    def create_body_model(*, fields: Sequence['ModelField'], model_name: str) -> Type[BaseModel]:
        field_params: Dict[str, Tuple[Any, FieldInfo]] = {f.name: (f.field_info.annotation, f.field_info) for f in fields}
        BodyModel: Type[BaseModel] = create_model(model_name, **field_params)
        return BodyModel

    def get_model_fields(model: Type[BaseModel]) -> List['ModelField']:
        return [ModelField(field_info=field_info, name=name) for (name, field_info) in model.model_fields.items()]
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
    
    GetJsonSchemaHandler: Type[Any] = Any
    JsonSchemaValue: Type[Dict[str, Any]] = Dict[str, Any]
    CoreSchema: Type[Any] = Any
    sequence_shapes: Set[int] = {SHAPE_LIST, SHAPE_SET, SHAPE_FROZENSET, SHAPE_TUPLE, SHAPE_SEQUENCE, SHAPE_TUPLE_ELLIPSIS}
    sequence_shape_to_type: Dict[int, Type[Any]] = {SHAPE_LIST: list, SHAPE_SET: set, SHAPE_TUPLE: tuple, SHAPE_SEQUENCE: list, SHAPE_TUPLE_ELLIPSIS: list}

    @dataclass
    class GenerateJsonSchema:
        ref_template: str

    class PydanticSchemaGenerationError(Exception):
        pass

    def with_info_plain_validator_function(function: Callable[..., Any], *, ref: Optional[str] = None, metadata: Any = None, serialization: Any = None) -> Any:
        return {}

    def get_model_definitions(*, flat_models: Set[Union[Type[BaseModel], Type[Enum]]], model_name_map: Dict[Union[Type[BaseModel], Type[Enum]], str]) -> Dict[str, Any]:
        definitions: Dict[str, Dict[str, Any]] = {}
        for model in flat_models:
            m_schema: Dict[str, Any]
            m_definitions: Dict[str, Any]
            m_nested_models: List[Any]
            (m_schema, m_definitions, m_nested_models) = model_process_schema(model, model_name_map=model_name_map, ref_prefix=REF_PREFIX)
            definitions.update(m_definitions)
            model_name: str = model_name_map[model]
            if 'description' in m_schema:
                m_schema['description'] = m_schema['description'].split('\x0c')[0]
            definitions[model_name] = m_schema
        return definitions

    def is_pv1_scalar_field(field: ModelField) -> bool:
        from fastapi import params
        field_info: FieldInfo = field.field_info
        if not (field.shape == SHAPE_SINGLETON and (not lenient_issubclass(field.type_, BaseModel)) and (not lenient_issubclass(field.type_, dict)) and (not field_annotation_is_sequence(field.type_)) and (not is_dataclass(field.type_)) and (not isinstance(field_info, params.Body))):
            return False
        if field.sub_fields:
            if not all((is_pv1_scalar_field(f) for f in field.sub_fields)):
                return False
        return True

    def is_pv1_scalar_sequence_field(field: ModelField) -> bool:
        if field.shape in sequence_shapes and (not lenient_issubclass(field.type_, BaseModel)):
            if field.sub_fields is not None:
                for sub_field in field.sub_fields:
                    if not is_pv1_scalar_field(sub_field):
                        return False
            return True
        if _annotation_is_sequence(field.type_):
            return True
        return False

    def _normalize_errors(errors: Sequence[Any]) -> List[Dict[str, Any]]:
        use_errors: List[Any] = []
        for error in errors:
            if isinstance(error, ErrorWrapper):
                new_errors: List[Dict[str, Any]] = ValidationError(errors=[error], model=RequestErrorModel).errors()
                use_errors.extend(new_errors)
            elif isinstance(error, list):
                use_errors.extend(_normalize_errors(error))
            else:
                use_errors.append(error)
        return use_errors

    def _model_rebuild(model: Type[BaseModel]) -> None:
        model.update_forward_refs()

    def _model_dump(model: BaseModel, mode: Literal['json', 'python'] = 'json', **kwargs: Any) -> Any:
        return model.dict(**kwargs)

    def _get_model_config(model: BaseModel) -> Any:
        return model.__config__

    def get_schema_from_model_field(*, field: ModelField, schema_generator: GenerateJsonSchema, model_name_map: ModelNameMap, field_mapping: Dict[Tuple[ModelField, Literal['validation', 'serialization']], JsonSchemaValue], separate_input_output_schemas: bool = True) -> Dict[str, Any]:
        return field_schema(field, model_name_map=model_name_map, ref_prefix=REF_PREFIX)[0]

    def get_compat_model_name_map(fields: List[ModelField]) -> ModelNameMap:
        models: Set[Type[BaseModel]] = get_flat_models_from_fields(fields, known_models=set())
        return get_model_name_map(models)

    def get_definitions(*, fields: List[ModelField], schema_generator: GenerateJsonSchema, model_name_map: ModelNameMap, separate_input_output_schemas: bool = True) -> Tuple[Dict[Tuple[ModelField, Literal['validation', 'serialization']], JsonSchemaValue], Dict[str, Dict[str, Any]]]:
        models: Set[Type[BaseModel]] = get_flat_models_from_fields(fields, known_models=set())
        return ({}, get_model_definitions(flat_models=models, model_name_map=model_name_map))

    def is_scalar_field(field: ModelField) -> bool:
        return is_pv1_scalar_field(field)

    def is_sequence_field(field: ModelField) -> bool:
        return field.shape in sequence_shapes or _annotation_is_sequence(field.type_)

    def is_scalar_sequence_field(field: ModelField) -> bool:
        return is_pv1_scalar_sequence_field(field)

    def is_bytes_field(field: ModelField) -> bool:
        return lenient_issubclass(field.type_, bytes)

    def is_bytes_sequence_field(field: ModelField) -> bool:
        return field.shape in sequence_shapes and lenient_issubclass(field.type_, bytes)

    def copy_field_info(*, field_info: FieldInfo, annotation: Any) -> FieldInfo:
        return copy(field_info)

    def serialize_sequence_value(*, field: ModelField, value: Any) -> Sequence[Any]:
        return sequence_shape_to_type[field.shape](value)

    def get_missing_field_error(loc: Tuple[str, ...]) -> Dict[str, Any]:
        missing_field_error: ErrorWrapper = ErrorWrapper(MissingError(), loc=loc)
        new_error: ValidationError = ValidationError([missing_field_error], RequestErrorModel)
        return new_error.errors()[0]

    def create_body_model(*, fields: Sequence[ModelField], model_name: str) -> Type[BaseModel]:
        BodyModel: Type[BaseModel] = create_model(model_name)
        for f in fields:
            BodyModel.__fields__[f.name] = f
        return BodyModel

    def get_model_fields(model: Type[BaseModel]) -> List[ModelField]:
        return list(model.__fields__.values())

def _regenerate_error_with_loc