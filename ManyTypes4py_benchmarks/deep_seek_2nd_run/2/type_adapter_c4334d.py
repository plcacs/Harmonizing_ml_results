"""Type adapter specification."""
from __future__ import annotations as _annotations
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import is_dataclass
from types import FrameType
from typing import Any, Generic, Literal, TypeVar, cast, final, overload, Optional, Union, Dict, Tuple, List
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator, Some
from typing_extensions import ParamSpec, is_typeddict
from pydantic.errors import PydanticUserError
from pydantic.main import BaseModel, IncEx
from ._internal import _config, _generate_schema, _mock_val_ser, _namespace_utils, _repr, _typing_extra, _utils
from .config import ConfigDict
from .errors import PydanticUndefinedAnnotation
from .json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaKeyT, JsonSchemaMode, JsonSchemaValue
from .plugin._schema_validator import PluggableSchemaValidator, create_schema_validator

T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')
TypeAdapterT = TypeVar('TypeAdapterT', bound='TypeAdapter[Any]')

def _getattr_no_parents(obj: Any, attribute: str) -> Any:
    """Returns the attribute value without attempting to look up attributes from parent types."""
    if hasattr(obj, '__dict__'):
        try:
            return obj.__dict__[attribute]
        except KeyError:
            pass
    slots = getattr(obj, '__slots__', None)
    if slots is not None and attribute in slots:
        return getattr(obj, attribute)
    else:
        raise AttributeError(attribute)

def _type_has_config(type_: Any) -> bool:
    """Returns whether the type has config."""
    type_ = _typing_extra.annotated_type(type_) or type_
    try:
        return issubclass(type_, BaseModel) or is_dataclass(type_) or is_typeddict(type_)
    except TypeError:
        return False

@final
class TypeAdapter(Generic[T]):
    def __init__(
        self,
        type: type[T],
        *,
        config: Optional[ConfigDict] = None,
        _parent_depth: int = 2,
        module: Optional[str] = None
    ) -> None:
        if _type_has_config(type) and config is not None:
            raise PydanticUserError(
                'Cannot use `config` when the type is a BaseModel, dataclass or TypedDict. These types can have their own config and setting the config via the `config` parameter to TypeAdapter will not override it, thus the `config` you passed to TypeAdapter becomes meaningless, which is probably not what you want.',
                code='type-adapter-config-unused'
            )
        self._type = type
        self._config = config
        self._parent_depth = _parent_depth
        self.pydantic_complete = False
        parent_frame = self._fetch_parent_frame()
        if parent_frame is not None:
            globalns = parent_frame.f_globals
            localns = parent_frame.f_locals if parent_frame.f_locals is not globalns else {}
        else:
            globalns = {}
            localns = {}
        self._module_name = module or cast(str, globalns.get('__name__', ''))
        self._init_core_attrs(
            ns_resolver=_namespace_utils.NsResolver(
                namespaces_tuple=_namespace_utils.NamespacesTuple(locals=localns, globals=globalns),
                parent_namespace=localns
            ),
            force=False
        )

    def _fetch_parent_frame(self) -> Optional[FrameType]:
        frame = sys._getframe(self._parent_depth)
        if frame.f_globals.get('__name__') == 'typing':
            return frame.f_back
        return frame

    def _init_core_attrs(
        self,
        ns_resolver: _namespace_utils.NsResolver,
        force: bool,
        raise_errors: bool = False
    ) -> bool:
        if not force and self._defer_build:
            _mock_val_ser.set_type_adapter_mocks(self)
            self.pydantic_complete = False
            return False
        try:
            self.core_schema = _getattr_no_parents(self._type, '__pydantic_core_schema__')
            self.validator = _getattr_no_parents(self._type, '__pydantic_validator__')
            self.serializer = _getattr_no_parents(self._type, '__pydantic_serializer__')
            if isinstance(self.core_schema, _mock_val_ser.MockCoreSchema) or isinstance(self.validator, _mock_val_ser.MockValSer) or isinstance(self.serializer, _mock_val_ser.MockValSer):
                raise AttributeError()
        except AttributeError:
            config_wrapper = _config.ConfigWrapper(self._config)
            schema_generator = _generate_schema.GenerateSchema(config_wrapper, ns_resolver=ns_resolver)
            try:
                core_schema = schema_generator.generate_schema(self._type)
            except PydanticUndefinedAnnotation:
                if raise_errors:
                    raise
                _mock_val_ser.set_type_adapter_mocks(self)
                return False
            try:
                self.core_schema = schema_generator.clean_schema(core_schema)
            except _generate_schema.InvalidSchemaError:
                _mock_val_ser.set_type_adapter_mocks(self)
                return False
            core_config = config_wrapper.core_config(None)
            self.validator = create_schema_validator(
                schema=self.core_schema,
                schema_type=self._type,
                schema_type_module=self._module_name,
                schema_type_name=str(self._type),
                schema_kind='TypeAdapter',
                config=core_config,
                plugin_settings=config_wrapper.plugin_settings
            )
            self.serializer = SchemaSerializer(self.core_schema, core_config)
        self.pydantic_complete = True
        return True

    @property
    def _defer_build(self) -> bool:
        config = self._config if self._config is not None else self._model_config
        if config:
            return config.get('defer_build') is True
        return False

    @property
    def _model_config(self) -> Optional[ConfigDict]:
        type_ = _typing_extra.annotated_type(self._type) or self._type
        if _utils.lenient_issubclass(type_, BaseModel):
            return type_.model_config
        return getattr(type_, '__pydantic_config__', None)

    def __repr__(self) -> str:
        return f'TypeAdapter({_repr.display_as_type(self._type)})'

    def rebuild(
        self,
        *,
        force: bool = False,
        raise_errors: bool = True,
        _parent_namespace_depth: int = 2,
        _types_namespace: Optional[Dict[str, Any]] = None
    ) -> Optional[bool]:
        if not force and self.pydantic_complete:
            return None
        if _types_namespace is not None:
            rebuild_ns = _types_namespace
        elif _parent_namespace_depth > 0:
            rebuild_ns = _typing_extra.parent_frame_namespace(parent_depth=_parent_namespace_depth, force=True) or {}
        else:
            rebuild_ns = {}
        globalns = sys._getframe(max(_parent_namespace_depth - 1, 1)).f_globals
        ns_resolver = _namespace_utils.NsResolver(
            namespaces_tuple=_namespace_utils.NamespacesTuple(locals=rebuild_ns, globals=globalns),
            parent_namespace=rebuild_ns
        )
        return self._init_core_attrs(ns_resolver=ns_resolver, force=True, raise_errors=raise_errors)

    def validate_python(
        self,
        object: Any,
        /,
        *,
        strict: Optional[bool] = None,
        from_attributes: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
        experimental_allow_partial: Union[bool, Literal['off', 'on', 'trailing-strings']] = False
    ) -> T:
        return self.validator.validate_python(
            object,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            allow_partial=experimental_allow_partial
        )

    def validate_json(
        self,
        data: Union[str, bytes],
        /,
        *,
        strict: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
        experimental_allow_partial: Union[bool, Literal['off', 'on', 'trailing-strings']] = False
    ) -> T:
        return self.validator.validate_json(
            data,
            strict=strict,
            context=context,
            allow_partial=experimental_allow_partial
        )

    def validate_strings(
        self,
        obj: Any,
        /,
        *,
        strict: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
        experimental_allow_partial: Union[bool, Literal['off', 'on', 'trailing-strings']] = False
    ) -> T:
        return self.validator.validate_strings(
            obj,
            strict=strict,
            context=context,
            allow_partial=experimental_allow_partial
        )

    def get_default_value(
        self,
        *,
        strict: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Some[T]]:
        return self.validator.get_default_value(strict=strict, context=context)

    def dump_python(
        self,
        instance: T,
        /,
        *,
        mode: Literal['python', 'json'] = 'python',
        include: Optional[IncEx] = None,
        exclude: Optional[IncEx] = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: Union[bool, Literal['none', 'warn', 'error']] = True,
        fallback: Optional[Callable[[Any], Any]] = None,
        serialize_as_any: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        return self.serializer.to_python(
            instance,
            mode=mode,
            by_alias=by_alias,
            include=include,
            exclude=exclude,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            fallback=fallback,
            serialize_as_any=serialize_as_any,
            context=context
        )

    def dump_json(
        self,
        instance: T,
        /,
        *,
        indent: Optional[int] = None,
        include: Optional[IncEx] = None,
        exclude: Optional[IncEx] = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: Union[bool, Literal['none', 'warn', 'error']] = True,
        fallback: Optional[Callable[[Any], Any]] = None,
        serialize_as_any: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> bytes:
        return self.serializer.to_json(
            instance,
            indent=indent,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            fallback=fallback,
            serialize_as_any=serialize_as_any,
            context=context
        )

    def json_schema(
        self,
        *,
        by_alias: bool = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = 'validation'
    ) -> JsonSchemaValue:
        schema_generator_instance = schema_generator(by_alias=by_alias, ref_template=ref_template)
        if isinstance(self.core_schema, _mock_val_ser.MockCoreSchema):
            self.core_schema.rebuild()
            assert not isinstance(self.core_schema, _mock_val_ser.MockCoreSchema), 'this is a bug! please report it'
        return schema_generator_instance.generate(self.core_schema, mode=mode)

    @staticmethod
    def json_schemas(
        inputs: List[Tuple[JsonSchemaKeyT, JsonSchemaMode, 'TypeAdapter[Any]']],
        *,
        by_alias: bool = True,
        title: Optional[str] = None,
        description: Optional[str] = None,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema
    ) -> Tuple[Dict[Tuple[JsonSchemaKeyT, JsonSchemaMode], JsonSchemaValue], JsonSchemaValue]:
        schema_generator_instance = schema_generator(by_alias=by_alias, ref_template=ref_template)
        inputs_: List[Tuple[JsonSchemaKeyT, JsonSchemaMode, CoreSchema]] = []
        for key, mode, adapter in inputs:
            if isinstance(adapter.core_schema, _mock_val_ser.MockCoreSchema):
                adapter.core_schema.rebuild()
                assert not isinstance(adapter.core_schema, _mock_val_ser.MockCoreSchema), 'this is a bug! please report it'
            inputs_.append((key, mode, adapter.core_schema))
        json_schemas_map, definitions = schema_generator_instance.generate_definitions(inputs_)
        json_schema: Dict[str, Any] = {}
        if definitions:
            json_schema['$defs'] = definitions
        if title:
            json_schema['title'] = title
        if description:
            json_schema['description'] = description
        return (json_schemas_map, json_schema)
