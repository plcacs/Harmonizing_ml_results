from __future__ import annotations as _annotations
import operator
import sys
import types
import typing
import warnings
from collections.abc import Generator, Mapping
from copy import copy, deepcopy
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Literal, Optional, Tuple, Type, TypeVar, Union, cast, overload
import pydantic_core
import typing_extensions
from pydantic_core import PydanticUndefined, ValidationError
from typing_extensions import Self, TypeAlias, Unpack
from . import PydanticDeprecatedSince20, PydanticDeprecatedSince211
from ._internal import _config, _decorators, _fields, _forward_ref, _generics, _mock_val_ser, _model_construction, _namespace_utils, _repr, _typing_extra, _utils
from ._migration import getattr_migration
from .aliases import AliasChoices, AliasPath
from .annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from .config import ConfigDict
from .errors import PydanticUndefinedAnnotation, PydanticUserError
from .json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue, model_json_schema
from .plugin._schema_validator import PluggableSchemaValidator

if TYPE_CHECKING:
    from inspect import Signature
    from pathlib import Path
    from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator
    from ._internal._namespace_utils import MappingNamespace
    from ._internal._utils import AbstractSetIntStr, MappingIntStrAny
    from .deprecated.parse import Protocol as DeprecatedParseProtocol
    from .fields import ComputedFieldInfo, FieldInfo, ModelPrivateAttr
else:
    DeprecationWarning = PydanticDeprecatedSince20

__all__ = ('BaseModel', 'create_model')

TupleGenerator = Generator[Tuple[str, Any], None, None]
IncEx = Union[set[int], set[str], Mapping[int, Union['IncEx', bool]], Mapping[str, Union['IncEx', bool]]]
_object_setattr = _model_construction.object_setattr

def _check_frozen(model_cls: Type['BaseModel'], name: str, value: Any) -> None:
    if model_cls.model_config.get('frozen'):
        error_type = 'frozen_instance'
    elif getattr(model_cls.__pydantic_fields__.get(name), 'frozen', False):
        error_type = 'frozen_field'
    else:
        return
    raise ValidationError.from_exception_data(model_cls.__name__, [{'type': error_type, 'loc': (name,), 'input': value}])

def _model_field_setattr_handler(model: 'BaseModel', name: str, val: Any) -> None:
    model.__dict__[name] = val
    model.__pydantic_fields_set__.add(name)

_SIMPLE_SETATTR_HANDLERS = {
    'model_field': _model_field_setattr_handler,
    'validate_assignment': lambda model, name, val: model.__pydantic_validator__.validate_assignment(model, name, val),
    'private': lambda model, name, val: model.__pydantic_private__.__setitem__(name, val),
    'cached_property': lambda model, name, val: model.__dict__.__setitem__(name, val),
    'extra_known': lambda model, name, val: _object_setattr(model, name, val)
}

class BaseModel(metaclass=_model_construction.ModelMetaclass):
    model_config: ClassVar[ConfigDict] = ConfigDict()
    __class_vars__: ClassVar[set[str]]
    __private_attributes__: ClassVar[dict[str, ModelPrivateAttr]]
    __signature__: ClassVar[Signature]
    
    __pydantic_complete__: ClassVar[bool] = False
    __pydantic_core_schema__: ClassVar[CoreSchema]
    __pydantic_custom_init__: ClassVar[bool]
    __pydantic_decorators__: ClassVar[_decorators.DecoratorInfos] = _decorators.DecoratorInfos()
    __pydantic_generic_metadata__: ClassVar[dict[str, Any]]
    __pydantic_parent_namespace__: ClassVar[Optional[dict[str, Any]]] = None
    __pydantic_post_init__: ClassVar[Optional[str]]
    __pydantic_root_model__: ClassVar[bool] = False
    __pydantic_serializer__: ClassVar[SchemaSerializer]
    __pydantic_validator__: ClassVar[SchemaValidator]
    
    __pydantic_fields__: ClassVar[dict[str, FieldInfo]]
    __pydantic_setattr_handlers__: ClassVar[dict[str, Callable[['BaseModel', str, Any], None]]]
    __pydantic_computed_fields__: ClassVar[dict[str, ComputedFieldInfo]]
    
    __pydantic_extra__: Optional[dict[str, Any]] = _model_construction.NoInitField(init=False)
    __pydantic_fields_set__: set[str] = _model_construction.NoInitField(init=False)
    __pydantic_private__: Optional[dict[str, Any]] = _model_construction.NoInitField(init=False)
    
    if not TYPE_CHECKING:
        __pydantic_core_schema__ = _mock_val_ser.MockCoreSchema('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', code='base-model-instantiated')
        __pydantic_validator__ = _mock_val_ser.MockValSer('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', val_or_ser='validator', code='base-model-instantiated')
        __pydantic_serializer__ = _mock_val_ser.MockValSer('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', val_or_ser='serializer', code='base-model-instantiated')
    
    __slots__ = ('__dict__', '__pydantic_fields_set__', '__pydantic_extra__', '__pydantic_private__')

    def __init__(self, /, **data: Any) -> None:
        __tracebackhide__ = True
        validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
        if self is not validated_self:
            warnings.warn("A custom validator is returning a value other than `self`.\nReturning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.\nSee the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.", stacklevel=2)
    
    __init__.__pydantic_base_init__ = True

    @_utils.deprecated_instance_property
    @classmethod
    def model_fields(cls) -> dict[str, FieldInfo]:
        return getattr(cls, '__pydantic_fields__', {})

    @_utils.deprecated_instance_property
    @classmethod
    def model_computed_fields(cls) -> dict[str, ComputedFieldInfo]:
        return getattr(cls, '__pydantic_computed_fields__', {})

    @property
    def model_extra(self) -> Optional[dict[str, Any]]:
        return self.__pydantic_extra__

    @property
    def model_fields_set(self) -> set[str]:
        return self.__pydantic_fields_set__

    @classmethod
    def model_construct(cls: Type[ModelT], _fields_set: Optional[set[str]] = None, **values: Any) -> ModelT:
        m = cls.__new__(cls)
        fields_values = {}
        fields_set = set()
        for name, field in cls.__pydantic_fields__.items():
            if field.alias is not None and field.alias in values:
                fields_values[name] = values.pop(field.alias)
                fields_set.add(name)
            if name not in fields_set and field.validation_alias is not None:
                validation_aliases = field.validation_alias.choices if isinstance(field.validation_alias, AliasChoices) else [field.validation_alias]
                for alias in validation_aliases:
                    if isinstance(alias, str) and alias in values:
                        fields_values[name] = values.pop(alias)
                        fields_set.add(name)
                        break
                    elif isinstance(alias, AliasPath):
                        value = alias.search_dict_for_path(values)
                        if value is not PydanticUndefined:
                            fields_values[name] = value
                            fields_set.add(name)
                            break
            if name not in fields_set:
                if name in values:
                    fields_values[name] = values.pop(name)
                    fields_set.add(name)
                elif not field.is_required():
                    fields_values[name] = field.get_default(call_default_factory=True, validated_data=fields_values)
        if _fields_set is None:
            _fields_set = fields_set
        _extra = values if cls.model_config.get('extra') == 'allow' else None
        _object_setattr(m, '__dict__', fields_values)
        _object_setattr(m, '__pydantic_fields_set__', _fields_set)
        if not cls.__pydantic_root_model__:
            _object_setattr(m, '__pydantic_extra__', _extra)
        if cls.__pydantic_post_init__:
            m.model_post_init(None)
            if hasattr(m, '__pydantic_private__') and m.__pydantic_private__ is not None:
                for k, v in values.items():
                    if k in m.__private_attributes__:
                        m.__pydantic_private__[k] = v
        elif not cls.__pydantic_root_model__:
            _object_setattr(m, '__pydantic_private__', None)
        return m

    def model_copy(self: ModelT, *, update: Optional[dict[str, Any]] = None, deep: bool = False) -> ModelT:
        copied = self.__deepcopy__() if deep else self.__copy__()
        if update:
            if self.model_config.get('extra') == 'allow':
                for k, v in update.items():
                    if k in self.__pydantic_fields__:
                        copied.__dict__[k] = v
                    else:
                        if copied.__pydantic_extra__ is None:
                            copied.__pydantic_extra__ = {}
                        copied.__pydantic_extra__[k] = v
            else:
                copied.__dict__.update(update)
            copied.__pydantic_fields_set__.update(update.keys())
        return copied

    def model_dump(self, *, mode: Literal['json', 'python'] = 'python', include: Optional[IncEx] = None, exclude: Optional[IncEx] = None, context: Optional[dict[str, Any]] = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, round_trip: bool = False, warnings: Union[bool, Literal['none', 'warn', 'error']] = True, fallback: Optional[Callable[[Any], Any]] = None, serialize_as_any: bool = False) -> dict[str, Any]:
        return self.__pydantic_serializer__.to_python(self, mode=mode, by_alias=by_alias, include=include, exclude=exclude, context=context, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings, fallback=fallback, serialize_as_any=serialize_as_any)

    def model_dump_json(self, *, indent: Optional[int] = None, include: Optional[IncEx] = None, exclude: Optional[IncEx] = None, context: Optional[dict[str, Any]] = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, round_trip: bool = False, warnings: Union[bool, Literal['none', 'warn', 'error']] = True, fallback: Optional[Callable[[Any], Any]] = None, serialize_as_any: bool = False) -> str:
        return self.__pydantic_serializer__.to_json(self, indent=indent, include=include, exclude=exclude, context=context, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings, fallback=fallback, serialize_as_any=serialize_as_any).decode()

    @classmethod
    def model_json_schema(cls, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE, schema_generator: Type[GenerateJsonSchema] = GenerateJsonSchema, mode: JsonSchemaMode = 'validation') -> dict[str, Any]:
        return model_json_schema(cls, by_alias=by_alias, ref_template=ref_template, schema_generator=schema_generator, mode=mode)

    @classmethod
    def model_parametrized_name(cls, params: tuple[type[Any], ...]) -> str:
        if not issubclass(cls, typing.Generic):
            raise TypeError('Concrete names should only be generated for generic models.')
        param_names = [param if isinstance(param, str) else _repr.display_as_type(param) for param in params]
        params_component = ', '.join(param_names)
        return f'{cls.__name__}[{params_component}]'

    def model_post_init(self, context: Any, /) -> None:
        pass

    @classmethod
    def model_rebuild(cls, *, force: bool = False, raise_errors: bool = True, _parent_namespace_depth: int = 2, _types_namespace: Optional[dict[str, Any]] = None) -> Optional[bool]:
        if not force and cls.__pydantic_complete__:
            return None
        for attr in ('__pydantic_core_schema__', '__pydantic_validator__', '__pydantic_serializer__'):
            if attr in cls.__dict__:
                delattr(cls, attr)
        cls.__pydantic_complete__ = False
        if _types_namespace is not None:
            rebuild_ns = _types_namespace
        elif _parent_namespace_depth > 0:
            rebuild_ns = _typing_extra.parent_frame_namespace(parent_depth=_parent_namespace_depth, force=True) or {}
        else:
            rebuild_ns = {}
        parent_ns = _model_construction.unpack_lenient_weakvaluedict(cls.__pydantic_parent_namespace__) or {}
        ns_resolver = _namespace_utils.NsResolver(parent_namespace={**rebuild_ns, **parent_ns})
        if not cls.__pydantic_fields_complete__:
            typevars_map = _generics.get_model_typevars_map(cls)
            try:
                cls.__pydantic_fields__ = _fields.rebuild_model_fields(cls, ns_resolver=ns_resolver, typevars_map=typevars_map)
            except NameError as e:
                exc = PydanticUndefinedAnnotation.from_name_error(e)
                _mock_val_ser.set_model_mocks(cls, f'`{exc.name}`')
                if raise_errors:
                    raise exc from e
            if not raise_errors and (not cls.__pydantic_fields_complete__):
                return False
            assert cls.__pydantic_fields_complete__
        return _model_construction.complete_model_class(cls, _config.ConfigWrapper(cls.model_config, check=False), raise_errors=raise_errors, ns_resolver=ns_resolver)

    @classmethod
    def model_validate(cls: Type[ModelT], obj: Any, *, strict: Optional[bool] = None, from_attributes: Optional[bool] = None, context: Optional[dict[str, Any]] = None) -> ModelT:
        __tracebackhide__ = True
        return cls.__pydantic_validator__.validate_python(obj, strict=strict, from_attributes=from_attributes, context=context)

    @classmethod
    def model_validate_json(cls: Type[ModelT], json_data: str, *, strict: Optional[bool] = None, context: Optional[dict[str, Any]] = None) -> ModelT:
        __tracebackhide__ = True
        return cls.__pydantic_validator__.validate_json(json_data, strict=strict, context=context)

    @classmethod
    def model_validate_strings(cls: Type[ModelT], obj: Any, *, strict: Optional[bool] = None, context: Optional[dict[str, Any]] = None) -> ModelT:
        __tracebackhide__ = True
        return cls.__pydantic_validator__.validate_strings(obj, strict=strict, context=context)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler, /) -> CoreSchema:
        warnings.warn('The `__get_pydantic_core_schema__` method of the `BaseModel` class is deprecated. If you are calling `super().__get_pydantic_core_schema__` when overriding the method on a Pydantic model, consider using `handler(source)` instead. However, note that overriding this method on models can lead to unexpected side effects.', PydanticDeprecatedSince211, stacklevel=2)
        schema = cls.__dict__.get('__pydantic_core_schema__')
        if schema is not None and (not isinstance(schema, _mock_val_ser.MockCoreSchema)):
            return cls.__pydantic_core_schema__
        return handler(source)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler, /) -> JsonSchemaValue:
        return handler(core_schema)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        pass

    def __class_getitem__(cls, typevar_values