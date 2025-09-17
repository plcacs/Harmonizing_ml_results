from __future__ import annotations
import operator
import sys
import types
import typing
import warnings
from collections.abc import Generator, Mapping, Iterator
from copy import copy, deepcopy
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Iterable, Optional, Set, Tuple, Type, TypeVar, Union, cast, overload
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

TupleGenerator: TypeAlias = Generator[tuple[str, Any], None, None]
IncEx: TypeAlias = Union[set[int], set[str], Mapping[int, Union['IncEx', bool]], Mapping[str, Union['IncEx', bool]]]
_object_setattr: Callable[[Any, str, Any], None] = _model_construction.object_setattr

def _check_frozen(model_cls: type[BaseModel], name: str, value: Any) -> None:
    if model_cls.model_config.get('frozen'):
        error_type: str = 'frozen_instance'
    elif getattr(model_cls.__pydantic_fields__.get(name), 'frozen', False):
        error_type = 'frozen_field'
    else:
        return
    raise ValidationError.from_exception_data(model_cls.__name__, [{'type': error_type, 'loc': (name,), 'input': value}])

def _model_field_setattr_handler(model: BaseModel, name: str, val: Any) -> None:
    model.__dict__[name] = val
    model.__pydantic_fields_set__.add(name)

_SIMPLE_SETATTR_HANDLERS: Dict[str, Callable[[BaseModel, str, Any], None]] = {
    'model_field': _model_field_setattr_handler,
    'validate_assignment': lambda model, name, val: model.__pydantic_validator__.validate_assignment(model, name, val),
    'private': lambda model, name, val: model.__pydantic_private__.__setitem__(name, val),
    'cached_property': lambda model, name, val: model.__dict__.__setitem__(name, val),
    'extra_known': lambda model, name, val: _object_setattr(model, name, val)
}

class BaseModel(metaclass=_model_construction.ModelMetaclass):
    model_config: ClassVar[ConfigDict] = ConfigDict()
    __pydantic_complete__: ClassVar[bool] = False
    __pydantic_decorators__: ClassVar[_decorators.DecoratorInfos] = _decorators.DecoratorInfos()
    __pydantic_parent_namespace__: ClassVar[Optional[Any]] = None
    __pydantic_root_model__: ClassVar[bool] = False
    __pydantic_extra__: Any = _model_construction.NoInitField(init=False)
    __pydantic_fields_set__: Any = _model_construction.NoInitField(init=False)
    __pydantic_private__: Any = _model_construction.NoInitField(init=False)
    if not TYPE_CHECKING:
        __pydantic_core_schema__ = _mock_val_ser.MockCoreSchema('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', code='base-model-instantiated')
        __pydantic_validator__ = _mock_val_ser.MockValSer('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', val_or_ser='validator', code='base-model-instantiated')
        __pydantic_serializer__ = _mock_val_ser.MockValSer('Pydantic models should inherit from BaseModel, BaseModel cannot be instantiated directly', val_or_ser='serializer', code='base-model-instantiated')
    __slots__ = ('__dict__', '__pydantic_fields_set__', '__pydantic_extra__', '__pydantic_private__')

    def __init__(self, /, **data: Any) -> None:
        __tracebackhide__ = True
        validated_self: Any = self.__pydantic_validator__.validate_python(data, self_instance=self)
        if self is not validated_self:
            warnings.warn(
                "A custom validator is returning a value other than `self`.\nReturning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.\nSee the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.",
                stacklevel=2
            )
    __init__.__pydantic_base_init__ = True

    @_utils.deprecated_instance_property
    @classmethod
    def model_fields(cls) -> Mapping[str, Any]:
        return getattr(cls, '__pydantic_fields__', {})

    @_utils.deprecated_instance_property
    @classmethod
    def model_computed_fields(cls) -> Mapping[str, Any]:
        return getattr(cls, '__pydantic_computed_fields__', {})

    @property
    def model_extra(self) -> Any:
        return self.__pydantic_extra__

    @property
    def model_fields_set(self) -> Set[str]:
        return self.__pydantic_fields_set__

    @classmethod
    def model_construct(cls: Type[Self], _fields_set: Optional[Set[str]] = None, **values: Any) -> Self:
        m: Self = cls.__new__(cls)
        fields_values: Dict[str, Any] = {}
        fields_set: Set[str] = set()
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
        _extra: Optional[Dict[str, Any]] = values if cls.model_config.get('extra') == 'allow' else None
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

    def model_copy(self, *, update: Optional[Dict[str, Any]] = None, deep: bool = False) -> Self:
        copied: Self = self.__deepcopy__() if deep else self.__copy__()
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

    def model_dump(
        self,
        *,
        mode: str = 'python',
        include: Optional[Any] = None,
        exclude: Optional[Any] = None,
        context: Any = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: Union[bool, str] = True,
        fallback: Optional[Callable[[Any], Any]] = None,
        serialize_as_any: bool = False
    ) -> Dict[str, Any]:
        return self.__pydantic_serializer__.to_python(
            self,
            mode=mode,
            by_alias=by_alias,
            include=include,
            exclude=exclude,
            context=context,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            fallback=fallback,
            serialize_as_any=serialize_as_any,
        )

    def model_dump_json(
        self,
        *,
        indent: Optional[int] = None,
        include: Optional[Any] = None,
        exclude: Optional[Any] = None,
        context: Any = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: Union[bool, str] = True,
        fallback: Optional[Callable[[Any], Any]] = None,
        serialize_as_any: bool = False
    ) -> str:
        return self.__pydantic_serializer__.to_json(
            self,
            indent=indent,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings,
            fallback=fallback,
            serialize_as_any=serialize_as_any,
        ).decode()

    @classmethod
    def model_json_schema(
        cls, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE, schema_generator: Type[GenerateJsonSchema] = GenerateJsonSchema, mode: str = 'validation'
    ) -> Any:
        return model_json_schema(cls, by_alias=by_alias, ref_template=ref_template, schema_generator=schema_generator, mode=mode)

    @classmethod
    def model_parametrized_name(cls, params: Tuple[Any, ...]) -> str:
        if not issubclass(cls, typing.Generic):
            raise TypeError('Concrete names should only be generated for generic models.')
        param_names = [param if isinstance(param, str) else _repr.display_as_type(param) for param in params]
        params_component = ', '.join(param_names)
        return f'{cls.__name__}[{params_component}]'

    def model_post_init(self, context: Any, /) -> None:
        pass

    @classmethod
    def model_rebuild(
        cls,
        *,
        force: bool = False,
        raise_errors: bool = True,
        _parent_namespace_depth: int = 2,
        _types_namespace: Optional[Any] = None
    ) -> Optional[bool]:
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
    def model_validate(cls: Type[Self], obj: Any, *, strict: Optional[bool] = None, from_attributes: Optional[bool] = None, context: Any = None) -> Self:
        __tracebackhide__ = True
        return cls.__pydantic_validator__.validate_python(obj, strict=strict, from_attributes=from_attributes, context=context)

    @classmethod
    def model_validate_json(cls: Type[Self], json_data: Any, *, strict: Optional[bool] = None, context: Any = None) -> Self:
        __tracebackhide__ = True
        return cls.__pydantic_validator__.validate_json(json_data, strict=strict, context=context)

    @classmethod
    def model_validate_strings(cls: Type[Self], obj: Any, *, strict: Optional[bool] = None, context: Any = None) -> Self:
        __tracebackhide__ = True
        return cls.__pydantic_validator__.validate_strings(obj, strict=strict, context=context)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Callable[[Any], Any], /) -> Any:
        warnings.warn(
            'The `__get_pydantic_core_schema__` method of the `BaseModel` class is deprecated. If you are calling `super().__get_pydantic_core_schema__` when overriding the method on a Pydantic model, consider using `handler(source)` instead. However, note that overriding this method on models can lead to unexpected side effects.',
            PydanticDeprecatedSince211,
            stacklevel=2
        )
        schema = cls.__dict__.get('__pydantic_core_schema__')
        if schema is not None and (not isinstance(schema, _mock_val_ser.MockCoreSchema)):
            return cls.__pydantic_core_schema__
        return handler(source)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Callable[[Any], Any], /) -> Any:
        return handler(core_schema)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        pass

    def __class_getitem__(cls, typevar_values: Any) -> Any:
        cached = _generics.get_cached_generic_type_early(cls, typevar_values)
        if cached is not None:
            return cached
        if cls is BaseModel:
            raise TypeError('Type parameters should be placed on typing.Generic, not BaseModel')
        if not hasattr(cls, '__parameters__'):
            raise TypeError(f'{cls} cannot be parametrized because it does not inherit from typing.Generic')
        if not cls.__pydantic_generic_metadata__['parameters'] and typing.Generic not in cls.__bases__:
            raise TypeError(f'{cls} is not a generic class')
        if not isinstance(typevar_values, tuple):
            typevar_values = (typevar_values,)
        typevars_map = _generics.map_generic_model_arguments(cls, typevar_values)
        typevar_values = tuple((v for v in typevars_map.values()))
        if _utils.all_identical(typevars_map.keys(), typevars_map.values()) and typevars_map:
            submodel = cls
            _generics.set_cached_generic_type(cls, typevar_values, submodel)
        else:
            parent_args = cls.__pydantic_generic_metadata__['args']
            if not parent_args:
                args = typevar_values
            else:
                args = tuple((_generics.replace_types(arg, typevars_map) for arg in parent_args))
            origin = cls.__pydantic_generic_metadata__['origin'] or cls
            model_name = origin.model_parametrized_name(args)
            params = tuple({param: None for param in _generics.iter_contained_typevars(typevars_map.values())})
            with _generics.generic_recursion_self_type(origin, args) as maybe_self_type:
                cached = _generics.get_cached_generic_type_late(cls, typevar_values, origin, args)
                if cached is not None:
                    return cached
                if maybe_self_type is not None:
                    return maybe_self_type
                try:
                    parent_ns = _typing_extra.parent_frame_namespace(parent_depth=2) or {}
                    origin.model_rebuild(_types_namespace=parent_ns)
                except PydanticUndefinedAnnotation:
                    pass
                submodel = _generics.create_generic_submodel(model_name, origin, args, params)
                if len(_generics.recursively_defined_type_refs()) == 1:
                    _generics.set_cached_generic_type(cls, typevar_values, submodel, origin, args)
        return submodel

    def __copy__(self) -> Self:
        cls: Type[Self] = type(self)
        m: Self = cls.__new__(cls)
        _object_setattr(m, '__dict__', copy(self.__dict__))
        _object_setattr(m, '__pydantic_extra__', copy(self.__pydantic_extra__))
        _object_setattr(m, '__pydantic_fields_set__', copy(self.__pydantic_fields_set__))
        if not hasattr(self, '__pydantic_private__') or self.__pydantic_private__ is None:
            _object_setattr(m, '__pydantic_private__', None)
        else:
            _object_setattr(m, '__pydantic_private__', {k: v for k, v in self.__pydantic_private__.items() if v is not PydanticUndefined})
        return m

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> Self:
        cls: Type[Self] = type(self)
        m: Self = cls.__new__(cls)
        _object_setattr(m, '__dict__', deepcopy(self.__dict__, memo=memo))
        _object_setattr(m, '__pydantic_extra__', deepcopy(self.__pydantic_extra__, memo=memo))
        _object_setattr(m, '__pydantic_fields_set__', copy(self.__pydantic_fields_set__))
        if not hasattr(self, '__pydantic_private__') or self.__pydantic_private__ is None:
            _object_setattr(m, '__pydantic_private__', None)
        else:
            _object_setattr(m, '__pydantic_private__', deepcopy({k: v for k, v in self.__pydantic_private__.items() if v is not PydanticUndefined}, memo=memo))
        return m

    if not TYPE_CHECKING:

        def __getattr__(self, item: str) -> Any:
            private_attributes = object.__getattribute__(self, '__private_attributes__')
            if item in private_attributes:
                attribute = private_attributes[item]
                if hasattr(attribute, '__get__'):
                    return attribute.__get__(self, type(self))
                try:
                    return self.__pydantic_private__[item]
                except KeyError as exc:
                    raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}') from exc
            else:
                try:
                    pydantic_extra = object.__getattribute__(self, '__pydantic_extra__')
                except AttributeError:
                    pydantic_extra = None
                if pydantic_extra:
                    try:
                        return pydantic_extra[item]
                    except KeyError as exc:
                        raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}') from exc
                elif hasattr(self.__class__, item):
                    return super().__getattribute__(item)
                else:
                    raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')

        def __setattr__(self, name: str, value: Any) -> None:
            if (setattr_handler := self.__pydantic_setattr_handlers__.get(name)) is not None:
                setattr_handler(self, name, value)
            elif (setattr_handler := self._setattr_handler(name, value)) is not None:
                setattr_handler(self, name, value)
                self.__pydantic_setattr_handlers__[name] = setattr_handler

        def _setattr_handler(self, name: str, value: Any) -> Optional[Callable[[BaseModel, str, Any], None]]:
            cls = self.__class__
            if name in cls.__class_vars__:
                raise AttributeError(f'{name!r} is a ClassVar of `{cls.__name__}` and cannot be set on an instance. If you want to set a value on the class, use `{cls.__name__}.{name} = value`.')
            elif not _fields.is_valid_field_name(name):
                if (attribute := cls.__private_attributes__.get(name)) is not None:
                    if hasattr(attribute, '__set__'):
                        return lambda model, _name, val: attribute.__set__(model, val)
                    else:
                        return _SIMPLE_SETATTR_HANDLERS['private']
                else:
                    _object_setattr(self, name, value)
                    return None
            attr = getattr(cls, name, None)
            if isinstance(attr, cached_property):
                return _SIMPLE_SETATTR_HANDLERS['cached_property']
            _check_frozen(cls, name, value)
            if isinstance(attr, property):
                return lambda model, _name, val: attr.__set__(model, val)
            elif cls.model_config.get('validate_assignment'):
                return _SIMPLE_SETATTR_HANDLERS['validate_assignment']
            elif name not in cls.__pydantic_fields__:
                if cls.model_config.get('extra') != 'allow':
                    raise ValueError(f'"{cls.__name__}" object has no field "{name}"')
                elif attr is None:
                    self.__pydantic_extra__[name] = value
                    return None
                else:
                    return _SIMPLE_SETATTR_HANDLERS['extra_known']
            else:
                return _SIMPLE_SETATTR_HANDLERS['model_field']

        def __delattr__(self, item: str) -> None:
            cls = self.__class__
            if item in self.__private_attributes__:
                attribute = self.__private_attributes__[item]
                if hasattr(attribute, '__delete__'):
                    attribute.__delete__(self)
                    return
                try:
                    del self.__pydantic_private__[item]
                    return
                except KeyError as exc:
                    raise AttributeError(f'{cls.__name__!r} object has no attribute {item!r}') from exc
            attr = getattr(cls, item, None)
            if isinstance(attr, cached_property):
                return object.__delattr__(self, item)
            _check_frozen(cls, name=item, value=None)
            if item in self.__pydantic_fields__:
                object.__delattr__(self, item)
            elif self.__pydantic_extra__ is not None and item in self.__pydantic_extra__:
                del self.__pydantic_extra__[item]
            else:
                try:
                    object.__delattr__(self, item)
                except AttributeError:
                    raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')

        def __replace__(self, **changes: Any) -> Self:
            return self.model_copy(update=changes)

    def __getstate__(self) -> Dict[str, Any]:
        private = self.__pydantic_private__
        if private:
            private = {k: v for k, v in private.items() if v is not PydanticUndefined}
        return {
            '__dict__': self.__dict__,
            '__pydantic_extra__': self.__pydantic_extra__,
            '__pydantic_fields_set__': self.__pydantic_fields_set__,
            '__pydantic_private__': private,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        _object_setattr(self, '__pydantic_fields_set__', state.get('__pydantic_fields_set__', {}))
        _object_setattr(self, '__pydantic_extra__', state.get('__pydantic_extra__', {}))
        _object_setattr(self, '__pydantic_private__', state.get('__pydantic_private__', {}))
        _object_setattr(self, '__dict__', state.get('__dict__', {}))
    if not TYPE_CHECKING:

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, BaseModel):
                self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
                other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__
                if not (self_type == other_type and getattr(self, '__pydantic_private__', None) == getattr(other, '__pydantic_private__', None) and (self.__pydantic_extra__ == other.__pydantic_extra__)):
                    return False
                if self.__dict__ == other.__dict__:
                    return True
                model_fields = type(self).__pydantic_fields__.keys()
                if self.__dict__.keys() <= model_fields and other.__dict__.keys() <= model_fields:
                    return False
                getter = operator.itemgetter(*model_fields) if model_fields else lambda _: _utils._SENTINEL
                try:
                    return getter(self.__dict__) == getter(other.__dict__)
                except KeyError:
                    self_fields_proxy = _utils.SafeGetItemProxy(self.__dict__)
                    other_fields_proxy = _utils.SafeGetItemProxy(other.__dict__)
                    return getter(self_fields_proxy) == getter(other_fields_proxy)
            else:
                return NotImplemented
    if TYPE_CHECKING:

        def __init_subclass__(cls, **kwargs: Any) -> None:
            pass

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        yield from [(k, v) for k, v in self.__dict__.items() if not k.startswith('_')]
        extra = self.__pydantic_extra__
        if extra:
            yield from extra.items()

    def __repr__(self) -> str:
        return f'{self.__repr_name__()}({self.__repr_str__(", ")})'

    def __repr_args__(self) -> Iterable[tuple[str, Any]]:
        for k, v in self.__dict__.items():
            field = self.__pydantic_fields__.get(k)
            if field and field.repr:
                if v is not self:
                    yield (k, v)
                else:
                    yield (k, self.__repr_recursion__(v))
        try:
            pydantic_extra = object.__getattribute__(self, '__pydantic_extra__')
        except AttributeError:
            pydantic_extra = None
        if pydantic_extra is not None:
            yield from ((k, v) for k, v in pydantic_extra.items())
        yield from ((k, getattr(self, k)) for k, v in self.__pydantic_computed_fields__.items() if v.repr)
    __repr_name__ = _repr.Representation.__repr_name__
    __repr_recursion__ = _repr.Representation.__repr_recursion__
    __repr_str__ = _repr.Representation.__repr_str__
    __pretty__ = _repr.Representation.__pretty__
    __rich_repr__ = _repr.Representation.__rich_repr__

    def __str__(self) -> str:
        return self.__repr_str__(' ')

    @property
    @typing_extensions.deprecated('The `__fields__` attribute is deprecated, use `model_fields` instead.', category=None)
    def __fields__(self) -> Mapping[str, Any]:
        warnings.warn('The `__fields__` attribute is deprecated, use `model_fields` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return getattr(type(self), '__pydantic_fields__', {})

    @property
    @typing_extensions.deprecated('The `__fields_set__` attribute is deprecated, use `model_fields_set` instead.', category=None)
    def __fields_set__(self) -> Set[str]:
        warnings.warn('The `__fields_set__` attribute is deprecated, use `model_fields_set` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return self.__pydantic_fields_set__

    @typing_extensions.deprecated('The `dict` method is deprecated; use `model_dump` instead.', category=None)
    def dict(self, *, include: Optional[Any] = None, exclude: Optional[Any] = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False) -> Dict[str, Any]:
        warnings.warn('The `dict` method is deprecated; use `model_dump` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return self.model_dump(include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)

    @typing_extensions.deprecated('The `json` method is deprecated; use `model_dump_json` instead.', category=None)
    def json(self, *, include: Optional[Any] = None, exclude: Optional[Any] = None, by_alias: bool = False, exclude_unset: bool = False, exclude_defaults: bool = False, exclude_none: bool = False, encoder: Any = PydanticUndefined, models_as_dict: Any = PydanticUndefined, **dumps_kwargs: Any) -> str:
        warnings.warn('The `json` method is deprecated; use `model_dump_json` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        if encoder is not PydanticUndefined:
            raise TypeError('The `encoder` argument is no longer supported; use field serializers instead.')
        if models_as_dict is not PydanticUndefined:
            raise TypeError('The `models_as_dict` argument is no longer supported; use a model serializer instead.')
        if dumps_kwargs:
            raise TypeError('`dumps_kwargs` keyword arguments are no longer supported.')
        return self.model_dump_json(include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)

    @classmethod
    @typing_extensions.deprecated('The `parse_obj` method is deprecated; use `model_validate` instead.', category=None)
    def parse_obj(cls: Type[Self], obj: Any) -> Self:
        warnings.warn('The `parse_obj` method is deprecated; use `model_validate` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return cls.model_validate(obj)

    @classmethod
    @typing_extensions.deprecated('The `parse_raw` method is deprecated; if your data is JSON use `model_validate_json`, otherwise load the data then use `model_validate` instead.', category=None)
    def parse_raw(cls: Type[Self], b: Any, *, content_type: Any = None, encoding: str = 'utf8', proto: Any = None, allow_pickle: bool = False) -> Self:
        warnings.warn('The `parse_raw` method is deprecated; if your data is JSON use `model_validate_json`, otherwise load the data then use `model_validate` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        from .deprecated import parse
        try:
            obj = parse.load_str_bytes(b, proto=proto, content_type=content_type, encoding=encoding, allow_pickle=allow_pickle)
        except (ValueError, TypeError) as exc:
            import json
            if isinstance(exc, UnicodeDecodeError):
                type_str = 'value_error.unicodedecode'
            elif isinstance(exc, json.JSONDecodeError):
                type_str = 'value_error.jsondecode'
            elif isinstance(exc, ValueError):
                type_str = 'value_error'
            else:
                type_str = 'type_error'
            error = {'type': pydantic_core.PydanticCustomError(type_str, str(exc)), 'loc': ('__root__',), 'input': b}
            raise pydantic_core.ValidationError.from_exception_data(cls.__name__, [error])
        return cls.model_validate(obj)

    @classmethod
    @typing_extensions.deprecated('The `parse_file` method is deprecated; load the data from file, then if your data is JSON use `model_validate_json`, otherwise `model_validate` instead.', category=None)
    def parse_file(cls: Type[Self], path: Any, *, content_type: Any = None, encoding: str = 'utf8', proto: Any = None, allow_pickle: bool = False) -> Self:
        warnings.warn('The `parse_file` method is deprecated; load the data from file, then if your data is JSON use `model_validate_json`, otherwise `model_validate` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        from .deprecated import parse
        obj = parse.load_file(path, proto=proto, content_type=content_type, encoding=encoding, allow_pickle=allow_pickle)
        return cls.parse_obj(obj)

    @classmethod
    @typing_extensions.deprecated("The `from_orm` method is deprecated; set `model_config['from_attributes']=True` and use `model_validate` instead.", category=None)
    def from_orm(cls: Type[Self], obj: Any) -> Self:
        warnings.warn("The `from_orm` method is deprecated; set `model_config['from_attributes']=True` and use `model_validate` instead.", category=PydanticDeprecatedSince20, stacklevel=2)
        if not cls.model_config.get('from_attributes', None):
            raise PydanticUserError('You must set the config attribute `from_attributes=True` to use from_orm', code=None)
        return cls.model_validate(obj)

    @classmethod
    @typing_extensions.deprecated('The `construct` method is deprecated; use `model_construct` instead.', category=None)
    def construct(cls: Type[Self], _fields_set: Optional[Set[str]] = None, **values: Any) -> Self:
        warnings.warn('The `construct` method is deprecated; use `model_construct` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return cls.model_construct(_fields_set=_fields_set, **values)

    @typing_extensions.deprecated('The `copy` method is deprecated; use `model_copy` instead. See the docstring of `BaseModel.copy` for details about how to handle `include` and `exclude`.', category=None)
    def copy(self, *, include: Optional[Any] = None, exclude: Optional[Any] = None, update: Optional[Dict[str, Any]] = None, deep: bool = False) -> Self:
        warnings.warn('The `copy` method is deprecated; use `model_copy` instead. See the docstring of `BaseModel.copy` for details about how to handle `include` and `exclude`.', category=PydanticDeprecatedSince20, stacklevel=2)
        from .deprecated import copy_internals
        values = dict(copy_internals._iter(self, to_dict=False, by_alias=False, include=include, exclude=exclude, exclude_unset=False), **(update or {}))
        if self.__pydantic_private__ is None:
            private = None
        else:
            private = {k: v for k, v in self.__pydantic_private__.items() if v is not PydanticUndefined}
        if self.__pydantic_extra__ is None:
            extra = None
        else:
            extra = self.__pydantic_extra__.copy()
            for k in list(self.__pydantic_extra__):
                if k not in values:
                    extra.pop(k)
            for k in list(values):
                if k in self.__pydantic_extra__:
                    extra[k] = values.pop(k)
        if update:
            fields_set = self.__pydantic_fields_set__ | update.keys()
        else:
            fields_set = set(self.__pydantic_fields_set__)
        if exclude:
            fields_set -= set(exclude)
        return copy_internals._copy_and_set_values(self, values, fields_set, extra, private, deep=deep)

    @classmethod
    @typing_extensions.deprecated('The `schema` method is deprecated; use `model_json_schema` instead.', category=None)
    def schema(cls, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE) -> Any:
        warnings.warn('The `schema` method is deprecated; use `model_json_schema` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return cls.model_json_schema(by_alias=by_alias, ref_template=ref_template)

    @classmethod
    @typing_extensions.deprecated('The `schema_json` method is deprecated; use `model_json_schema` and json.dumps instead.', category=None)
    def schema_json(cls, *, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE, **dumps_kwargs: Any) -> str:
        warnings.warn('The `schema_json` method is deprecated; use `model_json_schema` and json.dumps instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        import json
        from .deprecated.json import pydantic_encoder
        return json.dumps(cls.model_json_schema(by_alias=by_alias, ref_template=ref_template), default=pydantic_encoder, **dumps_kwargs)

    @classmethod
    @typing_extensions.deprecated('The `validate` method is deprecated; use `model_validate` instead.', category=None)
    def validate(cls: Type[Self], value: Any) -> Self:
        warnings.warn('The `validate` method is deprecated; use `model_validate` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        return cls.model_validate(value)

    @classmethod
    @typing_extensions.deprecated('The `update_forward_refs` method is deprecated; use `model_rebuild` instead.', category=None)
    def update_forward_refs(cls, **localns: Any) -> None:
        warnings.warn('The `update_forward_refs` method is deprecated; use `model_rebuild` instead.', category=PydanticDeprecatedSince20, stacklevel=2)
        if localns:
            raise TypeError('`localns` arguments are not longer accepted.')
        cls.model_rebuild(force=True)

    @typing_extensions.deprecated('The private method `_iter` will be removed and should no longer be used.', category=None)
    def _iter(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn('The private method `_iter` will be removed and should no longer be used.', category=PydanticDeprecatedSince20, stacklevel=2)
        from .deprecated import copy_internals
        return copy_internals._iter(self, *args, **kwargs)

    @typing_extensions.deprecated('The private method `_copy_and_set_values` will be removed and should no longer be used.', category=None)
    def _copy_and_set_values(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn('The private method `_copy_and_set_values` will be removed and should no longer be used.', category=PydanticDeprecatedSince20, stacklevel=2)
        from .deprecated import copy_internals
        return copy_internals._copy_and_set_values(self, *args, **kwargs)

    @classmethod
    @typing_extensions.deprecated('The private method `_get_value` will be removed and should no longer be used.', category=None)
    def _get_value(cls, *args: Any, **kwargs: Any) -> Any:
        warnings.warn('The private method `_get_value` will be removed and should no longer be used.', category=PydanticDeprecatedSince20, stacklevel=2)
        from .deprecated import copy_internals
        return copy_internals._get_value(cls, *args, **kwargs)

    @typing_extensions.deprecated('The private method `_calculate_keys` will be removed and should no longer be used.', category=None)
    def _calculate_keys(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn('The private method `_calculate_keys` will be removed and should no longer be used.', category=PydanticDeprecatedSince20, stacklevel=2)
        from .deprecated import copy_internals
        return copy_internals._calculate_keys(self, *args, **kwargs)

ModelT = TypeVar('ModelT', bound=BaseModel)

@overload
def create_model(model_name: str, /, *, __config__: Any = None, __doc__: Optional[str] = None, __base__: None = None, __module__: str = __name__, __validators__: Any = None, __cls_kwargs__: Any = None, **field_definitions: Any) -> Type[BaseModel]:
    ...

@overload
def create_model(model_name: str, /, *, __config__: Any = None, __doc__: Optional[str] = None, __base__: Any, __module__: str = __name__, __validators__: Any = None, __cls_kwargs__: Any = None, **field_definitions: Any) -> Type[BaseModel]:
    ...

def create_model(
    model_name: str,
    /,
    *,
    __config__: Optional[Any] = None,
    __doc__: Optional[str] = None,
    __base__: Optional[Union[Type[ModelT], Tuple[Type[ModelT], ...]]] = None,
    __module__: Optional[str] = None,
    __validators__: Optional[Dict[str, Callable[..., Any]]] = None,
    __cls_kwargs__: Optional[Dict[str, Any]] = None,
    **field_definitions: Any
) -> Type[ModelT]:
    if __base__ is not None:
        if __config__ is not None:
            raise PydanticUserError('to avoid confusion `__config__` and `__base__` cannot be used together', code='create-model-config-base')
        if not isinstance(__base__, tuple):
            __base__ = (__base__,)
    else:
        __base__ = (cast(Type[ModelT], BaseModel),)
    __cls_kwargs__ = __cls_kwargs__ or {}
    fields: Dict[str, Any] = {}
    annotations: Dict[str, Any] = {}
    for f_name, f_def in field_definitions.items():
        if isinstance(f_def, tuple):
            if len(f_def) != 2:
                raise PydanticUserError(f'Field definition for {f_name!r} should a single element representing the type or a two-tuple, the first element being the type and the second element the assigned value (either a default or the `Field()` function).', code='create-model-field-definitions')
            annotations[f_name] = f_def[0]
            fields[f_name] = f_def[1]
        else:
            annotations[f_name] = f_def
    if __module__ is None:
        f = sys._getframe(1)
        __module__ = f.f_globals['__name__']
    namespace: Dict[str, Any] = {'__annotations__': annotations, '__module__': __module__}
    if __doc__:
        namespace.update({'__doc__': __doc__})
    if __validators__:
        namespace.update(__validators__)
    namespace.update(fields)
    if __config__:
        namespace['model_config'] = _config.ConfigWrapper(__config__).config_dict
    resolved_bases = types.resolve_bases(__base__)
    meta, ns, kwds = types.prepare_class(model_name, resolved_bases, kwds=__cls_kwargs__)
    if resolved_bases is not __base__:
        ns['__orig_bases__'] = __base__
    namespace.update(ns)
    return meta(model_name, resolved_bases, namespace, __pydantic_reset_parent_namespace__=False, _create_model_module=__module__, **kwds)

__getattr__ = getattr_migration(__name__)