from __future__ import annotations
import sys
import types
import typing
from collections import ChainMap
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import zip_longest
from types import prepare_class
from typing import TYPE_CHECKING, Any, TypeVar, Optional, Union, Tuple, Dict, List, Set, cast
from weakref import WeakValueDictionary
import typing_extensions
from . import _typing_extra
from ._core_utils import get_type_ref
from ._forward_ref import PydanticRecursiveRef
from ._utils import all_identical, is_model_class

if sys.version_info >= (3, 10):
    from typing import _UnionGenericAlias

if TYPE_CHECKING:
    from ..main import BaseModel

GenericTypesCacheKey = Tuple[Any, Any, Tuple[Any, ...]]
KT = TypeVar('KT')
VT = TypeVar('VT')
_LIMITED_DICT_SIZE = 100

class LimitedDict(dict[KT, VT]):
    def __init__(self, size_limit: int = _LIMITED_DICT_SIZE) -> None:
        self.size_limit = size_limit
        super().__init__()

    def __setitem__(self, key: KT, value: VT) -> None:
        super().__setitem__(key, value)
        if len(self) > self.size_limit:
            excess = len(self) - self.size_limit + self.size_limit // 10
            to_remove = list(self.keys())[:excess]
            for k in to_remove:
                del self[k]

GenericTypesCache = WeakValueDictionary[GenericTypesCacheKey, type[BaseModel]]

if TYPE_CHECKING:
    class DeepChainMap(ChainMap[KT, VT]):
        ...
else:
    class DeepChainMap(ChainMap):
        def clear(self) -> None:
            for mapping in self.maps:
                mapping.clear()

        def __setitem__(self, key: KT, value: VT) -> None:
            for mapping in self.maps:
                mapping[key] = value

        def __delitem__(self, key: KT) -> None:
            hit = False
            for mapping in self.maps:
                if key in mapping:
                    del mapping[key]
                    hit = True
            if not hit:
                raise KeyError(key)

_GENERIC_TYPES_CACHE: GenericTypesCache = GenericTypesCache()

class PydanticGenericMetadata(typing_extensions.TypedDict):
    pass

def create_generic_submodel(
    model_name: str,
    origin: type[BaseModel],
    args: Tuple[Any, ...],
    params: Tuple[Any, ...]
) -> type[BaseModel]:
    namespace = {'__module__': origin.__module__}
    bases = (origin,)
    meta, ns, kwds = prepare_class(model_name, bases)
    namespace.update(ns)
    created_model = meta(
        model_name,
        bases,
        namespace,
        __pydantic_generic_metadata__={'origin': origin, 'args': args, 'parameters': params},
        __pydantic_reset_parent_namespace__=False,
        **kwds
    )
    model_module, called_globally = _get_caller_frame_info(depth=3)
    if called_globally:
        object_by_reference = None
        reference_name = model_name
        reference_module_globals = sys.modules[created_model.__module__].__dict__
        while object_by_reference is not created_model:
            object_by_reference = reference_module_globals.setdefault(reference_name, created_model)
            reference_name += '_'
    return created_model

def _get_caller_frame_info(depth: int = 2) -> Tuple[Optional[str], bool]:
    try:
        previous_caller_frame = sys._getframe(depth)
    except ValueError as e:
        raise RuntimeError('This function must be used inside another function') from e
    except AttributeError:
        return (None, False)
    frame_globals = previous_caller_frame.f_globals
    return (frame_globals.get('__name__'), previous_caller_frame.f_locals is frame_globals)

DictValues = {}.values().__class__

def iter_contained_typevars(v: Any) -> Iterator[TypeVar]:
    if isinstance(v, TypeVar):
        yield v
    elif is_model_class(v):
        yield from v.__pydantic_generic_metadata__['parameters']
    elif isinstance(v, (DictValues, list)):
        for var in v:
            yield from iter_contained_typevars(var)
    else:
        args = get_args(v)
        for arg in args:
            yield from iter_contained_typevars(arg)

def get_args(v: Any) -> Tuple[Any, ...]:
    pydantic_generic_metadata = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('args', ())
    return typing_extensions.get_args(v)

def get_origin(v: Any) -> Optional[Any]:
    pydantic_generic_metadata = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('origin')
    return typing_extensions.get_origin(v)

def get_standard_typevars_map(cls: type) -> Optional[Dict[TypeVar, Any]]:
    origin = get_origin(cls)
    if origin is None:
        return None
    if not hasattr(origin, '__parameters__'):
        return None
    args = getattr(cls, '__args__', ())
    parameters = origin.__parameters__
    return dict(zip(parameters, args))

def get_model_typevars_map(cls: type) -> Dict[TypeVar, Any]:
    generic_metadata = cls.__pydantic_generic_metadata__
    origin = generic_metadata['origin']
    args = generic_metadata['args']
    if not args:
        return {}
    return dict(zip(iter_contained_typevars(origin), args))

def replace_types(type_: Any, type_map: Dict[TypeVar, Any]) -> Any:
    if not type_map:
        return type_
    type_args = get_args(type_)
    if _typing_extra.is_annotated(type_):
        annotated_type, *annotations = type_args
        annotated = replace_types(annotated_type, type_map)
        for annotation in annotations:
            annotated = typing.Annotated[annotated, annotation]
        return annotated
    origin_type = get_origin(type_)
    if type_args:
        resolved_type_args = tuple((replace_types(arg, type_map) for arg in type_args)
        if all_identical(type_args, resolved_type_args):
            return type_
        if origin_type is not None and isinstance(type_, _typing_extra.typing_base) and (not isinstance(origin_type, _typing_extra.typing_base)) and (getattr(type_, '_name', None) is not None):
            origin_type = getattr(typing, type_._name)
        assert origin_type is not None
        if _typing_extra.origin_is_union(origin_type):
            if any((_typing_extra.is_any(arg) for arg in resolved_type_args)):
                resolved_type_args = (Any,)
            resolved_type_args = tuple((arg for arg in resolved_type_args if not (_typing_extra.is_no_return(arg) or _typing_extra.is_never(arg))))
        if sys.version_info >= (3, 10) and origin_type is types.UnionType:
            return _UnionGenericAlias(origin_type, resolved_type_args)
        return origin_type[resolved_type_args[0] if len(resolved_type_args) == 1 else resolved_type_args]
    if not origin_type and is_model_class(type_):
        parameters = type_.__pydantic_generic_metadata__['parameters']
        if not parameters:
            return type_
        resolved_type_args = tuple((replace_types(t, type_map) for t in parameters))
        if all_identical(parameters, resolved_type_args):
            return type_
        return type_[resolved_type_args]
    if isinstance(type_, list):
        resolved_list = [replace_types(element, type_map) for element in type_]
        if all_identical(type_, resolved_list):
            return type_
        return resolved_list
    return type_map.get(type_, type_)

def map_generic_model_arguments(cls: type, args: Tuple[Any, ...]) -> Dict[TypeVar, Any]:
    parameters = cls.__pydantic_generic_metadata__['parameters']
    expected_len = len(parameters)
    typevars_map: Dict[TypeVar, Any] = {}
    _missing = object()
    for parameter, argument in zip_longest(parameters, args, fillvalue=_missing):
        if parameter is _missing:
            raise TypeError(f'Too many arguments for {cls}; actual {len(args)}, expected {expected_len}')
        if argument is _missing:
            param = cast(TypeVar, parameter)
            try:
                has_default = param.has_default()
            except AttributeError:
                has_default = False
            if has_default:
                typevars_map[param] = param.__default__
            else:
                expected_len -= sum((hasattr(p, 'has_default') and p.has_default() for p in parameters)
                raise TypeError(f'Too few arguments for {cls}; actual {len(args)}, expected at least {expected_len}')
        else:
            param = cast(TypeVar, parameter)
            typevars_map[param] = argument
    return typevars_map

_generic_recursion_cache: ContextVar[Optional[Set[str]] = ContextVar('_generic_recursion_cache', default=None)

@contextmanager
def generic_recursion_self_type(origin: type, args: Tuple[Any, ...]) -> Iterator[Optional[PydanticRecursiveRef]]:
    previously_seen_type_refs = _generic_recursion_cache.get()
    if previously_seen_type_refs is None:
        previously_seen_type_refs = set()
        token = _generic_recursion_cache.set(previously_seen_type_refs)
    else:
        token = None
    try:
        type_ref = get_type_ref(origin, args_override=args)
        if type_ref in previously_seen_type_refs:
            self_type = PydanticRecursiveRef(type_ref=type_ref)
            yield self_type
        else:
            previously_seen_type_refs.add(type_ref)
            yield
            previously_seen_type_refs.remove(type_ref)
    finally:
        if token:
            _generic_recursion_cache.reset(token)

def recursively_defined_type_refs() -> Set[str]:
    visited = _generic_recursion_cache.get()
    if not visited:
        return set()
    return visited.copy()

def get_cached_generic_type_early(
    parent: type,
    typevar_values: Union[Tuple[Any, ...], Any]
) -> Optional[type[BaseModel]]:
    return _GENERIC_TYPES_CACHE.get(_early_cache_key(parent, typevar_values))

def get_cached_generic_type_late(
    parent: type,
    typevar_values: Tuple[Any, ...],
    origin: type,
    args: Tuple[Any, ...]
) -> Optional[type[BaseModel]]:
    cached = _GENERIC_TYPES_CACHE.get(_late_cache_key(origin, args, typevar_values))
    if cached is not None:
        set_cached_generic_type(parent, typevar_values, cached, origin, args)
    return cached

def set_cached_generic_type(
    parent: type,
    typevar_values: Tuple[Any, ...],
    type_: type[BaseModel],
    origin: Optional[type] = None,
    args: Optional[Tuple[Any, ...]] = None
) -> None:
    _GENERIC_TYPES_CACHE[_early_cache_key(parent, typevar_values)] = type_
    if len(typevar_values) == 1:
        _GENERIC_TYPES_CACHE[_early_cache_key(parent, typevar_values[0])] = type_
    if origin and args:
        _GENERIC_TYPES_CACHE[_late_cache_key(origin, args, typevar_values)] = type_

def _union_orderings_key(typevar_values: Union[Tuple[Any, ...], Any]) -> Union[Tuple[Any, ...], Any]:
    if isinstance(typevar_values, tuple):
        args_data = []
        for value in typevar_values:
            args_data.append(_union_orderings_key(value))
        return tuple(args_data)
    elif _typing_extra.is_union(typevar_values):
        return get_args(typevar_values)
    else:
        return typevar_values

def _early_cache_key(cls: type, typevar_values: Union[Tuple[Any, ...], Any]) -> GenericTypesCacheKey:
    return (cls, typevar_values, _union_orderings_key(typevar_values))

def _late_cache_key(
    origin: type,
    args: Tuple[Any, ...],
    typevar_values: Tuple[Any, ...]
) -> GenericTypesCacheKey:
    return (_union_orderings_key(typevar_values), origin, args)
