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
from typing import TYPE_CHECKING, Any, TypeVar
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
GenericTypesCacheKey = tuple[Any, Any, tuple[Any, ...]]
KT = TypeVar('KT')
VT = TypeVar('VT')
_LIMITED_DICT_SIZE = 100

class LimitedDict(dict[KT, VT]):

    def __init__(self, size_limit: Union[int, typing.Callable[None, typing.Any]]=_LIMITED_DICT_SIZE) -> None:
        self.size_limit = size_limit
        super().__init__()

    def __setitem__(self, key, value, /) -> None:
        super().__setitem__(key, value)
        if len(self) > self.size_limit:
            excess = len(self) - self.size_limit + self.size_limit // 10
            to_remove = list(self.keys())[:excess]
            for k in to_remove:
                del self[k]
GenericTypesCache = WeakValueDictionary[GenericTypesCacheKey, 'type[BaseModel]']
if TYPE_CHECKING:

    class DeepChainMap(ChainMap[KT, VT]):
        ...
else:

    class DeepChainMap(ChainMap):
        """Variant of ChainMap that allows direct updates to inner scopes.

        Taken from https://docs.python.org/3/library/collections.html#collections.ChainMap,
        with some light modifications for this use case.
        """

        def clear(self) -> None:
            for mapping in self.maps:
                mapping.clear()

        def __setitem__(self, key, value) -> None:
            for mapping in self.maps:
                mapping[key] = value

        def __delitem__(self, key: str) -> None:
            hit = False
            for mapping in self.maps:
                if key in mapping:
                    del mapping[key]
                    hit = True
            if not hit:
                raise KeyError(key)
_GENERIC_TYPES_CACHE = GenericTypesCache()

class PydanticGenericMetadata(typing_extensions.TypedDict):
    pass

def create_generic_submodel(model_name: str, origin: Union[dict[str, typing.Any], str, typing.MutableMapping], args: Any, params: Union[dict[str, typing.Any], str, typing.Type]) -> Union[str, list[typing.Callable[None, typing.Any]], int]:
    """Dynamically create a submodel of a provided (generic) BaseModel.

    This is used when producing concrete parametrizations of generic models. This function
    only *creates* the new subclass; the schema/validators/serialization must be updated to
    reflect a concrete parametrization elsewhere.

    Args:
        model_name: The name of the newly created model.
        origin: The base class for the new model to inherit from.
        args: A tuple of generic metadata arguments.
        params: A tuple of generic metadata parameters.

    Returns:
        The created submodel.
    """
    namespace = {'__module__': origin.__module__}
    bases = (origin,)
    meta, ns, kwds = prepare_class(model_name, bases)
    namespace.update(ns)
    created_model = meta(model_name, bases, namespace, __pydantic_generic_metadata__={'origin': origin, 'args': args, 'parameters': params}, __pydantic_reset_parent_namespace__=False, **kwds)
    model_module, called_globally = _get_caller_frame_info(depth=3)
    if called_globally:
        object_by_reference = None
        reference_name = model_name
        reference_module_globals = sys.modules[created_model.__module__].__dict__
        while object_by_reference is not created_model:
            object_by_reference = reference_module_globals.setdefault(reference_name, created_model)
            reference_name += '_'
    return created_model

def _get_caller_frame_info(depth: int=2) -> Union[tuple[typing.Optional[bool]], tuple[bool]]:
    """Used inside a function to check whether it was called globally.

    Args:
        depth: The depth to get the frame.

    Returns:
        A tuple contains `module_name` and `called_globally`.

    Raises:
        RuntimeError: If the function is not called inside a function.
    """
    try:
        previous_caller_frame = sys._getframe(depth)
    except ValueError as e:
        raise RuntimeError('This function must be used inside another function') from e
    except AttributeError:
        return (None, False)
    frame_globals = previous_caller_frame.f_globals
    return (frame_globals.get('__name__'), previous_caller_frame.f_locals is frame_globals)
DictValues = {}.values().__class__

def iter_contained_typevars(v: Union[mypy.types.Type, typing.Type, str]) -> Union[typing.Generator[TypeVar], typing.Generator]:
    """Recursively iterate through all subtypes and type args of `v` and yield any typevars that are found.

    This is inspired as an alternative to directly accessing the `__parameters__` attribute of a GenericAlias,
    since __parameters__ of (nested) generic BaseModel subclasses won't show up in that list.
    """
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

def get_args(v: Union[str, travelfootprininsta.types.FeedItem, Exception]) -> str:
    pydantic_generic_metadata = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('args')
    return typing_extensions.get_args(v)

def get_origin(v: Union[str, typing.Type, bool]) -> str:
    pydantic_generic_metadata = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('origin')
    return typing_extensions.get_origin(v)

def get_standard_typevars_map(cls: typing.Type) -> None:
    """Package a generic type's typevars and parametrization (if present) into a dictionary compatible with the
    `replace_types` function. Specifically, this works with standard typing generics and typing._GenericAlias.
    """
    origin = get_origin(cls)
    if origin is None:
        return None
    if not hasattr(origin, '__parameters__'):
        return None
    args = cls.__args__
    parameters = origin.__parameters__
    return dict(zip(parameters, args))

def get_model_typevars_map(cls: typing.Type) -> dict:
    """Package a generic BaseModel's typevars and concrete parametrization (if present) into a dictionary compatible
    with the `replace_types` function.

    Since BaseModel.__class_getitem__ does not produce a typing._GenericAlias, and the BaseModel generic info is
    stored in the __pydantic_generic_metadata__ attribute, we need special handling here.
    """
    generic_metadata = cls.__pydantic_generic_metadata__
    origin = generic_metadata['origin']
    args = generic_metadata['args']
    if not args:
        return {}
    return dict(zip(iter_contained_typevars(origin), args))

def replace_types(type_: str, type_map: Any) -> Union[str, _UnionGenericAlias, int, list]:
    """Return type with all occurrences of `type_map` keys recursively replaced with their values.

    Args:
        type_: The class or generic alias.
        type_map: Mapping from `TypeVar` instance to concrete types.

    Returns:
        A new type representing the basic structure of `type_` with all
        `typevar_map` keys recursively replaced.

    Example:
        ```python
        from typing import List, Union

        from pydantic._internal._generics import replace_types

        replace_types(tuple[str, Union[List[str], float]], {str: int})
        #> tuple[int, Union[List[int], float]]
        ```
    """
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
        resolved_type_args = tuple((replace_types(arg, type_map) for arg in type_args))
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

def map_generic_model_arguments(cls: typing.Type, args: Any) -> dict:
    """Return a mapping between the arguments of a generic model and the provided arguments during parametrization.

    Raises:
        TypeError: If the number of arguments does not match the parameters (i.e. if providing too few or too many arguments).

    Example:
        ```python {test="skip" lint="skip"}
        class Model[T, U, V = int](BaseModel): ...

        map_generic_model_arguments(Model, (str, bytes))
        #> {T: str, U: bytes, V: int}

        map_generic_model_arguments(Model, (str,))
        #> TypeError: Too few arguments for <class '__main__.Model'>; actual 1, expected at least 2

        map_generic_model_argumenst(Model, (str, bytes, int, complex))
        #> TypeError: Too many arguments for <class '__main__.Model'>; actual 4, expected 3
        ```

    Note:
        This function is analogous to the private `typing._check_generic_specialization` function.
    """
    parameters = cls.__pydantic_generic_metadata__['parameters']
    expected_len = len(parameters)
    typevars_map = {}
    _missing = object()
    for parameter, argument in zip_longest(parameters, args, fillvalue=_missing):
        if parameter is _missing:
            raise TypeError(f'Too many arguments for {cls}; actual {len(args)}, expected {expected_len}')
        if argument is _missing:
            param = typing.cast(TypeVar, parameter)
            try:
                has_default = param.has_default()
            except AttributeError:
                has_default = False
            if has_default:
                typevars_map[param] = param.__default__
            else:
                expected_len -= sum((hasattr(p, 'has_default') and p.has_default() for p in parameters))
                raise TypeError(f'Too few arguments for {cls}; actual {len(args)}, expected at least {expected_len}')
        else:
            param = typing.cast(TypeVar, parameter)
            typevars_map[param] = argument
    return typevars_map
_generic_recursion_cache = ContextVar('_generic_recursion_cache', default=None)

@contextmanager
def generic_recursion_self_type(origin: str, args: Any) -> Union[typing.Generator[PydanticRecursiveRef], typing.Generator]:
    """This contextmanager should be placed around the recursive calls used to build a generic type,
    and accept as arguments the generic origin type and the type arguments being passed to it.

    If the same origin and arguments are observed twice, it implies that a self-reference placeholder
    can be used while building the core schema, and will produce a schema_ref that will be valid in the
    final parent schema.
    """
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

def recursively_defined_type_refs() -> set:
    visited = _generic_recursion_cache.get()
    if not visited:
        return set()
    return visited.copy()

def get_cached_generic_type_early(parent: Union[str, typing.Type], typevar_values: Union[str, typing.Type]) -> typing.Callable[None,None, typing.Any]:
    """The use of a two-stage cache lookup approach was necessary to have the highest performance possible for
    repeated calls to `__class_getitem__` on generic types (which may happen in tighter loops during runtime),
    while still ensuring that certain alternative parametrizations ultimately resolve to the same type.

    As a concrete example, this approach was necessary to make Model[List[T]][int] equal to Model[List[int]].
    The approach could be modified to not use two different cache keys at different points, but the
    _early_cache_key is optimized to be as quick to compute as possible (for repeated-access speed), and the
    _late_cache_key is optimized to be as "correct" as possible, so that two types that will ultimately be the
    same after resolving the type arguments will always produce cache hits.

    If we wanted to move to only using a single cache key per type, we would either need to always use the
    slower/more computationally intensive logic associated with _late_cache_key, or would need to accept
    that Model[List[T]][int] is a different type than Model[List[T]][int]. Because we rely on subclass relationships
    during validation, I think it is worthwhile to ensure that types that are functionally equivalent are actually
    equal.
    """
    return _GENERIC_TYPES_CACHE.get(_early_cache_key(parent, typevar_values))

def get_cached_generic_type_late(parent: Union[mypy.types.Type, str, None], typevar_values: Union[str, None, typing.Type, mypy.types.Instance], origin: Union[str, None, typing.Type, mypy.types.Instance], args: Any) -> Union[None, typing.Callable[None,None, typing.Any]]:
    """See the docstring of `get_cached_generic_type_early` for more information about the two-stage cache lookup."""
    cached = _GENERIC_TYPES_CACHE.get(_late_cache_key(origin, args, typevar_values))
    if cached is not None:
        set_cached_generic_type(parent, typevar_values, cached, origin, args)
    return cached

def set_cached_generic_type(parent: Union[mypy.types.Type, None, str], typevar_values: mypy.types.Type, type_: Union[typing.Type, str, None], origin: Union[None, mypy.types.Type, mypy.types.CallableType, mypy.types.Instance]=None, args: None=None) -> None:
    """See the docstring of `get_cached_generic_type_early` for more information about why items are cached with
    two different keys.
    """
    _GENERIC_TYPES_CACHE[_early_cache_key(parent, typevar_values)] = type_
    if len(typevar_values) == 1:
        _GENERIC_TYPES_CACHE[_early_cache_key(parent, typevar_values[0])] = type_
    if origin and args:
        _GENERIC_TYPES_CACHE[_late_cache_key(origin, args, typevar_values)] = type_

def _union_orderings_key(typevar_values: Union[pyspark.sql.types.DataType, list[typing.Type], typing.Any, None]) -> Union[tuple, typing.Sequence[str], str, set[str]]:
    """This is intended to help differentiate between Union types with the same arguments in different order.

    Thanks to caching internal to the `typing` module, it is not possible to distinguish between
    List[Union[int, float]] and List[Union[float, int]] (and similarly for other "parent" origins besides List)
    because `typing` considers Union[int, float] to be equal to Union[float, int].

    However, you _can_ distinguish between (top-level) Union[int, float] vs. Union[float, int].
    Because we parse items as the first Union type that is successful, we get slightly more consistent behavior
    if we make an effort to distinguish the ordering of items in a union. It would be best if we could _always_
    get the exact-correct order of items in the union, but that would require a change to the `typing` module itself.
    (See https://github.com/python/cpython/issues/86483 for reference.)
    """
    if isinstance(typevar_values, tuple):
        args_data = []
        for value in typevar_values:
            args_data.append(_union_orderings_key(value))
        return tuple(args_data)
    elif _typing_extra.is_union(typevar_values):
        return get_args(typevar_values)
    else:
        return ()

def _early_cache_key(cls: Union[pyspark.sql.types.DataType, typing.Iterable[typing.Any], typing.Sequence[T]], typevar_values: Union[pyspark.sql.types.DataType, typing.Iterable[typing.Any], typing.Sequence[T]]) -> tuple[typing.Union[pyspark.sql.types.DataType,typing.Iterable[typing.Any],typing.Sequence[T]]]:
    """This is intended for minimal computational overhead during lookups of cached types.

    Note that this is overly simplistic, and it's possible that two different cls/typevar_values
    inputs would ultimately result in the same type being created in BaseModel.__class_getitem__.
    To handle this, we have a fallback _late_cache_key that is checked later if the _early_cache_key
    lookup fails, and should result in a cache hit _precisely_ when the inputs to __class_getitem__
    would result in the same type.
    """
    return (cls, typevar_values, _union_orderings_key(typevar_values))

def _late_cache_key(origin: Union[list, int, T], args: Any, typevar_values: Union[list, int, T]) -> tuple[typing.Union[list,int,T]]:
    """This is intended for use later in the process of creating a new type, when we have more information
    about the exact args that will be passed. If it turns out that a different set of inputs to
    __class_getitem__ resulted in the same inputs to the generic type creation process, we can still
    return the cached type, and update the cache with the _early_cache_key as well.
    """
    return (_union_orderings_key(typevar_values), origin, args)