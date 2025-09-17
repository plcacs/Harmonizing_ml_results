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
from typing import TYPE_CHECKING, Any, TypeVar, Optional, Tuple, Dict, Iterator as TypingIterator
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
_LIMITED_DICT_SIZE: int = 100


class LimitedDict(dict[KT, VT]):
    def __init__(self, size_limit: int = _LIMITED_DICT_SIZE) -> None:
        self.size_limit: int = size_limit
        super().__init__()

    def __setitem__(self, key: KT, value: VT, /) -> None:
        super().__setitem__(key, value)
        if len(self) > self.size_limit:
            excess: int = len(self) - self.size_limit + self.size_limit // 10
            to_remove: list[KT] = list(self.keys())[:excess]
            for k in to_remove:
                del self[k]


GenericTypesCache: "WeakValueDictionary[GenericTypesCacheKey, type[BaseModel]]" = typing.cast(
    WeakValueDictionary[GenericTypesCacheKey, type[BaseModel]], 
    WeakValueDictionary()
)

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

        def __setitem__(self, key: Any, value: Any) -> None:
            for mapping in self.maps:
                mapping[key] = value

        def __delitem__(self, key: Any) -> None:
            hit: bool = False
            for mapping in self.maps:
                if key in mapping:
                    del mapping[key]
                    hit = True
            if not hit:
                raise KeyError(key)


_GENERIC_TYPES_CACHE: "WeakValueDictionary[GenericTypesCacheKey, type[BaseModel]]" = GenericTypesCache


class PydanticGenericMetadata(typing_extensions.TypedDict):
    ...


def create_generic_submodel(
    model_name: str, origin: Any, args: Tuple[Any, ...], params: Tuple[Any, ...]
) -> type:
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
    namespace: Dict[str, Any] = {'__module__': origin.__module__}
    bases: Tuple[Any, ...] = (origin,)
    meta, ns, kwds = prepare_class(model_name, bases)
    namespace.update(ns)
    created_model: type = meta(
        model_name,
        bases,
        namespace,
        __pydantic_generic_metadata__={'origin': origin, 'args': args, 'parameters': params},
        __pydantic_reset_parent_namespace__=False,
        **kwds,
    )
    model_module, called_globally = _get_caller_frame_info(depth=3)
    if called_globally:
        object_by_reference: Optional[Any] = None
        reference_name: str = model_name
        reference_module_globals: Dict[str, Any] = sys.modules[created_model.__module__].__dict__
        while object_by_reference is not created_model:
            object_by_reference = reference_module_globals.setdefault(reference_name, created_model)
            reference_name += '_'
    return created_model


def _get_caller_frame_info(depth: int = 2) -> Tuple[Optional[str], bool]:
    """Used inside a function to check whether it was called globally.

    Args:
        depth: The depth to get the frame.

    Returns:
        A tuple contains `module_name` and `called_globally`.

    Raises:
        RuntimeError: If the function is not called inside another function.
    """
    try:
        previous_caller_frame = sys._getframe(depth)
    except ValueError as e:
        raise RuntimeError('This function must be used inside another function') from e
    except AttributeError:
        return (None, False)
    frame_globals: Dict[str, Any] = previous_caller_frame.f_globals
    return (frame_globals.get('__name__'), previous_caller_frame.f_locals is frame_globals)


DictValues: type = {}.values().__class__


def iter_contained_typevars(v: Any) -> TypingIterator[Any]:
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


def get_args(v: Any) -> Any:
    pydantic_generic_metadata = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('args')
    return typing_extensions.get_args(v)


def get_origin(v: Any) -> Any:
    pydantic_generic_metadata = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('origin')
    return typing_extensions.get_origin(v)


def get_standard_typevars_map(cls: Any) -> Optional[Mapping[Any, Any]]:
    """Package a generic type's typevars and parametrization (if present) into a dictionary compatible with the
    `replace_types` function. Specifically, this works with standard typing generics and typing._GenericAlias.
    """
    origin = get_origin(cls)
    if origin is None:
        return None
    if not hasattr(origin, '__parameters__'):
        return None
    args: Tuple[Any, ...] = cls.__args__
    parameters: Tuple[Any, ...] = origin.__parameters__
    return dict(zip(parameters, args))


def get_model_typevars_map(cls: Any) -> Mapping[Any, Any]:
    """Package a generic BaseModel's typevars and concrete parametrization (if present) into a dictionary compatible
    with the `replace_types` function.

    Since BaseModel.__class_getitem__ does not produce a typing._GenericAlias, and the BaseModel generic info is
    stored in the __pydantic_generic_metadata__ attribute, we need special handling here.
    """
    generic_metadata = cls.__pydantic_generic_metadata__
    origin = generic_metadata['origin']
    args: Tuple[Any, ...] = generic_metadata['args']
    if not args:
        return {}
    return dict(zip(iter_contained_typevars(origin), args))


def replace_types(type_: Any, type_map: Mapping[Any, Any]) -> Any:
    """Return type with all occurrences of `type_map` keys recursively replaced with their values.

    Args:
        type_: The class or generic alias.
        type_map: Mapping from `TypeVar` instance to concrete types.

    Returns:
        A new type representing the basic structure of `type_` with all
        `typevar_map` keys recursively replaced.

    Example:
        >>> from typing import List, Union
        >>> from pydantic._internal._generics import replace_types
        >>> replace_types(tuple[str, Union[List[str], float]], {str: int})
        #> tuple[int, Union[List[int], float]]
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
        if (
            origin_type is not None
            and isinstance(type_, _typing_extra.typing_base)
            and (not isinstance(origin_type, _typing_extra.typing_base))
            and (getattr(type_, '_name', None) is not None)
        ):
            origin_type = getattr(typing, type_._name)
        assert origin_type is not None
        if _typing_extra.origin_is_union(origin_type):
            if any((_typing_extra.is_any(arg) for arg in resolved_type_args)):
                resolved_type_args = (Any,)
            resolved_type_args = tuple(
                (arg for arg in resolved_type_args if not (_typing_extra.is_no_return(arg) or _typing_extra.is_never(arg)))
            )
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


def map_generic_model_arguments(cls: Any, args: Tuple[Any, ...]) -> Mapping[Any, Any]:
    """Return a mapping between the arguments of a generic model and the provided arguments during parametrization.

    Raises:
        TypeError: If the number of arguments does not match the parameters (i.e. if providing too few or too many arguments).

    Example:
        >>>
        >>> class Model[T, U, V = int](BaseModel): ...
        >>>
        >>> map_generic_model_arguments(Model, (str, bytes))
        #> {T: str, U: bytes, V: int}
        >>>
        >>> map_generic_model_arguments(Model, (str,))
        #> TypeError: Too few arguments for <class '__main__.Model'>; actual 1, expected at least 2
        >>>
        >>> map_generic_model_arguments(Model, (str, bytes, int, complex))
        #> TypeError: Too many arguments for <class '__main__.Model'>; actual 4, expected 3

    Note:
        This function is analogous to the private `typing._check_generic_specialization` function.
    """
    parameters = cls.__pydantic_generic_metadata__['parameters']
    expected_len: int = len(parameters)
    typevars_map: Dict[Any, Any] = {}
    _missing = object()
    for parameter, argument in zip_longest(parameters, args, fillvalue=_missing):
        if parameter is _missing:
            raise TypeError(f'Too many arguments for {cls}; actual {len(args)}, expected {expected_len}')
        if argument is _missing:
            param = typing.cast(TypeVar, parameter)
            try:
                has_default = param.has_default()  # type: ignore[attr-defined]
            except AttributeError:
                has_default = False
            if has_default:
                typevars_map[param] = param.__default__  # type: ignore[attr-defined]
            else:
                expected_len -= sum((hasattr(p, 'has_default') and p.has_default() for p in parameters))  # type: ignore[attr-defined]
                raise TypeError(f'Too few arguments for {cls}; actual {len(args)}, expected at least {expected_len}')
        else:
            param = typing.cast(TypeVar, parameter)
            typevars_map[param] = argument
    return typevars_map


_generic_recursion_cache: ContextVar[Optional[set[Any]]] = ContextVar('_generic_recursion_cache', default=None)


@contextmanager
def generic_recursion_self_type(origin: Any, args: Tuple[Any, ...]) -> Iterator[Any]:
    """This contextmanager should be placed around the recursive calls used to build a generic type,
    and accept as arguments the generic origin type and the type arguments being passed to it.

    If the same origin and arguments are observed twice, it implies that a self-reference placeholder
    can be used while building the core schema, and will produce a schema_ref that will be valid in the
    final parent schema.
    """
    previously_seen_type_refs: Optional[set[Any]] = _generic_recursion_cache.get()
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
            yield  # type: ignore
            previously_seen_type_refs.remove(type_ref)
    finally:
        if token:
            _generic_recursion_cache.reset(token)


def recursively_defined_type_refs() -> set[Any]:
    visited: Optional[set[Any]] = _generic_recursion_cache.get()
    if not visited:
        return set()
    return visited.copy()


def get_cached_generic_type_early(parent: Any, typevar_values: Any) -> Optional[Any]:
    """The use of a two-stage cache lookup approach was necessary to have the highest performance possible for
    repeated calls to `__class_getitem__` on generic types (which may happen in tighter loops during runtime),
    while still ensuring that certain alternative parametrizations ultimately resolve to the same type.

    As a concrete example, this approach was necessary to make Model[List[T]][int] equal to Model[List[int]].
    The approach could be modified to not use two different cache keys at different points, but the
    _early_cache_key is optimized to be as quick to compute as possible (for repeated-access speed), and the
    _late_cache_key is optimized to be as "correct" as possible, so that two types that will ultimately be the
    same after resolving the type arguments will always produce cache hits.
    """
    return _GENERIC_TYPES_CACHE.get(_early_cache_key(parent, typevar_values))


def get_cached_generic_type_late(parent: Any, typevar_values: Any, origin: Any, args: Any) -> Optional[Any]:
    """See the docstring of `get_cached_generic_type_early` for more information about the two-stage cache lookup."""
    cached: Optional[Any] = _GENERIC_TYPES_CACHE.get(_late_cache_key(origin, args, typevar_values))
    if cached is not None:
        set_cached_generic_type(parent, typevar_values, cached, origin, args)
    return cached


def set_cached_generic_type(
    parent: Any, typevar_values: Any, type_: Any, origin: Optional[Any] = None, args: Optional[Any] = None
) -> None:
    """See the docstring of `get_cached_generic_type_early` for more information about why items are cached with
    two different keys.
    """
    _GENERIC_TYPES_CACHE[_early_cache_key(parent, typevar_values)] = type_
    if isinstance(typevar_values, tuple) and len(typevar_values) == 1:
        _GENERIC_TYPES_CACHE[_early_cache_key(parent, typevar_values[0])] = type_
    if origin and args:
        _GENERIC_TYPES_CACHE[_late_cache_key(origin, args, typevar_values)] = type_


def _union_orderings_key(typevar_values: Any) -> Any:
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
        args_data: list[Any] = []
        for value in typevar_values:
            args_data.append(_union_orderings_key(value))
        return tuple(args_data)
    elif _typing_extra.is_union(typevar_values):
        return get_args(typevar_values)
    else:
        return ()


def _early_cache_key(cls: Any, typevar_values: Any) -> Any:
    """This is intended for minimal computational overhead during lookups of cached types.

    Note that this is overly simplistic, and it's possible that two different cls/typevar_values
    inputs would ultimately result in the same type being created in BaseModel.__class_getitem__.
    To handle this, we have a fallback _late_cache_key that is checked later if the _early_cache_key
    lookup fails, and should result in a cache hit _precisely_ when the inputs to __class_getitem__
    would result in the same type.
    """
    return (cls, typevar_values, _union_orderings_key(typevar_values))


def _late_cache_key(origin: Any, args: Any, typevar_values: Any) -> Any:
    """This is intended for use later in the process of creating a new type, when we have more information
    about the exact args that will be passed. If it turns out that a different set of inputs to
    __class_getitem__ resulted in the same inputs to the generic type creation process, we can still
    return the cached type, and update the cache with the _early_cache_key as well.
    """
    return (_union_orderings_key(typevar_values), origin, args)