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
from typing import TYPE_CHECKING, Any, TypeVar, cast
from weakref import WeakValueDictionary

import typing_extensions

from . import _typing_extra
from ._core_utils import get_type_ref
from ._forward_ref import PydanticRecursiveRef
from ._utils import all_identical, is_model_class

if sys.version_info >= (3, 10):
    from typing import _UnionGenericAlias  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from ..main import BaseModel

GenericTypesCacheKey = tuple[Any, Any, tuple[Any, ...]]

# Note: We want to remove LimitedDict, but to do this, we'd need to improve the handling of generics caching.
#   Right now, to handle recursive generics, we some types must remain cached for brief periods without references.
#   By chaining the WeakValuesDict with a LimitedDict, we have a way to retain caching for all types with references,
#   while also retaining a limited number of types even without references. This is generally enough to build
#   specific recursive generic models without losing required items out of the cache.

KT = TypeVar('KT')
VT = TypeVar('VT')
_LIMITED_DICT_SIZE = 100


class LimitedDict(dict[KT, VT]):
    def __init__(self, size_limit: int = _LIMITED_DICT_SIZE) -> None:
        self.size_limit = size_limit
        super().__init__()

    def __setitem__(self, key: KT, value: VT, /) -> None:
        super().__setitem__(key, value)
        if len(self) > self.size_limit:
            excess = len(self) - self.size_limit + self.size_limit // 10
            to_remove = list(self.keys())[:excess]
            for k in to_remove:
                del self[k]


# weak dictionaries allow the dynamically created parametrized versions of generic models to get collected
# once they are no longer referenced by the caller.
GenericTypesCache = WeakValueDictionary[GenericTypesCacheKey, 'type[BaseModel]']

if TYPE_CHECKING:

    class DeepChainMap(ChainMap[KT, VT]):  # type: ignore
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


# Despite the fact that LimitedDict _seems_ no longer necessary, I'm very nervous to actually remove it
# and discover later on that we need to re-add all this infrastructure...
# _GENERIC_TYPES_CACHE = DeepChainMap(GenericTypesCache(), LimitedDict())

_GENERIC_TYPES_CACHE = GenericTypesCache()


class PydanticGenericMetadata(typing_extensions.TypedDict):
    origin: type[BaseModel] | None  # analogous to typing._GenericAlias.__origin__
    args: tuple[Any, ...]  # analogous to typing._GenericAlias.__args__
    parameters: tuple[TypeVar, ...]  # analogous to typing.Generic.__parameters__


def create_generic_submodel(
    model_name: str, origin: type[BaseModel], args: tuple[Any, ...], params: tuple[Any, ...]
) -> type[BaseModel]:
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
    namespace: dict[str, Any] = {'__module__': origin.__module__}
    bases = (origin,)
    meta, ns, kwds = prepare_class(model_name, bases)
    namespace.update(ns)
    created_model = meta(
        model_name,
        bases,
        namespace,
        __pydantic_generic_metadata__={
            'origin': origin,
            'args': args,
            'parameters': params,
        },
        __pydantic_reset_parent_namespace__=False,
        **kwds,
    )

    model_module, called_globally = _get_caller_frame_info(depth=3)
    if called_globally:  # create global reference and therefore allow pickling
        object_by_reference = None
        reference_name = model_name
        reference_module_globals = sys.modules[created_model.__module__].__dict__
        while object_by_reference is not created_model:
            object_by_reference = reference_module_globals.setdefault(reference_name, created_model)
            reference_name += '_'

    return created_model


def _get_caller_frame_info(depth: int = 2) -> tuple[str | None, bool]:
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
    except AttributeError:  # sys module does not have _getframe function, so there's nothing we can do about it
        return None, False
    frame_globals = previous_caller_frame.f_globals
    return frame_globals.get('__name__'), previous_caller_frame.f_locals is frame_globals


DictValues: type[Any] = {}.values().__class__


def iter_contained_typevars(v: Any) -> Iterator[TypeVar]:
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
    pydantic_generic_metadata: PydanticGenericMetadata | None = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('args')
    return typing_extensions.get_args(v)


def get_origin(v: Any) -> Any:
    pydantic_generic_metadata: PydanticGenericMetadata | None = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('origin')
    return typing_extensions.get_origin(v)


def get_standard_typevars_map(cls: Any) -> dict[TypeVar, Any] | None:
    """Package a generic type's typevars and parametrization (if present) into a dictionary compatible with the
    `replace_types` function. Specifically, this works with standard typing generics and typing._GenericAlias.
    """
    origin = get_origin(cls)
    if origin is None:
        return None
    if not hasattr(origin, '__parameters__'):
        return None

    # In this case, we know that cls is a _GenericAlias, and origin is the generic type
    # So it is safe to access cls.__args__ and origin.__parameters__
    args: tuple[Any, ...] = cls.__args__  # type: ignore
    parameters: tuple[TypeVar, ...] = origin.__parameters__
    return dict(zip(parameters, args))


def get_model_typevars_map(cls: type[BaseModel]) -> dict[TypeVar, Any]:
    """Package a generic BaseModel's typevars and concrete parametrization (if present) into a dictionary compatible
    with the `replace_types` function.

    Since BaseModel.__class_getitem__ does not produce a typing._GenericAlias, and the BaseModel generic info is
    stored in the __pydantic_generic_metadata__ attribute, we need special handling here.
    """
    # TODO: This could be unified with `get_standard_typevars_map` if we stored the generic metadata
    #   in the __origin__, __args__, and __parameters__ attributes of the model.
    generic_metadata = cls.__pydantic_generic_metadata__
    origin = generic_metadata['origin']
    args = generic_metadata['args']
    if not args:
        # No need to go into `iter_contained_typevars`:
        return {}
    return dict(zip(iter_contained_typevars(origin), args))


def replace_types(type_: Any, type_map: Mapping[TypeVar, Any] | None) -> Any:
    """Return type with all occurrences of `type_map` keys recursively replaced with their values.

    Args:
        type_: The class or generic alias.
        type_map: Mapping from `TypeVar` instance to concrete types.

    Returns:
        A new type representing the basic structure of `type_` with all
        `typevar_map` keys recursively replaced.

    Example:
        