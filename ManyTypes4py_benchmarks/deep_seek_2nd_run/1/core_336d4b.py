import codecs
import enum
import math
import operator
import random
import re
import string
import sys
import typing
import warnings
from collections.abc import Collection, Hashable, Iterable, Sequence
from contextvars import ContextVar
from decimal import Context, Decimal, localcontext
from fractions import Fraction
from functools import reduce
from inspect import Parameter, Signature, isabstract, isclass
from re import Pattern
from types import FunctionType, GenericAlias
from typing import (
    Annotated, Any, AnyStr, Callable, Literal, Optional, Protocol, TypeVar, Union, 
    cast, get_args, get_origin, overload, Type, Dict, List, Tuple, Set, FrozenSet,
    Iterator, Mapping, Sequence as TypingSequence, AbstractSet
)
from uuid import UUID
import attr
from hypothesis._settings import note_deprecation
from hypothesis.control import RandomSeeder, cleanup, current_build_context, deprecate_random_in_strategy, note, should_note
from hypothesis.errors import HypothesisSideeffectWarning, HypothesisWarning, InvalidArgument, ResolutionFailed, RewindRecursive, SmallSearchSpaceWarning
from hypothesis.internal.cathetus import cathetus
from hypothesis.internal.charmap import Categories, CategoryName, as_general_categories, categories as all_categories
from hypothesis.internal.compat import Concatenate, ParamSpec, bit_count, ceil, floor, get_type_hints, is_typed_named_tuple
from hypothesis.internal.conjecture.utils import calc_label_from_cls, check_sample, identity
from hypothesis.internal.entropy import get_seeder_and_restorer
from hypothesis.internal.floats import float_of
from hypothesis.internal.reflection import define_function_signature, get_pretty_function_description, get_signature, is_first_param_referenced_in_function, nicerepr, repr_call, required_args
from hypothesis.internal.validation import check_type, check_valid_integer, check_valid_interval, check_valid_magnitude, check_valid_size, check_valid_sizes, try_convert
from hypothesis.strategies._internal import SearchStrategy, check_strategy
from hypothesis.strategies._internal.collections import FixedAndOptionalKeysDictStrategy, FixedKeysDictStrategy, ListStrategy, TupleStrategy, UniqueListStrategy, UniqueSampledListStrategy, tuples
from hypothesis.strategies._internal.deferred import DeferredStrategy
from hypothesis.strategies._internal.functions import FunctionStrategy
from hypothesis.strategies._internal.lazy import LazyStrategy, unwrap_strategies
from hypothesis.strategies._internal.misc import BooleansStrategy, just, none, nothing
from hypothesis.strategies._internal.numbers import IntegersStrategy, Real, floats, integers
from hypothesis.strategies._internal.recursive import RecursiveStrategy
from hypothesis.strategies._internal.shared import SharedStrategy
from hypothesis.strategies._internal.strategies import Ex, SampledFromStrategy, T, one_of
from hypothesis.strategies._internal.strings import BytesStrategy, OneCharStringStrategy, TextStrategy, _check_is_single_character
from hypothesis.strategies._internal.utils import cacheable, defines_strategy
from hypothesis.utils.conventions import not_set
from hypothesis.vendor.pretty import RepresentationPrinter

if sys.version_info >= (3, 10):
    from types import EllipsisType as EllipsisType
elif typing.TYPE_CHECKING:
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)

@cacheable
@defines_strategy(force_reusable_values=True)
def booleans() -> SearchStrategy[bool]:
    """Returns a strategy which generates instances of :class:`python:bool`."""
    return BooleansStrategy()

@overload
def sampled_from(elements: Sequence[Ex]) -> SearchStrategy[Ex]: ...
@overload
def sampled_from(elements: Type[enum.Enum]) -> SearchStrategy[enum.Enum]: ...
@overload
def sampled_from(elements: Type[enum.Flag]) -> SearchStrategy[enum.Flag]: ...

@defines_strategy(try_non_lazy=True)
def sampled_from(elements: Union[Sequence[Ex], Type[enum.Enum], Type[enum.Flag]]) -> SearchStrategy[Union[Ex, enum.Enum, enum.Flag]]:
    """Returns a strategy which generates any value present in ``elements``."""
    values = check_sample(elements, 'sampled_from')
    try:
        if isinstance(elements, type) and issubclass(elements, enum.Enum):
            repr_ = f'sampled_from({elements.__module__}.{elements.__name__})'
        else:
            repr_ = f'sampled_from({elements!r})'
    except Exception:
        repr_ = None
    if isclass(elements) and issubclass(elements, enum.Flag):
        flags = sorted(set(elements.__members__.values()), key=lambda v: (bit_count(v.value), v.value))
        flags_with_empty = flags
        if not flags or flags[0].value != 0:
            try:
                flags_with_empty = [*flags, elements(0)]
            except TypeError:
                pass
        inner = [sampled_from(flags_with_empty)]
        if len(flags) > 1:
            inner += [integers(min_value=1, max_value=len(flags)).flatmap(lambda r: sets(sampled_from(flags), min_size=r, max_size=r)).map(lambda s: elements(reduce(operator.or_, s)))]
        return LazyStrategy(one_of, args=inner, kwargs={}, force_repr=repr_)
    if not values:
        if isinstance(elements, type) and issubclass(elements, enum.Enum) and vars(elements).get('__annotations__'):
            raise InvalidArgument(f'Cannot sample from {elements.__module__}.{elements.__name__} because it contains no elements.  It does however have annotations, so maybe you tried to write an enum as if it was a dataclass?')
        raise InvalidArgument('Cannot sample from a length-zero sequence.')
    if len(values) == 1:
        return just(values[0])
    return SampledFromStrategy(values, repr_)

@cacheable
@defines_strategy()
def lists(
    elements: SearchStrategy[Ex], 
    *, 
    min_size: int = 0, 
    max_size: Optional[int] = None, 
    unique_by: Optional[Union[Callable[[Ex], Hashable], Tuple[Callable[[Ex], Hashable], ...]]] = None, 
    unique: bool = False
) -> SearchStrategy[List[Ex]]:
    """Returns a list containing values drawn from elements with length in the interval [min_size, max_size]."""
    check_valid_sizes(min_size, max_size)
    check_strategy(elements, 'elements')
    if unique:
        if unique_by is not None:
            raise InvalidArgument('cannot specify both unique and unique_by (you probably only want to set unique_by)')
        else:
            unique_by = identity
    if max_size == 0:
        return builds(list)
    if unique_by is not None:
        if not (callable(unique_by) or isinstance(unique_by, tuple)):
            raise InvalidArgument(f'unique_by={unique_by!r} is not a callable or tuple of callables')
        if callable(unique_by):
            unique_by = (unique_by,)
        if len(unique_by) == 0:
            raise InvalidArgument('unique_by is empty')
        for i, f in enumerate(unique_by):
            if not callable(f):
                raise InvalidArgument(f'unique_by[{i}]={f!r} is not a callable')
        tuple_suffixes = None
        if isinstance(elements, TupleStrategy) and len(elements.element_strategies) >= 1 and (len(unique_by) == 1) and (isinstance(unique_by[0], operator.itemgetter) and repr(unique_by[0]) == 'operator.itemgetter(0)' or (isinstance(unique_by[0], FunctionType) and re.fullmatch(get_pretty_function_description(unique_by[0]), 'lambda ([a-z]+): \\1\\[0\\]'))):
            unique_by = (identity,)
            tuple_suffixes = TupleStrategy(elements.element_strategies[1:])
            elements = elements.element_strategies[0]
        if isinstance(elements, IntegersStrategy) and elements.start is not None and (elements.end is not None) and (elements.end - elements.start <= 255):
            elements = SampledFromStrategy(sorted(range(elements.start, elements.end + 1), key=abs) if elements.end < 0 or elements.start > 0 else list(range(elements.end + 1)) + list(range(-1, elements.start - 1, -1)))
        if isinstance(elements, SampledFromStrategy):
            element_count = len(elements.elements)
            if min_size > element_count:
                raise InvalidArgument(f'Cannot create a collection of min_size={min_size!r} unique elements with values drawn from only {element_count} distinct elements')
            if max_size is not None:
                max_size = min(max_size, element_count)
            else:
                max_size = element_count
            return UniqueSampledListStrategy(elements=elements, max_size=max_size, min_size=min_size, keys=unique_by, tuple_suffixes=tuple_suffixes)
        return UniqueListStrategy(elements=elements, max_size=max_size, min_size=min_size, keys=unique_by, tuple_suffixes=tuple_suffixes)
    return ListStrategy(elements, min_size=min_size, max_size=max_size)

@cacheable
@defines_strategy()
def sets(
    elements: SearchStrategy[Ex], 
    *, 
    min_size: int = 0, 
    max_size: Optional[int] = None
) -> SearchStrategy[Set[Ex]]:
    """This has the same behaviour as lists, but returns sets instead."""
    return lists(elements=elements, min_size=min_size, max_size=max_size, unique=True).map(set)

@cacheable
@defines_strategy()
def frozensets(
    elements: SearchStrategy[Ex], 
    *, 
    min_size: int = 0, 
    max_size: Optional[int] = None
) -> SearchStrategy[FrozenSet[Ex]]:
    """This is identical to the sets function but instead returns frozensets."""
    return lists(elements=elements, min_size=min_size, max_size=max_size, unique=True).map(frozenset)

class PrettyIter:
    def __init__(self, values: Iterable[Ex]) -> None:
        self._values = values
        self._iter = iter(self._values)

    def __iter__(self) -> Iterator[Ex]:
        return self._iter

    def __next__(self) -> Ex:
        return next(self._iter)

    def __repr__(self) -> str:
        return f'iter({self._values!r})'

@defines_strategy()
def iterables(
    elements: SearchStrategy[Ex], 
    *, 
    min_size: int = 0, 
    max_size: Optional[int] = None, 
    unique_by: Optional[Union[Callable[[Ex], Hashable], Tuple[Callable[[Ex], Hashable], ...]]] = None, 
    unique: bool = False
) -> SearchStrategy[Iterable[Ex]]:
    """This has the same behaviour as lists, but returns iterables instead."""
    return lists(elements=elements, min_size=min_size, max_size=max_size, unique_by=unique_by, unique=unique).map(PrettyIter)

@defines_strategy()
def fixed_dictionaries(
    mapping: Dict[str, SearchStrategy[Ex]], 
    *, 
    optional: Optional[Dict[str, SearchStrategy[Ex]]] = None
) -> SearchStrategy[Dict[str, Ex]]:
    """Generates a dictionary with a fixed set of keys mapping to strategies."""
    check_type(dict, mapping, 'mapping')
    for k, v in mapping.items():
        check_strategy(v, f'mapping[{k!r}]')
    if optional is not None:
        check_type(dict, optional, 'optional')
        for k, v in optional.items():
            check_strategy(v, f'optional[{k!r}]')
        if type(mapping) != type(optional):
            raise InvalidArgument('Got arguments of different types: mapping=%s, optional=%s' % (nicerepr(type(mapping)), nicerepr(type(optional))))
        if set(mapping) & set(optional):
            raise InvalidArgument(f'The following keys were in both mapping and optional, which is invalid: {set(mapping) & set(optional)!r}')
        return FixedAndOptionalKeysDictStrategy(mapping, optional)
    return FixedKeysDictStrategy(mapping)

@cacheable
@defines_strategy()
def dictionaries(
    keys: SearchStrategy[K], 
    values: SearchStrategy[V], 
    *, 
    dict_class: Type[Dict[K, V]] = dict, 
    min_size: int = 0, 
    max_size: Optional[int] = None
) -> SearchStrategy[Dict[K, V]]:
    """Generates dictionaries with keys drawn from keys and values from values."""
    check_valid_sizes(min_size, max_size)
    if max_size == 0:
        return fixed_dictionaries(dict_class())
    check_strategy(keys, 'keys')
    check_strategy(values, 'values')
    return lists(tuples(keys, values), min_size=min_size, max_size=max_size, unique_by=operator.itemgetter(0)).map(dict_class)

@cacheable
@defines_strategy(force_reusable_values=True)
def characters(
    *, 
    codec: Optional[str] = None, 
    min_codepoint: Optional[int] = None, 
    max_codepoint: Optional[int] = None, 
    categories: Optional[Categories] = None, 
    exclude_categories: Optional[Categories] = None, 
    exclude_characters: str = '', 
    include_characters: str = '', 
    blacklist_categories: Optional[Categories] = None, 
    whitelist_categories: Optional[Categories] = None, 
    blacklist_characters: Optional[str] = None, 
    whitelist_characters: Optional[str] = None, 
    allow_nan: Optional[bool] = None, 
    allow_infinity: Optional[bool] = None
) -> SearchStrategy[str]:
    """Generates characters, length-one strings, following specified filtering rules."""
    check_valid_size(min_codepoint, 'min_codepoint')
    check_valid_size(max_codepoint, 'max_codepoint')
    check_valid_interval(min_codepoint, max_codepoint, 'min_codepoint', 'max_codepoint')
    categories = cast(Optional[Categories], categories)
    if categories is not None and exclude_categories is not None:
        raise InvalidArgument(f"Pass at most one of categories={categories!r} and exclude_categories={exclude_categories!r} - these arguments both specify which categories are allowed, so it doesn't make sense to use both in a single call.")
    has_old_arg = any((v is not None for k, v in locals().items() if 'list' in k))
    has_new_arg = any((v is not None for k, v in locals().items() if 'lude' in k))
    if has_old_arg and has_new_arg:
        raise InvalidArgument('The deprecated blacklist/whitelist arguments cannot be used in the same call as their replacement include/exclude arguments.')
    if blacklist_categories is not None:
        exclude_categories = blacklist_categories
    if whitelist_categories is not None:
        categories = whitelist_categories
    if blacklist_characters is not None:
        exclude_characters = blacklist_characters
    if whitelist_characters is not None:
        include_characters = whitelist_characters
    if min_codepoint is None and max_codepoint is None and (categories is None) and (exclude_categories is None) and (include_characters is not None) and (codec is None):
        raise InvalidArgument(f'Nothing is excluded by other arguments, so passing only include_characters={include_characters!r} would have no effect.  Also pass categories=(), or use sampled_from({include_characters!r}) instead.')
    exclude_characters = exclude_characters or ''
    include_characters = include_characters or ''
    overlap = set(exclude_characters).intersection(include_characters)
    if overlap:
        raise InvalidArgument(f'Characters {sorted(overlap)!r} are present in both include_characters={include_characters!r} and exclude_characters={exclude_characters!r}')
    if categories is not None:
        categories = as_general_categories(categories, 'categories')
    if exclude_categories is not None:
        exclude_categories = as_general_categories(exclude_categories, 'exclude_categories')
    if categories is not None and (not categories) and (not include_characters):
        raise InvalidArgument('When `categories` is an empty collection and there are no characters specified in include_characters, nothing can be generated by the characters() strategy.')
    both_cats = set(exclude_categories or ()).intersection(categories or ())
    if both_cats:
        raise InvalidArgument(f'Categories {sorted(both_cats)!r} are present in both categories={categories!r} and exclude_categories={exclude_categories!r}')
    elif exclude_categories is not None:
        categories = set(all_categories()) - set(exclude_categories)
    del exclude_categories
    if codec is not None:
        try:
            codec = codecs.lookup(codec).name
            ''.encode(codec)
        except LookupError:
            raise InvalidArgument(f'codec={codec!r} is not valid on this system') from None
        except Exception:
            raise InvalidArgument(f'codec={codec!r} is not a valid codec') from None
        for char in include_characters:
            try:
                char.encode(encoding=codec, errors='strict')
            except UnicodeEncodeError:
                raise InvalidArgument(f'Character {char!r} in include_characters={include_characters!r} cannot be encoded with codec={codec!r}') from None
        if codec == 'ascii':
            if max_codepoint is None or max_codepoint > 127:
                max_codepoint = 127
            codec = None
        elif