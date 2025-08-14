from __future__ import annotations
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
from collections.abc import Collection, Hashable, Iterable, Sequence, Iterator
from contextvars import ContextVar
from decimal import Context, Decimal, localcontext
from fractions import Fraction
from functools import reduce
from inspect import Parameter, Signature, isabstract, isclass
from re import Pattern
from types import FunctionType, GenericAlias
from typing import (
    Annotated,
    Any,
    AnyStr,
    Callable,
    Dict,
    Iterator as TypingIterator,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)
from uuid import UUID

import attr

from hypothesis._settings import note_deprecation
from hypothesis.control import (
    RandomSeeder,
    cleanup,
    current_build_context,
    deprecate_random_in_strategy,
    note,
    should_note,
)
from hypothesis.errors import (
    HypothesisSideeffectWarning,
    HypothesisWarning,
    InvalidArgument,
    ResolutionFailed,
    RewindRecursive,
    SmallSearchSpaceWarning,
)
from hypothesis.internal.cathetus import cathetus
from hypothesis.internal.charmap import (
    Categories,
    CategoryName,
    as_general_categories,
    categories as all_categories,
)
from hypothesis.internal.compat import (
    Concatenate,
    ParamSpec,
    bit_count,
    ceil,
    floor,
    get_type_hints,
    is_typed_named_tuple,
)
from hypothesis.internal.conjecture.utils import (
    calc_label_from_cls,
    check_sample,
    identity,
)
from hypothesis.internal.entropy import get_seeder_and_restorer
from hypothesis.internal.floats import float_of
from hypothesis.internal.reflection import (
    define_function_signature,
    get_pretty_function_description,
    get_signature,
    is_first_param_referenced_in_function,
    nicerepr,
    repr_call,
    required_args,
)
from hypothesis.internal.validation import (
    check_type,
    check_valid_integer,
    check_valid_interval,
    check_valid_magnitude,
    check_valid_size,
    check_valid_sizes,
    try_convert,
)
from hypothesis.strategies._internal import SearchStrategy, check_strategy
from hypothesis.strategies._internal.collections import (
    FixedAndOptionalKeysDictStrategy,
    FixedKeysDictStrategy,
    ListStrategy,
    TupleStrategy,
    UniqueListStrategy,
    UniqueSampledListStrategy,
    tuples,
)
from hypothesis.strategies._internal.deferred import DeferredStrategy
from hypothesis.strategies._internal.functions import FunctionStrategy
from hypothesis.strategies._internal.lazy import LazyStrategy, unwrap_strategies
from hypothesis.strategies._internal.misc import BooleansStrategy, just, none, nothing
from hypothesis.strategies._internal.numbers import (
    IntegersStrategy,
    Real,
    floats,
    integers,
)
from hypothesis.strategies._internal.recursive import RecursiveStrategy
from hypothesis.strategies._internal.shared import SharedStrategy
from hypothesis.strategies._internal.strategies import (
    Ex,
    SampledFromStrategy,
    T,
    one_of,
)
from hypothesis.strategies._internal.strings import (
    BytesStrategy,
    OneCharStringStrategy,
    TextStrategy,
    _check_is_single_character,
)
from hypothesis.strategies._internal.utils import cacheable, defines_strategy
from hypothesis.utils.conventions import not_set
from hypothesis.vendor.pretty import RepresentationPrinter

if sys.version_info >= (3, 10):
    from types import EllipsisType as EllipsisType
elif typing.TYPE_CHECKING:  # pragma: no cover
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)  # pragma: no cover


@cacheable
@defines_strategy(force_reusable_values=True)
def booleans() -> SearchStrategy[bool]:
    return BooleansStrategy()


@overload
def sampled_from(elements: Sequence[T]) -> SearchStrategy[T]:  # pragma: no cover
    ...


@overload
def sampled_from(elements: type[enum.Enum]) -> SearchStrategy[Any]:  # pragma: no cover
    ...


@overload
def sampled_from(
    elements: Union[type[enum.Enum], Sequence[Any]],
) -> SearchStrategy[Any]:  # pragma: no cover
    ...


@defines_strategy(try_non_lazy=True)
def sampled_from(
    elements: Union[type[enum.Enum], Sequence[Any]],
) -> SearchStrategy[Any]:
    values = check_sample(elements, "sampled_from")
    try:
        if isinstance(elements, type) and issubclass(elements, enum.Enum):
            repr_ = f"sampled_from({elements.__module__}.{elements.__name__})"
        else:
            repr_ = f"sampled_from({elements!r})"
    except Exception:  # pragma: no cover
        repr_ = None
    if isclass(elements) and issubclass(elements, enum.Flag):
        flags = sorted(
            set(elements.__members__.values()),
            key=lambda v: (bit_count(v.value), v.value),
        )
        flags_with_empty = flags
        if not flags or flags[0].value != 0:
            try:
                flags_with_empty = [*flags, elements(0)]
            except TypeError:  # pragma: no cover
                pass
        inner = [
            sampled_from(flags_with_empty),
        ]
        if len(flags) > 1:
            inner += [
                integers(min_value=1, max_value=len(flags))
                .flatmap(lambda r: sets(sampled_from(flags), min_size=r, max_size=r))
                .map(lambda s: elements(reduce(operator.or_, s))),
            ]
        return LazyStrategy(one_of, args=inner, kwargs={}, force_repr=repr_)
    if not values:
        if (
            isinstance(elements, type)
            and issubclass(elements, enum.Enum)
            and vars(elements).get("__annotations__")
        ):
            raise InvalidArgument(
                f"Cannot sample from {elements.__module__}.{elements.__name__} "
                "because it contains no elements.  It does however have annotations, "
                "so maybe you tried to write an enum as if it was a dataclass?"
            )
        raise InvalidArgument("Cannot sample from a length-zero sequence.")
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
    unique_by: Union[
        None,
        Callable[[Ex], Hashable],
        tuple[Callable[[Ex], Hashable], ...],
    ] = None,
    unique: bool = False,
) -> SearchStrategy[list[Ex]]:
    check_valid_sizes(min_size, max_size)
    check_strategy(elements, "elements")
    if unique:
        if unique_by is not None:
            raise InvalidArgument(
                "cannot specify both unique and unique_by "
                "(you probably only want to set unique_by)"
            )
        else:
            unique_by = identity

    if max_size == 0:
        return builds(list)
    if unique_by is not None:
        if not (callable(unique_by) or isinstance(unique_by, tuple)):
            raise InvalidArgument(
                f"{unique_by=} is not a callable or tuple of callables"
            )
        if callable(unique_by):
            unique_by = (unique_by,)
        if len(unique_by) == 0:
            raise InvalidArgument("unique_by is empty")
        for i, f in enumerate(unique_by):
            if not callable(f):
                raise InvalidArgument(f"unique_by[{i}]={f!r} is not a callable")
        tuple_suffixes = None
        if (
            isinstance(elements, TupleStrategy)
            and len(elements.element_strategies) >= 1
            and len(unique_by) == 1
            and (
                (isinstance(unique_by[0], operator.itemgetter)
                 and repr(unique_by[0]) == "operator.itemgetter(0)")
                or (
                    isinstance(unique_by[0], FunctionType)
                    and re.fullmatch(
                        get_pretty_function_description(unique_by[0]),
                        r"lambda ([a-z]+): \1\[0\]",
                    )
                )
            )
        ):
            unique_by = (identity,)
            tuple_suffixes = TupleStrategy(elements.element_strategies[1:])
            elements = elements.element_strategies[0]

        if (
            isinstance(elements, IntegersStrategy)
            and elements.start is not None  # type: ignore
            and elements.end is not None  # type: ignore
            and (elements.end - elements.start) <= 255  # type: ignore
        ):
            elements = SampledFromStrategy(
                sorted(range(elements.start, elements.end + 1), key=abs)  # type: ignore
                if elements.end < 0 or elements.start > 0  # type: ignore
                else list(range(elements.end + 1))  # type: ignore
                + list(range(-1, elements.start - 1, -1))  # type: ignore
            )

        if isinstance(elements, SampledFromStrategy):
            element_count = len(elements.elements)
            if min_size > element_count:
                raise InvalidArgument(
                    f"Cannot create a collection of {min_size=} unique "
                    f"elements with values drawn from only {element_count} distinct "
                    "elements"
                )

            if max_size is not None:
                max_size = min(max_size, element_count)
            else:
                max_size = element_count

            return UniqueSampledListStrategy(
                elements=elements,
                max_size=max_size,
                min_size=min_size,
                keys=unique_by,
                tuple_suffixes=tuple_suffixes,
            )

        return UniqueListStrategy(
            elements=elements,
            max_size=max_size,
            min_size=min_size,
            keys=unique_by,
            tuple_suffixes=tuple_suffixes,
        )
    return ListStrategy(elements, min_size=min_size, max_size=max_size)


@cacheable
@defines_strategy()
def sets(
    elements: SearchStrategy[Ex],
    *,
    min_size: int = 0,
    max_size: Optional[int] = None,
) -> SearchStrategy[set[Ex]]:
    return lists(
        elements=elements, min_size=min_size, max_size=max_size, unique=True
    ).map(set)


@cacheable
@defines_strategy()
def frozensets(
    elements: SearchStrategy[Ex],
    *,
    min_size: int = 0,
    max_size: Optional[int] = None,
) -> SearchStrategy[frozenset[Ex]]:
    return lists(
        elements=elements, min_size=min_size, max_size=max_size, unique=True
    ).map(frozenset)


class PrettyIter:
    def __init__(self, values: Sequence[Any]) -> None:
        self._values: Sequence[Any] = values
        self._iter: Iterator[Any] = iter(self._values)

    def __iter__(self) -> Iterator[Any]:
        return self._iter

    def __next__(self) -> Any:
        return next(self._iter)

    def __repr__(self) -> str:
        return f"iter({self._values!r})"


@defines_strategy()
def iterables(
    elements: SearchStrategy[Ex],
    *,
    min_size: int = 0,
    max_size: Optional[int] = None,
    unique_by: Union[
        None,
        Callable[[Ex], Hashable],
        tuple[Callable[[Ex], Hashable], ...],
    ] = None,
    unique: bool = False,
) -> SearchStrategy[Iterable[Ex]]:
    return lists(
        elements=elements,
        min_size=min_size,
        max_size=max_size,
        unique_by=unique_by,
        unique=unique,
    ).map(PrettyIter)


@defines_strategy()
def fixed_dictionaries(
    mapping: dict[T, SearchStrategy[Ex]],
    *,
    optional: Optional[dict[T, SearchStrategy[Ex]]] = None,
) -> SearchStrategy[dict[T, Ex]]:
    check_type(dict, mapping, "mapping")
    for k, v in mapping.items():
        check_strategy(v, f"mapping[{k!r}]")
    if optional is not None:
        check_type(dict, optional, "optional")
        for k, v in optional.items():
            check_strategy(v, f"optional[{k!r}]")
        if type(mapping) != type(optional):
            raise InvalidArgument(
                "Got arguments of different types: mapping=%s, optional=%s"
                % (nicerepr(type(mapping)), nicerepr(type(optional)))
            )
        if set(mapping) & set(optional):
            raise InvalidArgument(
                "The following keys were in both mapping and optional, "
                f"which is invalid: {set(mapping) & set(optional)!r}"
            )
        return FixedAndOptionalKeysDictStrategy(mapping, optional)
    return FixedKeysDictStrategy(mapping)


@cacheable
@defines_strategy()
def dictionaries(
    keys: SearchStrategy[Ex],
    values: SearchStrategy[T],
    *,
    dict_class: type = dict,
    min_size: int = 0,
    max_size: Optional[int] = None,
) -> SearchStrategy[dict[Ex, T]]:
    check_valid_sizes(min_size, max_size)
    if max_size == 0:
        return fixed_dictionaries(dict_class())
    check_strategy(keys, "keys")
    check_strategy(values, "values")

    return lists(
        tuples(keys, values),
        min_size=min_size,
        max_size=max_size,
        unique_by=operator.itemgetter(0),
    ).map(dict_class)


@cacheable
@defines_strategy(force_reusable_values=True)
def characters(
    *,
    codec: Optional[str] = None,
    min_codepoint: Optional[int] = None,
    max_codepoint: Optional[int] = None,
    categories: Optional[Collection[CategoryName]] = None,
    exclude_categories: Optional[Collection[CategoryName]] = None,
    exclude_characters: Optional[Collection[str]] = None,
    include_characters: Optional[Collection[str]] = None,
    blacklist_categories: Optional[Collection[CategoryName]] = None,
    whitelist_categories: Optional[Collection[CategoryName]] = None,
    blacklist_characters: Optional[Collection[str]] = None,
    whitelist_characters: Optional[Collection[str]] = None,
) -> SearchStrategy[str]:
    check_valid_size(min_codepoint, "min_codepoint")
    check_valid_size(max_codepoint, "max_codepoint")
    check_valid_interval(min_codepoint, max_codepoint, "min_codepoint", "max_codepoint")
    categories = cast(Optional[Categories], categories)
    if categories is not None and exclude_categories is not None:
        raise InvalidArgument(
            f"Pass at most one of {categories=} and {exclude_categories=} - "
            "these arguments both specify which categories are allowed, so it "
            "doesn't make sense to use both in a single call."
        )
    has_old_arg = any(v is not None for k, v in locals().items() if "list" in k)
    has_new_arg = any(v is not None for k, v in locals().items() if "lude" in k)
    if has_old_arg and has_new_arg:
        raise InvalidArgument(
            "The deprecated blacklist/whitelist arguments cannot be used in "
            "the same call as their replacement include/exclude arguments."
        )
    if blacklist_categories is not None:
        exclude_categories = blacklist_categories
    if whitelist_categories is not None:
        categories = whitelist_categories
    if blacklist_characters is not None:
        exclude_characters = blacklist_characters
    if whitelist_characters is not None:
        include_characters = whitelist_characters

    if (
        min_codepoint is None
        and max_codepoint is None
        and categories is None
        and exclude_categories is None
        and include_characters is not None
        and codec is None
    ):
        raise InvalidArgument(
            "Nothing is excluded by other arguments, so passing only "
            f"{include_characters=} would have no effect.  "
            "Also pass categories=(), or use "
            f"sampled_from({include_characters!r}) instead."
        )
    exclude_characters = exclude_characters or ""
    include_characters = include_characters or ""
    overlap = set(exclude_characters).intersection(include_characters)
    if overlap:
        raise InvalidArgument(
            f"Characters {sorted(overlap)!r} are present in both "
            f"{include_characters=} and {exclude_characters=}"
        )
    if categories is not None:
        categories = as_general_categories(categories, "categories")
    if exclude_categories is not None:
        categories = set(all_categories()) - set(exclude_categories)
    del exclude_categories

    if codec is not None:
        try:
            codec = codecs.lookup(codec).name
            "".encode(codec)
        except LookupError:
            raise InvalidArgument(f"{codec=} is not valid on this system") from None
        except Exception:
            raise InvalidArgument(f"{codec=} is not a valid codec") from None

        for char in include_characters:
            try:
                char.encode(encoding=codec, errors="strict")
            except UnicodeEncodeError:
                raise InvalidArgument(
                    f"Character {char!r} in {include_characters=} "
                    f"cannot be encoded with {codec=}"
                ) from None

        if codec == "ascii":
            if (max_codepoint is None) or (max_codepoint > 127):
                max_codepoint = 127
            codec = None
        elif codec == "utf-8":
            if categories is None:
                categories = all_categories()
            categories = tuple(c for c in categories if c != "Cs")

    return OneCharStringStrategy.from_characters_args(
        categories=categories,
        exclude_characters=exclude_characters,
        min_codepoint=min_codepoint,
        max_codepoint=max_codepoint,
        include_characters=include_characters,
        codec=codec,
    )


characters.__signature__ = (__sig := get_signature(characters)).replace(  # type: ignore
    parameters=[p for p in __sig.parameters.values() if "list" not in p.name]
)


@cacheable
@defines_strategy(force_reusable_values=True)
def text(
    alphabet: Union[Collection[str], SearchStrategy[str]] = characters(codec="utf-8"),
    *,
    min_size: int = 0,
    max_size: Optional[int] = None,
) -> SearchStrategy[str]:
    check_valid_sizes(min_size, max_size)
    if isinstance(alphabet, SearchStrategy):
        char_strategy = unwrap_strategies(alphabet)
        if isinstance(char_strategy, SampledFromStrategy):
            return text(char_strategy.elements, min_size=min_size, max_size=max_size)
        elif not isinstance(char_strategy, OneCharStringStrategy):
            char_strategy = char_strategy.map(_check_is_single_character)
    else:
        non_string = [c for c in alphabet if not isinstance(c, str)]
        if non_string:
            raise InvalidArgument(
                "The following elements in alphabet are not unicode "
                f"strings:  {non_string!r}"
            )
        not_one_char = [c for c in alphabet if len(c) != 1]
        if not_one_char:
            raise InvalidArgument(
                "The following elements in alphabet are not of length one, "
                f"which leads to violation of size constraints:  {not_one_char!r}"
            )
        if alphabet in ["ascii", "utf-8"]:
            warnings.warn(
                f"st.text({alphabet!r}): it seems like you are trying to use the "
                f"codec {alphabet!r}. st.text({alphabet!r}) instead generates "
                f"strings using the literal characters {list(alphabet)!r}. To specify "
                f"the {alphabet} codec, use st.text(st.characters(codec={alphabet!r})). "
                "If you intended to use character literals, you can silence this "
                "warning by reordering the characters.",
                HypothesisWarning,
                stacklevel=1,
            )
        char_strategy = (
            characters(categories=(), include_characters=alphabet)
            if alphabet
            else nothing()
        )
    if (max_size == 0 or char_strategy.is_empty) and not min_size:
        return just("")
    return TextStrategy(char_strategy, min_size=min_size, max_size=max_size)


@overload
def from_regex(
    regex: Union[bytes, Pattern[bytes]],
    *,
    fullmatch: bool = False,
) -> SearchStrategy[bytes]:  # pragma: no cover
    ...


@overload
def from_regex(
    regex: Union[str, Pattern[str]],
    *,
    fullmatch: bool = False,
    alphabet: Union[str, SearchStrategy[str]] = characters(codec="utf-8"),
) -> SearchStrategy[str]:  # pragma: no cover
    ...


@cacheable
@defines_strategy()
def from_regex(
    regex: Union[AnyStr, Pattern[AnyStr]],
    *,
    fullmatch: bool = False,
    alphabet: Union[str, SearchStrategy[str], None] = None,
) -> SearchStrategy[AnyStr]:
    check_type((str, bytes, re.Pattern), regex, "regex")
    check_type(bool, fullmatch, "fullmatch")
    pattern = regex.pattern if isinstance(regex, re.Pattern) else regex
    if alphabet is not None:
        check_type((str, SearchStrategy), alphabet, "alphabet")
        if not isinstance(pattern, str):
            raise InvalidArgument("alphabet= is not supported for bytestrings")
        alphabet = OneCharStringStrategy.from_alphabet(alphabet)
    elif isinstance(pattern, str):
        alphabet = characters(codec="utf-8")
    from hypothesis.strategies._internal.regex import regex_strategy
    return regex_strategy(regex, fullmatch, alphabet=alphabet)


@cacheable
@defines_strategy(force_reusable_values=True)
def binary(
    *,
    min_size: int = 0,
    max_size: Optional[int] = None,
) -> SearchStrategy[bytes]:
    check_valid_sizes(min_size, max_size)
    return BytesStrategy(min_size, max_size)


@cacheable
@defines_strategy()
def randoms(
    *,
    note_method_calls: bool = False,
    use_true_random: bool = False,
) -> SearchStrategy[random.Random]:
    check_type(bool, note_method_calls, "note_method_calls")
    check_type(bool, use_true_random, "use_true_random")
    from hypothesis.strategies._internal.random import RandomStrategy
    return RandomStrategy(
        use_true_random=use_true_random, note_method_calls=note_method_calls
    )


class RandomModule(SearchStrategy):
    def __init__(self, default: Any) -> None:
        self.default = default

    def do_draw(self, data: Any) -> RandomSeeder:
        seed = data.draw(integers(0, 2**32 - 1))
        seed_all, restore_all = get_seeder_and_restorer(seed)
        seed_all()
        cleanup(restore_all)
        return RandomSeeder(seed)


@cacheable
@defines_strategy()
def random_module() -> SearchStrategy[RandomSeeder]:
    return shared(RandomModule(not_set), key="hypothesis.strategies.random_module()")


class BuildsStrategy(SearchStrategy):
    def __init__(
        self,
        target: Callable[..., Any],
        args: Tuple[SearchStrategy[Any], ...],
        kwargs: Dict[str, SearchStrategy[Any]],
    ) -> None:
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def do_draw(self, data: Any) -> Any:
        args = [data.draw(a) for a in self.args]
        kwargs = {k: data.draw(v) for k, v in self.kwargs.items()}
        try:
            obj = self.target(*args, **kwargs)
        except TypeError as err:
            if (
                isinstance(self.target, type)
                and issubclass(self.target, enum.Enum)
                and not (self.args or self.kwargs)
            ):
                name = self.target.__module__ + "." + self.target.__qualname__
                raise InvalidArgument(
                    f"Calling {name} with no arguments raised an error - "
                    f"try using sampled_from({name}) instead of builds({name})"
                ) from err
            if not (self.args or self.kwargs):
                from .types import is_a_new_type, is_generic_type

                if is_a_new_type(self.target) or is_generic_type(self.target):
                    raise InvalidArgument(
                        f"Calling {self.target!r} with no arguments raised an "
                        f"error - try using from_type({self.target!r}) instead "
                        f"of builds({self.target!r})"
                    ) from err
            if getattr(self.target, "__no_type_check__", None) is True:
                raise TypeError(
                    "This might be because the @no_type_check decorator prevented "
                    "Hypothesis from inferring a strategy for some required arguments."
                ) from err
            raise
        current_build_context().record_call(obj, self.target, args, kwargs)
        return obj

    def validate(self) -> None:
        tuples(*self.args).validate()
        fixed_dictionaries(self.kwargs).validate()

    def __repr__(self) -> str:
        bits = [get_pretty_function_description(self.target)]
        bits.extend(map(repr, self.args))
        bits.extend(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"builds({', '.join(bits)})"


@cacheable
@defines_strategy()
def builds(
    target: Callable[..., Ex],
    *args: SearchStrategy[Any],
    **kwargs: Union[SearchStrategy[Any], EllipsisType],
) -> SearchStrategy[Ex]:
    if not callable(target):
        raise InvalidArgument(
            "The first positional argument to builds() must be a callable "
            "target to construct."
        )
    if ... in args:  # type: ignore
        raise InvalidArgument(
            "... was passed as a positional argument to "
            "builds(), but is only allowed as a keyword arg"
        )
    required = required_args(target, args, kwargs)
    to_infer = {k for k, v in kwargs.items() if v is ...}
    if required or to_infer:
        if isinstance(target, type) and attr.has(target):
            from hypothesis.strategies._internal.attrs import from_attrs

            return from_attrs(target, args, kwargs, required | to_infer)
        hints = get_type_hints(target)
        if to_infer - set(hints):
            badargs = ", ".join(sorted(to_infer - set(hints)))
            raise InvalidArgument(
                f"passed ... for {badargs}, but we cannot infer a strategy "
                "because these arguments have no type annotation"
            )
        infer_for = {k: v for k, v in hints.items() if k in (required | to_infer)}
        if infer_for:
            from hypothesis.strategies._internal.types import _global_type_lookup

            for kw, t in infer_for.items():
                if t in _global_type_lookup:
                    kwargs[kw] = from_type(t)
                else:
                    kwargs[kw] = deferred(lambda t=t: from_type(t))  # type: ignore
    return BuildsStrategy(target, args, kwargs)


@cacheable
@defines_strategy(never_lazy=True)
def from_type(thing: type[T]) -> SearchStrategy[T]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return _from_type(thing)
    except Exception:
        return _from_type_deferred(thing)


def _from_type_deferred(thing: type[Ex]) -> SearchStrategy[Ex]:
    try:
        thing_repr = nicerepr(thing)
        if hasattr(thing, "__module__"):
            module_prefix = f"{thing.__module__}."
            if not thing_repr.startswith(module_prefix):
                thing_repr = module_prefix + thing_repr
        repr_ = f"from_type({thing_repr})"
    except Exception:  # pragma: no cover
        repr_ = None
    return LazyStrategy(
        lambda thing: deferred(lambda: _from_type(thing)),
        (thing,),
        {},
        force_repr=repr_,
    )


_recurse_guard: ContextVar[list[type[Any]]] = ContextVar("recurse_guard")


def _from_type(thing: type[Ex]) -> SearchStrategy[Ex]:
    from hypothesis.strategies._internal import types

    def as_strategy(
        strat_or_callable: Union[SearchStrategy[Any], Callable[[Any], SearchStrategy[Any]]],
        thing: Any,
    ) -> SearchStrategy[Any]:
        if not isinstance(strat_or_callable, SearchStrategy):
            assert callable(strat_or_callable)
            strategy = strat_or_callable(thing)
        else:
            strategy = strat_or_callable
        if strategy is NotImplemented:
            return NotImplemented
        if not isinstance(strategy, SearchStrategy):
            raise ResolutionFailed(
                f"Error: {thing} was registered for {nicerepr(strat_or_callable)}, "
                f"but returned non-strategy {strategy!r}"
            )
        if strategy.is_empty:
            raise ResolutionFailed(f"Error: {thing!r} resolved to an empty strategy")
        return strategy

    def from_type_guarded(thing: type[Ex]) -> SearchStrategy[Any]:
        try:
            recurse_guard = _recurse_guard.get()
        except LookupError:
            _recurse_guard.set(recurse_guard := [])
        if thing in recurse_guard:
            raise RewindRecursive(thing)
        recurse_guard.append(thing)
        try:
            return _from_type(thing)
        except RewindRecursive as rr:
            if rr.target != thing:
                raise
            return ...  # type: ignore
        finally:
            recurse_guard.pop()

    try:
        known = thing in types._global_type_lookup
    except TypeError:
        pass
    else:
        if not known:
            for module, resolver in types._global_extra_lookup.items():
                if module in sys.modules:
                    strat = resolver(thing)
                    if strat is not None:
                        return strat
    if not isinstance(thing, type):
        if types.is_a_new_type(thing):
            if thing in types._global_type_lookup:
                strategy = as_strategy(types._global_type_lookup[thing], thing)
                if strategy is not NotImplemented:
                    return strategy
            return _from_type(thing.__supertype__)
        if types.is_a_union(thing):
            args = sorted(thing.__args__, key=types.type_sorting_key)
            return one_of([_from_type(t) for t in args])
        if thing in types.LiteralStringTypes:
            return text()
    if isinstance(thing, TypeVar) and type(thing) in types._global_type_lookup:
        strategy = as_strategy(types._global_type_lookup[type(thing)], thing)
        if strategy is not NotImplemented:
            return strategy
    if not types.is_a_type(thing):
        if isinstance(thing, str):
            raise InvalidArgument(
                f"Got {thing!r} as a type annotation, but the forward-reference "
                "could not be resolved from a string to a type.  Consider using "
                "`from __future__ import annotations` instead of forward-reference "
                "strings."
            )
        raise InvalidArgument(f"{thing=} must be a type")
    if thing in types.NON_RUNTIME_TYPES:
        raise InvalidArgument(
            f"Could not resolve {thing!r} to a strategy, "
            f"because there is no such thing as a runtime instance of {thing!r}"
        )
    try:
        if thing in types._global_type_lookup:
            strategy = as_strategy(types._global_type_lookup[thing], thing)
            if strategy is not NotImplemented:
                return strategy
        elif (
            isinstance(thing, GenericAlias)
            and (to := get_origin(thing)) in types._global_type_lookup
        ):
            strategy = as_strategy(types._global_type_lookup[to], thing)
            if strategy is not NotImplemented:
                return strategy
    except TypeError:  # pragma: no cover
        pass
    if (hasattr(typing, "_TypedDictMeta") and type(thing) is typing._TypedDictMeta) or (
        hasattr(types.typing_extensions, "_TypedDictMeta")
        and type(thing) is types.typing_extensions._TypedDictMeta
    ):  # pragma: no cover

        def _get_annotation_arg(key: str, annotation_type: Any) -> Any:
            try:
                return get_args(annotation_type)[0]
            except IndexError:
                raise InvalidArgument(
                    f"`{key}: {annotation_type.__name__}` is not a valid type annotation"
                ) from None

        def _get_typeddict_qualifiers(key: str, annotation_type: Any) -> tuple[set[Any], Any]:
            qualifiers = []
            while True:
                annotation_origin = types.extended_get_origin(annotation_type)
                if annotation_origin is Annotated:
                    if annotation_args := get_args(annotation_type):
                        annotation_type = annotation_args[0]
                    else:
                        break
                elif annotation_origin in types.RequiredTypes:
                    qualifiers.append(types.RequiredTypes)
                    annotation_type = _get_annotation_arg(key, annotation_type)
                elif annotation_origin in types.NotRequiredTypes:
                    qualifiers.append(types.NotRequiredTypes)
                    annotation_type = _get_annotation_arg(key, annotation_type)
                elif annotation_origin in types.ReadOnlyTypes:
                    qualifiers.append(types.ReadOnlyTypes)
                    annotation_type = _get_annotation_arg(key, annotation_type)
                else:
                    break
            return set(qualifiers), annotation_type

        optional = set(getattr(thing, "__optional_keys__", ()))
        required = set(
            getattr(thing, "__required_keys__", get_type_hints(thing).keys())
        )
        anns: dict[str, SearchStrategy[Any]] = {}
        for k, v in get_type_hints(thing).items():
            qualifiers, v = _get_typeddict_qualifiers(k, v)
            if types.RequiredTypes in qualifiers:
                optional.discard(k)
                required.add(k)
            if types.NotRequiredTypes in qualifiers:
                optional.add(k)
                required.discard(k)
            anns[k] = from_type_guarded(v)
            if anns[k] is ...:
                anns[k] = _from_type_deferred(v)
        if not required.isdisjoint(optional):  # pragma: no cover
            raise InvalidArgument(
                f"Required keys overlap with optional keys in a TypedDict:"
                f" {required=}, {optional=}"
            )
        if (
            (not anns)
            and thing.__annotations__
            and ".<locals>." in getattr(thing, "__qualname__", "")
        ):
            raise InvalidArgument("Failed to retrieve type annotations for local type")
        return fixed_dictionaries(  # type: ignore
            mapping={k: v for k, v in anns.items() if k in required},
            optional={k: v for k, v in anns.items() if k in optional},
        )
    if isinstance(thing, types.typing_root_type) or (
        isinstance(get_origin(thing), type) and get_args(thing)
    ):
        return types.from_typing_type(thing)
    strategies = [
        s
        for s in (
            as_strategy(v, thing)
            for k, v in sorted(types._global_type_lookup.items(), key=repr)
            if isinstance(k, type)
            and issubclass(k, thing)
            and sum(types.try_issubclass(k, typ) for typ in types._global_type_lookup)
            == 1
        )
        if s is not NotImplemented
    ]
    if any(not s.is_empty for s in strategies):
        return one_of(strategies)
    if issubclass(thing, enum.Enum):
        return sampled_from(thing)
    if not isabstract(thing):
        required = required_args(thing)
        if required and not (
            required.issubset(get_type_hints(thing))
            or attr.has(thing)
            or is_typed_named_tuple(thing)
        ):
            raise ResolutionFailed(
                f"Could not resolve {thing!r} to a strategy; consider "
                "using register_type_strategy"
            )
        try:
            hints = get_type_hints(thing)
            params = get_signature(thing).parameters
        except Exception:
            params = {}  # type: ignore
        posonly_args = []  # type: list[SearchStrategy[Any]]
        kwargs: Dict[str, SearchStrategy[Any]] = {}
        for k, p in params.items():
            if (
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                and k in hints
                and k != "return"
            ):
                ps = from_type_guarded(hints[k])
                if p.default is not Parameter.empty and ps is not ...:
                    ps = just(p.default) | ps
                if p.kind is Parameter.POSITIONAL_ONLY:
                    if ps is ...:  # pragma: no cover
                        if p.default is Parameter.empty:
                            raise ResolutionFailed(
                                f"Could not resolve {thing!r} to a strategy; "
                                "consider using register_type_strategy"
                            )
                        ps = just(p.default)
                    posonly_args.append(ps)
                else:
                    kwargs[k] = ps
        if (
            params
            and not (posonly_args or kwargs)
            and not issubclass(thing, BaseException)
        ):
            from_type_repr = repr_call(from_type, (thing,), {})
            builds_repr = repr_call(builds, (thing,), {})
            warnings.warn(
                f"{from_type_repr} resolved to {builds_repr}, because we could not "
                "find any (non-varargs) arguments. Use st.register_type_strategy() "
                "to resolve to a strategy which can generate more than one value, "
                "or silence this warning.",
                SmallSearchSpaceWarning,
                stacklevel=2,
            )
        return builds(thing, *posonly_args, **kwargs)
    subclasses = thing.__subclasses__()
    if not subclasses:
        raise ResolutionFailed(
            f"Could not resolve {thing!r} to a strategy, because it is an abstract "
            "type without any subclasses. Consider using register_type_strategy"
        )
    subclass_strategies: SearchStrategy[Any] = nothing()
    for sc in subclasses:
        try:
            subclass_strategies |= _from_type(sc)
        except Exception:
            pass
    if subclass_strategies.is_empty:
        return sampled_from(subclasses).flatmap(_from_type)
    return subclass_strategies


@cacheable
@defines_strategy(force_reusable_values=True)
def fractions(
    min_value: Optional[Union[Real, str]] = None,
    max_value: Optional[Union[Real, str]] = None,
    *,
    max_denominator: Optional[int] = None,
) -> SearchStrategy[Fraction]:
    min_value = try_convert(Fraction, min_value, "min_value")
    max_value = try_convert(Fraction, max_value, "max_value")
    assert min_value is None or isinstance(min_value, Fraction)
    assert max_value is None or isinstance(max_value, Fraction)

    check_valid_interval(min_value, max_value, "min_value", "max_value")
    check_valid_integer(max_denominator, "max_denominator")

    if max_denominator is not None:
        if max_denominator < 1:
            raise InvalidArgument(f"{max_denominator=} must be >= 1")
        if min_value is not None and min_value.denominator > max_denominator:
            raise InvalidArgument(
                f"The {min_value=} has a denominator greater than the "
                f"{max_denominator=}"
            )
        if max_value is not None and max_value.denominator > max_denominator:
            raise InvalidArgument(
                f"The {max_value=} has a denominator greater than the "
                f"{max_denominator=}"
            )

    if min_value is not None and min_value == max_value:
        return just(min_value)

    def dm_func(denom: int) -> SearchStrategy[Fraction]:
        min_num: Optional[int] = None
        max_num: Optional[int] = None
        if max_value is None and min_value is None:
            pass
        elif min_value is None:
            max_num = denom * max_value.numerator  # type: ignore
            denom *= max_value.denominator  # type: ignore
        elif max_value is None:
            min_num = denom * min_value.numerator  # type: ignore
            denom *= min_value.denominator  # type: ignore
        else:
            low = min_value.numerator * max_value.denominator  # type: ignore
            high = max_value.numerator * min_value.denominator  # type: ignore
            scale = min_value.denominator * max_value.denominator  # type: ignore
            div = math.gcd(scale, math.gcd(low, high))
            min_num = denom * low // div
            max_num = denom * high // div
            denom *= scale // div
        return builds(
            Fraction, integers(min_value=min_num, max_value=max_num), just(denom)
        )

    if max_denominator is None:
        return integers(min_value=1).flatmap(dm_func)

    return (
        integers(1, max_denominator)
        .flatmap(dm_func)
        .map(lambda f: f.limit_denominator(max_denominator))
    )


def _as_finite_decimal(
    value: Union[Real, str, None],
    name: str,
    allow_infinity: Optional[bool],
) -> Optional[Decimal]:
    assert name in ("min_value", "max_value")
    if value is None:
        return None
    if not isinstance(value, Decimal):
        with localcontext(Context()):
            value = try_convert(Decimal, value, name)
    assert isinstance(value, Decimal)
    if value.is_finite():
        return value
    if value.is_infinite() and (value < 0 if "min" in name else value > 0):
        if allow_infinity or allow_infinity is None:
            return None
        raise InvalidArgument(f"{allow_infinity=}, but {name}={value!r}")
    raise InvalidArgument(f"Invalid {name}={value!r}")


@cacheable
@defines_strategy(force_reusable_values=True)
def decimals(
    min_value: Optional[Union[Real, str]] = None,
    max_value: Optional[Union[Real, str]] = None,
    *,
    allow_nan: Optional[bool] = None,
    allow_infinity: Optional[bool] = None,
    places: Optional[int] = None,
) -> SearchStrategy[Decimal]:
    check_valid_integer(places, "places")
    if places is not None and places < 0:
        raise InvalidArgument(f"{places=} may not be negative")
    min_value = _as_finite_decimal(min_value, "min_value", allow_infinity)
    max_value = _as_finite_decimal(max_value, "max_value", allow_infinity)
    check_valid_interval(min_value, max_value, "min_value", "max_value")
    if allow_infinity and (None not in (min_value, max_value)):
        raise InvalidArgument("Cannot allow infinity between finite bounds")
    if places is not None:
        def ctx(val: Decimal) -> Context:
            precision = ceil(math.log10(abs(val) or 1)) + places + 1
            return Context(prec=max([precision, 1]))

        def int_to_decimal(val: int) -> Decimal:
            context = ctx(Decimal(val))
            return context.quantize(context.multiply(val, factor), factor)

        factor = Decimal(10) ** -places
        min_num: Optional[int] = None
        max_num: Optional[int] = None
        if min_value is not None:
            min_num = ceil(ctx(min_value).divide(min_value, factor))
        if max_value is not None:
            max_num = floor(ctx(max_value).divide(max_value, factor))
        if min_num is not None and max_num is not None and min_num > max_num:
            raise InvalidArgument(
                f"There are no decimals with {places} places between "
                f"{min_value=} and {max_value=}"
            )
        strat = integers(min_num, max_num).map(int_to_decimal)
    else:
        def fraction_to_decimal(val: Fraction) -> Decimal:
            precision = (
                ceil(math.log10(abs(val.numerator) or 1) + math.log10(val.denominator))
                + 1
            )
            return Context(prec=precision or 1).divide(
                Decimal(val.numerator), val.denominator
            )
        strat = fractions(min_value, max_value).map(fraction_to_decimal)
    special: list[Decimal] = []
    if allow_nan or (allow_nan is None and (None in (min_value, max_value))):
        special.extend(map(Decimal, ("NaN", "-NaN", "sNaN", "-sNaN")))
    if allow_infinity or (allow_infinity is None and max_value is None):
        special.append(Decimal("Infinity"))
    if allow_infinity or (allow_infinity is None and min_value is None):
        special.append(Decimal("-Infinity"))
    return strat | (sampled_from(special) if special else nothing())


@defines_strategy(never_lazy=True)
def recursive(
    base: SearchStrategy[Ex],
    extend: Callable[[SearchStrategy[Any]], SearchStrategy[T]],
    *,
    max_leaves: int = 100,
) -> SearchStrategy[Union[T, Ex]]:
    return RecursiveStrategy(base, extend, max_leaves)


class PermutationStrategy(SearchStrategy):
    def __init__(self, values: Sequence[T]) -> None:
        self.values = values

    def do_draw(self, data: Any) -> list[T]:
        result = list(self.values)
        for i in range(len(result) - 1):
            j = data.draw_integer(i, len(result) - 1)
            result[i], result[j] = result[j], result[i]
        return result


@defines_strategy()
def permutations(values: Sequence[T]) -> SearchStrategy[list[T]]:
    values = check_sample(values, "permutations")
    if not values:
        return builds(list)
    return PermutationStrategy(values)


class CompositeStrategy(SearchStrategy):
    def __init__(self, definition: Callable, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        self.definition = definition
        self.args = args
        self.kwargs = kwargs

    def do_draw(self, data: Any) -> Any:
        return self.definition(data.draw, *self.args, **self.kwargs)

    def calc_label(self) -> Any:
        return calc_label_from_cls(self.definition)

    def __repr__(self) -> str:
        bits = [get_pretty_function_description(self.definition)]
        bits.extend(map(repr, self.args))
        bits.extend(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"builds({', '.join(bits)})"


class DrawFn(Protocol):
    def __init__(self) -> None:
        raise TypeError("Protocols cannot be instantiated")  # pragma: no cover

    __signature__: Signature = Signature(parameters=[])

    def __call__(self, strategy: SearchStrategy[Ex], label: object = None) -> Ex:
        raise NotImplementedError


def _composite(f: Callable[..., Any]) -> Callable[..., SearchStrategy[Any]]:
    if isinstance(f, (classmethod, staticmethod)):
        special_method = type(f)
        f = f.__func__
    else:
        special_method = None

    sig = get_signature(f)
    params = tuple(sig.parameters.values())

    if not (params and "POSITIONAL" in params[0].kind.name):
        raise InvalidArgument(
            "Functions wrapped with composite must take at least one "
            "positional argument."
        )
    if params[0].default is not sig.empty:
        raise InvalidArgument("A default value for initial argument will never be used")
    if not (f is typing._overload_dummy or is_first_param_referenced_in_function(f)):
        note_deprecation(
            "There is no reason to use @st.composite on a function which "
            "does not call the provided draw() function internally.",
            since="2022-07-17",
            has_codemod=False,
        )
    if get_origin(sig.return_annotation) is SearchStrategy:
        ret_repr = repr(sig.return_annotation).replace("hypothesis.strategies.", "st.")
        warnings.warn(
            f"Return-type annotation is `{ret_repr}`, but the decorated "
            "function should return a value (not a strategy)",
            HypothesisWarning,
            stacklevel=3 if sys.version_info[:2] > (3, 9) else 5,
        )
    if params[0].kind.name != "VAR_POSITIONAL":
        params = params[1:]
    newsig = sig.replace(
        parameters=params,
        return_annotation=(
            SearchStrategy
            if sig.return_annotation is sig.empty
            else SearchStrategy[sig.return_annotation]
        ),
    )

    @defines_strategy()
    @define_function_signature(f.__name__, f.__doc__, newsig)
    def accept(*args: Any, **kwargs: Any) -> SearchStrategy[Any]:
        return CompositeStrategy(f, args, kwargs)

    accept.__module__ = f.__module__
    accept.__signature__ = newsig
    if special_method is not None:
        return special_method(accept)
    return accept


if typing.TYPE_CHECKING or ParamSpec is not None:
    P = ParamSpec("P")

    def composite(
        f: Callable[Concatenate[DrawFn, P], Ex],
    ) -> Callable[P, SearchStrategy[Ex]]:
        return _composite(f)

else:  # pragma: no cover
    @cacheable
    def composite(f: Callable[..., Ex]) -> Callable[..., SearchStrategy[Ex]]:
        return _composite(f)


@cacheable
@defines_strategy(force_reusable_values=True)
def complex_numbers(
    *,
    min_magnitude: Real = 0,
    max_magnitude: Optional[Real] = None,
    allow_infinity: Optional[bool] = None,
    allow_nan: Optional[bool] = None,
    allow_subnormal: bool = True,
    width: Literal[32, 64, 128] = 128,
) -> SearchStrategy[complex]:
    check_valid_magnitude(min_magnitude, "min_magnitude")
    check_valid_magnitude(max_magnitude, "max_magnitude")
    check_valid_interval(min_magnitude, max_magnitude, "min_magnitude", "max_magnitude")
    if max_magnitude == math.inf:
        max_magnitude = None

    if allow_infinity is None:
        allow_infinity = bool(max_magnitude is None)
    elif allow_infinity and max_magnitude is not None:
        raise InvalidArgument(f"Cannot have {allow_infinity=} with {max_magnitude=}")
    if allow_nan is None:
        allow_nan = bool(min_magnitude == 0 and max_magnitude is None)
    elif allow_nan and not (min_magnitude == 0 and max_magnitude is None):
        raise InvalidArgument(
            f"Cannot have {allow_nan=}, {min_magnitude=}, {max_magnitude=}"
        )
    check_type(bool, allow_subnormal, "allow_subnormal")
    if width not in (32, 64, 128):
        raise InvalidArgument(
            f"{width=}, but must be 32, 64 or 128 (other complex dtypes "
            "such as complex192 or complex256 are not supported)"
        )
    component_width = width // 2
    allow_kw = {
        "allow_nan": allow_nan,
        "allow_infinity": allow_infinity,
        "allow_subnormal": None if allow_subnormal else allow_subnormal,
        "width": component_width,
    }

    if min_magnitude == 0 and max_magnitude is None:
        return builds(complex, floats(**allow_kw), floats(**allow_kw))  # type: ignore

    @composite
    def constrained_complex(draw: DrawFn) -> complex:
        if max_magnitude is None:
            zi = draw(floats(**allow_kw))
            rmax: Optional[float] = None
        else:
            zi = draw(
                floats(
                    -float_of(max_magnitude, component_width),
                    float_of(max_magnitude, component_width),
                    **allow_kw,
                )
            )
            rmax = float_of(cathetus(max_magnitude, zi), component_width)
        if min_magnitude == 0 or math.fabs(zi) >= min_magnitude:
            zr = draw(floats(None if rmax is None else -rmax, rmax, **allow_kw))
        else:
            rmin = float_of(cathetus(min_magnitude, zi), component_width)
            zr = draw(floats(rmin, rmax, **allow_kw))
        if min_magnitude > 0 and draw(booleans()) and math.fabs(zi) <= min_magnitude:
            zr = -zr
        return complex(zr, zi)

    return constrained_complex()


@defines_strategy(never_lazy=True)
def shared(
    base: SearchStrategy[Ex],
    *,
    key: Optional[Hashable] = None,
) -> SearchStrategy[Ex]:
    return SharedStrategy(base, key)


@composite
def _maybe_nil_uuids(draw: DrawFn, uuid: UUID) -> UUID:
    if draw(data()).conjecture_data.draw_boolean(1 / 64):
        return UUID("00000000-0000-0000-0000-000000000000")
    return uuid


@cacheable
@defines_strategy(force_reusable_values=True)
def uuids(
    *, version: Optional[Literal[1, 2, 3, 4, 5]] = None, allow_nil: bool = False
) -> SearchStrategy[UUID]:
    check_type(bool, allow_nil, "allow_nil")
    if version not in (None, 1, 2, 3, 4, 5):
        raise InvalidArgument(
            f"{version=}, but version must be in "
            "(None, 1, 2, 3, 4, 5) to pass to the uuid.UUID constructor."
        )
    random_uuids: SearchStrategy[UUID] = shared(
        randoms(use_true_random=True), key="hypothesis.strategies.uuids.generator"
    ).map(lambda r: UUID(version=version, int=r.getrandbits(128)))

    if allow_nil:
        if version is not None:
            raise InvalidArgument("The nil UUID is not of any version")
        return random_uuids.flatmap(_maybe_nil_uuids)
    return random_uuids


class RunnerStrategy(SearchStrategy):
    def __init__(self, default: Any) -> None:
        self.default = default

    def do_draw(self, data: Any) -> Any:
        runner = getattr(data, "hypothesis_runner", not_set)
        if runner is not_set:
            if self.default is not_set:
                raise InvalidArgument(
                    "Cannot use runner() strategy with no "
                    "associated runner or explicit default."
                )
            else:
                return self.default
        else:
            return runner


@defines_strategy(force_reusable_values=True)
def runner(*, default: Any = not_set) -> SearchStrategy[Any]:
    return RunnerStrategy(default)


class DataObject:
    def __init__(self, data: Any) -> None:
        self.count: int = 0
        self.conjecture_data: Any = data

    __signature__ = Signature()

    def __repr__(self) -> str:
        return "data(...)"

    def draw(self, strategy: SearchStrategy[Ex], label: Any = None) -> Ex:
        check_strategy(strategy, "strategy")
        self.count += 1
        desc = f"Draw {self.count}{'' if label is None else f' ({label})'}"
        with deprecate_random_in_strategy("{}from {!r}", desc, strategy):
            result = self.conjecture_data.draw(strategy, observe_as=f"generate:{desc}")

        if should_note():
            printer = RepresentationPrinter(context=current_build_context())
            printer.text(f"{desc}: ")
            if self.conjecture_data.provider.avoid_realization:
                printer.text("<symbolic>")
            else:
                printer.pretty(result)
            note(printer.getvalue())
        return result


class DataStrategy(SearchStrategy):
    supports_find = False

    def do_draw(self, data: Any) -> DataObject:
        if not hasattr(data, "hypothesis_shared_data_strategy"):
            data.hypothesis_shared_data_strategy = DataObject(data)
        return data.hypothesis_shared_data_strategy

    def __repr__(self) -> str:
        return "data()"

    def map(self, f: Callable[[Any], Any]) -> Any:
        self.__not_a_first_class_strategy("map")

    def filter(self, f: Callable[[Any], bool]) -> Any:
        self.__not_a_first_class_strategy("filter")

    def flatmap(self, f: Callable[[Any], Any]) -> Any:
        self.__not_a_first_class_strategy("flatmap")

    def example(self) -> Any:
        self.__not_a_first_class_strategy("example")

    def __not_a_first_class_strategy(self, name: str) -> None:
        raise InvalidArgument(
            f"Cannot call {name} on a DataStrategy. You should probably "
            "be using @composite for whatever it is you're trying to do."
        )


@cacheable
@defines_strategy(never_lazy=True)
def data() -> SearchStrategy[DataObject]:
    return DataStrategy()


def register_type_strategy(
    custom_type: type[Ex],
    strategy: Union[SearchStrategy[Ex], Callable[[type[Ex]], SearchStrategy[Ex]]],
) -> None:
    from hypothesis.strategies._internal import types

    if not types.is_a_type(custom_type):
        raise InvalidArgument(f"{custom_type=} must be a type")
    if custom_type in types.NON_RUNTIME_TYPES:
        raise InvalidArgument(
            f"{custom_type=} is not allowed to be registered, "
            f"because there is no such thing as a runtime instance of {custom_type!r}"
        )
    if not (isinstance(strategy, SearchStrategy) or callable(strategy)):
        raise InvalidArgument(
            f"{strategy=} must be a SearchStrategy, or a function that takes "
            "a generic type and returns a specific SearchStrategy"
        )
    if isinstance(strategy, SearchStrategy):
        with warnings.catch_warnings():
            warnings.simplefilter("error", HypothesisSideeffectWarning)
            try:
                if strategy.is_empty:
                    raise InvalidArgument(f"{strategy=} must not be empty")
            except HypothesisSideeffectWarning:  # pragma: no cover
                pass
    if types.has_type_arguments(custom_type):
        raise InvalidArgument(
            f"Cannot register generic type {custom_type!r}, because it has type "
            "arguments which would not be handled.  Instead, register a function "
            f"for {get_origin(custom_type)!r} which can inspect specific type "
            "objects and return a strategy."
        )
    if (
        "pydantic.generics" in sys.modules
        and issubclass(custom_type, sys.modules["pydantic.generics"].GenericModel)
        and not re.search(r"[A-Za-z_]+\[.+\]", repr(custom_type))
        and callable(strategy)
    ):  # pragma: no cover
        raise InvalidArgument(
            f"Cannot register a function for {custom_type!r}, because parametrized "
            "`pydantic.generics.GenericModel` subclasses aren't actually generic "
            "types at runtime.  In this case, you should register a strategy "
            "directly for each parametrized form that you anticipate using."
        )

    types._global_type_lookup[custom_type] = strategy
    from_type.__clear_cache()  # type: ignore


@cacheable
@defines_strategy(never_lazy=True)
def deferred(definition: Callable[[], SearchStrategy[Ex]]) -> SearchStrategy[Ex]:
    return DeferredStrategy(definition)


def domains() -> Any:
    import hypothesis.provisional
    return hypothesis.provisional.domains()


@defines_strategy(force_reusable_values=True)
def emails(
    *, domains: SearchStrategy[str] = LazyStrategy(domains, (), {})
) -> SearchStrategy[str]:
    local_chars = string.ascii_letters + string.digits + "!#$%&'*+-/=^_`{|}~"
    local_part = text(local_chars, min_size=1, max_size=64)
    return builds("{}@{}".format, local_part, domains).filter(
        lambda addr: len(addr) <= 254
    )


def _functions(*, like: Callable[..., Any], returns: Optional[SearchStrategy[Any]], pure: bool) -> SearchStrategy[Callable[..., Any]]:
    check_type(bool, pure, "pure")
    if not callable(like):
        raise InvalidArgument(
            "The first argument to functions() must be a callable to imitate, "
            f"but got non-callable like={nicerepr(like)!r}"
        )
    if returns in (None, ...):
        hints = get_type_hints(like)
        returns = from_type(hints.get("return", type(None)))
    check_strategy(returns, "returns")
    return FunctionStrategy(like, returns, pure)


if typing.TYPE_CHECKING or ParamSpec is not None:

    @overload
    def functions(
        *, pure: bool = ...
    ) -> SearchStrategy[Callable[[], None]]:  # pragma: no cover
        ...

    @overload
    def functions(
        *, like: Callable[P, T],
        pure: bool = ...,
    ) -> SearchStrategy[Callable[P, T]]:  # pragma: no cover
        ...

    @overload
    def functions(
        *, returns: SearchStrategy[T],
        pure: bool = ...,
    ) -> SearchStrategy[Callable[[], T]]:  # pragma: no cover
        ...

    @overload
    def functions(
        *, like: Callable[P, Any],
        returns: SearchStrategy[T],
        pure: bool = ...,
    ) -> SearchStrategy[Callable[P, T]]:  # pragma: no cover
        ...

    @defines_strategy()
    def functions(*, like: Callable[..., Any] = lambda: None, returns: Optional[SearchStrategy[Any]] = ..., pure: bool = False) -> SearchStrategy[Callable[..., Any]]:
        return _functions(like=like, returns=returns, pure=pure)

else:  # pragma: no cover

    @defines_strategy()
    def functions(
        *,
        like: Callable[..., Any] = lambda: None,
        returns: Union[SearchStrategy[Any], EllipsisType] = ...,
        pure: bool = False,
    ) -> SearchStrategy[Callable[..., Any]]:
        return _functions(like=like, returns=returns, pure=pure)


@composite
def slices(draw: Any, size: int) -> slice:
    check_valid_size(size, "size")
    if size == 0:
        step = draw(none() | integers().filter(bool))
        return slice(None, None, step)
    start = draw(integers(0, size - 1) | none())
    stop = draw(integers(0, size) | none())
    if start is None and stop is None:
        max_step = size
    elif start is None:
        max_step = stop  # type: ignore
    elif stop is None:
        max_step = start
    else:
        max_step = abs(start - stop)
    step = draw(integers(1, max_step or 1))
    if (draw(booleans()) and start == stop) or (stop or 0) < (start or 0):
        step *= -1
    if draw(booleans()) and start is not None:
        start -= size
    if draw(booleans()) and stop is not None:
        stop -= size
    if (not draw(booleans())) and step == 1:
        step = None
    return slice(start, stop, step)