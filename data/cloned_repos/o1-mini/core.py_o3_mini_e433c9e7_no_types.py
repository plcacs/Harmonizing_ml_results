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
from typing import Annotated, Any, AnyStr, Callable, Literal, Optional, Protocol, TypeVar, Union, cast, get_args, get_origin, overload
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
def booleans():
    """Returns a strategy which generates instances of :class:`python:bool`.

    Examples from this strategy will shrink towards ``False`` (i.e.
    shrinking will replace ``True`` with ``False`` where possible).
    """
    return BooleansStrategy()


@overload
def sampled_from(elements):
    ...


@overload
def sampled_from(elements):
    ...


@overload
def sampled_from(elements):
    ...


@defines_strategy(try_non_lazy=True)
def sampled_from(elements):
    """Returns a strategy which generates any value present in ``elements``.

    Note that as with :func:`~hypothesis.strategies.just`, values will not be
    copied and thus you should be careful of using mutable data.

    ``sampled_from`` supports ordered collections, as well as
    :class:`~python:enum.Enum` objects.  :class:`~python:enum.Flag` objects
    may also generate any combination of their members.

    Examples from this strategy shrink by replacing them with values earlier in
    the list. So e.g. ``sampled_from([10, 1])`` will shrink by trying to replace
    1 values with 10, and ``sampled_from([1, 10])`` will shrink by trying to
    replace 10 values with 1.

    It is an error to sample from an empty sequence, because returning :func:`nothing`
    makes it too easy to silently drop parts of compound strategies.  If you need
    that behaviour, use ``sampled_from(seq) if seq else nothing()``.
    """
    values = check_sample(elements, 'sampled_from')
    try:
        if isinstance(elements, type) and issubclass(elements, enum.Enum):
            repr_ = f'sampled_from({elements.__module__}.{elements.__name__})'
        else:
            repr_ = f'sampled_from({elements!r})'
    except Exception:
        repr_ = None
    if isclass(elements) and issubclass(elements, enum.Flag):
        flags = sorted(set(elements.__members__.values()), key=lambda v: (
            bit_count(v.value), v.value))
        flags_with_empty = flags
        if not flags or flags[0].value != 0:
            try:
                flags_with_empty = [*flags, elements(0)]
            except TypeError:
                pass
        inner: list[SearchStrategy[Any]] = [sampled_from(flags_with_empty)]
        if len(flags) > 1:
            inner += [integers(min_value=1, max_value=len(flags)).flatmap(
                lambda r: sets(sampled_from(flags), min_size=r, max_size=r)
                ).map(lambda s: elements(reduce(operator.or_, s)))]
        return LazyStrategy(one_of, args=inner, kwargs={}, force_repr=repr_)
    if not values:
        if isinstance(elements, type) and issubclass(elements, enum.Enum
            ) and vars(elements).get('__annotations__'):
            raise InvalidArgument(
                f'Cannot sample from {elements.__module__}.{elements.__name__} because it contains no elements.  It does however have annotations, so maybe you tried to write an enum as if it was a dataclass?'
                )
        raise InvalidArgument('Cannot sample from a length-zero sequence.')
    if len(values) == 1:
        return just(values[0])
    return SampledFromStrategy(values, repr_)


@cacheable
@defines_strategy()
def lists(elements, *, min_size: int=0, max_size: Optional[int]=None,
    unique_by: Union[None, Callable[[Ex], Hashable], tuple[Callable[[Ex],
    Hashable], ...]]=None, unique: bool=False):
    """Returns a list containing values drawn from elements with length in the
    interval [min_size, max_size] (no bounds in that direction if these are
    None). If max_size is 0, only the empty list will be drawn.

    If ``unique`` is True (or something that evaluates to True), we compare direct
    object equality, as if unique_by was ``lambda x: x``. This comparison only
    works for hashable types.

    If ``unique_by`` is not None it must be a callable or tuple of callables
    returning a hashable type when given a value drawn from elements. The
    resulting list will satisfy the condition that for ``i`` != ``j``,
    ``unique_by(result[i])`` != ``unique_by(result[j])``.

    If ``unique_by`` is a tuple of callables the uniqueness will be respective
    to each callable.

    For example, the following will produce two columns of integers with both
    columns being unique respectively.

    .. code-block:: pycon

        >>> twoints = st.tuples(st.integers(), st.integers())
        >>> st.lists(twoints, unique_by=(lambda x: x[0], lambda x: x[1]))

    Examples from this strategy shrink by trying to remove elements from the
    list, and by shrinking each individual element of the list.
    """
    check_valid_sizes(min_size, max_size)
    check_strategy(elements, 'elements')
    if unique:
        if unique_by is not None:
            raise InvalidArgument(
                'cannot specify both unique and unique_by (you probably only want to set unique_by)'
                )
        else:
            unique_by = identity
    if max_size == 0:
        return builds(list)
    if unique_by is not None:
        if not (callable(unique_by) or isinstance(unique_by, tuple)):
            raise InvalidArgument(
                f'unique_by={unique_by!r} is not a callable or tuple of callables'
                )
        if callable(unique_by):
            unique_by = unique_by,
        if len(unique_by) == 0:
            raise InvalidArgument('unique_by is empty')
        for i, f in enumerate(unique_by):
            if not callable(f):
                raise InvalidArgument(f'unique_by[{i}]={f!r} is not a callable'
                    )
        tuple_suffixes: Optional[TupleStrategy] = None
        if isinstance(elements, TupleStrategy) and len(elements.
            element_strategies) >= 1 and len(unique_by) == 1 and (
            isinstance(unique_by[0], operator.itemgetter) and repr(
            unique_by[0]) == 'operator.itemgetter(0)' or isinstance(
            unique_by[0], FunctionType) and re.fullmatch(
            get_pretty_function_description(unique_by[0]),
            'lambda ([a-z]+): \\1\\[0\\]')):
            unique_by = identity,
            tuple_suffixes = TupleStrategy(elements.element_strategies[1:])
            elements = elements.element_strategies[0]
        if (isinstance(elements, IntegersStrategy) and elements.start is not
            None and elements.end is not None and elements.end - elements.
            start <= 255):
            elements = SampledFromStrategy(sorted(range(elements.start, 
                elements.end + 1), key=abs) if elements.end < 0 or elements
                .start > 0 else list(range(elements.end + 1)) + list(range(
                -1, elements.start - 1, -1)))
        if isinstance(elements, SampledFromStrategy):
            element_count = len(elements.elements)
            if min_size > element_count:
                raise InvalidArgument(
                    f'Cannot create a collection of min_size={min_size!r} unique elements with values drawn from only {element_count} distinct elements'
                    )
            if max_size is not None:
                max_size = min(max_size, element_count)
            else:
                max_size = element_count
            return UniqueSampledListStrategy(elements=elements, max_size=
                max_size, min_size=min_size, keys=unique_by, tuple_suffixes
                =tuple_suffixes)
        return UniqueListStrategy(elements=elements, max_size=max_size,
            min_size=min_size, keys=unique_by, tuple_suffixes=tuple_suffixes)
    return ListStrategy(elements, min_size=min_size, max_size=max_size)


@cacheable
@defines_strategy()
def sets(elements, *, min_size: int=0, max_size: Optional[int]=None):
    """This has the same behaviour as lists, but returns sets instead.

    Note that Hypothesis cannot tell if values are drawn from elements
    are hashable until running the test, so you can define a strategy
    for sets of an unhashable type but it will fail at test time.

    Examples from this strategy shrink by trying to remove elements from the
    set, and by shrinking each individual element of the set.
    """
    return lists(elements=elements, min_size=min_size, max_size=max_size,
        unique=True).map(set)


@cacheable
@defines_strategy()
def frozensets(elements, *, min_size: int=0, max_size: Optional[int]=None):
    """This is identical to the sets function but instead returns
    frozensets."""
    return lists(elements=elements, min_size=min_size, max_size=max_size,
        unique=True).map(frozenset)


class PrettyIter:

    def __init__(self, values):
        self._values = values
        self._iter = iter(self._values)

    def __iter__(self):
        return self._iter

    def __next__(self):
        return next(self._iter)

    def __repr__(self):
        return f'iter({self._values!r})'


@defines_strategy()
def iterables(elements, *, min_size: int=0, max_size: Optional[int]=None,
    unique_by: Union[None, Callable[[Ex], Hashable], tuple[Callable[[Ex],
    Hashable], ...]]=None, unique: bool=False):
    """This has the same behaviour as lists, but returns iterables instead.

    Some iterables cannot be indexed (e.g. sets) and some do not have a
    fixed length (e.g. generators). This strategy produces iterators,
    which cannot be indexed and do not have a fixed length. This ensures
    that you do not accidentally depend on sequence behaviour.
    """
    return lists(elements=elements, min_size=min_size, max_size=max_size,
        unique_by=unique_by, unique=unique).map(PrettyIter)


@defines_strategy()
def fixed_dictionaries(mapping, *, optional: Optional[dict[T,
    SearchStrategy[Ex]]]=None):
    """Generates a dictionary of the same type as mapping with a fixed set of
    keys mapping to strategies. ``mapping`` must be a dict subclass.

    Generated values have all keys present in mapping, in iteration order,
    with the corresponding values drawn from mapping[key].

    If ``optional`` is passed, the generated value *may or may not* contain each
    key from ``optional`` and a value drawn from the corresponding strategy.
    Generated values may contain optional keys in an arbitrary order.

    Examples from this strategy shrink by shrinking each individual value in
    the generated dictionary, and omitting optional key-value pairs.
    """
    check_type(dict, mapping, 'mapping')
    for k, v in mapping.items():
        check_strategy(v, f'mapping[{k!r}]')
    if optional is not None:
        check_type(dict, optional, 'optional')
        for k, v in optional.items():
            check_strategy(v, f'optional[{k!r}]')
        if type(mapping) != type(optional):
            raise InvalidArgument(
                'Got arguments of different types: mapping=%s, optional=%s' %
                (nicerepr(type(mapping)), nicerepr(type(optional))))
        if set(mapping) & set(optional):
            raise InvalidArgument(
                f'The following keys were in both mapping and optional, which is invalid: {set(mapping) & set(optional)!r}'
                )
        return FixedAndOptionalKeysDictStrategy(mapping, optional)
    return FixedKeysDictStrategy(mapping)


@cacheable
@defines_strategy()
def dictionaries(keys, values, *, dict_class: type=dict, min_size: int=0,
    max_size: Optional[int]=None):
    """Generates dictionaries of type ``dict_class`` with keys drawn from the ``keys``
    argument and values drawn from the ``values`` argument.

    The size parameters have the same interpretation as for
    :func:`~hypothesis.strategies.lists`.

    Examples from this strategy shrink by trying to remove keys from the
    generated dictionary, and by shrinking each generated key and value.
    """
    check_valid_sizes(min_size, max_size)
    if max_size == 0:
        return fixed_dictionaries(dict_class())
    check_strategy(keys, 'keys')
    check_strategy(values, 'values')
    return lists(tuples(keys, values), min_size=min_size, max_size=max_size,
        unique_by=operator.itemgetter(0)).map(dict_class)


@cacheable
@defines_strategy(force_reusable_values=True)
def characters(*, codec: Optional[str]=None, min_codepoint: Optional[int]=
    None, max_codepoint: Optional[int]=None, categories: Optional[
    Collection[CategoryName]]=None, exclude_categories: Optional[Collection
    [CategoryName]]=None, exclude_characters: Optional[Collection[str]]=
    None, include_characters: Optional[Collection[str]]=None,
    blacklist_categories: Optional[Collection[CategoryName]]=None,
    whitelist_categories: Optional[Collection[CategoryName]]=None,
    blacklist_characters: Optional[Collection[str]]=None,
    whitelist_characters: Optional[Collection[str]]=None):
    """Generates characters, length-one :class:`python:str`\\ ings,
    following specified filtering rules.

    - When no filtering rules are specified, any character can be produced.
    - If ``min_codepoint`` or ``max_codepoint`` is specified, then only
      characters having a codepoint in that range will be produced.
    - If ``categories`` is specified, then only characters from those
      Unicode categories will be produced. This is a further restriction,
      characters must also satisfy ``min_codepoint`` and ``max_codepoint``.
    - If ``exclude_categories`` is specified, then any character from those
      categories will not be produced.  You must not pass both ``categories``
      and ``exclude_categories``; these arguments are alternative ways to
      specify exactly the same thing.
    - If ``include_characters`` is specified, then any additional characters
      in that list will also be produced.
    - If ``exclude_characters`` is specified, then any characters in
      that list will be not be produced. Any overlap between
      ``include_characters`` and ``exclude_characters`` will raise an
      exception.
    - If ``codec`` is specified, only characters in the specified `codec encodings`_
      will be produced.

    The ``_codepoint`` arguments must be integers between zero and
    :obj:`python:sys.maxunicode`.  The ``_characters`` arguments must be
    collections of length-one unicode strings, such as a unicode string.

    The ``_categories`` arguments must be used to specify either the
    one-letter Unicode major category or the two-letter Unicode
    `general category`_.  For example, ``('Nd', 'Lu')`` signifies "Number,
    decimal digit" and "Letter, uppercase".  A single letter ('major category')
    can be given to match all corresponding categories, for example ``'P'``
    for characters in any punctuation category.

    We allow codecs from the :mod:`codecs` module and their aliases, platform
    specific and user-registered codecs if they are available, and
    `python-specific text encodings`_ (but not text or binary transforms).
    ``include_characters`` which cannot be encoded using this codec will
    raise an exception.  If non-encodable codepoints or categories are
    explicitly allowed, the ``codec`` argument will exclude them without
    raising an exception.

    .. _general category: https://en.wikipedia.org/wiki/Unicode_character_property
    .. _codec encodings: https://docs.python.org/3/library/codecs.html#encodings-and-unicode
    .. _python-specific text encodings: https://docs.python.org/3/library/codecs.html#python-specific-encodings

    Examples from this strategy shrink towards the codepoint for ``'0'``,
    or the first allowable codepoint after it if ``'0'`` is excluded.
    """
    check_valid_size(min_codepoint, 'min_codepoint')
    check_valid_size(max_codepoint, 'max_codepoint')
    check_valid_interval(min_codepoint, max_codepoint, 'min_codepoint',
        'max_codepoint')
    categories = cast(Optional[Categories], categories)
    if categories is not None and exclude_categories is not None:
        raise InvalidArgument(
            f"Pass at most one of categories={categories!r} and exclude_categories={exclude_categories!r} - these arguments both specify which categories are allowed, so it doesn't make sense to use both in a single call."
            )
    has_old_arg = any(v is not None for k, v in locals().items() if 'list' in k
        )
    has_new_arg = any(v is not None for k, v in locals().items() if 'lude' in k
        )
    if has_old_arg and has_new_arg:
        raise InvalidArgument(
            'The deprecated blacklist/whitelist arguments cannot be used in the same call as their replacement include/exclude arguments.'
            )
    if blacklist_categories is not None:
        exclude_categories = blacklist_categories
    if whitelist_categories is not None:
        categories = whitelist_categories
    if blacklist_characters is not None:
        exclude_characters = blacklist_characters
    if whitelist_characters is not None:
        include_characters = whitelist_characters
    if (min_codepoint is None and max_codepoint is None and categories is
        None and exclude_categories is None and include_characters is not
        None and codec is None):
        raise InvalidArgument(
            f'Nothing is excluded by other arguments, so passing only include_characters={include_characters!r} would have no effect.  Also pass categories=(), or use sampled_from({include_characters!r}) instead.'
            )
    exclude_characters = exclude_characters or ''
    include_characters = include_characters or ''
    overlap = set(exclude_characters).intersection(include_characters)
    if overlap:
        raise InvalidArgument(
            f'Characters {sorted(overlap)!r} are present in both include_characters={include_characters!r} and exclude_characters={exclude_characters!r}'
            )
    if categories is not None:
        categories = as_general_categories(categories, 'categories')
    if exclude_categories is not None:
        exclude_categories = as_general_categories(exclude_categories,
            'exclude_categories')
    if categories is not None and not categories and not include_characters:
        raise InvalidArgument(
            'When `categories` is an empty collection and there are no characters specified in include_characters, nothing can be generated by the characters() strategy.'
            )
    both_cats = set(exclude_categories or ()).intersection(categories or ())
    if both_cats:
        raise InvalidArgument(
            f'Categories {sorted(both_cats)!r} are present in both categories={categories!r} and exclude_categories={exclude_categories!r}'
            )
    elif exclude_categories is not None:
        categories = set(all_categories()) - set(exclude_categories)
    del exclude_categories
    if codec is not None:
        try:
            codec = codecs.lookup(codec).name
            """""".encode(codec)
        except LookupError:
            raise InvalidArgument(
                f'codec={codec!r} is not valid on this system') from None
        except Exception:
            raise InvalidArgument(f'codec={codec!r} is not a valid codec'
                ) from None
        for char in include_characters:
            try:
                char.encode(encoding=codec, errors='strict')
            except UnicodeEncodeError:
                raise InvalidArgument(
                    f'Character {char!r} in include_characters={include_characters!r} cannot be encoded with codec={codec!r}'
                    ) from None
        if codec == 'ascii':
            if max_codepoint is None or max_codepoint > 127:
                max_codepoint = 127
            codec = None
        elif codec == 'utf-8':
            if categories is None:
                categories = all_categories()
            categories = tuple(c for c in categories if c != 'Cs')
    return OneCharStringStrategy.from_characters_args(categories=categories,
        exclude_characters=exclude_characters, min_codepoint=min_codepoint,
        max_codepoint=max_codepoint, include_characters=include_characters,
        codec=codec)


characters.__signature__ = (__sig := get_signature(characters)).replace(
    parameters=[p for p in __sig.parameters.values() if 'list' not in p.name])


@cacheable
@defines_strategy(force_reusable_values=True)
def text(alphabet=characters(codec='utf-8'), *, min_size: int=0, max_size:
    Optional[int]=None):
    """Generates strings with characters drawn from ``alphabet``, which should
    be a collection of length one strings or a strategy generating such strings.

    The default alphabet strategy can generate the full unicode range but
    excludes surrogate characters because they are invalid in the UTF-8
    encoding.  You can use :func:`~hypothesis.strategies.characters` without
    arguments to find surrogate-related bugs such as :bpo:`34454`.

    ``min_size`` and ``max_size`` have the usual interpretations.
    Note that Python measures string length by counting codepoints: U+00C5
    ``Å`` is a single character, while U+0041 U+030A ``Å`` is two - the ``A``,
    and a combining ring above.

    Examples from this strategy shrink towards shorter strings, and with the
    characters in the text shrinking as per the alphabet strategy.
    This strategy does not :func:`~python:unicodedata.normalize` examples,
    so generated strings may be in any or none of the 'normal forms'.
    """
    check_valid_sizes(min_size, max_size)
    if isinstance(alphabet, SearchStrategy):
        char_strategy = unwrap_strategies(alphabet)
        if isinstance(char_strategy, SampledFromStrategy):
            return text(char_strategy.elements, min_size=min_size, max_size
                =max_size)
        elif not isinstance(char_strategy, OneCharStringStrategy):
            char_strategy = char_strategy.map(_check_is_single_character)
    else:
        non_string = [c for c in alphabet if not isinstance(c, str)]
        if non_string:
            raise InvalidArgument(
                f'The following elements in alphabet are not unicode strings:  {non_string!r}'
                )
        not_one_char = [c for c in alphabet if len(c) != 1]
        if not_one_char:
            raise InvalidArgument(
                f'The following elements in alphabet are not of length one, which leads to violation of size constraints:  {not_one_char!r}'
                )
        if alphabet in ['ascii', 'utf-8']:
            warnings.warn(
                f'st.text({alphabet!r}): it seems like you are trying to use the codec {alphabet!r}. st.text({alphabet!r}) instead generates strings using the literal characters {list(alphabet)!r}. To specify the {alphabet} codec, use st.text(st.characters(codec={alphabet!r})). If you intended to use character literals, you can silence this warning by reordering the characters.'
                , HypothesisWarning, stacklevel=1)
        char_strategy = characters(categories=(), include_characters=alphabet
            ) if alphabet else nothing()
    if (max_size == 0 or char_strategy.is_empty) and not min_size:
        return just('')
    return TextStrategy(char_strategy, min_size=min_size, max_size=max_size)


@overload
def from_regex(regex, *, fullmatch: bool=False):
    ...


@overload
def from_regex(regex, *, fullmatch: bool=False, alphabet: Union[str,
    SearchStrategy[str]]=characters(codec='utf-8')):
    ...


@cacheable
@defines_strategy()
def from_regex(regex, *, fullmatch: bool=False, alphabet: Union[str,
    SearchStrategy[str], None]=None):
    """Generates strings that contain a match for the given regex (i.e. ones
    for which :func:`python:re.search` will return a non-None result).

    ``regex`` may be a pattern or :func:`compiled regex <python:re.compile>`.
    Both byte-strings and unicode strings are supported, and will generate
    examples of the same type.

    You can use regex flags such as :obj:`python:re.IGNORECASE` or
    :obj:`python:re.DOTALL` to control generation. Flags can be passed either
    in compiled regex or inside the pattern with a ``(?iLmsux)`` group.

    Some regular expressions are only partly supported - the underlying
    strategy checks local matching and relies on filtering to resolve
    context-dependent expressions.  Using too many of these constructs may
    cause health-check errors as too many examples are filtered out. This
    mainly includes (positive or negative) lookahead and lookbehind groups.

    If you want the generated string to match the whole regex you should use
    boundary markers. So e.g. ``r"\\A.\\Z"`` will return a single character
    string, while ``"."`` will return any string, and ``r"\\A.$"`` will return
    a single character optionally followed by a ``"\\n"``.
    Alternatively, passing ``fullmatch=True`` will ensure that the whole
    string is a match, as if you had used the ``\\A`` and ``\\Z`` markers.

    The ``alphabet=`` argument constrains the characters in the generated
    string, as for :func:`text`, and is only supported for unicode strings.

    Examples from this strategy shrink towards shorter strings and lower
    character values, with exact behaviour that may depend on the pattern.
    """
    check_type((str, bytes, re.Pattern), regex, 'regex')
    check_type(bool, fullmatch, 'fullmatch')
    pattern = regex.pattern if isinstance(regex, re.Pattern) else regex
    if alphabet is not None:
        check_type((str, SearchStrategy), alphabet, 'alphabet')
        if not isinstance(pattern, str):
            raise InvalidArgument('alphabet= is not supported for bytestrings')
        alphabet = OneCharStringStrategy.from_alphabet(alphabet)
    elif isinstance(pattern, str):
        alphabet = characters(codec='utf-8')
    from hypothesis.strategies._internal.regex import regex_strategy
    return regex_strategy(regex, fullmatch, alphabet=alphabet)


@cacheable
@defines_strategy(force_reusable_values=True)
def binary(*, min_size: int=0, max_size: Optional[int]=None):
    """Generates :class:`python:bytes`.

    The generated :class:`python:bytes` will have a length of at least ``min_size``
    and at most ``max_size``.  If ``max_size`` is None there is no upper limit.

    Examples from this strategy shrink towards smaller strings and lower byte
    values.
    """
    check_valid_sizes(min_size, max_size)
    return BytesStrategy(min_size, max_size)


@cacheable
@defines_strategy()
def randoms(*, note_method_calls: bool=False, use_true_random: bool=False):
    """Generates instances of ``random.Random``. The generated Random instances
    are of a special HypothesisRandom subclass.

    - If ``note_method_calls`` is set to ``True``, Hypothesis will print the
      randomly drawn values in any falsifying test case. This can be helpful
      for debugging the behaviour of randomized algorithms.
    - If ``use_true_random`` is set to ``True`` then values will be drawn from
      their usual distribution, otherwise they will actually be Hypothesis
      generated values (and will be shrunk accordingly for any failing test
      case). Setting ``use_true_random=False`` will tend to expose bugs that
      would occur with very low probability when it is set to True, and this
      flag should only be set to True when your code relies on the distribution
      of values for correctness.

    For managing global state, see the :func:`~hypothesis.strategies.random_module`
    strategy and :func:`~hypothesis.register_random` function.
    """
    check_type(bool, note_method_calls, 'note_method_calls')
    check_type(bool, use_true_random, 'use_true_random')
    from hypothesis.strategies._internal.random import RandomStrategy
    return RandomStrategy(use_true_random=use_true_random,
        note_method_calls=note_method_calls)


class RandomModule(SearchStrategy[RandomSeeder]):

    def __init__(self, default=None):
        self.default = default

    def do_draw(self, data):
        seed = data.draw(integers(0, 2 ** 32 - 1))
        seed_all, restore_all = get_seeder_and_restorer(seed)
        seed_all()
        cleanup(restore_all)
        return RandomSeeder(seed)

    def __repr__(self):
        return 'random_module()'


@cacheable
@defines_strategy()
def random_module():
    """Hypothesis always seeds global PRNGs before running a test, and restores the
    previous state afterwards.

    If having a fixed seed would unacceptably weaken your tests, and you
    cannot use a ``random.Random`` instance provided by
    :func:`~hypothesis.strategies.randoms`, this strategy calls
    :func:`python:random.seed` with an arbitrary integer and passes you
    an opaque object whose repr displays the seed value for debugging.
    If ``numpy.random`` is available, that state is also managed, as is anything
    managed by :func:`hypothesis.register_random`.

    Examples from these strategy shrink to seeds closer to zero.
    """
    return shared(RandomModule(), key='hypothesis.strategies.random_module()')


class BuildsStrategy(SearchStrategy[Ex]):

    def __init__(self, target, args, kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def do_draw(self, data):
        args_drawn = [data.draw(a) for a in self.args]
        kwargs_drawn = {k: data.draw(v) for k, v in self.kwargs.items()}
        try:
            obj = self.target(*args_drawn, **kwargs_drawn)
        except TypeError as err:
            if isinstance(self.target, type) and issubclass(self.target,
                enum.Enum) and not (self.args or self.kwargs):
                name = self.target.__module__ + '.' + self.target.__qualname__
                raise InvalidArgument(
                    f'Calling {name} with no arguments raised an error - try using sampled_from({name}) instead of builds({name})'
                    ) from err
            if not (self.args or self.kwargs):
                from .types import is_a_new_type, is_generic_type
                if is_a_new_type(self.target) or is_generic_type(self.target):
                    raise InvalidArgument(
                        f'Calling {self.target!r} with no arguments raised an error - try using from_type({self.target!r}) instead of builds({self.target!r})'
                        ) from err
            if getattr(self.target, '__no_type_check__', None) is True:
                raise TypeError(
                    'This might be because the @no_type_check decorator prevented Hypothesis from inferring a strategy for some required arguments.'
                    ) from err
            raise
        current_build_context().record_call(obj, self.target, args_drawn,
            kwargs_drawn)
        return obj

    def validate(self):
        tuples(*self.args).validate()
        fixed_dictionaries(self.kwargs).validate()

    def __repr__(self):
        bits = [get_pretty_function_description(self.target)]
        bits.extend(map(repr, self.args))
        bits.extend(f'{k}={v!r}' for k, v in self.kwargs.items())
        return f"builds({', '.join(bits)})"


@cacheable
@defines_strategy()
def builds(target: Callable[..., Ex], /, *args: SearchStrategy[Any], **
    kwargs: Union[SearchStrategy[Any], EllipsisType]):
    """Generates values by drawing from ``args`` and ``kwargs`` and passing
    them to the callable (provided as the first positional argument) in the
    appropriate argument position.

    e.g. ``builds(target, integers(), flag=booleans())`` would draw an
    integer ``i`` and a boolean ``b`` and call ``target(i, flag=b)``.

    If the callable has type annotations, they will be used to infer a strategy
    for required arguments that were not passed to builds.  You can also tell
    builds to infer a strategy for an optional argument by passing ``...``
    (:obj:`python:Ellipsis`) as a keyword argument to builds, instead of a strategy for
    that argument to the callable.

    If the callable is a class defined with :pypi:`attrs`, missing required
    arguments will be inferred from the attribute on a best-effort basis,
    e.g. by checking :ref:`attrs standard validators <attrs:api-validators>`.
    Dataclasses are handled natively by the inference from type hints.

    Examples from this strategy shrink by shrinking the argument values to
    the callable.
    """
    if not callable(target):
        raise InvalidArgument(
            'The first positional argument to builds() must be a callable target to construct.'
            )
    if ... in args:
        raise InvalidArgument(
            '... was passed as a positional argument to builds(), but is only allowed as a keyword arg'
            )
    required = required_args(target, args, kwargs)
    to_infer = {k for k, v in kwargs.items() if v is ...}
    if required or to_infer:
        if isinstance(target, type) and attr.has(target):
            from hypothesis.strategies._internal.attrs import from_attrs
            return from_attrs(target, args, kwargs, required | to_infer)
        hints = get_type_hints(target)
        if to_infer - set(hints):
            badargs = ', '.join(sorted(to_infer - set(hints)))
            raise InvalidArgument(
                f'passed ... for {badargs}, but we cannot infer a strategy because these arguments have no type annotation'
                )
        infer_for = {k: v for k, v in hints.items() if k in required | to_infer
            }
        if infer_for:
            from hypothesis.strategies._internal.types import _global_type_lookup
            for kw, t in infer_for.items():
                if t in _global_type_lookup:
                    kwargs[kw] = from_type(t)
                else:
                    kwargs[kw] = deferred(lambda t=t: from_type(t))
    return BuildsStrategy(target, args, kwargs)


@cacheable
@defines_strategy(never_lazy=True)
def from_type(thing):
    """Looks up the appropriate search strategy for the given type.

    ``from_type`` is used internally to fill in missing arguments to
    :func:`~hypothesis.strategies.builds` and can be used interactively
    to explore what strategies are available or to debug type resolution.

    You can use :func:`~hypothesis.strategies.register_type_strategy` to
    handle your custom types, or to globally redefine certain strategies -
    for example excluding NaN from floats, or use timezone-aware instead of
    naive time and datetime strategies.

    The resolution logic may be changed in a future version, but currently
    tries these five options:

    1. If ``thing`` is in the default lookup mapping or user-registered lookup,
       return the corresponding strategy.  The default lookup covers all types
       with Hypothesis strategies, including extras where possible.
    2. If ``thing`` is from the :mod:`python:typing` module, return the
       corresponding strategy (special logic).
    3. If ``thing`` has one or more subtypes in the merged lookup, return
       the union of the strategies for those types that are not subtypes of
       other elements in the lookup.
    4. Finally, if ``thing`` has type annotations for all required arguments,
       and is not an abstract class, it is resolved via
       :func:`~hypothesis.strategies.builds`.
    5. Because :mod:`abstract types <python:abc>` cannot be instantiated,
       we treat abstract types as the union of their concrete subclasses.
       Note that this lookup works via inheritance but not via
       :obj:`~python:abc.ABCMeta.register`, so you may still need to use
       :func:`~hypothesis.strategies.register_type_strategy`.

    There is a valuable recipe for leveraging ``from_type()`` to generate
    "everything except" values from a specified type. I.e.

    .. code-block:: python

        def everything_except(excluded_types):
            return (
                from_type(type)
                .flatmap(from_type)
                .filter(lambda x: not isinstance(x, excluded_types))
            )

    For example, ``everything_except(int)`` returns a strategy that can
    generate anything that ``from_type()`` can ever generate, except for
    instances of :class:`python:int`, and excluding instances of types
    added via :func:`~hypothesis.strategies.register_type_strategy`.

    This is useful when writing tests which check that invalid input is
    rejected in a certain way.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            return _from_type(thing)
    except Exception:
        return _from_type_deferred(thing)


def _from_type_deferred(thing):
    try:
        thing_repr = nicerepr(thing)
        if hasattr(thing, '__module__'):
            module_prefix = f'{thing.__module__}.'
            if not thing_repr.startswith(module_prefix):
                thing_repr = module_prefix + thing_repr
        repr_ = f'from_type({thing_repr})'
    except Exception:
        repr_ = None
    return LazyStrategy(lambda thing_inner: deferred(lambda : _from_type(
        thing_inner)), (thing,), {}, force_repr=repr_)


_recurse_guard: ContextVar[Sequence[type[Ex]]] = ContextVar('recurse_guard',
    default=[])


def _from_type(thing):
    from hypothesis.strategies._internal import types

    def as_strategy(strat_or_callable, thing_inner):
        if not isinstance(strat_or_callable, SearchStrategy):
            assert callable(strat_or_callable)
            strategy = strat_or_callable(thing_inner)
        else:
            strategy = strat_or_callable
        if strategy is NotImplemented:
            return NotImplemented
        if not isinstance(strategy, SearchStrategy):
            raise ResolutionFailed(
                f'Error: {thing_inner} was registered for {nicerepr(strat_or_callable)}, but returned non-strategy {strategy!r}'
                )
        if strategy.is_empty:
            raise ResolutionFailed(
                f'Error: {thing_inner!r} resolved to an empty strategy')
        return strategy

    def from_type_guarded(thing_inner):
        """Returns the result of producer, or ... if recursion on thing is encountered"""
        recurse_guard = _recurse_guard.get().copy()
        if thing_inner in recurse_guard:
            raise RewindRecursive(thing_inner)
        recurse_guard.append(thing_inner)
        _recurse_guard.set(recurse_guard)
        try:
            return _from_type(thing_inner)
        except RewindRecursive as rr:
            if rr.target != thing_inner:
                raise
            return ...
        finally:
            recurse_guard.pop()
    try:
        known = thing in types._global_type_lookup
    except TypeError:
        known = False
    if known:
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
                f'Got {thing!r} as a type annotation, but the forward-reference could not be resolved from a string to a type.  Consider using `from __future__ import annotations` instead of forward-reference strings.'
                )
        raise InvalidArgument(f'thing={thing!r} must be a type')
    if thing in types.NON_RUNTIME_TYPES:
        raise InvalidArgument(
            f'Could not resolve {thing!r} to a strategy, because there is no such thing as a runtime instance of {thing!r}'
            )
    try:
        if thing in types._global_type_lookup:
            strategy = as_strategy(types._global_type_lookup[thing], thing)
            if strategy is not NotImplemented:
                return strategy
        elif isinstance(thing, GenericAlias) and (to := get_origin(thing)
            ) in types._global_type_lookup:
            strategy = as_strategy(types._global_type_lookup[to], thing)
            if strategy is not NotImplemented:
                return strategy
    except TypeError:
        pass
    if hasattr(typing, '_TypedDictMeta') and type(thing
        ) is typing._TypedDictMeta or hasattr(types.typing_extensions,
        '_TypedDictMeta') and type(thing
        ) is types.typing_extensions._TypedDictMeta:

        def _get_annotation_arg(key, annotation_type):
            try:
                return get_args(annotation_type)[0]
            except IndexError:
                raise InvalidArgument(
                    f'`{key}: {annotation_type.__name__}` is not a valid type annotation'
                    ) from None

        def _get_typeddict_qualifiers(key, annotation_type):
            qualifiers = set()
            while True:
                annotation_origin = types.extended_get_origin(annotation_type)
                if annotation_origin is Annotated:
                    annotation_args = get_args(annotation_type)
                    if annotation_args:
                        annotation_type = annotation_args[0]
                    else:
                        break
                elif annotation_origin in types.RequiredTypes:
                    qualifiers.add(types.RequiredTypes)
                    annotation_type = _get_annotation_arg(key, annotation_type)
                elif annotation_origin in types.NotRequiredTypes:
                    qualifiers.add(types.NotRequiredTypes)
                    annotation_type = _get_annotation_arg(key, annotation_type)
                elif annotation_origin in types.ReadOnlyTypes:
                    qualifiers.add(types.ReadOnlyTypes)
                    annotation_type = _get_annotation_arg(key, annotation_type)
                else:
                    break
            return qualifiers, annotation_type
        optional = set(getattr(thing, '__optional_keys__', ()))
        required = set(getattr(thing, '__required_keys__', get_type_hints(
            thing).keys()))
        anns: dict[str, SearchStrategy[Any] | EllipsisType] = {}
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
        if not required.isdisjoint(optional):
            raise InvalidArgument(
                f'Required keys overlap with optional keys in a TypedDict: required={required!r}, optional={optional!r}'
                )
        if not anns and thing.__annotations__ and '.<locals>.' in getattr(thing
            , '__qualname__', ''):
            raise InvalidArgument(
                'Failed to retrieve type annotations for local type')
        return fixed_dictionaries(mapping={k: v for k, v in anns.items() if
            k in required}, optional={k: v for k, v in anns.items() if k in
            optional})
    if isinstance(thing, types.typing_root_type) or isinstance(get_origin(
        thing), type) and get_args(thing):
        return types.from_typing_type(thing)
    strategies: list[SearchStrategy[Any]] = [s for s in (as_strategy(v, k) for
        k, v in sorted(types._global_type_lookup.items(), key=lambda item:
        repr(item)) if isinstance(k, type) and issubclass(k, thing) and all
        (not types.try_issubclass(k, other) for other in types.
        _global_type_lookup)) if s is not NotImplemented and not s.is_empty]
    if strategies:
        return one_of(strategies)
    if issubclass(thing, enum.Enum):
        return sampled_from(thing)
    if not isabstract(thing):
        required = required_args(thing)
        if required and not (required.issubset(get_type_hints(thing)) or
            attr.has(thing) or is_typed_named_tuple(thing)):
            raise ResolutionFailed(
                f'Could not resolve {thing!r} to a strategy; consider using register_type_strategy'
                )
        try:
            hints = get_type_hints(thing)
            params = get_signature(thing).parameters
        except Exception:
            params = {}
        posonly_args: list[SearchStrategy[Any]] = []
        kwargs: dict[str, SearchStrategy[Any]] = {}
        for k, p in params.items():
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.
                KEYWORD_ONLY) and k in hints and k != 'return':
                ps: SearchStrategy[Any] | EllipsisType = from_type_guarded(
                    hints[k])
                if p.default is not Parameter.empty and ps is not ...:
                    ps = just(p.default) | cast(SearchStrategy[Any], ps)
                if p.kind is Parameter.POSITIONAL_ONLY:
                    if ps is ...:
                        if p.default is Parameter.empty:
                            raise ResolutionFailed(
                                f'Could not resolve {thing!r} to a strategy; consider using register_type_strategy'
                                )
                        ps = just(p.default)
                    posonly_args.append(ps)
                else:
                    kwargs[k] = ps
        if params and not (posonly_args or kwargs) and not issubclass(thing,
            BaseException):
            from_type_repr = repr_call(from_type, (thing,), {})
            builds_repr = repr_call(builds, (thing,), {})
            warnings.warn(
                f'{from_type_repr} resolved to {builds_repr}, because we could not find any (non-varargs) arguments. Use st.register_type_strategy() to resolve to a strategy which can generate more than one value, or silence this warning.'
                , SmallSearchSpaceWarning, stacklevel=2 if sys.version_info
                [:2] > (3, 9) else 5)
        return builds(thing, *posonly_args, **kwargs)
    subclasses = thing.__subclasses__()
    if not subclasses:
        raise ResolutionFailed(
            f'Could not resolve {thing!r} to a strategy, because it is an abstract type without any subclasses. Consider using register_type_strategy'
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
def fractions(min_value=None, max_value=None, *, max_denominator: Optional[
    int]=None):
    """Returns a strategy which generates Fractions.

    If ``min_value`` is not None then all generated values are no less than
    ``min_value``.  If ``max_value`` is not None then all generated values are no
    greater than ``max_value``.  ``min_value`` and ``max_value`` may be anything accepted
    by the :class:`~fractions.Fraction` constructor.

    If ``max_denominator`` is not None then the denominator of any generated
    values is no greater than ``max_denominator``. Note that ``max_denominator`` must
    be None or a positive integer.

    Examples from this strategy shrink towards smaller denominators, then
    closer to zero.
    """
    min_value = try_convert(Fraction, min_value, 'min_value')
    max_value = try_convert(Fraction, max_value, 'max_value')
    assert min_value is None or isinstance(min_value, Fraction)
    assert max_value is None or isinstance(max_value, Fraction)
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    check_valid_integer(max_denominator, 'max_denominator')
    if max_denominator is not None:
        if max_denominator < 1:
            raise InvalidArgument(
                f'max_denominator={max_denominator!r} must be >= 1')
        if min_value is not None and min_value.denominator > max_denominator:
            raise InvalidArgument(
                f'The min_value={min_value!r} has a denominator greater than the max_denominator={max_denominator!r}'
                )
        if max_value is not None and max_value.denominator > max_denominator:
            raise InvalidArgument(
                f'The max_value={max_value!r} has a denominator greater than the max_denominator={max_denominator!r}'
                )
    if min_value is not None and min_value == max_value:
        return just(min_value)

    def dm_func(denom):
        """Take denom, construct numerator strategy, and build fraction."""
        min_num, max_num = None, None
        if max_value is None and min_value is None:
            pass
        elif min_value is None:
            max_num = denom * max_value.numerator
            denom *= max_value.denominator
        elif max_value is None:
            min_num = denom * min_value.numerator
            denom *= min_value.denominator
        else:
            low = min_value.numerator * max_value.denominator
            high = max_value.numerator * min_value.denominator
            scale = min_value.denominator * max_value.denominator
            div = math.gcd(scale, math.gcd(low, high))
            min_num = denom * low // div
            max_num = denom * high // div
            denom *= scale // div
        return builds(Fraction, integers(min_value=min_num, max_value=
            max_num), just(denom))
    if max_denominator is None:
        return integers(min_value=1).flatmap(dm_func)
    return integers(1, max_denominator).flatmap(dm_func).map(lambda f: f.
        limit_denominator(max_denominator))


def _as_finite_decimal(value, name, allow_infinity):
    """Convert decimal bounds to decimals, carefully."""
    assert name in ('min_value', 'max_value')
    if value is None:
        return None
    if not isinstance(value, Decimal):
        with localcontext(Context()):
            value = try_convert(Decimal, value, name)
    assert isinstance(value, Decimal)
    if value.is_finite():
        return value
    if value.is_infinite() and (value < 0 if 'min' in name else value > 0):
        if allow_infinity or allow_infinity is None:
            return None
        raise InvalidArgument(
            f'allow_infinity={allow_infinity!r}, but {name}={value!r}')
    raise InvalidArgument(f'Invalid {name}={value!r}')


@cacheable
@defines_strategy(force_reusable_values=True)
def decimals(min_value=None, max_value=None, *, allow_nan: Optional[bool]=
    None, allow_infinity: Optional[bool]=None, places: Optional[int]=None):
    """Generates instances of :class:`python:decimal.Decimal`, which may be:

    - A finite rational number, between ``min_value`` and ``max_value``.
    - Not a Number, if ``allow_nan`` is True.  None means "allow NaN, unless
      ``min_value`` and ``max_value`` are not None".
    - Positive or negative infinity, if ``max_value`` and ``min_value``
      respectively are None, and ``allow_infinity`` is not False.  None means
      "allow infinity, unless excluded by the min and max values".

    Note that where floats have one ``NaN`` value, Decimals have four: signed,
    and either *quiet* or *signalling*.  See `the decimal module docs
    <https://docs.python.org/3/library/decimal.html#special-values>`_ for
    more information on special values.

    If ``places`` is not None, all finite values drawn from the strategy will
    have that number of digits after the decimal place.

    Examples from this strategy do not have a well defined shrink order but
    try to maximize human readability when shrinking.
    """
    check_valid_integer(places, 'places')
    if places is not None and places < 0:
        raise InvalidArgument(f'places={places!r} may not be negative')
    min_value = _as_finite_decimal(min_value, 'min_value', allow_infinity)
    max_value = _as_finite_decimal(max_value, 'max_value', allow_infinity)
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    if allow_infinity and None not in (min_value, max_value):
        raise InvalidArgument('Cannot allow infinity between finite bounds')
    if allow_nan is None:
        allow_nan = bool(min_value == 0 and max_value is None)
    elif allow_nan and not (min_value == 0 and max_value is None):
        raise InvalidArgument(
            f'Cannot have allow_nan={allow_nan!r}, min_value={min_value!r}, max_value={max_value!r}'
            )
    check_type(bool, allow_subnormal, 'allow_subnormal')
    if places is not None:

        def ctx(val):
            """Return a context in which this value is lossless."""
            precision = ceil(math.log10(abs(val) or 1)) + places + 1
            return Context(prec=max([precision, 1]))

        def int_to_decimal(val):
            context = ctx(val)
            return context.quantize(context.multiply(val, factor), factor)
        factor: Decimal = Decimal(10) ** -places
        min_num: Optional[int] = None
        max_num: Optional[int] = None
        if min_value is not None:
            min_num = ceil(ctx(Decimal(min_value)).divide(Decimal(min_value
                ), factor))
        if max_value is not None:
            max_num = floor(ctx(Decimal(max_value)).divide(Decimal(
                max_value), factor))
        if min_num is not None and max_num is not None and min_num > max_num:
            raise InvalidArgument(
                f'There are no decimals with {places} places between min_value={min_value!r} and max_value={max_value!r}'
                )
        strat: SearchStrategy[Decimal] = integers(min_num, max_num).map(
            int_to_decimal)
    else:

        def fraction_to_decimal(val):
            precision = ceil(math.log10(abs(val.numerator) or 1) + math.
                log10(val.denominator)) + 1
            return Context(prec=precision or 1).divide(Decimal(val.
                numerator), val.denominator)
        strat: SearchStrategy[Decimal] = fractions(min_value, max_value).map(
            fraction_to_decimal)
    special: list[Decimal] = []
    if allow_nan or allow_nan is None and None in (min_value, max_value):
        special.extend(map(Decimal, ('NaN', '-NaN', 'sNaN', '-sNaN')))
    if allow_infinity or allow_infinity is None and max_value is None:
        special.append(Decimal('Infinity'))
    if allow_infinity or allow_infinity is None and min_value is None:
        special.append(Decimal('-Infinity'))
    return strat | (sampled_from(special) if special else nothing())


@defines_strategy(never_lazy=True)
def recursive(base, extend, *, max_leaves: int=100):
    """base: A strategy to start from.

    extend: A function which takes a strategy and returns a new strategy.

    max_leaves: The maximum number of elements to be drawn from base on a given
    run.

    This returns a strategy ``S`` such that ``S = extend(base | S)``. That is,
    values may be drawn from base, or from any strategy reachable by mixing
    applications of | and extend.

    An example may clarify: ``recursive(booleans(), lists)`` would return a
    strategy that may return arbitrarily nested and mixed lists of booleans.
    So e.g. ``False``, ``[True]``, ``[False, []]``, and ``[[[[True]]]]`` are
    all valid values to be drawn from that strategy.

    Examples from this strategy shrink by trying to reduce the amount of
    recursion and by shrinking according to the shrinking behaviour of base
    and the result of extend.

    """
    return RecursiveStrategy(base, extend, max_leaves)


class PermutationStrategy(SearchStrategy[list[T]]):

    def __init__(self, values):
        self.values = values

    def do_draw(self, data):
        result = list(self.values)
        for i in range(len(result) - 1):
            j = data.draw_integer(i, len(result) - 1)
            result[i], result[j] = result[j], result[i]
        return result

    def __repr__(self):
        return f'permutations({self.values!r})'


@defines_strategy()
def permutations(values):
    """Return a strategy which returns permutations of the ordered collection
    ``values``.

    Examples from this strategy shrink by trying to become closer to the
    original order of values.
    """
    values_checked = check_sample(values, 'permutations')
    if not values_checked:
        return builds(list)
    return PermutationStrategy(values_checked)


class CompositeStrategy(SearchStrategy[Any]):

    def __init__(self, definition, args, kwargs):
        self.definition = definition
        self.args = args
        self.kwargs = kwargs

    def do_draw(self, data):
        return self.definition(data.draw, *self.args, **self.kwargs)

    def calc_label(self):
        return calc_label_from_cls(self.definition)


class DrawFn(Protocol):
    """This type only exists so that you can write type hints for functions
    decorated with :func:`@composite <hypothesis.strategies.composite>`.
    Do not use it directly!

    .. code-block:: python

        @composite
        def list_and_index(draw: DrawFn) -> tuple[int, str]:
            i = draw(integers())  # type inferred as 'int'
            s = draw(text())  # type inferred as 'str'
            return i, s

    """

    def __init__(self):
        raise TypeError('Protocols cannot be instantiated')
    __signature__: Signature = Signature(parameters=[])

    def __call__(self, strategy, label=None):
        ...


def _composite(f):
    if isinstance(f, (classmethod, staticmethod)):
        special_method = type(f)
        f_func = f.__func__
    else:
        special_method = None
        f_func = f
    sig = get_signature(f_func)
    params = tuple(sig.parameters.values())
    if not (params and 'POSITIONAL' in params[0].kind.name):
        raise InvalidArgument(
            'Functions wrapped with composite must take at least one positional argument.'
            )
    if params[0].default is not sig.empty:
        raise InvalidArgument(
            'A default value for initial argument will never be used')
    if not (f_func is typing._overload_dummy or
        is_first_param_referenced_in_function(f_func)):
        note_deprecation(
            'There is no reason to use @st.composite on a function which does not call the provided draw() function internally.'
            , since='2022-07-17', has_codemod=False)
    if get_origin(sig.return_annotation) is SearchStrategy:
        ret_repr = repr(sig.return_annotation).replace('hypothesis.strategies.'
            , 'st.')
        warnings.warn(
            f'Return-type annotation is `{ret_repr}`, but the decorated function should return a value (not a strategy)'
            , HypothesisWarning, stacklevel=3 if sys.version_info[:2] > (3,
            9) else 5)
    if params[0].kind.name != 'VAR_POSITIONAL':
        params = params[1:]
    newsig = sig.replace(parameters=params, return_annotation=
        SearchStrategy if sig.return_annotation is sig.empty else
        SearchStrategy[sig.return_annotation])

    @defines_strategy()
    @define_function_signature(f_func.__name__, f_func.__doc__, newsig)
    def accept(*args: Any, **kwargs: Any):
        return CompositeStrategy(f_func, args, kwargs)
    accept.__module__ = f_func.__module__
    accept.__signature__ = newsig
    if special_method is not None:
        return special_method(accept)
    return accept


if typing.TYPE_CHECKING or ParamSpec is not None:
    P = ParamSpec('P')

    def composite(f):
        """Defines a strategy that is built out of potentially arbitrarily many
        other strategies.

        This is intended to be used as a decorator. See
        :ref:`the full documentation for more details <composite-strategies>`
        about how to use this function.

        Examples from this strategy shrink by shrinking the output of each draw
        call.
        """
        return _composite(f)
else:

    @defines_strategy()
    def composite(f):
        """Defines a strategy that is built out of potentially arbitrarily many
        other strategies.

        This is intended to be used as a decorator. See
        :ref:`the full documentation for more details <composite-strategies>`
        about how to use this function.

        Examples from this strategy shrink by shrinking the output of each draw
        call.
        """
        return _composite(f)


@cacheable
@defines_strategy(force_reusable_values=True)
def complex_numbers(*, min_magnitude: Real=0, max_magnitude: Optional[Real]
    =None, allow_infinity: Optional[bool]=None, allow_nan: Optional[bool]=
    None, allow_subnormal: bool=True, width: Literal[32, 64, 128]=128):
    """Returns a strategy that generates :class:`~python:complex`
    numbers.

    The strategy draws complex numbers with constrained magnitudes.
    The ``min_magnitude`` and ``max_magnitude`` parameters should be
    non-negative :class:`~python:numbers.Real` numbers; a value
    of ``None`` corresponds an infinite upper bound.

    If ``allow_infinity``, ``allow_nan``, and ``allow_subnormal`` are
    enabled or disabled, they respectively apply to each part of the complex number.

    The ``width`` argument specifies the maximum number of bits of precision
    required to represent the entire generated complex number.
    Valid values are 32, 64 or 128, which correspond to the real and imaginary
    components each having width 16, 32 or 64, respectively.
    Passing ``width=64`` will still use the builtin 128-bit
    :class:`~python:complex` class, but always for values which can be
    exactly represented as two 32-bit floats.

    Examples from this strategy shrink by shrinking their real and
    imaginary parts, as :func:`~hypothesis.strategies.floats`.
    """
    check_valid_magnitude(min_magnitude, 'min_magnitude')
    check_valid_magnitude(max_magnitude, 'max_magnitude')
    check_valid_interval(min_magnitude, max_magnitude, 'min_magnitude',
        'max_magnitude')
    if max_magnitude == math.inf:
        max_magnitude = None
    if allow_infinity is None:
        allow_infinity = bool(max_magnitude is None)
    elif allow_infinity and max_magnitude is not None:
        raise InvalidArgument(
            f'Cannot have allow_infinity={allow_infinity!r} with max_magnitude={max_magnitude!r}'
            )
    if allow_nan is None:
        allow_nan = bool(min_magnitude == 0 and max_magnitude is None)
    elif allow_nan and not (min_magnitude == 0 and max_magnitude is None):
        raise InvalidArgument(
            f'Cannot have allow_nan={allow_nan!r}, min_magnitude={min_magnitude!r}, max_magnitude={max_magnitude!r}'
            )
    check_type(bool, allow_subnormal, 'allow_subnormal')
    if width not in (32, 64, 128):
        raise InvalidArgument(
            f'width={width!r}, but must be 32, 64 or 128 (other complex dtypes such as complex192 or complex256 are not supported)'
            )
    component_width = width // 2
    allow_kw = {'allow_nan': allow_nan, 'allow_infinity': allow_infinity,
        'allow_subnormal': None if allow_subnormal else allow_subnormal,
        'width': component_width}
    if min_magnitude == 0 and max_magnitude is None:
        return builds(complex, floats(**allow_kw), floats(**allow_kw))

    @composite
    def constrained_complex(draw):
        if max_magnitude is None:
            zi = draw(floats(**allow_kw))
            rmax = None
        else:
            zi = draw(floats(-float_of(max_magnitude, component_width),
                float_of(max_magnitude, component_width), **allow_kw))
            rmax = float_of(cathetus(max_magnitude, zi), component_width)
        if min_magnitude == 0 or math.fabs(zi) >= min_magnitude:
            zr = draw(floats(None if rmax is None else -rmax, rmax, **allow_kw)
                )
        else:
            rmin = float_of(cathetus(min_magnitude, zi), component_width)
            zr = draw(floats(rmin, rmax, **allow_kw))
        if min_magnitude > 0 and draw(booleans()) and math.fabs(zi
            ) <= min_magnitude:
            zr = -zr
        return complex(zr, zi)
    return constrained_complex()


def _as_integer(value, name):
    """Convert a value to an integer, handling None."""
    if value is None:
        return None
    return try_convert(int, value, name)


@cacheable
@defines_strategy(never_lazy=True)
def shared(base, *, key: Optional[Hashable]=None):
    """Returns a strategy that draws a single shared value per run, drawn from
    base. Any two shared instances with the same key will share the same value,
    otherwise the identity of this strategy will be used. That is:

    >>> s = integers()  # or any other strategy
    >>> x = shared(s)
    >>> y = shared(s)

    In the above x and y may draw different (or potentially the same) values.
    In the following they will always draw the same:

    >>> x = shared(s, key="hi")
    >>> y = shared(s, key="hi")

    Examples from this strategy shrink as per their base strategy.
    """
    return SharedStrategy(base, key)


class DataObject:
    """This type only exists so that you can write type hints for tests using
    the :func:`~hypothesis.strategies.data` strategy.  Do not use it directly!
    """

    def __init__(self, data):
        self.count: int = 0
        self.conjecture_data = data
    __signature__: Signature = Signature(parameters=[])

    def __repr__(self):
        return 'data(...)'

    def draw(self, strategy, label=None):
        check_strategy(strategy, 'strategy')
        self.count += 1
        desc = f"Draw {self.count}{'' if label is None else f' ({label})'}"
        with deprecate_random_in_strategy('{}from {!r}', desc, strategy):
            result = self.conjecture_data.draw(strategy, observe_as=
                f'generate:{desc}')
        if should_note():
            printer = RepresentationPrinter(context=current_build_context())
            printer.text(f'{desc}: ')
            if self.conjecture_data.provider.avoid_realization:
                printer.text('<symbolic>')
            else:
                printer.pretty(result)
            note(printer.getvalue())
        return result


class DataStrategy(SearchStrategy[DataObject]):
    """This strategy provides a DataObject for interactive data drawing."""
    supports_find: bool = False

    def do_draw(self, data):
        if not hasattr(data, 'hypothesis_shared_data_strategy'):
            data.hypothesis_shared_data_strategy = DataObject(data)
        return data.hypothesis_shared_data_strategy

    def __repr__(self):
        return 'data()'

    def map(self, f):
        self.__not_a_first_class_strategy('map')
        return self

    def filter(self, f):
        self.__not_a_first_class_strategy('filter')
        return self

    def flatmap(self, f):
        self.__not_a_first_class_strategy('flatmap')
        return self

    def example(self):
        self.__not_a_first_class_strategy('example')

    def __not_a_first_class_strategy(self, name):
        raise InvalidArgument(
            f"Cannot call {name} on a DataStrategy. You should probably be using @composite for whatever it is you're trying to do."
            )


@cacheable
@defines_strategy(never_lazy=True)
def data():
    """This isn't really a normal strategy, but instead gives you an object
    which can be used to draw data interactively from other strategies.

    See :ref:`the rest of the documentation <interactive-draw>` for more
    complete information.

    Examples from this strategy do not shrink (because there is only one),
    but the result of calls to each ``data.draw()`` call shrink as they normally would.
    """
    return DataStrategy()


def register_type_strategy(custom_type, strategy):
    """Add an entry to the global type-to-strategy lookup.

    This lookup is used in :func:`~hypothesis.strategies.builds` and
    :func:`@given <hypothesis.given>`.

    :func:`~hypothesis.strategies.builds` will be used automatically for
    classes with type annotations on ``__init__`` , so you only need to
    register a strategy if one or more arguments need to be more tightly
    defined than their type-based default, or if you want to supply a strategy
    for an argument with a default value.

    ``strategy`` may be a search strategy, or a function that takes a type and
    returns a strategy (useful for generic types). The function may return
    :data:`NotImplemented` to conditionally not provide a strategy for the type
    (the type will still be resolved by other methods, if possible, as if the
    function was not registered).

    Note that you may not register a parametrised generic type (such as
    ``MyCollection[int]``) directly, because the resolution logic does not
    handle this case correctly.  Instead, you may register a *function* for
    ``MyCollection`` and `inspect the type parameters within that function
    <https://stackoverflow.com/q/48572831>`__.
    """
    from hypothesis.strategies._internal import types
    if not types.is_a_type(custom_type):
        raise InvalidArgument(f'custom_type={custom_type!r} must be a type')
    if custom_type in types.NON_RUNTIME_TYPES:
        raise InvalidArgument(
            f'custom_type={custom_type!r} is not allowed to be registered, because there is no such thing as a runtime instance of {custom_type!r}'
            )
    if not (isinstance(strategy, SearchStrategy) or callable(strategy)):
        raise InvalidArgument(
            f'strategy={strategy!r} must be a SearchStrategy, or a function that takes a generic type and returns a specific SearchStrategy'
            )
    if isinstance(strategy, SearchStrategy):
        with warnings.catch_warnings():
            warnings.simplefilter('error', HypothesisSideeffectWarning)
            try:
                if strategy.is_empty:
                    raise InvalidArgument(
                        f'strategy={strategy!r} must not be empty')
            except HypothesisSideeffectWarning:
                pass
    if types.has_type_arguments(custom_type):
        raise InvalidArgument(
            f'Cannot register generic type {custom_type!r}, because it has type arguments which would not be handled.  Instead, register a function for {get_origin(custom_type)!r} which can inspect specific type objects and return a strategy.'
            )
    if 'pydantic.generics' in sys.modules and issubclass(custom_type, sys.
        modules['pydantic.generics'].GenericModel) and not re.search(
        '[A-Za-z_]+\\[.+\\]', repr(custom_type)) and callable(strategy):
        raise InvalidArgument(
            f"Cannot register a function for {custom_type!r}, because parametrized `pydantic.generics.GenericModel` subclasses aren't actually generic types at runtime.  In this case, you should register a strategy directly for each parametrized form that you anticipate using."
            )
    types._global_type_lookup[custom_type] = strategy
    from_type.__clear_cache()


@cacheable
@defines_strategy(force_reusable_values=True)
def deferred(definition):
    """A deferred strategy allows you to write a strategy that references other
    strategies that have not yet been defined. This allows for the easy
    definition of recursive and mutually recursive strategies.

    The definition argument should be a zero-argument function that returns a
    strategy. It will be evaluated the first time the strategy is used to
    produce an example.

    Example usage:

    >>> import hypothesis.strategies as st
    >>> x = st.deferred(lambda: st.booleans() | st.tuples(x, x))
    >>> x.example()
    (((False, (True, True)), (False, True)), (True, True))
    >>> x.example()
    True

    Mutual recursion also works fine:

    >>> a = st.deferred(lambda: st.booleans() | b)
    >>> b = st.deferred(lambda: st.tuples(a, a))
    >>> a.example()
    True
    >>> b.example()
    (False, (False, ((False, True), False)))

    Examples from this strategy shrink as they normally would from the strategy
    returned by the definition.
    """
    return DeferredStrategy(definition)


def domains():
    import hypothesis.provisional
    return hypothesis.provisional.domains()


@cacheable
@defines_strategy(force_reusable_values=True)
def emails(*, domains: SearchStrategy[str]=LazyStrategy(domains, (), {})):
    """A strategy for generating email addresses as unicode strings. The
    address format is specified in :rfc:`5322#section-3.4.1`. Values shrink
    towards shorter local-parts and host domains.

    If ``domains`` is given then it must be a strategy that generates domain
    names for the emails, defaulting to :func:`~hypothesis.provisional.domains`.

    This strategy is useful for generating "user data" for tests, as
    mishandling of email addresses is a common source of bugs.
    """
    local_chars = string.ascii_letters + string.digits + "!#$%&'*+-/=^_`{|}~"
    local_part = text(local_chars, min_size=1, max_size=64)
    return builds('{}@{}'.format, local_part, domains).filter(lambda addr: 
        len(addr) <= 254)


class RunnerStrategy(SearchStrategy[Any]):

    def __init__(self, default=not_set):
        self.default = default

    def do_draw(self, data):
        runner = getattr(data, 'hypothesis_runner', not_set)
        if runner is not_set:
            if self.default is not_set:
                raise InvalidArgument(
                    'Cannot use runner() strategy with no associated runner or explicit default.'
                    )
            else:
                return self.default
        else:
            return runner

    def __repr__(self):
        return 'runner()'


@cacheable
@defines_strategy(force_reusable_values=True)
def runner(*, default: Any=not_set):
    """A strategy for getting "the current test runner", whatever that may be.
    The exact meaning depends on the entry point, but it will usually be the
    associated 'self' value for it.

    If you are using this in a rule for stateful testing, this strategy
    will return the instance of the :class:`~hypothesis.stateful.RuleBasedStateMachine`
    that the rule is running for.

    If there is no current test runner and a default is provided, return
    that default. If no default is provided, raises InvalidArgument.

    Examples from this strategy do not shrink (because there is only one).
    """
    return RunnerStrategy(default)


@defines_strategy(force_reusable_values=True)
@cacheable
def from_type(thing):
    return _from_type(thing)


@cacheable
@defines_strategy(force_reusable_values=True)
def from_type_deferred(thing):
    return _from_type_deferred(thing)


@defines_strategy()
@cacheable
def complex_numbers(*, min_magnitude: Real=0, max_magnitude: Optional[Real]
    =None, allow_infinity: Optional[bool]=None, allow_nan: Optional[bool]=
    None, allow_subnormal: bool=True, width: Literal[32, 64, 128]=128):
    return complex_numbers(min_magnitude=min_magnitude, max_magnitude=
        max_magnitude, allow_infinity=allow_infinity, allow_nan=allow_nan,
        allow_subnormal=allow_subnormal, width=width)


def _maybe_nil_uuids(draw, uuid):
    if draw(booleans()).bit_length() <= 1 / 64:
        return UUID('00000000-0000-0000-0000-000000000000')
    return uuid


@cacheable
@defines_strategy(force_reusable_values=True)
def uuids(*, version: Optional[Literal[1, 2, 3, 4, 5]]=None, allow_nil:
    bool=False):
    """Returns a strategy that generates :class:`UUIDs <uuid.UUID>`.

    If the optional version argument is given, value is passed through
    to :class:`~python:uuid.UUID` and only UUIDs of that version will
    be generated.

    If ``allow_nil`` is True, generate the nil UUID much more often.
    Otherwise, all returned values from this will be unique, so e.g. if you do
    ``lists(uuids())`` the resulting list will never contain duplicates.

    Examples from this strategy do not have any meaningful shrink order.
    """
    check_type(bool, allow_nil, 'allow_nil')
    if version not in (None, 1, 2, 3, 4, 5):
        raise InvalidArgument(
            f'version={version!r}, but version must be in (None, 1, 2, 3, 4, 5) to pass to the uuid.UUID constructor.'
            )
    random_uuids = shared(randoms(use_true_random=True), key=
        'hypothesis.strategies.uuids.generator').map(lambda r: UUID(version
        =version, int=r.getrandbits(128)))
    if allow_nil:
        if version is not None:
            raise InvalidArgument('The nil UUID is not of any version')
        return random_uuids.flatmap(lambda uuid: _maybe_nil_uuids(draw, uuid))
    return random_uuids


class RunnerStrategy(SearchStrategy[Any]):

    def __init__(self, default=not_set):
        self.default = default

    def do_draw(self, data):
        runner = getattr(data, 'hypothesis_runner', not_set)
        if runner is not_set:
            if self.default is not_set:
                raise InvalidArgument(
                    'Cannot use runner() strategy with no associated runner or explicit default.'
                    )
            else:
                return self.default
        else:
            return runner

    def __repr__(self):
        return 'runner()'


class OrderedStrategy(SearchStrategy[Any]):
    pass


@defines_strategy()
@cacheable
def random_module():
    return random_module()


@defines_strategy(force_reusable_values=True)
@cacheable
def builds(target: Callable[..., Ex], /, *args: SearchStrategy[Any], **
    kwargs: Union[SearchStrategy[Any], EllipsisType]):
    return builds(target, *args, **kwargs)


@defines_strategy(force_reusable_values=True)
@cacheable
def from_type(thing):
    return from_type(thing)


@cacheable
@defines_strategy(never_lazy=True)
def shared(base, *, key: Optional[Hashable]=None):
    return shared(base, key=key)


@composite
def slices(draw, size):
    """Generates slices that will select indices up to the supplied size

    Generated slices will have start and stop indices that range from -size to size - 1
    and will step in the appropriate direction. Slices should only produce an empty selection
    if the start and end are the same.

    Examples from this strategy shrink toward 0 and smaller values
    """
    check_valid_size(size, 'size')
    if size == 0:
        step = draw(none() | integers().filter(bool))
        return slice(None, None, step)
    start = draw(integers(0, size - 1) | none())
    stop = draw(integers(0, size) | none())
    if start is None and stop is None:
        max_step = size
    elif start is None:
        max_step = stop
    elif stop is None:
        max_step = start
    else:
        max_step = abs(start - stop)
    step = draw(integers(1, max_step or 1))
    if draw(booleans()) and start == stop or (stop or 0) < (start or 0):
        step *= -1
    if draw(booleans()) and start is not None:
        start -= size
    if draw(booleans()) and stop is not None:
        stop -= size
    if not draw(booleans()) and step == 1:
        step = None
    return slice(start, stop, step)


def _from_type(thing):
    return _from_type(thing)


def _from_type_deferred(thing):
    return _from_type_deferred(thing)


@cacheable
@defines_strategy(force_reusable_values=True)
def functions(*, like: Callable[..., Any]=lambda : None, returns: Union[
    SearchStrategy[Any], EllipsisType]=..., pure: bool=False):
    """functions(*, like=lambda: None, returns=..., pure=False)

    A strategy for functions, which can be used in callbacks.

    The generated functions will mimic the interface of ``like``, which must
    be a callable (including a class, method, or function).  The return value
    for the function is drawn from the ``returns`` argument, which must be a
    strategy.  If ``returns`` is not passed, we attempt to infer a strategy
    from the return-type annotation if present, falling back to :func:`~none`.

    If ``pure=True``, all arguments passed to the generated function must be
    hashable, and if passed identical arguments the original return value will
    be returned again - *not* regenerated, so beware mutable values.

    If ``pure=False``, generated functions do not validate their arguments, and
    may return a different value if called again with the same arguments.

    Generated functions can only be called within the scope of the ``@given``
    which created them.  This strategy does not support ``.example()``.
    """
    return _functions(like=like, returns=returns, pure=pure)


@defines_strategy(force_reusable_values=True)
@cacheable
def complex_numbers(*, min_magnitude: Real=0, max_magnitude: Optional[Real]
    =None, allow_infinity: Optional[bool]=None, allow_nan: Optional[bool]=
    None, allow_subnormal: bool=True, width: Literal[32, 64, 128]=128):
    return complex_numbers(min_magnitude=min_magnitude, max_magnitude=
        max_magnitude, allow_infinity=allow_infinity, allow_nan=allow_nan,
        allow_subnormal=allow_subnormal, width=width)
