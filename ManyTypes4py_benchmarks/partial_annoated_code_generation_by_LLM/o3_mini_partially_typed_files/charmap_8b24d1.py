#!/usr/bin/env python3
import codecs
import gzip
import json
import os
import sys
import tempfile
import unicodedata
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Iterable as IterableType, Tuple, Set, Any
from hypothesis.configuration import storage_directory
from hypothesis.control import _current_build_context
from hypothesis.errors import InvalidArgument
from hypothesis.internal.intervalsets import IntervalSet, IntervalsT

if TYPE_CHECKING:
    from typing import TypeAlias

CategoryName = Literal[
    'L', 'Lu', 'Ll', 'Lt', 'Lm', 'Lo',
    'M', 'Mn', 'Mc', 'Me',
    'N', 'Nd', 'Nl', 'No',
    'P', 'Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po',
    'S', 'Sm', 'Sc', 'Sk', 'So',
    'Z', 'Zs', 'Zl', 'Zp',
    'C', 'Cc', 'Cf', 'Cs', 'Co', 'Cn'
]
Categories = IterableType[CategoryName]
CategoriesTuple = Tuple[CategoryName, ...]

_charmap: Optional[dict[CategoryName, IntervalsT]] = None
_categories: Optional[CategoriesTuple] = None
category_index_cache: dict[frozenset[CategoryName], IntervalsT] = {frozenset(): ()}
limited_category_index_cache: dict[Tuple[CategoriesTuple, int, int, IntervalsT, IntervalsT], IntervalSet] = {}


def charmap_file(fname: str = 'charmap') -> Path:
    return storage_directory('unicode_data', unicodedata.unidata_version, f'{fname}.json.gz')


def charmap() -> dict[CategoryName, IntervalsT]:
    """Return a dict that maps a Unicode category to a tuple of 2-tuples covering the codepoint intervals for characters in that category.

    >>> charmap()['Co']
    ((57344, 63743), (983040, 1048573), (1048576, 1114109))
    """
    global _charmap
    if _charmap is None:
        f: Path = charmap_file()
        try:
            with gzip.GzipFile(f, 'rb') as d:
                tmp_charmap: dict[str, Any] = dict(json.load(d))
        except Exception:
            category = unicodedata.category
            tmp_charmap = {}  # type: dict[CategoryName, list[Tuple[int, int]]]
            last_cat: CategoryName = category(chr(0))  # type: ignore
            last_start: int = 0
            for i in range(1, sys.maxunicode + 1):
                cat: CategoryName = category(chr(i))  # type: ignore
                if cat != last_cat:
                    tmp_charmap.setdefault(last_cat, []).append((last_start, i - 1))
                    last_cat, last_start = cat, i
            tmp_charmap.setdefault(last_cat, []).append((last_start, sys.maxunicode))
            try:
                tmpdir: Path = storage_directory('tmp')
                tmpdir.mkdir(exist_ok=True, parents=True)
                (fd, tmpfile) = tempfile.mkstemp(dir=tmpdir)
                os.close(fd)
                with gzip.GzipFile(tmpfile, 'wb', mtime=1) as o:
                    result: str = json.dumps(sorted(tmp_charmap.items()))
                    o.write(result.encode())
                os.renames(tmpfile, str(f))
            except Exception:
                pass
        _charmap = {k: tuple(tuple(pair) for pair in pairs) for (k, pairs) in tmp_charmap.items()}
        for vs in _charmap.values():
            ints = list(sum(vs, ()))
            assert all(isinstance(x, int) for x in ints)
            assert ints == sorted(ints)
            assert all(len(tup) == 2 for tup in vs)
    assert _charmap is not None
    return _charmap


@lru_cache(maxsize=None)
def intervals_from_codec(codec_name: str) -> IntervalSet:
    """Return an IntervalSet of characters which are part of this codec."""
    assert codec_name == codecs.lookup(codec_name).name
    fname: Path = charmap_file(f'codec-{codec_name}')
    try:
        with gzip.GzipFile(fname) as gzf:
            encodable_intervals: IntervalsT = json.load(gzf)
    except Exception:
        encodable_intervals = []  # type: list[Tuple[int, int]]
        for i in range(sys.maxunicode + 1):
            try:
                chr(i).encode(codec_name)
            except Exception:
                pass
            else:
                encodable_intervals.append((i, i))
    res: IntervalSet = IntervalSet(encodable_intervals)
    res = res.union(res)
    try:
        tmpdir: Path = storage_directory('tmp')
        tmpdir.mkdir(exist_ok=True, parents=True)
        (fd, tmpfile) = tempfile.mkstemp(dir=tmpdir)
        os.close(fd)
        with gzip.GzipFile(tmpfile, 'wb', mtime=1) as o:
            o.write(json.dumps(res.intervals).encode())
        os.renames(tmpfile, str(fname))
    except Exception:
        pass
    return res


def categories() -> CategoriesTuple:
    """Return a tuple of Unicode categories in a normalised order.

    >>> categories() # doctest: +ELLIPSIS
    ('Zl', 'Zp', 'Co', 'Me', 'Pc', ..., 'Cc', 'Cs')
    """
    global _categories
    if _categories is None:
        cm: dict[CategoryName, IntervalsT] = charmap()
        cats: list[CategoryName] = sorted(list(cm.keys()), key=lambda c: len(cm[c]))
        cats.remove('Cc')
        cats.remove('Cs')
        cats.append('Cc')
        cats.append('Cs')
        _categories = tuple(cats)
    return _categories


def as_general_categories(cats: Categories, name: str = 'cats') -> CategoriesTuple:
    """Return a tuple of Unicode categories in a normalised order.

    This function expands one-letter designations of a major class to include
    all subclasses.

    >>> as_general_categories(['N'])
    ('Nd', 'Nl', 'No')

    If the collection ``cats`` includes any elements that do not represent a
    major class or a class with subclass, a deprecation warning is raised.
    """
    major_classes: Tuple[str, ...] = ('L', 'M', 'N', 'P', 'S', 'Z', 'C')
    cs: CategoriesTuple = categories()
    out: Set[CategoryName] = set(cats)
    for c in cats:
        if c in major_classes:
            out.discard(c)
            out.update(x for x in cs if x.startswith(c))
        elif c not in cs:
            raise InvalidArgument(f'In {name}={cats!r}, {c!r} is not a valid Unicode category.')
    return tuple(c for c in cs if c in out)


def _category_key(cats: Optional[Iterable[str]]) -> CategoriesTuple:
    """Return a normalised tuple of all Unicode categories that are in
    `cats`.

    If cats is None then default to including all categories.
    """
    cs: CategoriesTuple = categories()
    if cats is None:
        cats_set: Set[CategoryName] = set(cs)
    else:
        cats_set = set(c for c in cats if c in cs)  # type: ignore
    return tuple(c for c in cs if c in cats_set)


def _query_for_key(key: Categories) -> IntervalsT:
    """Return a tuple of codepoint intervals covering characters that match one
    or more categories in the tuple of categories `key`.

    >>> _query_for_key(categories())
    ((0, 1114111),)
    >>> _query_for_key(('Zl', 'Zp', 'Co'))
    ((8232, 8233), (57344, 63743), (983040, 1048573), (1048576, 1114109))
    """
    key_tuple: Tuple[CategoryName, ...] = tuple(key)  # type: ignore
    cache_key: frozenset[CategoryName] = frozenset(key_tuple)
    context = _current_build_context.value
    if context is None or not context.data.provider.avoid_realization:
        try:
            return category_index_cache[cache_key]
        except KeyError:
            pass
    elif not key_tuple:
        return ()
    assert key_tuple
    cs_all: CategoriesTuple = categories()
    if set(key_tuple) == set(cs_all):
        result = IntervalSet([(0, sys.maxunicode)])
    else:
        result = IntervalSet(_query_for_key(key_tuple[:-1])).union(IntervalSet(charmap()[key_tuple[-1]]))
    assert isinstance(result, IntervalSet)
    if context is None or not context.data.provider.avoid_realization:
        category_index_cache[cache_key] = result.intervals
    return result.intervals


def query(*, categories: Optional[Categories] = None, min_codepoint: Optional[int] = None, max_codepoint: Optional[int] = None, include_characters: str = '', exclude_characters: str = '') -> IntervalSet:
    """Return an IntervalSet covering the codepoints for all characters that meet the criteria.

    >>> query()
    ((0, 1114111),)
    >>> query(min_codepoint=0, max_codepoint=128)
    ((0, 128),)
    >>> query(min_codepoint=0, max_codepoint=128, categories=['Lu'])
    ((65, 90),)
    >>> query(min_codepoint=0, max_codepoint=128, categories=['Lu'],
    ...       include_characters='☃')
    ((65, 90), (9731, 9731))
    """
    if min_codepoint is None:
        min_codepoint = 0
    if max_codepoint is None:
        max_codepoint = sys.maxunicode
    catkey: CategoriesTuple = _category_key(categories)
    character_intervals: IntervalSet = IntervalSet.from_string(include_characters or '')
    exclude_intervals: IntervalSet = IntervalSet.from_string(exclude_characters or '')
    qkey: Tuple[CategoriesTuple, int, int, IntervalsT, IntervalsT] = (
        catkey,
        min_codepoint,
        max_codepoint,
        character_intervals.intervals,
        exclude_intervals.intervals,
    )
    context = _current_build_context.value
    if context is None or not context.data.provider.avoid_realization:
        try:
            return limited_category_index_cache[qkey]
        except KeyError:
            pass
    base: IntervalsT = _query_for_key(catkey)
    result_intervals = []
    for (u, v) in base:
        if v >= min_codepoint and u <= max_codepoint:
            result_intervals.append((max(u, min_codepoint), min(v, max_codepoint)))
    result = (IntervalSet(result_intervals) | character_intervals) - exclude_intervals
    if context is None or not context.data.provider.avoid_realization:
        limited_category_index_cache[qkey] = result
    return result
