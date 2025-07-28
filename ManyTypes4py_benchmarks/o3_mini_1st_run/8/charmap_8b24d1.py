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
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Dict, Union, Iterable as IterableType

from hypothesis.configuration import storage_directory
from hypothesis.control import _current_build_context
from hypothesis.errors import InvalidArgument
from hypothesis.internal.intervalsets import IntervalSet, IntervalsT

if TYPE_CHECKING:
    from typing import TypeAlias

CategoryName = Literal[
    'L', 'Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'M', 'Mn', 'Mc', 'Me', 'N', 'Nd', 'Nl', 'No', 'P', 'Pc', 'Pd', 'Ps', 'Pe',
    'Pi', 'Pf', 'Po', 'S', 'Sm', 'Sc', 'Sk', 'So', 'Z', 'Zs', 'Zl', 'Zp', 'C', 'Cc', 'Cf', 'Cs', 'Co', 'Cn'
]
Categories = IterableType[CategoryName]
CategoriesTuple = Tuple[CategoryName, ...]

_charmap: Optional[Dict[CategoryName, Tuple[Tuple[int, int], ...]]] = None
_categories: Optional[CategoriesTuple] = None
category_index_cache: Dict[frozenset, Tuple[Tuple[int, int], ...]] = {frozenset(): ()}
limited_category_index_cache: Dict[
    Tuple[CategoriesTuple, int, int, Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]],
    IntervalSet
] = {}


def charmap_file(fname: str = 'charmap') -> str:
    return storage_directory('unicode_data', unicodedata.unidata_version, f'{fname}.json.gz')


def charmap() -> Dict[CategoryName, Tuple[Tuple[int, int], ...]]:
    global _charmap
    if _charmap is None:
        f: str = charmap_file()
        try:
            with gzip.GzipFile(f, 'rb') as d:
                tmp_charmap: dict = json.load(d)
        except Exception:
            category = unicodedata.category
            tmp_charmap = {}  # type: Dict[CategoryName, list]
            last_cat: CategoryName = category(chr(0))  # type: ignore
            last_start = 0
            for i in range(1, sys.maxunicode + 1):
                cat: CategoryName = category(chr(i))  # type: ignore
                if cat != last_cat:
                    tmp_charmap.setdefault(last_cat, []).append((last_start, i - 1))
                    last_cat, last_start = (cat, i)
            tmp_charmap.setdefault(last_cat, []).append((last_start, sys.maxunicode))
            try:
                tmpdir: Path = storage_directory('tmp')
                tmpdir.mkdir(exist_ok=True, parents=True)
                fd, tmpfile = tempfile.mkstemp(dir=str(tmpdir))
                os.close(fd)
                with gzip.GzipFile(tmpfile, 'wb', mtime=1) as o:
                    result = json.dumps(sorted(tmp_charmap.items()))
                    o.write(result.encode())
                os.renames(tmpfile, f)
            except Exception:
                pass
        _charmap = {k: tuple((tuple(pair) for pair in pairs)) for k, pairs in tmp_charmap.items()}
        for vs in _charmap.values():
            ints = list(sum(vs, ()))
            assert all(isinstance(x, int) for x in ints)
            assert ints == sorted(ints)
            assert all(len(tup) == 2 for tup in vs)
    assert _charmap is not None
    return _charmap


@lru_cache(maxsize=None)
def intervals_from_codec(codec_name: str) -> IntervalSet:
    assert codec_name == codecs.lookup(codec_name).name
    fname: str = charmap_file(f'codec-{codec_name}')
    try:
        with gzip.GzipFile(fname) as gzf:
            encodable_intervals = json.load(gzf)
    except Exception:
        encodable_intervals = []
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
        fd, tmpfile = tempfile.mkstemp(dir=str(tmpdir))
        os.close(fd)
        with gzip.GzipFile(tmpfile, 'wb', mtime=1) as o:
            o.write(json.dumps(res.intervals).encode())
        os.renames(tmpfile, fname)
    except Exception:
        pass
    return res


def categories() -> CategoriesTuple:
    global _categories
    if _categories is None:
        cm = charmap()
        cats = sorted(cm.keys(), key=lambda c: len(cm[c]))  # type: list[CategoryName]
        cats.remove('Cc')  # type: ignore
        cats.remove('Cs')  # type: ignore
        cats.append('Cc')  # type: ignore
        cats.append('Cs')  # type: ignore
        _categories = tuple(cats)
    return _categories


def as_general_categories(cats: Iterable[str], name: str = 'cats') -> Tuple[CategoryName, ...]:
    major_classes: Tuple[str, ...] = ('L', 'M', 'N', 'P', 'S', 'Z', 'C')
    cs: CategoriesTuple = categories()
    out = set(cats)
    for c in cats:
        if c in major_classes:
            out.discard(c)
            out.update(x for x in cs if x.startswith(c))
        elif c not in cs:
            raise InvalidArgument(f'In {name}={cats!r}, {c!r} is not a valid Unicode category.')
    return tuple(c for c in cs if c in out)


def _category_key(cats: Optional[Iterable[str]]) -> Tuple[CategoryName, ...]:
    cs: CategoriesTuple = categories()
    if cats is None:
        cats_set = set(cs)
    else:
        cats_set = set(cats)
    return tuple(c for c in cs if c in cats_set)


def _query_for_key(key: Iterable[CategoryName]) -> Tuple[Tuple[int, int], ...]:
    key_tuple = tuple(key)
    cache_key = frozenset(key_tuple)
    context = _current_build_context.value
    if context is None or not context.data.provider.avoid_realization:
        try:
            return category_index_cache[cache_key]
        except KeyError:
            pass
    elif not key_tuple:
        return ()
    assert key_tuple
    if set(key_tuple) == set(categories()):
        result = IntervalSet([(0, sys.maxunicode)])
    else:
        result = IntervalSet(_query_for_key(key_tuple[:-1])).union(IntervalSet(charmap()[key_tuple[-1]]))
    assert isinstance(result, IntervalSet)
    if context is None or not context.data.provider.avoid_realization:
        category_index_cache[cache_key] = result.intervals
    return result.intervals


def query(*,
          categories: Optional[Iterable[str]] = None,
          min_codepoint: Optional[int] = None,
          max_codepoint: Optional[int] = None,
          include_characters: str = '',
          exclude_characters: str = '') -> IntervalSet:
    if min_codepoint is None:
        min_codepoint = 0
    if max_codepoint is None:
        max_codepoint = sys.maxunicode
    catkey: Tuple[CategoryName, ...] = _category_key(categories)
    character_intervals: IntervalSet = IntervalSet.from_string(include_characters or '')
    exclude_intervals: IntervalSet = IntervalSet.from_string(exclude_characters or '')
    qkey = (catkey, min_codepoint, max_codepoint, character_intervals.intervals, exclude_intervals.intervals)
    context = _current_build_context.value
    if context is None or not context.data.provider.avoid_realization:
        try:
            return limited_category_index_cache[qkey]
        except KeyError:
            pass
    base_intervals: Tuple[Tuple[int, int], ...] = _query_for_key(catkey)
    result_intervals = []
    for u, v in base_intervals:
        if v >= min_codepoint and u <= max_codepoint:
            result_intervals.append((max(u, min_codepoint), min(v, max_codepoint)))
    result: IntervalSet = (IntervalSet(result_intervals) | character_intervals) - exclude_intervals
    if context is None or not context.data.provider.avoid_realization:
        limited_category_index_cache[qkey] = result
    return result