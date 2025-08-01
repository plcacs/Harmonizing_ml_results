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
from typing import TYPE_CHECKING, Literal, Optional, Dict, Tuple, List, Union
from hypothesis.configuration import storage_directory
from hypothesis.control import _current_build_context
from hypothesis.errors import InvalidArgument
from hypothesis.internal.intervalsets import IntervalSet, IntervalsT

if TYPE_CHECKING:
    from typing import TypeAlias

CategoryName = Literal['L', 'Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'M', 'Mn', 'Mc', 'Me', 'N', 'Nd', 'Nl', 'No', 'P', 'Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po', 'S', 'Sm', 'Sc', 'Sk', 'So', 'Z', 'Zs', 'Zl', 'Zp', 'C', 'Cc', 'Cf', 'Cs', 'Co', 'Cn']
Categories = Iterable[CategoryName]
CategoriesTuple = tuple[CategoryName, ...]

def charmap_file(fname: str = 'charmap') -> Path:
    return storage_directory('unicode_data', unicodedata.unidata_version, f'{fname}.json.gz')

_charmap: Optional[Dict[str, Tuple[Tuple[int, int], ...]]] = None

def charmap() -> Dict[str, Tuple[Tuple[int, int], ...]]:
    global _charmap
    if _charmap is None:
        f = charmap_file()
        try:
            with gzip.GzipFile(f, 'rb') as d:
                tmp_charmap = dict(json.load(d))
        except Exception:
            category = unicodedata.category
            tmp_charmap = {}
            last_cat = category(chr(0))
            last_start = 0
            for i in range(1, sys.maxunicode + 1):
                cat = category(chr(i))
                if cat != last_cat:
                    tmp_charmap.setdefault(last_cat, []).append((last_start, i - 1))
                    last_cat, last_start = (cat, i)
            tmp_charmap.setdefault(last_cat, []).append((last_start, sys.maxunicode))
            try:
                tmpdir = storage_directory('tmp')
                tmpdir.mkdir(exist_ok=True, parents=True)
                fd, tmpfile = tempfile.mkstemp(dir=tmpdir)
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
            assert all((isinstance(x, int) for x in ints))
            assert ints == sorted(ints)
            assert all((len(tup) == 2 for tup in vs))
    assert _charmap is not None
    return _charmap

@lru_cache(maxsize=None)
def intervals_from_codec(codec_name: str) -> IntervalSet:
    assert codec_name == codecs.lookup(codec_name).name
    fname = charmap_file(f'codec-{codec_name}')
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
    res = IntervalSet(encodable_intervals)
    res = res.union(res)
    try:
        tmpdir = storage_directory('tmp')
        tmpdir.mkdir(exist_ok=True, parents=True)
        fd, tmpfile = tempfile.mkstemp(dir=tmpdir)
        os.close(fd)
        with gzip.GzipFile(tmpfile, 'wb', mtime=1) as o:
            o.write(json.dumps(res.intervals).encode())
        os.renames(tmpfile, fname)
    except Exception:
        pass
    return res

_categories: Optional[Tuple[str, ...]] = None

def categories() -> Tuple[str, ...]:
    global _categories
    if _categories is None:
        cm = charmap()
        categories = sorted(cm.keys(), key=lambda c: len(cm[c]))
        categories.remove('Cc')
        categories.remove('Cs')
        categories.append('Cc')
        categories.append('Cs')
        _categories = tuple(categories)
    return _categories

def as_general_categories(cats: Iterable[str], name: str = 'cats') -> Tuple[str, ...]:
    major_classes = ('L', 'M', 'N', 'P', 'S', 'Z', 'C')
    cs = categories()
    out = set(cats)
    for c in cats:
        if c in major_classes:
            out.discard(c)
            out.update((x for x in cs if x.startswith(c)))
        elif c not in cs:
            raise InvalidArgument(f'In {name}={cats!r}, {c!r} is not a valid Unicode category.')
    return tuple((c for c in cs if c in out))

category_index_cache: Dict[frozenset, Tuple[Tuple[int, int], ...]] = {frozenset(): ()}

def _category_key(cats: Optional[Iterable[str]]) -> Tuple[str, ...]:
    cs = categories()
    if cats is None:
        cats = set(cs)
    return tuple((c for c in cs if c in cats))

def _query_for_key(key: Tuple[str, ...]) -> Tuple[Tuple[int, int], ...]:
    key = tuple(key)
    cache_key = frozenset(key)
    context = _current_build_context.value
    if context is None or not context.data.provider.avoid_realization:
        try:
            return category_index_cache[cache_key]
        except KeyError:
            pass
    elif not key:
        return ()
    assert key
    if set(key) == set(categories()):
        result = IntervalSet([(0, sys.maxunicode)])
    else:
        result = IntervalSet(_query_for_key(key[:-1])).union(IntervalSet(charmap()[key[-1]]))
    assert isinstance(result, IntervalSet)
    if context is None or not context.data.provider.avoid_realization:
        category_index_cache[cache_key] = result.intervals
    return result.intervals

limited_category_index_cache: Dict[Tuple[Tuple[str, ...], int, int, Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]], IntervalSet] = {}

def query(*, categories: Optional[Iterable[str]] = None, min_codepoint: Optional[int] = None, max_codepoint: Optional[int] = None, include_characters: str = '', exclude_characters: str = '') -> IntervalSet:
    if min_codepoint is None:
        min_codepoint = 0
    if max_codepoint is None:
        max_codepoint = sys.maxunicode
    catkey = _category_key(categories)
    character_intervals = IntervalSet.from_string(include_characters or '')
    exclude_intervals = IntervalSet.from_string(exclude_characters or '')
    qkey = (catkey, min_codepoint, max_codepoint, character_intervals.intervals, exclude_intervals.intervals)
    context = _current_build_context.value
    if context is None or not context.data.provider.avoid_realization:
        try:
            return limited_category_index_cache[qkey]
        except KeyError:
            pass
    base = _query_for_key(catkey)
    result = []
    for u, v in base:
        if v >= min_codepoint and u <= max_codepoint:
            result.append((max(u, min_codepoint), min(v, max_codepoint)))
    result = (IntervalSet(result) | character_intervals) - exclude_intervals
    if context is None or not context.data.provider.avoid_realization:
        limited_category_index_cache[qkey] = result
    return result
