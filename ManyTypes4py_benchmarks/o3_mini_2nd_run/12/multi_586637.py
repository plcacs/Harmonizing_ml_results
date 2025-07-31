#!/usr/bin/env python
from __future__ import annotations
from collections.abc import Callable, Collection, Generator, Hashable, Iterable, Sequence
from functools import wraps
from sys import getsizeof
from typing import (Any, cast, List, Optional, Sequence as Seq, Tuple, Type,
                    TypeVar, Union, overload)
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import algos as libalgos, index as libindex, lib
from pandas._libs.hashtable import duplicated
from pandas._typing import AnyAll, AnyArrayLike, Axis, DropKeep, DtypeObj, F, IgnoreRaise, IndexLabel, IndexT, Scalar, Self, Shape, npt
from pandas.compat.numpy import function as nv
from pandas.errors import InvalidIndexError, PerformanceWarning, UnsortedIndexError
from pandas.util._decorators import Appender, cache_readonly, doc, set_module
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (ensure_int64, ensure_platform_int, is_hashable,
                                       is_integer, is_iterator, is_list_like,
                                       is_object_dtype, is_scalar, is_string_dtype, pandas_dtype)
from pandas.core.dtypes.dtypes import CategoricalDtype, ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import array_equivalent, isna
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import Categorical, ExtensionArray
from pandas.core.arrays.categorical import factorize_from_iterables, recode_for_categories
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import Index, _index_shared_docs, ensure_index, get_unanimous_names
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import get_group_index, lexsort_indexer
from pandas.io.formats.printing import pprint_thing
if False:  # TYPE_CHECKING placeholder
    from pandas import CategoricalIndex, DataFrame, Series

_index_doc_kwargs: dict[str, Any] = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({'klass': 'MultiIndex', 'target_klass': 'MultiIndex or list of tuples'})

T = TypeVar("T", bound="MultiIndex")

class MultiIndexUInt64Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt64Engine):
    """
    Manages a MultiIndex by mapping label combinations to positive integers.

    The number of possible label combinations must not overflow the 64 bits integers.
    """
    _base = libindex.UInt64Engine
    _codes_dtype = 'uint64'

class MultiIndexUInt32Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt32Engine):
    """
    Manages a MultiIndex by mapping label combinations to positive integers.

    The number of possible label combinations must not overflow the 32 bits integers.
    """
    _base = libindex.UInt32Engine
    _codes_dtype = 'uint32'

class MultiIndexUInt16Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt16Engine):
    """
    Manages a MultiIndex by mapping label combinations to positive integers.

    The number of possible label combinations must not overflow the 16 bits integers.
    """
    _base = libindex.UInt16Engine
    _codes_dtype = 'uint16'

class MultiIndexUInt8Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt8Engine):
    """
    Manages a MultiIndex by mapping label combinations to positive integers.

    The number of possible label combinations must not overflow the 8 bits integers.
    """
    _base = libindex.UInt8Engine
    _codes_dtype = 'uint8'

class MultiIndexPyIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.ObjectEngine):
    """
    Manages a MultiIndex by mapping label combinations to positive integers.

    This class manages those (extreme) cases in which the number of possible
    label combinations overflows the 64 bits integers, and uses an ObjectEngine
    containing Python integers.
    """
    _base = libindex.ObjectEngine
    _codes_dtype = 'object'

def names_compat(meth: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator to allow either `name` or `names` keyword but not both.

    This makes it easier to share code with base class.
    """
    @wraps(meth)
    def new_meth(self_or_cls: Any, *args: Any, **kwargs: Any) -> Any:
        if 'name' in kwargs and 'names' in kwargs:
            raise TypeError('Can only provide one of `names` and `name`')
        if 'name' in kwargs:
            kwargs['names'] = kwargs.pop('name')
        return meth(self_or_cls, *args, **kwargs)
    return cast(Callable[..., Any], new_meth)

@set_module('pandas')
class MultiIndex(Index):
    """
    A multi-level, or hierarchical, index object for pandas objects.
    """
    _hidden_attrs = Index._hidden_attrs | frozenset()
    _typ: str = 'multiindex'
    _names: List[Optional[Hashable]] = []
    _levels: FrozenList = FrozenList()
    _codes: FrozenList = FrozenList()
    _comparables: List[str] = ['names']

    def __new__(cls: Type[T],
                levels: Optional[Seq[Any]] = None,
                codes: Optional[Seq[Any]] = None,
                sortorder: Optional[int] = None,
                names: Any = None,
                dtype: Any = None,
                copy: bool = False,
                name: Any = None,
                verify_integrity: bool = True) -> T:
        if name is not None:
            names = name
        if levels is None or codes is None:
            raise TypeError('Must pass both levels and codes')
        if len(levels) != len(codes):
            raise ValueError('Length of levels and codes must be the same.')
        if len(levels) == 0:
            raise ValueError('Must pass non-zero number of levels/codes')
        result: MultiIndex = object.__new__(cls)
        result._cache = {}
        result._set_levels(levels, copy=copy, validate=False)
        result._set_codes(codes, copy=copy, validate=False)
        result._names = [None] * len(levels)
        if names is not None:
            result._set_names(names)
        if sortorder is not None:
            result.sortorder = int(sortorder)
        else:
            result.sortorder = sortorder
        if verify_integrity:
            new_codes = result._verify_integrity()
            result._codes = new_codes
        result._reset_identity()
        result._references = None
        return result

    def _validate_codes(self, level: int, code: Any) -> Any:
        """
        Reassign code values as -1 if their corresponding levels are NaN.
        """
        null_mask = isna(self.levels[level])
        if np.any(null_mask):
            code = np.where(null_mask[code], -1, code)
        return code

    def _verify_integrity(self,
                          codes: Optional[Seq[Any]] = None,
                          levels: Optional[Seq[Any]] = None,
                          levels_to_verify: Optional[Seq[int]] = None) -> FrozenList:
        """
        Verify the integrity of levels and codes.
        """
        codes = codes or self.codes
        levels = levels or self.levels
        if levels_to_verify is None:
            levels_to_verify = list(range(len(levels)))
        if len(levels) != len(codes):
            raise ValueError('Length of levels and codes must match. NOTE: this index is in an inconsistent state.')
        codes_length = len(codes[0])
        for i in levels_to_verify:
            level = levels[i]
            level_codes = codes[i]
            if len(level_codes) != codes_length:
                raise ValueError(f'Unequal code lengths: {[len(code_) for code_ in codes]}')
            if len(level_codes) and level_codes.max() >= len(level):
                raise ValueError(f'On level {i}, code max ({level_codes.max()}) >= length of level ({len(level)}). NOTE: this index is in an inconsistent state')
            if len(level_codes) and level_codes.min() < -1:
                raise ValueError(f'On level {i}, code value ({level_codes.min()}) < -1')
            if not level.is_unique:
                raise ValueError(f'Level values must be unique: {list(level)} on level {i}')
        if self.sortorder is not None:
            if self.sortorder > _lexsort_depth(self.codes, self.nlevels):
                raise ValueError(f'Value for sortorder must be inferior or equal to actual lexsort_depth: sortorder {self.sortorder} with lexsort_depth {_lexsort_depth(self.codes, self.nlevels)}')
        result_codes: List[Any] = []
        for i in range(len(levels)):
            if i in levels_to_verify:
                result_codes.append(self._validate_codes(i, codes[i]))
            else:
                result_codes.append(codes[i])
        new_codes = FrozenList(result_codes)
        return new_codes

    @classmethod
    def from_arrays(cls: Type[T],
                    arrays: Any,
                    sortorder: Optional[int] = None,
                    names: Any = lib.no_default) -> T:
        error_msg = 'Input must be a list / sequence of array-likes.'
        if not is_list_like(arrays):
            raise TypeError(error_msg)
        if is_iterator(arrays):
            arrays = list(arrays)
        for array in arrays:
            if not is_list_like(array):
                raise TypeError(error_msg)
        for i in range(1, len(arrays)):
            if len(arrays[i]) != len(arrays[i - 1]):
                raise ValueError('all arrays must be same length')
        codes, levels = factorize_from_iterables(arrays)
        if names is lib.no_default:
            names = [getattr(arr, 'name', None) for arr in arrays]
        return cls(levels=levels, codes=codes, sortorder=sortorder, names=names, verify_integrity=False)

    @classmethod
    @names_compat
    def from_tuples(cls: Type[T],
                    tuples: Any,
                    sortorder: Optional[int] = None,
                    names: Any = None) -> T:
        if not is_list_like(tuples):
            raise TypeError('Input must be a list / sequence of tuple-likes.')
        if is_iterator(tuples):
            tuples = list(tuples)
        tuples = cast(Collection[tuple[Hashable, ...]], tuples)
        if len(tuples) and all((isinstance(e, tuple) and (not e) for e in tuples)):
            codes = [np.zeros(len(tuples))]
            levels = [Index(com.asarray_tuplesafe(tuples, dtype=np.dtype('object')))]
            return cls(levels=levels, codes=codes, sortorder=sortorder, names=names, verify_integrity=False)
        if len(tuples) == 0:
            if names is None:
                raise TypeError('Cannot infer number of levels from empty list')
            arrays = [[]] * len(names)
        elif isinstance(tuples, (np.ndarray, Index)):
            if isinstance(tuples, Index):
                tuples = np.asarray(tuples._values)
            arrays = list(lib.tuples_to_object_array(tuples).T)
        elif isinstance(tuples, list):
            arrays = list(lib.to_object_array_tuples(tuples).T)
        else:
            arrs = zip(*tuples)
            arrays = cast(list[Sequence[Hashable]], list(arrs))
        return cls.from_arrays(arrays, sortorder=sortorder, names=names)

    @classmethod
    def from_product(cls: Type[T],
                     iterables: Any,
                     sortorder: Optional[int] = None,
                     names: Any = lib.no_default) -> T:
        if not is_list_like(iterables):
            raise TypeError('Input must be a list / sequence of iterables.')
        if is_iterator(iterables):
            iterables = list(iterables)
        codes, levels = factorize_from_iterables(iterables)
        if names is lib.no_default:
            names = [getattr(it, 'name', None) for it in iterables]
        codes = cartesian_product(codes)
        return cls(levels, codes, sortorder=sortorder, names=names)

    @classmethod
    def from_frame(cls: Type[T],
                   df: Any,
                   sortorder: Optional[int] = None,
                   names: Any = None) -> T:
        from pandas import DataFrame
        if not isinstance(df, ABCDataFrame):
            raise TypeError('Input must be a DataFrame')
        column_names, columns = zip(*df.items())
        names = column_names if names is None else names
        return cls.from_arrays(columns, sortorder=sortorder, names=names)

    @cache_readonly
    def _values(self) -> np.ndarray:
        values: List[np.ndarray] = []
        for i in range(self.nlevels):
            index = self.levels[i]
            codes = self.codes[i]
            vals = index
            if isinstance(vals.dtype, CategoricalDtype):
                vals = cast('CategoricalIndex', vals)
                vals = vals._data._internal_get_values()
            if isinstance(vals.dtype, ExtensionDtype) or lib.is_np_dtype(vals.dtype, 'mM'):
                vals = vals.astype(object)
            array_vals = np.asarray(vals)
            array_vals = algos.take_nd(array_vals, codes, fill_value=index._na_value)
            values.append(array_vals)
        arr = lib.fast_zip(values)
        return arr

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def array(self) -> None:
        raise ValueError("MultiIndex has no single backing array. Use 'MultiIndex.to_numpy()' to get a NumPy array of tuples.")

    @cache_readonly
    def dtypes(self) -> Any:
        from pandas import Series
        names = com.fill_missing_names(self.names)
        return Series([level.dtype for level in self.levels], index=Index(names))

    def __len__(self) -> int:
        return len(self.codes[0])

    @property
    def size(self) -> int:
        return len(self)

    @cache_readonly
    def levels(self) -> FrozenList:
        result = [x._rename(name=name) for x, name in zip(self._levels, self._names)]
        for level in result:
            level._no_setting_name = True
        return FrozenList(result)

    def _set_levels(self, levels: Any, *, level: Optional[Any] = None, copy: bool = False, validate: bool = True, verify_integrity: bool = False) -> None:
        if validate:
            if len(levels) == 0:
                raise ValueError('Must set non-zero number of levels.')
            if level is None and len(levels) != self.nlevels:
                raise ValueError('Length of levels must match number of levels.')
            if level is not None and len(levels) != len(level):
                raise ValueError('Length of levels must match length of level.')
        if level is None:
            new_levels = FrozenList((ensure_index(lev, copy=copy)._view() for lev in levels))
            level_numbers = list(range(len(new_levels)))
        else:
            level_numbers = [self._get_level_number(lev) for lev in level]
            new_levels_list = list(self._levels)
            for lev_num, lev in zip(level_numbers, levels):
                new_levels_list[lev_num] = ensure_index(lev, copy=copy)._view()
            new_levels = FrozenList(new_levels_list)
        if verify_integrity:
            new_codes = self._verify_integrity(levels=new_levels, levels_to_verify=level_numbers)
            self._codes = new_codes
        names = self.names
        self._levels = new_levels
        if any(names):
            self._set_names(names)
        self._reset_cache()

    def set_levels(self, levels: Any, *, level: Optional[Any] = None, verify_integrity: bool = True) -> MultiIndex:
        if isinstance(levels, Index):
            pass
        elif is_array_like(levels):
            levels = Index(levels)
        elif is_list_like(levels):
            levels = list(levels)
        level, levels = _require_listlike(level, levels, 'Levels')
        idx = self._view()
        idx._reset_identity()
        idx._set_levels(levels, level=level, validate=True, verify_integrity=verify_integrity)
        return idx

    @property
    def nlevels(self) -> int:
        return len(self._levels)

    @property
    def levshape(self) -> Tuple[int, ...]:
        return tuple((len(x) for x in self.levels))

    @property
    def codes(self) -> FrozenList:
        return self._codes

    def _set_codes(self, codes: Any, *, level: Optional[Any] = None, copy: bool = False, validate: bool = True, verify_integrity: bool = False) -> None:
        if validate:
            if level is None and len(codes) != self.nlevels:
                raise ValueError('Length of codes must match number of levels')
            if level is not None and len(codes) != len(level):
                raise ValueError('Length of codes must match length of levels.')
        if level is None:
            new_codes = FrozenList((_coerce_indexer_frozen(level_codes, lev, copy=copy).view() for lev, level_codes in zip(self._levels, codes)))
            level_numbers = list(range(len(new_codes)))
        else:
            level_numbers = [self._get_level_number(lev) for lev in level]
            new_codes_list = list(self._codes)
            for lev_num, level_codes in zip(level_numbers, codes):
                lev = self.levels[lev_num]
                new_codes_list[lev_num] = _coerce_indexer_frozen(level_codes, lev, copy=copy)
            new_codes = FrozenList(new_codes_list)
        if verify_integrity:
            new_codes = self._verify_integrity(codes=new_codes, levels_to_verify=level_numbers)
        self._codes = new_codes
        self._reset_cache()

    def set_codes(self, codes: Any, *, level: Optional[Any] = None, verify_integrity: bool = True) -> MultiIndex:
        level, codes = _require_listlike(level, codes, 'Codes')
        idx = self._view()
        idx._reset_identity()
        idx._set_codes(codes, level=level, verify_integrity=verify_integrity)
        return idx

    @cache_readonly
    def _engine(self) -> Any:
        sizes = np.ceil(np.log2([len(level) + libindex.multiindex_nulls_shift for level in self.levels]))
        lev_bits = np.cumsum(sizes[::-1])[::-1]
        offsets = np.concatenate([lev_bits[1:], [0]])
        offsets = offsets.astype(np.min_scalar_type(int(offsets[0])))
        if lev_bits[0] > 64:
            return MultiIndexPyIntEngine(self.levels, self.codes, offsets)
        if lev_bits[0] > 32:
            return MultiIndexUInt64Engine(self.levels, self.codes, offsets)
        if lev_bits[0] > 16:
            return MultiIndexUInt32Engine(self.levels, self.codes, offsets)
        if lev_bits[0] > 8:
            return MultiIndexUInt16Engine(self.levels, self.codes, offsets)
        return MultiIndexUInt8Engine(self.levels, self.codes, offsets)

    @property
    def _constructor(self) -> Callable[..., MultiIndex]:
        return type(self).from_tuples

    @doc(Index._shallow_copy)
    def _shallow_copy(self, values: Any, name: Any = lib.no_default) -> MultiIndex:
        names = name if name is not lib.no_default else self.names
        return type(self).from_tuples(values, sortorder=None, names=names)

    def _view(self) -> MultiIndex:
        result = type(self)(levels=self.levels, codes=self.codes, sortorder=self.sortorder, names=self.names, verify_integrity=False)
        result._cache = self._cache.copy()
        result._reset_cache('levels')
        return result

    def copy(self, names: Optional[Any] = None, deep: bool = False, name: Any = None) -> MultiIndex:
        names = self._validate_names(name=name, names=names, deep=deep)
        keep_id = not deep
        levels, codes = (None, None)
        if deep:
            from copy import deepcopy
            levels = deepcopy(self.levels)
            codes = deepcopy(self.codes)
        levels = levels if levels is not None else self.levels
        codes = codes if codes is not None else self.codes
        new_index = type(self)(levels=levels, codes=codes, sortorder=self.sortorder, names=names, verify_integrity=False)
        new_index._cache = self._cache.copy()
        new_index._reset_cache('levels')
        if keep_id:
            new_index._id = self._id
        return new_index

    def __array__(self, dtype: Optional[Any] = None, copy: Optional[bool] = None) -> np.ndarray:
        if copy is False:
            raise ValueError('Unable to avoid copy while creating an array as requested.')
        if copy is True:
            return np.array(self.values, dtype=dtype)
        return self.values

    def view(self, cls: Optional[Type[Any]] = None) -> MultiIndex:
        result = self.copy()
        result._id = self._id
        return result

    @doc(Index.__contains__)
    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            self.get_loc(key)
            return True
        except (LookupError, TypeError, ValueError):
            return False

    @cache_readonly
    def dtype(self) -> np.dtype:
        return np.dtype('O')

    @cache_readonly
    def _is_memory_usage_qualified(self) -> bool:
        def f(dtype: Any) -> bool:
            return is_object_dtype(dtype) or (is_string_dtype(dtype) and getattr(dtype, "storage", None) == 'python')
        return any((f(level.dtype) for level in self.levels))

    @doc(Index.memory_usage)
    def memory_usage(self, deep: bool = False) -> int:
        return self._nbytes(deep)

    @cache_readonly
    def nbytes(self) -> int:
        return self._nbytes(False)

    def _nbytes(self, deep: bool = False) -> int:
        objsize = 24
        level_nbytes = sum((i.memory_usage(deep=deep) for i in self.levels))
        label_nbytes = sum((i.nbytes for i in self.codes))
        names_nbytes = sum((getsizeof(i, objsize) for i in self.names))
        result = level_nbytes + label_nbytes + names_nbytes
        if '_engine' in self._cache:
            result += self._engine.sizeof(deep=deep)
        return result

    def _formatter_func(self, tup: Tuple[Any, ...]) -> Tuple[Any, ...]:
        formatter_funcs = (level._formatter_func for level in self.levels)
        return tuple((func(val) for func, val in zip(formatter_funcs, tup)))

    def _get_values_for_csv(self, *, na_rep: str = 'nan', **kwargs: Any) -> Any:
        new_levels: List[Any] = []
        new_codes: List[Any] = []
        for level, level_codes in zip(self.levels, self.codes):
            level_strs = level._get_values_for_csv(na_rep=na_rep, **kwargs)
            mask = level_codes == -1
            if mask.any():
                nan_index = len(level_strs)
                level_strs = level_strs.astype(str)
                level_strs = np.append(level_strs, na_rep)
                assert not level_codes.flags.writeable
                level_codes = level_codes.copy()
                level_codes[mask] = nan_index
            new_levels.append(level_strs)
            new_codes.append(level_codes)
        if len(new_levels) == 1:
            return Index(new_levels[0].take(new_codes[0]))._get_values_for_csv()
        else:
            mi = MultiIndex(levels=new_levels, codes=new_codes, names=self.names, sortorder=self.sortorder, verify_integrity=False)
            return mi._values

    def _format_multi(self, *, include_names: bool, sparsify: Optional[Union[bool, str]] = None, formatter: Optional[Callable[[Any], str]] = None) -> List[Any]:
        if len(self) == 0:
            return []
        stringified_levels: List[List[Any]] = []
        for lev, level_codes in zip(self.levels, self.codes):
            na = _get_na_rep(lev.dtype)
            if len(lev) > 0:
                taken = lev.take(level_codes)
                formatted = taken._format_flat(include_name=False, formatter=formatter)
                mask = level_codes == -1
                if mask.any():
                    formatted = np.array(formatted, dtype=object)
                    formatted[mask] = na
                    formatted = formatted.tolist()
            else:
                formatted = [pprint_thing(na if isna(x) else x, escape_chars=('\t', '\r', '\n')) for x in algos.take_nd(lev._values, level_codes)]
            stringified_levels.append(formatted)
        result_levels: List[List[Any]] = []
        for lev, lev_name in zip(stringified_levels, self.names):
            level: List[Any] = []
            if include_names:
                level.append(pprint_thing(lev_name, escape_chars=('\t', '\r', '\n')) if lev_name is not None else '')
            level.extend(np.array(lev, dtype=object))
            result_levels.append(level)
        if sparsify is None:
            sparsify = get_option('display.multi_sparse')
        if sparsify:
            sentinel = ''
            assert isinstance(sparsify, bool) or sparsify is lib.no_default
            if sparsify is lib.no_default:
                sentinel = sparsify
            result_levels = sparsify_labels(result_levels, start=int(include_names), sentinel=sentinel)
        return result_levels

    def _get_names(self) -> FrozenList:
        return FrozenList(self._names)

    def _set_names(self, names: Any, *, level: Optional[Any] = None) -> None:
        if names is not None and (not is_list_like(names)):
            raise ValueError('Names should be list-like for a MultiIndex')
        names_list = list(names)
        if level is not None and len(names_list) != len(level):
            raise ValueError('Length of names must match length of level.')
        if level is None and len(names_list) != self.nlevels:
            raise ValueError('Length of names must match number of levels in MultiIndex.')
        if level is None:
            level = range(self.nlevels)
        else:
            level = [self._get_level_number(lev) for lev in level]
        for lev, name in zip(level, names_list):
            if name is not None:
                if not is_hashable(name):
                    raise TypeError(f'{type(self).__name__}.name must be a hashable type')
            self._names[lev] = name
        self._reset_cache('levels')
    names = property(fset=_set_names, fget=_get_names, doc="\n        Names of levels in MultiIndex.\n\n        This attribute provides access to the names of the levels in a `MultiIndex`.\n        The names are stored as a `FrozenList`, which is an immutable list-like\n        container. Each name corresponds to a level in the `MultiIndex`, and can be\n        used to identify or manipulate the levels individually.\n\n        See Also\n        --------\n        MultiIndex.set_names : Set Index or MultiIndex name.\n        MultiIndex.rename : Rename specific levels in a MultiIndex.\n        Index.names : Get names on index.\n\n        Examples\n        --------\n        >>> mi = pd.MultiIndex.from_arrays(\n        ...     [[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'z']\n        ... )\n        >>> mi\n        MultiIndex([('1', '3', '5'),\n                    ('2', '4', '6')],\n                   names=['x', 'y', 'z'])\n        >>> mi.names\n        FrozenList(['x', 'y', 'z'])\n        ")

    @cache_readonly
    def inferred_type(self) -> str:
        return 'mixed'

    def _get_level_values(self, level: int, unique: bool = False) -> Index:
        lev = self.levels[level]
        level_codes = self.codes[level]
        name = self._names[level]
        if unique:
            level_codes = algos.unique(level_codes)
        filled = algos.take_nd(lev._values, level_codes, fill_value=lev._na_value)
        return lev._shallow_copy(filled, name=name)

    def get_level_values(self, level: Union[int, str]) -> Index:
        level_num = self._get_level_number(level)
        values = self._get_level_values(level_num)
        return values

    @doc(Index.unique)
    def unique(self, level: Optional[Union[int, str]] = None) -> Any:
        if level is None:
            return self.drop_duplicates()
        else:
            level_num = self._get_level_number(level)
            return self._get_level_values(level=level_num, unique=True)

    def to_frame(self, index: bool = True, name: Any = lib.no_default, allow_duplicates: bool = False) -> Any:
        from pandas import DataFrame
        if name is not lib.no_default:
            if not is_list_like(name):
                raise TypeError("'name' must be a list / sequence of column names.")
            if len(name) != len(self.levels):
                raise ValueError("'name' should have same length as number of levels on index.")
            idx_names = name
        else:
            idx_names = self._get_level_names()
        if not allow_duplicates and len(set(idx_names)) != len(idx_names):
            raise ValueError('Cannot create duplicate column labels if allow_duplicates is False')
        result = DataFrame({level: self._get_level_values(level) for level in range(len(self.levels))}, copy=False)
        result.columns = idx_names
        if index:
            result.index = self
        return result

    def to_flat_index(self) -> Index:
        return Index(self._values, tupleize_cols=False)

    def _is_lexsorted(self) -> bool:
        return self._lexsort_depth == self.nlevels

    @cache_readonly
    def _lexsort_depth(self) -> int:
        if self.sortorder is not None:
            return self.sortorder
        return _lexsort_depth(self.codes, self.nlevels)

    def _sort_levels_monotonic(self, raise_if_incomparable: bool = False) -> MultiIndex:
        if self._is_lexsorted() and self.is_monotonic_increasing:
            return self
        new_levels: List[Any] = []
        new_codes: List[Any] = []
        for lev, level_codes in zip(self.levels, self.codes):
            if not lev.is_monotonic_increasing:
                try:
                    indexer = lev.argsort()
                except TypeError:
                    if raise_if_incomparable:
                        raise
                else:
                    lev = lev.take(indexer)
                    indexer = ensure_platform_int(indexer)
                    ri = lib.get_reverse_indexer(indexer, len(indexer))
                    level_codes = algos.take_nd(ri, level_codes, fill_value=-1)
            new_levels.append(lev)
            new_codes.append(level_codes)
        return MultiIndex(new_levels, new_codes, names=self.names, sortorder=self.sortorder, verify_integrity=False)

    def remove_unused_levels(self) -> MultiIndex:
        new_levels: List[Any] = []
        new_codes: List[Any] = []
        changed = False
        for lev, level_codes in zip(self.levels, self.codes):
            uniques = np.where(np.bincount(level_codes + 1) > 0)[0] - 1
            has_na = int(len(uniques) and uniques[0] == -1)
            if len(uniques) != len(lev) + has_na:
                if lev.isna().any() and len(uniques) == len(lev):
                    break
                changed = True
                uniques = algos.unique(level_codes)
                if has_na:
                    na_idx = np.where(uniques == -1)[0]
                    uniques[[0, na_idx[0]]] = uniques[[na_idx[0], 0]]
                code_mapping = np.zeros(len(lev) + has_na)
                code_mapping[uniques] = np.arange(len(uniques)) - has_na
                level_codes = code_mapping[level_codes]
                lev = lev.take(uniques[has_na:])
            new_levels.append(lev)
            new_codes.append(level_codes)
        result = self.view()
        if changed:
            result._reset_identity()
            result._set_levels(new_levels, validate=False)
            result._set_codes(new_codes, validate=False)
        return result

    def __reduce__(self) -> Tuple[Any, Tuple[Any, ...], None]:
        d = {'levels': list(self.levels), 'codes': list(self.codes), 'sortorder': self.sortorder, 'names': list(self.names)}
        return (ibase._new_Index, (type(self), d), None)

    def __getitem__(self, key: Any) -> Union[tuple, MultiIndex]:
        if is_scalar(key):
            key = com.cast_scalar_indexer(key)
            retval = []
            for lev, level_codes in zip(self.levels, self.codes):
                if level_codes[key] == -1:
                    retval.append(np.nan)
                else:
                    retval.append(lev[level_codes[key]])
            return tuple(retval)
        else:
            sortorder = None
            if com.is_bool_indexer(key):
                key = np.asarray(key, dtype=bool)
                sortorder = self.sortorder
            elif isinstance(key, slice):
                if key.step is None or key.step > 0:
                    sortorder = self.sortorder
            elif isinstance(key, Index):
                key = np.asarray(key)
            new_codes = [level_codes[key] for level_codes in self.codes]
            return MultiIndex(levels=self.levels, codes=new_codes, names=self.names, sortorder=sortorder, verify_integrity=False)

    def _getitem_slice(self, slobj: slice) -> MultiIndex:
        sortorder = None
        if slobj.step is None or slobj.step > 0:
            sortorder = self.sortorder
        new_codes = [level_codes[slobj] for level_codes in self.codes]
        return type(self)(levels=self.levels, codes=new_codes, names=self._names, sortorder=sortorder, verify_integrity=False)

    @Appender(_index_shared_docs['take'] % _index_doc_kwargs)
    def take(self, indices: np.ndarray, axis: int = 0, allow_fill: bool = True, fill_value: Optional[Any] = None, **kwargs: Any) -> MultiIndex:
        nv.validate_take((), kwargs)
        indices = ensure_platform_int(indices)
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)
        if indices.ndim == 1 and lib.is_range_indexer(indices, len(self)):
            return self.copy()
        na_value = -1
        taken = [lab.take(indices) for lab in self.codes]
        if allow_fill:
            mask = indices == -1
            if mask.any():
                masked = []
                for new_label in taken:
                    label_values = new_label
                    label_values[mask] = na_value
                    masked.append(np.asarray(label_values))
                taken = masked
        return MultiIndex(levels=self.levels, codes=taken, names=self.names, verify_integrity=False)

    def append(self, other: Union[Index, List[Index]]) -> MultiIndex:
        if not isinstance(other, (list, tuple)):
            other = [other]
        if all((isinstance(o, MultiIndex) and o.nlevels >= self.nlevels for o in other)):
            codes: List[Any] = []
            levels: List[Any] = []
            names: List[Any] = []
            for i in range(self.nlevels):
                level_values = self.levels[i]
                for mi in other:
                    level_values = level_values.union(mi.levels[i])
                level_codes = [recode_for_categories(mi.codes[i], mi.levels[i], level_values, copy=False) for mi in [self, *other]]
                level_name = self.names[i]
                if any((mi.names[i] != level_name for mi in other)):
                    level_name = None
                codes.append(np.concatenate(level_codes))
                levels.append(level_values)
                names.append(level_name)
            return MultiIndex(codes=codes, levels=levels, names=names, verify_integrity=False)
        to_concat = (self._values,) + tuple((k._values for k in other))
        new_tuples = np.concatenate(to_concat)
        try:
            return MultiIndex.from_tuples(new_tuples)
        except (TypeError, IndexError):
            return Index(new_tuples)

    def argsort(self, *args: Any, na_position: str = 'last', **kwargs: Any) -> np.ndarray:
        target = self._sort_levels_monotonic(raise_if_incomparable=True)
        keys = [lev.codes for lev in target._get_codes_for_sorting()]
        return lexsort_indexer(keys, na_position=na_position, codes_given=True)

    @Appender(_index_shared_docs['repeat'] % _index_doc_kwargs)
    def repeat(self, repeats: int, axis: Optional[int] = None) -> MultiIndex:
        nv.validate_repeat((), {'axis': axis})
        repeats = ensure_platform_int(repeats)
        return MultiIndex(levels=self.levels, codes=[level_codes.view(np.ndarray).astype(np.intp, copy=False).repeat(repeats) for level_codes in self.codes], names=self.names, sortorder=self.sortorder, verify_integrity=False)

    def drop(self, codes: Any, level: Optional[Union[int, str]] = None, errors: str = 'raise') -> MultiIndex:
        if level is not None:
            return self._drop_from_level(codes, level, errors)
        if not isinstance(codes, (np.ndarray, Index)):
            try:
                codes = com.index_labels_to_array(codes, dtype=np.dtype('object'))
            except ValueError:
                pass
        inds: List[Any] = []
        for level_codes in codes:
            try:
                loc = self.get_loc(level_codes)
                if isinstance(loc, int):
                    inds.append(loc)
                elif isinstance(loc, slice):
                    step = loc.step if loc.step is not None else 1
                    inds.extend(range(loc.start, loc.stop, step))
                elif com.is_bool_indexer(loc):
                    if get_option('performance_warnings') and self._lexsort_depth == 0:
                        warnings.warn('dropping on a non-lexsorted multi-index without a level parameter may impact performance.', PerformanceWarning, stacklevel=find_stack_level())
                    loc = loc.nonzero()[0]
                    inds.extend(loc)
                else:
                    msg = f'unsupported indexer of type {type(loc)}'
                    raise AssertionError(msg)
            except KeyError:
                if errors != 'ignore':
                    raise
        return self.delete(inds)

    def _drop_from_level(self, codes: Any, level: Union[int, str], errors: str = 'raise') -> MultiIndex:
        codes_arr = com.index_labels_to_array(codes)
        i = self._get_level_number(level)
        index = self.levels[i]
        values = index.get_indexer(codes_arr)
        nan_codes = isna(codes_arr)
        values[np.equal(nan_codes, False) & (values == -1)] = -2
        if index.shape[0] == self.shape[0]:
            values[np.equal(nan_codes, True)] = -2
        not_found = codes_arr[values == -2]
        if len(not_found) != 0 and errors != 'ignore':
            raise KeyError(f'labels {not_found} not found in level')
        mask = ~algos.isin(self.codes[i], values)
        return self[mask]

    def swaplevel(self, i: Union[int, str] = -2, j: Union[int, str] = -1) -> MultiIndex:
        new_levels = list(self.levels)
        new_codes = list(self.codes)
        new_names = list(self.names)
        i = self._get_level_number(i)
        j = self._get_level_number(j)
        new_levels[i], new_levels[j] = new_levels[j], new_levels[i]
        new_codes[i], new_codes[j] = new_codes[j], new_codes[i]
        new_names[i], new_names[j] = new_names[j], new_names[i]
        return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

    def reorder_levels(self, order: List[Union[int, str]]) -> MultiIndex:
        order = [self._get_level_number(i) for i in order]
        result = self._reorder_ilevels(order)
        return result

    def _reorder_ilevels(self, order: List[int]) -> MultiIndex:
        if len(order) != self.nlevels:
            raise AssertionError(f'Length of order must be same as number of levels ({self.nlevels}), got {len(order)}')
        new_levels = [self.levels[i] for i in order]
        new_codes = [self.codes[i] for i in order]
        new_names = [self.names[i] for i in order]
        return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

    def _recode_for_new_levels(self, new_levels: Seq[Any], copy: bool = True) -> Any:
        if len(new_levels) > self.nlevels:
            raise AssertionError(f'Length of new_levels ({len(new_levels)}) must be <= self.nlevels ({self.nlevels})')
        for i in range(len(new_levels)):
            yield recode_for_categories(self.codes[i], self.levels[i], new_levels[i], copy=copy)

    def _get_codes_for_sorting(self) -> List[Any]:
        def cats(level_codes: np.ndarray) -> np.ndarray:
            return np.arange(level_codes.max() + 1 if len(level_codes) else 0, dtype=level_codes.dtype)
        return [Categorical.from_codes(level_codes, cats(level_codes), True, validate=False) for level_codes in self.codes]

    def sortlevel(self,
                  level: Union[int, str, List[Union[int, str]]] = 0,
                  ascending: Union[bool, List[bool]] = True,
                  sort_remaining: bool = True,
                  na_position: str = 'first') -> Tuple[MultiIndex, np.ndarray]:
        if not is_list_like(level):
            level = [level]
        level = [self._get_level_number(lev) for lev in level]
        sortorder: Optional[int] = None
        codes_for_sorting = [self.codes[lev] for lev in level]
        if isinstance(ascending, list):
            if not len(level) == len(ascending):
                raise ValueError('level must have same length as ascending')
        elif sort_remaining:
            codes_for_sorting.extend([self.codes[lev] for lev in range(len(self.levels)) if lev not in level])
        else:
            sortorder = level[0]
        indexer = lexsort_indexer(codes_for_sorting, orders=ascending, na_position=na_position, codes_given=True)
        indexer = ensure_platform_int(indexer)
        new_codes = [level_codes.take(indexer) for level_codes in self.codes]
        new_index = MultiIndex(codes=new_codes, levels=self.levels, names=self.names, sortorder=sortorder, verify_integrity=False)
        return (new_index, indexer)

    def _wrap_reindex_result(self, target: Any, indexer: Any, preserve_names: bool) -> Any:
        if not isinstance(target, MultiIndex):
            if indexer is None:
                target = self
            elif (indexer >= 0).all():
                target = self.take(indexer)
            else:
                try:
                    target = MultiIndex.from_tuples(target)
                except TypeError:
                    return target
        target = self._maybe_preserve_names(target, preserve_names)
        return target

    def _maybe_preserve_names(self, target: MultiIndex, preserve_names: bool) -> MultiIndex:
        if preserve_names and target.nlevels == self.nlevels and (target.names != self.names):
            target = target.copy(deep=False)
            target.names = self.names
        return target

    def _check_indexing_error(self, key: Any) -> None:
        if not is_hashable(key) or is_iterator(key):
            raise InvalidIndexError(key)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        return self.levels[0]._should_fallback_to_positional

    def _get_indexer_strict(self, key: Any, axis_name: str) -> Tuple[Any, Any]:
        keyarr = key
        if not isinstance(keyarr, Index):
            keyarr = com.asarray_tuplesafe(keyarr)
        if len(keyarr) and (not isinstance(keyarr[0], tuple)):
            indexer = self._get_indexer_level_0(keyarr)
            self._raise_if_missing(key, indexer, axis_name)
            return (self[indexer], indexer)
        return super()._get_indexer_strict(key, axis_name)

    def _raise_if_missing(self, key: Any, indexer: Any, axis_name: str) -> None:
        keyarr = key
        if not isinstance(key, Index):
            keyarr = com.asarray_tuplesafe(key)
        if len(keyarr) and (not isinstance(keyarr[0], tuple)):
            mask = indexer == -1
            if mask.any():
                check = self.levels[0].get_indexer(keyarr)
                cmask = check == -1
                if cmask.any():
                    raise KeyError(f'{keyarr[cmask]} not in index')
                raise KeyError(f'{keyarr} not in index')
        else:
            return super()._raise_if_missing(key, indexer, axis_name)

    def _get_indexer_level_0(self, target: Any) -> Any:
        lev = self.levels[0]
        codes = self._codes[0]
        cat = Categorical.from_codes(codes=codes, categories=lev, validate=False)
        ci = Index(cat)
        return ci.get_indexer_for(target)

    def get_slice_bound(self, label: Any, side: str) -> int:
        if not isinstance(label, tuple):
            label = (label,)
        return self._partial_tup_index(label, side=side)

    def slice_locs(self, start: Optional[Any] = None, end: Optional[Any] = None, step: Optional[Any] = None) -> Tuple[int, int]:
        return super().slice_locs(start, end, step)

    def _partial_tup_index(self, tup: Tuple[Any, ...], side: str = 'left') -> int:
        if len(tup) > self._lexsort_depth:
            raise UnsortedIndexError(f'Key length ({len(tup)}) was greater than MultiIndex lexsort depth ({self._lexsort_depth})')
        n = len(tup)
        start, end = (0, len(self))
        zipped = zip(tup, self.levels, self.codes)
        for k, (lab, lev, level_codes) in enumerate(zipped):
            section = level_codes[start:end]
            if lab not in lev and (not isna(lab)):
                try:
                    loc = algos.searchsorted(lev, lab, side=side)
                except TypeError as err:
                    raise TypeError(f'Level type mismatch: {lab}') from err
                if not is_integer(loc):
                    raise TypeError(f'Level type mismatch: {lab}')
                if side == 'right' and loc >= 0:
                    loc -= 1
                return start + algos.searchsorted(section, loc, side=side)
            idx = self._get_loc_single_level_index(lev, lab)
            if isinstance(idx, slice) and k < n - 1:
                start = idx.start
                end = idx.stop
            elif k < n - 1:
                end = start + algos.searchsorted(section, idx, side='right')
                start = start + algos.searchsorted(section, idx, side='left')
            elif isinstance(idx, slice):
                idx = idx.start
                return start + algos.searchsorted(section, idx, side=side)
            else:
                return start + algos.searchsorted(section, idx, side=side)
        return start

    def _get_loc_single_level_index(self, level_index: Index, key: Any) -> Union[int, slice]:
        if is_scalar(key) and isna(key):
            return -1
        else:
            return level_index.get_loc(key)

    def get_loc(self, key: Any) -> Union[int, slice, np.ndarray]:
        self._check_indexing_error(key)
        def _maybe_to_slice(loc: Any) -> Any:
            if not isinstance(loc, np.ndarray) or loc.dtype != np.intp:
                return loc
            loc = lib.maybe_indices_to_slice(loc, len(self))
            if isinstance(loc, slice):
                return loc
            mask = np.empty(len(self), dtype='bool')
            mask.fill(False)
            mask[loc] = True
            return mask
        if not isinstance(key, tuple):
            loc = self._get_level_indexer(key, level=0)
            return _maybe_to_slice(loc)
        keylen = len(key)
        if self.nlevels < keylen:
            raise KeyError(f'Key length ({keylen}) exceeds index depth ({self.nlevels})')
        if keylen == self.nlevels and self.is_unique:
            try:
                return self._engine.get_loc(key)
            except KeyError as err:
                raise KeyError(key) from err
            except TypeError:
                loc, _ = self.get_loc_level(key, range(self.nlevels))
                return loc
        i = self._lexsort_depth
        lead_key, follow_key = (key[:i], key[i:])
        if not lead_key:
            start = 0
            stop = len(self)
        else:
            try:
                start, stop = self.slice_locs(lead_key, lead_key)
            except TypeError as err:
                raise KeyError(key) from err
        if start == stop:
            raise KeyError(key)
        if not follow_key:
            return slice(start, stop)
        if get_option('performance_warnings'):
            warnings.warn('indexing past lexsort depth may impact performance.', PerformanceWarning, stacklevel=find_stack_level())
        loc = np.arange(start, stop, dtype=np.intp)
        for i, k in enumerate(follow_key, len(lead_key)):
            mask = self.codes[i][loc] == self._get_loc_single_level_index(self.levels[i], k)
            if not mask.all():
                loc = loc[mask]
            if not len(loc):
                raise KeyError(key)
        return _maybe_to_slice(loc) if len(loc) != stop - start else slice(start, stop)

    def get_loc_level(self, key: Any, level: Union[int, str] = 0, drop_level: bool = True) -> Tuple[Any, Optional[MultiIndex]]:
        if not isinstance(level, (range, list, tuple)):
            level = self._get_level_number(level)
        else:
            level = [self._get_level_number(lev) for lev in level]
        loc, mi = self._get_loc_level(key, level=level)
        if not drop_level:
            if isinstance(loc, int):
                mi = self[loc:loc + 1]
            else:
                mi = self[loc]
        return (loc, mi)

    def _get_loc_level(self, key: Any, level: Union[int, List[int]]) -> Tuple[Any, Optional[MultiIndex]]:
        def maybe_mi_droplevels(indexer: Any, levels: List[int]) -> MultiIndex:
            new_index = self[indexer]
            for i in sorted(levels, reverse=True):
                new_index = new_index._drop_level_numbers([i])
            return new_index
        if isinstance(level, (tuple, list)):
            if len(key) != len(level):
                raise AssertionError('Key for location must have same length as number of levels')
            result = None
            for lev, k in zip(level, key):
                loc, new_index = self._get_loc_level(k, level=lev)
                if isinstance(loc, slice):
                    mask = np.zeros(len(self), dtype=bool)
                    mask[loc] = True
                    loc = mask
                result = loc if result is None else result & loc
            try:
                mi = maybe_mi_droplevels(result, level)
            except ValueError:
                mi = self[result]
            return (result, mi)
        if isinstance(key, list):
            key = tuple(key)
        if isinstance(key, tuple) and level == 0:
            try:
                if key in self.levels[0]:
                    indexer = self._get_level_indexer(key, level=level)
                    new_index = maybe_mi_droplevels(indexer, [0])
                    return (indexer, new_index)
            except (TypeError, InvalidIndexError):
                pass
            if not any((isinstance(k, slice) for k in key)):
                if len(key) == self.nlevels and self.is_unique:
                    try:
                        return (self._engine.get_loc(key), None)
                    except KeyError as err:
                        raise KeyError(key) from err
                    except TypeError:
                        pass
                indexer = self.get_loc(key)
                ilevels = [i for i in range(len(key)) if key[i] != slice(None, None)]
                if len(ilevels) == self.nlevels:
                    if is_integer(indexer):
                        return (indexer, None)
                    ilevels = [i for i in range(len(key)) if (not isinstance(key[i], str) or not self.levels[i]._supports_partial_string_indexing) and key[i] != slice(None, None)]
                    if len(ilevels) == self.nlevels:
                        ilevels = []
                return (indexer, maybe_mi_droplevels(indexer, ilevels))
            else:
                indexer = None
                for i, k in enumerate(key):
                    if not isinstance(k, slice):
                        loc_level = self._get_level_indexer(k, level=i)
                        if isinstance(loc_level, slice):
                            if com.is_null_slice(loc_level) or com.is_full_slice(loc_level, len(self)):
                                continue
                            k_index = np.zeros(len(self), dtype=bool)
                            k_index[loc_level] = True
                        else:
                            k_index = loc_level
                    elif com.is_null_slice(k):
                        continue
                    else:
                        raise TypeError(f'Expected label or tuple of labels, got {key}')
                    if indexer is None:
                        indexer = k_index
                    else:
                        indexer &= k_index
                if indexer is None:
                    indexer = slice(None, None)
                ilevels = [i for i in range(len(key)) if key[i] != slice(None, None)]
                return (indexer, maybe_mi_droplevels(indexer, ilevels))
        else:
            indexer = self._get_level_indexer(key, level=level)
            if isinstance(key, str) and self.levels[level]._supports_partial_string_indexing:
                check = self.levels[level].get_loc(key)
                if not is_integer(check):
                    return (indexer, self[indexer])
            try:
                result_index = maybe_mi_droplevels(indexer, [level])
            except ValueError:
                result_index = self[indexer]
            return (indexer, result_index)

    def _get_level_indexer(self, key: Any, level: int, indexer: Optional[np.ndarray] = None) -> Union[np.ndarray, slice]:
        level_index = self.levels[level]
        level_codes = self.codes[level]
        def convert_indexer(start: int, stop: int, step: Optional[int]) -> Any:
            if indexer is not None:
                codes = level_codes[indexer]
            else:
                codes = level_codes
            if step is None or step == 1:
                new_indexer = (codes >= start) & (codes < stop)
            else:
                r = np.arange(start, stop, step, dtype=codes.dtype)
                new_indexer = algos.isin(codes, r)
            if indexer is None:
                return new_indexer
            indexer_copy = indexer.copy()
            indexer_copy[indexer_copy] = new_indexer
            return indexer_copy
        if isinstance(key, slice):
            step = key.step
            is_negative_step = step is not None and step < 0
            try:
                if key.start is not None:
                    start = level_index.get_loc(key.start)
                elif is_negative_step:
                    start = len(level_index) - 1
                else:
                    start = 0
                if key.stop is not None:
                    stop = level_index.get_loc(key.stop)
                elif is_negative_step:
                    stop = 0
                elif isinstance(start, slice):
                    stop = len(level_index)
                else:
                    stop = len(level_index) - 1
            except KeyError:
                start = stop = level_index.slice_indexer(key.start, key.stop, key.step)
                step = start.step
            if isinstance(start, slice) or isinstance(stop, slice):
                start_val = getattr(start, 'start', start)
                stop_val = getattr(stop, 'stop', stop)
                return convert_indexer(start_val, stop_val, step)
            elif level > 0 or self._lexsort_depth == 0 or step is not None:
                stop = stop - 1 if is_negative_step else stop + 1
                return convert_indexer(start, stop, step)
            else:
                i = algos.searchsorted(level_codes, start, side='left')
                j = algos.searchsorted(level_codes, stop, side='right')
                return slice(i, j, step)
        else:
            idx = self._get_loc_single_level_index(level_index, key)
            if level > 0 or self._lexsort_depth == 0:
                if isinstance(idx, slice):
                    locs = (level_codes >= idx.start) & (level_codes < idx.stop)
                    return locs
                locs = np.asarray(level_codes == idx, dtype=bool)
                if not locs.any():
                    raise KeyError(key)
                return locs
            if isinstance(idx, slice):
                start = algos.searchsorted(level_codes, idx.start, side='left')
                end = algos.searchsorted(level_codes, idx.stop, side='left')
            else:
                start = algos.searchsorted(level_codes, idx, side='left')
                end = algos.searchsorted(level_codes, idx, side='right')
            if start == end:
                raise KeyError(key)
            return slice(start, end)

    def get_locs(self, seq: Any) -> np.ndarray:
        true_slices = [i for i, s in enumerate(com.is_true_slices(seq)) if s]
        if true_slices and true_slices[-1] >= self._lexsort_depth:
            raise UnsortedIndexError(f'MultiIndex slicing requires the index to be lexsorted: slicing on levels {true_slices}, lexsort depth {self._lexsort_depth}')
        if any((x is Ellipsis for x in seq)):
            raise NotImplementedError('MultiIndex does not support indexing with Ellipsis')
        n = len(self)
        def _to_bool_indexer(indexer: Any) -> Any:
            if isinstance(indexer, slice):
                new_indexer = np.zeros(n, dtype=np.bool_)
                new_indexer[indexer] = True
                return new_indexer
            return indexer
        indexer: Optional[np.ndarray] = None
        for i, k in enumerate(seq):
            lvl_indexer = None
            if com.is_bool_indexer(k):
                if len(k) != n:
                    raise ValueError('cannot index with a boolean indexer that is not the same length as the index')
                if isinstance(k, (ABCSeries, Index)):
                    k = k._values
                lvl_indexer = np.asarray(k)
                if indexer is None:
                    lvl_indexer = lvl_indexer.copy()
            elif is_list_like(k):
                try:
                    lvl_indexer = self._get_level_indexer(k, level=i, indexer=indexer)
                except (InvalidIndexError, TypeError, KeyError) as err:
                    for x in k:
                        if not is_hashable(x):
                            raise err
                        item_indexer = self._get_level_indexer(x, level=i, indexer=indexer)
                        if lvl_indexer is None:
                            lvl_indexer = _to_bool_indexer(item_indexer)
                        elif isinstance(item_indexer, slice):
                            lvl_indexer[item_indexer] = True
                        else:
                            lvl_indexer |= item_indexer
                if lvl_indexer is None:
                    return np.array([], dtype=np.intp)
            elif com.is_null_slice(k):
                if indexer is None and i == len(seq) - 1:
                    return np.arange(n, dtype=np.intp)
                continue
            else:
                lvl_indexer = self._get_level_indexer(k, level=i, indexer=indexer)
            lvl_indexer = _to_bool_indexer(lvl_indexer)
            if indexer is None:
                indexer = lvl_indexer
            else:
                indexer &= lvl_indexer
                if not np.any(indexer) and np.any(lvl_indexer):
                    raise KeyError(seq)
        if indexer is None:
            return np.array([], dtype=np.intp)
        pos_indexer = indexer.nonzero()[0]
        return self._reorder_indexer(seq, pos_indexer)

    def _reorder_indexer(self, seq: Any, indexer: np.ndarray) -> np.ndarray:
        need_sort = False
        for i, k in enumerate(seq):
            if com.is_null_slice(k) or com.is_bool_indexer(k) or is_scalar(k):
                pass
            elif is_list_like(k):
                if len(k) <= 1:
                    pass
                elif self._is_lexsorted():
                    k_codes = self.levels[i].get_indexer(k)
                    k_codes = k_codes[k_codes >= 0]
                    need_sort = (k_codes[:-1] > k_codes[1:]).any()
                else:
                    need_sort = True
            elif isinstance(k, slice):
                if self._is_lexsorted():
                    need_sort = k.step is not None and k.step < 0
                else:
                    need_sort = True
            else:
                need_sort = True
            if need_sort:
                break
        if not need_sort:
            return indexer
        n = len(self)
        keys: Tuple[Any, ...] = ()
        for i, k in enumerate(seq):
            if is_scalar(k):
                k = [k]
            if com.is_bool_indexer(k):
                new_order = np.arange(n)[indexer]
            elif is_list_like(k):
                if not isinstance(k, (np.ndarray, ExtensionArray, Index, ABCSeries)):
                    k = sanitize_array(k, None)
                k = algos.unique(k)
                key_order_map = np.ones(len(self.levels[i]), dtype=np.uint64) * len(self.levels[i])
                level_indexer = self.levels[i].get_indexer(k)
                level_indexer = level_indexer[level_indexer >= 0]
                key_order_map[level_indexer] = np.arange(len(level_indexer))
                new_order = key_order_map[self.codes[i][indexer]]
            elif isinstance(k, slice) and k.step is not None and (k.step < 0):
                new_order = np.arange(n - 1, -1, -1)[indexer]
            elif isinstance(k, slice) and k.start is None and (k.stop is None):
                new_order = np.ones((n,), dtype=np.intp)[indexer]
            else:
                new_order = np.arange(n)[indexer]
            keys = (new_order,) + keys
        ind = np.lexsort(keys)
        return indexer[ind]

    def truncate(self, before: Optional[Any] = None, after: Optional[Any] = None) -> MultiIndex:
        if after and before and (after < before):
            raise ValueError('after < before')
        i, j = self.levels[0].slice_locs(before, after)
        left, right = self.slice_locs(before, after)
        new_levels = list(self.levels)
        new_levels[0] = new_levels[0][i:j]
        new_codes = [level_codes[left:right] for level_codes in self.codes]
        new_codes[0] = new_codes[0] - i
        return MultiIndex(levels=new_levels, codes=new_codes, names=self._names, verify_integrity=False)

    def equals(self, other: Any) -> bool:
        if self.is_(other):
            return True
        if not isinstance(other, Index):
            return False
        if len(self) != len(other):
            return False
        if not isinstance(other, MultiIndex):
            if not self._should_compare(other):
                return False
            return array_equivalent(self._values, other._values)
        if self.nlevels != other.nlevels:
            return False
        for i in range(self.nlevels):
            self_codes = self.codes[i]
            other_codes = other.codes[i]
            self_mask = self_codes == -1
            other_mask = other_codes == -1
            if not np.array_equal(self_mask, other_mask):
                return False
            self_level = self.levels[i]
            other_level = other.levels[i]
            new_codes = recode_for_categories(other_codes, other_level, self_level, copy=False)
            if not np.array_equal(self_codes, new_codes):
                return False
            if not self_level[:0].equals(other_level[:0]):
                return False
        return True

    def equal_levels(self, other: MultiIndex) -> bool:
        if self.nlevels != other.nlevels:
            return False
        for i in range(self.nlevels):
            if not self.levels[i].equals(other.levels[i]):
                return False
        return True

    def _union(self, other: Any, sort: bool) -> MultiIndex:
        other, result_names = self._convert_can_do_setop(other)
        if other.has_duplicates:
            result = super()._union(other, sort)
            if isinstance(result, MultiIndex):
                return result
            return MultiIndex.from_arrays(zip(*result), sortorder=None, names=result_names)
        else:
            right_missing = other.difference(self, sort=False)
            if len(right_missing):
                result = self.append(right_missing)
            else:
                result = self._get_reconciled_name_object(other)
            if sort is not False:
                try:
                    result = result.sort_values()
                except TypeError:
                    if sort is True:
                        raise
                    warnings.warn('The values in the array are unorderable. Pass `sort=False` to suppress this warning.', RuntimeWarning, stacklevel=find_stack_level())
            return result

    def _is_comparable_dtype(self, dtype: Any) -> bool:
        return is_object_dtype(dtype)

    def _get_reconciled_name_object(self, other: Any) -> MultiIndex:
        names = self._maybe_match_names(other)
        if self.names != names:
            return self.rename(names)
        return self

    def _maybe_match_names(self, other: Any) -> List[Any]:
        if len(self.names) != len(other.names):
            return [None] * len(self.names)
        names: List[Any] = []
        for a_name, b_name in zip(self.names, other.names):
            if a_name == b_name:
                names.append(a_name)
            else:
                names.append(None)
        return names

    def _wrap_intersection_result(self, other: Any, result: MultiIndex) -> MultiIndex:
        _, result_names = self._convert_can_do_setop(other)
        return result.set_names(result_names)

    def _wrap_difference_result(self, other: Any, result: MultiIndex) -> MultiIndex:
        _, result_names = self._convert_can_do_setop(other)
        if len(result) == 0:
            return result.remove_unused_levels().set_names(result_names)
        else:
            return result.set_names(result_names)

    def _convert_can_do_setop(self, other: Any) -> Tuple[Any, List[Any]]:
        result_names = self.names
        if not isinstance(other, Index):
            if len(other) == 0:
                return (self[:0], self.names)
            else:
                msg = 'other must be a MultiIndex or a list of tuples'
                try:
                    other = MultiIndex.from_tuples(other, names=self.names)
                except (ValueError, TypeError) as err:
                    raise TypeError(msg) from err
        else:
            result_names = get_unanimous_names(self, other)
        return (other, result_names)

    @doc(Index.astype)
    def astype(self, dtype: Any, copy: bool = True) -> MultiIndex:
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, CategoricalDtype):
            msg = '> 1 ndim Categorical are not supported at this time'
            raise NotImplementedError(msg)
        if not is_object_dtype(dtype):
            raise TypeError('Setting a MultiIndex dtype to anything other than object is not supported')
        if copy is True:
            return self._view()
        return self

    def _validate_fill_value(self, item: Any) -> Tuple[Any, ...]:
        if isinstance(item, MultiIndex):
            if item.nlevels != self.nlevels:
                raise ValueError('Item must have length equal to number of levels.')
            return item._values
        elif not isinstance(item, tuple):
            item = (item,) + ('',) * (self.nlevels - 1)
        elif len(item) != self.nlevels:
            raise ValueError('Item must have length equal to number of levels.')
        return item

    def putmask(self, mask: Any, value: MultiIndex) -> MultiIndex:
        mask, noop = validate_putmask(self, mask)
        if noop:
            return self.copy()
        if len(mask) == len(value):
            subset = value[mask].remove_unused_levels()
        else:
            subset = value.remove_unused_levels()
        new_levels: List[Any] = []
        new_codes: List[Any] = []
        for i, (value_level, level, level_codes) in enumerate(zip(subset.levels, self.levels, self.codes)):
            new_level = level.union(value_level, sort=False)
            value_codes = new_level.get_indexer_for(subset.get_level_values(i))
            new_code = ensure_int64(level_codes)
            new_code[mask] = value_codes
            new_levels.append(new_level)
            new_codes.append(new_code)
        return MultiIndex(levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False)

    def insert(self, loc: int, item: Any) -> MultiIndex:
        item = self._validate_fill_value(item)
        new_levels: List[Any] = []
        new_codes: List[np.ndarray] = []
        for k, level, level_codes in zip(item, self.levels, self.codes):
            if k not in level:
                lev_loc = len(level)
                level = level.insert(lev_loc, k)
                if isna(level[lev_loc]):
                    lev_loc = -1
            else:
                lev_loc = level.get_loc(k)
            new_levels.append(level)
            new_codes.append(np.insert(ensure_int64(level_codes), loc, lev_loc))
        return MultiIndex(levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False)

    def delete(self, loc: Union[int, List[int]]) -> MultiIndex:
        new_codes = [np.delete(level_codes, loc) for level_codes in self.codes]
        return MultiIndex(levels=self.levels, codes=new_codes, names=self.names, verify_integrity=False)

    @doc(Index.isin)
    def isin(self, values: Any, level: Optional[Union[int, str]] = None) -> np.ndarray:
        if isinstance(values, Generator):
            values = list(values)
        if level is None:
            if len(values) == 0:
                return np.zeros((len(self),), dtype=np.bool_)
            if not isinstance(values, MultiIndex):
                values = MultiIndex.from_tuples(values)
            return values.unique().get_indexer_for(self) != -1
        else:
            num = self._get_level_number(level)
            levs = self.get_level_values(num)
            if levs.size == 0:
                return np.zeros(len(levs), dtype=np.bool_)
            return levs.isin(values)
    rename = Index.set_names

    __add__ = make_invalid_op('__add__')
    __radd__ = make_invalid_op('__radd__')
    __iadd__ = make_invalid_op('__iadd__')
    __sub__ = make_invalid_op('__sub__')
    __rsub__ = make_invalid_op('__rsub__')
    __isub__ = make_invalid_op('__isub__')
    __pow__ = make_invalid_op('__pow__')
    __rpow__ = make_invalid_op('__rpow__')
    __mul__ = make_invalid_op('__mul__')
    __rmul__ = make_invalid_op('__rmul__')
    __floordiv__ = make_invalid_op('__floordiv__')
    __rfloordiv__ = make_invalid_op('__rfloordiv__')
    __truediv__ = make_invalid_op('__truediv__')
    __rtruediv__ = make_invalid_op('__rtruediv__')
    __mod__ = make_invalid_op('__mod__')
    __rmod__ = make_invalid_op('__rmod__')
    __divmod__ = make_invalid_op('__divmod__')
    __rdivmod__ = make_invalid_op('__rdivmod__')
    __neg__ = make_invalid_op('__neg__')
    __pos__ = make_invalid_op('__pos__')
    __abs__ = make_invalid_op('__abs__')
    __invert__ = make_invalid_op('__invert__')

def _lexsort_depth(codes: Seq[np.ndarray], nlevels: int) -> int:
    int64_codes = [ensure_int64(level_codes) for level_codes in codes]
    for k in range(nlevels, 0, -1):
        if libalgos.is_lexsorted(int64_codes[:k]):
            return k
    return 0

def sparsify_labels(label_list: List[List[Any]], start: int = 0, sentinel: str = '') -> List[Tuple[Any, ...]]:
    pivoted = list(zip(*label_list))
    k = len(label_list)
    result = pivoted[:start + 1]
    prev = pivoted[start]
    for cur in pivoted[start + 1:]:
        sparse_cur: List[Any] = []
        for i, (p, t) in enumerate(zip(prev, cur)):
            if i == k - 1:
                sparse_cur.append(t)
                result.append(tuple(sparse_cur))
                break
            if p == t:
                sparse_cur.append(sentinel)
            else:
                sparse_cur.extend(cur[i:])
                result.append(tuple(sparse_cur))
                break
        prev = cur
    return list(zip(*result))

def _get_na_rep(dtype: Any) -> str:
    if isinstance(dtype, ExtensionDtype):
        return f'{dtype.na_value}'
    else:
        dtype_type = dtype.type
    return {np.datetime64: 'NaT', np.timedelta64: 'NaT'}.get(dtype_type, 'NaN')

def maybe_droplevels(index: Index, key: Union[Any, Tuple[Any, ...]]) -> Index:
    original_index = index
    if isinstance(key, tuple):
        for _ in key:
            try:
                index = index._drop_level_numbers([0])
            except ValueError:
                return original_index
    else:
        try:
            index = index._drop_level_numbers([0])
        except ValueError:
            pass
    return index

def _coerce_indexer_frozen(array_like: Any, categories: Any, copy: bool = False) -> np.ndarray:
    array_like = coerce_indexer_dtype(array_like, categories)
    if copy:
        array_like = array_like.copy()
    array_like.flags.writeable = False
    return array_like

def _require_listlike(level: Any, arr: Any, arrname: str) -> Tuple[Any, Any]:
    if level is not None and (not is_list_like(level)):
        if not is_list_like(arr):
            raise TypeError(f'{arrname} must be list-like')
        if len(arr) > 0 and is_list_like(arr[0]):
            raise TypeError(f'{arrname} must be list-like')
        level = [level]
        arr = [arr]
    elif level is None or is_list_like(level):
        if not is_list_like(arr) or not is_list_like(arr[0]):
            raise TypeError(f'{arrname} must be list of lists-like')
    return (level, arr)

def cartesian_product(X: Iterable[Iterable[Any]]) -> List[np.ndarray]:
    msg = 'Input must be a list-like of list-likes'
    if not is_list_like(X):
        raise TypeError(msg)
    for x in X:
        if not is_list_like(x):
            raise TypeError(msg)
    if len(list(X)) == 0:
        return []
    lenX = np.fromiter((len(x) for x in X), dtype=np.intp)
    cumprodX = np.cumprod(lenX)
    if np.any(cumprodX < 0):
        raise ValueError('Product space too large to allocate arrays!')
    a = np.roll(cumprodX, 1)
    a[0] = 1
    if cumprodX[-1] != 0:
        b = cumprodX[-1] / cumprodX
    else:
        b = np.zeros_like(cumprodX)
    return [np.tile(np.repeat(x, b[i]), np.prod(a[i])) for i, x in enumerate(X)]
