from __future__ import annotations
from collections.abc import Callable, Collection, Generator, Hashable, Iterable, Sequence
from functools import wraps
from sys import getsizeof
from typing import TYPE_CHECKING, Any, Literal, cast, Optional, Union, List, Tuple, Dict, Set, FrozenSet, Iterator, TypeVar, Generic, overload
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
from pandas.core.dtypes.common import ensure_int64, ensure_platform_int, is_hashable, is_integer, is_iterator, is_list_like, is_object_dtype, is_scalar, is_string_dtype, pandas_dtype
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

if TYPE_CHECKING:
    from pandas import CategoricalIndex, DataFrame, Series

_index_doc_kwargs: Dict[str, Any] = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({'klass': 'MultiIndex', 'target_klass': 'MultiIndex or list of tuples'})

T = TypeVar('T')

class MultiIndexUInt64Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt64Engine):
    _base: Any = libindex.UInt64Engine
    _codes_dtype: str = 'uint64'

class MultiIndexUInt32Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt32Engine):
    _base: Any = libindex.UInt32Engine
    _codes_dtype: str = 'uint32'

class MultiIndexUInt16Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt16Engine):
    _base: Any = libindex.UInt16Engine
    _codes_dtype: str = 'uint16'

class MultiIndexUInt8Engine(libindex.BaseMultiIndexCodesEngine, libindex.UInt8Engine):
    _base: Any = libindex.UInt8Engine
    _codes_dtype: str = 'uint8'

class MultiIndexPyIntEngine(libindex.BaseMultiIndexCodesEngine, libindex.ObjectEngine):
    _base: Any = libindex.ObjectEngine
    _codes_dtype: str = 'object'

def names_compat(meth: F) -> F:
    @wraps(meth)
    def new_meth(self_or_cls: Any, *args: Any, **kwargs: Any) -> Any:
        if 'name' in kwargs and 'names' in kwargs:
            raise TypeError('Can only provide one of `names` and `name`')
        if 'name' in kwargs:
            kwargs['names'] = kwargs.pop('name')
        return meth(self_or_cls, *args, **kwargs)
    return cast(F, new_meth)

@set_module('pandas')
class MultiIndex(Index):
    _hidden_attrs: FrozenSet[str] = Index._hidden_attrs | frozenset()
    _typ: str = 'multiindex'
    _names: List[Optional[str]] = []
    _levels: FrozenList = FrozenList()
    _codes: FrozenList = FrozenList()
    _comparables: List[str] = ['names']

    def __new__(
        cls,
        levels: Optional[Sequence[Any]] = None,
        codes: Optional[Sequence[Any]] = None,
        sortorder: Optional[int] = None,
        names: Optional[Sequence[Any]] = None,
        dtype: Optional[DtypeObj] = None,
        copy: bool = False,
        name: Optional[str] = None,
        verify_integrity: bool = True
    ) -> MultiIndex:
        if name is not None:
            names = name
        if levels is None or codes is None:
            raise TypeError('Must pass both levels and codes')
        if len(levels) != len(codes):
            raise ValueError('Length of levels and codes must be the same.')
        if len(levels) == 0:
            raise ValueError('Must pass non-zero number of levels/codes')
        result = object.__new__(cls)
        result._cache: Dict[str, Any] = {}
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

    def _validate_codes(self, level: np.ndarray, code: np.ndarray) -> np.ndarray:
        null_mask = isna(level)
        if np.any(null_mask):
            code = np.where(null_mask[code], -1, code)
        return code

    def _verify_integrity(
        self,
        codes: Optional[Sequence[np.ndarray]] = None,
        levels: Optional[Sequence[Any]] = None,
        levels_to_verify: Optional[Sequence[int]] = None
    ) -> FrozenList:
        codes = codes or self.codes
        levels = levels or self.levels
        if levels_to_verify is None:
            levels_to_verify = range(len(levels))
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
        result_codes = []
        for i in range(len(levels)):
            if i in levels_to_verify:
                result_codes.append(self._validate_codes(levels[i], codes[i]))
            else:
                result_codes.append(codes[i])
        new_codes = FrozenList(result_codes)
        return new_codes

    @classmethod
    def from_arrays(
        cls,
        arrays: Sequence[AnyArrayLike],
        sortorder: Optional[int] = None,
        names: Union[Sequence[Any], Literal[lib.no_default]] = lib.no_default
    ) -> MultiIndex:
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
    def from_tuples(
        cls,
        tuples: Sequence[Tuple[Any, ...]],
        sortorder: Optional[int] = None,
        names: Optional[Sequence[str]] = None
    ) -> MultiIndex:
        if not is_list_like(tuples):
            raise TypeError('Input must be a list / sequence of tuple-likes.')
        if is_iterator(tuples):
            tuples = list(tuples)
        tuples = cast(Collection[Tuple[Hashable, ...]], tuples)
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
            arrays = cast(List[Sequence[Hashable]], arrs)
        return cls.from_arrays(arrays, sortorder=sortorder, names=names)

    @classmethod
    def from_product(
        cls,
        iterables: Sequence[Iterable[Any]],
        sortorder: Optional[int] = None,
        names: Union[Sequence[Any], Literal[lib.no_default]] = lib.no_default
    ) -> MultiIndex:
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
    def from_frame(
        cls,
        df: DataFrame,
        sortorder: Optional[int] = None,
        names: Optional[Sequence[str]] = None
    ) -> MultiIndex:
        if not isinstance(df, ABCDataFrame):
            raise TypeError('Input must be a DataFrame')
        column_names, columns = zip(*df.items())
        names = column_names if names is None else names
        return cls.from_arrays(columns, sortorder=sortorder, names=names)

    @cache_readonly
    def _values(self) -> np.ndarray:
        values = []
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
    def dtypes(self) -> Series:
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

    def _set_levels(
        self,
        levels: Sequence[Any],
        *,
        level: Optional[Sequence[int]] = None,
        copy: bool = False,
        validate: bool = True,
        verify_integrity: bool = False
    ) -> None:
        if validate:
            if len(levels) == 0:
                raise ValueError('Must set non-zero number of levels.')
            if level is None and len(levels) != self.nlevels:
                raise ValueError('Length of levels must match number of levels.')
            if level is not None and len(levels) != len(level):
                raise ValueError('Length of levels must match length of level.')
        if level is None:
            new_levels = FrozenList((ensure_index(lev, copy=copy)._view() for lev in levels))
            level_numbers = range(len(new_levels))
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

    def set_levels(
        self,
        levels: Sequence[Any],
        *,
        level: Optional[Sequence[Union[int, str]]] = None,
        verify_integrity: bool = True
    ) -> MultiIndex:
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

    def _set_codes(
        self,
        codes: Sequence[Any],
        *,
        level: Optional[Sequence[int]] = None,
        copy: bool = False,
        validate: bool = True,
        verify_integrity: bool = False
    ) -> None:
        if validate:
            if level is None and len(codes) != self.nlevels:
                raise ValueError('Length of codes must match number of levels')
            if level is not None and len(codes) != len(level):
                raise ValueError('Length of codes must match length of levels.')
        if level is None:
            new_codes = FrozenList((_coerce_indexer_frozen(level_codes, lev, copy=copy).view() for lev, level_codes in zip(self._levels, codes)))
            level_numbers = range(len(new_codes))
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

    def set_codes(
        self,
        codes: Sequence[Any],
        *,
        level: Optional[Sequence[Union[int, str]]] = None,
        verify_integrity: bool = True
    ) -> MultiIndex:
       