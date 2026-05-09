from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import functools
import operator
import sys
from textwrap import dedent
from typing import IO, TYPE_CHECKING, Any, Literal, cast, overload
import warnings
import numpy as np
from pandas._libs import lib, properties, reshape
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import ChainedAssignmentError, InvalidIndexError
from pandas.errors.cow import _chained_assignment_method_msg, _chained_assignment_msg
from pandas.util._decorators import Appender, Substitution, deprecate_nonkeyword_arguments, doc, set_module
from pandas.util._validators import validate_ascending, validate_bool_kwarg, validate_percentile
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import LossySetitemError, construct_1d_arraylike_from_scalar, find_common_type, infer_dtype_from, maybe_box_native, maybe_cast_pointwise_result
from pandas.core.dtypes.common import is_dict_like, is_float, is_integer, is_iterator, is_list_like, is_object_dtype, is_scalar, pandas_dtype, validate_all_hashable
from pandas.core.dtypes.dtypes import CategoricalDtype, ExtensionDtype, SparseDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import isna, na_value_for_dtype, notna, remove_na_arraylike
from pandas.core import algorithms, base, common as com, nanops, ops, roperator
from pandas.core.accessor import Accessor
from pandas.core.apply import SeriesApply
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.arrow import ListAccessor, StructAccessor
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import array as pd_array, extract_array, sanitize_array
from pandas.core.generic import NDFrame, make_doc
from pandas.core.indexers import disallow_ndim_indexing, unpack_1tuple
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
from pandas.core.indexes.api import DatetimeIndex, Index, MultiIndex, PeriodIndex, default_index, ensure_index, maybe_sequence_to_range
import pandas.core.indexes.base as ibase
from pandas.core.indexes.multi import maybe_droplevels
from pandas.core.indexing import check_bool_indexer, check_dict_or_set_indexers
from pandas.core.internals import SingleBlockManager
from pandas.core.methods import selectn
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import ensure_key_mapped, nargsort
from pandas.core.strings.accessor import StringMethods
from pandas.core.tools.datetimes import to_datetime
import pandas.io.formats.format as fmt
from pandas.io.formats.info import INFO_DOCSTRING, SeriesInfo, series_sub_kwargs
import pandas.plotting
if TYPE_CHECKING:
    from pandas._libs.internals import BlockValuesRefs
    from pandas._typing import AggFuncType, AnyAll, AnyArrayLike, ArrayLike, Axis, AxisInt, CorrelationMethod, DropKeep, Dtype, DtypeObj, FilePath, Frequency, IgnoreRaise, IndexKeyFunc, IndexLabel, Level, ListLike, MutableMappingT, NaPosition, NumpySorter, NumpyValueArrayLike, QuantileInterpolation, ReindexMethod, Renamer, Scalar, Self, SortKind, StorageOptions, Suffixes, ValueKeyFunc, WriteBuffer, npt
    from pandas.core.frame import DataFrame
    from pandas.core.groupby.generic import SeriesGroupBy
__all__ = ['Series']

_shared_doc_kwargs = {'axes': 'index', 'klass': 'Series', 'axes_single_arg': "{0 or 'index'}", 'axis': "axis : {0 or 'index'}\n        Unused. Parameter needed for compatibility with DataFrame.", 'inplace': 'inplace : bool, default False\n        If True, performs operation inplace and returns None.', 'unique': 'np.ndarray', 'duplicated': 'Series', 'optional_by': '', 'optional_reindex': '\nindex : array-like, optional\n    New labels for the index. Preferably an Index object to avoid\n    duplicating data.\naxis : int or str, optional\n    Unused.'}

@set_module('pandas')
class Series(base.IndexOpsMixin, NDFrame):
    _typ = 'series'
    _HANDLED_TYPES = (Index, ExtensionArray, np.ndarray)
    _metadata = ['_name']
    _internal_names_set = {'index', 'name'} | NDFrame._internal_names_set
    _accessors = {'dt', 'cat', 'str', 'sparse'}
    _hidden_attrs = base.IndexOpsMixin._hidden_attrs | NDFrame._hidden_attrs | frozenset([])
    __pandas_priority__ = 3000
    hasnans = property(base.IndexOpsMixin.hasnans.fget, doc=base.IndexOpsMixin.hasnans.__doc__)

    def __init__(self, data: Any = None, index: Any = None, dtype: Any = None, name: Any = None, copy: Any = None) -> None:
        ...

    def _init_dict(self, data: Mapping, index: Any = None, dtype: Any = None) -> tuple:
        ...

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        ...

    @property
    def _constructor(self) -> type:
        return Series

    def _constructor_from_mgr(self, mgr: Any, axes: Any) -> Series:
        ...

    @property
    def _constructor_expanddim(self) -> type:
        ...

    def _constructor_expanddim_from_mgr(self, mgr: Any, axes: Any) -> DataFrame:
        ...

    @property
    def _can_hold_na(self) -> bool:
        return self._mgr._can_hold_na

    @property
    def dtype(self) -> DtypeObj:
        ...

    @property
    def dtypes(self) -> DtypeObj:
        ...

    @property
    def name(self) -> Any:
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        ...

    @property
    def values(self) -> np.ndarray:
        ...

    @property
    def _values(self) -> np.ndarray:
        ...

    @property
    def _references(self) -> BlockValuesRefs:
        return self._mgr._block.refs

    @Appender(base.IndexOpsMixin.array.__doc__)
    @property
    def array(self) -> ExtensionArray:
        return self._mgr.array_values()

    def __len__(self) -> int:
        return len(self._mgr)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        ...

    def _ixs(self, i: int, axis: int = 0) -> Any:
        return self._values[i]

    def _slice(self, slobj: slice, axis: int = 0) -> Series:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def _get_with(self, key: Any) -> Series:
        ...

    def _get_values_tuple(self, key: tuple) -> Series:
        ...

    def _get_rows_with_mask(self, indexer: np.ndarray) -> Series:
        ...

    def _get_value(self, label: Any, takeable: bool = False) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def _set_with_engine(self, key: Any, value: Any) -> None:
        ...

    def _set_with(self, key: Any, value: Any) -> None:
        ...

    def _set_labels(self, key: Any, value: Any) -> None:
        ...

    def _set_values(self, key: Any, value: Any) -> None:
        ...

    def _set_value(self, label: Any, value: Any, takeable: bool = False) -> None:
        ...

    def repeat(self, repeats: int, axis: Any = None) -> Series:
        ...

    @overload
    def reset_index(self, level: Any = ..., *, drop: Any = ..., name: Any = ..., inplace: Any = ..., allow_duplicates: Any = ...) -> Series:
        ...

    @overload
    def reset_index(self, level: Any = ..., *, drop: Any, name: Any = ..., inplace: Any = ..., allow_duplicates: Any = ...) -> Series:
        ...

    @overload
    def reset_index(self, level: Any = ..., *, drop: Any = ..., name: Any = ..., inplace: Any, allow_duplicates: Any = ...) -> Series:
        ...

    def reset_index(self, level: Any = None, *, drop: bool = False, name: Any = lib.no_default, inplace: bool = False, allow_duplicates: bool = False) -> Series:
        ...

    def __repr__(self) -> str:
        return self.to_string(**fmt.get_series_repr_params())

    @overload
    def to_string(self, buf: Any = ..., *, na_rep: Any = ..., float_format: Any = ..., header: Any = ..., index: Any = ..., length: Any = ..., dtype: Any = ..., name: Any = ..., max_rows: Any = ..., min_rows: Any = ...) -> None:
        ...

    @overload
    def to_string(self, buf: Any, *, na_rep: Any = ..., float_format: Any = ..., header: Any = ..., index: Any = ..., length: Any = ..., dtype: Any = ..., name: Any = ..., max_rows: Any = ..., min_rows: Any = ...) -> None:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self', 'buf'], name='to_string')
    def to_string(self, buf: Any = None, na_rep: str = 'NaN', float_format: Any = None, header: bool = True, index: bool = True, length: bool = False, dtype: bool = False, name: bool = False, max_rows: Any = None, min_rows: Any = None) -> str:
        ...

    @overload
    def to_markdown(self, buf: Any = ..., *, mode: Any = ..., index: Any = ..., storage_options: Any = ..., **kwargs: Any) -> str:
        ...

    @overload
    def to_markdown(self, buf: Any, *, mode: Any = ..., index: Any = ..., storage_options: Any = ..., **kwargs: Any) -> str:
        ...

    @overload
    def to_markdown(self, buf: Any, *, mode: Any = ..., index: Any = ..., storage_options: Any = ..., **kwargs: Any) -> str:
        ...

    @doc(klass=_shared_doc_kwargs['klass'], storage_options=_shared_docs['storage_options'], examples=dedent('Examples\n            --------\n            >>> s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")\n            >>> print(s.to_markdown())\n            |    | animal   |\n            |---:|:---------|\n            |  0 | elk      |\n            |  1 | pig      |\n            |  2 | dog      |\n            |  3 | quetzal  |\n\n            Output markdown with a tabulate option.\n\n            >>> print(s.to_markdown(tablefmt="grid"))\n            +----+----------+\n            |    | animal   |\n            +====+==========+\n            |  0 | elk      |\n            +----+----------+\n            |  1 | pig      |\n            +----+----------+\n            |  2 | dog      |\n            +----+----------+\n            |  3 | quetzal  |\n            +----+----------+'))
    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self', 'buf'], name='to_markdown')
    def to_markdown(self, buf: Any = None, mode: str = 'wt', index: bool = True, storage_options: Any = None, **kwargs: Any) -> str:
        ...

    def items(self) -> Iterable[tuple]:
        return zip(iter(self.index), iter(self))

    def keys(self) -> Index:
        return self.index

    @overload
    def to_dict(self, *, into: Any) -> MutableMappingT:
        ...

    @overload
    def to_dict(self, *, into: Any = ...) -> MutableMappingT:
        ...

    def to_dict(self, *, into: type = dict) -> MutableMappingT:
        ...

    def to_frame(self, name: Any = lib.no_default) -> DataFrame:
        ...

    def _set_name(self, name: Any, inplace: bool = False, deep: Any = None) -> Series:
        ...

    @Appender(dedent('\n        Examples\n        --------\n        >>> ser = pd.Series([390., 350., 30., 20.],\n        ...                 index=[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'],\n        ...                 name="Max Speed")\n        >>> ser\n        Falcon    390.0\n        Falcon    350.0\n        Parrot     30.0\n        Parrot     20.0\n        Name: Max Speed, dtype: float64\n\n        We can pass a list of values to group the Series data by custom labels:\n\n        >>> ser.groupby(["a", "b", "a", "b"]).mean()\n        a    210.0\n        b    185.0\n        Name: Max Speed, dtype: float64\n\n        Grouping by numeric labels yields similar results:\n\n        >>> ser.groupby([0, 1, 0, 1]).mean()\n        0    210.0\n        1    185.0\n        Name: Max Speed, dtype: float64\n\n        We can group by a level of the index:\n\n        >>> ser.groupby(level=0).mean()\n        Falcon    370.0\n        Parrot     25.0\n        Name: Max Speed, dtype: float64\n\n        We can group by a condition applied to the Series values:\n\n        >>> ser.groupby(ser > 100).mean()\n        Max Speed\n        False     25.0\n        True     370.0\n        Name: Max Speed, dtype: float64\n\n        **Grouping by Indexes**\n\n        We can groupby different levels of a hierarchical index\n        using the `level` parameter:\n\n        >>> arrays = [[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'],\n        ...           [\'Captive\', \'Wild\', \'Captive\', \'Wild\']]\n        >>> index = pd.MultiIndex.from_arrays(arrays, names=(\'Animal\', \'Type\'))\n        >>> ser = pd.Series([390., 350., 30., 20.], index=index, name="Max Speed")\n        >>> ser\n        Animal  Type\n        Falcon  Captive    390.0\n                Wild       350.0\n        Parrot  Captive     30.0\n                Wild        20.0\n        Name: Max Speed, dtype: float64\n\n        >>> ser.groupby(level=0).mean()\n        Animal\n        Falcon    370.0\n        Parrot     25.0\n        Name: Max Speed, dtype: float64\n\n        We can also group by the \'Type\' level of the hierarchical index\n        to get the mean speed for each type:\n\n        >>> ser.groupby(level="Type").mean()\n        Type\n        Captive    210.0\n        Wild       185.0\n        Name: Max Speed, dtype: float64\n\n        We can also choose to include `NA` in group keys or not by defining\n        `dropna` parameter, the default setting is `True`.\n\n        >>> ser = pd.Series([1, 2, 3, 3], index=["a", \'a\', \'b\', np.nan])\n        >>> ser.groupby(level=0).sum()\n        a    3\n        b    3\n        dtype: int64\n\n        To include `NA` values in the group keys, set `dropna=False`:\n\n        >>> ser.groupby(level=0, dropna=False).sum()\n        a    3\n        b    3\n        NaN  3\n        dtype: int64\n\n        We can also group by a custom list with NaN values to handle\n        missing group labels:\n\n        >>> arrays = [\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\']\n        >>> ser = pd.Series([390., 350., 30., 20.], index=arrays, name="Max Speed")\n        >>> ser.groupby(["a", "b", "a", np.nan]).mean()\n        a    210.0\n        b    350.0\n        Name: Max Speed, dtype: float64\n\n        >>> ser.groupby(["a", "b", "a", np.nan], dropna=False).mean()\n        a    210.0\n        b    350.0\n        NaN   20.0\n        Name: Max Speed, dtype: float64\n        '))
    @Appender(_shared_docs['groupby'] % _shared_doc_kwargs)
    def groupby(self, by: Any = None, level: Any = None, as_index: bool = True, sort: bool = True, group_keys: bool = True, observed: bool = False, dropna: bool = True) -> SeriesGroupBy:
        ...

    def count(self) -> int:
        return notna(self._values).sum().astype('int64')

    def mode(self, dropna: bool = True) -> Series:
        ...

    def unique(self) -> np.ndarray:
        ...

    @overload
    def drop_duplicates(self, *, keep: Any = ..., inplace: Any = ..., ignore_index: Any = ...) -> Series:
        ...

    @overload
    def drop_duplicates(self, *, keep: Any = ..., inplace: Any, ignore_index: Any = ...) -> Series:
        ...

    @overload
    def drop_duplicates(self, *, keep: Any = ..., inplace: Any = ..., ignore_index: Any = ...) -> Series:
        ...

    def drop_duplicates(self, *, keep: Literal['first', 'last', False] = 'first', inplace: bool = False, ignore_index: bool = False) -> Series:
        ...

    def duplicated(self, keep: Literal['first', 'last', False] = 'first') -> Series:
        ...

    def idxmin(self, axis: Any = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Index:
        ...

    def idxmax(self, axis: Any = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Index:
        ...

    def round(self, decimals: int = 0, *args: Any, **kwargs: Any) -> Series:
        ...

    @overload
    def quantile(self, q: float, interpolation: str = ...) -> float:
        ...

    @overload
    def quantile(self, q: list, interpolation: str = ...) -> Series:
        ...

    @overload
    def quantile(self, q: Any = ..., interpolation: str = ...) -> Any:
        ...

    def quantile(self, q: Any = 0.5, interpolation: str = 'linear') -> Any:
        ...

    def corr(self, other: Series, method: str = 'pearson', min_periods: Any = None) -> float:
        ...

    def cov(self, other: Series, min_periods: Any = None, ddof: int = 1) -> float:
        ...

    def diff(self, periods: int = 1) -> Series:
        ...

    def autocorr(self, lag: int = 1) -> float:
        ...

    def dot(self, other: Any) -> Any:
        ...

    def __matmul__(self, other: Any) -> Any:
        return self.dot(other)

    def __rmatmul__(self, other: Any) -> Any:
        return self.dot(np.transpose(other))

    @doc(base.IndexOpsMixin.searchsorted, klass='Series')
    def searchsorted(self, value: Any, side: str = 'left', sorter: Any = None) -> Any:
        ...

    def _append(self, to_append: Any, ignore_index: bool = False, verify_integrity: bool = False) -> Series:
        ...

    @doc(_shared_docs['compare'], dedent('\n        Returns\n        -------\n        Series or DataFrame\n            If axis is 0 or \'index\' the result will be a Series.\n            The resulting index will be a MultiIndex with \'self\' and \'other\'\n            stacked alternately at the inner level.\n\n            If axis is 1 or \'columns\' the result will be a DataFrame.\n            It will have two columns namely \'self\' and \'other\'.\n\n        See Also\n        --------\n        DataFrame.compare : Compare with another DataFrame and show differences.\n\n        Notes\n        -----\n        Matching NaNs will not appear as a difference.\n\n        Examples\n        --------\n        >>> s1 = pd.Series(["a", "b", "c", "d", "e"])\n        >>> s2 = pd.Series(["a", "a", "c", "b", "e"])\n\n        Align the differences on columns\n\n        >>> s1.compare(s2)\n          self other\n        1    b     a\n        3    d     b\n\n        Stack the differences on indices\n\n        >>> s1.compare(s2, align_axis=0)\n        1  self     b\n           other    a\n        3  self     d\n           other    b\n        dtype: object\n\n        Keep all original rows\n\n        >>> s1.compare(s2, keep_shape=True)\n          self other\n        0  NaN   NaN\n        1    b     a\n        2  NaN   NaN\n        3    d     b\n        4  NaN   NaN\n\n        Keep all original rows and also all original values\n\n        >>> s1.compare(s2, keep_shape=True, keep_equal=True)\n          self other\n        0    a     a\n        1    b     a\n        2    c     c\n        3    d     b\n        4    e     e\n        '))
    def compare(self, other: Any, align_axis: int = 1, keep_shape: bool = False, keep_equal: bool = False, result_names: tuple = ('self', 'other')) -> Series:
        ...

    def combine(self, other: Any, func: Callable, fill_value: Any = None) -> Series:
        ...

    def combine_first(self, other: Series) -> Series:
        ...

    def update(self, other: Any) -> None:
        ...

    @overload
    def sort_values(self, *, axis: Any = ..., ascending: Any = ..., inplace: Any = ..., kind: Any = ..., na_position: Any = ..., ignore_index: Any = ..., key: Any = ...) -> Series:
        ...

    @overload
    def sort_values(self, *, axis: Any = ..., ascending: Any = ..., inplace: Any, kind: Any = ..., na_position: Any = ..., ignore_index: Any = ..., key: Any = ...) -> Series:
        ...

    @overload
    def sort_values(self, *, axis: Any = ..., ascending: Any = ..., inplace: Any = ..., kind: Any = ..., na_position: Any = ..., ignore_index: Any = ..., key: Any = ...) -> Series:
        ...

    def sort_values(self, *, axis: int = 0, ascending: bool = True, inplace: bool = False, kind: str = 'quicksort', na_position: str = 'last', ignore_index: bool = False, key: Callable = None) -> Series:
        ...

    @overload
    def sort_index(self, *, axis: Any = ..., level: Any = ..., ascending: Any = ..., inplace: Any = ..., kind: Any = ..., na_position: Any = ..., sort_remaining: Any = ..., ignore_index: Any = ..., key: Any = ...) -> Series:
        ...

    @overload
    def sort_index(self, *, axis: Any = ..., level: Any = ..., ascending: Any = ..., inplace: Any, kind: Any = ..., na_position: Any = ..., sort_remaining: Any = ..., ignore_index: Any = ..., key: Any = ...) -> Series:
        ...

    @overload
    def sort_index(self, *, axis: Any = ..., level: Any = ..., ascending: Any = ..., inplace: Any = ..., kind: Any = ..., na_position: Any = ..., sort_remaining: Any = ..., ignore_index: Any = ..., key: Any = ...) -> Series:
        ...

    def sort_index(self, *, axis: int = 0, level: Any = None, ascending: bool = True, inplace: bool = False, kind: str = 'quicksort', na_position: str = 'last', sort_remaining: bool = True, ignore_index: bool = False, key: Callable = None) -> Series:
        ...

    def argsort(self, axis: Any = 0, kind: str = 'quicksort', order: Any = None, stable: Any = None) -> Series:
        ...

    def nlargest(self, n: int = 5, keep: Literal['first', 'last', 'all'] = 'first') -> Series:
        ...

    def nsmallest(self, n: int = 5, keep: Literal['first', 'last', 'all'] = 'first') -> Series:
        ...

    def swaplevel(self, i: int = -2, j: int = -1, copy: Any = lib.no_default) -> Series:
        ...

    def reorder_levels(self, order: list) -> Series:
        ...

    def explode(self, ignore_index: bool = False) -> Series:
        ...

    def unstack(self, level: Any = -1, fill_value: Any = None, sort: bool = True) -> DataFrame:
        ...

    def map(self, arg: Any, na_action: Any = None, **kwargs: Any) -> Series:
        ...

    def _gotitem(self, key: Any, ndim: int, subset: Any = None) -> Series:
        ...

    @doc(_shared_docs['aggregate'], klass=_shared_doc_kwargs['klass'], axis=_shared_doc_kwargs['axis'], see_also=_agg_see_also_doc, examples=_agg_examples_doc)
    def aggregate(self, func: Any = None, axis: int = 0, *args: Any, **kwargs: Any) -> Any:
        ...

    agg = aggregate

    @doc(_shared_docs['transform'], klass=_shared_doc_kwargs['klass'], axis=_shared_doc_kwargs['axis'])
    def transform(self, func: Any, axis: int = 0, *args: Any, **kwargs: Any) -> Any:
        ...

    def apply(self, func: Any, args: Any = (), *, by_row: str = 'compat', **kwargs: Any) -> Any:
        ...

    def _reindex_indexer(self, new_index: Any, indexer: Any) -> Series:
        ...

    def _needs_reindex_multi(self, axes: Any, method: Any, level: Any) -> bool:
        return False

    @overload
    def rename(self, index: Any = ..., *, axis: Any = ..., copy: Any = ..., inplace: Any = ..., level: Any = ..., errors: Any = ...) -> Series:
        ...

    @overload
    def rename(self, index: Any = ..., *, axis: Any = ..., copy: Any = ..., inplace: Any, level: Any = ..., errors: Any = ...) -> Series:
        ...

    @overload
    def rename(self, index: Any = ..., *, axis: Any = ..., copy: Any = ..., inplace: Any = ..., level: Any = ..., errors: Any = ...) -> Series:
        ...

    def rename(self, index: Any = None, *, axis: Any = None, copy: Any = lib.no_default, inplace: bool = False, level: Any = None, errors: str = 'ignore') -> Series:
        ...

    @Appender("\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3])\n        >>> s\n        0    1\n        1    2\n        2    3\n        dtype: int64\n\n        >>> s.set_axis(['a', 'b', 'c'], axis=0)\n        a    1\n        b    2\n        c    3\n        dtype: int64\n    ")
    @Substitution(klass=_shared_doc_kwargs['klass'], axes_single_arg=_shared_doc_kwargs['axes_single_arg'], extended_summary_sub='', axis_description_sub='', see_also_sub='')
    @Appender(NDFrame.set_axis.__doc__)
    def set_axis(self, labels: Any, *, axis: int = 0, copy: Any = lib.no_default) -> Series:
        ...

    @doc(NDFrame.reindex, klass=_shared_doc_kwargs['klass'], optional_reindex=_shared_doc_kwargs['optional_reindex'])
    def reindex(self, index: Any = None, *, axis: Any = None, method: Any = None, copy: Any = lib.no_default, level: Any = None, fill_value: Any = None, limit: Any = None, tolerance: Any = None) -> Series:
        ...

    @overload
    def rename_axis(self, mapper: Any = ..., *, index: Any = ..., axis: Any = ..., copy: Any = ..., inplace: Any) -> Series:
        ...

    @overload
    def rename_axis(self, mapper: Any = ..., *, index: Any = ..., axis: Any = ..., copy: Any = ..., inplace: Any = ...) -> Series:
        ...

    @overload
    def rename_axis(self, mapper: Any = ..., *, index: Any = ..., axis: Any = ..., copy: Any = ..., inplace: Any = ...) -> Series:
        ...

    def rename_axis(self, mapper: Any = lib.no_default, *, index: Any = lib.no_default, axis: int = 0, copy: Any = lib.no_default, inplace: bool = False) -> Series:
        ...

    @overload
    def drop(self, labels: Any = ..., *, axis: Any = ..., index: Any = ..., columns: Any = ..., level: Any = ..., inplace: Any, errors: Any = ...) -> Series:
        ...

    @overload
    def drop(self, labels: Any = ..., *, axis: Any = ..., index: Any = ..., columns: Any = ..., level: Any = ..., inplace: Any, errors: Any = ...) -> Series:
        ...

    @overload
    def drop(self, labels: Any = ..., *, axis: Any = ..., index: Any = ..., columns: Any = ..., level: Any = ..., inplace: Any, errors: Any = ...) -> Series:
        ...

    def drop(self, labels: Any = None, *, axis: int = 0, index: Any = None, columns: Any = None, level: Any = None, inplace: bool = False, errors: str = 'raise') -> Series:
        ...

    def pop(self, item: Any) -> Any:
        ...

    @doc(INFO_DOCSTRING, **series_sub_kwargs)
    def info(self, verbose: Any = None, buf: Any = None, max_cols: Any = None, memory_usage: Any = None, show_counts: bool = True) -> None:
        ...

    def memory_usage(self, index: bool = True, deep: bool = False) -> int:
        ...

    def isin(self, values: Any) -> Series:
        ...

    def between(self, left: Any, right: Any, inclusive: str = 'both') -> Series:
        ...

    def case_when(self, caselist: Any) -> Series:
        ...

    @doc(NDFrame.isna, klass=_shared_doc_kwargs['klass'])
    def isna(self) -> Series:
        ...

    @doc(NDFrame.isna, klass=_shared_doc_kwargs['klass'])
    def isnull(self) -> Series:
        ...

    @doc(NDFrame.notna, klass=_shared_doc_kwargs['klass'])
    def notna(self) -> Series:
        ...

    @doc(NDFrame.notna, klass=_shared_doc_kwargs['klass'])
    def notnull(self) -> Series:
        ...

    @overload
    def dropna(self, *, axis: Any = ..., inplace: Any = ..., how: Any = ..., ignore_index: Any = ...) -> Series:
        ...

    @overload
    def dropna(self, *, axis: Any = ..., inplace: Any, how: Any = ..., ignore_index: Any = ...) -> Series:
        ...

    def dropna(self, *, axis: int = 0, inplace: bool = False, how: Any = None, ignore_index: bool = False) -> Series:
        ...

    def to_timestamp(self, freq: Any = None, how: str = 'start', copy: Any = lib.no_default) -> Series:
        ...

    def to_period(self, freq: Any = None, copy: Any = lib.no_default) -> Series:
        ...

    @property
    def _AXIS_ORDERS(self) -> list:
        return ['index']

    @property
    def _AXIS_LEN(self) -> int:
        return len(self._AXIS_ORDERS)

    @property
    def _info_axis_number(self) -> int:
        return 0

    @property
    def _info_axis_name(self) -> str:
        return 'index'

    @property
    def index(self) -> Index:
        ...

    str = Accessor('str', StringMethods)
    dt = Accessor('dt', CombinedDatetimelikeProperties)
    cat = Accessor('cat', CategoricalAccessor)
    plot = Accessor('plot', pandas.plotting.PlotAccessor)
    sparse = Accessor('sparse', SparseAccessor)
    struct = Accessor('struct', StructAccessor)
    list = Accessor('list', ListAccessor)
    hist = pandas.plotting.hist_series

    def _cmp_method(self, other: Any, op: Any) -> Series:
        ...

    def _logical_method(self, other: Any, op: Any) -> Series:
        ...

    def _arith_method(self, other: Any, op: Any) -> Series:
        ...

    def _align_for_op(self, right: Any, align_asobject: bool = False) -> tuple:
        ...

    def _binop(self, other: Any, func: Any, level: Any = None, fill_value: Any = None) -> Series:
        ...

    def _construct_result(self, result: Any, name: Any) -> Series:
        ...

    def _reduce(self, op: Any, name: Any, *, axis: Any = 0, skipna: bool = True, numeric_only: bool = False, filter_type: Any = None, **kwds: Any) -> Any:
        ...

    @Appender(make_doc('any', ndim=1))
    def any(self, *, axis: Any = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='all')
    @Appender(make_doc('all', ndim=1))
    def all(self, axis: int = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='min')
    def min(self, axis: Any = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='max')
    def max(self, axis: Any = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='sum')
    def sum(self, axis: Any = None, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='prod')
    @doc(make_doc('prod', ndim=1))
    def prod(self, axis: Any = None, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='mean')
    def mean(self, axis: Any = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='median')
    def median(self, axis: Any = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='sem')
    @doc(make_doc('sem', ndim=1))
    def sem(self, axis: Any = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='var')
    def var(self, axis: Any = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='std')
    @doc(make_doc('std', ndim=1))
    def std(self, axis: Any = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='skew')
    @doc(make_doc('skew', ndim=1))
    def skew(self, axis: Any = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='kurt')
    def kurt(self, axis: Any = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        ...

    kurtosis = kurt
    product = prod

    @doc(make_doc('cummin', ndim=1))
    def cummin(self, axis: Any = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        ...

    @doc(make_doc('cummax', ndim=1))
    def cummax(self, axis: Any = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        ...

    @doc(make_doc('cumsum', ndim=1))
    def cumsum(self, axis: Any = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        ...

    @doc(make_doc('cumprod', 1))
    def cumprod(self, axis: Any = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        ...

    def eq(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    @Appender(ops.make_flex_doc('ne', 'series'))
    def ne(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    def le(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    @Appender(ops.make_flex_doc('lt', 'series'))
    def lt(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    def ge(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    @Appender(ops.make_flex_doc('gt', 'series'))
    def gt(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    def add(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    @Appender(ops.make_flex_doc('radd', 'series'))
    def radd(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    @Appender(ops.make_flex_doc('sub', 'series'))
    def sub(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    subtract = sub

    @Appender(ops.make_flex_doc('rsub', 'series'))
    def rsub(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    def mul(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    @Appender(ops.make_flex_doc('rmul', 'series'))
    def rmul(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    def truediv(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    @Appender(ops.make_flex_doc('rtruediv', 'series'))
    def rtruediv(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    rdiv = rtruediv

    @Appender(ops.make_flex_doc('floordiv', 'series'))
    def floordiv(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    @Appender(ops.make_flex_doc('rfloordiv', 'series'))
    def rfloordiv(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    def mod(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    @Appender(ops.make_flex_doc('rmod', 'series'))
    def rmod(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    def pow(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    @Appender(ops.make_flex_doc('rpow', 'series'))
    def rpow(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> Series:
        ...

    def divmod(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> tuple:
        ...

    @Appender(ops.make_flex_doc('rdivmod', 'series'))
    def rdivmod(self, other: Any, level: Any = None, fill_value: Any = None, axis: Any = 0) -> tuple:
        ...
