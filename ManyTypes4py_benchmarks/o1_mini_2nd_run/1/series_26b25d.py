"""
Data structure for 1-dimensional cross-sectional and time series data
"""
from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import functools
import operator
import sys
from textwrap import dedent
from typing import IO, TYPE_CHECKING, Any, Literal, cast, overload, Optional, Union, Sequence as TypingSequence
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
    """
    One-dimensional ndarray with axis labels (including time series).

    Labels need not be unique but must be a hashable type. The object
    supports both integer- and label-based indexing and provides a host of
    methods for performing operations involving the index. Statistical
    methods from ndarray have been overridden to automatically exclude
    missing data (currently represented as NaN).

    Operations between Series (+, -, /, \*, \\*\\*) align values based on their
    associated index values-- they need not be the same length. The result
    index will be the sorted union of the two indexes.

    Parameters
    ----------
    data : array-like, Iterable, dict, or scalar value
        Contains data stored in Series. If data is a dict, argument order is
        maintained. Unordered sets are not supported.
    index : array-like or Index (1d)
        Values must be hashable and have the same length as `data`.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, ..., n) if not provided. If data is dict-like
        and index is None, then the keys in the data are used as the index. If the
        index is not None, the resulting Series is reindexed with the index values.
    dtype : Union[str, np.dtype, ExtensionDtype], optional
        Data type for the output Series. If not specified, this will be
        inferred from `data`.
        See the :ref:`user guide <basics.dtypes>` for more usages.
        If ``data`` is Series then is ignored.
    name : Hashable, default None
        The name to give to the Series.
    copy : bool, default False
        Copy input data. Only affects Series or 1d ndarray input. See examples.

    See Also
    --------
    DataFrame : Two-dimensional, size-mutable, potentially heterogeneous tabular data.
    Index : Immutable sequence used for indexing and alignment.

    Notes
    -----
    Please reference the :ref:`User Guide <basics.series>` for more information.

    Examples
    --------
    Constructing Series from a dictionary with an Index specified

    >>> d = {"a": 1, "b": 2, "c": 3}
    >>> ser = pd.Series(data=d, index=["a", "b", "c"])
    >>> ser
    a   1
    b   2
    c   3
    dtype: int64

    The keys of the dictionary match with the Index values, hence the Index
    values have no effect.

    >>> d = {"a": 1, "b": 2, "c": 3}
    >>> ser = pd.Series(data=d, index=["x", "y", "z"])
    >>> ser
    x   NaN
    y   NaN
    z   NaN
    dtype: float64

    Note that the Index is first built with the keys from the dictionary.
    After this the Series is reindexed with the given Index values, hence we
    get all NaN as a result.

    Constructing Series from a list with `copy=False`.

    >>> r = [1, 2]
    >>> ser = pd.Series(r, copy=False)
    >>> ser.iloc[0] = 999
    >>> r
    [1, 2]
    >>> ser
    0    999
    1      2
    dtype: int64

    Due to input data type the Series has a `copy` of
    the original data even though `copy=False`, so
    the data is unchanged.

    Constructing Series from a 1d ndarray with `copy=False`.

    >>> r = np.array([1, 2])
    >>> ser = pd.Series(r, copy=False)
    >>> ser.iloc[0] = 999
    >>> r
    array([999,   2])
    >>> ser
    0    999
    1      2
    dtype: int64

    Due to input data type the Series has a `view` on
    the original data, so
    the data is changed as well.
    """
    _typ: str = 'series'
    _HANDLED_TYPES: tuple = (Index, ExtensionArray, np.ndarray)
    _metadata: list = ['_name']
    _internal_names_set: set = {'index', 'name'} | NDFrame._internal_names_set
    _accessors: dict[str, Callable] = {'dt', 'cat', 'str', 'sparse'}
    _hidden_attrs: frozenset[str] = base.IndexOpsMixin._hidden_attrs | NDFrame._hidden_attrs | frozenset([])
    __pandas_priority__: int = 3000
    hasnans = property(base.IndexOpsMixin.hasnans.fget, doc=base.IndexOpsMixin.hasnans.__doc__)

    def __init__(
        self,
        data: Optional[
            Union[
                array_like,
                Iterable[Any],
                Mapping[Any, Any],
                Scalar
            ]
        ] = None,
        index: Optional[Index] = None,
        dtype: Optional[Union[str, np.dtype, ExtensionDtype]] = None,
        name: Optional[Hashable] = None,
        copy: Optional[bool] = None
    ) -> None:
        allow_mgr = False
        if isinstance(data, SingleBlockManager) and index is None and (dtype is None) and (copy is False or copy is None):
            if not allow_mgr:
                warnings.warn(f'Passing a {type(data).__name__} to {type(self).__name__} is deprecated and will raise in a future version. Use public APIs instead.', DeprecationWarning, stacklevel=2)
            data = data.copy(deep=False)
            NDFrame.__init__(self, data)
            self.name = name
            return
        if isinstance(data, (ExtensionArray, np.ndarray)):
            if copy is not False:
                if dtype is None or astype_is_view(data.dtype, pandas_dtype(dtype)):
                    data = data.copy()
        if copy is None:
            copy = False
        if isinstance(data, SingleBlockManager) and (not copy):
            data = data.copy(deep=False)
            if not allow_mgr:
                warnings.warn(f'Passing a {type(data).__name__} to {type(self).__name__} is deprecated and will raise in a future version. Use public APIs instead.', DeprecationWarning, stacklevel=2)
        name = ibase.maybe_extract_name(name, data, type(self))
        if index is not None:
            index = ensure_index(index)
        if dtype is not None:
            dtype = self._validate_dtype(dtype)
        if data is None:
            index = index if index is not None else default_index(0)
            if len(index) or dtype is not None:
                data = na_value_for_dtype(pandas_dtype(dtype), compat=False)
            else:
                data = []
        if isinstance(data, MultiIndex):
            raise NotImplementedError('initializing a Series from a MultiIndex is not supported')
        refs: Optional[BlockValuesRefs] = None
        if isinstance(data, Index):
            if dtype is not None:
                data = data.astype(dtype)
            refs = data._references
            copy = False
        elif isinstance(data, np.ndarray):
            if len(data.dtype):
                raise ValueError('Cannot construct a Series from an ndarray with compound dtype.  Use DataFrame instead.')
        elif isinstance(data, Series):
            if index is None:
                index = data.index
                data = data._mgr.copy(deep=False)
            else:
                data = data.reindex(index)
                copy = False
                data = data._mgr
        elif isinstance(data, Mapping):
            data, index = self._init_dict(data, index, dtype)
            dtype = None
            copy = False
        elif isinstance(data, SingleBlockManager):
            if index is None:
                index = data.index
            elif not data.index.equals(index) or copy:
                raise AssertionError('Cannot pass both SingleBlockManager `data` argument and a different `index` argument. `copy` must be False.')
            if not allow_mgr:
                warnings.warn(f'Passing a {type(data).__name__} to {type(self).__name__} is deprecated and will raise in a future version. Use public APIs instead.', DeprecationWarning, stacklevel=2)
                allow_mgr = True
        elif isinstance(data, ExtensionArray):
            pass
        else:
            data = com.maybe_iterable_to_list(data)
            if is_list_like(data) and (not len(data)) and (dtype is None):
                dtype = np.dtype(object)
        if index is None:
            if not is_list_like(data):
                data = [data]
            index = default_index(len(data))
        elif is_list_like(data):
            com.require_length_match(data, index)
        if isinstance(data, SingleBlockManager):
            if dtype is not None:
                data = data.astype(dtype=dtype)
            elif copy:
                data = data.copy()
        else:
            data = sanitize_array(data, index, dtype, copy)
            data = SingleBlockManager.from_array(data, index, refs=refs)
        NDFrame.__init__(self, data)
        self.name = name
        self._set_axis(0, index)

    def _init_dict(
        self,
        data: Mapping[Any, Any],
        index: Optional[Index] = None,
        dtype: Optional[Union[str, np.dtype, ExtensionDtype]] = None
    ) -> tuple[SingleBlockManager, Index]:
        """
        Derive the "_mgr" and "index" attributes of a new Series from a
        dictionary input.

        Parameters
        ----------
        data : dict or dict-like
            Data used to populate the new Series.
        index : Index or None, default None
            Index for the new Series: if None, use dict keys.
        dtype : np.dtype, ExtensionDtype, or None, default None
            The dtype for the new Series: if None, infer from data.

        Returns
        -------
        _data : BlockManager for the new Series
        index : index for the new Series
        """
        if data:
            keys = maybe_sequence_to_range(tuple(data.keys()))
            values = list(data.values())
        elif index is not None:
            if len(index) or dtype is not None:
                values = na_value_for_dtype(pandas_dtype(dtype), compat=False)
            else:
                values = []
            keys = index
        else:
            keys, values = (default_index(0), [])
        s: Series = Series(values, index=keys, dtype=dtype)
        if data and index is not None:
            s = s.reindex(index)
        return (s._mgr, s.index)

    def __arrow_c_stream__(self, requested_schema: Optional[IO] = None) -> Any:
        """
        Export the pandas Series as an Arrow C stream PyCapsule.

        This relies on pyarrow to convert the pandas Series to the Arrow
        format (and follows the default behavior of ``pyarrow.Array.from_pandas``
        in its handling of the index, i.e. to ignore it).
        This conversion is not necessarily zero-copy.

        Parameters
        ----------
        requested_schema : PyCapsule, default None
            The schema to which the dataframe should be casted, passed as a
            PyCapsule containing a C ArrowSchema representation of the
            requested schema.

        Returns
        -------
        PyCapsule
        """
        pa = import_optional_dependency('pyarrow', min_version='16.0.0')
        type_: Optional[Any] = pa.DataType._import_from_c_capsule(requested_schema) if requested_schema is not None else None
        ca = pa.array(self, type=type_)
        if not isinstance(ca, pa.ChunkedArray):
            ca = pa.chunked_array([ca])
        return ca.__arrow_c_stream__()

    @property
    def _constructor(self) -> type[Series]:
        return Series

    def _constructor_from_mgr(self, mgr: SingleBlockManager, axes: list[Index]) -> Series:
        ser = Series._from_mgr(mgr, axes=axes)
        ser._name = None
        if type(self) is Series:
            return ser
        return self._constructor(ser)

    @property
    def _constructor_expanddim(self) -> type[NDFrame]:
        """
        Used when a manipulation result has one higher dimension as the
        original, such as Series.to_frame()
        """
        from pandas.core.frame import DataFrame
        return DataFrame

    def _constructor_expanddim_from_mgr(self, mgr: SingleBlockManager, axes: list[Index]) -> NDFrame:
        from pandas.core.frame import DataFrame
        df = DataFrame._from_mgr(mgr, axes=mgr.axes)
        if type(self) is Series:
            return df
        return self._constructor_expanddim(df)

    @property
    def _can_hold_na(self) -> bool:
        return self._mgr._can_hold_na

    @property
    def dtype(self) -> Union[np.dtype, ExtensionDtype]:
        """
        Return the dtype object of the underlying data.

        See Also
        --------
        Series.dtypes : Return the dtype object of the underlying data.
        Series.astype : Cast a pandas object to a specified dtype dtype.
        Series.convert_dtypes : Convert columns to the best possible dtypes using dtypes
            supporting pd.NA.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.dtype
        dtype('int64')
        """
        return self._mgr.dtype

    @property
    def dtypes(self) -> Union[np.dtype, ExtensionDtype]:
        """
        Return the dtype object of the underlying data.

        See Also
        --------
        DataFrame.dtypes :  Return the dtypes in the DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.dtypes
        dtype('int64')
        """
        return self.dtype

    @property
    def name(self) -> Optional[Hashable]:
        """
        Return the name of the Series.

        The name of a Series becomes its index or column name if it is used
        to form a DataFrame. It is also used whenever displaying the Series
        using the interpreter.

        Returns
        -------
        Optional[Hashable]
            The name of the Series, also the column name if part of a DataFrame.

        See Also
        --------
        Series.rename : Sets the Series name when given a scalar input.
        Index.name : Corresponding Index property.

        Examples
        --------
        The Series name can be set initially when calling the constructor.

        >>> s = pd.Series([1, 2, 3], dtype=np.int64, name="Numbers")
        >>> s
        0    1
        1    2
        2    3
        Name: Numbers, dtype: int64
        >>> s.name = "Integers"
        >>> s
        0    1
        1    2
        2    3
        Name: Integers, dtype: int64

        The name of a Series within a DataFrame is its column name.

        >>> df = pd.DataFrame(
        ...     [[1, 2], [3, 4], [5, 6]], columns=["Odd Numbers", "Even Numbers"]
        ... )
        >>> df
           Odd Numbers  Even Numbers
        0            1             2
        1            3             4
        2            5             6
        >>> df["Even Numbers"].name
        'Even Numbers'
        """
        return self._name

    @name.setter
    def name(self, value: Optional[Hashable]) -> None:
        validate_all_hashable(value, error_name=f'{type(self).__name__}.name')
        object.__setattr__(self, '_name', value)

    @property
    def values(self) -> Union[np.ndarray, ExtensionArray]:
        """
        Return Series as ndarray or ndarray-like depending on the dtype.

        .. warning::

           We recommend using :attr:`Series.array` or
           :meth:`Series.to_numpy`, depending on whether you need
           a reference to the underlying data or a NumPy array.

        Returns
        -------
        Union[np.ndarray, ExtensionArray]
            The unique values returned as a NumPy array. See Notes.

        See Also
        --------
        Series.array : Reference to the underlying data.
        Series.to_numpy : A NumPy array representing the underlying data.

        Examples
        --------
        >>> pd.Series([1, 2, 3]).values
        array([1, 2, 3])

        >>> pd.Series(list("aabc")).values
        array(['a', 'a', 'b', 'c'], dtype=object)

        >>> pd.Series(list("aabc")).astype("category").values
        ['a', 'a', 'b', 'c']
        Categories (3, object): ['a', 'b', 'c']

        Timezone aware datetime data is converted to UTC:

        >>> pd.Series(pd.date_range("20130101", periods=3, tz="US/Eastern")).values
        array(['2013-01-01T05:00:00.000000000',
               '2013-01-02T05:00:00.000000000',
               '2013-01-03T05:00:00.000000000'], dtype='datetime64[ns]')
        """
        return self._mgr.external_values()

    @property
    def _values(self) -> Union[np.ndarray, ExtensionArray]:
        """
        Return the internal repr of this data (defined by Block.interval_values).
        This are the values as stored in the Block (ndarray or ExtensionArray
        depending on the Block class), with datetime64[ns] and timedelta64[ns]
        wrapped in ExtensionArrays to match Index._values behavior.

        Differs from the public ``.values`` for certain data types, because of
        historical backwards compatibility of the public attribute (e.g. period
        returns object ndarray and datetimetz a datetime64[ns] ndarray for
        ``.values`` while it returns an ExtensionArray for ``._values`` in those
        cases).

        Differs from ``.array`` in that this still returns the numpy array if
        the Block is backed by a numpy array (except for datetime64 and
        timedelta64 dtypes), while ``.array`` ensures to always return an
        ExtensionArray.

        Overview:

        dtype       | values        | _values       | array                 |
        ----------- | ------------- | ------------- | --------------------- |
        Numeric     | ndarray       | ndarray       | NumpyExtensionArray   |
        Category    | Categorical   | Categorical   | Categorical           |
        dt64[ns]    | ndarray[M8ns] | DatetimeArray | DatetimeArray         |
        dt64[ns tz] | ndarray[M8ns] | DatetimeArray | DatetimeArray         |
        td64[ns]    | ndarray[m8ns] | TimedeltaArray| TimedeltaArray        |
        Period      | ndarray[obj]  | PeriodArray   | PeriodArray           |
        Nullable    | EA            | EA            | EA                    |

        """
        return self._mgr.internal_values()

    @property
    def _references(self) -> Any:
        return self._mgr._block.refs

    @Appender(base.IndexOpsMixin.array.__doc__)
    @property
    def array(self) -> ExtensionArray:
        return self._mgr.array_values()

    def __len__(self) -> int:
        """
        Return the length of the Series.
        """
        return len(self._mgr)

    def __array__(
        self,
        dtype: Optional[Union[str, np.dtype]] = None,
        copy: Optional[bool] = None
    ) -> np.ndarray:
        """
        Return the values as a NumPy array.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : Union[str, np.dtype], optional
            The dtype to use for the resulting NumPy array. By default,
            the dtype is inferred from the data.

        copy : Optional[bool], optional
            See :func:`numpy.asarray`.

        Returns
        -------
        np.ndarray
            The values in the series converted to a :class:`numpy.ndarray`
            with the specified `dtype`.

        See Also
        --------
        array : Create a new array from data.
        Series.array : Zero-copy view to the array backing the Series.
        Series.to_numpy : Series method for similar behavior.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])
        >>> np.asarray(ser)
        array([1, 2, 3])

        For timezone-aware data, the timezones may be retained with
        ``dtype='object'``

        >>> tzser = pd.Series(pd.date_range("2000", periods=2, tz="CET"))
        >>> np.asarray(tzser, dtype="object")
        array([Timestamp('2000-01-01 00:00:00+0100', tz='CET'),
               Timestamp('2000-01-02 00:00:00+0100', tz='CET')],
              dtype=object)

        Or the values may be localized to UTC and the tzinfo discarded with
        ``dtype='datetime64[ns]'``

        >>> np.asarray(tzser, dtype="datetime64[ns]")  # doctest: +ELLIPSIS
        array(['1999-12-31T23:00:00.000000000', ...],
              dtype='datetime64[ns]')
        """
        values = self._values
        if copy is None:
            arr = np.asarray(values, dtype=dtype)
        else:
            arr = np.array(values, dtype=dtype, copy=copy)
        if copy is True:
            return arr
        if copy is False or astype_is_view(values.dtype, arr.dtype):
            arr = arr.view()
            arr.flags.writeable = False
        return arr

    @property
    def axes(self) -> list[Index]:
        """
        Return a list of the row axis labels.
        """
        return [self.index]

    def _ixs(self, i: int, axis: int = 0) -> Any:
        """
        Return the i-th value or values in the Series by location.

        Parameters
        ----------
        i : int

        Returns
        -------
        Any
        """
        return self._values[i]

    def _slice(self, slobj: slice, axis: int = 0) -> Series:
        mgr = self._mgr.get_slice(slobj, axis=axis)
        out = self._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self._name
        return out.__finalize__(self)

    def __getitem__(self, key: Any) -> Union[Any, Series]:
        check_dict_or_set_indexers(key)
        key = com.apply_if_callable(key, self)
        if key is Ellipsis:
            return self.copy(deep=False)
        key_is_scalar = is_scalar(key)
        if isinstance(key, (list, tuple)):
            key = unpack_1tuple(key)
        elif key_is_scalar:
            return self._get_value(key)
        if is_iterator(key):
            key = list(key)
        if is_hashable(key) and (not isinstance(key, slice)):
            try:
                result = self._get_value(key)
                return result
            except (KeyError, TypeError, InvalidIndexError):
                if isinstance(key, tuple) and isinstance(self.index, MultiIndex):
                    return self._get_values_tuple(key)
        if isinstance(key, slice):
            return self._getitem_slice(key)
        if com.is_bool_indexer(key):
            key = check_bool_indexer(self.index, key)
            key = np.asarray(key, dtype=bool)
            return self._get_rows_with_mask(key)
        return self._get_with(key)

    def _get_with(self, key: Any) -> Union[Any, Series]:
        if isinstance(key, ABCDataFrame):
            raise TypeError('Indexing a Series with DataFrame is not supported, use the appropriate DataFrame column')
        elif isinstance(key, tuple):
            return self._get_values_tuple(key)
        return self.loc[key]

    def _get_values_tuple(self, key: tuple) -> Series:
        if com.any_none(*key):
            result = np.asarray(self._values[key])
            disallow_ndim_indexing(result)
            return self._constructor(result, index=self.index, copy=False)
        if not isinstance(self.index, MultiIndex):
            raise KeyError('key of type tuple not found and not a MultiIndex')
        indexer, new_index = self.index.get_loc_level(key)
        new_ser = self._constructor(self._values[indexer], index=new_index, copy=False)
        if isinstance(indexer, slice):
            new_ser._mgr.add_references(self._mgr)
        return new_ser.__finalize__(self)

    def _get_rows_with_mask(self, indexer: np.ndarray) -> Series:
        new_mgr = self._mgr.get_rows_with_mask(indexer)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self)

    def _get_value(self, label: Any, takeable: bool = False) -> Any:
        """
        Quickly retrieve single value at passed index label.

        Parameters
        ----------
        label : object
        takeable : interpret the index as indexers, default False

        Returns
        -------
        Any
            Scalar value.
        """
        if takeable:
            return self._values[label]
        loc = self.index.get_loc(label)
        if is_integer(loc):
            return self._values[loc]
        if isinstance(self.index, MultiIndex):
            mi = self.index
            new_values = self._values[loc]
            if len(new_values) == 1 and mi.nlevels == 1:
                return new_values[0]
            new_index = mi[loc]
            new_index = maybe_droplevels(new_index, label)
            new_ser = self._constructor(new_values, index=new_index, name=self.name, copy=False)
            if isinstance(loc, slice):
                new_ser._mgr.add_references(self._mgr)
            return new_ser.__finalize__(self)
        else:
            return self.iloc[loc]

    def __setitem__(self, key: Any, value: Any) -> None:
        if not PYPY:
            if sys.getrefcount(self) <= 3:
                warnings.warn(_chained_assignment_msg, ChainedAssignmentError, stacklevel=2)
        check_dict_or_set_indexers(key)
        key = com.apply_if_callable(key, self)
        if key is Ellipsis:
            key = slice(None)
        if isinstance(key, slice):
            indexer = self.index._convert_slice_indexer(key, kind='getitem')
            return self._set_values(indexer, value)
        try:
            self._set_with_engine(key, value)
        except KeyError:
            self.loc[key] = value
        except (TypeError, ValueError, LossySetitemError):
            indexer = self.index.get_loc(key)
            self._set_values(indexer, value)
        except InvalidIndexError as err:
            if isinstance(key, tuple) and (not isinstance(self.index, MultiIndex)):
                raise KeyError('key of type tuple not found and not a MultiIndex') from err
            if com.is_bool_indexer(key):
                key = check_bool_indexer(self.index, key)
                key = np.asarray(key, dtype=bool)
                if is_list_like(value) and len(value) != len(self) and (not isinstance(value, Series)) and (not is_object_dtype(self.dtype)):
                    indexer = key.nonzero()[0]
                    self._set_values(indexer, value)
                    return
                try:
                    self._where(~key, value, inplace=True)
                except InvalidIndexError:
                    self.iloc[key] = value
                return
            else:
                self._set_with(key, value)

    def _set_with_engine(self, key: Any, value: Any) -> None:
        loc = self.index.get_loc(key)
        self._mgr.setitem_inplace(loc, value)

    def _set_with(self, key: Any, value: Any) -> None:
        assert not isinstance(key, tuple)
        if is_iterator(key):
            key = list(key)
        self._set_labels(key, value)

    def _set_labels(self, key: Any, value: Any) -> None:
        key = com.asarray_tuplesafe(key)
        indexer = self.index.get_indexer(key)
        mask = indexer == -1
        if mask.any():
            raise KeyError(f'{key[mask]} not in index')
        self._set_values(indexer, value)

    def _set_values(self, key: Union[int, slice, np.ndarray], value: Any) -> None:
        if isinstance(key, (Index, Series)):
            key = key._values
        self._mgr = self._mgr.setitem(indexer=key, value=value)

    def _set_value(self, label: Any, value: Any, takeable: bool = False) -> None:
        """
        Quickly set single value at passed label.

        If label is not contained, a new object is created with the label
        placed at the end of the result index.

        Parameters
        ----------
        label : object
            Partial indexing with MultiIndex not allowed.
        value : object
            Scalar value.
        takeable : interpret the index as indexers, default False
        """
        if not takeable:
            try:
                loc = self.index.get_loc(label)
            except KeyError:
                self.loc[label] = value
                return
        else:
            loc = label
        self._set_values(loc, value)

    def repeat(
        self,
        repeats: Union[int, TypingSequence[int]],
        axis: Optional[Any] = None
    ) -> Series:
        """
        Repeat elements of a Series.

        Returns a new Series where each element of the current Series
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Series.
        axis : None
            Unused. Parameter needed for compatibility with DataFrame.

        Returns
        -------
        Series
            Newly created Series with repeated elements.

        See Also
        --------
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c"])
        >>> s
        0    a
        1    b
        2    c
        dtype: object
        >>> s.repeat(2)
        0    a
        0    a
        1    b
        1    b
        2    c
        2    c
        dtype: object
        >>> s.repeat([1, 2, 3])
        0    a
        1    b
        1    b
        2    c
        2    c
        2    c
        dtype: object
        """
        nv.validate_repeat((), {'axis': axis})
        new_index = self.index.repeat(repeats)
        new_values = self._values.repeat(repeats)
        return self._constructor(new_values, index=new_index, copy=False).__finalize__(self, method='repeat')

    @overload
    def reset_index(
        self,
        level: Optional[Union[int, str, Sequence[Union[int, str]]]] = ...,
        *,
        drop: bool = ...,
        name: Optional[Hashable] = ...,
        inplace: bool = ...,
        allow_duplicates: bool = ...
    ) -> Optional[Union[Series, DataFrame]]:
        ...

    @overload
    def reset_index(
        self,
        level: int,
        *,
        drop: bool = ...,
        name: Optional[Hashable] = ...,
        inplace: bool,
        allow_duplicates: bool = ...
    ) -> Optional[Union[Series, DataFrame]]:
        ...

    @overload
    def reset_index(
        self,
        level: str,
        *,
        drop: bool = ...,
        name: Optional[Hashable] = ...,
        inplace: bool,
        allow_duplicates: bool = ...
    ) -> Optional[Union[Series, DataFrame]]:
        ...

    def reset_index(
        self,
        level: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        *,
        drop: bool = False,
        name: Optional[Hashable] = lib.no_default,
        inplace: bool = False,
        allow_duplicates: bool = False
    ) -> Optional[Union[Series, DataFrame]]:
        """
        Generate a new DataFrame or Series with the index reset.

        This is useful when the index needs to be treated as a column, or
        when the index is meaningless and needs to be reset to the default
        before another operation.

        Parameters
        ----------
        level : int, str, tuple, or list, default optional
            For a Series with a MultiIndex, only remove the specified levels
            from the index. Removes all levels by default.
        drop : bool, default False
            Just reset the index, without inserting it as a column in
            the new DataFrame.
        name : object, optional
            The name to use for the column containing the original Series
            values. Uses ``self.name`` by default. This argument is ignored
            when `drop` is True.
        inplace : bool, default False
            Modify the Series in place (do not create a new object).
        allow_duplicates : bool, default False
            Allow duplicate column labels to be created.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series or DataFrame or None
            When `drop` is False (the default), a DataFrame is returned.
            The newly created columns will come first in the DataFrame,
            followed by the original Series values.
            When `drop` is True, a `Series` is returned.
            In either case, if ``inplace=True``, no value is returned.

        See Also
        --------
        DataFrame.reset_index: Analogous function for DataFrame.

        Examples
        --------
        >>> s = pd.Series(
        ...     [1, 2, 3, 4],
        ...     name="foo",
        ...     index=pd.Index(["a", "b", "c", "d"], name="idx"),
        ... )

        Generate a DataFrame with default index.

        >>> s.reset_index()
          idx  foo
        0   a    1
        1   b    2
        2   c    3
        3   d    4

        To specify the name of the new column use `name`.

        >>> s.reset_index(name="values")
          idx  values
        0   a       1
        1   b       2
        2   c       3
        3   d       4

        To generate a new Series with the default set `drop` to True.

        >>> s.reset_index(drop=True)
        0    1
        1    2
        2    3
        3    4
        Name: foo, dtype: int64

        The `level` parameter is interesting for Series with a multi-level
        index.

        >>> arrays = [
        ...     np.array(["bar", "bar", "baz", "baz"]),
        ...     np.array(["one", "two", "one", "two"]),
        ... ]
        >>> s2 = pd.Series(
        ...     range(4),
        ...     name="foo",
        ...     index=pd.MultiIndex.from_arrays(arrays, names=["a", "b"]),
        ... )

        To remove a specific level from the Index, use `level`.

        >>> s2.reset_index(level="a")
               a  foo
        b
        one  bar    0
        two  bar    1
        one  baz    2
        two  baz    3

        If `level` is not set, all levels are removed from the Index.

        >>> s2.reset_index()
             a    b  foo
        0  bar  one    0
        1  bar  two    1
        2  baz  one    2
        3  baz  two    3
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if drop:
            new_index: Index
            if level is not None:
                if not isinstance(level, (tuple, list)):
                    level_list: TypingSequence[Union[int, str]] = [level]
                else:
                    level_list = level
                level_list = [self.index._get_level_number(lev) for lev in level_list]
                if len(level_list) < self.index.nlevels:
                    new_index = self.index.droplevel(level_list)
                else:
                    new_index = default_index(len(self))
            else:
                new_index = default_index(len(self))
            if inplace:
                self.index = new_index
            else:
                new_ser = self.copy(deep=False)
                new_ser.index = new_index
                return new_ser.__finalize__(self, method='reset_index')
        elif inplace:
            raise TypeError('Cannot reset_index inplace on a Series to create a DataFrame')
        else:
            if name is lib.no_default:
                if self.name is None:
                    name = 0
                else:
                    name = self.name
            df = self.to_frame(name)
            return df.reset_index(level=level, drop=drop, allow_duplicates=allow_duplicates)
        return None

    def __repr__(self) -> str:
        """
        Return a string representation for a particular Series.
        """
        repr_params = fmt.get_series_repr_params()
        return self.to_string(**repr_params)

    @overload
    def to_string(
        self,
        buf: Optional[IO[str]] = ...,
        *,
        na_rep: str = ...,
        float_format: Optional[Callable[[Any], str]] = ...,
        header: bool = ...,
        index: bool = ...,
        length: bool = ...,
        dtype: bool = ...,
        name: bool = ...,
        max_rows: Optional[int] = ...,
        min_rows: Optional[int] = ...
    ) -> Union[str, None]:
        ...

    @overload
    def to_string(
        self,
        buf: IO[str],
        *,
        na_rep: str = ...,
        float_format: Optional[Callable[[Any], str]] = ...,
        header: bool = ...,
        index: bool = ...,
        length: bool = ...,
        dtype: bool = ...,
        name: bool = ...,
        max_rows: Optional[int] = ...,
        min_rows: Optional[int] = ...
    ) -> None:
        ...

    @overload
    def to_string(
        self,
        buf: Optional[IO[str]] = ...,
        *,
        na_rep: str = ...,
        float_format: Optional[Callable[[Any], str]] = ...,
        header: bool = ...,
        index: bool = ...,
        length: bool = ...,
        dtype: bool = ...,
        name: bool = ...,
        max_rows: Optional[int] = ...,
        min_rows: Optional[int] = ...
    ) -> Union[str, None]:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self', 'buf'], name='to_string')
    def to_string(
        self,
        buf: Optional[IO[str]] = None,
        na_rep: str = 'NaN',
        float_format: Optional[Callable[[Any], str]] = None,
        header: bool = True,
        index: bool = True,
        length: bool = False,
        dtype: bool = False,
        name: bool = False,
        max_rows: Optional[int] = None,
        min_rows: Optional[int] = None
    ) -> Optional[str]:
        """
        Render a string representation of the Series.

        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        na_rep : str, optional
            String representation of NaN to use, default 'NaN'.
        float_format : one-parameter function, optional
            Formatter function to apply to columns' elements if they are
            floats, default None.
        header : bool, default True
            Add the Series header (index name).
        index : bool, optional
            Add index (row) labels, default True.
        length : bool, default False
            Add the Series length.
        dtype : bool, default False
            Add the Series dtype.
        name : bool, default False
            Add the Series name if not None.
        max_rows : int, optional
            Maximum number of rows to show before truncating. If None, show
            all.
        min_rows : int, optional
            The number of rows to display in a truncated repr (when number
            of rows is above `max_rows`).

        Returns
        -------
        Optional[str]
            String representation of Series if ``buf=None``, otherwise None.

        See Also
        --------
        Series.to_dict : Convert Series to dict object.
        Series.to_frame : Convert Series to DataFrame object.
        Series.to_markdown : Print Series in Markdown-friendly format.
        Series.to_timestamp : Cast to DatetimeIndex of Timestamps.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3]).to_string()
        >>> ser
        '0    1\\n1    2\\n2    3'
        """
        formatter = fmt.SeriesFormatter(self, name=name, length=length, header=header, index=index, dtype=dtype, na_rep=na_rep, float_format=float_format, min_rows=min_rows, max_rows=max_rows)
        result = formatter.to_string()
        if not isinstance(result, str):
            raise AssertionError(f'result must be of type str, type of result is {type(result).__name__!r}')
        if buf is None:
            return result
        elif hasattr(buf, 'write'):
            buf.write(result)
        else:
            with open(buf, 'w', encoding='utf-8') as f:
                f.write(result)
        return None

    @overload
    def to_markdown(
        self,
        buf: Optional[Union[str, Path, IO[str]]] = ...,
        *,
        mode: str = ...,
        index: bool = ...,
        storage_options: Optional[dict[str, Any]] = ...,
        **kwargs: Any
    ) -> Optional[str]:
        ...

    @overload
    def to_markdown(
        self,
        buf: Union[str, Path, IO[str]],
        *,
        mode: str = ...,
        index: bool = ...,
        storage_options: Optional[dict[str, Any]] = ...,
        **kwargs: Any
    ) -> None:
        ...

    @overload
    def to_markdown(
        self,
        buf: Optional[Union[str, Path, IO[str]]] = ...,
        *,
        mode: str = ...,
        index: bool = ...,
        storage_options: Optional[dict[str, Any]] = ...,
        **kwargs: Any
    ) -> Optional[str]:
        ...

    @doc(klass=_shared_doc_kwargs['klass'], storage_options=_shared_docs['storage_options'], examples=dedent('Examples\n        --------\n        >>> s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")\n        >>> print(s.to_markdown())\n        |    | animal   |\n        |---:|:---------|\n        |  0 | elk      |\n        |  1 | pig      |\n        |  2 | dog      |\n        |  3 | quetzal  |\n\n        Output markdown with a tabulate option.\n\n        >>> print(s.to_markdown(tablefmt="grid"))\n        +----+----------+\n        |    | animal   |\n        +====+==========+\n        |  0 | elk      |\n        +----+----------+\n        |  1 | pig      |\n        +----+----------+\n        |  2 | dog      |\n        +----+----------+\n        |  3 | quetzal  |\n        +----+----------+'))
    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self', 'buf'], name='to_markdown')
    def to_markdown(
        self,
        buf: Optional[Union[str, IO[str], 'Path']] = None,
        mode: str = 'wt',
        index: bool = True,
        storage_options: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> Optional[str]:
        """
        Print {klass} in Markdown-friendly format.

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        mode : str, optional
            Mode in which file is opened, "wt" by default.
        index : bool, optional, default True
            Add index (row) labels.

        {storage_options}

        **kwargs
            These parameters will be passed to `tabulate                 <https://pypi.org/project/tabulate>`_.

        Returns
        -------
        Optional[str]
            {klass} in Markdown-friendly format.

        See Also
        --------
        Series.to_frame : Rrite a text representation of object to the system clipboard.
        Series.to_latex : Render Series to LaTeX-formatted table.

        Notes
        -----
        Requires the `tabulate <https://pypi.org/project/tabulate>`_ package.

        {examples}
        """
        return self.to_frame().to_markdown(buf, mode=mode, index=index, storage_options=storage_options, **kwargs)

    def items(self) -> Iterable[tuple[Any, Any]]:
        """
        Lazily iterate over (index, value) tuples.

        This method returns an iterable tuple (index, value). This is
        convenient if you want to create a lazy iterator.

        Returns
        -------
        Iterable[tuple[Any, Any]]
            Iterable of tuples containing the (index, value) pairs from a
            Series.

        See Also
        --------
        DataFrame.items : Iterate over (column name, Series) pairs.
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series) pairs.

        Examples
        --------
        >>> s = pd.Series(["A", "B", "C"])
        >>> for index, value in s.items():
        ...     print(f"Index : {index}, Value : {value}")
        Index : 0, Value : A
        Index : 1, Value : B
        Index : 2, Value : C
        """
        return zip(iter(self.index), iter(self))

    def keys(self) -> Index:
        """
        Return alias for index.

        Returns
        -------
        Index
            Index of the Series.

        See Also
        --------
        Series.index : The index (axis labels) of the Series.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3], index=[0, 1, 2])
        >>> s.keys()
        Index([0, 1, 2], dtype='int64')
        """
        return self.index

    @overload
    def to_dict(
        self,
        *,
        into: type[MutableMappingT] = ...,
    ) -> MutableMappingT:
        ...

    @overload
    def to_dict(
        self,
        *,
        into: type[MutableMappingT] = dict
    ) -> MutableMappingT:
        ...

    def to_dict(
        self,
        *,
        into: type[MutableMappingT] = dict
    ) -> MutableMappingT:
        """
        Convert Series to {label -> value} dict or dict-like object.

        Parameters
        ----------
        into : class, default dict
            The collections.abc.MutableMapping subclass to use as the return
            object. Can be the actual class or an empty instance of the mapping
            type you want.  If you want a collections.defaultdict, you must
            pass it initialized.

        Returns
        -------
        MutableMappingT
            Key-value representation of Series.

        See Also
        --------
        Series.to_list: Converts Series to a list of the values.
        Series.to_numpy: Converts Series to NumPy ndarray.
        Series.array: ExtensionArray of the data backing this Series.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.to_dict()
        {0: 1, 1: 2, 2: 3, 3: 4}
        >>> from collections import OrderedDict, defaultdict
        >>> s.to_dict(into=OrderedDict)
        OrderedDict([(0, 1), (1, 2), (2, 3), (3, 4)])
        >>> dd = defaultdict(list)
        >>> s.to_dict(into=dd)
        defaultdict(<class 'list'>, {0: 1, 1: 2, 2: 3, 3: 4})
        """
        into_c = com.standardize_mapping(into)
        if is_object_dtype(self.dtype) or isinstance(self.dtype, ExtensionDtype):
            return cast(
                MutableMappingT,
                into_c(((k, maybe_box_native(v)) for k, v in self.items()))
            )
        else:
            return cast(MutableMappingT, into_c(self.items()))

    def to_frame(self, name: Optional[Hashable] = lib.no_default) -> DataFrame:
        """
        Convert Series to DataFrame.

        Parameters
        ----------
        name : object, optional
            The passed name should substitute for the series name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame representation of Series.

        See Also
        --------
        Series.to_dict : Convert Series to dict object.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c"], name="vals")
        >>> s.to_frame()
          vals
        0    a
        1    b
        2    c
        """
        if name is lib.no_default:
            name = self.name
            if name is None:
                columns = default_index(1)
            else:
                columns = Index([name])
        else:
            columns = Index([name])
        mgr = self._mgr.to_2d_mgr(columns)
        df = self._constructor_expanddim_from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(self, method='to_frame')

    def _set_name(
        self,
        name: Optional[Hashable],
        inplace: bool = False,
        deep: Optional[Any] = None
    ) -> Optional[Series]:
        """
        Set the Series name.

        Parameters
        ----------
        name : Optional[Hashable]
            The new name of the Series.
        inplace : bool
            Whether to modify `self` directly or return a copy.
        deep : Optional[Any]
            Unused parameter, included for compatibility.

        Returns
        -------
        Optional[Series]
            The modified Series or None if `inplace=True`.
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        ser = self if inplace else self.copy(deep=False)
        ser.name = name
        return ser

    @Appender(dedent('\n        Examples\n        --------\n        >>> ser = pd.Series([390., 350., 30., 20.],\n        ...                 index=[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'],\n        ...                 name="Max Speed")\n        >>> ser\n        Falcon    390.0\n        Falcon    350.0\n        Parrot     30.0\n        Parrot     20.0\n        Name: Max Speed, dtype: float64\n\n        We can pass a list of values to group the Series data by custom labels:\n\n        >>> ser.groupby(["a", "b", "a", "b"]).mean()\n        a    210.0\n        b    185.0\n        Name: Max Speed, dtype: float64\n\n        Grouping by numeric labels yields similar results:\n\n        >>> ser.groupby([0, 1, 0, 1]).mean()\n        0    210.0\n        1    185.0\n        Name: Max Speed, dtype: float64\n\n        We can group by a level of the index:\n\n        >>> ser.groupby(level=0).mean()\n        Falcon    370.0\n        Parrot     25.0\n        Name: Max Speed, dtype: float64\n\n        We can group by a condition applied to the Series values:\n\n        >>> ser.groupby(ser > 100).mean()\n        Max Speed\n        False     25.0\n        True     370.0\n        Name: Max Speed, dtype: float64\n\n        **Grouping by Indexes**\n\n        We can groupby different levels of a hierarchical index\n        using the `level` parameter:\n\n        >>> arrays = [[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'],\n        ...           [\'Captive\', \'Wild\', \'Captive\', \'Wild\']]\n        >>> index = pd.MultiIndex.from_arrays(arrays, names=(\'Animal\', \'Type\'))\n        >>> ser = pd.Series([390., 350., 30., 20.], index=index, name="Max Speed")\n        >>> ser\n        Animal  Type\n        Falcon  Captive    390.0\n                Wild       350.0\n        Parrot  Captive     30.0\n                Wild        20.0\n        Name: Max Speed, dtype: float64\n\n        >>> ser.groupby(level=0).mean()\n        Animal\n        Falcon    370.0\n        Parrot     25.0\n        Name: Max Speed, dtype: float64\n\n        We can also group by the \'Type\' level of the hierarchical index\n        to get the mean speed for each type:\n\n        >>> ser.groupby(level="Type").mean()\n        Type\n        Captive    210.0\n        Wild       185.0\n        Name: Max Speed, dtype: float64\n\n        We can also choose to include `NA` in group keys or not by defining\n        `dropna` parameter, the default setting is `True`.\n\n        >>> ser = pd.Series([1, 2, 3, 3], index=["a", \'a\', \'b\', np.nan])\n        >>> ser.groupby(level=0).sum()\n        a    3\n        b    3\n        dtype: int64\n\n        To include `NA` values in the group keys, set `dropna=False`:\n\n        >>> ser.groupby(level=0, dropna=False).sum()\n        a    3\n        b    3\n        NaN  3\n        dtype: int64\n\n        We can also group by a custom list with NaN values to handle\n        missing group labels:\n\n        >>> arrays = [\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\']\n        >>> ser = pd.Series([390., 350., 30., 20.], index=arrays, name="Max Speed")\n        >>> ser.groupby(["a", "b", "a", np.nan]).mean()\n        a    210.0\n        b    350.0\n        Name: Max Speed, dtype: float64\n\n        >>> ser.groupby(["a", "b", "a", np.nan], dropna=False).mean()\n        a    210.0\n        b    350.0\n        NaN   20.0\n        Name: Max Speed, dtype: float64\n        '))
    @Appender(_shared_docs['groupby'] % _shared_doc_kwargs)
    def groupby(
        self,
        by: Optional[Union[
            IndexLabel,
            Callable[[Any], Any],
            Sequence[Any],
            Mapping[Any, Any],
            ABCDataFrame
        ]] = None,
        level: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True
    ) -> SeriesGroupBy:
        from pandas.core.groupby.generic import SeriesGroupBy
        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        if not as_index:
            raise TypeError('as_index=False only valid with DataFrame')
        return SeriesGroupBy(
            obj=self,
            keys=by,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna
        )

    def count(self) -> int:
        """
        Return number of non-NA/null observations in the Series.

        Returns
        -------
        int
            Number of non-null values in the Series.

        See Also
        --------
        DataFrame.count : Count non-NA cells for each column or row.

        Examples
        --------
        >>> s = pd.Series([0.0, 1.0, np.nan])
        >>> s.count()
        2
        """
        return notna(self._values).sum().astype('int64')

    def mode(self, dropna: bool = True) -> Series:
        """
        Return the mode(s) of the Series.

        The mode is the value that appears most often. There can be multiple modes.

        Always returns Series even if only one value is returned.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NaN/NaT.

        Returns
        -------
        Series
            Modes of the Series in sorted order.

        See Also
        --------
        numpy.mode : Equivalent numpy function for computing median.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.

        Examples
        --------
        >>> s = pd.Series([2, 4, 2, 2, 4, None])
        >>> s.mode()
        0    2.0
        dtype: float64

        More than one mode:

        >>> s = pd.Series([2, 4, 8, 2, 4, None])
        >>> s.mode()
        0    2.0
        1    4.0
        dtype: float64

        With and without considering null value:

        >>> s = pd.Series([2, 4, None, None, 4, None])
        >>> s.mode(dropna=False)
        0   NaN
        dtype: float64
        >>> s = pd.Series([2, 4, None, None, 4, None])
        >>> s.mode()
        0    4.0
        dtype: float64
        """
        values = self._values
        if isinstance(values, np.ndarray):
            res_values = algorithms.mode(values, dropna=dropna)
        else:
            res_values = values._mode(dropna=dropna)
        return self._constructor(
            res_values,
            index=range(len(res_values)),
            name=self.name,
            copy=False,
            dtype=self.dtype
        ).__finalize__(self, method='mode')

    def unique(self) -> Union[np.ndarray, ExtensionArray]:
        """
        Return unique values of Series object.

        Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.

        Returns
        -------
        Union[np.ndarray, ExtensionArray]
            The unique values returned as a NumPy array. See Notes.

        See Also
        --------
        Series.drop_duplicates : Return Series with duplicate values removed.
        unique : Top-level unique method for any 1-d array-like object.
        Index.unique : Return Index with unique values from an Index object.

        Notes
        -----
        Returns the unique values as a NumPy array. In case of an
        extension-array backed Series, a new
        :class:`~api.extensions.ExtensionArray` of that type with just
        the unique values is returned. This includes

            * Categorical
            * Period
            * Datetime with Timezone
            * Datetime without Timezone
            * Timedelta
            * Interval
            * Sparse
            * IntegerNA

        See Examples section.

        Examples
        --------
        >>> pd.Series([2, 1, 3, 3], name="A").unique()
        array([2, 1, 3])

        >>> pd.Series([pd.Timestamp("2016-01-01") for _ in range(3)]).unique()
        <DatetimeArray>
        ['2016-01-01 00:00:00']
        Length: 1, dtype: datetime64[s]

        >>> pd.Series(
        ...     [pd.Timestamp("2016-01-01", tz="US/Eastern") for _ in range(3)]
        ... ).unique()
        <DatetimeArray>
        ['2016-01-01 00:00:00-05:00']
        Length: 1, dtype: datetime64[s, US/Eastern]

        An Categorical will return categories in the order of
        appearance and with the same dtype.

        >>> pd.Series(pd.Categorical(list("baabc"))).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> pd.Series(
        ...     pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
        ... ).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a' < 'b' < 'c']
        """
        return super().unique()

    @overload
    def drop_duplicates(
        self,
        *,
        keep: Literal['first', 'last', False] = ...,
        inplace: bool = ...,
        ignore_index: bool = ...
    ) -> Optional[Series]:
        ...

    @overload
    def drop_duplicates(
        self,
        *,
        keep: Literal['first', 'last', False],
        inplace: bool,
        ignore_index: bool = ...
    ) -> Optional[Series]:
        ...

    @overload
    def drop_duplicates(
        self,
        *,
        keep: Literal['first', 'last', False] = ...,
        inplace: bool,
        ignore_index: bool = ...
    ) -> Optional[Series]:
        ...

    def drop_duplicates(
        self,
        *,
        keep: Union[Literal['first', 'last'], bool] = 'first',
        inplace: bool = False,
        ignore_index: bool = False
    ) -> Optional[Series]:
        """
        Return Series with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.
        inplace : bool, default ``False``
            If ``True``, performs operation inplace and returns None.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        Optional[Series]
            Series with duplicates dropped or None if ``inplace=True``.

        See Also
        --------
        Index.drop_duplicates : Equivalent method on Index.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.
        Series.duplicated : Related method on Series, indicating duplicate
            Series values.
        Series.unique : Return unique values as an array.

        Examples
        --------
        Generate a Series with duplicated entries.

        >>> s = pd.Series(
        ...     ["llama", "cow", "llama", "beetle", "llama", "hippo"], name="animal"
        ... )
        >>> s
        0     llama
        1       cow
        2     llama
        3    beetle
        4     llama
        5     hippo
        Name: animal, dtype: object

        With the 'keep' parameter, the selection behavior of duplicated values
        can be changed. The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> s.drop_duplicates()
        0     llama
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object

        The value 'last' for parameter 'keep' keeps the last occurrence for
        each set of duplicated entries.

        >>> s.drop_duplicates(keep="last")
        1       cow
        3    beetle
        4     llama
        5     hippo
        Name: animal, dtype: object

        The value ``False`` for parameter 'keep' discards all sets of
        duplicated entries.

        >>> s.drop_duplicates(keep=False)
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        result = super().drop_duplicates(keep=keep)
        if ignore_index:
            result.index = default_index(len(result))
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def duplicated(self, keep: Union[Literal['first', 'last'], bool] = 'first') -> Series:
        """
        Indicate duplicate Series values.

        Duplicated values are indicated as ``True`` values in the resulting
        Series. Either all duplicates, all except the first or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Mark duplicates as ``True`` except for the first occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        Series[bool]
            Series indicating whether each value has occurred in the
            preceding values.

        See Also
        --------
        Index.duplicated : Equivalent method on pandas.Index.
        DataFrame.duplicated : Equivalent method on pandas.DataFrame.
        Series.drop_duplicates : Remove duplicate values from Series.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set on False and all others on True:

        >>> animals = pd.Series(["llama", "cow", "llama", "beetle", "llama"])
        >>> animals.duplicated()
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        which is equivalent to

        >>> animals.duplicated(keep="first")
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> animals.duplicated(keep="last")
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        By setting keep on ``False``, all duplicates are True:

        >>> animals.duplicated(keep=False)
        0     True
        1    False
        2     True
        3    False
        4     True
        dtype: bool
        """
        res = self._duplicated(keep=keep)
        result = self._constructor(res, index=self.index, copy=False)
        return result.__finalize__(self, method='duplicated')

    def idxmin(
        self,
        axis: Union[int, Literal[0]] = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Return the row label of the minimum value.

        If multiple values equal the minimum, the first row label with that
        value is returned.

        Parameters
        ----------
        axis : Union[int, Literal[0]]
            {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, or if ``skipna=False``
            and there is an NA value, this method will raise a ``ValueError``.
        *args : Any
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.
        **kwargs : Any
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Any
            Label of the minimum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        numpy.argmin : Return indices of the minimum values
            along the given axis.
        DataFrame.idxmin : Return index of first occurrence of minimum
            over requested axis.
        Series.idxmax : Return index *label* of the first occurrence
            of maximum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmin``. This method
        returns the label of the minimum, while ``ndarray.argmin`` returns
        the position. To get the position, use ``series.values.argmin()``.

        Examples
        --------
        >>> s = pd.Series(data=[1, None, 4, 1], index=["A", "B", "C", "D"])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    1.0
        dtype: float64

        >>> s.idxmin()
        'A'
        """
        axis = self._get_axis_number(axis)
        iloc = self.argmin(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def idxmax(
        self,
        axis: Union[int, Literal[0]] = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Return the row label of the maximum value.

        If multiple values equal the maximum, the first row label with that
        value is returned.

        Parameters
        ----------
        axis : Union[int, Literal[0]]
            {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, or if ``skipna=False``
            and there is an NA value, this method will raise a ``ValueError``.
        *args : Any
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.
        **kwargs : Any
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Any
            Label of the maximum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        numpy.argmax : Return indices of the maximum values
            along the given axis.
        DataFrame.idxmax : Return index of first occurrence of maximum
            over requested axis.
        Series.idxmin : Return index *label* of the first occurrence
            of minimum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmax``. This method
        returns the label of the maximum, while ``ndarray.argmax`` returns
        the position. To get the position, use ``series.values.argmax()``.

        Examples
        --------
        >>> s = pd.Series(data=[1, None, 4, 3, 4], index=["A", "B", "C", "D", "E"])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    3.0
        E    4.0
        dtype: float64

        >>> s.idxmax()
        'C'
        """
        axis = self._get_axis_number(axis)
        iloc = self.argmax(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def round(
        self,
        decimals: Union[int, Tuple[int, ...], list[int]] = 0,
        *args: Any,
        **kwargs: Any
    ) -> Series:
        """
        Round each value in a Series to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args : Any
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.
        **kwargs : Any
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        Series
            Rounded values of the Series.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.
        Series.dt.round : Round values of data to the specified freq.

        Notes
        -----
        For values exactly halfway between rounded decimal values, pandas rounds
        to the nearest even value (e.g. -0.5 and 0.5 round to 0.0, 1.5 and 2.5
        round to 2.0, etc.).

        Examples
        --------
        >>> s = pd.Series([-0.5, 0.1, 2.5, 1.3, 2.7])
        >>> s.round()
        0   -0.0
        1    0.0
        2    2.0
        3    1.0
        4    3.0
        dtype: float64
        """
        nv.validate_round(args, kwargs)
        new_mgr = self._mgr.round(decimals=decimals)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self, method='round')

    @overload
    def quantile(
        self,
        q: Union[float, TypingSequence[float]] = ...,
        interpolation: Literal['linear', 'lower', 'higher', 'midpoint', 'nearest'] = ...
    ) -> Union[float, Series]:
        ...

    @overload
    def quantile(
        self,
        q: float,
        interpolation: Literal['linear', 'lower', 'higher', 'midpoint', 'nearest']
    ) -> float:
        ...

    @overload
    def quantile(
        self,
        q: TypingSequence[float],
        interpolation: Literal['linear', 'lower', 'higher', 'midpoint', 'nearest']
    ) -> Series:
        ...

    def quantile(
        self,
        q: Union[float, TypingSequence[float]] = 0.5,
        interpolation: Literal['linear', 'lower', 'higher', 'midpoint', 'nearest'] = 'linear'
    ) -> Union[float, Series]:
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : Union[float, Sequence[float]], default 0.5 (50% quantile)
            The quantile(s) to compute, which can lie in range: 0 <= q <= 1.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

                * linear: `i + (j - i) * (x-i)/(j-i)`, where `(x-i)/(j-i)` is
                  the fractional part of the index surrounded by `i > j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.

        Returns
        -------
        Union[float, Series]
            If ``q`` is an array, a Series will be returned where the
            index is ``q`` and the values are the quantiles, otherwise
            a float will be returned.

        See Also
        --------
        core.window.Rolling.quantile : Calculate the rolling quantile.
        numpy.percentile : Returns the q-th percentile(s) of the array elements.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.quantile(0.5)
        2.5
        >>> s.quantile([0.25, 0.5, 0.75])
        0.25    1.75
        0.50    2.50
        0.75    3.25
        dtype: float64
        """
        validate_percentile(q)
        df = self.to_frame()
        result = df.quantile(q=q, interpolation=interpolation, numeric_only=False)
        if result.ndim == 2:
            result = result.iloc[:, 0]
        if is_list_like(q):
            result.name = self.name
            idx = Index(q, dtype=np.float64)
            return self._constructor(result, index=idx, name=self.name)
        else:
            return result.iloc[0]

    def corr(
        self,
        other: Series,
        method: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'pearson',
        min_periods: Optional[int] = None
    ) -> float:
        """
        Compute correlation with `other` Series, excluding missing values.

        The two `Series` objects are not required to be the same length and will be
        aligned internally before the correlation function is applied.

        Parameters
        ----------
        other : Series
            Series with which to compute the correlation.
        method : Union[str, Callable[[np.ndarray, np.ndarray], float]]
            Method used to compute correlation:

            - pearson : Standard correlation coefficient
            - kendall : Kendall Tau correlation coefficient
            - spearman : Spearman rank correlation
            - callable: Callable with input two 1d ndarrays and returning a float.

            .. warning::
                Note that the returned matrix from corr will have 1 along the
                diagonals and will be symmetric regardless of the callable's
                behavior.
        min_periods : Optional[int], default None
            Minimum number of observations needed to have a valid result.

        Returns
        -------
        float
            Correlation with other.

        See Also
        --------
        DataFrame.corr : Compute pairwise correlation between columns.
        DataFrame.corrwith : Compute pairwise correlation with another
            DataFrame or Series.

        Notes
        -----
        Pearson, Kendall and Spearman correlation are currently computed using pairwise complete observations.

        * `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
        * `Kendall rank correlation coefficient <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_
        * `Spearman's rank correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_

        Automatic data alignment: as with all pandas operations, automatic data alignment is performed for this method.
        ``corr()`` automatically considers values with matching indices.

        Examples
        --------
        >>> def histogram_intersection(a, b):
        ...     v = np.minimum(a, b).sum().round(decimals=1)
        ...     return v
        >>> s1 = pd.Series([0.2, 0.0, 0.6, 0.2])
        >>> s2 = pd.Series([0.3, 0.6, 0.0, 0.1])
        >>> s1.corr(s2, method=histogram_intersection)
        0.3

        Pandas auto-aligns the values with matching indices

        >>> s1 = pd.Series([1, 2, 3], index=[0, 1, 2])
        >>> s2 = pd.Series([1, 2, 3], index=[2, 1, 0])
        >>> s1.corr(s2)
        -1.0

        If the input is a constant array, the correlation is not defined in this case,
        and ``np.nan`` is returned.

        >>> s1 = pd.Series([0.45, 0.45])
        >>> s1.corr(s1)
        nan
        """
        this, other_aligned = self.align(other, join='inner')
        if len(this) == 0:
            return np.nan
        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other_aligned.to_numpy(dtype=float, na_value=np.nan, copy=False)
        if method in ['pearson', 'spearman', 'kendall'] or callable(method):
            return nanops.nancorr(this_values, other_values, method=method, min_periods=min_periods)
        raise ValueError(f"method must be either 'pearson', 'spearman', 'kendall', or a callable, '{method}' was supplied")

    def cov(
        self,
        other: Series,
        min_periods: Optional[int] = None,
        ddof: int = 1
    ) -> float:
        """
        Compute covariance with Series, excluding missing values.

        The two `Series` objects are not required to be the same length and
        will be aligned internally before the covariance is calculated.

        Parameters
        ----------
        other : Series
            Series with which to compute the covariance.
        min_periods : Optional[int], default None
            Minimum number of observations needed to have a valid result.
        ddof : int, default 1
            Delta degrees of freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.

        Returns
        -------
        float
            Covariance between Series and other normalized by N-1
            (unbiased estimator).

        See Also
        --------
        DataFrame.cov : Compute pairwise covariance of columns.

        Examples
        --------
        >>> s1 = pd.Series([0.90010907, 0.13484424, 0.62036035])
        >>> s2 = pd.Series([0.12528585, 0.26962463, 0.51111198])
        >>> s1.cov(s2)
        -0.01685762652715874
        """
        this, other_aligned = self.align(other, join='inner')
        if len(this) == 0:
            return np.nan
        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other_aligned.to_numpy(dtype=float, na_value=np.nan, copy=False)
        return nanops.nancov(this_values, other_values, min_periods=min_periods, ddof=ddof)

    @doc(_shared_docs['compare'], dedent('\n        Returns\n        -------\n        Series or DataFrame\n            If axis is 0 or \'index\' the result will be a Series.\n            The resulting index will be a MultiIndex with \'self\' and \'other\'\n            stacked alternately at the inner level.\n\n            If axis is 1 or \'columns\' the result will be a DataFrame.\n            It will have two columns namely \'self\' and \'other\'.\n\n        See Also\n        --------\n        DataFrame.compare : Compare with another DataFrame and show differences.\n\n        Notes\n        -----\n        Matching NaNs will not appear as a difference.\n\n        Examples\n        --------\n        >>> s1 = pd.Series(["a", "b", "c", "d", "e"])\n        >>> s2 = pd.Series(["a", "a", "c", "b", "e"])\n\n        Align the differences on columns\n\n        >>> s1.compare(s2)\n          self other\n        1    b     a\n        3    d     b\n\n        Stack the differences on indices\n\n        >>> s1.compare(s2, align_axis=0)\n        1  self     b\n           other    a\n        3  self     d\n           other    b\n        dtype: object\n\n        Keep all original rows\n\n        >>> s1.compare(s2, keep_shape=True)\n          self other\n        0  NaN   NaN\n        1    b     a\n        2  NaN   NaN\n        3    d     b\n        4  NaN   NaN\n\n        Keep all original rows and also all original values\n\n        >>> s1.compare(s2, keep_shape=True, keep_equal=True)\n          self other\n        0    a     a\n        1    b     a\n        2    c     c\n        3    d     b\n        4    e     e\n        """
    def compare(
        self,
        other: Series,
        align_axis: int = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: tuple[str, str] = ('self', 'other')
    ) -> Union[Series, DataFrame]:
        return super().compare(other=other, align_axis=align_axis, keep_shape=keep_shape, keep_equal=keep_equal, result_names=result_names)

    def combine(
        self,
        other: Union[Series, Scalar],
        func: Callable[[Any, Any], Any],
        fill_value: Optional[Union[float, Any]] = None
    ) -> Series:
        """
        Combine the Series with a Series or scalar according to `func`.

        Combine the Series and `other` using `func` to perform elementwise
        selection for combined Series.
        `fill_value` is assumed when value is missing at some index
        from one of the two objects being combined.

        Parameters
        ----------
        other : Union[Series, Scalar]
            The value(s) to be combined with the `Series`.
        func : Callable[[Any, Any], Any]
            Function that takes two scalars as inputs and returns an element.
        fill_value : Optional[Union[float, Any]], optional
            The value to assume when an index is missing from
            one Series or the other. If ``other`` is a scalar or not a
            Series, this is ignored.

        Returns
        -------
        Series
            The result of combining the Series with the other object.

        See Also
        --------
        Series.combine_first : Combine Series values, choosing the calling
            Series' values first.

        Examples
        --------
        Consider 2 Datasets ``s1`` and ``s2`` containing
        highest clocked speeds of different birds.

        >>> s1 = pd.Series([390., 350., 30., 20.],\n        ...                 index=[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'],\n        ...                 name="Max Speed")
        >>> s1
        Falcon    390.0
        Falcon    350.0
        Parrot     30.0
        Parrot     20.0
        Name: Max Speed, dtype: float64

        Now, to combine the two datasets and view the highest speeds
        of the birds across the two datasets

        >>> s1.combine(s2, max)
        duck        NaN
        eagle     200.0
        falcon    345.0
        dtype: float64

        In the previous example, the resulting value for duck is missing,
        because the maximum of a NaN and a float is a NaN.
        So, in the example, we set ``fill_value=0``,
        so the maximum value returned will be the value from some dataset.

        >>> s1.combine(s2, max, fill_value=0)
        duck       30.0
        eagle     200.0
        falcon    345.0
        dtype: float64
        """
        if fill_value is None:
            fill_value = na_value_for_dtype(self.dtype, compat=False)
        if isinstance(other, Series):
            new_index = self.index.union(other.index)
            new_name = ops.get_op_result_name(self, other)
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all='ignore'):
                for i, idx in enumerate(new_index):
                    lv = self.get(idx, fill_value)
                    rv = other.get(idx, fill_value)
                    new_values[i] = func(lv, rv)
        else:
            new_index = self.index
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all='ignore'):
                new_values[:] = [func(lv, other) for lv in self._values]
            new_name = self.name
        npvalues = lib.maybe_convert_objects(new_values, try_float=False)
        same_dtype = isinstance(self.dtype, (StringDtype, CategoricalDtype))
        res_values = maybe_cast_pointwise_result(npvalues, self.dtype, same_dtype=same_dtype)
        return self._constructor(res_values, index=new_index, name=new_name, copy=False)

    def combine_first(self, other: Series) -> Series:
        """
        Update null elements with value in the same location in 'other'.

        Combine two Series objects by filling null values in one Series with
        non-null values from the other Series. Result index will be the union
        of the two indexes.

        Parameters
        ----------
        other : Series
            The value(s) to be used for filling null values.

        Returns
        -------
        Series
            The result of combining the provided Series with the other object.

        See Also
        --------
        Series.combine : Perform element-wise operation on two Series
            using a given function.

        Examples
        --------
        >>> s1 = pd.Series([1.0, np.nan])
        >>> s2 = pd.Series([3.0, 4.0, 5.0])
        >>> s1.combine_first(s2)
        0    1.0
        1    4.0
        2    5.0
        dtype: float64

        Null values still persist if the location of that null value
        does not exist in `other`

        >>> s1 = pd.Series({"falcon": np.nan, "eagle": 160.0})
        >>> s2 = pd.Series({"eagle": 200.0, "duck": 30.0})
        >>> s1.combine_first(s2)
        duck       30.0
        eagle     160.0
        falcon      NaN
        dtype: float64
        """
        from pandas.core.reshape.concat import concat
        if self.dtype == other.dtype:
            if self.index.equals(other.index):
                return self.mask(self.isna(), other)
            elif self._can_hold_na and (not isinstance(self.dtype, SparseDtype)):
                this, other_aligned = self.align(other, join='outer')
                return this.mask(this.isna(), other_aligned)
        new_index = self.index.union(other.index)
        this = self
        keep_other = other.index.difference(this.index[notna(this)])
        keep_this = this.index.difference(keep_other)
        this = this.reindex(keep_this)
        other = other.reindex(keep_other)
        if this.dtype.kind == 'M' and other.dtype.kind != 'M':
            other = to_datetime(other)
        combined = concat([this, other])
        combined = combined.reindex(new_index)
        return combined.__finalize__(self, method='combine_first')

    def update(self, other: Union[Series, array_like, Mapping[Any, Any], list, tuple, dict], /) -> None:
        """
        Modify Series in place using values from passed Series.

        Uses non-NA values from passed Series to make updates. Aligns
        on index.

        Parameters
        ----------
        other : Union[Series, array_like, Mapping[Any, Any], list, tuple, dict]
            Other Series that provides values to update the current Series.

        See Also
        --------
        Series.combine : Perform element-wise operation on two Series
            using a given function.
        Series.transform: Modify a Series using a function.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])

        >>> ser.update(pd.Series([4, 5, 6]))
        >>> ser
        0    4
        1    5
        2    6
        dtype: int64

        >>> ser = pd.Series(["a", "b", "c"])
        >>> ser.update(pd.Series(["d", "e"], index=[0, 2]))
        >>> ser
        0    d
        1    b
        2    e
        dtype: object

        >>> ser = pd.Series([1, 2, 3])
        >>> ser.update(pd.Series([4, 5, 6, 7, 8]))
        >>> ser
        0    4
        1    5
        2    6
        dtype: int64

        If ``other`` contains NaNs the corresponding values are not updated
        in the original Series.

        >>> ser = pd.Series([1, 2, 3])
        >>> ser.update(pd.Series([4, np.nan, 6]))
        >>> ser
        0    4
        1    2
        2    6
        dtype: int64

        ``other`` can also be a non-Series object type
        that is coercible into a Series

        >>> ser = pd.Series([1, 2, 3])
        >>> ser.update([4, np.nan, 6])
        >>> ser
        0    4
        1    2
        2    6
        dtype: int64

        >>> ser = pd.Series([1, 2, 3])
        >>> ser.update({1: 9})
        >>> ser
        0    1
        1    9
        2    3
        dtype: int64
        """
        if not PYPY:
            if sys.getrefcount(self) <= REF_COUNT:
                warnings.warn(_chained_assignment_method_msg, ChainedAssignmentError, stacklevel=2)
        if not isinstance(other, Series):
            other = Series(other)
        other = other.reindex_like(self)
        mask = notna(other)
        self._mgr = self._mgr.putmask(mask=mask, new=other)

    @overload
    def sort_values(
        self,
        *,
        axis: Union[int, Literal[0]] = ...,
        ascending: Union[bool, TypingSequence[bool]] = ...,
        inplace: bool = ...,
        kind: Literal['quicksort', 'mergesort', 'heapsort', 'stable'] = ...,
        na_position: Literal['first', 'last'] = ...,
        ignore_index: bool = ...,
        key: Optional[Callable[[Any], Any]] = ...
    ) -> Optional[Series]:
        ...

    @overload
    def sort_values(
        self,
        *,
        axis: Union[int, Literal[0]],
        ascending: Union[bool, TypingSequence[bool]],
        inplace: bool,
        kind: Literal['quicksort', 'mergesort', 'heapsort', 'stable'],
        na_position: Literal['first', 'last'],
        ignore_index: bool,
        key: Optional[Callable[[Any], Any]] = ...
    ) -> Optional[Series]:
        ...

    @overload
    def sort_values(
        self,
        *,
        axis: Union[int, Literal[0]] = ...,
        ascending: Union[bool, TypingSequence[bool]] = ...,
        inplace: bool = ...,
        kind: Literal['quicksort', 'mergesort', 'heapsort', 'stable'] = ...,
        na_position: Literal['first', 'last'] = ...,
        ignore_index: bool = ...,
        key: Optional[Callable[[Any], Any]] = ...
    ) -> Optional[Series]:
        ...

    def sort_values(
        self,
        *,
        axis: Union[int, Literal[0]] = 0,
        ascending: Union[bool, TypingSequence[bool]] = True,
        inplace: bool = False,
        kind: Literal['quicksort', 'mergesort', 'heapsort', 'stable'] = 'quicksort',
        na_position: Literal['first', 'last'] = 'last',
        ignore_index: bool = False,
        key: Optional[Callable[[Any], Any]] = None
    ) -> Optional[Series]:
        """
        Sort by the values.

        Sort a Series in ascending or descending order by some
        criterion.

        Parameters
        ----------
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        ascending : Union[bool, Sequence[bool]], default True
            If True, sort values in ascending order, otherwise descending.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : Literal['quicksort', 'mergesort', 'heapsort', 'stable'], default 'quicksort'
            Choice of sorting algorithm. See also :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable algorithms.
        na_position : Literal['first', 'last'], default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
        key : Optional[Callable[[Any], Any]], optional
            If not None, apply the key function to the series values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect a
            ``Series`` and return an array-like.

        Returns
        -------
        Optional[Series]
            Series ordered by values or None if ``inplace=True``.

        See Also
        --------
        Series.sort_index : Sort by the Series indices.
        DataFrame.sort_values : Sort DataFrame by the values along either axis.
        DataFrame.sort_index : Sort DataFrame by indices.
        Series.sort_values : Sort Series by the value.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c", "d"], index=[3, 2, 1, 4])
        >>> s.sort_values()
        1    c
        2    b
        3    a
        4    d
        dtype: object

        Sort Descending

        >>> s.sort_values(ascending=False)
        4    d
        3    a
        2    b
        1    c
        dtype: object

        By default NaNs are put at the end, but use `na_position` to place
        them at the beginning

        >>> s = pd.Series(["a", "b", "c", "d"], index=[3, 2, 1, np.nan])
        >>> s.sort_values(na_position="first")
        NaN     d
         1.0    c
         2.0    b
         3.0    a
        dtype: object

        Specify index level to sort

        >>> arrays = [
        ...     np.array(["qux", "qux", "foo", "foo", "baz", "baz", "bar", "bar"]),
        ...     np.array(["two", "one", "two", "one", "two", "one", "two", "one"]),
        ... ]
        >>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=arrays)
        >>> s.sort_values(level=1)
        bar  one    8
        baz  one    6
        foo  one    4
        qux  one    2
        bar  two    7
        baz  two    5
        foo  two    3
        qux  two    1
        dtype: int64

        Does not sort by remaining levels when sorting by levels

        >>> s.sort_values(level=1, sort_remaining=False)
        qux  one    2
        foo  one    4
        baz  one    6
        bar  one    8
        qux  two    1
        foo  two    3
        baz  two    5
        bar  two    7
        dtype: int64

        Apply a key function before sorting

        >>> s = pd.Series([1, 2, 3, 4], index=["A", "b", "C", "d"])
        >>> s.sort_values(key=lambda x: x.str.lower())
        A    1
        b    2
        C    3
        d    4
        dtype: int64

        NumPy ufuncs work well here. For example, we can
        sort by the ``sin`` of the value

        >>> s = pd.Series([-4, -2, 0, 2, 4])
        >>> s.sort_values(key=np.sin)
        1   -2
        4    4
        2    0
        0   -4
        3    2
        dtype: int64

        More complicated user-defined functions can be used,
        as long as they expect a Series and return an array-like

        >>> s.sort_values(key=lambda x: (np.tan(x.cumsum())))
        0   -4
        3    2
        4    4
        1   -2
        2    0
        dtype: int64
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        self._get_axis_number(axis)
        if is_list_like(ascending):
            ascending = cast(TypingSequence[bool], ascending)
            if len(ascending) != 1:
                raise ValueError(f'Length of ascending ({len(ascending)}) must be 1 for Series')
            ascending = ascending[0]
        ascending = validate_ascending(ascending)
        if na_position not in ['first', 'last']:
            raise ValueError(f'invalid na_position: {na_position}')
        if key is not None:
            values_to_sort = cast(Series, ensure_key_mapped(self, key))._values
        else:
            values_to_sort = self._values
        sorted_index = nargsort(values_to_sort, kind, bool(ascending), na_position)
        if is_range_indexer(sorted_index, len(sorted_index)):
            if inplace:
                return self._update_inplace(self)
            return self.copy(deep=False)
        result = self._constructor(
            self._values[sorted_index],
            index=self.index[sorted_index],
            copy=False
        ).__finalize__(self, method='sort_values')
        if ignore_index:
            result.index = default_index(len(sorted_index))
        if not inplace:
            return result
        else:
            self._update_inplace(result)
            return None

    @overload
    def sort_index(
        self,
        *,
        axis: Union[int, Literal[0]] = ...,
        level: Optional[Union[int, str, Sequence[Union[int, str]]]] = ...,
        ascending: Union[bool, TypingSequence[bool]] = ...,
        inplace: bool = ...,
        kind: Literal['quicksort', 'mergesort', 'heapsort', 'stable'] = ...,
        na_position: Literal['first', 'last'] = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: Optional[Callable[[Any], Any]] = ...
    ) -> Optional[Series]:
        ...

    @overload
    def sort_index(
        self,
        *,
        axis: Union[int, Literal[0]],
        level: Optional[Union[int, str, Sequence[Union[int, str]]]],
        ascending: Union[bool, TypingSequence[bool]],
        inplace: bool,
        kind: Literal['quicksort', 'mergesort', 'heapsort', 'stable'],
        na_position: Literal['first', 'last'],
        sort_remaining: bool,
        ignore_index: bool,
        key: Optional[Callable[[Any], Any]]
    ) -> Optional[Series]:
        ...

    @overload
    def sort_index(
        self,
        *,
        axis: Union[int, Literal[0]] = ...,
        level: Optional[Union[int, str, Sequence[Union[int, str]]]] = ...,
        ascending: Union[bool, TypingSequence[bool]] = ...,
        inplace: bool = ...,
        kind: Literal['quicksort', 'mergesort', 'heapsort', 'stable'] = ...,
        na_position: Literal['first', 'last'] = ...,
        sort_remaining: bool = ...,
        ignore_index: bool = ...,
        key: Optional[Callable[[Any], Any]] = ...
    ) -> Optional[Series]:
        ...

    def sort_index(
        self,
        *,
        axis: Union[int, Literal[0]] = 0,
        level: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        ascending: Union[bool, TypingSequence[bool]] = True,
        inplace: bool = False,
        kind: Literal['quicksort', 'mergesort', 'heapsort', 'stable'] = 'quicksort',
        na_position: Literal['first', 'last'] = 'last',
        sort_remaining: bool = True,
        ignore_index: bool = False,
        key: Optional[Callable[[Any], Any]] = None
    ) -> Optional[Series]:
        """
        Sort Series by index labels.

        Returns a new Series sorted by label if `inplace` argument is
        ``False``, otherwise updates the original series and returns None.

        Parameters
        ----------
        mapper : scalar, hashable sequence, dict-like or function, optional
            Value to set the axis name attribute.

            Use either ``mapper`` and ``axis`` to
            specify the axis to target with ``mapper``, or ``index``.
        axis : {0 or 'index'}, default 0
            The axis to rename. For `Series` this parameter is unused and defaults to 0.
        level : int or level name, default None
            In case of MultiIndex, only rename labels in the specified level.
        ascending : Union[bool, Sequence[bool]], default True
            Sort ascending vs. descending. When the index is a MultiIndex the
            sort direction can be controlled for each level individually.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : Literal['quicksort', 'mergesort', 'heapsort', 'stable'], default 'quicksort'
            Choice of sorting algorithm. See also :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable algorithms. For
            DataFrames, this option is only applied when sorting on a single
            column or label.
        na_position : Literal['first', 'last'], default 'last'
            If 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end. Not implemented for MultiIndex.
        sort_remaining : bool, default True
            If True and sorting by level and index is multilevel, sort by other
            levels too (in order) after sorting by specified level.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
        key : Optional[Callable[[Any], Any]], optional
            If not None, apply the key function to the index values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect an
            ``Index`` and return an ``Index`` of the same shape.

        Returns
        -------
        Optional[Series]
            The original Series sorted by the labels or None if ``inplace=True``.

        See Also
        --------
        DataFrame.sort_index: Sort DataFrame by the index.
        DataFrame.sort_values: Sort DataFrame by the value.
        Series.sort_values : Sort Series by the value.

        Examples
        --------
        >>> s = pd.Series(["a", "b", "c", "d"], index=[3, 2, 1, 4])
        >>> s.sort_index()
        1    c
        2    b
        3    a
        4    d
        dtype: object

        >>> s.sort_index(ascending=False)
        4    d
        3    a
        2    b
        1    c
        dtype: object

        >>> s = pd.Series(["a", "b", "c", "d"], index=[3, 2, 1, np.nan])
        >>> s.sort_index(na_position="first")
        NaN     d
         1.0    c
         2.0    b
         3.0    a
        dtype: object

        >>> arrays = [
        ...     np.array(["qux", "qux", "foo", "foo", "baz", "baz", "bar", "bar"]),
        ...     np.array(["two", "one", "two", "one", "two", "one", "two", "one"]),
        ... ]
        >>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=arrays)
        >>> s.sort_index(level=1)
        bar  one    8
        baz  one    6
        foo  one    4
        qux  one    2
        bar  two    7
        baz  two    5
        foo  two    3
        qux  two    1
        dtype: int64

        >>> s.sort_index(level=1, sort_remaining=False)
        qux  one    2
        foo  one    4
        baz  one    6
        bar  one    8
        qux  two    1
        foo  two    3
        baz  two    5
        bar  two    7
        dtype: int64

        >>> s = pd.Series([1, 2, 3, 4], index=["A", "b", "C", "d"])
        >>> s.sort_index(key=lambda x: x.str.lower())
        A    1
        b    2
        C    3
        d    4
        dtype: int64
        """
        return super().sort_index(
            axis=axis,
            level=level,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
            sort_remaining=sort_remaining,
            ignore_index=ignore_index,
            key=key
        )

    def argsort(
        self,
        axis: Union[int, Literal[0]] = 0,
        kind: Literal['mergesort', 'quicksort', 'heapsort', 'stable'] = 'quicksort',
        order: Optional[Any] = None,
        stable: Optional[bool] = None
    ) -> Series:
        """
        Return the integer indices that would sort the Series values.

        Override ndarray.argsort. Argsorts the value, omitting NA/null values,
        and places the result in the same locations as the non-NA values.

        Parameters
        ----------
        axis : Union[int, Literal[0]], default 0
            Unused. Parameter needed for compatibility with DataFrame.
        kind : Literal['mergesort', 'quicksort', 'heapsort', 'stable'], default 'quicksort'
            Choice of sorting algorithm. See :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable algorithms.
        order : None
            Has no effect but is accepted for compatibility with numpy.
        stable : Optional[bool], default None
            Has no effect but is accepted for compatibility with numpy.

        Returns
        -------
        Series[np.intp]
            Positions of values within the sort order with -1 indicating
            nan values.

        See Also
        --------
        numpy.ndarray.argsort : Returns the indices that would sort this array.

        Examples
        --------
        >>> s = pd.Series([3, 2, 1])
        >>> s.argsort()
        0    2
        1    1
        2    0
        dtype: int64
        """
        if axis != -1:
            self._get_axis_number(axis)
        result = self.array.argsort(kind=kind)
        res = self._constructor(result, index=self.index, name=self.name, dtype=np.intp, copy=False)
        return res.__finalize__(self, method='argsort')

    def nlargest(self, n: int = 5, keep: Union[str, bool] = 'first') -> Series:
        """
        Return the largest `n` elements.

        Parameters
        ----------
        n : int, default 5
            Return this many descending sorted values.
        keep : Union[str, bool], default 'first'
            When there are duplicate values that cannot all fit in a
            Series of `n` elements:

            - ``'first'`` : return the first `n` occurrences in order
              of appearance.
            - ``'last'`` : return the last `n` occurrences in reverse
              order of appearance.
            - ``'all'`` : keep all occurrences. This can result in a Series of
              size larger than `n`.

        Returns
        -------
        Series
            The `n` largest values in the Series, sorted in decreasing order.

        See Also
        --------
        Series.nsmallest: Get the `n` smallest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values(ascending=False).head(n)`` for small `n`
        relative to the size of the ``Series`` object.

        Examples
        --------
        >>> countries_population = {
        ...     "Italy": 59000000,
        ...     "France": 65000000,
        ...     "Malta": 434000,
        ...     "Maldives": 434000,
        ...     "Brunei": 434000,
        ...     "Iceland": 337000,
        ...     "Nauru": 11300,
        ...     "Tuvalu": 11300,
        ...     "Anguilla": 11300,
        ...     "Montserrat": 5200,
        ... }
        >>> s = pd.Series(countries_population)
        >>> s
        Italy       59000000
        France      65000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        Iceland       337000
        Nauru          11300
        Tuvalu         11300
        Anguilla       11300
        Montserrat      5200
        dtype: int64

        The `n` largest elements where ``n=5`` by default.

        >>> s.nlargest()
        France      65000000
        Italy       59000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        dtype: int64

        The `n` largest elements where ``n=3``. Default `keep` value is 'first'
        so Malta will be kept.

        >>> s.nlargest(3)
        France    65000000
        Italy     59000000
        Malta       434000
        dtype: int64

        The `n` largest elements where ``n=3`` and keeping the last duplicates.
        Brunei will be kept since it is the last with value 434000 based on
        the index order.

        >>> s.nlargest(3, keep="last")
        France      65000000
        Italy       59000000
        Brunei        434000
        dtype: int64

        The `n` largest elements where ``n=3`` with all duplicates kept. Note
        that the returned Series has five elements due to the three duplicates.

        >>> s.nlargest(3, keep="all")
        France      65000000
        Italy       59000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        dtype: int64
        """
        return selectn.SelectNSeries(self, n=n, keep=keep).nlargest()

    def nsmallest(self, n: int = 5, keep: Union[str, bool] = 'first') -> Series:
        """
        Return the smallest `n` elements.

        Parameters
        ----------
        n : int, default 5
            Return this many ascending sorted values.
        keep : Union[str, bool], default 'first'
            When there are duplicate values that cannot all fit in a
            Series of `n` elements:

            - ``'first'`` : return the first `n` occurrences in order
              of appearance.
            - ``'last'`` : return the last `n` occurrences in reverse
              order of appearance.
            - ``'all'`` : keep all occurrences. This can result in a Series of
              size larger than `n`.

        Returns
        -------
        Series
            The `n` smallest values in the Series, sorted in increasing order.

        See Also
        --------
        Series.nlargest: Get the `n` largest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values().head(n)`` for small `n` relative to
        the size of the ``Series`` object.

        Examples
        --------
        >>> countries_population = {
        ...     "Italy": 59000000,
        ...     "France": 65000000,
        ...     "Brunei": 434000,
        ...     "Malta": 434000,
        ...     "Maldives": 434000,
        ...     "Iceland": 337000,
        ...     "Nauru": 11300,
        ...     "Tuvalu": 11300,
        ...     "Anguilla": 11300,
        ...     "Montserrat": 5200,
        ... }
        >>> s = pd.Series(countries_population)
        >>> s
        Italy       59000000
        France      65000000
        Brunei        434000
        Malta         434000
        Maldives      434000
        Iceland       337000
        Nauru          11300
        Tuvalu         11300
        Anguilla       11300
        Montserrat      5200
        dtype: int64

        The `n` smallest elements where ``n=5`` by default.

        >>> s.nsmallest()
        Montserrat    5200
        Nauru        11300
        Tuvalu       11300
        Anguilla     11300
        Iceland     337000
        dtype: int64

        The `n` smallest elements where ``n=3``. Default `keep` value is
        'first' so Nauru and Tuvalu will be kept.

        >>> s.nsmallest(3)
        Montserrat   5200
        Nauru       11300
        Tuvalu      11300
        dtype: int64

        The `n` smallest elements where ``n=3`` and keeping the last
        duplicates. Anguilla and Tuvalu will be kept since they are the last
        with value 11300 based on the index order.

        >>> s.nsmallest(3, keep="last")
        Montserrat   5200
        Anguilla    11300
        Tuvalu      11300
        dtype: int64

        The `n` smallest elements where ``n=3`` with all duplicates kept. Note
        that the returned Series has four elements due to the three duplicates.

        >>> s.nsmallest(3, keep="all")
        Montserrat   5200
        Nauru       11300
        Tuvalu      11300
        Anguilla    11300
        dtype: int64
        """
        return selectn.SelectNSeries(self, n=n, keep=keep).nsmallest()

    def swaplevel(
        self,
        i: Union[int, str] = -2,
        j: Union[int, str] = -1,
        copy: Optional[bool] = None
    ) -> Series:
        """
        Swap levels i and j in a :class:`MultiIndex`.

        Default is to swap the two innermost levels of the index.

        Parameters
        ----------
        i : Union[int, str], default -2
            Levels of the indices to be swapped. Can pass level name as string.
        j : Union[int, str], default -1
            Levels of the indices to be swapped. Can pass level name as string.
        copy : Optional[bool], default None
                        Whether to copy underlying data.

                        .. note::
                            The `copy` keyword will change behavior in pandas 3.0.
                            `Copy-on-Write
                            <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                            will be enabled by default, which means that all methods with a
                            `copy` keyword will use a lazy copy mechanism to defer the copy
                            and ignore the `copy` keyword. The `copy` keyword will be
                            removed in a future version of pandas.

                            You can already get the future behavior and improvements through
                            enabling copy on write ``pd.options.mode.copy_on_write = True``

                        .. deprecated:: 3.0.0

        Returns
        -------
        Series
            Series with levels swapped in MultiIndex.

        See Also
        --------
        DataFrame.swaplevel : Swap levels i and j in a :class:`DataFrame`.
        Series.reorder_levels : Rearrange index levels using input order.
        MultiIndex.swaplevel : Swap levels i and j in a :class:`MultiIndex`.

        Examples
        --------
        >>> s = pd.Series(
        ...     ["A", "B", "A", "C"],
        ...     index=[
        ...         ["Final exam", "Final exam", "Coursework", "Coursework"],
        ...         ["History", "Geography", "History", "Geography"],
        ...         ["January", "February", "March", "April"],
        ...     ],
        ... )
        >>> s
        Final exam  History     January      A
                    Geography   February     B
        Coursework  History     March        A
                    Geography   April        C
        dtype: object

        In the following example, we will swap the levels of the indices.
        Here, we will swap the levels column-wise, but levels can be swapped row-wise
        in a similar manner. Note that column-wise is the default behavior.
        By not supplying any arguments for i and j, we swap the last and second to
        last indices.

        >>> s.swaplevel()
        Final exam  January     History         A
                    February    Geography       B
        Coursework  March       History         A
                    April       Geography       C
        dtype: object

        By supplying one argument, we can choose which index to swap the last
        index with. We can for example swap the first index with the last one as
        follows.

        >>> s.swaplevel(0)
        January     History     Final exam      A
        February    Geography   Final exam      B
        March       History     Coursework      A
        April       Geography   Coursework      C
        dtype: object

        We can also define explicitly which indices we want to swap by supplying values
        for both i and j. Here, we for example swap the first and second indices.

        >>> s.swaplevel(0, 1)
        History     Final exam  January         A
        Geography   Final exam  February        B
        History     Coursework  March           A
        Geography   Coursework  April           C
        dtype: object
        """
        self._check_copy_deprecation(copy)
        assert isinstance(self.index, MultiIndex)
        result = self.copy(deep=False)
        result.index = self.index.swaplevel(i, j)
        return result

    def reorder_levels(
        self,
        order: Sequence[Union[int, str]]
    ) -> Series:
        """
        Rearrange index levels using input order.

        May not drop or duplicate levels.

        Parameters
        ----------
        order : Sequence[Union[int, str]]
            Reference level by number or key.

        Returns
        -------
        Series
            Type of caller with index as MultiIndex (new object).

        See Also
        --------
        DataFrame.reorder_levels : Rearrange index or column levels using
            input ``order``.

        Examples
        --------
        >>> arrays = [
        ...     np.array(["dog", "dog", "cat", "cat", "bird", "bird"]),
        ...     np.array(["white", "black", "white", "black", "white", "black"]),
        ... ]
        >>> s = pd.Series([1, 2, 3, 4], index=arrays)
        >>> s
        dog   white    1
              black    2
        cat   white    3
              black    4
        bird  white    5
              black    6
        dtype: int64
        >>> s.reorder_levels([1, 0])
        white  bird    5
        black  bird    6
        white  cat     3
        black  cat     4
        white  dog     1
        black  dog     2
        dtype: int64
        """
        if not isinstance(self.index, MultiIndex):
            raise Exception('Can only reorder levels on a hierarchical axis.')
        result = self.copy(deep=False)
        assert isinstance(result.index, MultiIndex)
        result.index = result.index.reorder_levels(order)
        return result

    def explode(self, ignore_index: bool = False) -> Series:
        """
        Transform each element of a list-like to a row.

        Parameters
        ----------
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        Series
            Exploded lists to rows; index will be duplicated for these rows.

        See Also
        --------
        Series.str.split : Split string values on specified separator.
        Series.unstack : Unstack, a.k.a. pivot, Series with MultiIndex
            to produce DataFrame.
        DataFrame.melt : Unpivot a DataFrame from wide format to long format.
        DataFrame.explode : Explode a DataFrame from list-like
            columns to long format.

        Notes
        -----
        This routine will explode list-likes including lists, tuples, sets,
        Series, and np.ndarray. The result dtype of the subset rows will
        be object. Scalars will be returned unchanged, and empty list-likes will
        result in a np.nan for that row. In addition, the ordering of elements in
        the output will be non-deterministic when exploding sets.

        Reference :ref:`the user guide <reshaping.explode>` for more examples.

        Examples
        --------
        >>> s = pd.Series([[1, 2, 3], "foo", [], [3, 4]])
        >>> s
        0    [1, 2, 3]
        1          foo
        2           []
        3       [3, 4]
        dtype: object

        >>> s.explode()
        0      1
        0      2
        0      3
        1    foo
        2    NaN
        3      3
        3      4
        dtype: object
        """
        if isinstance(self.dtype, ExtensionDtype):
            values, counts = self._values._explode()
        elif len(self) and is_object_dtype(self.dtype):
            values, counts = reshape.explode(np.asarray(self._values))
        else:
            result = self.copy()
            return result.reset_index(drop=True) if ignore_index else result
        if ignore_index:
            index = default_index(len(values))
        else:
            index = self.index.repeat(counts)
        return self._constructor(values, index=index, name=self.name, copy=False)

    def unstack(
        self,
        level: Union[int, str, Sequence[Union[int, str]]] = -1,
        fill_value: Optional[Any] = None,
        sort: bool = True
    ) -> DataFrame:
        """
        Unstack, also known as pivot, Series with MultiIndex to produce DataFrame.

        Parameters
        ----------
        level : Union[int, str, Sequence[Union[int, str]]], default -1
            Level(s) to unstack, can pass level name.
        fill_value : Optional[Any], default None
            Value to use when replacing NaN values.
        sort : bool, default True
            Sort the level(s) in the resulting MultiIndex columns.

        Returns
        -------
        DataFrame
            Unstacked Series.

        See Also
        --------
        DataFrame.unstack : Pivot the MultiIndex of a DataFrame.

        Notes
        -----
        Reference :ref:`the user guide <reshaping.stacking>` for more examples.

        Examples
        --------
        >>> s = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.MultiIndex.from_product([["one", "two"], ["a", "b"]]),
        ... )
        >>> s
        one  a    1
             b    2
        two  a    3
             b    4
        dtype: int64

        >>> s.unstack(level=-1)
             a  b
        one  1  2
        two  3  4

        >>> s.unstack(level=0)
           one  two
        a    1    3
        b    2    4
        """
        from pandas.core.reshape.reshape import unstack
        return unstack(self, level, fill_value, sort)

    def map(
        self,
        arg: Union[Callable[[Any], Any], Mapping[Any, Any], Series],
        na_action: Optional[Literal['ignore']] = None,
        **kwargs: Any
    ) -> Series:
        """
        Map values of Series according to an input mapping or function.

        Used for substituting each value in a Series with another value,
        that may be derived from a function, a ``dict`` or
        a :class:`Series`.

        Parameters
        ----------
        arg : Union[Callable[[Any], Any], Mapping[Any, Any], Series]
            Mapping correspondence.
        na_action : Optional[Literal['ignore']], default None
            If 'ignore', propagate NaN values, without passing them to the
            mapping correspondence.
        **kwargs : Any
            Additional keyword arguments passed to func.

            .. versionadded:: 3.0.0

        Returns
        -------
        Series
            Same index as caller.

        See Also
        --------
        Series.apply : For applying more complex functions on a Series.
        Series.replace: Replace values given in `to_replace` with `value`.
        DataFrame.apply : Apply a function row-/column-wise.
        DataFrame.map : Apply a function elementwise on a whole DataFrame.

        Notes
        -----
        When ``arg`` is a dictionary, values in Series that are not in the
        dictionary (as keys) are converted to ``NaN``. However, if the
        dictionary is a ``dict`` subclass that defines ``__missing__`` (i.e.
        provides a method for default values), then this default is used
        rather than ``NaN``.

        Examples
        --------
        >>> s = pd.Series(["cat", "dog", np.nan, "rabbit"])
        >>> s
        0      cat
        1      dog
        2      NaN
        3   rabbit
        dtype: object

        ``map`` accepts a ``dict`` or a ``Series``. Values that are not found
        in the ``dict`` are converted to ``NaN``, unless the dict has a default
        value (e.g. ``defaultdict``):

        >>> s.map({"cat": "kitten", "dog": "puppy"})
        0   kitten
        1    puppy
        2      NaN
        3      NaN
        dtype: object

        It also accepts a function:

        >>> s.map("I am a {}".format)
        0       I am a cat
        1       I am a dog
        2       I am a nan
        3    I am a rabbit
        dtype: object

        To avoid applying the function to missing values (and keep them as
        ``NaN``) ``na_action='ignore'`` can be used:

        >>> s.map("I am a {}".format, na_action="ignore")
        0     I am a cat
        1     I am a dog
        2            NaN
        3  I am a rabbit
        dtype: object

        Passing a single string as ``s.map('llama')`` will raise an error. Use
        a list of one element instead:

        >>> s.map(["llama"])
        0         NaN
        1         NaN
        2         NaN
        3         NaN
        dtype: object

        Strings and integers are distinct and are therefore not comparable:

        >>> pd.Series([1]).map(["1"])
        0    NaN
        dtype: float64
        >>> pd.Series([1.1]).map(["1.1"])
        0    NaN
        dtype: float64
        """
        if callable(arg):
            arg = functools.partial(arg, **kwargs)
        new_values = self._map_values(arg, na_action=na_action)
        return self._constructor(new_values, index=self.index, copy=False).__finalize__(self, method='map')

    def _gotitem(
        self,
        key: Any,
        ndim: int,
        subset: Optional[Any] = None
    ) -> Series:
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : object
        ndim : {1, 2}
            Requested ndim of result.
        subset : Optional[Any], default None
            Subset to act on.

        Returns
        -------
        Series
            A sliced Series object.
        """
        return self

    _agg_see_also_doc = dedent('\n    See Also\n    --------\n    Series.apply : Invoke function on a Series.\n    Series.transform : Transform function producing a Series with like indexes.\n    ')
    _agg_examples_doc = dedent("\n    Examples\n    --------\n    >>> s = pd.Series([1, 2, 3, 4])\n    >>> s\n    0    1\n    1    2\n    2    3\n    3    4\n    dtype: int64\n\n    >>> s.agg('min')\n    1\n\n    >>> s.agg(['min', 'max'])\n    min   1\n    max   4\n    dtype: int64\n    ")

    @doc(_shared_docs['aggregate'], klass=_shared_doc_kwargs['klass'], axis=_shared_doc_kwargs['axis'], see_also=_agg_see_also_doc, examples=_agg_examples_doc)
    def aggregate(
        self,
        func: Optional[
            Union[
                Callable[[Any], Any],
                List[Callable[[Any], Any]],
                Dict[Any, Callable[[Any], Any]]
            ]
        ] = None,
        axis: int = 0,
        *args: Any,
        **kwargs: Any
    ) -> Union[Any, Series, DataFrame]:
        self._get_axis_number(axis)
        if func is None:
            func = dict(kwargs.items())
        op = SeriesApply(self, func, args=args, kwargs=kwargs)
        result = op.agg()
        return result

    agg = aggregate

    @doc(_shared_docs['transform'], klass=_shared_doc_kwargs['klass'], axis=_shared_doc_kwargs['axis'])
    def transform(
        self,
        func: Callable[..., Any],
        axis: int = 0,
        *args: Any,
        **kwargs: Any
    ) -> Series:
        self._get_axis_number(axis)
        ser = self.copy(deep=False)
        result = SeriesApply(ser, func=func, args=args, kwargs=kwargs).transform()
        return result

    def apply(
        self,
        func: Union[Callable[..., Any], str],
        args: tuple = (),
        *,
        by_row: Union[str, bool] = 'compat',
        **kwargs: Any
    ) -> Union[Any, Series, DataFrame]:
        """
        Invoke function on values of Series.

        Can be ufunc (a NumPy function that applies to the entire Series)
        or a Python function that only works on single values.

        Parameters
        ----------
        func : Union[Callable[..., Any], str]
            Python function or NumPy ufunc to apply.
        args : tuple, default ()
            Positional arguments passed to func after the series value.
        by_row : Union[str, bool], default "compat"
            If ``"compat"`` and func is a callable, func will be passed each element of
            the Series, like ``Series.map``. If func is a list or dict of
            callables, will first try to translate each func into pandas methods. If
            that doesn't work, will try call to apply again with ``by_row="compat"``
            and if that fails, will call apply again with ``by_row=False``
            (backward compatible).
            If False, the func will be passed the whole Series at once.

            ``by_row`` has no effect when ``func`` is a string.

            .. versionadded:: 2.1.0
        **kwargs : Any
            Additional keyword arguments passed to func.

        Returns
        -------
        Union[Any, Series, DataFrame]
            If func returns a Series object the result will be a DataFrame.

        See Also
        --------
        Series.map: For element-wise operations.
        Series.agg: Only perform aggregating type operations.
        Series.transform: Only perform transforming type operations.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        Create a series with typical summer temperatures for each city.

        >>> s = pd.Series([20, 21, 12, 15], index=["London", "New York", "Helsinki", "Tokyo"])
        >>> s
        London      20
        New York    21
        Helsinki    12
        Tokyo       15
        dtype: int64

        Square the values by defining a function and passing it as an
        argument to ``apply()``.

        >>> def square(x):
        ...     return x**2
        >>> s.apply(square)
        London      400
        New York    441
        Helsinki    144
        Tokyo       225
        dtype: int64

        Square the values by passing an anonymous function as an
        argument to ``apply()``.

        >>> s.apply(lambda x: x**2)
        London      400
        New York    441
        Helsinki    144
        Tokyo       225
        dtype: int64

        Define a custom function that needs additional positional
        arguments and pass these additional arguments using the
        ``args`` keyword.

        >>> def subtract_custom_value(x, custom_value):
        ...     return x - custom_value

        >>> s.apply(subtract_custom_value, args=(5,))
        London      15
        New York    16
        Helsinki     7
        Tokyo       10
        dtype: int64

        Define a custom function that takes keyword arguments
        and pass these arguments to ``apply``.
    
        >>> def add_custom_values(x, **kwargs):
        ...     for month in kwargs:
        ...         x += kwargs[month]
        ...     return x

        >>> s.apply(add_custom_values, june=30, july=20, august=25)
        London      95
        New York    96
        Helsinki    87
        Tokyo       90
        dtype: int64

        Use a function from the Numpy library.

        >>> s.apply(np.log)
        London      2.995732
        New York    3.044522
        Helsinki    2.484907
        Tokyo       2.708050
        dtype: float64
        """
        return SeriesApply(self, func, by_row=by_row, args=args, kwargs=kwargs).apply()

    def _reindex_indexer(
        self,
        new_index: Optional[Index],
        indexer: Optional[Union[np.ndarray, list, slice, int]]
    ) -> Series:
        if indexer is None and (new_index is None or new_index.names == self.index.names):
            return self.copy(deep=False)
        new_values = algorithms.take_nd(self._values, indexer, allow_fill=True, fill_value=None)
        return self._constructor(new_values, index=new_index, copy=False)

    def _needs_reindex_multi(
        self,
        axes: list[Index],
        method: Optional[str],
        level: Optional[Union[int, str, Sequence[Union[int, str]]]]
    ) -> bool:
        """
        Check if we do need a multi reindex; this is for compat with
        higher dims.
        """
        return False

    @overload
    def rename(
        self,
        index: Optional[Union[Hashable, Callable[[Any], Any], Mapping[Any, Any], Sequence[Any]]] = ...,
        *,
        axis: Optional[Union[int, str]] = ...,
        copy: bool = ...,
        inplace: bool = ...,
        level: Optional[Union[int, str]] = ...,
        errors: Literal['ignore', 'raise'] = ...
    ) -> Optional[Series]:
        ...

    @overload
    def rename(
        self,
        index: Optional[Union[Hashable, Callable[[Any], Any], Mapping[Any, Any], Sequence[Any]]] = ...,
        *,
        axis: Optional[Union[int, str]],
        copy: bool,
        inplace: bool,
        level: Optional[Union[int, str]],
        errors: Literal['ignore', 'raise']
    ) -> Optional[Series]:
        ...

    @overload
    def rename(
        self,
        index: Optional[Union[Hashable, Callable[[Any], Any], Mapping[Any, Any], Sequence[Any]]] = ...,
        *,
        axis: Optional[Union[int, str]] = ...,
        copy: bool = ...,
        inplace: bool,
        level: Optional[Union[int, str]] = ...,
        errors: Literal['ignore', 'raise'] = ...
    ) -> Optional[Series]:
        ...

    def rename(
        self,
        index: Optional[Union[Hashable, Callable[[Any], Any], Mapping[Any, Any], Sequence[Any]]] = None,
        *,
        axis: Optional[Union[int, str]] = None,
        copy: Optional[bool] = lib.no_default,
        inplace: bool = False,
        level: Optional[Union[int, str]] = None,
        errors: Literal['ignore', 'raise'] = 'ignore'
    ) -> Optional[Series]:
        """
        Alter Series index labels or name.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is. Extra labels listed don't throw an
        error.
    
        Alternatively, change ``Series.name`` with a scalar value.

        See the :ref:`user guide <basics.rename>` for more.

        Parameters
        ----------
        index : Optional[Union[Hashable, Callable[[Any], Any], Mapping[Any, Any], Sequence[Any]]]
            Functions or dict-like are transformations to apply to
            the index.
            Scalar or hashable sequence-like will alter the ``Series.name``
            attribute.
        axis : Optional[Union[int, str]], default None
            Unused. Parameter needed for compatibility with DataFrame.
        copy : Optional[bool], default False
            Also copy underlying data.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

            .. deprecated:: 3.0.0
        inplace : bool, default False
            Whether to return a new Series. If True the value of copy is ignored.
        level : Optional[Union[int, str]], default None
            In case of MultiIndex, level for which the labels will be removed.
        errors : Literal['ignore', 'raise'], default 'ignore'
            If 'raise', raise `KeyError` when a `dict-like mapper` or
            `index` contains labels that are not present in the index being transformed.
            If 'ignore', existing keys will be renamed and extra keys will be ignored.

        Returns
        -------
        Optional[Series]
            Series with index labels or name altered or None if ``inplace=True``.

        See Also
        --------
        DataFrame.rename : Corresponding DataFrame method.
        Series.rename_axis : Set the name of the axis.

        Examples
        --------

        >>> s = pd.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.rename("my_name")  # scalar, changes Series.name
        0    1
        1    2
        2    3
        Name: my_name, dtype: int64
        >>> s.rename(lambda x: x**2)  # function, changes labels
        0    1
        1    2
        4    3
        dtype: int64
        >>> s.rename({1: 3, 2: 5})  # mapping, changes labels
        0    1
        3    2
        5    3
        dtype: int64
        """
        self._check_copy_deprecation(copy)
        if axis is not None:
            axis = self._get_axis_number(axis)
        if callable(index) or is_dict_like(index):
            return super()._rename(index, inplace=inplace, level=level, errors=errors)
        else:
            return self._set_name(index, inplace=inplace)

    @Appender("\n        Examples\n        --------\n        >>> s = pd.Series([1, 2, 3])\n        >>> s.set_axis(['a', 'b', 'c'], axis=0)\n        a    1\n        b    2\n        c    3\n        dtype: int64\n    ")
    @Substitution(
        klass=_shared_doc_kwargs['klass'],
        axes_single_arg=_shared_doc_kwargs['axes_single_arg'],
        extended_summary_sub='',
        axis_description_sub='',
        see_also_sub=''
    )
    @Appender(NDFrame.set_axis.__doc__)
    def set_axis(
        self,
        labels: Union[Index, Sequence[Any], Hashable],
        *,
        axis: Optional[Union[int, str]] = 0,
        copy: Optional[bool] = lib.no_default
    ) -> Optional[Series]:
        return super().set_axis(labels=labels, axis=axis, copy=copy)

    @doc(NDFrame.reindex, klass=_shared_doc_kwargs['klass'], optional_reindex=_shared_doc_kwargs['optional_reindex'])
    def reindex(
        self,
        index: Optional[Index] = None,
        *,
        axis: Optional[int] = None,
        method: Optional[str] = None,
        copy: Optional[bool] = lib.no_default,
        level: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        fill_value: Optional[Any] = None,
        limit: Optional[int] = None,
        tolerance: Optional[Union[str, float]] = None
    ) -> Series:
        return super().reindex(
            index=index,
            method=method,
            level=level,
            fill_value=fill_value,
            limit=limit,
            tolerance=tolerance,
            copy=copy
        )

    @overload
    def rename_axis(
        self,
        mapper: Optional[Union[Hashable, Sequence[Hashable], Callable[[Any], Any]]] = ...,
        *,
        index: Optional[Union[Hashable, Sequence[Hashable], Callable[[Any], Any]]] = ...,
        axis: Optional[Union[int, str]] = ...,
        copy: Optional[bool] = ...,
        inplace: bool = ...,
    ) -> Optional[Series]:
        ...

    @overload
    def rename_axis(
        self,
        mapper: Optional[Union[Hashable, Sequence[Hashable], Callable[[Any], Any]]] = ...,
        *,
        index: Optional[Union[Hashable, Sequence[Hashable], Callable[[Any], Any]]] = ...,
        axis: Optional[Union[int, str]] = ...,
        copy: Optional[bool] = ...,
        inplace: bool,
    ) -> Optional[Series]:
        ...

    @overload
    def rename_axis(
        self,
        mapper: Optional[Union[Hashable, Sequence[Hashable], Callable[[Any], Any]]] = ...,
        *,
        index: Optional[Union[Hashable, Sequence[Hashable], Callable[[Any], Any]]] = ...,
        axis: Optional[Union[int, str]] = ...,
        copy: Optional[bool] = ...,
        inplace: bool,
    ) -> Optional[Series]:
        ...

    def rename_axis(
        self,
        mapper: Optional[Union[Hashable, Sequence[Hashable], Callable[[Any], Any]]] = lib.no_default,
        *,
        index: Optional[Union[Hashable, Sequence[Hashable], Callable[[Any], Any]]] = lib.no_default,
        axis: Union[int, str] = 0,
        copy: Optional[bool] = lib.no_default,
        inplace: bool = False
    ) -> Optional[Series]:
        """
        Set the name of the axis for the index.

        Parameters
        ----------
        mapper : Optional[Union[Hashable, Sequence[Hashable], Callable[[Any], Any]]], default None
            Value to set the axis name attribute.

            Use either ``mapper`` and ``axis`` to
            specify the axis to target with ``mapper``, or ``index``.

        index : Optional[Union[Hashable, Sequence[Hashable], Callable[[Any], Any]]], default None
            A scalar, list-like, dict-like or functions transformations to
            apply to that axis' values.
        axis : Union[int, str], default 0
            The axis to rename. For `Series` this parameter is unused and defaults to 0.
        copy : Optional[bool], default False
            Whether or not to return a copy.
    
            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.
    
                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``
    
            .. deprecated:: 3.0.0
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Series
            or DataFrame.

        Returns
        -------
        Optional[Series]
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Series.rename : Alter Series index labels or name.
        DataFrame.rename : Alter DataFrame index labels or name.
        Index.rename : Set new names on index.

        Examples
        --------

        >>> s = pd.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.rename_axis("animal")
        animal
        0    1
        1    2
        2    3
        dtype: int64
        """
        return super().rename_axis(mapper=mapper, index=index, axis=axis, inplace=inplace, copy=copy)

    @overload
    def drop(
        self,
        labels: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        *,
        axis: Union[int, str] = ...,
        index: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        columns: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        level: Optional[Union[int, str]] = ...,
        inplace: bool = ...,
        errors: Literal['raise', 'ignore'] = ...
    ) -> Optional[Series]:
        ...

    @overload
    def drop(
        self,
        labels: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        *,
        axis: Union[int, str],
        index: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        columns: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        level: Optional[Union[int, str]] = ...,
        inplace: bool,
        errors: Literal['raise', 'ignore'] = ...
    ) -> Optional[Series]:
        ...

    @overload
    def drop(
        self,
        labels: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        *,
        axis: Union[int, str] = ...,
        index: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        columns: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        level: Optional[Union[int, str]] = ...,
        inplace: bool,
        errors: Literal['raise', 'ignore'] = ...
    ) -> Optional[Series]:
        ...

    def drop(
        self,
        labels: Optional[Union[Hashable, Sequence[Hashable]]] = None,
        *,
        axis: Union[int, str] = 0,
        index: Optional[Union[Hashable, Sequence[Hashable]]] = None,
        columns: Optional[Union[Hashable, Sequence[Hashable]]] = None,
        level: Optional[Union[int, str]] = None,
        inplace: bool = False,
        errors: Literal['raise', 'ignore'] = 'raise'
    ) -> Optional[Series]:
        """
        Return Series with specified index labels removed.

        Remove elements of a Series based on specifying the index labels.
        When using a multi-index, labels on different levels can be removed
        by specifying the level.

        Parameters
        ----------
        labels : Optional[Union[Hashable, Sequence[Hashable]]]
            Index labels to drop.
        axis : Union[int, str], default 0
            {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        index : Optional[Union[Hashable, Sequence[Hashable]]], default None
            Redundant for application on Series, but 'index' can be used instead
            of 'labels'.
        columns : Optional[Union[Hashable, Sequence[Hashable]]], default None
            No change is made to the Series; use 'index' or 'labels' instead.
        level : Optional[Union[int, str]], default None
            For MultiIndex, level for which the labels will be removed.
        inplace : bool, default False
            If True, do operation inplace and return None.
        errors : Literal['raise', 'ignore'], default 'raise'
            If 'raise', raise ``KeyError`` when none of the labels are found in the index.
            If 'ignore', existing labels will be dropped and if none of the new labels are
            found then no error is raised.

        Returns
        -------
        Optional[Series]
            Series with specified index labels removed or None if ``inplace=True``.

        Raises
        ------
        KeyError
            If none of the labels are found in the index.

        See Also
        --------
        Series.reindex : Return only specified index labels of Series.
        Series.dropna : Return series without null values.
        Series.drop_duplicates : Return Series with duplicate values removed.
        DataFrame.drop : Drop specified labels from rows or columns.
        """
        return super().drop(
            labels=labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            inplace=inplace,
            errors=errors
        )

    def pop(self, item: Hashable) -> Any:
        """
        Return item and drops from series. Raise KeyError if not found.

        Parameters
        ----------
        item : Hashable
            Index of the element that needs to be removed.

        Returns
        -------
        Any
            Value that is popped from series.

        See Also
        --------
        Series.drop: Drop specified values from Series.
        Series.drop_duplicates: Return Series with duplicate values removed.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3])

        >>> ser.pop(0)
        1

        >>> ser
        1    2
        2    3
        dtype: int64
        """
        return super().pop(item=item)

    def info(
        self,
        verbose: Optional[bool] = None,
        buf: Optional[IO[str]] = None,
        max_cols: Optional[int] = None,
        memory_usage: Optional[bool] = None,
        show_counts: bool = True
    ) -> None:
        """
        Provide a concise summary of the Series.

        Parameters
        ----------
        verbose : Optional[bool], default None
            Whether to show full summary information.
        buf : Optional[IO[str]], default None
            Buffer to write the output to.
        max_cols : Optional[int], default None
            Not used. Kept for compatibility.
        memory_usage : Optional[bool], default None
            Whether to include the memory usage of the Series.
        show_counts : bool, default True
            Whether to show the counts of unique values.

        Returns
        -------
        None
        """
        return SeriesInfo(self, memory_usage).render(buf=buf, max_cols=max_cols, verbose=verbose, show_counts=show_counts)

    def memory_usage(
        self,
        index: bool = True,
        deep: bool = False
    ) -> int:
        """
        Return the memory usage of the Series.

        The memory usage can optionally include the contribution of
        the index and of elements of `object` dtype.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the Series index.
        deep : bool, default False
            If True, introspect the data deeply by interrogating
            `object` dtypes for system-level memory consumption, and include
            it in the returned value.

        Returns
        -------
        int
            Bytes of memory consumed.

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of the
            array.
        DataFrame.memory_usage : Bytes consumed by a DataFrame.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.memory_usage()
        152

        Not including the index gives the size of the rest of the data, which
        is necessarily smaller:

        >>> s.memory_usage(index=False)
        24

        The memory footprint of `object` values is ignored by default:

        >>> s = pd.Series(["a", "b"])
        >>> s.values
        array(['a', 'b'], dtype=object)
        >>> s.memory_usage()
        144
        >>> s.memory_usage(deep=True)
        244
        """
        v = self._memory_usage(deep=deep)
        if index:
            v += self.index.memory_usage(deep=deep)
        return v

    def isin(self, values: Union[Iterable[Any], Mapping[Any, Any]]) -> Series:
        """
        Whether elements in Series are contained in `values`.

        Return a boolean Series showing whether each element in the Series
        matches an element in the passed sequence of `values` exactly.

        Parameters
        ----------
        values : Union[Iterable[Any], Mapping[Any, Any]]
            The sequence of values to test. Passing in a single string will
            raise a ``TypeError``. Instead, turn a single string into a
            list of one element.

        Returns
        -------
        Series[bool]
            Series of booleans indicating if each element is in values.

        Raises
        ------
        TypeError
            If `values` is a string

        See Also
        --------
        DataFrame.isin : Equivalent method on DataFrame.

        Examples
        --------
        >>> s = pd.Series(["llama", "cow", "llama", "beetle", "llama", "hippo"], name="animal")
        >>> s.isin(["cow", "llama"])
        0     True
        1     True
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        To invert the boolean values, use the ``~`` operator:

        >>> ~s.isin(["cow", "llama"])
        0    False
        1    False
        2    False
        3     True
        4    False
        5     True
        Name: animal, dtype: bool

        Passing a single string as ``s.isin('llama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(["llama"])
        0     True
        1    False
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        Strings and integers are distinct and are therefore not comparable:

        >>> pd.Series([1]).isin(["1"])
        0    False
        dtype: bool
        >>> pd.Series([1.1]).isin(["1.1"])
        0    False
        dtype: bool
        """
        result = algorithms.isin(self._values, values)
        return self._constructor(result, index=self.index, copy=False).__finalize__(self, method='isin')

    def between(
        self,
        left: Any,
        right: Any,
        inclusive: Literal['both', 'neither', 'left', 'right'] = 'both'
    ) -> Series:
        """
        Return boolean Series equivalent to left <= series <= right.

        This function returns a boolean vector containing `True` wherever the
        corresponding Series element is between the boundary values `left` and
        `right`. NA values are treated as `False`.

        Parameters
        ----------
        left : Any
            Left boundary.
        right : Any
            Right boundary.
        inclusive : Literal['both', 'neither', 'left', 'right'], default 'both'
            Include boundaries. Whether to set each bound as closed or open.

            .. versionchanged:: 1.3.0

        Returns
        -------
        Series[bool]
            Series representing whether each element is between left and
            right (inclusive).

        See Also
        --------
        Series.gt : Greater than of series and other.
        Series.lt : Less than of series and other.

        Notes
        -----
        This function is equivalent to ``(left <= ser) & (ser <= right)``

        Examples
        --------
        >>> s = pd.Series([2, 0, 4, 8, np.nan])

        Boundary values are included by default:

        >>> s.between(1, 4)
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        With `inclusive` set to ``"neither"`` boundary values are excluded:

        >>> s.between(1, 4, inclusive="neither")
        0     True
        1    False
        2    False
        3    False
        4    False
        dtype: bool

        `left` and `right` can be any scalar value:

        >>> s = pd.Series(["Alice", "Bob", "Carol", "Eve"])
        >>> s.between("Anna", "Daniel")
        0    False
        1     True
        2     True
        3    False
        dtype: bool
        """
        if inclusive == 'both':
            lmask = self >= left
            rmask = self <= right
        elif inclusive == 'left':
            lmask = self >= left
            rmask = self < right
        elif inclusive == 'right':
            lmask = self > left
            rmask = self <= right
        elif inclusive == 'neither':
            lmask = self > left
            rmask = self < right
        else:
            raise ValueError("Inclusive has to be either string of 'both','left', 'right', or 'neither'.")
        return lmask & rmask

    def case_when(
        self,
        caselist: list[tuple[Union[bool, Sequence[bool], Callable[[Any], bool]], Union[Any, Sequence[Any], Callable[[Any], Any]]]]
    ) -> Series:
        """
        Replace values where the conditions are True.

        .. versionadded:: 2.2.0

        Parameters
        ----------
        caselist : list of tuples
            A list of tuples of conditions and expected replacements
            Takes the form:  ``(condition0, replacement0)``,
            ``(condition1, replacement1)``, ... .
            ``condition`` should be a 1-D boolean array-like object
            or a callable. If ``condition`` is a callable,
            it is computed on the Series
            and should return a boolean Series or array.
            The callable must not change the input Series
            (though pandas doesn't check it). ``replacement`` should be a
            1-D array-like object, a scalar or a callable.
            If ``replacement`` is a callable, it is computed on the Series
            and should return a scalar or Series. The callable
            must not change the input Series
            (though pandas doesn't check it).

        Returns
        -------
        Series
            A new Series with values replaced based on the provided conditions.

        See Also
        --------
        Series.mask : Replace values where the condition is True.

        Examples
        --------
        >>> c = pd.Series([6, 7, 8, 9], name="c")
        >>> a = pd.Series([0, 0, 1, 2])
        >>> b = pd.Series([0, 3, 4, 5])

        >>> c.case_when(
        ...     caselist=[
        ...         (a.gt(0), a),  # condition, replacement
        ...         (b.gt(0), b),
        ...     ]
        ... )
        0    6.0
        1    3.0
        2    1.0
        3    2.0
        Name: c, dtype: float64
        """
        if not isinstance(caselist, list):
            raise TypeError(f'The caselist argument should be a list; instead got {type(caselist)}')
        if not caselist:
            raise ValueError('provide at least one boolean condition, with a corresponding replacement.')
        for num, entry in enumerate(caselist):
            if not isinstance(entry, tuple):
                raise TypeError(f'Argument {num} must be a tuple; instead got {type(entry)}.')
            if len(entry) != 2:
                raise ValueError(f'Argument {num} must have length 2; a condition and replacement; instead got length {len(entry)}.')
        caselist = [
            (com.apply_if_callable(condition, self), com.apply_if_callable(replacement, self))
            for condition, replacement in caselist
        ]
        default = self.copy(deep=False)
        conditions, replacements = zip(*caselist)
        common_dtypes = [infer_dtype_from(arg)[0] for arg in [*replacements, default]]
        if len(set(common_dtypes)) > 1:
            common_dtype = find_common_type(common_dtypes)
            updated_replacements = []
            for condition, replacement in zip(conditions, replacements):
                if is_scalar(replacement):
                    replacement = construct_1d_arraylike_from_scalar(
                        value=replacement,
                        length=len(condition),
                        dtype=common_dtype
                    )
                elif isinstance(replacement, ABCSeries):
                    replacement = replacement.astype(common_dtype)
                else:
                    replacement = pd_array(replacement, dtype=common_dtype)
                updated_replacements.append(replacement)
            replacements = updated_replacements
            default = default.astype(common_dtype)
        counter = range(len(conditions) - 1, -1, -1)
        for position, condition, replacement in zip(counter, reversed(conditions), reversed(replacements)):
            try:
                default = default.mask(condition, other=replacement, axis=0, inplace=False, level=None)
            except Exception as error:
                raise ValueError(f'Failed to apply condition{position} and replacement{position}.') from error
        return default

    def isna(self) -> Series:
        """
        Return a boolean same-sized object indicating if values are NA.

        NA values, such as None or numpy.NaN, get mapped to True. Everything else gets mapped to False.

        Returns
        -------
        Series[bool]
            Boolean Series indicating if each value is NA.
        """
        return NDFrame.isna(self)

    def isnull(self) -> Series:
        """
        Series.isnull is an alias for Series.isna.
        """
        return super().isnull()

    def notna(self) -> Series:
        """
        Return a boolean same-sized object indicating if values are not NA.

        Non-NA values get mapped to True. NA values, such as None or numpy.NaN,
        get mapped to False.

        Returns
        -------
        Series[bool]
            Boolean Series indicating if each value is not NA.
        """
        return super().notna()

    def notnull(self) -> Series:
        """
        Series.notnull is an alias for Series.notna.
        """
        return super().notnull()

    @overload
    def dropna(
        self,
        *,
        axis: Union[int, Literal[0]] = ...,
        inplace: bool = ...,
        how: Optional[Any] = ...,
        ignore_index: bool = ...
    ) -> Optional[Series]:
        ...

    @overload
    def dropna(
        self,
        *,
        axis: Union[int, Literal[0]],
        inplace: bool,
        how: Optional[Any] = ...,
        ignore_index: bool = ...
    ) -> Optional[Series]:
        ...

    def dropna(
        self,
        *,
        axis: Union[int, Literal[0]] = 0,
        inplace: bool = False,
        how: Optional[Any] = None,
        ignore_index: bool = False
    ) -> Optional[Series]:
        """
        Return a new Series with missing values removed.

        See the :ref:`User Guide <missing_data>` for more on which values are
        considered missing, and how to work with missing data.

        Parameters
        ----------
        axis : Union[int, Literal[0]], default 0
            {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        inplace : bool, default False
            If True, do operation inplace and return None.
        how : Optional[Any], default None
            Not in use. Kept for compatibility.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        Optional[Series]
            Series with NA entries dropped from it or None if ``inplace=True``.

        See Also
        --------
        Series.isna: Indicate missing values.
        Series.notna : Indicate existing (non-missing) values.
        Series.fillna : Replace missing values.
        DataFrame.dropna : Drop rows or columns which contain NA values.
        Index.dropna : Drop missing indices.

        Examples
        --------
        >>> ser = pd.Series([1.0, 2.0, np.nan])
        >>> ser
        0    1.0
        1    2.0
        2    NaN
        dtype: float64

        Drop NA values from a Series.

        >>> ser.dropna()
        0    1.0
        1    2.0
        dtype: float64

        Empty strings are not considered NA values. ``None`` is considered an
        NA value.

        >>> ser = pd.Series([np.nan, 2, pd.NaT, "", None, "I stay"])
        >>> ser
        0       NaN
        1         2
        2       NaT
        3
        4      None
        5    I stay
        dtype: object
        >>> ser.dropna()
        1         2
        3
        5    I stay
        dtype: object
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        ignore_index = validate_bool_kwarg(ignore_index, 'ignore_index')
        self._get_axis_number(axis or 0)
        if self._can_hold_na:
            result = remove_na_arraylike(self)
        elif not inplace:
            result = self.copy(deep=False)
        else:
            result = self
        if ignore_index:
            result.index = default_index(len(result))
        if inplace:
            return self._update_inplace(result)
        else:
            return result

    def to_timestamp(
        self,
        freq: Optional[Union[str, pd.offsets.DateOffset]] = None,
        how: Literal['start', 'end', 'e'] = 'start',
        copy: Optional[bool] = None
    ) -> Series:
        """
        Cast to DatetimeIndex of Timestamps, at *beginning* of period.

        This can be changed to the *end* of the period, by specifying `how="e"`.

        Parameters
        ----------
        freq : Optional[Union[str, pd.offsets.DateOffset]], default None
            Desired frequency.
        how : Literal['start', 'end', 'e'], default 'start'
            Convention for converting period to timestamp; start of period
            vs. end.
        copy : Optional[bool], default None
            Whether or not to return a copy.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

            .. deprecated:: 3.0.0

        Returns
        -------
        Series
            Series with index converted to PeriodIndex.

        See Also
        --------
        Series.to_period: Inverse method to cast DatetimeIndex to PeriodIndex.
        DataFrame.to_timestamp: Equivalent method for DataFrame.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> s1 = pd.Series([1, 2, 3], index=idx)
        >>> s1
        2023    1
        2024    2
        2025    3
        Freq: Y-DEC, dtype: int64

        The resulting frequency of the Timestamps is `YearBegin`

        >>> s1 = s1.to_timestamp()
        >>> s1
        2023-01-01    1
        2024-01-01    2
        2025-01-01    3
        Freq: YS-JAN, dtype: int64

        Using `freq` which is the offset that the Timestamps will have

        >>> s2 = pd.Series([1, 2, 3], index=idx)
        >>> s2 = s2.to_timestamp(freq="M")
        >>> s2
        2023-01-31    1
        2024-01-31    2
        2025-01-31    3
        Freq: YE-JAN, dtype: int64
        """
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, PeriodIndex):
            raise TypeError(f'unsupported Type {type(self.index).__name__}')
        new_obj = self.copy(deep=False)
        new_index = self.index.to_timestamp(freq=freq, how=how)
        setattr(new_obj, 'index', new_index)
        return new_obj

    def to_period(
        self,
        freq: Optional[Union[str, pd.offsets.DateOffset]] = None,
        copy: Optional[bool] = None
    ) -> Series:
        """
        Convert Series from DatetimeIndex to PeriodIndex.

        Parameters
        ----------
        freq : Optional[Union[str, pd.offsets.DateOffset]], default None
            Frequency associated with the PeriodIndex.
        copy : Optional[bool], default False
            Whether or not to return a copy.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``

            .. deprecated:: 3.0.0

        Returns
        -------
        Series
            Series with index converted to PeriodIndex.

        See Also
        --------
        DataFrame.to_period: Equivalent method for DataFrame.
        Series.dt.to_period: Convert DateTime column values.

        Examples
        --------
        >>> idx = pd.DatetimeIndex(["2023", "2024", "2025"])
        >>> s = pd.Series([1, 2, 3], index=idx)
        >>> s.to_period()
        2023    1
        2024    2
        2025    3
        Freq: Y-DEC, dtype: int64

        Viewing the index

        >>> s.index
        PeriodIndex(['2023', '2024', '2025'], dtype='period[Y-DEC]')
        """
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError(f'unsupported Type {type(self.index).__name__}')
        new_obj = self.copy(deep=False)
        new_index = self.index.to_period(freq=freq)
        setattr(new_obj, 'index', new_index)
        return new_obj

    _AXIS_ORDERS: list[str] = ['index']
    _AXIS_LEN: int = len(_AXIS_ORDERS)
    _info_axis_number: int = 0
    _info_axis_name: str = 'index'
    index: Index = properties.AxisProperty(
        axis=0,
        doc="""
        The index (axis labels) of the Series.

        The index of a Series is used to label and identify each element of the
        underlying data. The index can be thought of as an immutable ordered set
        (technically a multi-set, as it may contain duplicate labels), and is
        used to index and align data in pandas.

        Returns
        -------
        Index
            The index labels of the Series.

        See Also
        --------
        Series.reindex : Conform Series to new index.
        Index : The base pandas index type.

        Notes
        -----
        For more information on pandas indexing, see the `indexing user guide
        <https://pandas.pydata.org/docs/user_guide/indexing.html>`__.

        Examples
        --------
        To create a Series with a custom index and view the index labels:

        >>> cities = ['Kolkata', 'Chicago', 'Toronto', 'Lisbon']
        >>> populations = [14.85, 2.71, 2.93, 0.51]
        >>> city_series = pd.Series(populations, index=cities)
        >>> city_series.index
        Index(['Kolkata', 'Chicago', 'Toronto', 'Lisbon'], dtype='object')

        To change the index labels of an existing Series:

        >>> city_series.index = ['KOL', 'CHI', 'TOR', 'LIS']
        >>> city_series.index
        Index(['KOL', 'CHI', 'TOR', 'LIS'], dtype='object')
        """
    )
    str = Accessor('str', StringMethods)
    dt = Accessor('dt', CombinedDatetimelikeProperties)
    cat = Accessor('cat', CategoricalAccessor)
    plot = Accessor('plot', pandas.plotting.PlotAccessor)
    sparse = Accessor('sparse', SparseAccessor)
    struct = Accessor('struct', StructAccessor)
    list = Accessor('list', ListAccessor)
    hist = pandas.plotting.hist_series

    def _cmp_method(
        self,
        other: Any,
        op: Callable[[Any, Any], Any]
    ) -> Series:
        res_name = ops.get_op_result_name(self, other)
        if isinstance(other, Series) and (not self._indexed_same(other)):
            raise ValueError('Can only compare identically-labeled Series objects')
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        res_values = ops.comparison_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    def _logical_method(
        self,
        other: Any,
        op: Callable[[Any, Any], Any]
    ) -> Series:
        res_name = ops.get_op_result_name(self, other)
        self_aligned, other_aligned = self._align_for_op(other, align_asobject=True)
        lvalues = self_aligned._values
        rvalues = extract_array(other_aligned, extract_numpy=True, extract_range=True)
        res_values = ops.logical_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    def _arith_method(
        self,
        other: Any,
        op: Callable[[Any, Any], Any]
    ) -> Series:
        self_aligned, other_aligned = self._align_for_op(other)
        return base.IndexOpsMixin._arith_method(self_aligned, other_aligned, op)

    def _align_for_op(
        self,
        right: Any,
        align_asobject: bool = False
    ) -> tuple[Series, Any]:
        """align lhs and rhs Series"""
        left = self
        if isinstance(right, Series):
            if not left.index.equals(right.index):
                if align_asobject:
                    if left.dtype not in (object, np.bool_) or right.dtype not in (object, np.bool_):
                        pass
                    else:
                        left = left.astype(object)
                        right = right.astype(object)
                left, right = left.align(right)
        return (left, right)

    def _binop(
        self,
        other: Any,
        func: Callable[[Any, Any], Any],
        level: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        fill_value: Optional[Any] = None
    ) -> Series:
        """
        Perform generic binary operation with optional fill value.

        Parameters
        ----------
        other : Any
            The other object to perform the operation with.
        func : Callable[[Any, Any], Any]
            The binary function to use.
        level : Optional[Union[int, str, Sequence[Union[int, str]]]], default None
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : Optional[Any], default None
            The value to use for missing data.

        Returns
        -------
        Series
            The result of the operation.
        """
        if isinstance(other, Series):
            this, other_aligned = self.align(other, join='inner')
        else:
            this = self
            other_aligned = other
        return super()._binop(other_aligned, func, level=level, fill_value=fill_value)

    def _construct_result(
        self,
        result: Union[np.ndarray, ExtensionArray, tuple],
        name: Any
    ) -> Union[Series, tuple[Series, Series]]:
        """
        Construct an appropriately-labelled Series from the result of an op.

        Parameters
        ----------
        result : Union[np.ndarray, ExtensionArray, tuple]
            The result of the operation.
        name : Any
            The name of the resulting Series.

        Returns
        -------
        Union[Series, tuple[Series, Series]]
            The constructed Series or tuple of Series.
        """
        if isinstance(result, tuple):
            res1 = self._construct_result(result[0], name=name)
            res2 = self._construct_result(result[1], name=name)
            assert isinstance(res1, Series)
            assert isinstance(res2, Series)
            return (res1, res2)
        dtype = getattr(result, 'dtype', None)
        out = self._constructor(result, index=self.index, dtype=dtype, copy=False)
        out = out.__finalize__(self)
        out.name = name
        return out

    def _flex_method(
        self,
        other: Any,
        op: Callable[[Any, Any], Any],
        *,
        level: Optional[Union[int, str]] = None,
        fill_value: Optional[Any] = None,
        axis: int = 0
    ) -> Union[Any, Series]:
        if axis is not None:
            self._get_axis_number(axis)
        res_name = ops.get_op_result_name(self, other)
        if isinstance(other, Series):
            return self._binop(other, op, level=level, fill_value=fill_value)
        elif isinstance(other, (np.ndarray, list, tuple)):
            if len(other) != len(self):
                raise ValueError('Lengths must be equal')
            other_series = self._constructor(other, self.index, copy=False)
            result = self._binop(other_series, op, level=level, fill_value=fill_value)
            result._name = res_name
            return result
        else:
            if fill_value is not None:
                if isna(other):
                    return op(self, fill_value)
                this_filled = self.fillna(fill_value)
                return op(this_filled, other)
            else:
                return op(self, other)

    def _reduce(
        self,
        op: Callable[..., Any],
        name: str,
        *,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        filter_type: Optional[str] = None,
        **kwds: Any
    ) -> Any:
        """
        Perform a reduction operation.

        If we have an ndarray as a value, then simply perform the operation,
        otherwise delegate to the object.

        Parameters
        ----------
        op : Callable[..., Any]
            The reduction operation to perform.
        name : str
            The name of the reduction operation.
        axis : int, default 0
            Axis for the function to be applied on.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        filter_type : Optional[str], default None
            The type of filtering to perform.

        Returns
        -------
        Any
            The result of the reduction.
        """
        delegate = self._values
        if axis is not None:
            self._get_axis_number(axis)
        if isinstance(delegate, ExtensionArray):
            return delegate._reduce(name, skipna=skipna, **kwds)
        else:
            if numeric_only and self.dtype.kind not in 'iufcb':
                kwd_name = 'numeric_only'
                if name in ['any', 'all']:
                    kwd_name = 'bool_only'
                raise TypeError(f'Series.{name} does not allow {kwd_name}={numeric_only} with non-numeric dtypes.')
            return op(delegate, skipna=skipna, **kwds)

    @Appender(make_doc('any', ndim=1))
    def any(
        self,
        *,
        axis: int = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any
    ) -> bool:
        """
        Return True if any element is True over the requested axis.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.
        bool_only : bool, default False
            Include only boolean columns.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs : Any
            Additional arguments to be passed to the function.

        Returns
        -------
        bool
            Whether any element is True.

        See Also
        --------
        Series.all : Return False only if all elements are False.

        Examples
        --------
        >>> s = pd.Series([True, False, True])
        >>> s.any()
        True
        """
        nv.validate_logical_func((), kwargs, fname='any')
        skipna = validate_bool_kwarg(skipna, 'skipna', none_allowed=False)
        return self._reduce(
            nanops.nanany,
            name='any',
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type='bool'
        )

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='all')
    @Appender(make_doc('all', ndim=1))
    def all(
        self,
        *,
        axis: int = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any
    ) -> bool:
        """
        Return True if all elements are True over the requested axis.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.
        bool_only : bool, default False
            Include only boolean columns.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs : Any
            Additional arguments to be passed to the function.

        Returns
        -------
        bool
            Whether all elements are True.

        See Also
        --------
        Series.any : Return True if any element is True.

        Examples
        --------
        >>> s = pd.Series([True, False, True])
        >>> s.all()
        False
        """
        nv.validate_logical_func((), kwargs, fname='all')
        skipna = validate_bool_kwarg(skipna, 'skipna', none_allowed=False)
        return self._reduce(
            nanops.nanall,
            name='all',
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type='bool'
        )

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='min')
    def min(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return the minimum of the values over the requested axis.

        If you want the *index* of the minimum, use ``idxmin``.
        This is the equivalent of the ``numpy.ndarray`` method ``argmin``.


        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. warning::

                The behavior of DataFrame.min with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            The minimum of the values in the Series.

        See Also
        --------
        numpy.min : Equivalent numpy function for arrays.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.max : Maximum value.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays(
        ...     [["warm", "warm", "cold", "cold"], ["dog", "falcon", "fish", "spider"]],
        ...     names=["blooded", "animal"],
        ... )
        >>> s = pd.Series([4, 2, 0, 8], name="legs", index=idx)
        >>> s
        blooded  animal
        warm     dog       4
                falcon    2
        cold     fish      0
                spider    8
        Name: legs, dtype: float64

        >>> s.min()
        0.0
        """
        return NDFrame.min(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='max')
    def max(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return the maximum of the values over the requested axis.

        If you want the *index* of the maximum, use ``idxmax``.
        This is the equivalent of the ``numpy.ndarray`` method ``argmax``.


        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. warning::

                The behavior of DataFrame.max with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            The maximum of the values in the Series.

        See Also
        --------
        numpy.max : Equivalent numpy function for arrays.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.min : Minimum value.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays(
        ...     [["warm", "warm", "cold", "cold"], ["dog", "falcon", "fish", "spider"]],
        ...     names=["blooded", "animal"],
        ... )
        >>> s = pd.Series([4, 2, 0, 8], name="legs", index=idx)
        >>> s
        blooded  animal
        warm     dog       4
                falcon    2
        cold     fish      0
                spider    8
        Name: legs, dtype: float64

        >>> s.max()
        8.0
        """
        return NDFrame.max(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='sum')
    def sum(
        self,
        axis: Optional[Union[int, Literal[0]]] = None,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs: Any
    ) -> Any:
        """
        Return the sum of the values over the requested axis.

        This is equivalent to the method ``numpy.sum``.

        Parameters
        ----------
        axis : Optional[Union[int, Literal[0]]], default None
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

            .. warning::

                The behavior of DataFrame.sum with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer than
            ``min_count`` non-NA values are present the result will be NA.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            Sum of the values for the requested axis.

        See Also
        --------
        numpy.sum : Equivalent numpy function for computing sum.
        Series.mean : Mean of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> idx = pd.MultiIndex.from_arrays(
        ...     [["warm", "warm", "cold", "cold"], ["dog", "falcon", "fish", "spider"]],
        ...     names=["blooded", "animal"],
        ... )
        >>> s = pd.Series([4, 2, 0, 8], name="legs", index=idx)
        >>> s
        blooded  animal
        warm     dog       4
                falcon    2
        cold     fish      0
                spider    8
        Name: legs, dtype: float64

        >>> s.sum()
        14.0

        Alternatively, ``ddof=0`` can be set to normalize by N instead of N-1:
    
        >>> s.var(ddof=0)
        41.25
        """
        return NDFrame.sum(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='prod')
    @doc(make_doc('prod', ndim=1))
    def prod(
        self,
        axis: Optional[Union[int, Literal[0]]] = None,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs: Any
    ) -> Any:
        return NDFrame.prod(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='mean')
    def mean(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return the mean of the values over the requested axis.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            Mean of the values for the requested axis.

        See Also
        --------
        numpy.mean : Equivalent numpy function for arrays.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.mean()
        2.0
        """
        return NDFrame.mean(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='median')
    def median(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return the median of the values over the requested axis.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            Median of the values for the requested axis.

        See Also
        --------
        numpy.median : Equivalent numpy function for arrays.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.median()
        2.0

        With a DataFrame

        >>> df = pd.DataFrame({"a": [1, 2], "b": [2, 3]}, index=["tiger", "zebra"])
        >>> df
           a  b
        tiger  1  2
        zebra  2  3
        >>> df.median()
        a    1.5
        b    2.5
        dtype: float64

        Alternatively, calculate the median by rows using axis=1

        >>> df.median(axis=1)
        tiger    1.5
        zebra    2.5
        dtype: float64

        Using axis=1 can be useful when you want the median across columns.

        >>> df = pd.DataFrame({"a": [1, 2], "b": ["T", "Z"]}, index=["tiger", "zebra"])
        >>> df.median(numeric_only=True)
        a    1.5
        dtype: float64
        """
        return NDFrame.median(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='sem')
    def sem(
        self,
        axis: Optional[Union[int, Literal[0]]] = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return the standard error of the mean over the requested axis.

        Parameters
        ----------
        axis : Optional[Union[int, Literal[0]]], default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        ddof : int, default 1
            Degrees of freedom.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            The standard error of the mean for the requested axis.

        See Also
        --------
        numpy.std : Equivalent numpy function for arrays.
        Series.std : Return the standard deviation.
        DataFrame.std : Return the standard deviation over the requested axis.
        DataFrame.sem : Return the standard error of the mean over the requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.sem()
        0.5773502691896257
        """
        return NDFrame.sem(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    def var(
        self,
        axis: Optional[Union[int, Literal[0]]] = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return unbiased variance over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : Optional[Union[int, Literal[0]]], default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

            .. warning::

                The behavior of DataFrame.var with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            Unbiased variance over requested axis.

        See Also
        --------
        numpy.var : Equivalent function in NumPy.
        Series.std : Return unbiased standard deviation over requested axis.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "person_id": [0, 1, 2, 3],
        ...         "age": [21, 25, 62, 43],
        ...         "height": [1.61, 1.87, 1.49, 2.01],
        ...     }
        ... ).set_index("person_id")
        >>> df
                   age  height
        person_id
        0           21    1.61
        1           25    1.87
        2           62    1.49
        3           43    2.01

        >>> df.var()
        age       352.916667
        height      0.056367
        dtype: float64

        Alternatively, ``ddof=0`` can be set to normalize by N instead of N-1:

        >>> df.var(ddof=0)
        age       264.687500
        height      0.042275
        dtype: float64
        """
        return NDFrame.var(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='std')
    def std(
        self,
        axis: Optional[Union[int, Literal[0]]] = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return the standard deviation of the values over the requested axis.

        Parameters
        ----------
        axis : Optional[Union[int, Literal[0]]], default None
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

            .. warning::

                The behavior of DataFrame.std with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        ddof : int, default 1
            Degrees of freedom.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            The standard deviation for the requested axis.

        See Also
        --------
        numpy.std : Equivalent numpy function for arrays.
        Series.var : Return the variance.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.var : Return the variance over the requested axis.
        DataFrame.std : Return the standard deviation over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "person_id": [0, 1, 2, 3],
        ...         "age": [21, 25, 62, 43],
        ...         "height": [1.61, 1.87, 1.49, 2.01],
        ...     }
        ... ).set_index("person_id")
        >>> df
                   age  height
        person_id
        0           21    1.61
        1           25    1.87
        2           62    1.49
        3           43    2.01

        >>> df.std()
        age       18.204155
        height     0.245519
        dtype: float64

        Alternatively, ``ddof=0`` can be set to normalize by N instead of N-1:

        >>> df.std(ddof=0)
        age       16.436296
        height     0.205851
        dtype: float64
        """
        return NDFrame.std(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    def skew(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return the skewness of the values over the requested axis.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            Skewness of the values for the requested axis.

        See Also
        --------
        numpy.skew : Equivalent numpy function for arrays.
        Series.kurt : Return unbiased kurtosis over requested axis.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.index.min : Return the index of the minimum over the requested axis.
        DataFrame.index.max : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        >>> s.skew()
        0.0
        >>> s = pd.Series([1, 2, 3, 10], index=["a", "b", "c", "d"])
        >>> s.skew()
        1.1461932206205813
        """
        return NDFrame.skew(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def kurt(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return unbiased kurtosis over requested axis.

        Normalized by N-1 by default, using Fisher's definition of kurtosis (kurtosis of normal == 0.0).

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            Kurtosis of the values for the requested axis.

        See Also
        --------
        numpy.kurtosis : Equivalent numpy function for arrays.
        Series.std : Return unbiased standard deviation over requested axis.
        Series.var : Return unbiased variance over requested axis.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.std : Return the standard deviation over the requested axis.
        DataFrame.var : Return the variance over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        >>> s.kurt()
        -1.2
        >>> s = pd.Series([1, 2, 3, 10], index=["a", "b", "c", "d"])
        >>> s.kurt()
        1.7
        """
        return NDFrame.kurt(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='kurt')
    @doc(make_doc('kurt', ndim=1))
    def kurtosis(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return unbiased kurtosis over requested axis.

        Normalized by N-1 by default.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            Kurtosis of the values for the requested axis.

        See Also
        --------
        numpy.kurtosis : Equivalent numpy function for arrays.
        Series.std : Return unbiased standard deviation over requested axis.
        Series.var : Return unbiased variance over requested axis.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.std : Return the standard deviation over the requested axis.
        DataFrame.var : Return the variance over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.kurtosis()
        -1.2
        >>> s = pd.Series([1, 2, 3, 10])
        >>> s.kurtosis()
        1.7
        """
        return self.kurt(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def cummin(
        self,
        axis: int = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any
    ) -> Series:
        return NDFrame.cummin(self, axis, skipna, *args, **kwargs)

    def cummax(
        self,
        axis: int = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any
    ) -> Series:
        return NDFrame.cummax(self, axis, skipna, *args, **kwargs)

    def cumsum(
        self,
        axis: int = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any
    ) -> Series:
        return NDFrame.cumsum(self, axis, skipna, *args, **kwargs)

    def cumprod(
        self,
        axis: int = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any
    ) -> Series:
        return NDFrame.cumprod(self, axis, skipna, *args, **kwargs)

    def _flex_method(
        self,
        other: Any,
        op: Callable[[Any, Any], Any],
        *,
        level: Optional[Union[int, str]] = None,
        fill_value: Optional[Any] = None,
        axis: int = 0
    ) -> Union[Any, Series]:
        if axis is not None:
            self._get_axis_number(axis)
        res_name = ops.get_op_result_name(self, other)
        if isinstance(other, Series):
            return self._binop(other, op, level=level, fill_value=fill_value)
        elif isinstance(other, (np.ndarray, list, tuple)):
            if len(other) != len(self):
                raise ValueError('Lengths must be equal')
            other_series = self._constructor(other, self.index, copy=False)
            result = self._binop(other_series, op, level=level, fill_value=fill_value)
            result._name = res_name
            return result
        else:
            if fill_value is not None:
                if isna(other):
                    return op(self, fill_value)
                this_filled = self.fillna(fill_value)
                return op(this_filled, other)
            else:
                return op(self, other)

    def _reduce(
        self,
        op: Callable[..., Any],
        name: str,
        *,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        filter_type: Optional[str] = None,
        **kwds: Any
    ) -> Any:
        """
        Perform a reduction operation.

        If we have an ndarray as a value, then simply perform the operation,
        otherwise delegate to the object.

        Parameters
        ----------
        op : Callable[..., Any]
            The reduction operation to perform.
        name : str
            The name of the reduction operation.
        axis : int, default 0
            Axis for the function to be applied on.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        filter_type : Optional[str], default None
            The type of filtering to perform.

        Returns
        -------
        Any
            The result of the reduction.
        """
        delegate = self._values
        if axis is not None:
            self._get_axis_number(axis)
        if isinstance(delegate, ExtensionArray):
            return delegate._reduce(name, skipna=skipna, **kwds)
        else:
            if numeric_only and self.dtype.kind not in 'iufcb':
                kwd_name = 'numeric_only'
                if name in ['any', 'all']:
                    kwd_name = 'bool_only'
                raise TypeError(f'Series.{name} does not allow {kwd_name}={numeric_only} with non-numeric dtypes.')
            return op(delegate, skipna=skipna, **kwds)

    @Appender(make_doc('any', ndim=1))
    def any(
        self,
        *,
        axis: int = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any
    ) -> bool:
        """
        Return True if any element is True over the requested axis.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.
        bool_only : bool, default False
            Include only boolean columns.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs : Any
            Additional arguments to be passed to the function.

        Returns
        -------
        bool
            Whether any element is True.

        See Also
        --------
        Series.all : Return False only if all elements are False.

        Examples
        --------
        >>> s = pd.Series([True, False, True])
        >>> s.any()
        True
        """
        nv.validate_logical_func((), kwargs, fname='any')
        skipna = validate_bool_kwarg(skipna, 'skipna', none_allowed=False)
        return self._reduce(
            nanops.nanany,
            name='any',
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type='bool'
        )

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='all')
    @Appender(make_doc('all', ndim=1))
    def all(
        self,
        *,
        axis: int = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any
    ) -> bool:
        """
        Return True if all elements are True over the requested axis.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.
        bool_only : bool, default False
            Include only boolean columns.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs : Any
            Additional arguments to be passed to the function.

        Returns
        -------
        bool
            Whether all elements are True.

        See Also
        --------
        Series.any : Return True if any element is True.

        Examples
        --------
        >>> s = pd.Series([True, False, True])
        >>> s.all()
        False
        """
        nv.validate_logical_func((), kwargs, fname='all')
        skipna = validate_bool_kwarg(skipna, 'skipna', none_allowed=False)
        return self._reduce(
            nanops.nanall,
            name='all',
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type='bool'
        )

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='min')
    def min(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return the minimum of the values over the requested axis.

        If you want the *index* of the minimum, use ``idxmin``.
        This is the equivalent of the ``numpy.ndarray`` method ``argmin``.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

            .. warning::

                The behavior of DataFrame.min with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            The minimum of the values in the Series.

        See Also
        --------
        numpy.min : Equivalent numpy function for arrays.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.max : Maximum value.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.min()
        1.0
        """
        return NDFrame.min(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='max')
    def max(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return the maximum of the values over the requested axis.

        If you want the *index* of the maximum, use ``idxmax``.
        This is the equivalent of the ``numpy.ndarray`` method ``argmax``.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

            .. warning::

                The behavior of DataFrame.max with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            The maximum of the values in the Series.

        See Also
        --------
        numpy.max : Equivalent numpy function for arrays.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.min : Minimum value.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.max()
        3.0
        """
        return NDFrame.max(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def _binop(
        self,
        other: Any,
        func: Callable[[Any, Any], Any],
        level: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        fill_value: Optional[Any] = None
    ) -> Series:
        """
        Perform generic binary operation with optional fill value.

        Parameters
        ----------
        other : Any
            The other object to perform the operation with.
        func : Callable[[Any, Any], Any]
            The binary function to use.
        level : Optional[Union[int, str, Sequence[Union[int, str]]]], default None
            Broadcast across a level, matching Index values on the
            passed MultiIndex level.
        fill_value : Optional[Any], default None
            The value to use for missing data.

        Returns
        -------
        Series
            The result of the operation.
        """
        if isinstance(other, Series):
            this, other_aligned = self.align(other, join='inner')
        else:
            this = self
            other_aligned = other
        return super()._binop(other_aligned, func, level=level, fill_value=fill_value)

    def _construct_result(
        self,
        result: Union[np.ndarray, ExtensionArray, tuple],
        name: Any
    ) -> Union[Series, tuple[Series, Series]]:
        """
        Construct an appropriately-labelled Series from the result of an op.

        Parameters
        ----------
        result : Union[np.ndarray, ExtensionArray, tuple]
            The result of the operation.
        name : Any
            The name of the resulting Series.

        Returns
        -------
        Union[Series, tuple[Series, Series]]
            The constructed Series or tuple of Series.
        """
        if isinstance(result, tuple):
            res1 = self._construct_result(result[0], name=name)
            res2 = self._construct_result(result[1], name=name)
            assert isinstance(res1, Series)
            assert isinstance(res2, Series)
            return (res1, res2)
        dtype = getattr(result, 'dtype', None)
        out = self._constructor(result, index=self.index, dtype=dtype, copy=False)
        out = out.__finalize__(self)
        out.name = name
        return out

    def _flex_method(
        self,
        other: Any,
        op: Callable[[Any, Any], Any],
        *,
        level: Optional[Union[int, str]] = None,
        fill_value: Optional[Any] = None,
        axis: int = 0
    ) -> Union[Any, Series]:
        if axis is not None:
            self._get_axis_number(axis)
        res_name = ops.get_op_result_name(self, other)
        if isinstance(other, Series):
            return self._binop(other, op, level=level, fill_value=fill_value)
        elif isinstance(other, (np.ndarray, list, tuple)):
            if len(other) != len(self):
                raise ValueError('Lengths must be equal')
            other_series = self._constructor(other, self.index, copy=False)
            result = self._binop(other_series, op, level=level, fill_value=fill_value)
            result._name = res_name
            return result
        else:
            if fill_value is not None:
                if isna(other):
                    return op(self, fill_value)
                this_filled = self.fillna(fill_value)
                return op(this_filled, other)
            else:
                return op(self, other)

    def _reduce(
        self,
        op: Callable[..., Any],
        name: str,
        *,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        filter_type: Optional[str] = None,
        **kwds: Any
    ) -> Any:
        """
        Perform a reduction operation.

        If we have an ndarray as a value, then simply perform the operation,
        otherwise delegate to the object.

        Parameters
        ----------
        op : Callable[..., Any]
            The reduction operation to perform.
        name : str
            The name of the reduction operation.
        axis : int, default 0
            Axis for the function to be applied on.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        filter_type : Optional[str]
            The type of filtering to perform.
        **kwds : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            The result of the reduction.
        """
        delegate = self._values
        if axis is not None:
            self._get_axis_number(axis)
        if isinstance(delegate, ExtensionArray):
            return delegate._reduce(name, skipna=skipna, **kwds)
        else:
            if numeric_only and self.dtype.kind not in 'iufcb':
                kwd_name = 'numeric_only'
                if name in ['any', 'all']:
                    kwd_name = 'bool_only'
                raise TypeError(f'Series.{name} does not allow {kwd_name}={numeric_only} with non-numeric dtypes.')
            return op(delegate, skipna=skipna, **kwds)

    @Appender(make_doc('any', ndim=1))
    def any(
        self,
        *,
        axis: int = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any
    ) -> bool:
        """
        Return True if any element is True over the requested axis.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.
        bool_only : bool, default False
            Include only boolean columns.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs : Any
            Additional arguments to be passed to the function.

        Returns
        -------
        bool
            Whether any element is True.

        See Also
        --------
        Series.all : Return False only if all elements are False.

        Examples
        --------
        >>> s = pd.Series([True, False, True])
        >>> s.any()
        True
        """
        nv.validate_logical_func((), kwargs, fname='any')
        skipna = validate_bool_kwarg(skipna, 'skipna', none_allowed=False)
        return self._reduce(
            nanops.nanany,
            name='any',
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type='bool'
        )

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='all')
    @Appender(make_doc('all', ndim=1))
    def all(
        self,
        *,
        axis: int = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any
    ) -> bool:
        """
        Return True if all elements are True over the requested axis.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.
        bool_only : bool, default False
            Include only boolean columns.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        **kwargs : Any
            Additional arguments to be passed to the function.

        Returns
        -------
        bool
            Whether all elements are True.

        See Also
        --------
        Series.any : Return True if any element is True.

        Examples
        --------
        >>> s = pd.Series([True, False, True])
        >>> s.all()
        False
        """
        nv.validate_logical_func((), kwargs, fname='all')
        skipna = validate_bool_kwarg(skipna, 'skipna', none_allowed=False)
        return self._reduce(
            nanops.nanall,
            name='all',
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type='bool'
        )

    def var(
        self,
        axis: Optional[Union[int, Literal[0]]] = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return unbiased variance over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : Optional[Union[int, Literal[0]]], default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

            .. warning::

                The behavior of DataFrame.var with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            Unbiased variance over requested axis.

        See Also
        --------
        numpy.var : Equivalent function in NumPy.
        Series.std : Return unbiased standard deviation over requested axis.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "person_id": [0, 1, 2, 3],
        ...         "age": [21, 25, 62, 43],
        ...         "height": [1.61, 1.87, 1.49, 2.01],
        ...     }
        ... ).set_index("person_id")
        >>> df
                   age  height
        person_id
        0           21    1.61
        1           25    1.87
        2           62    1.49
        3           43    2.01

        >>> df.var()
        age       352.916667
        height      0.056367
        dtype: float64

        Alternatively, ``ddof=0`` can be set to normalize by N instead of N-1:

        >>> df.var(ddof=0)
        age       264.687500
        height      0.042275
        dtype: float64
        """
        return NDFrame.var(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    def std(
        self,
        axis: Optional[Union[int, Literal[0]]] = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return the standard deviation of the values over the requested axis.

        Parameters
        ----------
        axis : Optional[Union[int, Literal[0]]], default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

            .. warning::

                The behavior of DataFrame.std with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        ddof : int, default 1
            Degrees of freedom.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            The standard deviation for the requested axis.

        See Also
        --------
        numpy.std : Equivalent numpy function for arrays.
        Series.var : Return unbiased variance over requested axis.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.var : Return the variance over the requested axis.
        DataFrame.std : Return the standard deviation over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "person_id": [0, 1, 2, 3],
        ...         "age": [21, 25, 62, 43],
        ...         "height": [1.61, 1.87, 1.49, 2.01],
        ...     }
        ... ).set_index("person_id")
        >>> df
                   age  height
        person_id
        0           21    1.61
        1           25    1.87
        2           62    1.49
        3           43    2.01

        >>> df.std()
        age       18.204155
        height     0.245519
        dtype: float64

        Alternatively, ``ddof=0`` can be set to normalize by N instead of N-1:

        >>> df.std(ddof=0)
        age       16.436296
        height     0.205851
        dtype: float64
        """
        return NDFrame.std(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    def skew(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return the skewness of the values over the requested axis.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            Skewness of the values for the requested axis.

        See Also
        --------
        numpy.skew : Equivalent numpy function for arrays.
        Series.kurt : Return unbiased kurtosis over requested axis.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.std : Standard deviation of the values.
        Series.var : Variance of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.index.min : Return the index of the minimum over the requested axis.
        DataFrame.index.max : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        >>> s.skew()
        0.0
        >>> s = pd.Series([1, 2, 3, 10], index=["a", "b", "c", "d"])
        >>> s.skew()
        1.1461932206205813
        """
        return NDFrame.skew(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def kurt(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return unbiased kurtosis over requested axis.

        Normalized by N-1 by default, using Fisher's definition of kurtosis (kurtosis of normal == 0.0).

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            Kurtosis of the values for the requested axis.

        See Also
        --------
        numpy.kurtosis : Equivalent numpy function for arrays.
        Series.skew : Return unbiased skew over requested axis.
        Series.std : Return unbiased standard deviation over requested axis.
        Series.var : Return unbiased variance over requested axis.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.std : Return the standard deviation over the requested axis.
        DataFrame.var : Return the variance over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.kurt()
        -1.2
        >>> s = pd.Series([1, 2, 3, 10])
        >>> s.kurt()
        1.7
        """
        return NDFrame.kurt(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='kurtosis')
    @doc(make_doc('kurt', ndim=1))
    def kurtosis(
        self,
        axis: int = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any
    ) -> Any:
        """
        Return unbiased kurtosis over requested axis.

        Normalized by N-1 by default.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        **kwargs : Any
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Any
            Kurtosis of the values for the requested axis.

        See Also
        --------
        numpy.kurtosis : Equivalent numpy function for arrays.
        Series.skew : Return unbiased skew over requested axis.
        Series.std : Return unbiased standard deviation over requested axis.
        Series.var : Return unbiased variance over requested axis.
        Series.sum : Sum of the values.
        Series.median : Median of the values.
        Series.min : Minimum value.
        Series.max : Maximum value.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.std : Return the standard deviation over the requested axis.
        DataFrame.var : Return the variance over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.kurtosis()
        -1.2
        >>> s = pd.Series([1, 2, 3, 10])
        >>> s.kurtosis()
        1.7
        """
        return self.kurt(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
