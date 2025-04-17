from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
import functools
import operator
import sys
from textwrap import dedent
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    properties,
    reshape,
)
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
    ChainedAssignmentError,
    InvalidIndexError,
)
from pandas.errors.cow import (
    _chained_assignment_method_msg,
    _chained_assignment_msg,
)
from pandas.util._decorators import (
    Appender,
    Substitution,
    deprecate_nonkeyword_arguments,
    doc,
    set_module,
)
from pandas.util._validators import (
    validate_ascending,
    validate_bool_kwarg,
    validate_percentile,
)

from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
    LossySetitemError,
    construct_1d_arraylike_from_scalar,
    find_common_type,
    infer_dtype_from,
    maybe_box_native,
    maybe_cast_pointwise_result,
)
from pandas.core.dtypes.common import (
    is_dict_like,
    is_float,
    is_integer,
    is_iterator,
    is_list_like,
    is_object_dtype,
    is_scalar,
    pandas_dtype,
    validate_all_hashable,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
    remove_na_arraylike,
)

from pandas.core import (
    algorithms,
    base,
    common as com,
    nanops,
    ops,
    roperator,
)
from pandas.core.accessor import Accessor
from pandas.core.apply import SeriesApply
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.arrow import (
    ListAccessor,
    StructAccessor,
)
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
    array as pd_array,
    extract_array,
    sanitize_array,
)
from pandas.core.generic import (
    NDFrame,
    make_doc,
)
from pandas.core.indexers import (
    disallow_ndim_indexing,
    unpack_1tuple,
)
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    default_index,
    ensure_index,
    maybe_sequence_to_range,
)
import pandas.core.indexes.base as ibase
from pandas.core.indexes.multi import maybe_droplevels
from pandas.core.indexing import (
    check_bool_indexer,
    check_dict_or_set_indexers,
)
from pandas.core.internals import SingleBlockManager
from pandas.core.methods import selectn
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
    ensure_key_mapped,
    nargsort,
)
from pandas.core.strings.accessor import StringMethods
from pandas.core.tools.datetimes import to_datetime

import pandas.io.formats.format as fmt
from pandas.io.formats.info import (
    INFO_DOCSTRING,
    SeriesInfo,
    series_sub_kwargs,
)
import pandas.plotting

if TYPE_CHECKING:
    from pandas._libs.internals import BlockValuesRefs
    from pandas._typing import (
        AggFuncType,
        AnyAll,
        AnyArrayLike,
        ArrayLike,
        Axis,
        AxisInt,
        CorrelationMethod,
        DropKeep,
        Dtype,
        DtypeObj,
        FilePath,
        Frequency,
        IgnoreRaise,
        IndexKeyFunc,
        IndexLabel,
        Level,
        ListLike,
        MutableMappingT,
        NaPosition,
        NumpySorter,
        NumpyValueArrayLike,
        QuantileInterpolation,
        ReindexMethod,
        Renamer,
        Scalar,
        Self,
        SortKind,
        StorageOptions,
        Suffixes,
        ValueKeyFunc,
        WriteBuffer,
        npt,
    )

    from pandas.core.frame import DataFrame
    from pandas.core.groupby.generic import SeriesGroupBy

__all__ = ["Series"]

_shared_doc_kwargs = {
    "axes": "index",
    "klass": "Series",
    "axes_single_arg": "{0 or 'index'}",
    "axis": """axis : {0 or 'index'}
        Unused. Parameter needed for compatibility with DataFrame.""",
    "inplace": """inplace : bool, default False
        If True, performs operation inplace and returns None.""",
    "unique": "np.ndarray",
    "duplicated": "Series",
    "optional_by": "",
    "optional_reindex": """
index : array-like, optional
    New labels for the index. Preferably an Index object to avoid
    duplicating data.
axis : int or str, optional
    Unused.""",
}

# ----------------------------------------------------------------------
# Series class


# error: Cannot override final attribute "ndim" (previously declared in base
# class "NDFrame")
# error: Cannot override final attribute "size" (previously declared in base
# class "NDFrame")
# definition in base class "NDFrame"
@set_module("pandas")
class Series(base.IndexOpsMixin, NDFrame):  # type: ignore[misc]
    """
    One-dimensional ndarray with axis labels (including time series).

    Labels need not be unique but must be a hashable type. The object
    supports both integer- and label-based indexing and provides a host of
    methods for performing operations involving the index. Statistical
    methods from ndarray have been overridden to automatically exclude
    missing data (currently represented as NaN).

    Operations between Series (+, -, /, \\*, \\*\\*) align values based on their
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
    dtype : str, numpy.dtype, or ExtensionDtype, optional
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

    _typ = "series"
    _HANDLED_TYPES = (Index, ExtensionArray, np.ndarray)

    _name: Hashable
    _metadata: list[str] = ["_name"]
    _internal_names_set = {"index", "name"} | NDFrame._internal_names_set
    _accessors = {"dt", "cat", "str", "sparse"}
    _hidden_attrs = (
        base.IndexOpsMixin._hidden_attrs | NDFrame._hidden_attrs | frozenset([])
    )

    # similar to __array_priority__, positions Series after DataFrame
    #  but before Index and ExtensionArray.  Should NOT be overridden by subclasses.
    __pandas_priority__ = 3000

    # Override cache_readonly bc Series is mutable
    # error: Incompatible types in assignment (expression has type "property",
    # base class "IndexOpsMixin" defined the type as "Callable[[IndexOpsMixin], bool]")
    hasnans = property(  # type: ignore[assignment]
        # error: "Callable[[IndexOpsMixin], bool]" has no attribute "fget"
        base.IndexOpsMixin.hasnans.fget,  # type: ignore[attr-defined]
        doc=base.IndexOpsMixin.hasnans.__doc__,
    )
    _mgr: SingleBlockManager

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
        self,
        data=None,
        index=None,
        dtype: Dtype | None = None,
        name=None,
        copy: bool | None = None,
    ) -> None:
        allow_mgr = False
        if (
            isinstance(data, SingleBlockManager)
            and index is None
            and dtype is None
            and (copy is False or copy is None)
        ):
            if not allow_mgr:
                # GH#52419
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} "
                    "is deprecated and will raise in a future version. "
                    "Use public APIs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            data = data.copy(deep=False)
            # GH#33357 called with just the SingleBlockManager
            NDFrame.__init__(self, data)
            self.name = name
            return

        if isinstance(data, (ExtensionArray, np.ndarray)):
            if copy is not False:
                if dtype is None or astype_is_view(data.dtype, pandas_dtype(dtype)):
                    data = data.copy()
        if copy is None:
            copy = False

        if isinstance(data, SingleBlockManager) and not copy:
            data = data.copy(deep=False)

            if not allow_mgr:
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} "
                    "is deprecated and will raise in a future version. "
                    "Use public APIs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

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
            raise NotImplementedError(
                "initializing a Series from a MultiIndex is not supported"
            )

        refs = None
        if isinstance(data, Index):
            if dtype is not None:
                data = data.astype(dtype)

            refs = data._references
            copy = False

        elif isinstance(data, np.ndarray):
            if len(data.dtype):
                # GH#13296 we are dealing with a compound dtype, which
                #  should be treated as 2D
                raise ValueError(
                    "Cannot construct a Series from an ndarray with "
                    "compound dtype.  Use DataFrame instead."
                )
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
                # GH#19275 SingleBlockManager input should only be called
                # internally
                raise AssertionError(
                    "Cannot pass both SingleBlockManager "
                    "`data` argument and a different "
                    "`index` argument. `copy` must be False."
                )

            if not allow_mgr:
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} "
                    "is deprecated and will raise in a future version. "
                    "Use public APIs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                allow_mgr = True

        elif isinstance(data, ExtensionArray):
            pass
        else:
            data = com.maybe_iterable_to_list(data)
            if is_list_like(data) and not len(data) and dtype is None:
                # GH 29405: Pre-2.0, this defaulted to float.
                dtype = np.dtype(object)

        if index is None:
            if not is_list_like(data):
                data = [data]
            index = default_index(len(data))
        elif is_list_like(data):
            com.require_length_match(data, index)

        # create/copy the manager
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
        self, data: Mapping, index: Index | None = None, dtype: DtypeObj | None = None
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
        # Looking for NaN in dict doesn't work ({np.nan : 1}[float('nan')]
        # raises KeyError), so we iterate the entire dict, and align
        if data:
            # GH:34717, issue was using zip to extract key and values from data.
            # using generators in effects the performance.
            # Below is the new way of extracting the keys and values

            keys = maybe_sequence_to_range(tuple(data.keys()))
            values = list(data.values())  # Generating list of values- faster way
        elif index is not None:
            # fastpath for Series(data=None). Just use broadcasting a scalar
            # instead of reindexing.
            if len(index) or dtype is not None:
                values = na_value_for_dtype(pandas_dtype(dtype), compat=False)
            else:
                values = []
            keys = index
        else:
            keys, values = default_index(0), []

        # Input is now list-like, so rely on "standard" construction:
        s = Series(values, index=keys, dtype=dtype)

        # Now we just make sure the order is respected, if any
        if data and index is not None:
            s = s.reindex(index)
        return s._mgr, s.index

    # ----------------------------------------------------------------------

    def __arrow_c_stream__(self, requested_schema=None) -> Any:
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
        pa = import_optional_dependency("pyarrow", min_version="16.0.0")
        type = (
            pa.DataType._import_from_c_capsule(requested_schema)
            if requested_schema is not None
            else None
        )
        ca = pa.array(self, type=type)
        if not isinstance(ca, pa.ChunkedArray):
            ca = pa.chunked_array([ca])
        return ca.__arrow_c_stream__()

    # ----------------------------------------------------------------------

    @property
    def _constructor(self) -> type[Series]:
        return Series

    def _constructor_from_mgr(self, mgr, axes) -> Series:
        ser = Series._from_mgr(mgr, axes=axes)
        ser._name = None  # caller is responsible for setting real name

        if type(self) is Series:
            # This would also work `if self._constructor is Series`, but
            #  this check is slightly faster, benefiting the most-common case.
            return ser

        # We assume that the subclass __init__ knows how to handle a
        #  pd.Series object.
        return self._constructor(ser)

    @property
    def _constructor_expanddim(self) -> Callable[..., DataFrame]:
        """
        Used when a manipulation result has one higher dimension as the
        original, such as Series.to_frame()
        """
        from pandas.core.frame import DataFrame

        return DataFrame

    def _constructor_expanddim_from_mgr(self, mgr, axes) -> DataFrame:
        from pandas.core.frame import DataFrame

        df = DataFrame._from_mgr(mgr, axes=mgr.axes)

        if type(self) is Series:
            # This would also work `if self._constructor_expanddim is DataFrame`,
            #  but this check is slightly faster, benefiting the most-common case.
            return df

        # We assume that the subclass __init__ knows how to handle a
        #  pd.DataFrame object.
        return self._constructor_expanddim(df)

    # types
    @property
    def _can_hold_na(self) -> bool:
        return self._mgr._can_hold_na

    # ndarray compatibility
    @property
    def dtype(self) -> DtypeObj:
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
    def dtypes(self) -> DtypeObj:
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
        # DataFrame compatibility
        return self.dtype

    @property
    def name(self) -> Hashable:
        """
        Return the name of the Series.

        The name of a Series becomes its index or column name if it is used
        to form a DataFrame. It is also used whenever displaying the Series
        using the interpreter.

        Returns
        -------
        label (hashable object)
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
    def name(self, value: Hashable) -> None:
        validate_all_hashable(value, error_name=f"{type(self).__name__}.name")
        object.__setattr__(self, "_name", value)

    @property
    def values(self) -> np.ndarray:
        """
        Return Series as ndarray or ndarray-like depending on the dtype.

        .. warning::

           We recommend using :attr:`Series.array` or
           :meth:`Series.to_numpy`, depending on whether you need
           a reference to the underlying data or a NumPy array.

        Returns
        -------
        numpy.ndarray or ndarray-like

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
    def _values(self) -> ArrayLike:
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
    def _references(self) -> BlockValuesRefs:
        return self._mgr._block.refs

    @Appender(base.IndexOpsMixin.array.__doc__)  # type: ignore[prop-decorator]
    @property
    def array(self) -> ExtensionArray:
        return self._mgr.array_values()

    def __len__(self) -> int:
        """
        Return the length of the Series.
        """
        return len(self._mgr)

    # ----------------------------------------------------------------------
    # NDArray Compat
    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """
        Return the values as a NumPy array.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to use for the resulting NumPy array. By default,
            the dtype is inferred from the data.

        copy : bool or None, optional
            See :func:`numpy.asarray`.

        Returns
        -------
        numpy.ndarray
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
            # Note: branch avoids `copy=None` for NumPy 1.x support
            arr = np.asarray(values, dtype=dtype)
        else:
            arr = np.array(values, dtype=dtype, copy=copy)

        if copy is True:
            return arr
        if copy is False or astype_is_view(values.dtype, arr.dtype):
            arr = arr.view()
            arr.flags.writeable = False
        return arr

    # ----------------------------------------------------------------------

    # indexers
    @property
    def axes(self) -> list[Index]:
        """
        Return a list of the row axis labels.
        """
        return [self.index]

    # ----------------------------------------------------------------------
    # Indexing Methods

    def _ixs(self, i: int, axis: AxisInt = 0) -> Any:
        """
        Return the i-th value or values in the Series by location.

        Parameters
        ----------
        i : int

        Returns
        -------
        scalar
        """
        return self._values[i]

    def _slice(self, slobj: slice, axis: AxisInt = 0) -> Series:
        # axis kwarg is retained for compat with NDFrame method
        #  _slice is *always* positional
        mgr = self._mgr.get_slice(slobj, axis=axis)
        out = self._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self._name
        return out.__finalize__(self)

    def __getitem__(self, key) -> Any:
        check_dict_or_set_indexers(key)
        key = com.apply_if_callable(key, self)

        if key is Ellipsis:
            return self.copy(deep=False)

        key_is_scalar = is_scalar(key)
        if isinstance(key, (list, tuple)):
            key = unpack_1tuple(key)

        elif key_is_scalar:
            # Note: GH#50617 in 3.0 we changed int key to always be treated as
            #  a label, matching DataFrame behavior.
            return self._get_value(key)

        # Convert generator to list before going through hashable part
        # (We will iterate through the generator there to check for slices)
        if is_iterator(key):
            key = list(key)

        if is_hashable(key) and not isinstance(key, slice):
            # Otherwise index.get_value will raise InvalidIndexError
            try:
                # For labels that don't resolve as scalars like tuples and frozensets
                result = self._get_value(key)

                return result

            except (KeyError, TypeError, InvalidIndexError):
                # InvalidIndexError for e.g. generator
                #  see test_series_getitem_corner_generator
                if isinstance(key, tuple) and isinstance(self.index, MultiIndex):
                    # We still have the corner case where a tuple is a key
                    # in the first level of our MultiIndex
                    return self._get_values_tuple(key)

        if isinstance(key, slice):
            # Do slice check before somewhat-costly is_bool_indexer
            return self._getitem_slice(key)

        if com.is_bool_indexer(key):
            key = check_bool_indexer(self.index, key)
            key = np.asarray(key, dtype=bool)
            return self._get_rows_with_mask(key)

        return self._get_with(key)

    def _get_with(self, key) -> Series:
        # other: fancy integer or otherwise
        if isinstance(key, ABCDataFrame):
            raise TypeError(
                "Indexing a Series with DataFrame is not "
                "supported, use the appropriate DataFrame column"
            )
        elif isinstance(key, tuple):
            return self._get_values_tuple(key)

        return self.loc[key]

    def _get_values_tuple(self, key: tuple) -> Series:
        # mpl hackaround
        if com.any_none(*key):
            # mpl compat if we look up e.g. ser[:, np.newaxis];
            #  see tests.series.timeseries.test_mpl_compat_hack
            # the asarray is needed to avoid returning a 2D DatetimeArray
            result = np.asarray(self._values[key])
            disallow_ndim_indexing(result)
            return result

        if not isinstance(self.index, MultiIndex):
            raise KeyError("key of type tuple not found and not a MultiIndex")

        # If key is contained, would have returned by now
        indexer, new_index = self.index.get_loc_level(key)
        new_ser = self._constructor(self._values[indexer], index=new_index, copy=False)
        if isinstance(indexer, slice):
            new_ser._mgr.add_references(self._mgr)
        return new_ser.__finalize__(self)

    def _get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Series:
        new_mgr = self._mgr.get_rows_with_mask(indexer)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self)

    def _get_value(self, label, takeable: bool = False) -> Any:
        """
        Quickly retrieve single value at passed index label.

        Parameters
        ----------
        label : object
        takeable : interpret the index as indexers, default False

        Returns
        -------
        scalar value
        """
        if takeable:
            return self._values[label]

        # Similar to Index.get_value, but we do not fall back to positional
        loc = self.index.get_loc(label)

        if is_integer(loc):
            return self._values[loc]

        if isinstance(self.index, MultiIndex):
            mi = self.index
            new_values = self._values[loc]
            if len(new_values) == 1 and mi.nlevels == 1:
                # If more than one level left, we can not return a scalar
                return new_values[0]

            new_index = mi[loc]
            new_index = maybe_droplevels(new_index, label)
            new_ser = self._constructor(
                new_values, index=new_index, name=self.name, copy=False
            )
            if isinstance(l