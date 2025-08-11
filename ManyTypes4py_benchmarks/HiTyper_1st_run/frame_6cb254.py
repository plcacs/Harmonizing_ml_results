"""
DataFrame
---------
An efficient 2D container for potentially mixed-type time series or other
labeled data series.

Similar to its R counterpart, data.frame, except providing automatic data
alignment and a host of useful data manipulation methods having to do with the
labeling information
"""
from __future__ import annotations
import collections
from collections import abc
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
import functools
from io import StringIO
import itertools
import operator
import sys
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, cast, overload
import warnings
import numpy as np
from numpy import ma
from pandas._config import get_option
from pandas._libs import algos as libalgos, lib, properties
from pandas._libs.hashtable import duplicated
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import ChainedAssignmentError, InvalidIndexError
from pandas.errors.cow import _chained_assignment_method_msg, _chained_assignment_msg
from pandas.util._decorators import Appender, Substitution, deprecate_nonkeyword_arguments, doc, set_module
from pandas.util._exceptions import find_stack_level, rewrite_warning
from pandas.util._validators import validate_ascending, validate_bool_kwarg, validate_percentile
from pandas.core.dtypes.cast import LossySetitemError, can_hold_element, construct_1d_arraylike_from_scalar, construct_2d_arraylike_from_scalar, find_common_type, infer_dtype_from_scalar, invalidate_string_dtypes, maybe_downcast_to_dtype
from pandas.core.dtypes.common import infer_dtype_from_object, is_1d_only_ea_dtype, is_array_like, is_bool_dtype, is_dataclass, is_dict_like, is_float, is_float_dtype, is_hashable, is_integer, is_integer_dtype, is_iterator, is_list_like, is_scalar, is_sequence, needs_i8_conversion, pandas_dtype
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ArrowDtype, BaseMaskedDtype, ExtensionDtype
from pandas.core.dtypes.missing import isna, notna
from pandas.core import algorithms, common as com, nanops, ops, roperator
from pandas.core.accessor import Accessor
from pandas.core.apply import reconstruct_and_relabel_result
from pandas.core.array_algos.take import take_2d_multi
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import BaseMaskedArray, DatetimeArray, ExtensionArray, PeriodArray, TimedeltaArray
from pandas.core.arrays.sparse import SparseFrameAccessor
from pandas.core.construction import ensure_wrapped_if_datetimelike, sanitize_array, sanitize_masked_array
from pandas.core.generic import NDFrame, make_doc
from pandas.core.indexers import check_key_length
from pandas.core.indexes.api import DatetimeIndex, Index, PeriodIndex, default_index, ensure_index, ensure_index_from_sequences
from pandas.core.indexes.multi import MultiIndex, maybe_droplevels
from pandas.core.indexing import check_bool_indexer, check_dict_or_set_indexers
from pandas.core.internals import BlockManager
from pandas.core.internals.construction import arrays_to_mgr, dataclasses_to_dicts, dict_to_mgr, ndarray_to_mgr, nested_data_to_arrays, rec_array_to_mgr, reorder_arrays, to_arrays, treat_as_nested
from pandas.core.methods import selectn
from pandas.core.reshape.melt import melt
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_group_index, lexsort_indexer, nargsort
from pandas.io.common import get_handle
from pandas.io.formats import console, format as fmt
from pandas.io.formats.info import INFO_DOCSTRING, DataFrameInfo, frame_sub_kwargs
import pandas.plotting
if TYPE_CHECKING:
    import datetime
    from pandas._libs.internals import BlockValuesRefs
    from pandas._typing import AggFuncType, AnyAll, AnyArrayLike, ArrayLike, Axes, Axis, AxisInt, ColspaceArgType, CompressionOptions, CorrelationMethod, DropKeep, Dtype, DtypeObj, FilePath, FloatFormatType, FormattersType, Frequency, FromDictOrient, HashableT, HashableT2, IgnoreRaise, IndexKeyFunc, IndexLabel, JoinValidate, Level, ListLike, MergeHow, MergeValidate, MutableMappingT, NaPosition, NsmallestNlargestKeep, PythonFuncType, QuantileInterpolation, ReadBuffer, ReindexMethod, Renamer, Scalar, Self, SequenceNotStr, SortKind, StorageOptions, Suffixes, T, ToStataByteorder, ToTimestampHow, UpdateJoin, ValueKeyFunc, WriteBuffer, XMLParsers, npt
    from pandas.core.groupby.generic import DataFrameGroupBy
    from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
    from pandas.core.internals.managers import SingleBlockManager
    from pandas.io.formats.style import Styler
_shared_doc_kwargs = {'axes': 'index, columns', 'klass': 'DataFrame', 'axes_single_arg': "{0 or 'index', 1 or 'columns'}", 'axis': "axis : {0 or 'index', 1 or 'columns'}, default 0\n        If 0 or 'index': apply function to each column.\n        If 1 or 'columns': apply function to each row.", 'inplace': '\n    inplace : bool, default False\n        Whether to modify the DataFrame rather than creating a new one.', 'optional_by': "\nby : str or list of str\n    Name or list of names to sort by.\n\n    - if `axis` is 0 or `'index'` then `by` may contain index\n      levels and/or column labels.\n    - if `axis` is 1 or `'columns'` then `by` may contain column\n      levels and/or index labels.", 'optional_reindex': "\nlabels : array-like, optional\n    New labels / index to conform the axis specified by 'axis' to.\nindex : array-like, optional\n    New labels for the index. Preferably an Index object to avoid\n    duplicating data.\ncolumns : array-like, optional\n    New labels for the columns. Preferably an Index object to avoid\n    duplicating data.\naxis : int or str, optional\n    Axis to target. Can be either the axis name ('index', 'columns')\n    or number (0, 1)."}
_merge_doc = '\nMerge DataFrame or named Series objects with a database-style join.\n\nA named Series object is treated as a DataFrame with a single named column.\n\nThe join is done on columns or indexes. If joining columns on\ncolumns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes\non indexes or indexes on a column or columns, the index will be passed on.\nWhen performing a cross merge, no column specifications to merge on are\nallowed.\n\n.. warning::\n\n    If both key columns contain rows where the key is a null value, those\n    rows will be matched against each other. This is different from usual SQL\n    join behaviour and can lead to unexpected results.\n\nParameters\n----------%s\nright : DataFrame or named Series\n    Object to merge with.\nhow : {\'left\', \'right\', \'outer\', \'inner\', \'cross\', \'left_anti\', \'right_anti\'},\n    default \'inner\'\n    Type of merge to be performed.\n\n    * left: use only keys from left frame, similar to a SQL left outer join;\n      preserve key order.\n    * right: use only keys from right frame, similar to a SQL right outer join;\n      preserve key order.\n    * outer: use union of keys from both frames, similar to a SQL full outer\n      join; sort keys lexicographically.\n    * inner: use intersection of keys from both frames, similar to a SQL inner\n      join; preserve the order of the left keys.\n    * cross: creates the cartesian product from both frames, preserves the order\n      of the left keys.\n    * left_anti: use only keys from left frame that are not in right frame, similar\n      to SQL left anti join; preserve key order.\n    * right_anti: use only keys from right frame that are not in left frame, similar\n      to SQL right anti join; preserve key order.\non : label or list\n    Column or index level names to join on. These must be found in both\n    DataFrames. If `on` is None and not merging on indexes then this defaults\n    to the intersection of the columns in both DataFrames.\nleft_on : label or list, or array-like\n    Column or index level names to join on in the left DataFrame. Can also\n    be an array or list of arrays of the length of the left DataFrame.\n    These arrays are treated as if they are columns.\nright_on : label or list, or array-like\n    Column or index level names to join on in the right DataFrame. Can also\n    be an array or list of arrays of the length of the right DataFrame.\n    These arrays are treated as if they are columns.\nleft_index : bool, default False\n    Use the index from the left DataFrame as the join key(s). If it is a\n    MultiIndex, the number of keys in the other DataFrame (either the index\n    or a number of columns) must match the number of levels.\nright_index : bool, default False\n    Use the index from the right DataFrame as the join key. Same caveats as\n    left_index.\nsort : bool, default False\n    Sort the join keys lexicographically in the result DataFrame. If False,\n    the order of the join keys depends on the join type (how keyword).\nsuffixes : list-like, default is ("_x", "_y")\n    A length-2 sequence where each element is optionally a string\n    indicating the suffix to add to overlapping column names in\n    `left` and `right` respectively. Pass a value of `None` instead\n    of a string to indicate that the column name from `left` or\n    `right` should be left as-is, with no suffix. At least one of the\n    values must not be None.\ncopy : bool, default False\n    If False, avoid copy if possible.\n\n    .. note::\n        The `copy` keyword will change behavior in pandas 3.0.\n        `Copy-on-Write\n        <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__\n        will be enabled by default, which means that all methods with a\n        `copy` keyword will use a lazy copy mechanism to defer the copy and\n        ignore the `copy` keyword. The `copy` keyword will be removed in a\n        future version of pandas.\n\n        You can already get the future behavior and improvements through\n        enabling copy on write ``pd.options.mode.copy_on_write = True``\n\n    .. deprecated:: 3.0.0\nindicator : bool or str, default False\n    If True, adds a column to the output DataFrame called "_merge" with\n    information on the source of each row. The column can be given a different\n    name by providing a string argument. The column will have a Categorical\n    type with the value of "left_only" for observations whose merge key only\n    appears in the left DataFrame, "right_only" for observations\n    whose merge key only appears in the right DataFrame, and "both"\n    if the observation\'s merge key is found in both DataFrames.\n\nvalidate : str, optional\n    If specified, checks if merge is of specified type.\n\n    * "one_to_one" or "1:1": check if merge keys are unique in both\n      left and right datasets.\n    * "one_to_many" or "1:m": check if merge keys are unique in left\n      dataset.\n    * "many_to_one" or "m:1": check if merge keys are unique in right\n      dataset.\n    * "many_to_many" or "m:m": allowed, but does not result in checks.\n\nReturns\n-------\nDataFrame\n    A DataFrame of the two merged objects.\n\nSee Also\n--------\nmerge_ordered : Merge with optional filling/interpolation.\nmerge_asof : Merge on nearest keys.\nDataFrame.join : Similar method using indices.\n\nExamples\n--------\n>>> df1 = pd.DataFrame({\'lkey\': [\'foo\', \'bar\', \'baz\', \'foo\'],\n...                     \'value\': [1, 2, 3, 5]})\n>>> df2 = pd.DataFrame({\'rkey\': [\'foo\', \'bar\', \'baz\', \'foo\'],\n...                     \'value\': [5, 6, 7, 8]})\n>>> df1\n    lkey value\n0   foo      1\n1   bar      2\n2   baz      3\n3   foo      5\n>>> df2\n    rkey value\n0   foo      5\n1   bar      6\n2   baz      7\n3   foo      8\n\nMerge df1 and df2 on the lkey and rkey columns. The value columns have\nthe default suffixes, _x and _y, appended.\n\n>>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\')\n  lkey  value_x rkey  value_y\n0  foo        1  foo        5\n1  foo        1  foo        8\n2  bar        2  bar        6\n3  baz        3  baz        7\n4  foo        5  foo        5\n5  foo        5  foo        8\n\nMerge DataFrames df1 and df2 with specified left and right suffixes\nappended to any overlapping columns.\n\n>>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\',\n...           suffixes=(\'_left\', \'_right\'))\n  lkey  value_left rkey  value_right\n0  foo           1  foo            5\n1  foo           1  foo            8\n2  bar           2  bar            6\n3  baz           3  baz            7\n4  foo           5  foo            5\n5  foo           5  foo            8\n\nMerge DataFrames df1 and df2, but raise an exception if the DataFrames have\nany overlapping columns.\n\n>>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\', suffixes=(False, False))\nTraceback (most recent call last):\n...\nValueError: columns overlap but no suffix specified:\n    Index([\'value\'], dtype=\'object\')\n\n>>> df1 = pd.DataFrame({\'a\': [\'foo\', \'bar\'], \'b\': [1, 2]})\n>>> df2 = pd.DataFrame({\'a\': [\'foo\', \'baz\'], \'c\': [3, 4]})\n>>> df1\n      a  b\n0   foo  1\n1   bar  2\n>>> df2\n      a  c\n0   foo  3\n1   baz  4\n\n>>> df1.merge(df2, how=\'inner\', on=\'a\')\n      a  b  c\n0   foo  1  3\n\n>>> df1.merge(df2, how=\'left\', on=\'a\')\n      a  b  c\n0   foo  1  3.0\n1   bar  2  NaN\n\n>>> df1 = pd.DataFrame({\'left\': [\'foo\', \'bar\']})\n>>> df2 = pd.DataFrame({\'right\': [7, 8]})\n>>> df1\n    left\n0   foo\n1   bar\n>>> df2\n    right\n0   7\n1   8\n\n>>> df1.merge(df2, how=\'cross\')\n   left  right\n0   foo      7\n1   foo      8\n2   bar      7\n3   bar      8\n'

@set_module('pandas')
class DataFrame(NDFrame, OpsMixin):
    """
    Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    Data structure also contains labeled axes (rows and columns).
    Arithmetic operations align on both row and column labels. Can be
    thought of as a dict-like container for Series objects. The primary
    pandas data structure.

    Parameters
    ----------
    data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
        Dict can contain Series, arrays, constants, dataclass or list-like objects. If
        data is a dict, column order follows insertion-order. If a dict contains Series
        which have an index defined, it is aligned by its index. This alignment also
        occurs if data is a Series or a DataFrame itself. Alignment is done on
        Series/DataFrame inputs.

        If data is a list of dicts, column order follows insertion-order.

    index : Index or array-like
        Index to use for resulting frame. Will default to RangeIndex if
        no indexing information part of input data and no index provided.
    columns : Index or array-like
        Column labels to use for resulting frame when data does not have them,
        defaulting to RangeIndex(0, 1, 2, ..., n). If data contains column labels,
        will perform column selection instead.
    dtype : dtype, default None
        Data type to force. Only a single dtype is allowed. If None, infer.
        If ``data`` is DataFrame then is ignored.
    copy : bool or None, default None
        Copy data from inputs.
        For dict data, the default of None behaves like ``copy=True``.  For DataFrame
        or 2d ndarray input, the default of None behaves like ``copy=False``.
        If data is a dict containing one or more Series (possibly of different dtypes),
        ``copy=False`` will ensure that these inputs are not copied.

        .. versionchanged:: 1.3.0

    See Also
    --------
    DataFrame.from_records : Constructor from tuples, also record arrays.
    DataFrame.from_dict : From dicts of Series, arrays, or dicts.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_table : Read general delimited file into DataFrame.
    read_clipboard : Read text from clipboard into DataFrame.

    Notes
    -----
    Please reference the :ref:`User Guide <basics.dataframe>` for more information.

    Examples
    --------
    Constructing DataFrame from a dictionary.

    >>> d = {"col1": [1, 2], "col2": [3, 4]}
    >>> df = pd.DataFrame(data=d)
    >>> df
       col1  col2
    0     1     3
    1     2     4

    Notice that the inferred dtype is int64.

    >>> df.dtypes
    col1    int64
    col2    int64
    dtype: object

    To enforce a single dtype:

    >>> df = pd.DataFrame(data=d, dtype=np.int8)
    >>> df.dtypes
    col1    int8
    col2    int8
    dtype: object

    Constructing DataFrame from a dictionary including Series:

    >>> d = {"col1": [0, 1, 2, 3], "col2": pd.Series([2, 3], index=[2, 3])}
    >>> pd.DataFrame(data=d, index=[0, 1, 2, 3])
       col1  col2
    0     0   NaN
    1     1   NaN
    2     2   2.0
    3     3   3.0

    Constructing DataFrame from numpy ndarray:

    >>> df2 = pd.DataFrame(
    ...     np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=["a", "b", "c"]
    ... )
    >>> df2
       a  b  c
    0  1  2  3
    1  4  5  6
    2  7  8  9

    Constructing DataFrame from a numpy ndarray that has labeled columns:

    >>> data = np.array(
    ...     [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
    ...     dtype=[("a", "i4"), ("b", "i4"), ("c", "i4")],
    ... )
    >>> df3 = pd.DataFrame(data, columns=["c", "a"])
    >>> df3
       c  a
    0  3  1
    1  6  4
    2  9  7

    Constructing DataFrame from dataclass:

    >>> from dataclasses import make_dataclass
    >>> Point = make_dataclass("Point", [("x", int), ("y", int)])
    >>> pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)])
       x  y
    0  0  0
    1  0  3
    2  2  3

    Constructing DataFrame from Series/DataFrame:

    >>> ser = pd.Series([1, 2, 3], index=["a", "b", "c"])
    >>> df = pd.DataFrame(data=ser, index=["a", "c"])
    >>> df
       0
    a  1
    c  3

    >>> df1 = pd.DataFrame([1, 2, 3], index=["a", "b", "c"], columns=["x"])
    >>> df2 = pd.DataFrame(data=df1, index=["a", "c"])
    >>> df2
       x
    a  1
    c  3
    """
    _internal_names_set = {'columns', 'index'} | NDFrame._internal_names_set
    _typ = 'dataframe'
    _HANDLED_TYPES = (Series, Index, ExtensionArray, np.ndarray)
    _accessors = {'sparse'}
    _hidden_attrs = NDFrame._hidden_attrs | frozenset([])
    __pandas_priority__ = 4000

    @property
    def _constructor(self) -> DataFrame:
        return DataFrame

    def _constructor_from_mgr(self, mgr: Any, axes: Any):
        df = DataFrame._from_mgr(mgr, axes=axes)
        if type(self) is DataFrame:
            return df
        elif type(self).__name__ == 'GeoDataFrame':
            return self._constructor(mgr)
        return self._constructor(df)
    _constructor_sliced = Series

    def _constructor_sliced_from_mgr(self, mgr: Any, axes: Any):
        ser = Series._from_mgr(mgr, axes)
        ser._name = None
        if type(self) is DataFrame:
            return ser
        return self._constructor_sliced(ser)

    def __init__(self, data: None=None, index: None=None, columns: None=None, dtype: None=None, copy: None=None) -> None:
        allow_mgr = False
        if dtype is not None:
            dtype = self._validate_dtype(dtype)
        if isinstance(data, DataFrame):
            data = data._mgr
            allow_mgr = True
            if not copy:
                data = data.copy(deep=False)
        if isinstance(data, BlockManager):
            if not allow_mgr:
                warnings.warn(f'Passing a {type(data).__name__} to {type(self).__name__} is deprecated and will raise in a future version. Use public APIs instead.', DeprecationWarning, stacklevel=2)
            data = data.copy(deep=False)
            if index is None and columns is None and (dtype is None) and (not copy):
                NDFrame.__init__(self, data)
                return
        if isinstance(index, set):
            raise ValueError('index cannot be a set')
        if isinstance(columns, set):
            raise ValueError('columns cannot be a set')
        if copy is None:
            if isinstance(data, dict):
                copy = True
            elif not isinstance(data, (Index, DataFrame, Series)):
                copy = True
            else:
                copy = False
        if data is None:
            index = index if index is not None else default_index(0)
            columns = columns if columns is not None else default_index(0)
            dtype = dtype if dtype is not None else pandas_dtype(object)
            data = []
        if isinstance(data, BlockManager):
            mgr = self._init_mgr(data, axes={'index': index, 'columns': columns}, dtype=dtype, copy=copy)
        elif isinstance(data, dict):
            mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy)
        elif isinstance(data, ma.MaskedArray):
            from numpy.ma import mrecords
            if isinstance(data, mrecords.MaskedRecords):
                raise TypeError('MaskedRecords are not supported. Pass {name: data[name] for name in data.dtype.names} instead')
            data = sanitize_masked_array(data)
            mgr = ndarray_to_mgr(data, index, columns, dtype=dtype, copy=copy)
        elif isinstance(data, (np.ndarray, Series, Index, ExtensionArray)):
            if data.dtype.names:
                data = cast(np.ndarray, data)
                mgr = rec_array_to_mgr(data, index, columns, dtype, copy)
            elif getattr(data, 'name', None) is not None:
                mgr = dict_to_mgr({data.name: data}, index, columns, dtype=dtype, copy=copy)
            else:
                mgr = ndarray_to_mgr(data, index, columns, dtype=dtype, copy=copy)
        elif is_list_like(data):
            if not isinstance(data, abc.Sequence):
                if hasattr(data, '__array__'):
                    data = np.asarray(data)
                else:
                    data = list(data)
            if len(data) > 0:
                if is_dataclass(data[0]):
                    data = dataclasses_to_dicts(data)
                if not isinstance(data, np.ndarray) and treat_as_nested(data):
                    if columns is not None:
                        columns = ensure_index(columns)
                    arrays, columns, index = nested_data_to_arrays(data, columns, index, dtype)
                    mgr = arrays_to_mgr(arrays, columns, index, dtype=dtype)
                else:
                    mgr = ndarray_to_mgr(data, index, columns, dtype=dtype, copy=copy)
            else:
                mgr = dict_to_mgr({}, index, columns if columns is not None else default_index(0), dtype=dtype)
        else:
            if index is None or columns is None:
                raise ValueError('DataFrame constructor not properly called!')
            index = ensure_index(index)
            columns = ensure_index(columns)
            if not dtype:
                dtype, _ = infer_dtype_from_scalar(data)
            if isinstance(dtype, ExtensionDtype):
                values = [construct_1d_arraylike_from_scalar(data, len(index), dtype) for _ in range(len(columns))]
                mgr = arrays_to_mgr(values, columns, index, dtype=None)
            else:
                arr2d = construct_2d_arraylike_from_scalar(data, len(index), len(columns), dtype, copy)
                mgr = ndarray_to_mgr(arr2d, index, columns, dtype=arr2d.dtype, copy=False)
        NDFrame.__init__(self, mgr)

    def __dataframe__(self, nan_as_null: bool=False, allow_copy: bool=True) -> PandasDataFrameXchg:
        """
        Return the dataframe interchange object implementing the interchange protocol.

        .. note::

           For new development, we highly recommend using the Arrow C Data Interface
           alongside the Arrow PyCapsule Interface instead of the interchange protocol

        .. warning::

            Due to severe implementation issues, we recommend only considering using the
            interchange protocol in the following cases:

            - converting to pandas: for pandas >= 2.0.3
            - converting from pandas: for pandas >= 3.0.0

        Parameters
        ----------
        nan_as_null : bool, default False
            `nan_as_null` is DEPRECATED and has no effect. Please avoid using
            it; it will be removed in a future release.
        allow_copy : bool, default True
            Whether to allow memory copying when exporting. If set to False
            it would cause non-zero-copy exports to fail.

        Returns
        -------
        DataFrame interchange object
            The object which consuming library can use to ingress the dataframe.

        See Also
        --------
        DataFrame.from_records : Constructor from tuples, also record arrays.
        DataFrame.from_dict : From dicts of Series, arrays, or dicts.

        Notes
        -----
        Details on the interchange protocol:
        https://data-apis.org/dataframe-protocol/latest/index.html

        Examples
        --------
        >>> df_not_necessarily_pandas = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> interchange_object = df_not_necessarily_pandas.__dataframe__()
        >>> interchange_object.column_names()
        Index(['A', 'B'], dtype='object')
        >>> df_pandas = pd.api.interchange.from_dataframe(
        ...     interchange_object.select_columns_by_name(["A"])
        ... )
        >>> df_pandas
             A
        0    1
        1    2

        These methods (``column_names``, ``select_columns_by_name``) should work
        for any dataframe library which implements the interchange protocol.
        """
        from pandas.core.interchange.dataframe import PandasDataFrameXchg
        return PandasDataFrameXchg(self, allow_copy=allow_copy)

    def __arrow_c_stream__(self, requested_schema: None=None):
        """
        Export the pandas DataFrame as an Arrow C stream PyCapsule.

        This relies on pyarrow to convert the pandas DataFrame to the Arrow
        format (and follows the default behaviour of ``pyarrow.Table.from_pandas``
        in its handling of the index, i.e. store the index as a column except
        for RangeIndex).
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
        pa = import_optional_dependency('pyarrow', min_version='14.0.0')
        if requested_schema is not None:
            requested_schema = pa.Schema._import_from_c_capsule(requested_schema)
        table = pa.Table.from_pandas(self, schema=requested_schema)
        return table.__arrow_c_stream__()

    @property
    def axes(self) -> list:
        """
        Return a list representing the axes of the DataFrame.

        It has the row axis labels and column axis labels as the only members.
        They are returned in that order.

        See Also
        --------
        DataFrame.index: The index (row labels) of the DataFrame.
        DataFrame.columns: The column labels of the DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df.axes
        [RangeIndex(start=0, stop=2, step=1), Index(['col1', 'col2'],
        dtype='object')]
        """
        return [self.index, self.columns]

    @property
    def shape(self) -> tuple[int]:
        """
        Return a tuple representing the dimensionality of the DataFrame.

        Unlike the `len()` method, which only returns the number of rows, `shape`
        provides both row and column counts, making it a more informative method for
        understanding dataset size.

        See Also
        --------
        numpy.ndarray.shape : Tuple of array dimensions.

        Examples
        --------
        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df.shape
        (2, 2)

        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
        >>> df.shape
        (2, 3)
        """
        return (len(self.index), len(self.columns))

    @property
    def _is_homogeneous_type(self) -> bool:
        """
        Whether all the columns in a DataFrame have the same type.

        Returns
        -------
        bool

        Examples
        --------
        >>> DataFrame({"A": [1, 2], "B": [3, 4]})._is_homogeneous_type
        True
        >>> DataFrame({"A": [1, 2], "B": [3.0, 4.0]})._is_homogeneous_type
        False

        Items with the same type but different sizes are considered
        different types.

        >>> DataFrame(
        ...     {
        ...         "A": np.array([1, 2], dtype=np.int32),
        ...         "B": np.array([1, 2], dtype=np.int64),
        ...     }
        ... )._is_homogeneous_type
        False
        """
        return len({block.values.dtype for block in self._mgr.blocks}) <= 1

    @property
    def _can_fast_transpose(self) -> bool:
        """
        Can we transpose this DataFrame without creating any new array objects.
        """
        blocks = self._mgr.blocks
        if len(blocks) != 1:
            return False
        dtype = blocks[0].dtype
        return not is_1d_only_ea_dtype(dtype)

    @property
    def _values(self):
        """
        Analogue to ._values that may return a 2D ExtensionArray.
        """
        mgr = self._mgr
        blocks = mgr.blocks
        if len(blocks) != 1:
            return ensure_wrapped_if_datetimelike(self.values)
        arr = blocks[0].values
        if arr.ndim == 1:
            return self.values
        arr = cast('np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray', arr)
        return arr.T

    def _repr_fits_vertical_(self) -> bool:
        """
        Check length against max_rows.
        """
        max_rows = get_option('display.max_rows')
        return len(self) <= max_rows

    def _repr_fits_horizontal_(self) -> bool:
        """
        Check if full repr fits in horizontal boundaries imposed by the display
        options width and max_columns.
        """
        width, height = console.get_console_size()
        max_columns = get_option('display.max_columns')
        nb_columns = len(self.columns)
        if max_columns and nb_columns > max_columns or (width and nb_columns > width // 2):
            return False
        if width is None or not console.in_interactive_session():
            return True
        if get_option('display.width') is not None or console.in_ipython_frontend():
            max_rows = 1
        else:
            max_rows = get_option('display.max_rows')
        buf = StringIO()
        d = self
        if max_rows is not None:
            d = d.iloc[:min(max_rows, len(d))]
        else:
            return True
        d.to_string(buf=buf)
        value = buf.getvalue()
        repr_width = max((len(line) for line in value.split('\n')))
        return repr_width < width

    def _info_repr(self) -> bool:
        """
        True if the repr should show the info view.
        """
        info_repr_option = get_option('display.large_repr') == 'info'
        return info_repr_option and (not (self._repr_fits_horizontal_() and self._repr_fits_vertical_()))

    def __repr__(self):
        """
        Return a string representation for a particular DataFrame.
        """
        if self._info_repr():
            buf = StringIO()
            self.info(buf=buf)
            return buf.getvalue()
        repr_params = fmt.get_dataframe_repr_params()
        return self.to_string(**repr_params)

    def _repr_html_(self) -> Union[typing.Text, None]:
        """
        Return a html representation for a particular DataFrame.

        Mainly for IPython notebook.
        """
        if self._info_repr():
            buf = StringIO()
            self.info(buf=buf)
            val = buf.getvalue().replace('<', '&lt;', 1)
            val = val.replace('>', '&gt;', 1)
            return f'<pre>{val}</pre>'
        if get_option('display.notebook_repr_html'):
            max_rows = get_option('display.max_rows')
            min_rows = get_option('display.min_rows')
            max_cols = get_option('display.max_columns')
            show_dimensions = get_option('display.show_dimensions')
            show_floats = get_option('display.float_format')
            formatter = fmt.DataFrameFormatter(self, columns=None, col_space=None, na_rep='NaN', formatters=None, float_format=show_floats, sparsify=None, justify=None, index_names=True, header=True, index=True, bold_rows=True, escape=True, max_rows=max_rows, min_rows=min_rows, max_cols=max_cols, show_dimensions=show_dimensions, decimal='.')
            return fmt.DataFrameRenderer(formatter).to_html(notebook=True)
        else:
            return None

    @overload
    def to_string(self, buf: Any=..., *, columns: Any=..., col_space: Any=..., header: Any=..., index: Any=..., na_rep: Any=..., formatters: Any=..., float_format: Any=..., sparsify: Any=..., index_names: Any=..., justify: Any=..., max_rows: Any=..., max_cols: Any=..., show_dimensions: Any=..., decimal: Any=..., line_width: Any=..., min_rows: Any=..., max_colwidth: Any=..., encoding: Any=...) -> None:
        ...

    @overload
    def to_string(self, buf: Any, *, columns: Any=..., col_space: Any=..., header: Any=..., index: Any=..., na_rep: Any=..., formatters: Any=..., float_format: Any=..., sparsify: Any=..., index_names: Any=..., justify: Any=..., max_rows: Any=..., max_cols: Any=..., show_dimensions: Any=..., decimal: Any=..., line_width: Any=..., min_rows: Any=..., max_colwidth: Any=..., encoding: Any=...) -> None:
        ...

    @Substitution(header_type='bool or list of str', header='Write out the column names. If a list of columns is given, it is assumed to be aliases for the column names', col_space_type='int, list or dict of int', col_space='The minimum width of each column. If a list of ints is given every integers corresponds with one column. If a dict is given, the key references the column, while the value defines the space to use.')
    @Substitution(shared_params=fmt.common_docstring, returns=fmt.return_docstring)
    def to_string(self, buf: Any=None, *, columns: Any=None, col_space: Any=None, header: Any=True, index: Any=True, na_rep: Any='NaN', formatters: Any=None, float_format: Any=None, sparsify: Any=None, index_names: Any=True, justify: Any=None, max_rows: Any=None, max_cols: Any=None, show_dimensions: Any=False, decimal: Any='.', line_width: Any=None, min_rows: Any=None, max_colwidth: Any=None, encoding: Any=None) -> None:
        """
        Render a DataFrame to a console-friendly tabular output.
        %(shared_params)s
        line_width : int, optional
            Width to wrap a line in characters.
        min_rows : int, optional
            The number of rows to display in the console in a truncated repr
            (when number of rows is above `max_rows`).
        max_colwidth : int, optional
            Max width to truncate each column in characters. By default, no limit.
        encoding : str, default "utf-8"
            Set character encoding.
        %(returns)s
        See Also
        --------
        to_html : Convert DataFrame to HTML.

        Examples
        --------
        >>> d = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        >>> df = pd.DataFrame(d)
        >>> print(df.to_string())
           col1  col2
        0     1     4
        1     2     5
        2     3     6
        """
        from pandas import option_context
        with option_context('display.max_colwidth', max_colwidth):
            formatter = fmt.DataFrameFormatter(self, columns=columns, col_space=col_space, na_rep=na_rep, formatters=formatters, float_format=float_format, sparsify=sparsify, justify=justify, index_names=index_names, header=header, index=index, min_rows=min_rows, max_rows=max_rows, max_cols=max_cols, show_dimensions=show_dimensions, decimal=decimal)
            return fmt.DataFrameRenderer(formatter).to_string(buf=buf, encoding=encoding, line_width=line_width)

    def _get_values_for_csv(self, *, float_format: Any, date_format: Any, decimal: Any, na_rep: Any, quoting: Any):
        mgr = self._mgr.get_values_for_csv(float_format=float_format, date_format=date_format, decimal=decimal, na_rep=na_rep, quoting=quoting)
        return self._constructor_from_mgr(mgr, axes=mgr.axes)

    @property
    def style(self) -> Styler:
        """
        Returns a Styler object.

        Contains methods for building a styled HTML representation of the DataFrame.

        See Also
        --------
        io.formats.style.Styler : Helps style a DataFrame or Series according to the
            data with HTML and CSS.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3]})
        >>> df.style  # doctest: +SKIP

        Please see
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
        has_jinja2 = import_optional_dependency('jinja2', errors='ignore')
        if not has_jinja2:
            raise AttributeError("The '.style' accessor requires jinja2")
        from pandas.io.formats.style import Styler
        return Styler(self)
    _shared_docs['items'] = "\n        Iterate over (column name, Series) pairs.\n\n        Iterates over the DataFrame columns, returning a tuple with\n        the column name and the content as a Series.\n\n        Yields\n        ------\n        label : object\n            The column names for the DataFrame being iterated over.\n        content : Series\n            The column entries belonging to each label, as a Series.\n\n        See Also\n        --------\n        DataFrame.iterrows : Iterate over DataFrame rows as\n            (index, Series) pairs.\n        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples\n            of the values.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'species': ['bear', 'bear', 'marsupial'],\n        ...                   'population': [1864, 22000, 80000]},\n        ...                   index=['panda', 'polar', 'koala'])\n        >>> df\n                species   population\n        panda   bear      1864\n        polar   bear      22000\n        koala   marsupial 80000\n        >>> for label, content in df.items():\n        ...     print(f'label: {label}')\n        ...     print(f'content: {content}', sep='\\n')\n        ...\n        label: species\n        content:\n        panda         bear\n        polar         bear\n        koala    marsupial\n        Name: species, dtype: object\n        label: population\n        content:\n        panda     1864\n        polar    22000\n        koala    80000\n        Name: population, dtype: int64\n        "

    @Appender(_shared_docs['items'])
    def items(self) -> typing.Generator[tuple]:
        for i, k in enumerate(self.columns):
            yield (k, self._ixs(i, axis=1))

    def iterrows(self) -> typing.Generator[tuple]:
        """
        Iterate over DataFrame rows as (index, Series) pairs.

        Yields
        ------
        index : label or tuple of label
            The index of the row. A tuple for a `MultiIndex`.
        data : Series
            The data of the row as a Series.

        See Also
        --------
        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples of the values.
        DataFrame.items : Iterate over (column name, Series) pairs.

        Notes
        -----
        1. Because ``iterrows`` returns a Series for each row,
           it does **not** preserve dtypes across the rows (dtypes are
           preserved across columns for DataFrames).

           To preserve dtypes while iterating over the rows, it is better
           to use :meth:`itertuples` which returns namedtuples of the values
           and which is generally faster than ``iterrows``.

        2. You should **never modify** something you are iterating over.
           This is not guaranteed to work in all cases. Depending on the
           data types, the iterator returns a copy and not a view, and writing
           to it will have no effect.

        Examples
        --------

        >>> df = pd.DataFrame([[1, 1.5]], columns=["int", "float"])
        >>> row = next(df.iterrows())[1]
        >>> row
        int      1.0
        float    1.5
        Name: 0, dtype: float64
        >>> print(row["int"].dtype)
        float64
        >>> print(df["int"].dtype)
        int64
        """
        columns = self.columns
        klass = self._constructor_sliced
        for k, v in zip(self.index, self.values):
            s = klass(v, index=columns, name=k).__finalize__(self)
            if self._mgr.is_single_block:
                s._mgr.add_references(self._mgr)
            yield (k, s)

    def itertuples(self, index: bool=True, name: typing.Text='Pandas'):
        """
        Iterate over DataFrame rows as namedtuples.

        Parameters
        ----------
        index : bool, default True
            If True, return the index as the first element of the tuple.
        name : str or None, default "Pandas"
            The name of the returned namedtuples or None to return regular
            tuples.

        Returns
        -------
        iterator
            An object to iterate over namedtuples for each row in the
            DataFrame with the first field possibly being the index and
            following fields being the column values.

        See Also
        --------
        DataFrame.iterrows : Iterate over DataFrame rows as (index, Series)
            pairs.
        DataFrame.items : Iterate over (column name, Series) pairs.

        Notes
        -----
        The column names will be renamed to positional names if they are
        invalid Python identifiers, repeated, or start with an underscore.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"num_legs": [4, 2], "num_wings": [0, 2]}, index=["dog", "hawk"]
        ... )
        >>> df
              num_legs  num_wings
        dog          4          0
        hawk         2          2
        >>> for row in df.itertuples():
        ...     print(row)
        Pandas(Index='dog', num_legs=4, num_wings=0)
        Pandas(Index='hawk', num_legs=2, num_wings=2)

        By setting the `index` parameter to False we can remove the index
        as the first element of the tuple:

        >>> for row in df.itertuples(index=False):
        ...     print(row)
        Pandas(num_legs=4, num_wings=0)
        Pandas(num_legs=2, num_wings=2)

        With the `name` parameter set we set a custom name for the yielded
        namedtuples:

        >>> for row in df.itertuples(name="Animal"):
        ...     print(row)
        Animal(Index='dog', num_legs=4, num_wings=0)
        Animal(Index='hawk', num_legs=2, num_wings=2)
        """
        arrays = []
        fields = list(self.columns)
        if index:
            arrays.append(self.index)
            fields.insert(0, 'Index')
        arrays.extend((self.iloc[:, k] for k in range(len(self.columns))))
        if name is not None:
            itertuple = collections.namedtuple(name, fields, rename=True)
            return map(itertuple._make, zip(*arrays))
        return zip(*arrays)

    def __len__(self) -> int:
        """
        Returns length of info axis, but here we use the index.
        """
        return len(self.index)

    @overload
    def dot(self, other: Any) -> None:
        ...

    @overload
    def dot(self, other: Any) -> None:
        ...

    def dot(self, other: Any) -> None:
        """
        Compute the matrix multiplication between the DataFrame and other.

        This method computes the matrix product between the DataFrame and the
        values of an other Series, DataFrame or a numpy array.

        It can also be called using ``self @ other``.

        Parameters
        ----------
        other : Series, DataFrame or array-like
            The other object to compute the matrix product with.

        Returns
        -------
        Series or DataFrame
            If other is a Series, return the matrix product between self and
            other as a Series. If other is a DataFrame or a numpy.array, return
            the matrix product of self and other in a DataFrame of a np.array.

        See Also
        --------
        Series.dot: Similar method for Series.

        Notes
        -----
        The dimensions of DataFrame and other must be compatible in order to
        compute the matrix multiplication. In addition, the column names of
        DataFrame and the index of other must contain the same values, as they
        will be aligned prior to the multiplication.

        The dot method for Series computes the inner product, instead of the
        matrix product here.

        Examples
        --------
        Here we multiply a DataFrame with a Series.

        >>> df = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
        >>> s = pd.Series([1, 1, 2, 1])
        >>> df.dot(s)
        0    -4
        1     5
        dtype: int64

        Here we multiply a DataFrame with another DataFrame.

        >>> other = pd.DataFrame([[0, 1], [1, 2], [-1, -1], [2, 0]])
        >>> df.dot(other)
            0   1
        0   1   4
        1   2   2

        Note that the dot method give the same result as @

        >>> df @ other
            0   1
        0   1   4
        1   2   2

        The dot method works also if other is an np.array.

        >>> arr = np.array([[0, 1], [1, 2], [-1, -1], [2, 0]])
        >>> df.dot(arr)
            0   1
        0   1   4
        1   2   2

        Note how shuffling of the objects does not change the result.

        >>> s2 = s.reindex([1, 0, 2, 3])
        >>> df.dot(s2)
        0    -4
        1     5
        dtype: int64
        """
        if isinstance(other, (Series, DataFrame)):
            common = self.columns.union(other.index)
            if len(common) > len(self.columns) or len(common) > len(other.index):
                raise ValueError('matrices are not aligned')
            left = self.reindex(columns=common)
            right = other.reindex(index=common)
            lvals = left.values
            rvals = right._values
        else:
            left = self
            lvals = self.values
            rvals = np.asarray(other)
            if lvals.shape[1] != rvals.shape[0]:
                raise ValueError(f'Dot product shape mismatch, {lvals.shape} vs {rvals.shape}')
        if isinstance(other, DataFrame):
            common_type = find_common_type(list(self.dtypes) + list(other.dtypes))
            return self._constructor(np.dot(lvals, rvals), index=left.index, columns=other.columns, copy=False, dtype=common_type)
        elif isinstance(other, Series):
            common_type = find_common_type(list(self.dtypes) + [other.dtypes])
            return self._constructor_sliced(np.dot(lvals, rvals), index=left.index, copy=False, dtype=common_type)
        elif isinstance(rvals, (np.ndarray, Index)):
            result = np.dot(lvals, rvals)
            if result.ndim == 2:
                return self._constructor(result, index=left.index, copy=False)
            else:
                return self._constructor_sliced(result, index=left.index, copy=False)
        else:
            raise TypeError(f'unsupported type: {type(other)}')

    @overload
    def __matmul__(self, other: Any) -> None:
        ...

    @overload
    def __matmul__(self, other: Any) -> None:
        ...

    def __matmul__(self, other: Any) -> None:
        """
        Matrix multiplication using binary `@` operator.
        """
        return self.dot(other)

    def __rmatmul__(self, other: Any):
        """
        Matrix multiplication using binary `@` operator.
        """
        try:
            return self.T.dot(np.transpose(other)).T
        except ValueError as err:
            if 'shape mismatch' not in str(err):
                raise
            msg = f'shapes {np.shape(other)} and {self.shape} not aligned'
            raise ValueError(msg) from err

    @classmethod
    def from_dict(cls: Any, data: Any, orient: typing.Text='columns', dtype: None=None, columns: None=None):
        """
        Construct DataFrame from dict of array-like or dicts.

        Creates DataFrame object from dictionary by columns or by index
        allowing dtype specification.

        Parameters
        ----------
        data : dict
            Of the form {field : array-like} or {field : dict}.
        orient : {'columns', 'index', 'tight'}, default 'columns'
            The "orientation" of the data. If the keys of the passed dict
            should be the columns of the resulting DataFrame, pass 'columns'
            (default). Otherwise if the keys should be rows, pass 'index'.
            If 'tight', assume a dict with keys ['index', 'columns', 'data',
            'index_names', 'column_names'].

            .. versionadded:: 1.4.0
               'tight' as an allowed value for the ``orient`` argument

        dtype : dtype, default None
            Data type to force after DataFrame construction, otherwise infer.
        columns : list, default None
            Column labels to use when ``orient='index'``. Raises a ValueError
            if used with ``orient='columns'`` or ``orient='tight'``.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.from_records : DataFrame from structured ndarray, sequence
            of tuples or dicts, or DataFrame.
        DataFrame : DataFrame object creation using constructor.
        DataFrame.to_dict : Convert the DataFrame to a dictionary.

        Examples
        --------
        By default the keys of the dict become the DataFrame columns:

        >>> data = {"col_1": [3, 2, 1, 0], "col_2": ["a", "b", "c", "d"]}
        >>> pd.DataFrame.from_dict(data)
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d

        Specify ``orient='index'`` to create the DataFrame using dictionary
        keys as rows:

        >>> data = {"row_1": [3, 2, 1, 0], "row_2": ["a", "b", "c", "d"]}
        >>> pd.DataFrame.from_dict(data, orient="index")
               0  1  2  3
        row_1  3  2  1  0
        row_2  a  b  c  d

        When using the 'index' orientation, the column names can be
        specified manually:

        >>> pd.DataFrame.from_dict(data, orient="index", columns=["A", "B", "C", "D"])
               A  B  C  D
        row_1  3  2  1  0
        row_2  a  b  c  d

        Specify ``orient='tight'`` to create the DataFrame using a 'tight'
        format:

        >>> data = {
        ...     "index": [("a", "b"), ("a", "c")],
        ...     "columns": [("x", 1), ("y", 2)],
        ...     "data": [[1, 3], [2, 4]],
        ...     "index_names": ["n1", "n2"],
        ...     "column_names": ["z1", "z2"],
        ... }
        >>> pd.DataFrame.from_dict(data, orient="tight")
        z1     x  y
        z2     1  2
        n1 n2
        a  b   1  3
           c   2  4
        """
        index = None
        orient = orient.lower()
        if orient == 'index':
            if len(data) > 0:
                if isinstance(next(iter(data.values())), (Series, dict)):
                    data = _from_nested_dict(data)
                else:
                    index = list(data.keys())
                    data = list(data.values())
        elif orient in ('columns', 'tight'):
            if columns is not None:
                raise ValueError(f"cannot use columns parameter with orient='{orient}'")
        else:
            raise ValueError(f"Expected 'index', 'columns' or 'tight' for orient parameter. Got '{orient}' instead")
        if orient != 'tight':
            return cls(data, index=index, columns=columns, dtype=dtype)
        else:
            realdata = data['data']

            def create_index(indexlist: Any, namelist: Any) -> Index:
                if len(namelist) > 1:
                    index = MultiIndex.from_tuples(indexlist, names=namelist)
                else:
                    index = Index(indexlist, name=namelist[0])
                return index
            index = create_index(data['index'], data['index_names'])
            columns = create_index(data['columns'], data['column_names'])
            return cls(realdata, index=index, columns=columns, dtype=dtype)

    def to_numpy(self, dtype: None=None, copy: bool=False, na_value: Any=lib.no_default):
        """
        Convert the DataFrame to a NumPy array.

        By default, the dtype of the returned array will be the common NumPy
        dtype of all types in the DataFrame. For example, if the dtypes are
        ``float16`` and ``float32``, the results dtype will be ``float32``.
        This may require copying data and coercing values, which may be
        expensive.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the dtypes of the DataFrame columns.

        Returns
        -------
        numpy.ndarray
            The NumPy array representing the values in the DataFrame.

        See Also
        --------
        Series.to_numpy : Similar method for Series.

        Examples
        --------
        >>> pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_numpy()
        array([[1, 3],
               [2, 4]])

        With heterogeneous data, the lowest common type will have to
        be used.

        >>> df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.5]})
        >>> df.to_numpy()
        array([[1. , 3. ],
               [2. , 4.5]])

        For a mix of numeric and non-numeric types, the output array will
        have object dtype.

        >>> df["C"] = pd.date_range("2000", periods=2)
        >>> df.to_numpy()
        array([[1, 3.0, Timestamp('2000-01-01 00:00:00')],
               [2, 4.5, Timestamp('2000-01-02 00:00:00')]], dtype=object)
        """
        if dtype is not None:
            dtype = np.dtype(dtype)
        result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
        if result.dtype is not dtype:
            result = np.asarray(result, dtype=dtype)
        return result

    @overload
    def to_dict(self, orient: Any=..., *, into: Any, index: Any=...) -> None:
        ...

    @overload
    def to_dict(self, orient: Any, *, into: Any, index: Any=...) -> None:
        ...

    @overload
    def to_dict(self, orient: Any=..., *, into: Any=..., index: Any=...) -> None:
        ...

    @overload
    def to_dict(self, orient: Any, *, into: Any=..., index: Any=...) -> None:
        ...

    def to_dict(self, orient: Any='dict', *, into: Any=dict, index: Any=True) -> None:
        """
        Convert the DataFrame to a dictionary.

        The type of the key-value pairs can be customized with the parameters
        (see below).

        Parameters
        ----------
        orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
            Determines the type of the values of the dictionary.

            - 'dict' (default) : dict like {column -> {index -> value}}
            - 'list' : dict like {column -> [values]}
            - 'series' : dict like {column -> Series(values)}
            - 'split' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
            - 'tight' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values],
              'index_names' -> [index.names], 'column_names' -> [column.names]}
            - 'records' : list like
              [{column -> value}, ... , {column -> value}]
            - 'index' : dict like {index -> {column -> value}}

            .. versionadded:: 1.4.0
                'tight' as an allowed value for the ``orient`` argument

        into : class, default dict
            The collections.abc.MutableMapping subclass used for all Mappings
            in the return value.  Can be the actual class or an empty
            instance of the mapping type you want.  If you want a
            collections.defaultdict, you must pass it initialized.

        index : bool, default True
            Whether to include the index item (and index_names item if `orient`
            is 'tight') in the returned dictionary. Can only be ``False``
            when `orient` is 'split' or 'tight'. Note that when `orient` is
            'records', this parameter does not take effect (index item always
            not included).

            .. versionadded:: 2.0.0

        Returns
        -------
        dict, list or collections.abc.MutableMapping
            Return a collections.abc.MutableMapping object representing the
            DataFrame. The resulting transformation depends on the `orient`
            parameter.

        See Also
        --------
        DataFrame.from_dict: Create a DataFrame from a dictionary.
        DataFrame.to_json: Convert a DataFrame to JSON format.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"col1": [1, 2], "col2": [0.5, 0.75]}, index=["row1", "row2"]
        ... )
        >>> df
              col1  col2
        row1     1  0.50
        row2     2  0.75
        >>> df.to_dict()
        {'col1': {'row1': 1, 'row2': 2}, 'col2': {'row1': 0.5, 'row2': 0.75}}

        You can specify the return orientation.

        >>> df.to_dict("series")
        {'col1': row1    1
                 row2    2
        Name: col1, dtype: int64,
        'col2': row1    0.50
                row2    0.75
        Name: col2, dtype: float64}

        >>> df.to_dict("split")
        {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],
         'data': [[1, 0.5], [2, 0.75]]}

        >>> df.to_dict("records")
        [{'col1': 1, 'col2': 0.5}, {'col1': 2, 'col2': 0.75}]

        >>> df.to_dict("index")
        {'row1': {'col1': 1, 'col2': 0.5}, 'row2': {'col1': 2, 'col2': 0.75}}

        >>> df.to_dict("tight")
        {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],
         'data': [[1, 0.5], [2, 0.75]], 'index_names': [None], 'column_names': [None]}

        You can also specify the mapping type.

        >>> from collections import OrderedDict, defaultdict
        >>> df.to_dict(into=OrderedDict)
        OrderedDict([('col1', OrderedDict([('row1', 1), ('row2', 2)])),
                     ('col2', OrderedDict([('row1', 0.5), ('row2', 0.75)]))])

        If you want a `defaultdict`, you need to initialize it:

        >>> dd = defaultdict(list)
        >>> df.to_dict("records", into=dd)
        [defaultdict(<class 'list'>, {'col1': 1, 'col2': 0.5}),
         defaultdict(<class 'list'>, {'col1': 2, 'col2': 0.75})]
        """
        from pandas.core.methods.to_dict import to_dict
        return to_dict(self, orient, into=into, index=index)

    @classmethod
    def from_records(cls: Any, data: Any, index: None=None, exclude: None=None, columns: None=None, coerce_float: bool=False, nrows: None=None):
        """
        Convert structured or record ndarray to DataFrame.

        Creates a DataFrame object from a structured ndarray, or sequence of
        tuples or dicts.

        Parameters
        ----------
        data : structured ndarray, sequence of tuples or dicts
            Structured input data.
        index : str, list of fields, array-like
            Field of array to use as the index, alternately a specific set of
            input labels to use.
        exclude : sequence, default None
            Columns or fields to exclude.
        columns : sequence, default None
            Column names to use. If the passed data do not have names
            associated with them, this argument provides names for the
            columns. Otherwise, this argument indicates the order of the columns
            in the result (any names not found in the data will become all-NA
            columns) and limits the data to these columns if not all column names
            are provided.
        coerce_float : bool, default False
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        nrows : int, default None
            Number of rows to read if data is an iterator.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.from_dict : DataFrame from dict of array-like or dicts.
        DataFrame : DataFrame object creation using constructor.

        Examples
        --------
        Data can be provided as a structured ndarray:

        >>> data = np.array(
        ...     [(3, "a"), (2, "b"), (1, "c"), (0, "d")],
        ...     dtype=[("col_1", "i4"), ("col_2", "U1")],
        ... )
        >>> pd.DataFrame.from_records(data)
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d

        Data can be provided as a list of dicts:

        >>> data = [
        ...     {"col_1": 3, "col_2": "a"},
        ...     {"col_1": 2, "col_2": "b"},
        ...     {"col_1": 1, "col_2": "c"},
        ...     {"col_1": 0, "col_2": "d"},
        ... ]
        >>> pd.DataFrame.from_records(data)
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d

        Data can be provided as a list of tuples with corresponding columns:

        >>> data = [(3, "a"), (2, "b"), (1, "c"), (0, "d")]
        >>> pd.DataFrame.from_records(data, columns=["col_1", "col_2"])
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d
        """
        if isinstance(data, DataFrame):
            raise TypeError('Passing a DataFrame to DataFrame.from_records is not supported. Use set_index and/or drop to modify the DataFrame instead.')
        result_index = None
        if columns is not None:
            columns = ensure_index(columns)

        def maybe_reorder(arrays: Any, arr_columns: Any, columns: Any, index: Any) -> tuple[None]:
            """
            If our desired 'columns' do not match the data's pre-existing 'arr_columns',
            we re-order our arrays.  This is like a preemptive (cheap) reindex.
            """
            if len(arrays):
                length = len(arrays[0])
            else:
                length = 0
            result_index = None
            if len(arrays) == 0 and index is None and (length == 0):
                result_index = default_index(0)
            arrays, arr_columns = reorder_arrays(arrays, arr_columns, columns, length)
            return (arrays, arr_columns, result_index)
        if is_iterator(data):
            if nrows == 0:
                return cls()
            try:
                first_row = next(data)
            except StopIteration:
                return cls(index=index, columns=columns)
            dtype = None
            if hasattr(first_row, 'dtype') and first_row.dtype.names:
                dtype = first_row.dtype
            values = [first_row]
            if nrows is None:
                values += data
            else:
                values.extend(itertools.islice(data, nrows - 1))
            if dtype is not None:
                data = np.array(values, dtype=dtype)
            else:
                data = values
        if isinstance(data, dict):
            if columns is None:
                columns = arr_columns = ensure_index(sorted(data))
                arrays = [data[k] for k in columns]
            else:
                arrays = []
                arr_columns_list = []
                for k, v in data.items():
                    if k in columns:
                        arr_columns_list.append(k)
                        arrays.append(v)
                arr_columns = Index(arr_columns_list)
                arrays, arr_columns, result_index = maybe_reorder(arrays, arr_columns, columns, index)
        elif isinstance(data, np.ndarray):
            arrays, columns = to_arrays(data, columns)
            arr_columns = columns
        else:
            arrays, arr_columns = to_arrays(data, columns)
            if coerce_float:
                for i, arr in enumerate(arrays):
                    if arr.dtype == object:
                        arrays[i] = lib.maybe_convert_objects(arr, try_float=True)
            arr_columns = ensure_index(arr_columns)
            if columns is None:
                columns = arr_columns
            else:
                arrays, arr_columns, result_index = maybe_reorder(arrays, arr_columns, columns, index)
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)
        if index is not None:
            if isinstance(index, str) or not hasattr(index, '__iter__'):
                i = columns.get_loc(index)
                exclude.add(index)
                if len(arrays) > 0:
                    result_index = Index(arrays[i], name=index)
                else:
                    result_index = Index([], name=index)
            else:
                try:
                    index_data = [arrays[arr_columns.get_loc(field)] for field in index]
                except (KeyError, TypeError):
                    result_index = index
                else:
                    result_index = ensure_index_from_sequences(index_data, names=index)
                    exclude.update(index)
        if any(exclude):
            arr_exclude = (x for x in exclude if x in arr_columns)
            to_remove = {arr_columns.get_loc(col) for col in arr_exclude}
            arrays = [v for i, v in enumerate(arrays) if i not in to_remove]
            columns = columns.drop(exclude)
        mgr = arrays_to_mgr(arrays, columns, result_index)
        df = DataFrame._from_mgr(mgr, axes=mgr.axes)
        if cls is not DataFrame:
            return cls(df, copy=False)
        return df

    def to_records(self, index: bool=True, column_dtypes: Any=None, index_dtypes: Any=None):
        """
        Convert DataFrame to a NumPy record array.

        Index will be included as the first field of the record array if
        requested.

        Parameters
        ----------
        index : bool, default True
            Include index in resulting record array, stored in 'index'
            field or using the index label, if set.
        column_dtypes : str, type, dict, default None
            If a string or type, the data type to store all columns. If
            a dictionary, a mapping of column names and indices (zero-indexed)
            to specific data types.
        index_dtypes : str, type, dict, default None
            If a string or type, the data type to store all index levels. If
            a dictionary, a mapping of index level names and indices
            (zero-indexed) to specific data types.

            This mapping is applied only if `index=True`.

        Returns
        -------
        numpy.rec.recarray
            NumPy ndarray with the DataFrame labels as fields and each row
            of the DataFrame as entries.

        See Also
        --------
        DataFrame.from_records: Convert structured or record ndarray
            to DataFrame.
        numpy.rec.recarray: An ndarray that allows field access using
            attributes, analogous to typed columns in a
            spreadsheet.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [0.5, 0.75]}, index=["a", "b"])
        >>> df
           A     B
        a  1  0.50
        b  2  0.75
        >>> df.to_records()
        rec.array([('a', 1, 0.5 ), ('b', 2, 0.75)],
                  dtype=[('index', 'O'), ('A', '<i8'), ('B', '<f8')])

        If the DataFrame index has no label then the recarray field name
        is set to 'index'. If the index has a label then this is used as the
        field name:

        >>> df.index = df.index.rename("I")
        >>> df.to_records()
        rec.array([('a', 1, 0.5 ), ('b', 2, 0.75)],
                  dtype=[('I', 'O'), ('A', '<i8'), ('B', '<f8')])

        The index can be excluded from the record array:

        >>> df.to_records(index=False)
        rec.array([(1, 0.5 ), (2, 0.75)],
                  dtype=[('A', '<i8'), ('B', '<f8')])

        Data types can be specified for the columns:

        >>> df.to_records(column_dtypes={"A": "int32"})
        rec.array([('a', 1, 0.5 ), ('b', 2, 0.75)],
                  dtype=[('I', 'O'), ('A', '<i4'), ('B', '<f8')])

        As well as for the index:

        >>> df.to_records(index_dtypes="<S2")
        rec.array([(b'a', 1, 0.5 ), (b'b', 2, 0.75)],
                  dtype=[('I', 'S2'), ('A', '<i8'), ('B', '<f8')])

        >>> index_dtypes = f"<S{df.index.str.len().max()}"
        >>> df.to_records(index_dtypes=index_dtypes)
        rec.array([(b'a', 1, 0.5 ), (b'b', 2, 0.75)],
                  dtype=[('I', 'S1'), ('A', '<i8'), ('B', '<f8')])
        """
        if index:
            ix_vals = [np.asarray(self.index.get_level_values(i)) for i in range(self.index.nlevels)]
            arrays = ix_vals + [np.asarray(self.iloc[:, i]) for i in range(len(self.columns))]
            index_names = list(self.index.names)
            if isinstance(self.index, MultiIndex):
                index_names = com.fill_missing_names(index_names)
            elif index_names[0] is None:
                index_names = ['index']
            names = [str(name) for name in itertools.chain(index_names, self.columns)]
        else:
            arrays = [np.asarray(self.iloc[:, i]) for i in range(len(self.columns))]
            names = [str(c) for c in self.columns]
            index_names = []
        index_len = len(index_names)
        formats = []
        for i, v in enumerate(arrays):
            index_int = i
            if index_int < index_len:
                dtype_mapping = index_dtypes
                name = index_names[index_int]
            else:
                index_int -= index_len
                dtype_mapping = column_dtypes
                name = self.columns[index_int]
            if is_dict_like(dtype_mapping):
                if name in dtype_mapping:
                    dtype_mapping = dtype_mapping[name]
                elif index_int in dtype_mapping:
                    dtype_mapping = dtype_mapping[index_int]
                else:
                    dtype_mapping = None
            if dtype_mapping is None:
                formats.append(v.dtype)
            elif isinstance(dtype_mapping, (type, np.dtype, str)):
                formats.append(dtype_mapping)
            else:
                element = 'row' if i < index_len else 'column'
                msg = f'Invalid dtype {dtype_mapping} specified for {element} {name}'
                raise ValueError(msg)
        return np.rec.fromarrays(arrays, dtype={'names': names, 'formats': formats})

    @classmethod
    def _from_arrays(cls: Any, arrays: Any, columns: Any, index: Any, dtype: None=None, verify_integrity: bool=True):
        """
        Create DataFrame from a list of arrays corresponding to the columns.

        Parameters
        ----------
        arrays : list-like of arrays
            Each array in the list corresponds to one column, in order.
        columns : list-like, Index
            The column names for the resulting DataFrame.
        index : list-like, Index
            The rows labels for the resulting DataFrame.
        dtype : dtype, optional
            Optional dtype to enforce for all arrays.
        verify_integrity : bool, default True
            Validate and homogenize all input. If set to False, it is assumed
            that all elements of `arrays` are actual arrays how they will be
            stored in a block (numpy ndarray or ExtensionArray), have the same
            length as and are aligned with the index, and that `columns` and
            `index` are ensured to be an Index object.

        Returns
        -------
        DataFrame
        """
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        columns = ensure_index(columns)
        if len(columns) != len(arrays):
            raise ValueError('len(columns) must match len(arrays)')
        mgr = arrays_to_mgr(arrays, columns, index, dtype=dtype, verify_integrity=verify_integrity)
        return cls._from_mgr(mgr, axes=mgr.axes)

    @doc(storage_options=_shared_docs['storage_options'], compression_options=_shared_docs['compression_options'] % 'path')
    def to_stata(self, path: Any, *, convert_dates: None=None, write_index: bool=True, byteorder: None=None, time_stamp: None=None, data_label: None=None, variable_labels: None=None, version: int=114, convert_strl: None=None, compression: typing.Text='infer', storage_options: None=None, value_labels: None=None) -> None:
        """
        Export DataFrame object to Stata dta format.

        Writes the DataFrame to a Stata dataset file.
        "dta" files contain a Stata dataset.

        Parameters
        ----------
        path : str, path object, or buffer
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a binary ``write()`` function.

        convert_dates : dict
            Dictionary mapping columns containing datetime types to stata
            internal format to use when writing the dates. Options are 'tc',
            'td', 'tm', 'tw', 'th', 'tq', 'ty'. Column can be either an integer
            or a name. Datetime columns that do not have a conversion type
            specified will be converted to 'tc'. Raises NotImplementedError if
            a datetime column has timezone information.
        write_index : bool
            Write the index to Stata dataset.
        byteorder : str
            Can be ">", "<", "little", or "big". default is `sys.byteorder`.
        time_stamp : datetime
            A datetime to use as file creation date.  Default is the current
            time.
        data_label : str, optional
            A label for the data set.  Must be 80 characters or smaller.
        variable_labels : dict
            Dictionary containing columns as keys and variable labels as
            values. Each label must be 80 characters or smaller.
        version : {{114, 117, 118, 119, None}}, default 114
            Version to use in the output dta file. Set to None to let pandas
            decide between 118 or 119 formats depending on the number of
            columns in the frame. Version 114 can be read by Stata 10 and
            later. Version 117 can be read by Stata 13 or later. Version 118
            is supported in Stata 14 and later. Version 119 is supported in
            Stata 15 and later. Version 114 limits string variables to 244
            characters or fewer while versions 117 and later allow strings
            with lengths up to 2,000,000 characters. Versions 118 and 119
            support Unicode characters, and version 119 supports more than
            32,767 variables.

            Version 119 should usually only be used when the number of
            variables exceeds the capacity of dta format 118. Exporting
            smaller datasets in format 119 may have unintended consequences,
            and, as of November 2020, Stata SE cannot read version 119 files.

        convert_strl : list, optional
            List of column names to convert to string columns to Stata StrL
            format. Only available if version is 117.  Storing strings in the
            StrL format can produce smaller dta files if strings have more than
            8 characters and values are repeated.
        {compression_options}

            .. versionchanged:: 1.4.0 Zstandard support.

        {storage_options}

        value_labels : dict of dicts
            Dictionary containing columns as keys and dictionaries of column value
            to labels as values. Labels for a single variable must be 32,000
            characters or smaller.

            .. versionadded:: 1.4.0

        Raises
        ------
        NotImplementedError
            * If datetimes contain timezone information
            * Column dtype is not representable in Stata
        ValueError
            * Columns listed in convert_dates are neither datetime64[ns]
              or datetime.datetime
            * Column listed in convert_dates is not in DataFrame
            * Categorical label contains more than 32,000 characters

        See Also
        --------
        read_stata : Import Stata data files.
        io.stata.StataWriter : Low-level writer for Stata data files.
        io.stata.StataWriter117 : Low-level writer for version 117 files.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [["falcon", 350], ["parrot", 18]], columns=["animal", "parrot"]
        ... )
        >>> df.to_stata("animals.dta")  # doctest: +SKIP
        """
        if version not in (114, 117, 118, 119, None):
            raise ValueError('Only formats 114, 117, 118 and 119 are supported.')
        if version == 114:
            if convert_strl is not None:
                raise ValueError('strl is not supported in format 114')
            from pandas.io.stata import StataWriter as statawriter
        elif version == 117:
            from pandas.io.stata import StataWriter117 as statawriter
        else:
            from pandas.io.stata import StataWriterUTF8 as statawriter
        kwargs = {}
        if version is None or version >= 117:
            kwargs['convert_strl'] = convert_strl
        if version is None or version >= 118:
            kwargs['version'] = version
        writer = statawriter(path, self, convert_dates=convert_dates, byteorder=byteorder, time_stamp=time_stamp, data_label=data_label, write_index=write_index, variable_labels=variable_labels, compression=compression, storage_options=storage_options, value_labels=value_labels, **kwargs)
        writer.write_file()

    def to_feather(self, path: Any, **kwargs) -> None:
        """
        Write a DataFrame to the binary Feather format.

        Parameters
        ----------
        path : str, path object, file-like object
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a binary ``write()`` function. If a string or a path,
            it will be used as Root Directory path when writing a partitioned dataset.
        **kwargs :
            Additional keywords passed to :func:`pyarrow.feather.write_feather`.
            This includes the `compression`, `compression_level`, `chunksize`
            and `version` keywords.

        See Also
        --------
        DataFrame.to_parquet : Write a DataFrame to the binary parquet format.
        DataFrame.to_excel : Write object to an Excel sheet.
        DataFrame.to_sql : Write to a sql table.
        DataFrame.to_csv : Write a csv file.
        DataFrame.to_json : Convert the object to a JSON string.
        DataFrame.to_html : Render a DataFrame as an HTML table.
        DataFrame.to_string : Convert DataFrame to a string.

        Notes
        -----
        This function writes the dataframe as a `feather file
        <https://arrow.apache.org/docs/python/feather.html>`_. Requires a default
        index. For saving the DataFrame with your custom index use a method that
        supports custom indices e.g. `to_parquet`.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        >>> df.to_feather("file.feather")  # doctest: +SKIP
        """
        from pandas.io.feather_format import to_feather
        to_feather(self, path, **kwargs)

    @overload
    def to_markdown(self, buf: Any=..., *, mode: Any=..., index: Any=..., storage_options: Any=..., **kwargs) -> None:
        ...

    @overload
    def to_markdown(self, buf: Any, *, mode: Any=..., index: Any=..., storage_options: Any=..., **kwargs) -> None:
        ...

    @overload
    def to_markdown(self, buf: Any, *, mode: Any=..., index: Any=..., storage_options: Any=..., **kwargs) -> None:
        ...

    def to_markdown(self, buf: Any=None, *, mode: Any='wt', index: Any=True, storage_options: Any=None, **kwargs) -> None:
        """
        Print DataFrame in Markdown-friendly format.

        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        mode : str, optional
            Mode in which file is opened, "wt" by default.
        index : bool, optional, default True
            Add index (row) labels.

        storage_options : dict, optional
            Extra options that make sense for a particular storage connection, e.g.
            host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
            are forwarded to ``urllib.request.Request`` as header options. For other
            URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
            forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
            details, and for more examples on storage options refer `here
            <https://pandas.pydata.org/docs/user_guide/io.html?
            highlight=storage_options#reading-writing-remote-files>`_.

        **kwargs
            These parameters will be passed to `tabulate <https://pypi.org/project/tabulate>`_.

        Returns
        -------
        str
            DataFrame in Markdown-friendly format.

        See Also
        --------
        DataFrame.to_html : Render DataFrame to HTML-formatted table.
        DataFrame.to_latex : Render DataFrame to LaTeX-formatted table.

        Notes
        -----
        Requires the `tabulate <https://pypi.org/project/tabulate>`_ package.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     data={"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]}
        ... )
        >>> print(df.to_markdown())
        |    | animal_1   | animal_2   |
        |---:|:-----------|:-----------|
        |  0 | elk        | dog        |
        |  1 | pig        | quetzal    |

        Output markdown with a tabulate option.

        >>> print(df.to_markdown(tablefmt="grid"))
        +----+------------+------------+
        |    | animal_1   | animal_2   |
        +====+============+============+
        |  0 | elk        | dog        |
        +----+------------+------------+
        |  1 | pig        | quetzal    |
        +----+------------+------------+
        """
        if 'showindex' in kwargs:
            raise ValueError("Pass 'index' instead of 'showindex")
        kwargs.setdefault('headers', 'keys')
        kwargs.setdefault('tablefmt', 'pipe')
        kwargs.setdefault('showindex', index)
        tabulate = import_optional_dependency('tabulate')
        result = tabulate.tabulate(self, **kwargs)
        if buf is None:
            return result
        with get_handle(buf, mode, storage_options=storage_options) as handles:
            handles.handle.write(result)
        return None

    @overload
    def to_parquet(self, path: Any=..., *, engine: Any=..., compression: Any=..., index: Any=..., partition_cols: Any=..., storage_options: Any=..., **kwargs) -> None:
        ...

    @overload
    def to_parquet(self, path: Any, *, engine: Any=..., compression: Any=..., index: Any=..., partition_cols: Any=..., storage_options: Any=..., **kwargs) -> None:
        ...

    @doc(storage_options=_shared_docs['storage_options'])
    def to_parquet(self, path: Any=None, *, engine: Any='auto', compression: Any='snappy', index: Any=None, partition_cols: Any=None, storage_options: Any=None, **kwargs) -> None:
        """
        Write a DataFrame to the binary parquet format.

        This function writes the dataframe as a `parquet file
        <https://parquet.apache.org/>`_. You can choose different parquet
        backends, and have the option of compression. See
        :ref:`the user guide <io.parquet>` for more details.

        Parameters
        ----------
        path : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a binary ``write()`` function. If None, the result is
            returned as bytes. If a string or path, it will be used as Root Directory
            path when writing a partitioned dataset.
        engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
            Parquet library to use. If 'auto', then the option
            ``io.parquet.engine`` is used. The default ``io.parquet.engine``
            behavior is to try 'pyarrow', falling back to 'fastparquet' if
            'pyarrow' is unavailable.
        compression : str or None, default 'snappy'
            Name of the compression to use. Use ``None`` for no compression.
            Supported options: 'snappy', 'gzip', 'brotli', 'lz4', 'zstd'.
        index : bool, default None
            If ``True``, include the dataframe's index(es) in the file output.
            If ``False``, they will not be written to the file.
            If ``None``, similar to ``True`` the dataframe's index(es)
            will be saved. However, instead of being saved as values,
            the RangeIndex will be stored as a range in the metadata so it
            doesn't require much space and is faster. Other indexes will
            be included as columns in the file output.
        partition_cols : list, optional, default None
            Column names by which to partition the dataset.
            Columns are partitioned in the order they are given.
            Must be None if path is not a string.
        {storage_options}

        **kwargs
            Additional arguments passed to the parquet library. See
            :ref:`pandas io <io.parquet>` for more details.

        Returns
        -------
        bytes if no path argument is provided else None
            Returns the DataFrame converted to the binary parquet format as bytes if no
            path argument. Returns None and writes the DataFrame to the specified
            location in the Parquet format if the path argument is provided.

        See Also
        --------
        read_parquet : Read a parquet file.
        DataFrame.to_orc : Write an orc file.
        DataFrame.to_csv : Write a csv file.
        DataFrame.to_sql : Write to a sql table.
        DataFrame.to_hdf : Write to hdf.

        Notes
        -----
        * This function requires either the `fastparquet
          <https://pypi.org/project/fastparquet>`_ or `pyarrow
          <https://arrow.apache.org/docs/python/>`_ library.
        * When saving a DataFrame with categorical columns to parquet,
          the file size may increase due to the inclusion of all possible
          categories, not just those present in the data. This behavior
          is expected and consistent with pandas' handling of categorical data.
          To manage file size and ensure a more predictable roundtrip process,
          consider using :meth:`Categorical.remove_unused_categories` on the
          DataFrame before saving.

        Examples
        --------
        >>> df = pd.DataFrame(data={{"col1": [1, 2], "col2": [3, 4]}})
        >>> df.to_parquet("df.parquet.gzip", compression="gzip")  # doctest: +SKIP
        >>> pd.read_parquet("df.parquet.gzip")  # doctest: +SKIP
           col1  col2
        0     1     3
        1     2     4

        If you want to get a buffer to the parquet content you can use a io.BytesIO
        object, as long as you don't use partition_cols, which creates multiple files.

        >>> import io
        >>> f = io.BytesIO()
        >>> df.to_parquet(f)
        >>> f.seek(0)
        0
        >>> content = f.read()
        """
        from pandas.io.parquet import to_parquet
        return to_parquet(self, path, engine, compression=compression, index=index, partition_cols=partition_cols, storage_options=storage_options, **kwargs)

    @overload
    def to_orc(self, path: Any=..., *, engine: Any=..., index: Any=..., engine_kwargs: Any=...) -> None:
        ...

    @overload
    def to_orc(self, path: Any, *, engine: Any=..., index: Any=..., engine_kwargs: Any=...) -> None:
        ...

    @overload
    def to_orc(self, path: Any, *, engine: Any=..., index: Any=..., engine_kwargs: Any=...) -> None:
        ...

    def to_orc(self, path: Any=None, *, engine: Any='pyarrow', index: Any=None, engine_kwargs: Any=None) -> None:
        """
        Write a DataFrame to the Optimized Row Columnar (ORC) format.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        path : str, file-like object or None, default None
            If a string, it will be used as Root Directory path
            when writing a partitioned dataset. By file-like object,
            we refer to objects with a write() method, such as a file handle
            (e.g. via builtin open function). If path is None,
            a bytes object is returned.
        engine : {'pyarrow'}, default 'pyarrow'
            ORC library to use.
        index : bool, optional
            If ``True``, include the dataframe's index(es) in the file output.
            If ``False``, they will not be written to the file.
            If ``None``, similar to ``infer`` the dataframe's index(es)
            will be saved. However, instead of being saved as values,
            the RangeIndex will be stored as a range in the metadata so it
            doesn't require much space and is faster. Other indexes will
            be included as columns in the file output.
        engine_kwargs : dict[str, Any] or None, default None
            Additional keyword arguments passed to :func:`pyarrow.orc.write_table`.

        Returns
        -------
        bytes if no ``path`` argument is provided else None
            Bytes object with DataFrame data if ``path`` is not specified else None.

        Raises
        ------
        NotImplementedError
            Dtype of one or more columns is category, unsigned integers, interval,
            period or sparse.
        ValueError
            engine is not pyarrow.

        See Also
        --------
        read_orc : Read a ORC file.
        DataFrame.to_parquet : Write a parquet file.
        DataFrame.to_csv : Write a csv file.
        DataFrame.to_sql : Write to a sql table.
        DataFrame.to_hdf : Write to hdf.

        Notes
        -----
        * Find more information on ORC
          `here <https://en.wikipedia.org/wiki/Apache_ORC>`__.
        * Before using this function you should read the :ref:`user guide about
          ORC <io.orc>` and :ref:`install optional dependencies <install.warn_orc>`.
        * This function requires `pyarrow <https://arrow.apache.org/docs/python/>`_
          library.
        * For supported dtypes please refer to `supported ORC features in Arrow
          <https://arrow.apache.org/docs/cpp/orc.html#data-types>`__.
        * Currently timezones in datetime columns are not preserved when a
          dataframe is converted into ORC files.

        Examples
        --------
        >>> df = pd.DataFrame(data={"col1": [1, 2], "col2": [4, 3]})
        >>> df.to_orc("df.orc")  # doctest: +SKIP
        >>> pd.read_orc("df.orc")  # doctest: +SKIP
           col1  col2
        0     1     4
        1     2     3

        If you want to get a buffer to the orc content you can write it to io.BytesIO

        >>> import io
        >>> b = io.BytesIO(df.to_orc())  # doctest: +SKIP
        >>> b.seek(0)  # doctest: +SKIP
        0
        >>> content = b.read()  # doctest: +SKIP
        """
        from pandas.io.orc import to_orc
        return to_orc(self, path, engine=engine, index=index, engine_kwargs=engine_kwargs)

    @overload
    def to_html(self, buf: Any, *, columns: Any=..., col_space: Any=..., header: Any=..., index: Any=..., na_rep: Any=..., formatters: Any=..., float_format: Any=..., sparsify: Any=..., index_names: Any=..., justify: Any=..., max_rows: Any=..., max_cols: Any=..., show_dimensions: Any=..., decimal: Any=..., bold_rows: Any=..., classes: Any=..., escape: Any=..., notebook: Any=..., border: Any=..., table_id: Any=..., render_links: Any=..., encoding: Any=...) -> None:
        ...

    @overload
    def to_html(self, buf: Any=..., *, columns: Any=..., col_space: Any=..., header: Any=..., index: Any=..., na_rep: Any=..., formatters: Any=..., float_format: Any=..., sparsify: Any=..., index_names: Any=..., justify: Any=..., max_rows: Any=..., max_cols: Any=..., show_dimensions: Any=..., decimal: Any=..., bold_rows: Any=..., classes: Any=..., escape: Any=..., notebook: Any=..., border: Any=..., table_id: Any=..., render_links: Any=..., encoding: Any=...) -> None:
        ...

    @Substitution(header_type='bool', header='Whether to print column labels, default True', col_space_type='str or int, list or dict of int or str', col_space='The minimum width of each column in CSS length units.  An int is assumed to be px units.')
    @Substitution(shared_params=fmt.common_docstring, returns=fmt.return_docstring)
    def to_html(self, buf: Any=None, *, columns: Any=None, col_space: Any=None, header: Any=True, index: Any=True, na_rep: Any='NaN', formatters: Any=None, float_format: Any=None, sparsify: Any=None, index_names: Any=True, justify: Any=None, max_rows: Any=None, max_cols: Any=None, show_dimensions: Any=False, decimal: Any='.', bold_rows: Any=True, classes: Any=None, escape: Any=True, notebook: Any=False, border: Any=None, table_id: Any=None, render_links: Any=False, encoding: Any=None) -> None:
        """
        Render a DataFrame as an HTML table.
        %(shared_params)s
        bold_rows : bool, default True
            Make the row labels bold in the output.
        classes : str or list or tuple, default None
            CSS class(es) to apply to the resulting html table.
        escape : bool, default True
            Convert the characters <, >, and & to HTML-safe sequences.
        notebook : {True, False}, default False
            Whether the generated HTML is for IPython Notebook.
        border : int or bool
            When an integer value is provided, it sets the border attribute in
            the opening tag, specifying the thickness of the border.
            If ``False`` or ``0`` is passed, the border attribute will not
            be present in the ``<table>`` tag.
            The default value for this parameter is governed by
            ``pd.options.display.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.
        encoding : str, default "utf-8"
            Set character encoding.
        %(returns)s
        See Also
        --------
        to_string : Convert DataFrame to a string.

        Examples
        --------
        >>> df = pd.DataFrame(data={"col1": [1, 2], "col2": [4, 3]})
        >>> html_string = '''<table border="1" class="dataframe">
        ...   <thead>
        ...     <tr style="text-align: right;">
        ...       <th></th>
        ...       <th>col1</th>
        ...       <th>col2</th>
        ...     </tr>
        ...   </thead>
        ...   <tbody>
        ...     <tr>
        ...       <th>0</th>
        ...       <td>1</td>
        ...       <td>4</td>
        ...     </tr>
        ...     <tr>
        ...       <th>1</th>
        ...       <td>2</td>
        ...       <td>3</td>
        ...     </tr>
        ...   </tbody>
        ... </table>'''
        >>> assert html_string == df.to_html()
        """
        if justify is not None and justify not in fmt.VALID_JUSTIFY_PARAMETERS:
            raise ValueError('Invalid value for justify parameter')
        formatter = fmt.DataFrameFormatter(self, columns=columns, col_space=col_space, na_rep=na_rep, header=header, index=index, formatters=formatters, float_format=float_format, bold_rows=bold_rows, sparsify=sparsify, justify=justify, index_names=index_names, escape=escape, decimal=decimal, max_rows=max_rows, max_cols=max_cols, show_dimensions=show_dimensions)
        return fmt.DataFrameRenderer(formatter).to_html(buf=buf, classes=classes, notebook=notebook, border=border, encoding=encoding, table_id=table_id, render_links=render_links)

    @overload
    def to_xml(self, path_or_buffer: Any=..., *, index: Any=..., root_name: Any=..., row_name: Any=..., na_rep: Any=..., attr_cols: Any=..., elem_cols: Any=..., namespaces: Any=..., prefix: Any=..., encoding: Any=..., xml_declaration: Any=..., pretty_print: Any=..., parser: Any=..., stylesheet: Any=..., compression: Any=..., storage_options: Any=...) -> None:
        ...

    @overload
    def to_xml(self, path_or_buffer: Any, *, index: Any=..., root_name: Any=..., row_name: Any=..., na_rep: Any=..., attr_cols: Any=..., elem_cols: Any=..., namespaces: Any=..., prefix: Any=..., encoding: Any=..., xml_declaration: Any=..., pretty_print: Any=..., parser: Any=..., stylesheet: Any=..., compression: Any=..., storage_options: Any=...) -> None:
        ...

    @doc(storage_options=_shared_docs['storage_options'], compression_options=_shared_docs['compression_options'] % 'path_or_buffer')
    def to_xml(self, path_or_buffer: Any=None, *, index: Any=True, root_name: Any='data', row_name: Any='row', na_rep: Any=None, attr_cols: Any=None, elem_cols: Any=None, namespaces: Any=None, prefix: Any=None, encoding: Any='utf-8', xml_declaration: Any=True, pretty_print: Any=True, parser: Any='lxml', stylesheet: Any=None, compression: Any='infer', storage_options: Any=None) -> None:
        """
        Render a DataFrame to an XML document.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        path_or_buffer : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a ``write()`` function. If None, the result is returned
            as a string.
        index : bool, default True
            Whether to include index in XML document.
        root_name : str, default 'data'
            The name of root element in XML document.
        row_name : str, default 'row'
            The name of row element in XML document.
        na_rep : str, optional
            Missing data representation.
        attr_cols : list-like, optional
            List of columns to write as attributes in row element.
            Hierarchical columns will be flattened with underscore
            delimiting the different levels.
        elem_cols : list-like, optional
            List of columns to write as children in row element. By default,
            all columns output as children of row element. Hierarchical
            columns will be flattened with underscore delimiting the
            different levels.
        namespaces : dict, optional
            All namespaces to be defined in root element. Keys of dict
            should be prefix names and values of dict corresponding URIs.
            Default namespaces should be given empty string key. For
            example, ::

                namespaces = {{"": "https://example.com"}}

        prefix : str, optional
            Namespace prefix to be used for every element and/or attribute
            in document. This should be one of the keys in ``namespaces``
            dict.
        encoding : str, default 'utf-8'
            Encoding of the resulting document.
        xml_declaration : bool, default True
            Whether to include the XML declaration at start of document.
        pretty_print : bool, default True
            Whether output should be pretty printed with indentation and
            line breaks.
        parser : {{'lxml','etree'}}, default 'lxml'
            Parser module to use for building of tree. Only 'lxml' and
            'etree' are supported. With 'lxml', the ability to use XSLT
            stylesheet is supported.
        stylesheet : str, path object or file-like object, optional
            A URL, file-like object, or a raw string containing an XSLT
            script used to transform the raw XML output. Script should use
            layout of elements and attributes from original output. This
            argument requires ``lxml`` to be installed. Only XSLT 1.0
            scripts and not later versions is currently supported.
        {compression_options}

            .. versionchanged:: 1.4.0 Zstandard support.

        {storage_options}

        Returns
        -------
        None or str
            If ``io`` is None, returns the resulting XML format as a
            string. Otherwise returns None.

        See Also
        --------
        to_json : Convert the pandas object to a JSON string.
        to_html : Convert DataFrame to a html.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [["square", 360, 4], ["circle", 360, np.nan], ["triangle", 180, 3]],
        ...     columns=["shape", "degrees", "sides"],
        ... )

        >>> df.to_xml()  # doctest: +SKIP
        <?xml version='1.0' encoding='utf-8'?>
        <data>
          <row>
            <index>0</index>
            <shape>square</shape>
            <degrees>360</degrees>
            <sides>4.0</sides>
          </row>
          <row>
            <index>1</index>
            <shape>circle</shape>
            <degrees>360</degrees>
            <sides/>
          </row>
          <row>
            <index>2</index>
            <shape>triangle</shape>
            <degrees>180</degrees>
            <sides>3.0</sides>
          </row>
        </data>

        >>> df.to_xml(
        ...     attr_cols=["index", "shape", "degrees", "sides"]
        ... )  # doctest: +SKIP
        <?xml version='1.0' encoding='utf-8'?>
        <data>
          <row index="0" shape="square" degrees="360" sides="4.0"/>
          <row index="1" shape="circle" degrees="360"/>
          <row index="2" shape="triangle" degrees="180" sides="3.0"/>
        </data>

        >>> df.to_xml(
        ...     namespaces={{"doc": "https://example.com"}}, prefix="doc"
        ... )  # doctest: +SKIP
        <?xml version='1.0' encoding='utf-8'?>
        <doc:data xmlns:doc="https://example.com">
          <doc:row>
            <doc:index>0</doc:index>
            <doc:shape>square</doc:shape>
            <doc:degrees>360</doc:degrees>
            <doc:sides>4.0</doc:sides>
          </doc:row>
          <doc:row>
            <doc:index>1</doc:index>
            <doc:shape>circle</doc:shape>
            <doc:degrees>360</doc:degrees>
            <doc:sides/>
          </doc:row>
          <doc:row>
            <doc:index>2</doc:index>
            <doc:shape>triangle</doc:shape>
            <doc:degrees>180</doc:degrees>
            <doc:sides>3.0</doc:sides>
          </doc:row>
        </doc:data>
        """
        from pandas.io.formats.xml import EtreeXMLFormatter, LxmlXMLFormatter
        lxml = import_optional_dependency('lxml.etree', errors='ignore')
        if parser == 'lxml':
            if lxml is not None:
                TreeBuilder = LxmlXMLFormatter
            else:
                raise ImportError('lxml not found, please install or use the etree parser.')
        elif parser == 'etree':
            TreeBuilder = EtreeXMLFormatter
        else:
            raise ValueError('Values for parser can only be lxml or etree.')
        xml_formatter = TreeBuilder(self, path_or_buffer=path_or_buffer, index=index, root_name=root_name, row_name=row_name, na_rep=na_rep, attr_cols=attr_cols, elem_cols=elem_cols, namespaces=namespaces, prefix=prefix, encoding=encoding, xml_declaration=xml_declaration, pretty_print=pretty_print, stylesheet=stylesheet, compression=compression, storage_options=storage_options)
        return xml_formatter.write_output()

    @doc(INFO_DOCSTRING, **frame_sub_kwargs)
    def info(self, verbose: None=None, buf: None=None, max_cols: None=None, memory_usage: None=None, show_counts: None=None) -> None:
        info = DataFrameInfo(data=self, memory_usage=memory_usage)
        info.render(buf=buf, max_cols=max_cols, verbose=verbose, show_counts=show_counts)

    def memory_usage(self, index: bool=True, deep: bool=False):
        """
        Return the memory usage of each column in bytes.

        The memory usage can optionally include the contribution of
        the index and elements of `object` dtype.

        This value is displayed in `DataFrame.info` by default. This can be
        suppressed by setting ``pandas.options.display.memory_usage`` to False.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the DataFrame's
            index in returned Series. If ``index=True``, the memory usage of
            the index is the first item in the output.
        deep : bool, default False
            If True, introspect the data deeply by interrogating
            `object` dtypes for system-level memory consumption, and include
            it in the returned values.

        Returns
        -------
        Series
            A Series whose index is the original column names and whose values
            is the memory usage of each column in bytes.

        See Also
        --------
        numpy.ndarray.nbytes : Total bytes consumed by the elements of an
            ndarray.
        Series.memory_usage : Bytes consumed by a Series.
        Categorical : Memory-efficient array for string values with
            many repeated values.
        DataFrame.info : Concise summary of a DataFrame.

        Notes
        -----
        See the :ref:`Frequently Asked Questions <df-memory-usage>` for more
        details.

        Examples
        --------
        >>> dtypes = ["int64", "float64", "complex128", "object", "bool"]
        >>> data = dict([(t, np.ones(shape=5000, dtype=int).astype(t)) for t in dtypes])
        >>> df = pd.DataFrame(data)
        >>> df.head()
           int64  float64            complex128  object  bool
        0      1      1.0              1.0+0.0j       1  True
        1      1      1.0              1.0+0.0j       1  True
        2      1      1.0              1.0+0.0j       1  True
        3      1      1.0              1.0+0.0j       1  True
        4      1      1.0              1.0+0.0j       1  True

        >>> df.memory_usage()
        Index           128
        int64         40000
        float64       40000
        complex128    80000
        object        40000
        bool           5000
        dtype: int64

        >>> df.memory_usage(index=False)
        int64         40000
        float64       40000
        complex128    80000
        object        40000
        bool           5000
        dtype: int64

        The memory footprint of `object` dtype columns is ignored by default:

        >>> df.memory_usage(deep=True)
        Index            128
        int64          40000
        float64        40000
        complex128     80000
        object        180000
        bool            5000
        dtype: int64

        Use a Categorical for efficient storage of an object-dtype column with
        many repeated values.

        >>> df["object"].astype("category").memory_usage(deep=True)
        5136
        """
        result = self._constructor_sliced([c.memory_usage(index=False, deep=deep) for col, c in self.items()], index=self.columns, dtype=np.intp)
        if index:
            index_memory_usage = self._constructor_sliced(self.index.memory_usage(deep=deep), index=['Index'])
            result = index_memory_usage._append(result)
        return result

    def transpose(self, *args, copy: Any=lib.no_default):
        """
        Transpose index and columns.

        Reflect the DataFrame over its main diagonal by writing rows as columns
        and vice-versa. The property :attr:`.T` is an accessor to the method
        :meth:`transpose`.

        Parameters
        ----------
        *args : tuple, optional
            Accepted for compatibility with NumPy.
        copy : bool, default False
            Whether to copy the data after transposing, even for DataFrames
            with a single dtype.

            Note that a copy is always required for mixed dtype DataFrames,
            or for DataFrames with any extension types.

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
        DataFrame
            The transposed DataFrame.

        See Also
        --------
        numpy.transpose : Permute the dimensions of a given array.

        Notes
        -----
        Transposing a DataFrame with mixed dtypes will result in a homogeneous
        DataFrame with the `object` dtype. In such a case, a copy of the data
        is always made.

        Examples
        --------
        **Square DataFrame with homogeneous dtype**

        >>> d1 = {"col1": [1, 2], "col2": [3, 4]}
        >>> df1 = pd.DataFrame(data=d1)
        >>> df1
           col1  col2
        0     1     3
        1     2     4

        >>> df1_transposed = df1.T  # or df1.transpose()
        >>> df1_transposed
              0  1
        col1  1  2
        col2  3  4

        When the dtype is homogeneous in the original DataFrame, we get a
        transposed DataFrame with the same dtype:

        >>> df1.dtypes
        col1    int64
        col2    int64
        dtype: object
        >>> df1_transposed.dtypes
        0    int64
        1    int64
        dtype: object

        **Non-square DataFrame with mixed dtypes**

        >>> d2 = {
        ...     "name": ["Alice", "Bob"],
        ...     "score": [9.5, 8],
        ...     "employed": [False, True],
        ...     "kids": [0, 0],
        ... }
        >>> df2 = pd.DataFrame(data=d2)
        >>> df2
            name  score  employed  kids
        0  Alice    9.5     False     0
        1    Bob    8.0      True     0

        >>> df2_transposed = df2.T  # or df2.transpose()
        >>> df2_transposed
                      0     1
        name      Alice   Bob
        score       9.5   8.0
        employed  False  True
        kids          0     0

        When the DataFrame has mixed dtypes, we get a transposed DataFrame with
        the `object` dtype:

        >>> df2.dtypes
        name         object
        score       float64
        employed       bool
        kids          int64
        dtype: object
        >>> df2_transposed.dtypes
        0    object
        1    object
        dtype: object
        """
        self._check_copy_deprecation(copy)
        nv.validate_transpose(args, {})
        first_dtype = self.dtypes.iloc[0] if len(self.columns) else None
        if self._can_fast_transpose:
            new_vals = self._values.T
            result = self._constructor(new_vals, index=self.columns, columns=self.index, copy=False, dtype=new_vals.dtype)
            if len(self) > 0:
                result._mgr.add_references(self._mgr)
        elif self._is_homogeneous_type and first_dtype is not None and isinstance(first_dtype, ExtensionDtype):
            if isinstance(first_dtype, BaseMaskedDtype):
                from pandas.core.arrays.masked import transpose_homogeneous_masked_arrays
                new_values = transpose_homogeneous_masked_arrays(cast(Sequence[BaseMaskedArray], self._iter_column_arrays()))
            elif isinstance(first_dtype, ArrowDtype):
                from pandas.core.arrays.arrow.array import ArrowExtensionArray, transpose_homogeneous_pyarrow
                new_values = transpose_homogeneous_pyarrow(cast(Sequence[ArrowExtensionArray], self._iter_column_arrays()))
            else:
                arr_typ = first_dtype.construct_array_type()
                values = self.values
                new_values = [arr_typ._from_sequence(row, dtype=first_dtype) for row in values]
            result = type(self)._from_arrays(new_values, index=self.columns, columns=self.index, verify_integrity=False)
        else:
            new_arr = self.values.T
            result = self._constructor(new_arr, index=self.columns, columns=self.index, dtype=new_arr.dtype, copy=False)
        return result.__finalize__(self, method='transpose')

    @property
    def T(self):
        """
        The transpose of the DataFrame.

        Returns
        -------
        DataFrame
            The transposed DataFrame.

        See Also
        --------
        DataFrame.transpose : Transpose index and columns.

        Examples
        --------
        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df
           col1  col2
        0     1     3
        1     2     4

        >>> df.T
              0  1
        col1  1  2
        col2  3  4
        """
        return self.transpose()

    def _ixs(self, i: Any, axis: int=0):
        """
        Parameters
        ----------
        i : int
        axis : int

        Returns
        -------
        Series
        """
        if axis == 0:
            new_mgr = self._mgr.fast_xs(i)
            result = self._constructor_sliced_from_mgr(new_mgr, axes=new_mgr.axes)
            result._name = self.index[i]
            return result.__finalize__(self)
        else:
            col_mgr = self._mgr.iget(i)
            return self._box_col_values(col_mgr, i)

    def _get_column_array(self, i: Any):
        """
        Get the values of the i'th column (ndarray or ExtensionArray, as stored
        in the Block)

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution (for read-only purposes).
        """
        return self._mgr.iget_values(i)

    def _iter_column_arrays(self) -> typing.Generator:
        """
        Iterate over the arrays of all columns in order.
        This returns the values as stored in the Block (ndarray or ExtensionArray).

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution (for read-only purposes).
        """
        for i in range(len(self.columns)):
            yield self._get_column_array(i)

    def __getitem__(self, key: Any):
        check_dict_or_set_indexers(key)
        key = lib.item_from_zerodim(key)
        key = com.apply_if_callable(key, self)
        if is_hashable(key) and (not is_iterator(key)) and (not isinstance(key, slice)):
            is_mi = isinstance(self.columns, MultiIndex)
            if not is_mi and (self.columns.is_unique and key in self.columns or key in self.columns.drop_duplicates(keep=False)):
                return self._get_item(key)
            elif is_mi and self.columns.is_unique and (key in self.columns):
                return self._getitem_multilevel(key)
        if isinstance(key, slice):
            return self._getitem_slice(key)
        if isinstance(key, DataFrame):
            return self.where(key)
        if com.is_bool_indexer(key):
            return self._getitem_bool_array(key)
        is_single_key = isinstance(key, tuple) or not is_list_like(key)
        if is_single_key:
            if self.columns.nlevels > 1:
                return self._getitem_multilevel(key)
            indexer = self.columns.get_loc(key)
            if is_integer(indexer):
                indexer = [indexer]
        else:
            if is_iterator(key):
                key = list(key)
            indexer = self.columns._get_indexer_strict(key, 'columns')[1]
        if getattr(indexer, 'dtype', None) == bool:
            indexer = np.where(indexer)[0]
        if isinstance(indexer, slice):
            return self._slice(indexer, axis=1)
        data = self.take(indexer, axis=1)
        if is_single_key:
            if data.shape[1] == 1 and (not isinstance(self.columns, MultiIndex)):
                return data._get_item(key)
        return data

    def _getitem_bool_array(self, key: Any):
        if isinstance(key, Series) and (not key.index.equals(self.index)):
            warnings.warn('Boolean Series key will be reindexed to match DataFrame index.', UserWarning, stacklevel=find_stack_level())
        elif len(key) != len(self.index):
            raise ValueError(f'Item wrong length {len(key)} instead of {len(self.index)}.')
        key = check_bool_indexer(self.index, key)
        if key.all():
            return self.copy(deep=False)
        indexer = key.nonzero()[0]
        return self.take(indexer, axis=0)

    def _getitem_multilevel(self, key: Any):
        loc = self.columns.get_loc(key)
        if isinstance(loc, (slice, np.ndarray)):
            new_columns = self.columns[loc]
            result_columns = maybe_droplevels(new_columns, key)
            result = self.iloc[:, loc]
            result.columns = result_columns
            if len(result.columns) == 1:
                top = result.columns[0]
                if isinstance(top, tuple):
                    top = top[0]
                if top == '':
                    result = result['']
                    if isinstance(result, Series):
                        result = self._constructor_sliced(result, index=self.index, name=key)
            return result
        else:
            return self._ixs(loc, axis=1)

    def _get_value(self, index: Any, col: Any, takeable: bool=False):
        """
        Quickly retrieve single value at passed column and index.

        Parameters
        ----------
        index : row label
        col : column label
        takeable : interpret the index/col as indexers, default False

        Returns
        -------
        scalar

        Notes
        -----
        Assumes that both `self.index._index_as_unique` and
        `self.columns._index_as_unique`; Caller is responsible for checking.
        """
        if takeable:
            series = self._ixs(col, axis=1)
            return series._values[index]
        series = self._get_item(col)
        if not isinstance(self.index, MultiIndex):
            row = self.index.get_loc(index)
            return series._values[row]
        loc = self.index._engine.get_loc(index)
        return series._values[loc]

    def isetitem(self, loc: Any, value: Any) -> None:
        """
        Set the given value in the column with position `loc`.

        This is a positional analogue to ``__setitem__``.

        Parameters
        ----------
        loc : int or sequence of ints
            Index position for the column.
        value : scalar or arraylike
            Value(s) for the column.

        See Also
        --------
        DataFrame.iloc : Purely integer-location based indexing for selection by
            position.

        Notes
        -----
        ``frame.isetitem(loc, value)`` is an in-place method as it will
        modify the DataFrame in place (not returning a new object). In contrast to
        ``frame.iloc[:, i] = value`` which will try to update the existing values in
        place, ``frame.isetitem(loc, value)`` will not update the values of the column
        itself in place, it will instead insert a new array.

        In cases where ``frame.columns`` is unique, this is equivalent to
        ``frame[frame.columns[i]] = value``.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> df.isetitem(1, [5, 6])
        >>> df
              A  B
        0     1  5
        1     2  6
        """
        if isinstance(value, DataFrame):
            if is_integer(loc):
                loc = [loc]
            if len(loc) != len(value.columns):
                raise ValueError(f'Got {len(loc)} positions but value has {len(value.columns)} columns.')
            for i, idx in enumerate(loc):
                arraylike, refs = self._sanitize_column(value.iloc[:, i])
                self._iset_item_mgr(idx, arraylike, inplace=False, refs=refs)
            return
        arraylike, refs = self._sanitize_column(value)
        self._iset_item_mgr(loc, arraylike, inplace=False, refs=refs)

    def __setitem__(self, key: Any, value: Any):
        if not PYPY:
            if sys.getrefcount(self) <= 3:
                warnings.warn(_chained_assignment_msg, ChainedAssignmentError, stacklevel=2)
        key = com.apply_if_callable(key, self)
        if isinstance(key, slice):
            slc = self.index._convert_slice_indexer(key, kind='getitem')
            return self._setitem_slice(slc, value)
        if isinstance(key, DataFrame) or getattr(key, 'ndim', None) == 2:
            self._setitem_frame(key, value)
        elif isinstance(key, (Series, np.ndarray, list, Index)):
            self._setitem_array(key, value)
        elif isinstance(value, DataFrame):
            self._set_item_frame_value(key, value)
        elif is_list_like(value) and (not self.columns.is_unique) and (1 < len(self.columns.get_indexer_for([key])) == len(value)):
            self._setitem_array([key], value)
        else:
            self._set_item(key, value)

    def _setitem_slice(self, key: Any, value: Any) -> None:
        self.iloc[key] = value

    def _setitem_array(self, key: Any, value: Any) -> None:
        if com.is_bool_indexer(key):
            if len(key) != len(self.index):
                raise ValueError(f'Item wrong length {len(key)} instead of {len(self.index)}!')
            key = check_bool_indexer(self.index, key)
            indexer = key.nonzero()[0]
            if isinstance(value, DataFrame):
                value = value.reindex(self.index.take(indexer))
            self.iloc[indexer] = value
        elif isinstance(value, DataFrame):
            check_key_length(self.columns, key, value)
            for k1, k2 in zip(key, value.columns):
                self[k1] = value[k2]
        elif not is_list_like(value):
            for col in key:
                self[col] = value
        elif isinstance(value, np.ndarray) and value.ndim == 2:
            self._iset_not_inplace(key, value)
        elif np.ndim(value) > 1:
            value = DataFrame(value).values
            self._setitem_array(key, value)
        else:
            self._iset_not_inplace(key, value)

    def _iset_not_inplace(self, key: Any, value: Any) -> None:

        def igetitem(obj: Any, i: Any):
            if isinstance(obj, np.ndarray):
                return obj[..., i]
            else:
                return obj[i]
        if self.columns.is_unique:
            if np.shape(value)[-1] != len(key):
                raise ValueError('Columns must be same length as key')
            for i, col in enumerate(key):
                self[col] = igetitem(value, i)
        else:
            ilocs = self.columns.get_indexer_non_unique(key)[0]
            if (ilocs < 0).any():
                raise NotImplementedError
            if np.shape(value)[-1] != len(ilocs):
                raise ValueError('Columns must be same length as key')
            assert np.ndim(value) <= 2
            orig_columns = self.columns
            try:
                self.columns = Index(range(len(self.columns)))
                for i, iloc in enumerate(ilocs):
                    self[iloc] = igetitem(value, i)
            finally:
                self.columns = orig_columns

    def _setitem_frame(self, key: Any, value: Any) -> None:
        if isinstance(key, np.ndarray):
            if key.shape != self.shape:
                raise ValueError('Array conditional must be same shape as self')
            key = self._constructor(key, **self._construct_axes_dict(), copy=False)
        if key.size and (not all((is_bool_dtype(dtype) for dtype in key.dtypes))):
            raise TypeError('Must pass DataFrame or 2-d ndarray with boolean values only')
        self._where(-key, value, inplace=True)

    def _set_item_frame_value(self, key: Any, value: Any) -> None:
        self._ensure_valid_index(value)
        if key in self.columns:
            loc = self.columns.get_loc(key)
            cols = self.columns[loc]
            len_cols = 1 if is_scalar(cols) or isinstance(cols, tuple) else len(cols)
            if len_cols != len(value.columns):
                raise ValueError('Columns must be same length as key')
            if isinstance(self.columns, MultiIndex) and isinstance(loc, (slice, Series, np.ndarray, Index)):
                cols_droplevel = maybe_droplevels(cols, key)
                if len(cols_droplevel) and (not cols_droplevel.equals(value.columns)):
                    value = value.reindex(cols_droplevel, axis=1)
                for col, col_droplevel in zip(cols, cols_droplevel):
                    self[col] = value[col_droplevel]
                return
            if is_scalar(cols):
                self[cols] = value[value.columns[0]]
                return
            if isinstance(loc, slice):
                locs = np.arange(loc.start, loc.stop, loc.step)
            elif is_scalar(loc):
                locs = [loc]
            else:
                locs = loc.nonzero()[0]
            return self.isetitem(locs, value)
        if len(value.columns) > 1:
            raise ValueError(f'Cannot set a DataFrame with multiple columns to the single column {key}')
        elif len(value.columns) == 0:
            raise ValueError(f'Cannot set a DataFrame without columns to the column {key}')
        self[key] = value[value.columns[0]]

    def _iset_item_mgr(self, loc: Any, value: Any, inplace: bool=False, refs: None=None) -> None:
        self._mgr.iset(loc, value, inplace=inplace, refs=refs)

    def _set_item_mgr(self, key: Any, value: Any, refs: None=None) -> None:
        try:
            loc = self._info_axis.get_loc(key)
        except KeyError:
            self._mgr.insert(len(self._info_axis), key, value, refs)
        else:
            self._iset_item_mgr(loc, value, refs=refs)

    def _iset_item(self, loc: Any, value: Any, inplace: bool=True) -> None:
        self._iset_item_mgr(loc, value._values, inplace=inplace, refs=value._references)

    def _set_item(self, key: Any, value: Any) -> None:
        """
        Add series to DataFrame in specified column.

        If series is a numpy-array (not a Series/TimeSeries), it must be the
        same length as the DataFrames index or an error will be thrown.

        Series/TimeSeries will be conformed to the DataFrames index to
        ensure homogeneity.
        """
        value, refs = self._sanitize_column(value)
        if key in self.columns and value.ndim == 1 and (not isinstance(value.dtype, ExtensionDtype)):
            if not self.columns.is_unique or isinstance(self.columns, MultiIndex):
                existing_piece = self[key]
                if isinstance(existing_piece, DataFrame):
                    value = np.tile(value, (len(existing_piece.columns), 1)).T
                    refs = None
        self._set_item_mgr(key, value, refs)

    def _set_value(self, index: Any, col: Any, value: Any, takeable: bool=False) -> None:
        """
        Put single value at passed column and index.

        Parameters
        ----------
        index : Label
            row label
        col : Label
            column label
        value : scalar
        takeable : bool, default False
            Sets whether or not index/col interpreted as indexers
        """
        try:
            if takeable:
                icol = col
                iindex = cast(int, index)
            else:
                icol = self.columns.get_loc(col)
                iindex = self.index.get_loc(index)
            self._mgr.column_setitem(icol, iindex, value, inplace_only=True)
        except (KeyError, TypeError, ValueError, LossySetitemError):
            if takeable:
                self.iloc[index, col] = value
            else:
                self.loc[index, col] = value
        except InvalidIndexError as ii_err:
            raise InvalidIndexError(f'You can only assign a scalar value not a {type(value)}') from ii_err

    def _ensure_valid_index(self, value: Any) -> None:
        """
        Ensure that if we don't have an index, that we can create one from the
        passed value.
        """
        if not len(self.index) and is_list_like(value) and len(value):
            if not isinstance(value, DataFrame):
                try:
                    value = Series(value)
                except (ValueError, NotImplementedError, TypeError) as err:
                    raise ValueError('Cannot set a frame with no defined index and a value that cannot be converted to a Series') from err
            index_copy = value.index.copy()
            if self.index.name is not None:
                index_copy.name = self.index.name
            self._mgr = self._mgr.reindex_axis(index_copy, axis=1, fill_value=np.nan)

    def _box_col_values(self, values: Any, loc: Any):
        """
        Provide boxed values for a column.
        """
        name = self.columns[loc]
        obj = self._constructor_sliced_from_mgr(values, axes=values.axes)
        obj._name = name
        return obj.__finalize__(self)

    def _get_item(self, item: Any):
        loc = self.columns.get_loc(item)
        return self._ixs(loc, axis=1)

    @overload
    def query(self, expr: Any, *, inplace: Any=..., **kwargs) -> None:
        ...

    @overload
    def query(self, expr: Any, *, inplace: Any, **kwargs) -> None:
        ...

    @overload
    def query(self, expr: Any, *, inplace: Any=..., **kwargs) -> None:
        ...

    def query(self, expr: Any, *, inplace: Any=False, **kwargs) -> None:
        """
        Query the columns of a DataFrame with a boolean expression.

        .. warning::

            This method can run arbitrary code which can make you vulnerable to code
            injection if you pass user input to this function.

        Parameters
        ----------
        expr : str
            The query string to evaluate.

            See the documentation for :func:`eval` for details of
            supported operations and functions in the query string.

            See the documentation for :meth:`DataFrame.eval` for details on
            referring to column names and variables in the query string.
        inplace : bool
            Whether to modify the DataFrame rather than creating a new one.
        **kwargs
            See the documentation for :func:`eval` for complete details
            on the keyword arguments accepted by :meth:`DataFrame.query`.

        Returns
        -------
        DataFrame or None
            DataFrame resulting from the provided query expression or
            None if ``inplace=True``.

        See Also
        --------
        eval : Evaluate a string describing operations on
            DataFrame columns.
        DataFrame.eval : Evaluate a string describing operations on
            DataFrame columns.

        Notes
        -----
        The result of the evaluation of this expression is first passed to
        :attr:`DataFrame.loc` and if that fails because of a
        multidimensional key (e.g., a DataFrame) then the result will be passed
        to :meth:`DataFrame.__getitem__`.

        This method uses the top-level :func:`eval` function to
        evaluate the passed query.

        The :meth:`~pandas.DataFrame.query` method uses a slightly
        modified Python syntax by default. For example, the ``&`` and ``|``
        (bitwise) operators have the precedence of their boolean cousins,
        :keyword:`and` and :keyword:`or`. This *is* syntactically valid Python,
        however the semantics are different.

        You can change the semantics of the expression by passing the keyword
        argument ``parser='python'``. This enforces the same semantics as
        evaluation in Python space. Likewise, you can pass ``engine='python'``
        to evaluate an expression using Python itself as a backend. This is not
        recommended as it is inefficient compared to using ``numexpr`` as the
        engine.

        The :attr:`DataFrame.index` and
        :attr:`DataFrame.columns` attributes of the
        :class:`~pandas.DataFrame` instance are placed in the query namespace
        by default, which allows you to treat both the index and columns of the
        frame as a column in the frame.
        The identifier ``index`` is used for the frame index; you can also
        use the name of the index to identify it in a query. Please note that
        Python keywords may not be used as identifiers.

        For further details and examples see the ``query`` documentation in
        :ref:`indexing <indexing.query>`.

        *Backtick quoted variables*

        Backtick quoted variables are parsed as literal Python code and
        are converted internally to a Python valid identifier.
        This can lead to the following problems.

        During parsing a number of disallowed characters inside the backtick
        quoted string are replaced by strings that are allowed as a Python identifier.
        These characters include all operators in Python, the space character, the
        question mark, the exclamation mark, the dollar sign, and the euro sign.

        A backtick can be escaped by double backticks.

        See also the `Python documentation about lexical analysis
        <https://docs.python.org/3/reference/lexical_analysis.html>`__
        in combination with the source code in :mod:`pandas.core.computation.parsing`.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"A": range(1, 6), "B": range(10, 0, -2), "C&C": range(10, 5, -1)}
        ... )
        >>> df
           A   B  C&C
        0  1  10   10
        1  2   8    9
        2  3   6    8
        3  4   4    7
        4  5   2    6
        >>> df.query("A > B")
           A  B  C&C
        4  5  2    6

        The previous expression is equivalent to

        >>> df[df.A > df.B]
           A  B  C&C
        4  5  2    6

        For columns with spaces in their name, you can use backtick quoting.

        >>> df.query("B == `C&C`")
           A   B  C&C
        0  1  10   10

        The previous expression is equivalent to

        >>> df[df.B == df["C&C"]]
           A   B  C&C
        0  1  10   10

        Using local variable:

        >>> local_var = 2
        >>> df.query("A <= @local_var")
        A   B  C&C
        0  1  10   10
        1  2   8    9
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if not isinstance(expr, str):
            msg = f'expr must be a string to be evaluated, {type(expr)} given'
            raise ValueError(msg)
        kwargs['level'] = kwargs.pop('level', 0) + 1
        kwargs['target'] = None
        res = self.eval(expr, **kwargs)
        try:
            result = self.loc[res]
        except ValueError:
            result = self[res]
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    @overload
    def eval(self, expr: Any, *, inplace: Any=..., **kwargs) -> None:
        ...

    @overload
    def eval(self, expr: Any, *, inplace: Any, **kwargs) -> None:
        ...

    def eval(self, expr: Any, *, inplace: Any=False, **kwargs) -> None:
        """
        Evaluate a string describing operations on DataFrame columns.

        .. warning::

            This method can run arbitrary code which can make you vulnerable to code
            injection if you pass user input to this function.

        Operates on columns only, not specific rows or elements.  This allows
        `eval` to run arbitrary code, which can make you vulnerable to code
        injection if you pass user input to this function.

        Parameters
        ----------
        expr : str
            The expression string to evaluate.

            You can refer to variables
            in the environment by prefixing them with an '@' character like
            ``@a + b``.

            You can refer to column names that are not valid Python variable names
            by surrounding them in backticks. Thus, column names containing spaces
            or punctuation (besides underscores) or starting with digits must be
            surrounded by backticks. (For example, a column named "Area (cm^2)" would
            be referenced as ```Area (cm^2)```). Column names which are Python keywords
            (like "if", "for", "import", etc) cannot be used.

            For example, if one of your columns is called ``a a`` and you want
            to sum it with ``b``, your query should be ```a a` + b``.

            See the documentation for :func:`eval` for full details of
            supported operations and functions in the expression string.
        inplace : bool, default False
            If the expression contains an assignment, whether to perform the
            operation inplace and mutate the existing DataFrame. Otherwise,
            a new DataFrame is returned.
        **kwargs
            See the documentation for :func:`eval` for complete details
            on the keyword arguments accepted by
            :meth:`~pandas.DataFrame.eval`.

        Returns
        -------
        ndarray, scalar, pandas object, or None
            The result of the evaluation or None if ``inplace=True``.

        See Also
        --------
        DataFrame.query : Evaluates a boolean expression to query the columns
            of a frame.
        DataFrame.assign : Can evaluate an expression or function to create new
            values for a column.
        eval : Evaluate a Python expression as a string using various
            backends.

        Notes
        -----
        For more details see the API documentation for :func:`~eval`.
        For detailed examples see :ref:`enhancing performance with eval
        <enhancingperf.eval>`.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"A": range(1, 6), "B": range(10, 0, -2), "C&C": range(10, 5, -1)}
        ... )
        >>> df
           A   B  C&C
        0  1  10   10
        1  2   8    9
        2  3   6    8
        3  4   4    7
        4  5   2    6
        >>> df.eval("A + B")
        0    11
        1    10
        2     9
        3     8
        4     7
        dtype: int64

        Assignment is allowed though by default the original DataFrame is not
        modified.

        >>> df.eval("D = A + B")
           A   B  C&C   D
        0  1  10   10  11
        1  2   8    9  10
        2  3   6    8   9
        3  4   4    7   8
        4  5   2    6   7
        >>> df
           A   B  C&C
        0  1  10   10
        1  2   8    9
        2  3   6    8
        3  4   4    7
        4  5   2    6

        Multiple columns can be assigned to using multi-line expressions:

        >>> df.eval(
        ...     '''
        ... D = A + B
        ... E = A - B
        ... '''
        ... )
           A   B  C&C   D  E
        0  1  10   10  11 -9
        1  2   8    9  10 -6
        2  3   6    8   9 -3
        3  4   4    7   8  0
        4  5   2    6   7  3

        For columns with spaces or other disallowed characters in their name, you can
        use backtick quoting.

        >>> df.eval("B * `C&C`")
        0    100
        1     72
        2     48
        3     28
        4     12

        Local variables shall be explicitly referenced using ``@``
        character in front of the name:

        >>> local_var = 2
        >>> df.eval("@local_var * A")
        0     2
        1     4
        2     6
        3     8
        4    10
        """
        from pandas.core.computation.eval import eval as _eval
        inplace = validate_bool_kwarg(inplace, 'inplace')
        kwargs['level'] = kwargs.pop('level', 0) + 1
        index_resolvers = self._get_index_resolvers()
        column_resolvers = self._get_cleaned_column_resolvers()
        resolvers = (column_resolvers, index_resolvers)
        if 'target' not in kwargs:
            kwargs['target'] = self
        kwargs['resolvers'] = tuple(kwargs.get('resolvers', ())) + resolvers
        return _eval(expr, inplace=inplace, **kwargs)

    def select_dtypes(self, include: Any=None, exclude: Any=None):
        """
        Return a subset of the DataFrame's columns based on the column dtypes.

        This method allows for filtering columns based on their data types.
        It is useful when working with heterogeneous DataFrames where operations
        need to be performed on a specific subset of data types.

        Parameters
        ----------
        include, exclude : scalar or list-like
            A selection of dtypes or strings to be included/excluded. At least
            one of these parameters must be supplied.

        Returns
        -------
        DataFrame
            The subset of the frame including the dtypes in ``include`` and
            excluding the dtypes in ``exclude``.

        Raises
        ------
        ValueError
            * If both of ``include`` and ``exclude`` are empty
            * If ``include`` and ``exclude`` have overlapping elements
        TypeError
            * If any kind of string dtype is passed in.

        See Also
        --------
        DataFrame.dtypes: Return Series with the data type of each column.

        Notes
        -----
        * To select all *numeric* types, use ``np.number`` or ``'number'``
        * To select strings you must use the ``object`` dtype, but note that
          this will return *all* object dtype columns. With
          ``pd.options.future.infer_string`` enabled, using ``"str"`` will
          work to select all string columns.
        * See the `numpy dtype hierarchy
          <https://numpy.org/doc/stable/reference/arrays.scalars.html>`__
        * To select datetimes, use ``np.datetime64``, ``'datetime'`` or
          ``'datetime64'``
        * To select timedeltas, use ``np.timedelta64``, ``'timedelta'`` or
          ``'timedelta64'``
        * To select Pandas categorical dtypes, use ``'category'``
        * To select Pandas datetimetz dtypes, use ``'datetimetz'``
          or ``'datetime64[ns, tz]'``

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"a": [1, 2] * 3, "b": [True, False] * 3, "c": [1.0, 2.0] * 3}
        ... )
        >>> df
                a      b  c
        0       1   True  1.0
        1       2  False  2.0
        2       1   True  1.0
        3       2  False  2.0
        4       1   True  1.0
        5       2  False  2.0

        >>> df.select_dtypes(include="bool")
           b
        0  True
        1  False
        2  True
        3  False
        4  True
        5  False

        >>> df.select_dtypes(include=["float64"])
           c
        0  1.0
        1  2.0
        2  1.0
        3  2.0
        4  1.0
        5  2.0

        >>> df.select_dtypes(exclude=["int64"])
               b    c
        0   True  1.0
        1  False  2.0
        2   True  1.0
        3  False  2.0
        4   True  1.0
        5  False  2.0
        """
        if not is_list_like(include):
            include = (include,) if include is not None else ()
        if not is_list_like(exclude):
            exclude = (exclude,) if exclude is not None else ()
        selection = (frozenset(include), frozenset(exclude))
        if not any(selection):
            raise ValueError('at least one of include or exclude must be nonempty')

        def check_int_infer_dtype(dtypes: Any) -> set:
            converted_dtypes = []
            for dtype in dtypes:
                if isinstance(dtype, str) and dtype == 'int' or dtype is int:
                    converted_dtypes.append(np.int32)
                    converted_dtypes.append(np.int64)
                elif dtype == 'float' or dtype is float:
                    converted_dtypes.extend([np.float64, np.float32])
                else:
                    converted_dtypes.append(infer_dtype_from_object(dtype))
            return frozenset(converted_dtypes)
        include = check_int_infer_dtype(include)
        exclude = check_int_infer_dtype(exclude)
        for dtypes in (include, exclude):
            invalidate_string_dtypes(dtypes)
        if not include.isdisjoint(exclude):
            raise ValueError(f'include and exclude overlap on {include & exclude}')

        def dtype_predicate(dtype: Any, dtypes_set: Any) -> bool:
            dtype = dtype if not isinstance(dtype, ArrowDtype) else dtype.numpy_dtype
            return issubclass(dtype.type, tuple(dtypes_set)) or (np.number in dtypes_set and getattr(dtype, '_is_numeric', False) and (not is_bool_dtype(dtype)))

        def predicate(arr: Any) -> bool:
            dtype = arr.dtype
            if include:
                if not dtype_predicate(dtype, include):
                    return False
            if exclude:
                if dtype_predicate(dtype, exclude):
                    return False
            return True
        mgr = self._mgr._get_data_subset(predicate).copy(deep=False)
        return self._constructor_from_mgr(mgr, axes=mgr.axes).__finalize__(self)

    def insert(self, loc: Any, column: Any, value: Any, allow_duplicates: Any=lib.no_default) -> None:
        """
        Insert column into DataFrame at specified location.

        Raises a ValueError if `column` is already contained in the DataFrame,
        unless `allow_duplicates` is set to True.

        Parameters
        ----------
        loc : int
            Insertion index. Must verify 0 <= loc <= len(columns).
        column : str, number, or hashable object
            Label of the inserted column.
        value : Scalar, Series, or array-like
            Content of the inserted column.
        allow_duplicates : bool, optional, default lib.no_default
            Allow duplicate column labels to be created.

        See Also
        --------
        Index.insert : Insert new item by index.

        Examples
        --------
        >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        >>> df
           col1  col2
        0     1     3
        1     2     4
        >>> df.insert(1, "newcol", [99, 99])
        >>> df
           col1  newcol  col2
        0     1      99     3
        1     2      99     4
        >>> df.insert(0, "col1", [100, 100], allow_duplicates=True)
        >>> df
           col1  col1  newcol  col2
        0   100     1      99     3
        1   100     2      99     4

        Notice that pandas uses index alignment in case of `value` from type `Series`:

        >>> df.insert(0, "col0", pd.Series([5, 6], index=[1, 2]))
        >>> df
           col0  col1  col1  newcol  col2
        0   NaN   100     1      99     3
        1   5.0   100     2      99     4
        """
        if allow_duplicates is lib.no_default:
            allow_duplicates = False
        if allow_duplicates and (not self.flags.allows_duplicate_labels):
            raise ValueError("Cannot specify 'allow_duplicates=True' when 'self.flags.allows_duplicate_labels' is False.")
        if not allow_duplicates and column in self.columns:
            raise ValueError(f'cannot insert {column}, already exists')
        if not is_integer(loc):
            raise TypeError('loc must be int')
        loc = int(loc)
        if isinstance(value, DataFrame) and len(value.columns) > 1:
            raise ValueError(f'Expected a one-dimensional object, got a DataFrame with {len(value.columns)} columns instead.')
        elif isinstance(value, DataFrame):
            value = value.iloc[:, 0]
        value, refs = self._sanitize_column(value)
        self._mgr.insert(loc, column, value, refs=refs)

    def assign(self, **kwargs):
        """
        Assign new columns to a DataFrame.

        Returns a new object with all original columns in addition to new ones.
        Existing columns that are re-assigned will be overwritten.

        Parameters
        ----------
        **kwargs : callable or Series
            The column names are keywords. If the values are
            callable, they are computed on the DataFrame and
            assigned to the new columns. The callable must not
            change input DataFrame (though pandas doesn't check it).
            If the values are not callable, (e.g. a Series, scalar, or array),
            they are simply assigned.

        Returns
        -------
        DataFrame
            A new DataFrame with the new columns in addition to
            all the existing columns.

        See Also
        --------
        DataFrame.loc : Select a subset of a DataFrame by labels.
        DataFrame.iloc : Select a subset of a DataFrame by positions.

        Notes
        -----
        Assigning multiple columns within the same ``assign`` is possible.
        Later items in '\\*\\*kwargs' may refer to newly created or modified
        columns in 'df'; items are computed and assigned into 'df' in order.

        Examples
        --------
        >>> df = pd.DataFrame({"temp_c": [17.0, 25.0]}, index=["Portland", "Berkeley"])
        >>> df
                  temp_c
        Portland    17.0
        Berkeley    25.0

        Where the value is a callable, evaluated on `df`:

        >>> df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32)
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0

        Alternatively, the same behavior can be achieved by directly
        referencing an existing Series or sequence:

        >>> df.assign(temp_f=df["temp_c"] * 9 / 5 + 32)
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0

        You can create multiple columns within the same assign where one
        of the columns depends on another one defined within the same assign:

        >>> df.assign(
        ...     temp_f=lambda x: x["temp_c"] * 9 / 5 + 32,
        ...     temp_k=lambda x: (x["temp_f"] + 459.67) * 5 / 9,
        ... )
                  temp_c  temp_f  temp_k
        Portland    17.0    62.6  290.15
        Berkeley    25.0    77.0  298.15
        """
        data = self.copy(deep=False)
        for k, v in kwargs.items():
            data[k] = com.apply_if_callable(v, data)
        return data

    def _sanitize_column(self, value: Any) -> tuple[None]:
        """
        Ensures new columns (which go into the BlockManager as new blocks) are
        always copied (or a reference is being tracked to them under CoW)
        and converted into an array.

        Parameters
        ----------
        value : scalar, Series, or array-like

        Returns
        -------
        tuple of numpy.ndarray or ExtensionArray and optional BlockValuesRefs
        """
        self._ensure_valid_index(value)
        assert not isinstance(value, DataFrame)
        if is_dict_like(value):
            if not isinstance(value, Series):
                value = Series(value)
            return _reindex_for_setitem(value, self.index)
        if is_list_like(value):
            com.require_length_match(value, self.index)
        return (sanitize_array(value, self.index, copy=True, allow_2d=True), None)

    @property
    def _series(self) -> dict:
        return {item: self._ixs(idx, axis=1) for idx, item in enumerate(self.columns)}

    def _reindex_multi(self, axes: Any, fill_value: Any):
        """
        We are guaranteed non-Nones in the axes.
        """
        new_index, row_indexer = self.index.reindex(axes['index'])
        new_columns, col_indexer = self.columns.reindex(axes['columns'])
        if row_indexer is not None and col_indexer is not None:
            indexer = (row_indexer, col_indexer)
            new_values = take_2d_multi(self.values, indexer, fill_value=fill_value)
            return self._constructor(new_values, index=new_index, columns=new_columns, copy=False)
        else:
            return self._reindex_with_indexers({0: [new_index, row_indexer], 1: [new_columns, col_indexer]}, fill_value=fill_value)

    @Appender('\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})\n\n        Change the row labels.\n\n        >>> df.set_axis([\'a\', \'b\', \'c\'], axis=\'index\')\n           A  B\n        a  1  4\n        b  2  5\n        c  3  6\n\n        Change the column labels.\n\n        >>> df.set_axis([\'I\', \'II\'], axis=\'columns\')\n           I  II\n        0  1   4\n        1  2   5\n        2  3   6\n        ')
    @Substitution(klass=_shared_doc_kwargs['klass'], axes_single_arg=_shared_doc_kwargs['axes_single_arg'], extended_summary_sub=' column or', axis_description_sub=', and 1 identifies the columns', see_also_sub=' or columns')
    @Appender(NDFrame.set_axis.__doc__)
    def set_axis(self, labels: Any, *, axis: int=0, copy: Any=lib.no_default):
        return super().set_axis(labels, axis=axis, copy=copy)

    @doc(NDFrame.reindex, klass=_shared_doc_kwargs['klass'], optional_reindex=_shared_doc_kwargs['optional_reindex'])
    def reindex(self, labels: None=None, *, index: None=None, columns: None=None, axis: None=None, method: None=None, copy: Any=lib.no_default, level: None=None, fill_value: Any=np.nan, limit: None=None, tolerance: None=None):
        return super().reindex(labels=labels, index=index, columns=columns, axis=axis, method=method, level=level, fill_value=fill_value, limit=limit, tolerance=tolerance, copy=copy)

    @overload
    def drop(self, labels: Any=..., *, axis: Any=..., index: Any=..., columns: Any=..., level: Any=..., inplace: Any, errors: Any=...) -> None:
        ...

    @overload
    def drop(self, labels: Any=..., *, axis: Any=..., index: Any=..., columns: Any=..., level: Any=..., inplace: Any=..., errors: Any=...) -> None:
        ...

    @overload
    def drop(self, labels: Any=..., *, axis: Any=..., index: Any=..., columns: Any=..., level: Any=..., inplace: Any=..., errors: Any=...) -> None:
        ...

    def drop(self, labels: Any=None, *, axis: Any=0, index: Any=None, columns: Any=None, level: Any=None, inplace: Any=False, errors: Any='raise') -> None:
        """
        Drop specified labels from rows or columns.

        Remove rows or columns by specifying label names and corresponding
        axis, or by directly specifying index or column names. When using a
        multi-index, labels on different levels can be removed by specifying
        the level. See the :ref:`user guide <advanced.shown_levels>`
        for more information about the now unused levels.

        Parameters
        ----------
        labels : single label or list-like
            Index or column labels to drop. A tuple will be used as a single
            label and not treated as a list-like.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Whether to drop labels from the index (0 or 'index') or
            columns (1 or 'columns').
        index : single label or list-like
            Alternative to specifying axis (``labels, axis=0``
            is equivalent to ``index=labels``).
        columns : single label or list-like
            Alternative to specifying axis (``labels, axis=1``
            is equivalent to ``columns=labels``).
        level : int or level name, optional
            For MultiIndex, level from which the labels will be removed.
        inplace : bool, default False
            If False, return a copy. Otherwise, do operation
            in place and return None.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are
            dropped.

        Returns
        -------
        DataFrame or None
            Returns DataFrame or None DataFrame with the specified
            index or column labels removed or None if inplace=True.

        Raises
        ------
        KeyError
            If any of the labels is not found in the selected axis.

        See Also
        --------
        DataFrame.loc : Label-location based indexer for selection by label.
        DataFrame.dropna : Return DataFrame with labels on given axis omitted
            where (all or any) data are missing.
        DataFrame.drop_duplicates : Return DataFrame with duplicate rows
            removed, optionally only considering certain columns.
        Series.drop : Return Series with specified index labels removed.

        Examples
        --------
        >>> df = pd.DataFrame(np.arange(12).reshape(3, 4), columns=["A", "B", "C", "D"])
        >>> df
           A  B   C   D
        0  0  1   2   3
        1  4  5   6   7
        2  8  9  10  11

        Drop columns

        >>> df.drop(["B", "C"], axis=1)
           A   D
        0  0   3
        1  4   7
        2  8  11

        >>> df.drop(columns=["B", "C"])
           A   D
        0  0   3
        1  4   7
        2  8  11

        Drop a row by index

        >>> df.drop([0, 1])
           A  B   C   D
        2  8  9  10  11

        Drop columns and/or rows of MultiIndex DataFrame

        >>> midx = pd.MultiIndex(
        ...     levels=[["llama", "cow", "falcon"], ["speed", "weight", "length"]],
        ...     codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        ... )
        >>> df = pd.DataFrame(
        ...     index=midx,
        ...     columns=["big", "small"],
        ...     data=[
        ...         [45, 30],
        ...         [200, 100],
        ...         [1.5, 1],
        ...         [30, 20],
        ...         [250, 150],
        ...         [1.5, 0.8],
        ...         [320, 250],
        ...         [1, 0.8],
        ...         [0.3, 0.2],
        ...     ],
        ... )
        >>> df
                        big     small
        llama   speed   45.0    30.0
                weight  200.0   100.0
                length  1.5     1.0
        cow     speed   30.0    20.0
                weight  250.0   150.0
                length  1.5     0.8
        falcon  speed   320.0   250.0
                weight  1.0     0.8
                length  0.3     0.2

        Drop a specific index combination from the MultiIndex
        DataFrame, i.e., drop the combination ``'falcon'`` and
        ``'weight'``, which deletes only the corresponding row

        >>> df.drop(index=("falcon", "weight"))
                        big     small
        llama   speed   45.0    30.0
                weight  200.0   100.0
                length  1.5     1.0
        cow     speed   30.0    20.0
                weight  250.0   150.0
                length  1.5     0.8
        falcon  speed   320.0   250.0
                length  0.3     0.2

        >>> df.drop(index="cow", columns="small")
                        big
        llama   speed   45.0
                weight  200.0
                length  1.5
        falcon  speed   320.0
                weight  1.0
                length  0.3

        >>> df.drop(index="length", level=1)
                        big     small
        llama   speed   45.0    30.0
                weight  200.0   100.0
        cow     speed   30.0    20.0
                weight  250.0   150.0
        falcon  speed   320.0   250.0
                weight  1.0     0.8
        """
        return super().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)

    @overload
    def rename(self, mapper: Any=..., *, index: Any=..., columns: Any=..., axis: Any=..., copy: Any=lib.no_default, inplace: Any, level: Any=..., errors: Any=...) -> None:
        ...

    @overload
    def rename(self, mapper: Any=..., *, index: Any=..., columns: Any=..., axis: Any=..., copy: Any=lib.no_default, inplace: Any=..., level: Any=..., errors: Any=...) -> None:
        ...

    @overload
    def rename(self, mapper: Any=..., *, index: Any=..., columns: Any=..., axis: Any=..., copy: Any=lib.no_default, inplace: Any=..., level: Any=..., errors: Any=...) -> None:
        ...

    def rename(self, mapper: Any=None, *, index: Any=None, columns: Any=None, axis: Any=None, copy: Any=lib.no_default, inplace: Any=False, level: Any=None, errors: Any='ignore') -> None:
        """
        Rename columns or index labels.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is. Extra labels listed don't throw an
        error.

        See the :ref:`user guide <basics.rename>` for more.

        Parameters
        ----------
        mapper : dict-like or function
            Dict-like or function transformations to apply to
            that axis' values. Use either ``mapper`` and ``axis`` to
            specify the axis to target with ``mapper``, or ``index`` and
            ``columns``.
        index : dict-like or function
            Alternative to specifying axis (``mapper, axis=0``
            is equivalent to ``index=mapper``).
        columns : dict-like or function
            Alternative to specifying axis (``mapper, axis=1``
            is equivalent to ``columns=mapper``).
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis to target with ``mapper``. Can be either the axis name
            ('index', 'columns') or number (0, 1). The default is 'index'.
        copy : bool, default False
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
            Whether to modify the DataFrame rather than creating a new one.
            If True then value of copy is ignored.
        level : int or level name, default None
            In case of a MultiIndex, only rename labels in the specified
            level.
        errors : {'ignore', 'raise'}, default 'ignore'
            If 'raise', raise a `KeyError` when a dict-like `mapper`, `index`,
            or `columns` contains labels that are not present in the Index
            being transformed.
            If 'ignore', existing keys will be renamed and extra keys will be
            ignored.

        Returns
        -------
        DataFrame or None
            DataFrame with the renamed axis labels or None if ``inplace=True``.

        Raises
        ------
        KeyError
            If any of the labels is not found in the selected axis and
            "errors='raise'".

        See Also
        --------
        DataFrame.rename_axis : Set the name of the axis.

        Examples
        --------
        ``DataFrame.rename`` supports two calling conventions

        * ``(index=index_mapper, columns=columns_mapper, ...)``
        * ``(mapper, axis={'index', 'columns'}, ...)``

        We *highly* recommend using keyword arguments to clarify your
        intent.

        Rename columns using a mapping:

        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> df.rename(columns={"A": "a", "B": "c"})
           a  c
        0  1  4
        1  2  5
        2  3  6

        Rename index using a mapping:

        >>> df.rename(index={0: "x", 1: "y", 2: "z"})
           A  B
        x  1  4
        y  2  5
        z  3  6

        Cast index labels to a different type:

        >>> df.index
        RangeIndex(start=0, stop=3, step=1)
        >>> df.rename(index=str).index
        Index(['0', '1', '2'], dtype='object')

        >>> df.rename(columns={"A": "a", "B": "b", "C": "c"}, errors="raise")
        Traceback (most recent call last):
        KeyError: ['C'] not found in axis

        Using axis-style parameters:

        >>> df.rename(str.lower, axis="columns")
           a  b
        0  1  4
        1  2  5
        2  3  6

        >>> df.rename({1: 2, 2: 4}, axis="index")
           A  B
        0  1  4
        2  2  5
        4  3  6
        """
        self._check_copy_deprecation(copy)
        return super()._rename(mapper=mapper, index=index, columns=columns, axis=axis, inplace=inplace, level=level, errors=errors)

    def pop(self, item: Any):
        """
        Return item and drop it from DataFrame. Raise KeyError if not found.

        Parameters
        ----------
        item : label
            Label of column to be popped.

        Returns
        -------
        Series
            Series representing the item that is dropped.

        See Also
        --------
        DataFrame.drop: Drop specified labels from rows or columns.
        DataFrame.drop_duplicates: Return DataFrame with duplicate rows removed.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         ("falcon", "bird", 389.0),
        ...         ("parrot", "bird", 24.0),
        ...         ("lion", "mammal", 80.5),
        ...         ("monkey", "mammal", np.nan),
        ...     ],
        ...     columns=("name", "class", "max_speed"),
        ... )
        >>> df
             name   class  max_speed
        0  falcon    bird      389.0
        1  parrot    bird       24.0
        2    lion  mammal       80.5
        3  monkey  mammal        NaN

        >>> df.pop("class")
        0      bird
        1      bird
        2    mammal
        3    mammal
        Name: class, dtype: object

        >>> df
             name  max_speed
        0  falcon      389.0
        1  parrot       24.0
        2    lion       80.5
        3  monkey        NaN
        """
        return super().pop(item=item)

    @overload
    def _replace_columnwise(self, mapping: Any, inplace: Any, regex: Any) -> None:
        ...

    @overload
    def _replace_columnwise(self, mapping: Any, inplace: Any, regex: Any) -> None:
        ...

    def _replace_columnwise(self, mapping: Any, inplace: Any, regex: Any) -> None:
        """
        Dispatch to Series.replace column-wise.

        Parameters
        ----------
        mapping : dict
            of the form {col: (target, value)}
        inplace : bool
        regex : bool or same types as `to_replace` in DataFrame.replace

        Returns
        -------
        DataFrame or None
        """
        res = self if inplace else self.copy(deep=False)
        ax = self.columns
        for i, ax_value in enumerate(ax):
            if ax_value in mapping:
                ser = self.iloc[:, i]
                target, value = mapping[ax_value]
                newobj = ser.replace(target, value, regex=regex)
                res._iset_item(i, newobj, inplace=inplace)
        if inplace:
            return None
        return res.__finalize__(self)

    @doc(NDFrame.shift, klass=_shared_doc_kwargs['klass'])
    def shift(self, periods: int=1, freq: None=None, axis: int=0, fill_value: Any=lib.no_default, suffix: None=None) -> list[int]:
        if freq is not None and fill_value is not lib.no_default:
            raise ValueError("Passing a 'freq' together with a 'fill_value' is not allowed.")
        if self.empty and freq is None:
            return self.copy()
        axis = self._get_axis_number(axis)
        if is_list_like(periods):
            periods = cast(Sequence, periods)
            if axis == 1:
                raise ValueError('If `periods` contains multiple shifts, `axis` cannot be 1.')
            if len(periods) == 0:
                raise ValueError('If `periods` is an iterable, it cannot be empty.')
            from pandas.core.reshape.concat import concat
            shifted_dataframes = []
            for period in periods:
                if not is_integer(period):
                    raise TypeError(f'Periods must be integer, but {period} is {type(period)}.')
                period = cast(int, period)
                shifted_dataframes.append(super().shift(periods=period, freq=freq, axis=axis, fill_value=fill_value).add_suffix(f'{suffix}_{period}' if suffix else f'_{period}'))
            return concat(shifted_dataframes, axis=1)
        elif suffix:
            raise ValueError('Cannot specify `suffix` if `periods` is an int.')
        periods = cast(int, periods)
        ncols = len(self.columns)
        if axis == 1 and periods != 0 and (ncols > 0) and (freq is None):
            if fill_value is lib.no_default:
                label = self.columns[0]
                if periods > 0:
                    result = self.iloc[:, :-periods]
                    for col in range(min(ncols, abs(periods))):
                        filler = self.iloc[:, 0].shift(len(self))
                        result.insert(0, label, filler, allow_duplicates=True)
                else:
                    result = self.iloc[:, -periods:]
                    for col in range(min(ncols, abs(periods))):
                        filler = self.iloc[:, -1].shift(len(self))
                        result.insert(len(result.columns), label, filler, allow_duplicates=True)
                result.columns = self.columns.copy()
                return result
            elif len(self._mgr.blocks) > 1 or not can_hold_element(self._mgr.blocks[0].values, fill_value):
                nper = abs(periods)
                nper = min(nper, ncols)
                if periods > 0:
                    indexer = np.array([-1] * nper + list(range(ncols - periods)), dtype=np.intp)
                else:
                    indexer = np.array(list(range(nper, ncols)) + [-1] * nper, dtype=np.intp)
                mgr = self._mgr.reindex_indexer(self.columns, indexer, axis=0, fill_value=fill_value, allow_dups=True)
                res_df = self._constructor_from_mgr(mgr, axes=mgr.axes)
                return res_df.__finalize__(self, method='shift')
            else:
                return self.T.shift(periods=periods, fill_value=fill_value).T
        return super().shift(periods=periods, freq=freq, axis=axis, fill_value=fill_value)

    @overload
    def set_index(self, keys: Any, *, drop: Any=..., append: Any=..., inplace: Any=..., verify_integrity: Any=...) -> None:
        ...

    @overload
    def set_index(self, keys: Any, *, drop: Any=..., append: Any=..., inplace: Any, verify_integrity: Any=...) -> None:
        ...

    def set_index(self, keys: Any, *, drop: Any=True, append: Any=False, inplace: Any=False, verify_integrity: Any=False) -> None:
        """
        Set the DataFrame index using existing columns.

        Set the DataFrame index (row labels) using one or more existing
        columns or arrays (of the correct length). The index can replace the
        existing index or expand on it.

        Parameters
        ----------
        keys : label or array-like or list of labels/arrays
            This parameter can be either a single column key, a single array of
            the same length as the calling DataFrame, or a list containing an
            arbitrary combination of column keys and arrays. Here, "array"
            encompasses :class:`Series`, :class:`Index`, ``np.ndarray``, and
            instances of :class:`~collections.abc.Iterator`.
        drop : bool, default True
            Delete columns to be used as the new index.
        append : bool, default False
            Whether to append columns to existing index.
        inplace : bool, default False
            Whether to modify the DataFrame rather than creating a new one.
        verify_integrity : bool, default False
            Check the new index for duplicates. Otherwise defer the check until
            necessary. Setting to False will improve the performance of this
            method.

        Returns
        -------
        DataFrame or None
            Changed row labels or None if ``inplace=True``.

        See Also
        --------
        DataFrame.reset_index : Opposite of set_index.
        DataFrame.reindex : Change to new indices or expand indices.
        DataFrame.reindex_like : Change to same indices as other DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "month": [1, 4, 7, 10],
        ...         "year": [2012, 2014, 2013, 2014],
        ...         "sale": [55, 40, 84, 31],
        ...     }
        ... )
        >>> df
           month  year  sale
        0      1  2012    55
        1      4  2014    40
        2      7  2013    84
        3     10  2014    31

        Set the index to become the 'month' column:

        >>> df.set_index("month")
               year  sale
        month
        1      2012    55
        4      2014    40
        7      2013    84
        10     2014    31

        Create a MultiIndex using columns 'year' and 'month':

        >>> df.set_index(["year", "month"])
                    sale
        year  month
        2012  1     55
        2014  4     40
        2013  7     84
        2014  10    31

        Create a MultiIndex using an Index and a column:

        >>> df.set_index([pd.Index([1, 2, 3, 4]), "year"])
                 month  sale
           year
        1  2012  1      55
        2  2014  4      40
        3  2013  7      84
        4  2014  10     31

        Create a MultiIndex using two Series:

        >>> s = pd.Series([1, 2, 3, 4])
        >>> df.set_index([s, s**2])
              month  year  sale
        1 1       1  2012    55
        2 4       4  2014    40
        3 9       7  2013    84
        4 16     10  2014    31
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        self._check_inplace_and_allows_duplicate_labels(inplace)
        if not isinstance(keys, list):
            keys = [keys]
        err_msg = 'The parameter "keys" may be a column key, one-dimensional array, or a list containing only valid column keys and one-dimensional arrays.'
        missing = []
        for col in keys:
            if isinstance(col, (Index, Series, np.ndarray, list, abc.Iterator)):
                if getattr(col, 'ndim', 1) != 1:
                    raise ValueError(err_msg)
            else:
                try:
                    found = col in self.columns
                except TypeError as err:
                    raise TypeError(f'{err_msg}. Received column of type {type(col)}') from err
                else:
                    if not found:
                        missing.append(col)
        if missing:
            raise KeyError(f'None of {missing} are in the columns')
        if inplace:
            frame = self
        else:
            frame = self.copy(deep=False)
        arrays = []
        names = []
        if append:
            names = list(self.index.names)
            if isinstance(self.index, MultiIndex):
                arrays.extend((self.index._get_level_values(i) for i in range(self.index.nlevels)))
            else:
                arrays.append(self.index)
        to_remove = set()
        for col in keys:
            if isinstance(col, MultiIndex):
                arrays.extend((col._get_level_values(n) for n in range(col.nlevels)))
                names.extend(col.names)
            elif isinstance(col, (Index, Series)):
                arrays.append(col)
                names.append(col.name)
            elif isinstance(col, (list, np.ndarray)):
                arrays.append(col)
                names.append(None)
            elif isinstance(col, abc.Iterator):
                arrays.append(list(col))
                names.append(None)
            else:
                arrays.append(frame[col])
                names.append(col)
                if drop:
                    to_remove.add(col)
            if len(arrays[-1]) != len(self):
                raise ValueError(f'Length mismatch: Expected {len(self)} rows, received array of length {len(arrays[-1])}')
        index = ensure_index_from_sequences(arrays, names)
        if verify_integrity and (not index.is_unique):
            duplicates = index[index.duplicated()].unique()
            raise ValueError(f'Index has duplicate keys: {duplicates}')
        for c in to_remove:
            del frame[c]
        index._cleanup()
        frame.index = index
        if not inplace:
            return frame
        return None

    @overload
    def reset_index(self, level: Any=..., *, drop: Any=..., inplace: Any=..., col_level: Any=..., col_fill: Any=..., allow_duplicates: Any=..., names: None=None) -> None:
        ...

    @overload
    def reset_index(self, level: Any=..., *, drop: Any=..., inplace: Any, col_level: Any=..., col_fill: Any=..., allow_duplicates: Any=..., names: None=None) -> None:
        ...

    @overload
    def reset_index(self, level: Any=..., *, drop: Any=..., inplace: Any=..., col_level: Any=..., col_fill: Any=..., allow_duplicates: Any=..., names: None=None) -> None:
        ...

    def reset_index(self, level: Any=None, *, drop: Any=False, inplace: Any=False, col_level: Any=0, col_fill: Any='', allow_duplicates: Any=lib.no_default, names: None=None) -> None:
        """
        Reset the index, or a level of it.

        Reset the index of the DataFrame, and use the default one instead.
        If the DataFrame has a MultiIndex, this method can remove one or more
        levels.

        Parameters
        ----------
        level : int, str, tuple, or list, default None
            Only remove the given levels from the index. Removes all levels by
            default.
        drop : bool, default False
            Do not try to insert index into dataframe columns. This resets
            the index to the default integer index.
        inplace : bool, default False
            Whether to modify the DataFrame rather than creating a new one.
        col_level : int or str, default 0
            If the columns have multiple levels, determines which level the
            labels are inserted into. By default it is inserted into the first
            level.
        col_fill : object, default ''
            If the columns have multiple levels, determines how the other
            levels are named. If None then the index name is repeated.
        allow_duplicates : bool, optional, default lib.no_default
            Allow duplicate column labels to be created.

            .. versionadded:: 1.5.0

        names : int, str or 1-dimensional list, default None
            Using the given string, rename the DataFrame column which contains the
            index data. If the DataFrame has a MultiIndex, this has to be a list
            with length equal to the number of levels.

            .. versionadded:: 1.5.0

        Returns
        -------
        DataFrame or None
            DataFrame with the new index or None if ``inplace=True``.

        See Also
        --------
        DataFrame.set_index : Opposite of reset_index.
        DataFrame.reindex : Change to new indices or expand indices.
        DataFrame.reindex_like : Change to same indices as other DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [("bird", 389.0), ("bird", 24.0), ("mammal", 80.5), ("mammal", np.nan)],
        ...     index=["falcon", "parrot", "lion", "monkey"],
        ...     columns=("class", "max_speed"),
        ... )
        >>> df
                 class  max_speed
        falcon    bird      389.0
        parrot    bird       24.0
        lion    mammal       80.5
        monkey  mammal        NaN

        When we reset the index, the old index is added as a column, and a
        new sequential index is used:

        >>> df.reset_index()
            index   class  max_speed
        0  falcon    bird      389.0
        1  parrot    bird       24.0
        2    lion  mammal       80.5
        3  monkey  mammal        NaN

        We can use the `drop` parameter to avoid the old index being added as
        a column:

        >>> df.reset_index(drop=True)
            class  max_speed
        0    bird      389.0
        1    bird       24.0
        2  mammal       80.5
        3  mammal        NaN

        You can also use `reset_index` with `MultiIndex`.

        >>> index = pd.MultiIndex.from_tuples(
        ...     [
        ...         ("bird", "falcon"),
        ...         ("bird", "parrot"),
        ...         ("mammal", "lion"),
        ...         ("mammal", "monkey"),
        ...     ],
        ...     names=["class", "name"],
        ... )
        >>> columns = pd.MultiIndex.from_tuples([("speed", "max"), ("species", "type")])
        >>> df = pd.DataFrame(
        ...     [(389.0, "fly"), (24.0, "fly"), (80.5, "run"), (np.nan, "jump")],
        ...     index=index,
        ...     columns=columns,
        ... )
        >>> df
                       speed species
                         max    type
        class  name
        bird   falcon  389.0     fly
               parrot   24.0     fly
        mammal lion     80.5     run
               monkey    NaN    jump

        Using the `names` parameter, choose a name for the index column:

        >>> df.reset_index(names=["classes", "names"])
          classes   names  speed species
                             max    type
        0    bird  falcon  389.0     fly
        1    bird  parrot   24.0     fly
        2  mammal    lion   80.5     run
        3  mammal  monkey    NaN    jump

        If the index has multiple levels, we can reset a subset of them:

        >>> df.reset_index(level="class")
                 class  speed species
                          max    type
        name
        falcon    bird  389.0     fly
        parrot    bird   24.0     fly
        lion    mammal   80.5     run
        monkey  mammal    NaN    jump

        If we are not dropping the index, by default, it is placed in the top
        level. We can place it in another level:

        >>> df.reset_index(level="class", col_level=1)
                        speed species
                 class    max    type
        name
        falcon    bird  389.0     fly
        parrot    bird   24.0     fly
        lion    mammal   80.5     run
        monkey  mammal    NaN    jump

        When the index is inserted under another level, we can specify under
        which one with the parameter `col_fill`:

        >>> df.reset_index(level="class", col_level=1, col_fill="species")
                      species  speed species
                        class    max    type
        name
        falcon           bird  389.0     fly
        parrot           bird   24.0     fly
        lion           mammal   80.5     run
        monkey         mammal    NaN    jump

        If we specify a nonexistent level for `col_fill`, it is created:

        >>> df.reset_index(level="class", col_level=1, col_fill="genus")
                        genus  speed species
                        class    max    type
        name
        falcon           bird  389.0     fly
        parrot           bird   24.0     fly
        lion           mammal   80.5     run
        monkey         mammal    NaN    jump
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        self._check_inplace_and_allows_duplicate_labels(inplace)
        if inplace:
            new_obj = self
        else:
            new_obj = self.copy(deep=False)
        if allow_duplicates is not lib.no_default:
            allow_duplicates = validate_bool_kwarg(allow_duplicates, 'allow_duplicates')
        new_index = default_index(len(new_obj))
        if level is not None:
            if not isinstance(level, (tuple, list)):
                level = [level]
            level = [self.index._get_level_number(lev) for lev in level]
            if len(level) < self.index.nlevels:
                new_index = self.index.droplevel(level)
        if not drop:
            default = 'index' if 'index' not in self else 'level_0'
            names = self.index._get_default_index_names(names, default)
            if isinstance(self.index, MultiIndex):
                to_insert = zip(reversed(self.index.levels), reversed(self.index.codes))
            else:
                to_insert = ((self.index, None),)
            multi_col = isinstance(self.columns, MultiIndex)
            for j, (lev, lab) in enumerate(to_insert, start=1):
                i = self.index.nlevels - j
                if level is not None and i not in level:
                    continue
                name = names[i]
                if multi_col:
                    col_name = list(name) if isinstance(name, tuple) else [name]
                    if col_fill is None:
                        if len(col_name) not in (1, self.columns.nlevels):
                            raise ValueError(f'col_fill=None is incompatible with incomplete column name {name}')
                        col_fill = col_name[0]
                    lev_num = self.columns._get_level_number(col_level)
                    name_lst = [col_fill] * lev_num + col_name
                    missing = self.columns.nlevels - len(name_lst)
                    name_lst += [col_fill] * missing
                    name = tuple(name_lst)
                level_values = lev._values
                if level_values.dtype == np.object_:
                    level_values = lib.maybe_convert_objects(level_values)
                if lab is not None:
                    level_values = algorithms.take(level_values, lab, allow_fill=True, fill_value=lev._na_value)
                new_obj.insert(0, name, level_values, allow_duplicates=allow_duplicates)
        new_obj.index = new_index
        if not inplace:
            return new_obj
        return None

    @doc(NDFrame.isna, klass=_shared_doc_kwargs['klass'])
    def isna(self):
        res_mgr = self._mgr.isna(func=isna)
        result = self._constructor_from_mgr(res_mgr, axes=res_mgr.axes)
        return result.__finalize__(self, method='isna')

    @doc(NDFrame.isna, klass=_shared_doc_kwargs['klass'])
    def isnull(self):
        """
        DataFrame.isnull is an alias for DataFrame.isna.
        """
        return self.isna()

    @doc(NDFrame.notna, klass=_shared_doc_kwargs['klass'])
    def notna(self) -> int:
        return ~self.isna()

    @doc(NDFrame.notna, klass=_shared_doc_kwargs['klass'])
    def notnull(self) -> int:
        """
        DataFrame.notnull is an alias for DataFrame.notna.
        """
        return ~self.isna()

    @overload
    def dropna(self, *, axis: Any=..., how: Any=..., thresh: Any=..., subset: Any=..., inplace: Any=..., ignore_index: Any=...) -> None:
        ...

    @overload
    def dropna(self, *, axis: Any=..., how: Any=..., thresh: Any=..., subset: Any=..., inplace: Any, ignore_index: Any=...) -> None:
        ...

    def dropna(self, *, axis: Any=0, how: Any=lib.no_default, thresh: Any=lib.no_default, subset: Any=None, inplace: Any=False, ignore_index: Any=False) -> None:
        """
        Remove missing values.

        See the :ref:`User Guide <missing_data>` for more on which values are
        considered missing, and how to work with missing data.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Determine if rows or columns which contain missing values are
            removed.

            * 0, or 'index' : Drop rows which contain missing values.
            * 1, or 'columns' : Drop columns which contain missing value.

            Only a single axis is allowed.

        how : {'any', 'all'}, default 'any'
            Determine if row or column is removed from DataFrame, when we have
            at least one NA or all NA.

            * 'any' : If any NA values are present, drop that row or column.
            * 'all' : If all values are NA, drop that row or column.

        thresh : int, optional
            Require that many non-NA values. Cannot be combined with how.
        subset : column label or iterable of labels, optional
            Labels along other axis to consider, e.g. if you are dropping rows
            these would be a list of columns to include.
        inplace : bool, default False
            Whether to modify the DataFrame rather than creating a new one.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

            .. versionadded:: 2.0.0

        Returns
        -------
        DataFrame or None
            DataFrame with NA entries dropped from it or None if ``inplace=True``.

        See Also
        --------
        DataFrame.isna: Indicate missing values.
        DataFrame.notna : Indicate existing (non-missing) values.
        DataFrame.fillna : Replace missing values.
        Series.dropna : Drop missing values.
        Index.dropna : Drop missing indices.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "name": ["Alfred", "Batman", "Catwoman"],
        ...         "toy": [np.nan, "Batmobile", "Bullwhip"],
        ...         "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT],
        ...     }
        ... )
        >>> df
               name        toy       born
        0    Alfred        NaN        NaT
        1    Batman  Batmobile 1940-04-25
        2  Catwoman   Bullwhip        NaT

        Drop the rows where at least one element is missing.

        >>> df.dropna()
             name        toy       born
        1  Batman  Batmobile 1940-04-25

        Drop the columns where at least one element is missing.

        >>> df.dropna(axis="columns")
               name
        0    Alfred
        1    Batman
        2  Catwoman

        Drop the rows where all elements are missing.

        >>> df.dropna(how="all")
               name        toy       born
        0    Alfred        NaN        NaT
        1    Batman  Batmobile 1940-04-25
        2  Catwoman   Bullwhip        NaT

        Keep only the rows with at least 2 non-NA values.

        >>> df.dropna(thresh=2)
               name        toy       born
        1    Batman  Batmobile 1940-04-25
        2  Catwoman   Bullwhip        NaT

        Define in which columns to look for missing values.

        >>> df.dropna(subset=["name", "toy"])
               name        toy       born
        1    Batman  Batmobile 1940-04-25
        2  Catwoman   Bullwhip        NaT
        """
        if how is not lib.no_default and thresh is not lib.no_default:
            raise TypeError('You cannot set both the how and thresh arguments at the same time.')
        if how is lib.no_default:
            how = 'any'
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if isinstance(axis, (tuple, list)):
            raise TypeError('supplying multiple axes to axis is no longer supported.')
        axis = self._get_axis_number(axis)
        agg_axis = 1 - axis
        agg_obj = self
        if subset is not None:
            if not is_list_like(subset):
                subset = [cast(Hashable, subset)]
            ax = self._get_axis(agg_axis)
            indices = ax.get_indexer_for(subset)
            check = indices == -1
            if check.any():
                raise KeyError(np.array(subset)[check].tolist())
            agg_obj = self.take(indices, axis=agg_axis)
        if thresh is not lib.no_default:
            count = agg_obj.count(axis=agg_axis)
            mask = count >= thresh
        elif how == 'any':
            mask = notna(agg_obj).all(axis=agg_axis, bool_only=False)
        elif how == 'all':
            mask = notna(agg_obj).any(axis=agg_axis, bool_only=False)
        else:
            raise ValueError(f'invalid how option: {how}')
        if np.all(mask):
            result = self.copy(deep=False)
        else:
            result = self.loc(axis=axis)[mask]
        if ignore_index:
            result.index = default_index(len(result))
        if not inplace:
            return result
        self._update_inplace(result)
        return None

    @overload
    def drop_duplicates(self, subset: Any=..., *, keep: Any=..., inplace: Any, ignore_index: Any=...) -> None:
        ...

    @overload
    def drop_duplicates(self, subset: Any=..., *, keep: Any=..., inplace: Any=..., ignore_index: Any=...) -> None:
        ...

    @overload
    def drop_duplicates(self, subset: Any=..., *, keep: Any=..., inplace: Any=..., ignore_index: Any=...) -> None:
        ...

    def drop_duplicates(self, subset: Any=None, *, keep: Any='first', inplace: Any=False, ignore_index: Any=False) -> None:
        """
        Return DataFrame with duplicate rows removed.

        Considering certain columns is optional. Indexes, including time indexes
        are ignored.

        Parameters
        ----------
        subset : column label or iterable of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', ``False``}, default 'first'
            Determines which duplicates (if any) to keep.

            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        inplace : bool, default ``False``
            Whether to modify the DataFrame rather than creating a new one.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

        Returns
        -------
        DataFrame or None
            DataFrame with duplicates removed or None if ``inplace=True``.

        See Also
        --------
        DataFrame.value_counts: Count unique combinations of columns.

        Notes
        -----
        This method requires columns specified by ``subset`` to be of hashable type.
        Passing unhashable columns will raise a ``TypeError``.

        Examples
        --------
        Consider dataset containing ramen rating.

        >>> df = pd.DataFrame(
        ...     {
        ...         "brand": ["Yum Yum", "Yum Yum", "Indomie", "Indomie", "Indomie"],
        ...         "style": ["cup", "cup", "cup", "pack", "pack"],
        ...         "rating": [4, 4, 3.5, 15, 5],
        ...     }
        ... )
        >>> df
            brand style  rating
        0  Yum Yum   cup     4.0
        1  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        3  Indomie  pack    15.0
        4  Indomie  pack     5.0

        By default, it removes duplicate rows based on all columns.

        >>> df.drop_duplicates()
            brand style  rating
        0  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        3  Indomie  pack    15.0
        4  Indomie  pack     5.0

        To remove duplicates on specific column(s), use ``subset``.

        >>> df.drop_duplicates(subset=["brand"])
            brand style  rating
        0  Yum Yum   cup     4.0
        2  Indomie   cup     3.5

        To remove duplicates and keep last occurrences, use ``keep``.

        >>> df.drop_duplicates(subset=["brand", "style"], keep="last")
            brand style  rating
        1  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        4  Indomie  pack     5.0
        """
        if self.empty:
            return self.copy(deep=False)
        inplace = validate_bool_kwarg(inplace, 'inplace')
        ignore_index = validate_bool_kwarg(ignore_index, 'ignore_index')
        result = self[-self.duplicated(subset, keep=keep)]
        if ignore_index:
            result.index = default_index(len(result))
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def duplicated(self, subset: None=None, keep: typing.Text='first'):
        """
        Return boolean Series denoting duplicate rows.

        Considering certain columns is optional.

        Parameters
        ----------
        subset : column label or iterable of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', False}, default 'first'
            Determines which duplicates (if any) to mark.

            - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
            - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
            - False : Mark all duplicates as ``True``.

        Returns
        -------
        Series
            Boolean series for each duplicated rows.

        See Also
        --------
        Index.duplicated : Equivalent method on index.
        Series.duplicated : Equivalent method on Series.
        Series.drop_duplicates : Remove duplicate values from Series.
        DataFrame.drop_duplicates : Remove duplicate values from DataFrame.

        Examples
        --------
        Consider dataset containing ramen rating.

        >>> df = pd.DataFrame(
        ...     {
        ...         "brand": ["Yum Yum", "Yum Yum", "Indomie", "Indomie", "Indomie"],
        ...         "style": ["cup", "cup", "cup", "pack", "pack"],
        ...         "rating": [4, 4, 3.5, 15, 5],
        ...     }
        ... )
        >>> df
            brand style  rating
        0  Yum Yum   cup     4.0
        1  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        3  Indomie  pack    15.0
        4  Indomie  pack     5.0

        By default, for each set of duplicated values, the first occurrence
        is set on False and all others on True.

        >>> df.duplicated()
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True.

        >>> df.duplicated(keep="last")
        0     True
        1    False
        2    False
        3    False
        4    False
        dtype: bool

        By setting ``keep`` on False, all duplicates are True.

        >>> df.duplicated(keep=False)
        0     True
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        To find duplicates on specific column(s), use ``subset``.

        >>> df.duplicated(subset=["brand"])
        0    False
        1     True
        2    False
        3     True
        4     True
        dtype: bool
        """
        if self.empty:
            return self._constructor_sliced(dtype=bool)

        def f(vals: Any) -> tuple[int]:
            labels, shape = algorithms.factorize(vals, size_hint=len(self))
            return (labels.astype('i8'), len(shape))
        if subset is None:
            subset = self.columns
        elif not np.iterable(subset) or isinstance(subset, str) or (isinstance(subset, tuple) and subset in self.columns):
            subset = (subset,)
        subset = cast(Sequence, subset)
        diff = set(subset) - set(self.columns)
        if diff:
            raise KeyError(Index(diff))
        if len(subset) == 1 and self.columns.is_unique:
            result = self[next(iter(subset))].duplicated(keep)
            result.name = None
        else:
            vals = (col.values for name, col in self.items() if name in subset)
            labels, shape = map(list, zip(*map(f, vals)))
            ids = get_group_index(labels, tuple(shape), sort=False, xnull=False)
            result = self._constructor_sliced(duplicated(ids, keep), index=self.index)
        return result.__finalize__(self, method='duplicated')

    @overload
    def sort_values(self, by: Any, *, axis: Any=..., ascending: Any=..., inplace: Any=..., kind: Any=..., na_position: Any=..., ignore_index: Any=..., key: Any=...) -> None:
        ...

    @overload
    def sort_values(self, by: Any, *, axis: Any=..., ascending: Any=..., inplace: Any, kind: Any=..., na_position: Any=..., ignore_index: Any=..., key: Any=...) -> None:
        ...

    def sort_values(self, by: Any, *, axis: Any=0, ascending: Any=True, inplace: Any=False, kind: Any='quicksort', na_position: Any='last', ignore_index: Any=False, key: Any=None) -> None:
        """
        Sort by the values along either axis.

        Parameters
        ----------
        by : str or list of str
            Name or list of names to sort by.

            - if `axis` is 0 or `'index'` then `by` may contain index
              levels and/or column labels.
            - if `axis` is 1 or `'columns'` then `by` may contain column
              levels and/or index labels.
        axis : "{0 or 'index', 1 or 'columns'}", default 0
             Axis to be sorted.
        ascending : bool or list of bool, default True
             Sort ascending vs. descending. Specify list for multiple sort
             orders.  If this is a list of bools, must match the length of
             the by.
        inplace : bool, default False
             If True, perform operation in-place.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
             Choice of sorting algorithm. See also :func:`numpy.sort` for more
             information. `mergesort` and `stable` are the only stable algorithms. For
             DataFrames, this option is only applied when sorting on a single
             column or label.
        na_position : {'first', 'last'}, default 'last'
             Puts NaNs at the beginning if `first`; `last` puts NaNs at the
             end.
        ignore_index : bool, default False
             If True, the resulting axis will be labeled 0, 1, …, n - 1.
        key : callable, optional
            Apply the key function to the values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect a
            ``Series`` and return a Series with the same shape as the input.
            It will be applied to each column in `by` independently. The values in the
            returned Series will be used as the keys for sorting.

        Returns
        -------
        DataFrame or None
            DataFrame with sorted values or None if ``inplace=True``.

        See Also
        --------
        DataFrame.sort_index : Sort a DataFrame by the index.
        Series.sort_values : Similar method for a Series.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "col1": ["A", "A", "B", np.nan, "D", "C"],
        ...         "col2": [2, 1, 9, 8, 7, 4],
        ...         "col3": [0, 1, 9, 4, 2, 3],
        ...         "col4": ["a", "B", "c", "D", "e", "F"],
        ...     }
        ... )
        >>> df
          col1  col2  col3 col4
        0    A     2     0    a
        1    A     1     1    B
        2    B     9     9    c
        3  NaN     8     4    D
        4    D     7     2    e
        5    C     4     3    F

        **Sort by a single column**

        In this case, we are sorting the rows according to values in ``col1``:

        >>> df.sort_values(by=["col1"])
          col1  col2  col3 col4
        0    A     2     0    a
        1    A     1     1    B
        2    B     9     9    c
        5    C     4     3    F
        4    D     7     2    e
        3  NaN     8     4    D

        **Sort by multiple columns**

        You can also provide multiple columns to ``by`` argument, as shown below.
        In this example, the rows are first sorted according to ``col1``, and then
        the rows that have an identical value in ``col1`` are sorted according
        to ``col2``.

        >>> df.sort_values(by=["col1", "col2"])
          col1  col2  col3 col4
        1    A     1     1    B
        0    A     2     0    a
        2    B     9     9    c
        5    C     4     3    F
        4    D     7     2    e
        3  NaN     8     4    D

        **Sort in a descending order**

        The sort order can be reversed using ``ascending`` argument, as shown below:

        >>> df.sort_values(by="col1", ascending=False)
          col1  col2  col3 col4
        4    D     7     2    e
        5    C     4     3    F
        2    B     9     9    c
        0    A     2     0    a
        1    A     1     1    B
        3  NaN     8     4    D

        **Placing any** ``NA`` **first**

        Note that in the above example, the rows that contain an ``NA`` value in their
        ``col1`` are placed at the end of the dataframe. This behavior can be modified
        via ``na_position`` argument, as shown below:

        >>> df.sort_values(by="col1", ascending=False, na_position="first")
          col1  col2  col3 col4
        3  NaN     8     4    D
        4    D     7     2    e
        5    C     4     3    F
        2    B     9     9    c
        0    A     2     0    a
        1    A     1     1    B

        **Customized sort order**

        The ``key`` argument allows for a further customization of sorting behaviour.
        For example, you may want
        to ignore the `letter's case <https://en.wikipedia.org/wiki/Letter_case>`__
        when sorting strings:

        >>> df.sort_values(by="col4", key=lambda col: col.str.lower())
           col1  col2  col3 col4
        0    A     2     0    a
        1    A     1     1    B
        2    B     9     9    c
        3  NaN     8     4    D
        4    D     7     2    e
        5    C     4     3    F

        Another typical example is
        `natural sorting <https://en.wikipedia.org/wiki/Natural_sort_order>`__.
        This can be done using
        ``natsort`` `package <https://github.com/SethMMorton/natsort>`__,
        which provides sorted indices according
        to their natural order, as shown below:

        >>> df = pd.DataFrame(
        ...     {
        ...         "time": ["0hr", "128hr", "72hr", "48hr", "96hr"],
        ...         "value": [10, 20, 30, 40, 50],
        ...     }
        ... )
        >>> df
            time  value
        0    0hr     10
        1  128hr     20
        2   72hr     30
        3   48hr     40
        4   96hr     50
        >>> from natsort import index_natsorted
        >>> index_natsorted(df["time"])
        [0, 3, 2, 4, 1]
        >>> df.sort_values(
        ...     by="time",
        ...     key=lambda x: np.argsort(index_natsorted(x)),
        ... )
            time  value
        0    0hr     10
        3   48hr     40
        2   72hr     30
        4   96hr     50
        1  128hr     20
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        axis = self._get_axis_number(axis)
        ascending = validate_ascending(ascending)
        if not isinstance(by, list):
            by = [by]
        if is_sequence(ascending) and len(by) != len(ascending):
            raise ValueError(f'Length of ascending ({len(ascending)}) != length of by ({len(by)})')
        if len(by) > 1:
            keys = (self._get_label_or_level_values(x, axis=axis) for x in by)
            if key is not None:
                keys_data = [Series(k, name=name) for k, name in zip(keys, by)]
            else:
                keys_data = list(keys)
            indexer = lexsort_indexer(keys_data, orders=ascending, na_position=na_position, key=key)
        elif len(by):
            k = self._get_label_or_level_values(by[0], axis=axis)
            if key is not None:
                k = Series(k, name=by[0])
            if isinstance(ascending, (tuple, list)):
                ascending = ascending[0]
            indexer = nargsort(k, kind=kind, ascending=ascending, na_position=na_position, key=key)
        elif inplace:
            return self._update_inplace(self)
        else:
            return self.copy(deep=False)
        if is_range_indexer(indexer, len(indexer)):
            result = self.copy(deep=False)
            if ignore_index:
                result.index = default_index(len(result))
            if inplace:
                return self._update_inplace(result)
            else:
                return result
        new_data = self._mgr.take(indexer, axis=self._get_block_manager_axis(axis), verify=False)
        if ignore_index:
            new_data.set_axis(self._get_block_manager_axis(axis), default_index(len(indexer)))
        result = self._constructor_from_mgr(new_data, axes=new_data.axes)
        if inplace:
            return self._update_inplace(result)
        else:
            return result.__finalize__(self, method='sort_values')

    @overload
    def sort_index(self, *, axis: Any=..., level: Any=..., ascending: Any=..., inplace: Any, kind: Any=..., na_position: Any=..., sort_remaining: Any=..., ignore_index: Any=..., key: Any=...) -> None:
        ...

    @overload
    def sort_index(self, *, axis: Any=..., level: Any=..., ascending: Any=..., inplace: Any=..., kind: Any=..., na_position: Any=..., sort_remaining: Any=..., ignore_index: Any=..., key: Any=...) -> None:
        ...

    @overload
    def sort_index(self, *, axis: Any=..., level: Any=..., ascending: Any=..., inplace: Any=..., kind: Any=..., na_position: Any=..., sort_remaining: Any=..., ignore_index: Any=..., key: Any=...) -> None:
        ...

    def sort_index(self, *, axis: Any=0, level: Any=None, ascending: Any=True, inplace: Any=False, kind: Any='quicksort', na_position: Any='last', sort_remaining: Any=True, ignore_index: Any=False, key: Any=None) -> None:
        """
        Sort object by labels (along an axis).

        Returns a new DataFrame sorted by label if `inplace` argument is
        ``False``, otherwise updates the original DataFrame and returns None.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis along which to sort.  The value 0 identifies the rows,
            and 1 identifies the columns.
        level : int or level name or list of ints or list of level names
            If not None, sort on values in specified index level(s).
        ascending : bool or list-like of bools, default True
            Sort ascending vs. descending. When the index is a MultiIndex the
            sort direction can be controlled for each level individually.
        inplace : bool, default False
            Whether to modify the DataFrame rather than creating a new one.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See also :func:`numpy.sort` for more
            information. `mergesort` and `stable` are the only stable algorithms. For
            DataFrames, this option is only applied when sorting on a single
            column or label.
        na_position : {'first', 'last'}, default 'last'
            Puts NaNs at the beginning if `first`; `last` puts NaNs at the end.
            Not implemented for MultiIndex.
        sort_remaining : bool, default True
            If True and sorting by level and index is multilevel, sort by other
            levels too (in order) after sorting by specified level.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
        key : callable, optional
            If not None, apply the key function to the index values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect an
            ``Index`` and return an ``Index`` of the same shape. For MultiIndex
            inputs, the key is applied *per level*.

        Returns
        -------
        DataFrame or None
            The original DataFrame sorted by the labels or None if ``inplace=True``.

        See Also
        --------
        Series.sort_index : Sort Series by the index.
        DataFrame.sort_values : Sort DataFrame by the value.
        Series.sort_values : Sort Series by the value.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [1, 2, 3, 4, 5], index=[100, 29, 234, 1, 150], columns=["A"]
        ... )
        >>> df.sort_index()
             A
        1    4
        29   2
        100  1
        150  5
        234  3

        By default, it sorts in ascending order, to sort in descending order,
        use ``ascending=False``

        >>> df.sort_index(ascending=False)
             A
        234  3
        150  5
        100  1
        29   2
        1    4

        A key function can be specified which is applied to the index before
        sorting. For a ``MultiIndex`` this is applied to each level separately.

        >>> df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=["A", "b", "C", "d"])
        >>> df.sort_index(key=lambda x: x.str.lower())
           a
        A  1
        b  2
        C  3
        d  4
        """
        return super().sort_index(axis=axis, level=level, ascending=ascending, inplace=inplace, kind=kind, na_position=na_position, sort_remaining=sort_remaining, ignore_index=ignore_index, key=key)

    def value_counts(self, subset: None=None, normalize: bool=False, sort: bool=True, ascending: bool=False, dropna: bool=True):
        """
        Return a Series containing the frequency of each distinct row in the DataFrame.

        Parameters
        ----------
        subset : label or list of labels, optional
            Columns to use when counting unique combinations.
        normalize : bool, default False
            Return proportions rather than frequencies.
        sort : bool, default True
            Sort by frequencies when True. Preserve the order of the data when False.

            .. versionchanged:: 3.0.0

                Prior to 3.0.0, ``sort=False`` would sort by the columns values.
        ascending : bool, default False
            Sort in ascending order.
        dropna : bool, default True
            Do not include counts of rows that contain NA values.

            .. versionadded:: 1.3.0

        Returns
        -------
        Series
            Series containing the frequency of each distinct row in the DataFrame.

        See Also
        --------
        Series.value_counts: Equivalent method on Series.

        Notes
        -----
        The returned Series will have a MultiIndex with one level per input
        column but an Index (non-multi) for a single label. By default, rows
        that contain any NA values are omitted from the result. By default,
        the resulting Series will be sorted by frequencies in descending order so that
        the first element is the most frequently-occurring row.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        ...     index=["falcon", "dog", "cat", "ant"],
        ... )
        >>> df
                num_legs  num_wings
        falcon         2          2
        dog            4          0
        cat            4          0
        ant            6          0

        >>> df.value_counts()
        num_legs  num_wings
        4         0            2
        2         2            1
        6         0            1
        Name: count, dtype: int64

        >>> df.value_counts(sort=False)
        num_legs  num_wings
        2         2            1
        4         0            2
        6         0            1
        Name: count, dtype: int64

        >>> df.value_counts(ascending=True)
        num_legs  num_wings
        2         2            1
        6         0            1
        4         0            2
        Name: count, dtype: int64

        >>> df.value_counts(normalize=True)
        num_legs  num_wings
        4         0            0.50
        2         2            0.25
        6         0            0.25
        Name: proportion, dtype: float64

        With `dropna` set to `False` we can also count rows with NA values.

        >>> df = pd.DataFrame(
        ...     {
        ...         "first_name": ["John", "Anne", "John", "Beth"],
        ...         "middle_name": ["Smith", pd.NA, pd.NA, "Louise"],
        ...     }
        ... )
        >>> df
          first_name middle_name
        0       John       Smith
        1       Anne        <NA>
        2       John        <NA>
        3       Beth      Louise

        >>> df.value_counts()
        first_name  middle_name
        Beth        Louise         1
        John        Smith          1
        Name: count, dtype: int64

        >>> df.value_counts(dropna=False)
        first_name  middle_name
        Anne        NaN            1
        Beth        Louise         1
        John        Smith          1
                    NaN            1
        Name: count, dtype: int64

        >>> df.value_counts("first_name")
        first_name
        John    2
        Anne    1
        Beth    1
        Name: count, dtype: int64
        """
        if subset is None:
            subset = self.columns.tolist()
        name = 'proportion' if normalize else 'count'
        counts = self.groupby(subset, sort=False, dropna=dropna, observed=False)._grouper.size()
        counts.name = name
        if sort:
            counts = counts.sort_values(ascending=ascending)
        if normalize:
            counts /= counts.sum()
        if is_list_like(subset) and len(subset) == 1:
            counts.index = MultiIndex.from_arrays([counts.index], names=[counts.index.name])
        return counts

    def nlargest(self, n: Any, columns: Any, keep: typing.Text='first'):
        """
        Return the first `n` rows ordered by `columns` in descending order.

        Return the first `n` rows with the largest values in `columns`, in
        descending order. The columns that are not specified are returned as
        well, but not used for ordering.

        This method is equivalent to
        ``df.sort_values(columns, ascending=False).head(n)``, but more
        performant.

        Parameters
        ----------
        n : int
            Number of rows to return.
        columns : label or list of labels
            Column label(s) to order by.
        keep : {'first', 'last', 'all'}, default 'first'
            Where there are duplicate values:

            - ``first`` : prioritize the first occurrence(s)
            - ``last`` : prioritize the last occurrence(s)
            - ``all`` : keep all the ties of the smallest item even if it means
              selecting more than ``n`` items.

        Returns
        -------
        DataFrame
            The first `n` rows ordered by the given columns in descending
            order.

        See Also
        --------
        DataFrame.nsmallest : Return the first `n` rows ordered by `columns` in
            ascending order.
        DataFrame.sort_values : Sort DataFrame by the values.
        DataFrame.head : Return the first `n` rows without re-ordering.

        Notes
        -----
        This function cannot be used with all column types. For example, when
        specifying columns with `object` or `category` dtypes, ``TypeError`` is
        raised.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "population": [
        ...             59000000,
        ...             65000000,
        ...             434000,
        ...             434000,
        ...             434000,
        ...             337000,
        ...             11300,
        ...             11300,
        ...             11300,
        ...         ],
        ...         "GDP": [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311],
        ...         "alpha-2": ["IT", "FR", "MT", "MV", "BN", "IS", "NR", "TV", "AI"],
        ...     },
        ...     index=[
        ...         "Italy",
        ...         "France",
        ...         "Malta",
        ...         "Maldives",
        ...         "Brunei",
        ...         "Iceland",
        ...         "Nauru",
        ...         "Tuvalu",
        ...         "Anguilla",
        ...     ],
        ... )
        >>> df
                  population      GDP alpha-2
        Italy       59000000  1937894      IT
        France      65000000  2583560      FR
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN
        Iceland       337000    17036      IS
        Nauru          11300      182      NR
        Tuvalu         11300       38      TV
        Anguilla       11300      311      AI

        In the following example, we will use ``nlargest`` to select the three
        rows having the largest values in column "population".

        >>> df.nlargest(3, "population")
                population      GDP alpha-2
        France    65000000  2583560      FR
        Italy     59000000  1937894      IT
        Malta       434000    12011      MT

        When using ``keep='last'``, ties are resolved in reverse order:

        >>> df.nlargest(3, "population", keep="last")
                population      GDP alpha-2
        France    65000000  2583560      FR
        Italy     59000000  1937894      IT
        Brunei      434000    12128      BN

        When using ``keep='all'``, the number of element kept can go beyond ``n``
        if there are duplicate values for the smallest element, all the
        ties are kept:

        >>> df.nlargest(3, "population", keep="all")
                  population      GDP alpha-2
        France      65000000  2583560      FR
        Italy       59000000  1937894      IT
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN

        However, ``nlargest`` does not keep ``n`` distinct largest elements:

        >>> df.nlargest(5, "population", keep="all")
                  population      GDP alpha-2
        France      65000000  2583560      FR
        Italy       59000000  1937894      IT
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN

        To order by the largest values in column "population" and then "GDP",
        we can specify multiple columns like in the next example.

        >>> df.nlargest(3, ["population", "GDP"])
                population      GDP alpha-2
        France    65000000  2583560      FR
        Italy     59000000  1937894      IT
        Brunei      434000    12128      BN
        """
        return selectn.SelectNFrame(self, n=n, keep=keep, columns=columns).nlargest()

    def nsmallest(self, n: Any, columns: Any, keep: typing.Text='first'):
        """
        Return the first `n` rows ordered by `columns` in ascending order.

        Return the first `n` rows with the smallest values in `columns`, in
        ascending order. The columns that are not specified are returned as
        well, but not used for ordering.

        This method is equivalent to
        ``df.sort_values(columns, ascending=True).head(n)``, but more
        performant.

        Parameters
        ----------
        n : int
            Number of items to retrieve.
        columns : list or str
            Column name or names to order by.
        keep : {'first', 'last', 'all'}, default 'first'
            Where there are duplicate values:

            - ``first`` : take the first occurrence.
            - ``last`` : take the last occurrence.
            - ``all`` : keep all the ties of the largest item even if it means
              selecting more than ``n`` items.

        Returns
        -------
        DataFrame
            DataFrame with the first `n` rows ordered by `columns` in ascending order.

        See Also
        --------
        DataFrame.nlargest : Return the first `n` rows ordered by `columns` in
            descending order.
        DataFrame.sort_values : Sort DataFrame by the values.
        DataFrame.head : Return the first `n` rows without re-ordering.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "population": [
        ...             59000000,
        ...             65000000,
        ...             434000,
        ...             434000,
        ...             434000,
        ...             337000,
        ...             337000,
        ...             11300,
        ...             11300,
        ...         ],
        ...         "GDP": [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311],
        ...         "alpha-2": ["IT", "FR", "MT", "MV", "BN", "IS", "NR", "TV", "AI"],
        ...     },
        ...     index=[
        ...         "Italy",
        ...         "France",
        ...         "Malta",
        ...         "Maldives",
        ...         "Brunei",
        ...         "Iceland",
        ...         "Nauru",
        ...         "Tuvalu",
        ...         "Anguilla",
        ...     ],
        ... )
        >>> df
                  population      GDP alpha-2
        Italy       59000000  1937894      IT
        France      65000000  2583560      FR
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN
        Iceland       337000    17036      IS
        Nauru         337000      182      NR
        Tuvalu         11300       38      TV
        Anguilla       11300      311      AI

        In the following example, we will use ``nsmallest`` to select the
        three rows having the smallest values in column "population".

        >>> df.nsmallest(3, "population")
                  population    GDP alpha-2
        Tuvalu         11300     38      TV
        Anguilla       11300    311      AI
        Iceland       337000  17036      IS

        When using ``keep='last'``, ties are resolved in reverse order:

        >>> df.nsmallest(3, "population", keep="last")
                  population  GDP alpha-2
        Anguilla       11300  311      AI
        Tuvalu         11300   38      TV
        Nauru         337000  182      NR

        When using ``keep='all'``, the number of element kept can go beyond ``n``
        if there are duplicate values for the largest element, all the
        ties are kept.

        >>> df.nsmallest(3, "population", keep="all")
                  population    GDP alpha-2
        Tuvalu         11300     38      TV
        Anguilla       11300    311      AI
        Iceland       337000  17036      IS
        Nauru         337000    182      NR

        However, ``nsmallest`` does not keep ``n`` distinct
        smallest elements:

        >>> df.nsmallest(4, "population", keep="all")
                  population    GDP alpha-2
        Tuvalu         11300     38      TV
        Anguilla       11300    311      AI
        Iceland       337000  17036      IS
        Nauru         337000    182      NR

        To order by the smallest values in column "population" and then "GDP", we can
        specify multiple columns like in the next example.

        >>> df.nsmallest(3, ["population", "GDP"])
                  population  GDP alpha-2
        Tuvalu         11300   38      TV
        Anguilla       11300  311      AI
        Nauru         337000  182      NR
        """
        return selectn.SelectNFrame(self, n=n, keep=keep, columns=columns).nsmallest()

    def swaplevel(self, i: int=-2, j: int=-1, axis: int=0):
        """
        Swap levels i and j in a :class:`MultiIndex`.

        Default is to swap the two innermost levels of the index.

        Parameters
        ----------
        i, j : int or str
            Levels of the indices to be swapped. Can pass level name as string.
        axis : {0 or 'index', 1 or 'columns'}, default 0
                    The axis to swap levels on. 0 or 'index' for row-wise, 1 or
                    'columns' for column-wise.

        Returns
        -------
        DataFrame
            DataFrame with levels swapped in MultiIndex.

        See Also
        --------
        DataFrame.reorder_levels: Reorder levels of MultiIndex.
        DataFrame.sort_index: Sort MultiIndex.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"Grade": ["A", "B", "A", "C"]},
        ...     index=[
        ...         ["Final exam", "Final exam", "Coursework", "Coursework"],
        ...         ["History", "Geography", "History", "Geography"],
        ...         ["January", "February", "March", "April"],
        ...     ],
        ... )
        >>> df
                                            Grade
        Final exam  History     January      A
                    Geography   February     B
        Coursework  History     March        A
                    Geography   April        C

        In the following example, we will swap the levels of the indices.
        Here, we will swap the levels column-wise, but levels can be swapped row-wise
        in a similar manner. Note that column-wise is the default behaviour.
        By not supplying any arguments for i and j, we swap the last and second to
        last indices.

        >>> df.swaplevel()
                                            Grade
        Final exam  January     History         A
                    February    Geography       B
        Coursework  March       History         A
                    April       Geography       C

        By supplying one argument, we can choose which index to swap the last
        index with. We can for example swap the first index with the last one as
        follows.

        >>> df.swaplevel(0)
                                            Grade
        January     History     Final exam      A
        February    Geography   Final exam      B
        March       History     Coursework      A
        April       Geography   Coursework      C

        We can also define explicitly which indices we want to swap by supplying values
        for both i and j. Here, we for example swap the first and second indices.

        >>> df.swaplevel(0, 1)
                                            Grade
        History     Final exam  January         A
        Geography   Final exam  February        B
        History     Coursework  March           A
        Geography   Coursework  April           C
        """
        result = self.copy(deep=False)
        axis = self._get_axis_number(axis)
        if not isinstance(result._get_axis(axis), MultiIndex):
            raise TypeError('Can only swap levels on a hierarchical axis.')
        if axis == 0:
            assert isinstance(result.index, MultiIndex)
            result.index = result.index.swaplevel(i, j)
        else:
            assert isinstance(result.columns, MultiIndex)
            result.columns = result.columns.swaplevel(i, j)
        return result

    def reorder_levels(self, order: Any, axis: int=0):
        """
        Rearrange index or column levels using input ``order``.

        May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int or list of str
            List representing new level order. Reference level by number
            (position) or by key (label).
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Where to reorder levels.

        Returns
        -------
        DataFrame
            DataFrame with indices or columns with reordered levels.

        See Also
        --------
            DataFrame.swaplevel : Swap levels i and j in a MultiIndex.

        Examples
        --------
        >>> data = {
        ...     "class": ["Mammals", "Mammals", "Reptiles"],
        ...     "diet": ["Omnivore", "Carnivore", "Carnivore"],
        ...     "species": ["Humans", "Dogs", "Snakes"],
        ... }
        >>> df = pd.DataFrame(data, columns=["class", "diet", "species"])
        >>> df = df.set_index(["class", "diet"])
        >>> df
                                          species
        class      diet
        Mammals    Omnivore                Humans
                   Carnivore                 Dogs
        Reptiles   Carnivore               Snakes

        Let's reorder the levels of the index:

        >>> df.reorder_levels(["diet", "class"])
                                          species
        diet      class
        Omnivore  Mammals                  Humans
        Carnivore Mammals                    Dogs
                  Reptiles                 Snakes
        """
        axis = self._get_axis_number(axis)
        if not isinstance(self._get_axis(axis), MultiIndex):
            raise TypeError('Can only reorder levels on a hierarchical axis.')
        result = self.copy(deep=False)
        if axis == 0:
            assert isinstance(result.index, MultiIndex)
            result.index = result.index.reorder_levels(order)
        else:
            assert isinstance(result.columns, MultiIndex)
            result.columns = result.columns.reorder_levels(order)
        return result

    def _cmp_method(self, other: Any, op: Any):
        axis = 1
        self, other = self._align_for_op(other, axis, flex=False, level=None)
        new_data = self._dispatch_frame_op(other, op, axis=axis)
        return self._construct_result(new_data)

    def _arith_method(self, other: Any, op: Any):
        if self._should_reindex_frame_op(other, op, 1, None, None):
            return self._arith_method_with_reindex(other, op)
        axis = 1
        other = ops.maybe_prepare_scalar_for_op(other, (self.shape[axis],))
        self, other = self._align_for_op(other, axis, flex=True, level=None)
        with np.errstate(all='ignore'):
            new_data = self._dispatch_frame_op(other, op, axis=axis)
        return self._construct_result(new_data)
    _logical_method = _arith_method

    def _dispatch_frame_op(self, right: Any, func: Any, axis: None=None):
        """
        Evaluate the frame operation func(left, right) by evaluating
        column-by-column, dispatching to the Series implementation.

        Parameters
        ----------
        right : scalar, Series, or DataFrame
        func : arithmetic or comparison operator
        axis : {None, 0, 1}

        Returns
        -------
        DataFrame

        Notes
        -----
        Caller is responsible for setting np.errstate where relevant.
        """
        array_op = ops.get_array_op(func)
        right = lib.item_from_zerodim(right)
        if not is_list_like(right):
            bm = self._mgr.apply(array_op, right=right)
            return self._constructor_from_mgr(bm, axes=bm.axes)
        elif isinstance(right, DataFrame):
            assert self.index.equals(right.index)
            assert self.columns.equals(right.columns)
            bm = self._mgr.operate_blockwise(right._mgr, array_op)
            return self._constructor_from_mgr(bm, axes=bm.axes)
        elif isinstance(right, Series) and axis == 1:
            assert right.index.equals(self.columns)
            right = right._values
            assert not isinstance(right, np.ndarray)
            arrays = [array_op(_left, _right) for _left, _right in zip(self._iter_column_arrays(), right)]
        elif isinstance(right, Series):
            assert right.index.equals(self.index)
            right = right._values
            arrays = [array_op(left, right) for left in self._iter_column_arrays()]
        else:
            raise NotImplementedError(right)
        return type(self)._from_arrays(arrays, self.columns, self.index, verify_integrity=False)

    def _combine_frame(self, other: Any, func: Any, fill_value: None=None):
        if fill_value is None:
            _arith_op = func
        else:

            def _arith_op(left: Any, right: Any):
                left, right = ops.fill_binop(left, right, fill_value)
                return func(left, right)
        new_data = self._dispatch_frame_op(other, _arith_op)
        return new_data

    def _arith_method_with_reindex(self, right: Any, op: Any):
        """
        For DataFrame-with-DataFrame operations that require reindexing,
        operate only on shared columns, then reindex.

        Parameters
        ----------
        right : DataFrame
        op : binary operator

        Returns
        -------
        DataFrame
        """
        left = self
        cols, lcol_indexer, rcol_indexer = left.columns.join(right.columns, how='inner', return_indexers=True)
        new_left = left if lcol_indexer is None else left.iloc[:, lcol_indexer]
        new_right = right if rcol_indexer is None else right.iloc[:, rcol_indexer]
        if isinstance(cols, MultiIndex):
            new_left = new_left.copy(deep=False)
            new_right = new_right.copy(deep=False)
            new_left.columns = cols
            new_right.columns = cols
        result = op(new_left, new_right)
        join_columns = left.columns.join(right.columns, how='outer')
        if result.columns.has_duplicates:
            indexer, _ = result.columns.get_indexer_non_unique(join_columns)
            indexer = algorithms.unique1d(indexer)
            result = result._reindex_with_indexers({1: [join_columns, indexer]}, allow_dups=True)
        else:
            result = result.reindex(join_columns, axis=1)
        return result

    def _should_reindex_frame_op(self, right: Any, op: Any, axis: Any, fill_value: Any, level: Any) -> bool:
        """
        Check if this is an operation between DataFrames that will need to reindex.
        """
        if op is operator.pow or op is roperator.rpow:
            return False
        if not isinstance(right, DataFrame):
            return False
        if (isinstance(self.columns, MultiIndex) or isinstance(right.columns, MultiIndex)) and (not self.columns.equals(right.columns)) and (fill_value is None):
            return True
        if fill_value is None and level is None and (axis == 1):
            left_uniques = self.columns.unique()
            right_uniques = right.columns.unique()
            cols = left_uniques.intersection(right_uniques)
            if len(cols) and (not (len(cols) == len(left_uniques) and len(cols) == len(right_uniques))):
                return True
        return False

    def _align_for_op(self, other: Any, axis: Any, flex: bool=False, level: None=None):
        """
        Convert rhs to meet lhs dims if input is list, tuple or np.ndarray.

        Parameters
        ----------
        left : DataFrame
        right : Any
        axis : int
        flex : bool or None, default False
            Whether this is a flex op, in which case we reindex.
            None indicates not to check for alignment.
        level : int or level name, default None

        Returns
        -------
        left : DataFrame
        right : Any
        """
        left, right = (self, other)

        def to_series(right: Any):
            msg = 'Unable to coerce to Series, length must be {req_len}: given {given_len}'
            dtype = None
            if getattr(right, 'dtype', None) == object:
                dtype = object
            if axis == 0:
                if len(left.index) != len(right):
                    raise ValueError(msg.format(req_len=len(left.index), given_len=len(right)))
                right = left._constructor_sliced(right, index=left.index, dtype=dtype)
            else:
                if len(left.columns) != len(right):
                    raise ValueError(msg.format(req_len=len(left.columns), given_len=len(right)))
                right = left._constructor_sliced(right, index=left.columns, dtype=dtype)
            return right
        if isinstance(right, np.ndarray):
            if right.ndim == 1:
                right = to_series(right)
            elif right.ndim == 2:
                dtype = None
                if right.dtype == object:
                    dtype = object
                if right.shape == left.shape:
                    right = left._constructor(right, index=left.index, columns=left.columns, dtype=dtype)
                elif right.shape[0] == left.shape[0] and right.shape[1] == 1:
                    right = np.broadcast_to(right, left.shape)
                    right = left._constructor(right, index=left.index, columns=left.columns, dtype=dtype)
                elif right.shape[1] == left.shape[1] and right.shape[0] == 1:
                    right = to_series(right[0, :])
                else:
                    raise ValueError(f'Unable to coerce to DataFrame, shape must be {left.shape}: given {right.shape}')
            elif right.ndim > 2:
                raise ValueError(f'Unable to coerce to Series/DataFrame, dimension must be <= 2: {right.shape}')
        elif is_list_like(right) and (not isinstance(right, (Series, DataFrame))):
            if any((is_array_like(el) for el in right)):
                raise ValueError(f'Unable to coerce list of {type(right[0])} to Series/DataFrame')
            right = to_series(right)
        if flex is not None and isinstance(right, DataFrame):
            if not left._indexed_same(right):
                if flex:
                    left, right = left.align(right, join='outer', level=level)
                else:
                    raise ValueError('Can only compare identically-labeled (both index and columns) DataFrame objects')
        elif isinstance(right, Series):
            axis = axis if axis is not None else 1
            if not flex:
                if not left.axes[axis].equals(right.index):
                    raise ValueError('Operands are not aligned. Do `left, right = left.align(right, axis=1)` before operating.')
            left, right = left.align(right, join='outer', axis=axis, level=level)
            right = left._maybe_align_series_as_frame(right, axis)
        return (left, right)

    def _maybe_align_series_as_frame(self, series: Any, axis: Any):
        """
        If the Series operand is not EA-dtype, we can broadcast to 2D and operate
        blockwise.
        """
        rvalues = series._values
        if not isinstance(rvalues, np.ndarray):
            if rvalues.dtype in ('datetime64[ns]', 'timedelta64[ns]'):
                rvalues = np.asarray(rvalues)
            else:
                return series
        if axis == 0:
            rvalues = rvalues.reshape(-1, 1)
        else:
            rvalues = rvalues.reshape(1, -1)
        rvalues = np.broadcast_to(rvalues, self.shape)
        return self._constructor(rvalues, index=self.index, columns=self.columns, dtype=rvalues.dtype)

    def _flex_arith_method(self, other: Any, op: Any, *, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        axis = self._get_axis_number(axis) if axis is not None else 1
        if self._should_reindex_frame_op(other, op, axis, fill_value, level):
            return self._arith_method_with_reindex(other, op)
        if isinstance(other, Series) and fill_value is not None:
            raise NotImplementedError(f'fill_value {fill_value} not supported.')
        other = ops.maybe_prepare_scalar_for_op(other, self.shape)
        self, other = self._align_for_op(other, axis, flex=True, level=level)
        with np.errstate(all='ignore'):
            if isinstance(other, DataFrame):
                new_data = self._combine_frame(other, op, fill_value)
            elif isinstance(other, Series):
                new_data = self._dispatch_frame_op(other, op, axis=axis)
            else:
                if fill_value is not None:
                    self = self.fillna(fill_value)
                new_data = self._dispatch_frame_op(other, op)
        return self._construct_result(new_data)

    def _construct_result(self, result: Any):
        """
        Wrap the result of an arithmetic, comparison, or logical operation.

        Parameters
        ----------
        result : DataFrame

        Returns
        -------
        DataFrame
        """
        out = self._constructor(result, copy=False).__finalize__(self)
        out.columns = self.columns
        out.index = self.index
        return out

    def __divmod__(self, other: Any) -> tuple:
        div = self // other
        mod = self - div * other
        return (div, mod)

    def __rdivmod__(self, other: Any) -> tuple:
        div = other // self
        mod = other - div * self
        return (div, mod)

    def _flex_cmp_method(self, other: Any, op: Any, *, axis: typing.Text='columns', level: None=None):
        axis = self._get_axis_number(axis) if axis is not None else 1
        self, other = self._align_for_op(other, axis, flex=True, level=level)
        new_data = self._dispatch_frame_op(other, op, axis=axis)
        return self._construct_result(new_data)

    @Appender(ops.make_flex_doc('eq', 'dataframe'))
    def eq(self, other: Any, axis: typing.Text='columns', level: None=None):
        return self._flex_cmp_method(other, operator.eq, axis=axis, level=level)

    @Appender(ops.make_flex_doc('ne', 'dataframe'))
    def ne(self, other: Any, axis: typing.Text='columns', level: None=None):
        return self._flex_cmp_method(other, operator.ne, axis=axis, level=level)

    @Appender(ops.make_flex_doc('le', 'dataframe'))
    def le(self, other: Any, axis: typing.Text='columns', level: None=None):
        return self._flex_cmp_method(other, operator.le, axis=axis, level=level)

    @Appender(ops.make_flex_doc('lt', 'dataframe'))
    def lt(self, other: Any, axis: typing.Text='columns', level: None=None):
        return self._flex_cmp_method(other, operator.lt, axis=axis, level=level)

    @Appender(ops.make_flex_doc('ge', 'dataframe'))
    def ge(self, other: Any, axis: typing.Text='columns', level: None=None):
        return self._flex_cmp_method(other, operator.ge, axis=axis, level=level)

    @Appender(ops.make_flex_doc('gt', 'dataframe'))
    def gt(self, other: Any, axis: typing.Text='columns', level: None=None):
        return self._flex_cmp_method(other, operator.gt, axis=axis, level=level)

    @Appender(ops.make_flex_doc('add', 'dataframe'))
    def add(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, operator.add, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('radd', 'dataframe'))
    def radd(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, roperator.radd, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('sub', 'dataframe'))
    def sub(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, operator.sub, level=level, fill_value=fill_value, axis=axis)
    subtract = sub

    @Appender(ops.make_flex_doc('rsub', 'dataframe'))
    def rsub(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, roperator.rsub, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('mul', 'dataframe'))
    def mul(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, operator.mul, level=level, fill_value=fill_value, axis=axis)
    multiply = mul

    @Appender(ops.make_flex_doc('rmul', 'dataframe'))
    def rmul(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, roperator.rmul, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('truediv', 'dataframe'))
    def truediv(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, operator.truediv, level=level, fill_value=fill_value, axis=axis)
    div = truediv
    divide = truediv

    @Appender(ops.make_flex_doc('rtruediv', 'dataframe'))
    def rtruediv(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, roperator.rtruediv, level=level, fill_value=fill_value, axis=axis)
    rdiv = rtruediv

    @Appender(ops.make_flex_doc('floordiv', 'dataframe'))
    def floordiv(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, operator.floordiv, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('rfloordiv', 'dataframe'))
    def rfloordiv(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, roperator.rfloordiv, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('mod', 'dataframe'))
    def mod(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, operator.mod, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('rmod', 'dataframe'))
    def rmod(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, roperator.rmod, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('pow', 'dataframe'))
    def pow(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, operator.pow, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('rpow', 'dataframe'))
    def rpow(self, other: Any, axis: typing.Text='columns', level: None=None, fill_value: None=None):
        return self._flex_arith_method(other, roperator.rpow, level=level, fill_value=fill_value, axis=axis)

    @doc(_shared_docs['compare'], dedent('\n        Returns\n        -------\n        DataFrame\n            DataFrame that shows the differences stacked side by side.\n\n            The resulting index will be a MultiIndex with \'self\' and \'other\'\n            stacked alternately at the inner level.\n\n        Raises\n        ------\n        ValueError\n            When the two DataFrames don\'t have identical labels or shape.\n\n        See Also\n        --------\n        Series.compare : Compare with another Series and show differences.\n        DataFrame.equals : Test whether two objects contain the same elements.\n\n        Notes\n        -----\n        Matching NaNs will not appear as a difference.\n\n        Can only compare identically-labeled\n        (i.e. same shape, identical row and column labels) DataFrames\n\n        Examples\n        --------\n        >>> df = pd.DataFrame(\n        ...     {{\n        ...         "col1": ["a", "a", "b", "b", "a"],\n        ...         "col2": [1.0, 2.0, 3.0, np.nan, 5.0],\n        ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0]\n        ...     }},\n        ...     columns=["col1", "col2", "col3"],\n        ... )\n        >>> df\n          col1  col2  col3\n        0    a   1.0   1.0\n        1    a   2.0   2.0\n        2    b   3.0   3.0\n        3    b   NaN   4.0\n        4    a   5.0   5.0\n\n        >>> df2 = df.copy()\n        >>> df2.loc[0, \'col1\'] = \'c\'\n        >>> df2.loc[2, \'col3\'] = 4.0\n        >>> df2\n          col1  col2  col3\n        0    c   1.0   1.0\n        1    a   2.0   2.0\n        2    b   3.0   4.0\n        3    b   NaN   4.0\n        4    a   5.0   5.0\n\n        Align the differences on columns\n\n        >>> df.compare(df2)\n          col1       col3\n          self other self other\n        0    a     c  NaN   NaN\n        2  NaN   NaN  3.0   4.0\n\n        Assign result_names\n\n        >>> df.compare(df2, result_names=("left", "right"))\n          col1       col3\n          left right left right\n        0    a     c  NaN   NaN\n        2  NaN   NaN  3.0   4.0\n\n        Stack the differences on rows\n\n        >>> df.compare(df2, align_axis=0)\n                col1  col3\n        0 self     a   NaN\n          other    c   NaN\n        2 self   NaN   3.0\n          other  NaN   4.0\n\n        Keep the equal values\n\n        >>> df.compare(df2, keep_equal=True)\n          col1       col3\n          self other self other\n        0    a     c  1.0   1.0\n        2    b     b  3.0   4.0\n\n        Keep all original rows and columns\n\n        >>> df.compare(df2, keep_shape=True)\n          col1       col2       col3\n          self other self other self other\n        0    a     c  NaN   NaN  NaN   NaN\n        1  NaN   NaN  NaN   NaN  NaN   NaN\n        2  NaN   NaN  NaN   NaN  3.0   4.0\n        3  NaN   NaN  NaN   NaN  NaN   NaN\n        4  NaN   NaN  NaN   NaN  NaN   NaN\n\n        Keep all original rows and columns and also all original values\n\n        >>> df.compare(df2, keep_shape=True, keep_equal=True)\n          col1       col2       col3\n          self other self other self other\n        0    a     c  1.0   1.0  1.0   1.0\n        1    a     a  2.0   2.0  2.0   2.0\n        2    b     b  3.0   3.0  3.0   4.0\n        3    b     b  NaN   NaN  4.0   4.0\n        4    a     a  5.0   5.0  5.0   5.0\n        '), klass=_shared_doc_kwargs['klass'])
    def compare(self, other: Any, align_axis: int=1, keep_shape: bool=False, keep_equal: bool=False, result_names: tuple[typing.Text]=('self', 'other')):
        return super().compare(other=other, align_axis=align_axis, keep_shape=keep_shape, keep_equal=keep_equal, result_names=result_names)

    def combine(self, other: Any, func: Any, fill_value: None=None, overwrite: bool=True):
        """
        Perform column-wise combine with another DataFrame.

        Combines a DataFrame with `other` DataFrame using `func`
        to element-wise combine columns. The row and column indexes of the
        resulting DataFrame will be the union of the two.

        Parameters
        ----------
        other : DataFrame
            The DataFrame to merge column-wise.
        func : function
            Function that takes two series as inputs and return a Series or a
            scalar. Used to merge the two dataframes column by columns.
        fill_value : scalar value, default None
            The value to fill NaNs with prior to passing any column to the
            merge func.
        overwrite : bool, default True
            If True, columns in `self` that do not exist in `other` will be
            overwritten with NaNs.

        Returns
        -------
        DataFrame
            Combination of the provided DataFrames.

        See Also
        --------
        DataFrame.combine_first : Combine two DataFrame objects and default to
            non-null values in frame calling the method.

        Examples
        --------
        Combine using a simple function that chooses the smaller column.

        >>> df1 = pd.DataFrame({"A": [0, 0], "B": [4, 4]})
        >>> df2 = pd.DataFrame({"A": [1, 1], "B": [3, 3]})
        >>> take_smaller = lambda s1, s2: s1 if s1.sum() < s2.sum() else s2
        >>> df1.combine(df2, take_smaller)
           A  B
        0  0  3
        1  0  3

        Example using a true element-wise combine function.

        >>> df1 = pd.DataFrame({"A": [5, 0], "B": [2, 4]})
        >>> df2 = pd.DataFrame({"A": [1, 1], "B": [3, 3]})
        >>> df1.combine(df2, np.minimum)
           A  B
        0  1  2
        1  0  3

        Using `fill_value` fills Nones prior to passing the column to the
        merge function.

        >>> df1 = pd.DataFrame({"A": [0, 0], "B": [None, 4]})
        >>> df2 = pd.DataFrame({"A": [1, 1], "B": [3, 3]})
        >>> df1.combine(df2, take_smaller, fill_value=-5)
           A    B
        0  0 -5.0
        1  0  4.0

        However, if the same element in both dataframes is None, that None
        is preserved

        >>> df1 = pd.DataFrame({"A": [0, 0], "B": [None, 4]})
        >>> df2 = pd.DataFrame({"A": [1, 1], "B": [None, 3]})
        >>> df1.combine(df2, take_smaller, fill_value=-5)
            A    B
        0  0 -5.0
        1  0  3.0

        Example that demonstrates the use of `overwrite` and behavior when
        the axis differ between the dataframes.

        >>> df1 = pd.DataFrame({"A": [0, 0], "B": [4, 4]})
        >>> df2 = pd.DataFrame(
        ...     {
        ...         "B": [3, 3],
        ...         "C": [-10, 1],
        ...     },
        ...     index=[1, 2],
        ... )
        >>> df1.combine(df2, take_smaller)
             A    B     C
        0  NaN  NaN   NaN
        1  NaN  3.0 -10.0
        2  NaN  3.0   1.0

        >>> df1.combine(df2, take_smaller, overwrite=False)
             A    B     C
        0  0.0  NaN   NaN
        1  0.0  3.0 -10.0
        2  NaN  3.0   1.0

        Demonstrating the preference of the passed in dataframe.

        >>> df2 = pd.DataFrame(
        ...     {
        ...         "B": [3, 3],
        ...         "C": [1, 1],
        ...     },
        ...     index=[1, 2],
        ... )
        >>> df2.combine(df1, take_smaller)
           A    B   C
        0  0.0  NaN NaN
        1  0.0  3.0 NaN
        2  NaN  3.0 NaN

        >>> df2.combine(df1, take_smaller, overwrite=False)
             A    B   C
        0  0.0  NaN NaN
        1  0.0  3.0 1.0
        2  NaN  3.0 1.0
        """
        other_idxlen = len(other.index)
        other_columns = other.columns
        this, other = self.align(other)
        new_index = this.index
        if other.empty and len(new_index) == len(self.index):
            return self.copy()
        if self.empty and len(other) == other_idxlen:
            return other.copy()
        new_columns = self.columns.union(other_columns, sort=False)
        do_fill = fill_value is not None
        result = {}
        for col in new_columns:
            series = this[col]
            other_series = other[col]
            this_dtype = series.dtype
            other_dtype = other_series.dtype
            this_mask = isna(series)
            other_mask = isna(other_series)
            if not overwrite and other_mask.all():
                result[col] = this[col].copy()
                continue
            if do_fill:
                series = series.copy()
                other_series = other_series.copy()
                series[this_mask] = fill_value
                other_series[other_mask] = fill_value
            if col not in self.columns:
                new_dtype = other_dtype
                try:
                    series = series.astype(new_dtype)
                except ValueError:
                    pass
            else:
                new_dtype = find_common_type([this_dtype, other_dtype])
                series = series.astype(new_dtype)
                other_series = other_series.astype(new_dtype)
            arr = func(series, other_series)
            if isinstance(new_dtype, np.dtype):
                arr = maybe_downcast_to_dtype(arr, new_dtype)
            result[col] = arr
        frame_result = self._constructor(result, index=new_index, columns=new_columns)
        return frame_result.__finalize__(self, method='combine')

    def combine_first(self, other: Any):
        """
        Update null elements with value in the same location in `other`.

        Combine two DataFrame objects by filling null values in one DataFrame
        with non-null values from other DataFrame. The row and column indexes
        of the resulting DataFrame will be the union of the two. The resulting
        dataframe contains the 'first' dataframe values and overrides the
        second one values where both first.loc[index, col] and
        second.loc[index, col] are not missing values, upon calling
        first.combine_first(second).

        Parameters
        ----------
        other : DataFrame
            Provided DataFrame to use to fill null values.

        Returns
        -------
        DataFrame
            The result of combining the provided DataFrame with the other object.

        See Also
        --------
        DataFrame.combine : Perform series-wise operation on two DataFrames
            using a given function.

        Examples
        --------
        >>> df1 = pd.DataFrame({"A": [None, 0], "B": [None, 4]})
        >>> df2 = pd.DataFrame({"A": [1, 1], "B": [3, 3]})
        >>> df1.combine_first(df2)
             A    B
        0  1.0  3.0
        1  0.0  4.0

        Null values still persist if the location of that null value
        does not exist in `other`

        >>> df1 = pd.DataFrame({"A": [None, 0], "B": [4, None]})
        >>> df2 = pd.DataFrame({"B": [3, 3], "C": [1, 1]}, index=[1, 2])
        >>> df1.combine_first(df2)
             A    B    C
        0  NaN  4.0  NaN
        1  0.0  3.0  1.0
        2  NaN  3.0  1.0
        """
        from pandas.core.computation import expressions

        def combiner(x: Any, y: Any):
            mask = x.isna()._values
            x_values = x._values
            y_values = y._values
            if y.name not in self.columns:
                return y_values
            return expressions.where(mask, y_values, x_values)
        if len(other) == 0:
            combined = self.reindex(self.columns.append(other.columns.difference(self.columns)), axis=1)
            combined = combined.astype(other.dtypes)
        else:
            combined = self.combine(other, combiner, overwrite=False)
        dtypes = {col: find_common_type([self.dtypes[col], other.dtypes[col]]) for col in self.columns.intersection(other.columns) if combined.dtypes[col] != self.dtypes[col]}
        if dtypes:
            combined = combined.astype(dtypes)
        return combined.__finalize__(self, method='combine_first')

    def update(self, other: Any, join: typing.Text='left', overwrite: bool=True, filter_func: None=None, errors: typing.Text='ignore') -> None:
        """
        Modify in place using non-NA values from another DataFrame.

        Aligns on indices. There is no return value.

        Parameters
        ----------
        other : DataFrame, or object coercible into a DataFrame
            Should have at least one matching index/column label
            with the original DataFrame. If a Series is passed,
            its name attribute must be set, and that will be
            used as the column name to align with the original DataFrame.
        join : {'left'}, default 'left'
            Only left join is implemented, keeping the index and columns of the
            original object.
        overwrite : bool, default True
            How to handle non-NA values for overlapping keys:

            * True: overwrite original DataFrame's values
              with values from `other`.
            * False: only update values that are NA in
              the original DataFrame.

        filter_func : callable(1d-array) -> bool 1d-array, optional
            Can choose to replace values other than NA. Return True for values
            that should be updated.
        errors : {'raise', 'ignore'}, default 'ignore'
            If 'raise', will raise a ValueError if the DataFrame and `other`
            both contain non-NA data in the same place.

        Returns
        -------
        None
            This method directly changes calling object.

        Raises
        ------
        ValueError
            * When `errors='raise'` and there's overlapping non-NA data.
            * When `errors` is not either `'ignore'` or `'raise'`
        NotImplementedError
            * If `join != 'left'`

        See Also
        --------
        dict.update : Similar method for dictionaries.
        DataFrame.merge : For column(s)-on-column(s) operations.

        Notes
        -----
        1. Duplicate indices on `other` are not supported and raises `ValueError`.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [400, 500, 600]})
        >>> new_df = pd.DataFrame({"B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df.update(new_df)
        >>> df
           A  B
        0  1  4
        1  2  5
        2  3  6

        The DataFrame's length does not increase as a result of the update,
        only values at matching index/column labels are updated.

        >>> df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["x", "y", "z"]})
        >>> new_df = pd.DataFrame({"B": ["d", "e", "f", "g", "h", "i"]})
        >>> df.update(new_df)
        >>> df
           A  B
        0  a  d
        1  b  e
        2  c  f

        >>> df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["x", "y", "z"]})
        >>> new_df = pd.DataFrame({"B": ["d", "f"]}, index=[0, 2])
        >>> df.update(new_df)
        >>> df
           A  B
        0  a  d
        1  b  y
        2  c  f

        For Series, its name attribute must be set.

        >>> df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["x", "y", "z"]})
        >>> new_column = pd.Series(["d", "e", "f"], name="B")
        >>> df.update(new_column)
        >>> df
           A  B
        0  a  d
        1  b  e
        2  c  f

        If `other` contains NaNs the corresponding values are not updated
        in the original dataframe.

        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [400.0, 500.0, 600.0]})
        >>> new_df = pd.DataFrame({"B": [4, np.nan, 6]})
        >>> df.update(new_df)
        >>> df
           A      B
        0  1    4.0
        1  2  500.0
        2  3    6.0
        """
        if not PYPY:
            if sys.getrefcount(self) <= REF_COUNT:
                warnings.warn(_chained_assignment_method_msg, ChainedAssignmentError, stacklevel=2)
        if join != 'left':
            raise NotImplementedError('Only left join is supported')
        if errors not in ['ignore', 'raise']:
            raise ValueError("The parameter errors must be either 'ignore' or 'raise'")
        if not isinstance(other, DataFrame):
            other = DataFrame(other)
        if other.index.has_duplicates:
            raise ValueError('Update not allowed with duplicate indexes on other.')
        index_intersection = other.index.intersection(self.index)
        if index_intersection.empty:
            raise ValueError('Update not allowed when the index on `other` has no intersection with this dataframe.')
        other = other.reindex(index_intersection)
        this_data = self.loc[index_intersection]
        for col in self.columns.intersection(other.columns):
            this = this_data[col]
            that = other[col]
            if filter_func is not None:
                mask = ~filter_func(this) | isna(that)
            else:
                if errors == 'raise':
                    mask_this = notna(that)
                    mask_that = notna(this)
                    if any(mask_this & mask_that):
                        raise ValueError('Data overlaps.')
                if overwrite:
                    mask = isna(that)
                else:
                    mask = notna(this)
            if mask.all():
                continue
            self.loc[index_intersection, col] = this.where(mask, that)

    @Appender(dedent('\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'Animal\': [\'Falcon\', \'Falcon\',\n        ...                               \'Parrot\', \'Parrot\'],\n        ...                    \'Max Speed\': [380., 370., 24., 26.]})\n        >>> df\n           Animal  Max Speed\n        0  Falcon      380.0\n        1  Falcon      370.0\n        2  Parrot       24.0\n        3  Parrot       26.0\n        >>> df.groupby([\'Animal\']).mean()\n                Max Speed\n        Animal\n        Falcon      375.0\n        Parrot       25.0\n\n        **Hierarchical Indexes**\n\n        We can groupby different levels of a hierarchical index\n        using the `level` parameter:\n\n        >>> arrays = [[\'Falcon\', \'Falcon\', \'Parrot\', \'Parrot\'],\n        ...           [\'Captive\', \'Wild\', \'Captive\', \'Wild\']]\n        >>> index = pd.MultiIndex.from_arrays(arrays, names=(\'Animal\', \'Type\'))\n        >>> df = pd.DataFrame({\'Max Speed\': [390., 350., 30., 20.]},\n        ...                   index=index)\n        >>> df\n                        Max Speed\n        Animal Type\n        Falcon Captive      390.0\n               Wild         350.0\n        Parrot Captive       30.0\n               Wild          20.0\n        >>> df.groupby(level=0).mean()\n                Max Speed\n        Animal\n        Falcon      370.0\n        Parrot       25.0\n        >>> df.groupby(level="Type").mean()\n                 Max Speed\n        Type\n        Captive      210.0\n        Wild         185.0\n\n        We can also choose to include NA in group keys or not by setting\n        `dropna` parameter, the default setting is `True`.\n\n        >>> arr = [[1, 2, 3], [1, None, 4], [2, 1, 3], [1, 2, 2]]\n        >>> df = pd.DataFrame(arr, columns=["a", "b", "c"])\n\n        >>> df.groupby(by=["b"]).sum()\n            a   c\n        b\n        1.0 2   3\n        2.0 2   5\n\n        >>> df.groupby(by=["b"], dropna=False).sum()\n            a   c\n        b\n        1.0 2   3\n        2.0 2   5\n        NaN 1   4\n\n        >>> arr = [["a", 12, 12], [None, 12.3, 33.], ["b", 12.3, 123], ["a", 1, 1]]\n        >>> df = pd.DataFrame(arr, columns=["a", "b", "c"])\n\n        >>> df.groupby(by="a").sum()\n            b     c\n        a\n        a   13.0   13.0\n        b   12.3  123.0\n\n        >>> df.groupby(by="a", dropna=False).sum()\n            b     c\n        a\n        a   13.0   13.0\n        b   12.3  123.0\n        NaN 12.3   33.0\n\n        When using ``.apply()``, use ``group_keys`` to include or exclude the\n        group keys. The ``group_keys`` argument defaults to ``True`` (include).\n\n        >>> df = pd.DataFrame({\'Animal\': [\'Falcon\', \'Falcon\',\n        ...                               \'Parrot\', \'Parrot\'],\n        ...                    \'Max Speed\': [380., 370., 24., 26.]})\n        >>> df.groupby("Animal", group_keys=True)[[\'Max Speed\']].apply(lambda x: x)\n                  Max Speed\n        Animal\n        Falcon 0      380.0\n               1      370.0\n        Parrot 2       24.0\n               3       26.0\n\n        >>> df.groupby("Animal", group_keys=False)[[\'Max Speed\']].apply(lambda x: x)\n           Max Speed\n        0      380.0\n        1      370.0\n        2       24.0\n        3       26.0\n        '))
    @Appender(_shared_docs['groupby'] % _shared_doc_kwargs)
    def groupby(self, by: None=None, level: None=None, as_index: bool=True, sort: bool=True, group_keys: bool=True, observed: bool=True, dropna: bool=True) -> DataFrameGroupBy:
        from pandas.core.groupby.generic import DataFrameGroupBy
        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        return DataFrameGroupBy(obj=self, keys=by, level=level, as_index=as_index, sort=sort, group_keys=group_keys, observed=observed, dropna=dropna)
    _shared_docs['pivot'] = '\n        Return reshaped DataFrame organized by given index / column values.\n\n        Reshape data (produce a "pivot" table) based on column values. Uses\n        unique values from specified `index` / `columns` to form axes of the\n        resulting DataFrame. This function does not support data\n        aggregation, multiple values will result in a MultiIndex in the\n        columns. See the :ref:`User Guide <reshaping>` for more on reshaping.\n\n        Parameters\n        ----------%s\n        columns : str or object or a list of str\n            Column to use to make new frame\'s columns.\n        index : str or object or a list of str, optional\n            Column to use to make new frame\'s index. If not given, uses existing index.\n        values : str, object or a list of the previous, optional\n            Column(s) to use for populating new frame\'s values. If not\n            specified, all remaining columns will be used and the result will\n            have hierarchically indexed columns.\n\n        Returns\n        -------\n        DataFrame\n            Returns reshaped DataFrame.\n\n        Raises\n        ------\n        ValueError:\n            When there are any `index`, `columns` combinations with multiple\n            values. `DataFrame.pivot_table` when you need to aggregate.\n\n        See Also\n        --------\n        DataFrame.pivot_table : Generalization of pivot that can handle\n            duplicate values for one index/column pair.\n        DataFrame.unstack : Pivot based on the index values instead of a\n            column.\n        wide_to_long : Wide panel to long format. Less flexible but more\n            user-friendly than melt.\n\n        Notes\n        -----\n        For finer-tuned control, see hierarchical indexing documentation along\n        with the related stack/unstack methods.\n\n        Reference :ref:`the user guide <reshaping.pivot>` for more examples.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({\'foo\': [\'one\', \'one\', \'one\', \'two\', \'two\',\n        ...                            \'two\'],\n        ...                    \'bar\': [\'A\', \'B\', \'C\', \'A\', \'B\', \'C\'],\n        ...                    \'baz\': [1, 2, 3, 4, 5, 6],\n        ...                    \'zoo\': [\'x\', \'y\', \'z\', \'q\', \'w\', \'t\']})\n        >>> df\n            foo   bar  baz  zoo\n        0   one   A    1    x\n        1   one   B    2    y\n        2   one   C    3    z\n        3   two   A    4    q\n        4   two   B    5    w\n        5   two   C    6    t\n\n        >>> df.pivot(index=\'foo\', columns=\'bar\', values=\'baz\')\n        bar  A   B   C\n        foo\n        one  1   2   3\n        two  4   5   6\n\n        >>> df.pivot(index=\'foo\', columns=\'bar\')[\'baz\']\n        bar  A   B   C\n        foo\n        one  1   2   3\n        two  4   5   6\n\n        >>> df.pivot(index=\'foo\', columns=\'bar\', values=[\'baz\', \'zoo\'])\n              baz       zoo\n        bar   A  B  C   A  B  C\n        foo\n        one   1  2  3   x  y  z\n        two   4  5  6   q  w  t\n\n        You could also assign a list of column names or a list of index names.\n\n        >>> df = pd.DataFrame({\n        ...                   "lev1": [1, 1, 1, 2, 2, 2],\n        ...                   "lev2": [1, 1, 2, 1, 1, 2],\n        ...                   "lev3": [1, 2, 1, 2, 1, 2],\n        ...                   "lev4": [1, 2, 3, 4, 5, 6],\n        ...                   "values": [0, 1, 2, 3, 4, 5]})\n        >>> df\n            lev1 lev2 lev3 lev4 values\n        0   1    1    1    1    0\n        1   1    1    2    2    1\n        2   1    2    1    3    2\n        3   2    1    2    4    3\n        4   2    1    1    5    4\n        5   2    2    2    6    5\n\n        >>> df.pivot(index="lev1", columns=["lev2", "lev3"], values="values")\n        lev2    1         2\n        lev3    1    2    1    2\n        lev1\n        1     0.0  1.0  2.0  NaN\n        2     4.0  3.0  NaN  5.0\n\n        >>> df.pivot(index=["lev1", "lev2"], columns=["lev3"], values="values")\n              lev3    1    2\n        lev1  lev2\n           1     1  0.0  1.0\n                 2  2.0  NaN\n           2     1  4.0  3.0\n                 2  NaN  5.0\n\n        A ValueError is raised if there are any duplicates.\n\n        >>> df = pd.DataFrame({"foo": [\'one\', \'one\', \'two\', \'two\'],\n        ...                    "bar": [\'A\', \'A\', \'B\', \'C\'],\n        ...                    "baz": [1, 2, 3, 4]})\n        >>> df\n           foo bar  baz\n        0  one   A    1\n        1  one   A    2\n        2  two   B    3\n        3  two   C    4\n\n        Notice that the first two rows are the same for our `index`\n        and `columns` arguments.\n\n        >>> df.pivot(index=\'foo\', columns=\'bar\', values=\'baz\')\n        Traceback (most recent call last):\n           ...\n        ValueError: Index contains duplicate entries, cannot reshape\n        '

    @Substitution('')
    @Appender(_shared_docs['pivot'])
    def pivot(self, *, columns: Any, index: Any=lib.no_default, values: Any=lib.no_default):
        from pandas.core.reshape.pivot import pivot
        return pivot(self, index=index, columns=columns, values=values)
    _shared_docs['pivot_table'] = '\n        Create a spreadsheet-style pivot table as a DataFrame.\n\n        The levels in the pivot table will be stored in MultiIndex objects\n        (hierarchical indexes) on the index and columns of the result DataFrame.\n\n        Parameters\n        ----------%s\n        values : list-like or scalar, optional\n            Column or columns to aggregate.\n        index : column, Grouper, array, or list of the previous\n            Keys to group by on the pivot table index. If a list is passed,\n            it can contain any of the other types (except list). If an array is\n            passed, it must be the same length as the data and will be used in\n            the same manner as column values.\n        columns : column, Grouper, array, or list of the previous\n            Keys to group by on the pivot table column. If a list is passed,\n            it can contain any of the other types (except list). If an array is\n            passed, it must be the same length as the data and will be used in\n            the same manner as column values.\n        aggfunc : function, list of functions, dict, default "mean"\n            If a list of functions is passed, the resulting pivot table will have\n            hierarchical columns whose top level are the function names\n            (inferred from the function objects themselves).\n            If a dict is passed, the key is column to aggregate and the value is\n            function or list of functions. If ``margin=True``, aggfunc will be\n            used to calculate the partial aggregates.\n        fill_value : scalar, default None\n            Value to replace missing values with (in the resulting pivot table,\n            after aggregation).\n        margins : bool, default False\n            If ``margins=True``, special ``All`` columns and rows\n            will be added with partial group aggregates across the categories\n            on the rows and columns.\n        dropna : bool, default True\n            Do not include columns whose entries are all NaN. If True,\n            rows with a NaN value in any column will be omitted before\n            computing margins.\n        margins_name : str, default \'All\'\n            Name of the row / column that will contain the totals\n            when margins is True.\n        observed : bool, default False\n            This only applies if any of the groupers are Categoricals.\n            If True: only show observed values for categorical groupers.\n            If False: show all values for categorical groupers.\n\n            .. versionchanged:: 3.0.0\n\n                The default value is now ``True``.\n\n        sort : bool, default True\n            Specifies if the result should be sorted.\n\n            .. versionadded:: 1.3.0\n\n        **kwargs : dict\n            Optional keyword arguments to pass to ``aggfunc``.\n\n            .. versionadded:: 3.0.0\n\n        Returns\n        -------\n        DataFrame\n            An Excel style pivot table.\n\n        See Also\n        --------\n        DataFrame.pivot : Pivot without aggregation that can handle\n            non-numeric data.\n        DataFrame.melt: Unpivot a DataFrame from wide to long format,\n            optionally leaving identifiers set.\n        wide_to_long : Wide panel to long format. Less flexible but more\n            user-friendly than melt.\n\n        Notes\n        -----\n        Reference :ref:`the user guide <reshaping.pivot>` for more examples.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",\n        ...                          "bar", "bar", "bar", "bar"],\n        ...                    "B": ["one", "one", "one", "two", "two",\n        ...                          "one", "one", "two", "two"],\n        ...                    "C": ["small", "large", "large", "small",\n        ...                          "small", "large", "small", "small",\n        ...                          "large"],\n        ...                    "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],\n        ...                    "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})\n        >>> df\n             A    B      C  D  E\n        0  foo  one  small  1  2\n        1  foo  one  large  2  4\n        2  foo  one  large  2  5\n        3  foo  two  small  3  5\n        4  foo  two  small  3  6\n        5  bar  one  large  4  6\n        6  bar  one  small  5  8\n        7  bar  two  small  6  9\n        8  bar  two  large  7  9\n\n        This first example aggregates values by taking the sum.\n\n        >>> table = pd.pivot_table(df, values=\'D\', index=[\'A\', \'B\'],\n        ...                        columns=[\'C\'], aggfunc="sum")\n        >>> table\n        C        large  small\n        A   B\n        bar one    4.0    5.0\n            two    7.0    6.0\n        foo one    4.0    1.0\n            two    NaN    6.0\n\n        We can also fill missing values using the `fill_value` parameter.\n\n        >>> table = pd.pivot_table(df, values=\'D\', index=[\'A\', \'B\'],\n        ...                        columns=[\'C\'], aggfunc="sum", fill_value=0)\n        >>> table\n        C        large  small\n        A   B\n        bar one      4      5\n            two      7      6\n        foo one      4      1\n            two      0      6\n\n        The next example aggregates by taking the mean across multiple columns.\n\n        >>> table = pd.pivot_table(df, values=[\'D\', \'E\'], index=[\'A\', \'C\'],\n        ...                        aggfunc={\'D\': "mean", \'E\': "mean"})\n        >>> table\n                        D         E\n        A   C\n        bar large  5.500000  7.500000\n            small  5.500000  8.500000\n        foo large  2.000000  4.500000\n            small  2.333333  4.333333\n\n        We can also calculate multiple types of aggregations for any given\n        value column.\n\n        >>> table = pd.pivot_table(df, values=[\'D\', \'E\'], index=[\'A\', \'C\'],\n        ...                        aggfunc={\'D\': "mean",\n        ...                                 \'E\': ["min", "max", "mean"]})\n        >>> table\n                          D   E\n                       mean max      mean  min\n        A   C\n        bar large  5.500000   9  7.500000    6\n            small  5.500000   9  8.500000    8\n        foo large  2.000000   5  4.500000    4\n            small  2.333333   6  4.333333    2\n        '

    @Substitution('')
    @Appender(_shared_docs['pivot_table'])
    def pivot_table(self, values: None=None, index: None=None, columns: None=None, aggfunc: typing.Text='mean', fill_value: None=None, margins: bool=False, dropna: bool=True, margins_name: typing.Text='All', observed: bool=True, sort: bool=True, **kwargs):
        from pandas.core.reshape.pivot import pivot_table
        return pivot_table(self, values=values, index=index, columns=columns, aggfunc=aggfunc, fill_value=fill_value, margins=margins, dropna=dropna, margins_name=margins_name, observed=observed, sort=sort, **kwargs)

    def stack(self, level: Any=-1, dropna: Any=lib.no_default, sort: Any=lib.no_default, future_stack: bool=True):
        """
        Stack the prescribed level(s) from columns to index.

        Return a reshaped DataFrame or Series having a multi-level
        index with one or more new inner-most levels compared to the current
        DataFrame. The new inner-most levels are created by pivoting the
        columns of the current dataframe:

        - if the columns have a single level, the output is a Series;
        - if the columns have multiple levels, the new index level(s) is (are)
          taken from the prescribed level(s) and the output is a DataFrame.

        Parameters
        ----------
        level : int, str, list, default -1
            Level(s) to stack from the column axis onto the index
            axis, defined as one index or label, or a list of indices
            or labels.
        dropna : bool, default True
            Whether to drop rows in the resulting Frame/Series with
            missing values. Stacking a column level onto the index
            axis can create combinations of index and column values
            that are missing from the original dataframe. See Examples
            section.
        sort : bool, default True
            Whether to sort the levels of the resulting MultiIndex.
        future_stack : bool, default True
            Whether to use the new implementation that will replace the current
            implementation in pandas 3.0. When True, dropna and sort have no impact
            on the result and must remain unspecified. See :ref:`pandas 2.1.0 Release
            notes <whatsnew_210.enhancements.new_stack>` for more details.

        Returns
        -------
        DataFrame or Series
            Stacked dataframe or series.

        See Also
        --------
        DataFrame.unstack : Unstack prescribed level(s) from index axis
             onto column axis.
        DataFrame.pivot : Reshape dataframe from long format to wide
             format.
        DataFrame.pivot_table : Create a spreadsheet-style pivot table
             as a DataFrame.

        Notes
        -----
        The function is named by analogy with a collection of books
        being reorganized from being side by side on a horizontal
        position (the columns of the dataframe) to being stacked
        vertically on top of each other (in the index of the
        dataframe).

        Reference :ref:`the user guide <reshaping.stacking>` for more examples.

        Examples
        --------
        **Single level columns**

        >>> df_single_level_cols = pd.DataFrame(
        ...     [[0, 1], [2, 3]], index=["cat", "dog"], columns=["weight", "height"]
        ... )

        Stacking a dataframe with a single level column axis returns a Series:

        >>> df_single_level_cols
             weight height
        cat       0      1
        dog       2      3
        >>> df_single_level_cols.stack()
        cat  weight    0
             height    1
        dog  weight    2
             height    3
        dtype: int64

        **Multi level columns: simple case**

        >>> multicol1 = pd.MultiIndex.from_tuples(
        ...     [("weight", "kg"), ("weight", "pounds")]
        ... )
        >>> df_multi_level_cols1 = pd.DataFrame(
        ...     [[1, 2], [2, 4]], index=["cat", "dog"], columns=multicol1
        ... )

        Stacking a dataframe with a multi-level column axis:

        >>> df_multi_level_cols1
             weight
                 kg    pounds
        cat       1        2
        dog       2        4
        >>> df_multi_level_cols1.stack()
                    weight
        cat kg           1
            pounds       2
        dog kg           2
            pounds       4

        **Missing values**

        >>> multicol2 = pd.MultiIndex.from_tuples([("weight", "kg"), ("height", "m")])
        >>> df_multi_level_cols2 = pd.DataFrame(
        ...     [[1.0, 2.0], [3.0, 4.0]], index=["cat", "dog"], columns=multicol2
        ... )

        It is common to have missing values when stacking a dataframe
        with multi-level columns, as the stacked dataframe typically
        has more values than the original dataframe. Missing values
        are filled with NaNs:

        >>> df_multi_level_cols2
            weight height
                kg      m
        cat    1.0    2.0
        dog    3.0    4.0
        >>> df_multi_level_cols2.stack()
                weight  height
        cat kg     1.0     NaN
            m      NaN     2.0
        dog kg     3.0     NaN
            m      NaN     4.0

        **Prescribing the level(s) to be stacked**

        The first parameter controls which level or levels are stacked:

        >>> df_multi_level_cols2.stack(0)
                     kg    m
        cat weight  1.0  NaN
            height  NaN  2.0
        dog weight  3.0  NaN
            height  NaN  4.0
        >>> df_multi_level_cols2.stack([0, 1])
        cat  weight  kg    1.0
             height  m     2.0
        dog  weight  kg    3.0
             height  m     4.0
        dtype: float64
        """
        if not future_stack:
            from pandas.core.reshape.reshape import stack, stack_multiple
            warnings.warn("The previous implementation of stack is deprecated and will be removed in a future version of pandas. See the What's New notes for pandas 2.1.0 for details. Do not specify the future_stack argument to adopt the new implementation and silence this warning.", FutureWarning, stacklevel=find_stack_level())
            if dropna is lib.no_default:
                dropna = True
            if sort is lib.no_default:
                sort = True
            if isinstance(level, (tuple, list)):
                result = stack_multiple(self, level, dropna=dropna, sort=sort)
            else:
                result = stack(self, level, dropna=dropna, sort=sort)
        else:
            from pandas.core.reshape.reshape import stack_v3
            if dropna is not lib.no_default:
                raise ValueError('dropna must be unspecified as the new implementation does not introduce rows of NA values. This argument will be removed in a future version of pandas.')
            if sort is not lib.no_default:
                raise ValueError('Cannot specify sort, this argument will be removed in a future version of pandas. Sort the result using .sort_index instead.')
            if isinstance(level, (tuple, list)) and (not all((lev in self.columns.names for lev in level))) and (not all((isinstance(lev, int) for lev in level))):
                raise ValueError('level should contain all level names or all level numbers, not a mixture of the two.')
            if not isinstance(level, (tuple, list)):
                level = [level]
            level = [self.columns._get_level_number(lev) for lev in level]
            result = stack_v3(self, level)
        return result.__finalize__(self, method='stack')

    def explode(self, column: Any, ignore_index: bool=False):
        """
        Transform each element of a list-like to a row, replicating index values.

        Parameters
        ----------
        column : IndexLabel
            Column(s) to explode.
            For multiple columns, specify a non-empty list with each element
            be str or tuple, and all specified columns their list-like data
            on same row of the frame must have matching length.

            .. versionadded:: 1.3.0
                Multi-column explode

        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        DataFrame
            Exploded lists to rows of the subset columns;
            index will be duplicated for these rows.

        Raises
        ------
        ValueError :
            * If columns of the frame are not unique.
            * If specified columns to explode is empty list.
            * If specified columns to explode have not matching count of
              elements rowwise in the frame.

        See Also
        --------
        DataFrame.unstack : Pivot a level of the (necessarily hierarchical)
            index labels.
        DataFrame.melt : Unpivot a DataFrame from wide format to long format.
        Series.explode : Explode a DataFrame from list-like columns to long format.

        Notes
        -----
        This routine will explode list-likes including lists, tuples, sets,
        Series, and np.ndarray. The result dtype of the subset rows will
        be object. Scalars will be returned unchanged, and empty list-likes will
        result in a np.nan for that row. In addition, the ordering of rows in the
        output will be non-deterministic when exploding sets.

        Reference :ref:`the user guide <reshaping.explode>` for more examples.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "A": [[0, 1, 2], "foo", [], [3, 4]],
        ...         "B": 1,
        ...         "C": [["a", "b", "c"], np.nan, [], ["d", "e"]],
        ...     }
        ... )
        >>> df
                   A  B          C
        0  [0, 1, 2]  1  [a, b, c]
        1        foo  1        NaN
        2         []  1         []
        3     [3, 4]  1     [d, e]

        Single-column explode.

        >>> df.explode("A")
             A  B          C
        0    0  1  [a, b, c]
        0    1  1  [a, b, c]
        0    2  1  [a, b, c]
        1  foo  1        NaN
        2  NaN  1         []
        3    3  1     [d, e]
        3    4  1     [d, e]

        Multi-column explode.

        >>> df.explode(list("AC"))
             A  B    C
        0    0  1    a
        0    1  1    b
        0    2  1    c
        1  foo  1  NaN
        2  NaN  1  NaN
        3    3  1    d
        3    4  1    e
        """
        if not self.columns.is_unique:
            duplicate_cols = self.columns[self.columns.duplicated()].tolist()
            raise ValueError(f'DataFrame columns must be unique. Duplicate columns: {duplicate_cols}')
        if is_scalar(column) or isinstance(column, tuple):
            columns = [column]
        elif isinstance(column, list) and all((is_scalar(c) or isinstance(c, tuple) for c in column)):
            if not column:
                raise ValueError('column must be nonempty')
            if len(column) > len(set(column)):
                raise ValueError('column must be unique')
            columns = column
        else:
            raise ValueError('column must be a scalar, tuple, or list thereof')
        df = self.reset_index(drop=True)
        if len(columns) == 1:
            result = df[columns[0]].explode()
        else:
            mylen = lambda x: len(x) if is_list_like(x) and len(x) > 0 else 1
            counts0 = self[columns[0]].apply(mylen)
            for c in columns[1:]:
                if not all(counts0 == self[c].apply(mylen)):
                    raise ValueError('columns must have matching element counts')
            result = DataFrame({c: df[c].explode() for c in columns})
        result = df.drop(columns, axis=1).join(result)
        if ignore_index:
            result.index = default_index(len(result))
        else:
            result.index = self.index.take(result.index)
        result = result.reindex(columns=self.columns)
        return result.__finalize__(self, method='explode')

    def unstack(self, level: int=-1, fill_value: None=None, sort: bool=True):
        """
        Pivot a level of the (necessarily hierarchical) index labels.

        Returns a DataFrame having a new level of column labels whose inner-most level
        consists of the pivoted index labels.

        If the index is not a MultiIndex, the output will be a Series
        (the analogue of stack when the columns are not a MultiIndex).

        Parameters
        ----------
        level : int, str, or list of these, default -1 (last level)
            Level(s) of index to unstack, can pass level name.
        fill_value : int, str or dict
            Replace NaN with this value if the unstack produces missing values.
        sort : bool, default True
            Sort the level(s) in the resulting MultiIndex columns.

        Returns
        -------
        Series or DataFrame
            If index is a MultiIndex: DataFrame with pivoted index labels as new
            inner-most level column labels, else Series.

        See Also
        --------
        DataFrame.pivot : Pivot a table based on column values.
        DataFrame.stack : Pivot a level of the column labels (inverse operation
            from `unstack`).

        Notes
        -----
        Reference :ref:`the user guide <reshaping.stacking>` for more examples.

        Examples
        --------
        >>> index = pd.MultiIndex.from_tuples(
        ...     [("one", "a"), ("one", "b"), ("two", "a"), ("two", "b")]
        ... )
        >>> s = pd.Series(np.arange(1.0, 5.0), index=index)
        >>> s
        one  a   1.0
             b   2.0
        two  a   3.0
             b   4.0
        dtype: float64

        >>> s.unstack(level=-1)
             a   b
        one  1.0  2.0
        two  3.0  4.0

        >>> s.unstack(level=0)
           one  two
        a  1.0   3.0
        b  2.0   4.0

        >>> df = s.unstack(level=0)
        >>> df.unstack()
        one  a  1.0
             b  2.0
        two  a  3.0
             b  4.0
        dtype: float64
        """
        from pandas.core.reshape.reshape import unstack
        result = unstack(self, level, fill_value, sort)
        return result.__finalize__(self, method='unstack')

    def melt(self, id_vars: None=None, value_vars: None=None, var_name: None=None, value_name: typing.Text='value', col_level: None=None, ignore_index: bool=True):
        """
        Unpivot DataFrame from wide to long format, optionally leaving identifiers set.

        This function is useful to massage a DataFrame into a format where one
        or more columns are identifier variables (`id_vars`), while all other
        columns, considered measured variables (`value_vars`), are "unpivoted" to
        the row axis, leaving just two non-identifier columns, 'variable' and
        'value'.

        Parameters
        ----------
        id_vars : scalar, tuple, list, or ndarray, optional
            Column(s) to use as identifier variables.
        value_vars : scalar, tuple, list, or ndarray, optional
            Column(s) to unpivot. If not specified, uses all columns that
            are not set as `id_vars`.
        var_name : scalar, default None
            Name to use for the 'variable' column. If None it uses
            ``frame.columns.name`` or 'variable'.
        value_name : scalar, default 'value'
            Name to use for the 'value' column, can't be an existing column label.
        col_level : scalar, optional
            If columns are a MultiIndex then use this level to melt.
        ignore_index : bool, default True
            If True, original index is ignored. If False, original index is retained.
            Index labels will be repeated as necessary.

        Returns
        -------
        DataFrame
            Unpivoted DataFrame.

        See Also
        --------
        melt : Identical method.
        pivot_table : Create a spreadsheet-style pivot table as a DataFrame.
        DataFrame.pivot : Return reshaped DataFrame organized
            by given index / column values.
        DataFrame.explode : Explode a DataFrame from list-like
                columns to long format.

        Notes
        -----
        Reference :ref:`the user guide <reshaping.melt>` for more examples.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "A": {0: "a", 1: "b", 2: "c"},
        ...         "B": {0: 1, 1: 3, 2: 5},
        ...         "C": {0: 2, 1: 4, 2: 6},
        ...     }
        ... )
        >>> df
        A  B  C
        0  a  1  2
        1  b  3  4
        2  c  5  6

        >>> df.melt(id_vars=["A"], value_vars=["B"])
        A variable  value
        0  a        B      1
        1  b        B      3
        2  c        B      5

        >>> df.melt(id_vars=["A"], value_vars=["B", "C"])
        A variable  value
        0  a        B      1
        1  b        B      3
        2  c        B      5
        3  a        C      2
        4  b        C      4
        5  c        C      6

        The names of 'variable' and 'value' columns can be customized:

        >>> df.melt(
        ...     id_vars=["A"],
        ...     value_vars=["B"],
        ...     var_name="myVarname",
        ...     value_name="myValname",
        ... )
        A myVarname  myValname
        0  a         B          1
        1  b         B          3
        2  c         B          5

        Original index values can be kept around:

        >>> df.melt(id_vars=["A"], value_vars=["B", "C"], ignore_index=False)
        A variable  value
        0  a        B      1
        1  b        B      3
        2  c        B      5
        0  a        C      2
        1  b        C      4
        2  c        C      6

        If you have multi-index columns:

        >>> df.columns = [list("ABC"), list("DEF")]
        >>> df
        A  B  C
        D  E  F
        0  a  1  2
        1  b  3  4
        2  c  5  6

        >>> df.melt(col_level=0, id_vars=["A"], value_vars=["B"])
        A variable  value
        0  a        B      1
        1  b        B      3
        2  c        B      5

        >>> df.melt(id_vars=[("A", "D")], value_vars=[("B", "E")])
        (A, D) variable_0 variable_1  value
        0      a          B          E      1
        1      b          B          E      3
        2      c          B          E      5
        """
        return melt(self, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name, col_level=col_level, ignore_index=ignore_index).__finalize__(self, method='melt')

    @doc(Series.diff, klass='DataFrame', extra_params="axis : {0 or 'index', 1 or 'columns'}, default 0\n    Take difference over rows (0) or columns (1).\n", other_klass='Series', examples=dedent("\n        Difference with previous row\n\n        >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],\n        ...                    'b': [1, 1, 2, 3, 5, 8],\n        ...                    'c': [1, 4, 9, 16, 25, 36]})\n        >>> df\n           a  b   c\n        0  1  1   1\n        1  2  1   4\n        2  3  2   9\n        3  4  3  16\n        4  5  5  25\n        5  6  8  36\n\n        >>> df.diff()\n             a    b     c\n        0  NaN  NaN   NaN\n        1  1.0  0.0   3.0\n        2  1.0  1.0   5.0\n        3  1.0  1.0   7.0\n        4  1.0  2.0   9.0\n        5  1.0  3.0  11.0\n\n        Difference with previous column\n\n        >>> df.diff(axis=1)\n            a  b   c\n        0 NaN  0   0\n        1 NaN -1   3\n        2 NaN -1   7\n        3 NaN -1  13\n        4 NaN  0  20\n        5 NaN  2  28\n\n        Difference with 3rd previous row\n\n        >>> df.diff(periods=3)\n             a    b     c\n        0  NaN  NaN   NaN\n        1  NaN  NaN   NaN\n        2  NaN  NaN   NaN\n        3  3.0  2.0  15.0\n        4  3.0  4.0  21.0\n        5  3.0  6.0  27.0\n\n        Difference with following row\n\n        >>> df.diff(periods=-1)\n             a    b     c\n        0 -1.0  0.0  -3.0\n        1 -1.0 -1.0  -5.0\n        2 -1.0 -1.0  -7.0\n        3 -1.0 -2.0  -9.0\n        4 -1.0 -3.0 -11.0\n        5  NaN  NaN   NaN\n\n        Overflow in input dtype\n\n        >>> df = pd.DataFrame({'a': [1, 0]}, dtype=np.uint8)\n        >>> df.diff()\n               a\n        0    NaN\n        1  255.0"))
    def diff(self, periods: Any=1, axis: int=0):
        if not lib.is_integer(periods):
            if not (is_float(periods) and periods.is_integer()):
                raise ValueError('periods must be an integer')
            periods = int(periods)
        axis = self._get_axis_number(axis)
        if axis == 1:
            if periods != 0:
                return self - self.shift(periods, axis=axis)
            axis = 0
        new_data = self._mgr.diff(n=periods)
        res_df = self._constructor_from_mgr(new_data, axes=new_data.axes)
        return res_df.__finalize__(self, 'diff')

    def _gotitem(self, key: Any, ndim: Any, subset: None=None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        if subset is None:
            subset = self
        elif subset.ndim == 1:
            return subset
        return subset[key]
    _agg_see_also_doc = dedent('\n    See Also\n    --------\n    DataFrame.apply : Perform any type of operations.\n    DataFrame.transform : Perform transformation type operations.\n    DataFrame.groupby : Perform operations over groups.\n    DataFrame.resample : Perform operations over resampled bins.\n    DataFrame.rolling : Perform operations over rolling window.\n    DataFrame.expanding : Perform operations over expanding window.\n    core.window.ewm.ExponentialMovingWindow : Perform operation over exponential\n        weighted window.\n    ')
    _agg_examples_doc = dedent('\n    Examples\n    --------\n    >>> df = pd.DataFrame([[1, 2, 3],\n    ...                    [4, 5, 6],\n    ...                    [7, 8, 9],\n    ...                    [np.nan, np.nan, np.nan]],\n    ...                   columns=[\'A\', \'B\', \'C\'])\n\n    Aggregate these functions over the rows.\n\n    >>> df.agg([\'sum\', \'min\'])\n            A     B     C\n    sum  12.0  15.0  18.0\n    min   1.0   2.0   3.0\n\n    Different aggregations per column.\n\n    >>> df.agg({\'A\' : [\'sum\', \'min\'], \'B\' : [\'min\', \'max\']})\n            A    B\n    sum  12.0  NaN\n    min   1.0  2.0\n    max   NaN  8.0\n\n    Aggregate different functions over the columns and rename the index of the resulting\n    DataFrame.\n\n    >>> df.agg(x=(\'A\', \'max\'), y=(\'B\', \'min\'), z=(\'C\', \'mean\'))\n         A    B    C\n    x  7.0  NaN  NaN\n    y  NaN  2.0  NaN\n    z  NaN  NaN  6.0\n\n    Aggregate over the columns.\n\n    >>> df.agg("mean", axis="columns")\n    0    2.0\n    1    5.0\n    2    8.0\n    3    NaN\n    dtype: float64\n    ')

    @doc(_shared_docs['aggregate'], klass=_shared_doc_kwargs['klass'], axis=_shared_doc_kwargs['axis'], see_also=_agg_see_also_doc, examples=_agg_examples_doc)
    def aggregate(self, func: None=None, axis: int=0, *args, **kwargs):
        from pandas.core.apply import frame_apply
        axis = self._get_axis_number(axis)
        op = frame_apply(self, func=func, axis=axis, args=args, kwargs=kwargs)
        result = op.agg()
        result = reconstruct_and_relabel_result(result, func, **kwargs)
        return result
    agg = aggregate

    @doc(_shared_docs['transform'], klass=_shared_doc_kwargs['klass'], axis=_shared_doc_kwargs['axis'])
    def transform(self, func: Any, axis: int=0, *args, **kwargs):
        from pandas.core.apply import frame_apply
        op = frame_apply(self, func=func, axis=axis, args=args, kwargs=kwargs)
        result = op.transform()
        assert isinstance(result, DataFrame)
        return result

    def apply(self, func: Any, axis: int=0, raw: bool=False, result_type: None=None, args: tuple=(), by_row='compat', engine='python', engine_kwargs=None, **kwargs):
        """
        Apply a function along an axis of the DataFrame.

        Objects passed to the function are Series objects whose index is
        either the DataFrame's index (``axis=0``) or the DataFrame's columns
        (``axis=1``). By default (``result_type=None``), the final return type
        is inferred from the return type of the applied function. Otherwise,
        it depends on the `result_type` argument.

        Parameters
        ----------
        func : function
            Function to apply to each column or row.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis along which the function is applied:

            * 0 or 'index': apply function to each column.
            * 1 or 'columns': apply function to each row.

        raw : bool, default False
            Determines if row or column is passed as a Series or ndarray object:

            * ``False`` : passes each row or column as a Series to the
              function.
            * ``True`` : the passed function will receive ndarray objects
              instead.
              If you are just applying a NumPy reduction function this will
              achieve much better performance.

        result_type : {'expand', 'reduce', 'broadcast', None}, default None
            These only act when ``axis=1`` (columns):

            * 'expand' : list-like results will be turned into columns.
            * 'reduce' : returns a Series if possible rather than expanding
              list-like results. This is the opposite of 'expand'.
            * 'broadcast' : results will be broadcast to the original shape
              of the DataFrame, the original index and columns will be
              retained.

            The default behaviour (None) depends on the return value of the
            applied function: list-like results will be returned as a Series
            of those. However if the apply function returns a Series these
            are expanded to columns.
        args : tuple
            Positional arguments to pass to `func` in addition to the
            array/series.
        by_row : False or "compat", default "compat"
            Only has an effect when ``func`` is a listlike or dictlike of funcs
            and the func isn't a string.
            If "compat", will if possible first translate the func into pandas
            methods (e.g. ``Series().apply(np.sum)`` will be translated to
            ``Series().sum()``). If that doesn't work, will try call to apply again with
            ``by_row=True`` and if that fails, will call apply again with
            ``by_row=False`` (backward compatible).
            If False, the funcs will be passed the whole Series at once.

            .. versionadded:: 2.1.0

        engine : {'python', 'numba'}, default 'python'
            Choose between the python (default) engine or the numba engine in apply.

            The numba engine will attempt to JIT compile the passed function,
            which may result in speedups for large DataFrames.
            It also supports the following engine_kwargs :

            - nopython (compile the function in nopython mode)
            - nogil (release the GIL inside the JIT compiled function)
            - parallel (try to apply the function in parallel over the DataFrame)

              Note: Due to limitations within numba/how pandas interfaces with numba,
              you should only use this if raw=True

            Note: The numba compiler only supports a subset of
            valid Python/numpy operations.

            Please read more about the `supported python features
            <https://numba.pydata.org/numba-doc/dev/reference/pysupported.html>`_
            and `supported numpy features
            <https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html>`_
            in numba to learn what you can or cannot use in the passed function.

            .. versionadded:: 2.2.0

        engine_kwargs : dict
            Pass keyword arguments to the engine.
            This is currently only used by the numba engine,
            see the documentation for the engine argument for more information.
        **kwargs
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        Series or DataFrame
            Result of applying ``func`` along the given axis of the
            DataFrame.

        See Also
        --------
        DataFrame.map: For elementwise operations.
        DataFrame.aggregate: Only perform aggregating type operations.
        DataFrame.transform: Only perform transforming type operations.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame([[4, 9]] * 3, columns=["A", "B"])
        >>> df
           A  B
        0  4  9
        1  4  9
        2  4  9

        Using a numpy universal function (in this case the same as
        ``np.sqrt(df)``):

        >>> df.apply(np.sqrt)
             A    B
        0  2.0  3.0
        1  2.0  3.0
        2  2.0  3.0

        Using a reducing function on either axis

        >>> df.apply(np.sum, axis=0)
        A    12
        B    27
        dtype: int64

        >>> df.apply(np.sum, axis=1)
        0    13
        1    13
        2    13
        dtype: int64

        Returning a list-like will result in a Series

        >>> df.apply(lambda x: [1, 2], axis=1)
        0    [1, 2]
        1    [1, 2]
        2    [1, 2]
        dtype: object

        Passing ``result_type='expand'`` will expand list-like results
        to columns of a Dataframe

        >>> df.apply(lambda x: [1, 2], axis=1, result_type="expand")
           0  1
        0  1  2
        1  1  2
        2  1  2

        Returning a Series inside the function is similar to passing
        ``result_type='expand'``. The resulting column names
        will be the Series index.

        >>> df.apply(lambda x: pd.Series([1, 2], index=["foo", "bar"]), axis=1)
           foo  bar
        0    1    2
        1    1    2
        2    1    2

        Passing ``result_type='broadcast'`` will ensure the same shape
        result, whether list-like or scalar is returned by the function,
        and broadcast it along the axis. The resulting column names will
        be the originals.

        >>> df.apply(lambda x: [1, 2], axis=1, result_type="broadcast")
           A  B
        0  1  2
        1  1  2
        2  1  2
        """
        from pandas.core.apply import frame_apply
        op = frame_apply(self, func=func, axis=axis, raw=raw, result_type=result_type, by_row=by_row, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)
        return op.apply().__finalize__(self, method='apply')

    def map(self, func: Any, na_action: None=None, **kwargs):
        """
        Apply a function to a Dataframe elementwise.

        .. versionadded:: 2.1.0

           DataFrame.applymap was deprecated and renamed to DataFrame.map.

        This method applies a function that accepts and returns a scalar
        to every element of a DataFrame.

        Parameters
        ----------
        func : callable
            Python function, returns a single value from a single value.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to func.
        **kwargs
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        DataFrame
            Transformed DataFrame.

        See Also
        --------
        DataFrame.apply : Apply a function along input axis of DataFrame.
        DataFrame.replace: Replace values given in `to_replace` with `value`.
        Series.map : Apply a function elementwise on a Series.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2.12], [3.356, 4.567]])
        >>> df
               0      1
        0  1.000  2.120
        1  3.356  4.567

        >>> df.map(lambda x: len(str(x)))
           0  1
        0  3  4
        1  5  5

        Like Series.map, NA values can be ignored:

        >>> df_copy = df.copy()
        >>> df_copy.iloc[0, 0] = pd.NA
        >>> df_copy.map(lambda x: len(str(x)), na_action="ignore")
             0  1
        0  NaN  4
        1  5.0  5

        It is also possible to use `map` with functions that are not
        `lambda` functions:

        >>> df.map(round, ndigits=1)
             0    1
        0  1.0  2.1
        1  3.4  4.6

        Note that a vectorized version of `func` often exists, which will
        be much faster. You could square each number elementwise.

        >>> df.map(lambda x: x**2)
                   0          1
        0   1.000000   4.494400
        1  11.262736  20.857489

        But it's better to avoid map in that case.

        >>> df**2
                   0          1
        0   1.000000   4.494400
        1  11.262736  20.857489
        """
        if na_action not in {'ignore', None}:
            raise ValueError(f"na_action must be 'ignore' or None. Got {na_action!r}")
        if self.empty:
            return self.copy()
        func = functools.partial(func, **kwargs)

        def infer(x: Any):
            return x._map_values(func, na_action=na_action)
        return self.apply(infer).__finalize__(self, 'map')

    def _append(self, other: Any, ignore_index: bool=False, verify_integrity: bool=False, sort: bool=False):
        if isinstance(other, (Series, dict)):
            if isinstance(other, dict):
                if not ignore_index:
                    raise TypeError('Can only append a dict if ignore_index=True')
                other = Series(other)
            if other.name is None and (not ignore_index):
                raise TypeError('Can only append a Series if ignore_index=True or if the Series has a name')
            index = Index([other.name], name=self.index.names if isinstance(self.index, MultiIndex) else self.index.name)
            row_df = other.to_frame().T
            other = row_df.infer_objects().rename_axis(index.names)
        elif isinstance(other, list):
            if not other:
                pass
            elif not isinstance(other[0], DataFrame):
                other = DataFrame(other)
                if self.index.name is not None and (not ignore_index):
                    other.index.name = self.index.name
        from pandas.core.reshape.concat import concat
        if isinstance(other, (list, tuple)):
            to_concat = [self, *other]
        else:
            to_concat = [self, other]
        result = concat(to_concat, ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)
        return result.__finalize__(self, method='append')

    def join(self, other: Any, on: None=None, how: typing.Text='left', lsuffix: typing.Text='', rsuffix: typing.Text='', sort: bool=False, validate: None=None):
        """
        Join columns of another DataFrame.

        Join columns with `other` DataFrame either on index or on a key
        column. Efficiently join multiple DataFrame objects by index at once by
        passing a list.

        Parameters
        ----------
        other : DataFrame, Series, or a list containing any combination of them
            Index should be similar to one of the columns in this one. If a
            Series is passed, its name attribute must be set, and that will be
            used as the column name in the resulting joined DataFrame.
        on : str, list of str, or array-like, optional
            Column or index level name(s) in the caller to join on the index
            in `other`, otherwise joins index-on-index. If multiple
            values given, the `other` DataFrame must have a MultiIndex. Can
            pass an array as the join key if it is not already contained in
            the calling DataFrame. Like an Excel VLOOKUP operation.
        how : {'left', 'right', 'outer', 'inner', 'cross', 'left_anti', 'right_anti'},
            default 'left'
            How to handle the operation of the two objects.

            * left: use calling frame's index (or column if on is specified)
            * right: use `other`'s index.
            * outer: form union of calling frame's index (or column if on is
              specified) with `other`'s index, and sort it lexicographically.
            * inner: form intersection of calling frame's index (or column if
              on is specified) with `other`'s index, preserving the order
              of the calling's one.
            * cross: creates the cartesian product from both frames, preserves the order
              of the left keys.
            * left_anti: use set difference of calling frame's index and `other`'s
              index.
            * right_anti: use set difference of `other`'s index and calling frame's
              index.
        lsuffix : str, default ''
            Suffix to use from left frame's overlapping columns.
        rsuffix : str, default ''
            Suffix to use from right frame's overlapping columns.
        sort : bool, default False
            Order result DataFrame lexicographically by the join key. If False,
            the order of the join key depends on the join type (how keyword).
        validate : str, optional
            If specified, checks if join is of specified type.

            * "one_to_one" or "1:1": check if join keys are unique in both left
              and right datasets.
            * "one_to_many" or "1:m": check if join keys are unique in left dataset.
            * "many_to_one" or "m:1": check if join keys are unique in right dataset.
            * "many_to_many" or "m:m": allowed, but does not result in checks.

            .. versionadded:: 1.5.0

        Returns
        -------
        DataFrame
            A dataframe containing columns from both the caller and `other`.

        See Also
        --------
        DataFrame.merge : For column(s)-on-column(s) operations.

        Notes
        -----
        Parameters `on`, `lsuffix`, and `rsuffix` are not supported when
        passing a list of `DataFrame` objects.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "key": ["K0", "K1", "K2", "K3", "K4", "K5"],
        ...         "A": ["A0", "A1", "A2", "A3", "A4", "A5"],
        ...     }
        ... )

        >>> df
          key   A
        0  K0  A0
        1  K1  A1
        2  K2  A2
        3  K3  A3
        4  K4  A4
        5  K5  A5

        >>> other = pd.DataFrame({"key": ["K0", "K1", "K2"], "B": ["B0", "B1", "B2"]})

        >>> other
          key   B
        0  K0  B0
        1  K1  B1
        2  K2  B2

        Join DataFrames using their indexes.

        >>> df.join(other, lsuffix="_caller", rsuffix="_other")
          key_caller   A key_other    B
        0         K0  A0        K0   B0
        1         K1  A1        K1   B1
        2         K2  A2        K2   B2
        3         K3  A3       NaN  NaN
        4         K4  A4       NaN  NaN
        5         K5  A5       NaN  NaN

        If we want to join using the key columns, we need to set key to be
        the index in both `df` and `other`. The joined DataFrame will have
        key as its index.

        >>> df.set_index("key").join(other.set_index("key"))
              A    B
        key
        K0   A0   B0
        K1   A1   B1
        K2   A2   B2
        K3   A3  NaN
        K4   A4  NaN
        K5   A5  NaN

        Another option to join using the key columns is to use the `on`
        parameter. DataFrame.join always uses `other`'s index but we can use
        any column in `df`. This method preserves the original DataFrame's
        index in the result.

        >>> df.join(other.set_index("key"), on="key")
          key   A    B
        0  K0  A0   B0
        1  K1  A1   B1
        2  K2  A2   B2
        3  K3  A3  NaN
        4  K4  A4  NaN
        5  K5  A5  NaN

        Using non-unique key values shows how they are matched.

        >>> df = pd.DataFrame(
        ...     {
        ...         "key": ["K0", "K1", "K1", "K3", "K0", "K1"],
        ...         "A": ["A0", "A1", "A2", "A3", "A4", "A5"],
        ...     }
        ... )

        >>> df
          key   A
        0  K0  A0
        1  K1  A1
        2  K1  A2
        3  K3  A3
        4  K0  A4
        5  K1  A5

        >>> df.join(other.set_index("key"), on="key", validate="m:1")
          key   A    B
        0  K0  A0   B0
        1  K1  A1   B1
        2  K1  A2   B1
        3  K3  A3  NaN
        4  K0  A4   B0
        5  K1  A5   B1
        """
        from pandas.core.reshape.concat import concat
        from pandas.core.reshape.merge import merge
        if isinstance(other, Series):
            if other.name is None:
                raise ValueError('Other Series must have a name')
            other = DataFrame({other.name: other})
        if isinstance(other, DataFrame):
            if how == 'cross':
                return merge(self, other, how=how, on=on, suffixes=(lsuffix, rsuffix), sort=sort, validate=validate)
            return merge(self, other, left_on=on, how=how, left_index=on is None, right_index=True, suffixes=(lsuffix, rsuffix), sort=sort, validate=validate)
        else:
            if on is not None:
                raise ValueError('Joining multiple DataFrames only supported for joining on index')
            if rsuffix or lsuffix:
                raise ValueError('Suffixes not supported when joining multiple DataFrames')
            frames = [cast('DataFrame | Series', self)] + list(other)
            can_concat = all((df.index.is_unique for df in frames))
            if can_concat:
                if how == 'left':
                    res = concat(frames, axis=1, join='outer', verify_integrity=True, sort=sort)
                    return res.reindex(self.index)
                else:
                    return concat(frames, axis=1, join=how, verify_integrity=True, sort=sort)
            joined = frames[0]
            for frame in frames[1:]:
                joined = merge(joined, frame, how=how, left_index=True, right_index=True, validate=validate)
            return joined

    @Substitution('')
    @Appender(_merge_doc, indents=2)
    def merge(self, right: Any, how: typing.Text='inner', on: None=None, left_on: None=None, right_on: None=None, left_index: bool=False, right_index: bool=False, sort: bool=False, suffixes: tuple[typing.Text]=('_x', '_y'), copy=lib.no_default, indicator=False, validate=None):
        self._check_copy_deprecation(copy)
        from pandas.core.reshape.merge import merge
        return merge(self, right, how=how, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, sort=sort, suffixes=suffixes, indicator=indicator, validate=validate)

    def round(self, decimals: Any=0, *args, **kwargs):
        """
        Round numeric columns in a DataFrame to a variable number of decimal places.

        Parameters
        ----------
        decimals : int, dict, Series
            Number of decimal places to round each column to. If an int is
            given, round each column to the same number of places.
            Otherwise dict and Series round to variable numbers of places.
            Column names should be in the keys if `decimals` is a
            dict-like, or in the index if `decimals` is a Series. Any
            columns not included in `decimals` will be left as is. Elements
            of `decimals` which are not columns of the input will be
            ignored.
        *args
            Additional keywords have no effect but might be accepted for
            compatibility with numpy.
        **kwargs
            Additional keywords have no effect but might be accepted for
            compatibility with numpy.

        Returns
        -------
        DataFrame
            A DataFrame with the affected columns rounded to the specified
            number of decimal places.

        See Also
        --------
        numpy.around : Round a numpy array to the given number of decimals.
        Series.round : Round a Series to the given number of decimals.

        Notes
        -----
        For values exactly halfway between rounded decimal values, pandas rounds
        to the nearest even value (e.g. -0.5 and 0.5 round to 0.0, 1.5 and 2.5
        round to 2.0, etc.).

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [(0.21, 0.32), (0.01, 0.67), (0.66, 0.03), (0.21, 0.18)],
        ...     columns=["dogs", "cats"],
        ... )
        >>> df
            dogs  cats
        0  0.21  0.32
        1  0.01  0.67
        2  0.66  0.03
        3  0.21  0.18

        By providing an integer each column is rounded to the same number
        of decimal places

        >>> df.round(1)
            dogs  cats
        0   0.2   0.3
        1   0.0   0.7
        2   0.7   0.0
        3   0.2   0.2

        With a dict, the number of places for specific columns can be
        specified with the column names as key and the number of decimal
        places as value

        >>> df.round({"dogs": 1, "cats": 0})
            dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0

        Using a Series, the number of places for specific columns can be
        specified with the column names as index and the number of
        decimal places as value

        >>> decimals = pd.Series([0, 1], index=["cats", "dogs"])
        >>> df.round(decimals)
            dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0
        """
        from pandas.core.reshape.concat import concat

        def _dict_round(df: Any, decimals: Any) -> typing.Generator:
            for col, vals in df.items():
                try:
                    yield _series_round(vals, decimals[col])
                except KeyError:
                    yield vals

        def _series_round(ser: Any, decimals: Any):
            if is_integer_dtype(ser.dtype) or is_float_dtype(ser.dtype):
                return ser.round(decimals)
            return ser
        nv.validate_round(args, kwargs)
        if isinstance(decimals, (dict, Series)):
            if isinstance(decimals, Series) and (not decimals.index.is_unique):
                raise ValueError('Index of decimals must be unique')
            if is_dict_like(decimals) and (not all((is_integer(value) for _, value in decimals.items()))):
                raise TypeError('Values in decimals must be integers')
            new_cols = list(_dict_round(self, decimals))
        elif is_integer(decimals):
            new_mgr = self._mgr.round(decimals=decimals)
            return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self, method='round')
        else:
            raise TypeError('decimals must be an integer, a dict-like or a Series')
        if new_cols is not None and len(new_cols) > 0:
            return self._constructor(concat(new_cols, axis=1), index=self.index, columns=self.columns).__finalize__(self, method='round')
        else:
            return self.copy(deep=False)

    def corr(self, method: typing.Text='pearson', min_periods: int=1, numeric_only: bool=False):
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method of correlation:

            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * callable: callable with input two 1d ndarrays
                and returning a float. Note that the returned matrix from corr
                will have 1 along the diagonals and will be symmetric
                regardless of the callable's behavior.
        min_periods : int, optional
            Minimum number of observations required per pair of columns
            to have a valid result. Currently only available for Pearson
            and Spearman correlation.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.

        Returns
        -------
        DataFrame
            Correlation matrix.

        See Also
        --------
        DataFrame.corrwith : Compute pairwise correlation with another
            DataFrame or Series.
        Series.corr : Compute the correlation between two Series.

        Notes
        -----
        Pearson, Kendall and Spearman correlation are currently computed using pairwise complete observations.

        * `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
        * `Kendall rank correlation coefficient <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_
        * `Spearman's rank correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_

        Examples
        --------
        >>> def histogram_intersection(a, b):
        ...     v = np.minimum(a, b).sum().round(decimals=1)
        ...     return v
        >>> df = pd.DataFrame(
        ...     [(0.2, 0.3), (0.0, 0.6), (0.6, 0.0), (0.2, 0.1)],
        ...     columns=["dogs", "cats"],
        ... )
        >>> df.corr(method=histogram_intersection)
              dogs  cats
        dogs   1.0   0.3
        cats   0.3   1.0

        >>> df = pd.DataFrame(
        ...     [(1, 1), (2, np.nan), (np.nan, 3), (4, 4)], columns=["dogs", "cats"]
        ... )
        >>> df.corr(min_periods=3)
              dogs  cats
        dogs   1.0   NaN
        cats   NaN   1.0
        """
        data = self._get_numeric_data() if numeric_only else self
        cols = data.columns
        idx = cols.copy()
        mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
        if method == 'pearson':
            correl = libalgos.nancorr(mat, minp=min_periods)
        elif method == 'spearman':
            correl = libalgos.nancorr_spearman(mat, minp=min_periods)
        elif method == 'kendall' or callable(method):
            if min_periods is None:
                min_periods = 1
            mat = mat.T
            corrf = nanops.get_corr_func(method)
            K = len(cols)
            correl = np.empty((K, K), dtype=float)
            mask = np.isfinite(mat)
            for i, ac in enumerate(mat):
                for j, bc in enumerate(mat):
                    if i > j:
                        continue
                    valid = mask[i] & mask[j]
                    if valid.sum() < min_periods:
                        c = np.nan
                    elif i == j:
                        c = 1.0
                    elif not valid.all():
                        c = corrf(ac[valid], bc[valid])
                    else:
                        c = corrf(ac, bc)
                    correl[i, j] = c
                    correl[j, i] = c
        else:
            raise ValueError(f"method must be either 'pearson', 'spearman', 'kendall', or a callable, '{method}' was supplied")
        result = self._constructor(correl, index=idx, columns=cols, copy=False)
        return result.__finalize__(self, method='corr')

    def cov(self, min_periods: Any=None, ddof: int=1, numeric_only: bool=False):
        """
        Compute pairwise covariance of columns, excluding NA/null values.

        Compute the pairwise covariance among the series of a DataFrame.
        The returned data frame is the `covariance matrix
        <https://en.wikipedia.org/wiki/Covariance_matrix>`__ of the columns
        of the DataFrame.

        Both NA and null values are automatically excluded from the
        calculation. (See the note below about bias from missing values.)
        A threshold can be set for the minimum number of
        observations for each value created. Comparisons with observations
        below this threshold will be returned as ``NaN``.

        This method is generally used for the analysis of time series data to
        understand the relationship between different measures
        across time.

        Parameters
        ----------
        min_periods : int, optional
            Minimum number of observations required per pair of columns
            to have a valid result.

        ddof : int, default 1
            Delta degrees of freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            This argument is applicable only when no ``nan`` is in the dataframe.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.

        Returns
        -------
        DataFrame
            The covariance matrix of the series of the DataFrame.

        See Also
        --------
        Series.cov : Compute covariance with another Series.
        core.window.ewm.ExponentialMovingWindow.cov : Exponential weighted sample
            covariance.
        core.window.expanding.Expanding.cov : Expanding sample covariance.
        core.window.rolling.Rolling.cov : Rolling sample covariance.

        Notes
        -----
        Returns the covariance matrix of the DataFrame's time series.
        The covariance is normalized by N-ddof.

        For DataFrames that have Series that are missing data (assuming that
        data is `missing at random
        <https://en.wikipedia.org/wiki/Missing_data#Missing_at_random>`__)
        the returned covariance matrix will be an unbiased estimate
        of the variance and covariance between the member Series.

        However, for many applications this estimate may not be acceptable
        because the estimate covariance matrix is not guaranteed to be positive
        semi-definite. This could lead to estimate correlations having
        absolute values which are greater than one, and/or a non-invertible
        covariance matrix. See `Estimation of covariance matrices
        <https://en.wikipedia.org/w/index.php?title=Estimation_of_covariance_
        matrices>`__ for more details.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [(1, 2), (0, 3), (2, 0), (1, 1)], columns=["dogs", "cats"]
        ... )
        >>> df.cov()
                  dogs      cats
        dogs  0.666667 -1.000000
        cats -1.000000  1.666667

        >>> np.random.seed(42)
        >>> df = pd.DataFrame(
        ...     np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"]
        ... )
        >>> df.cov()
                  a         b         c         d         e
        a  0.998438 -0.020161  0.059277 -0.008943  0.014144
        b -0.020161  1.059352 -0.008543 -0.024738  0.009826
        c  0.059277 -0.008543  1.010670 -0.001486 -0.000271
        d -0.008943 -0.024738 -0.001486  0.921297 -0.013692
        e  0.014144  0.009826 -0.000271 -0.013692  0.977795

        **Minimum number of periods**

        This method also supports an optional ``min_periods`` keyword
        that specifies the required minimum number of non-NA observations for
        each column pair in order to have a valid result:

        >>> np.random.seed(42)
        >>> df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        >>> df.loc[df.index[:5], "a"] = np.nan
        >>> df.loc[df.index[5:10], "b"] = np.nan
        >>> df.cov(min_periods=12)
                  a         b         c
        a  0.316741       NaN -0.150812
        b       NaN  1.248003  0.191417
        c -0.150812  0.191417  0.895202
        """
        data = self._get_numeric_data() if numeric_only else self
        cols = data.columns
        idx = cols.copy()
        mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
        if notna(mat).all():
            if min_periods is not None and min_periods > len(mat):
                base_cov = np.empty((mat.shape[1], mat.shape[1]))
                base_cov.fill(np.nan)
            else:
                base_cov = np.cov(mat.T, ddof=ddof)
            base_cov = base_cov.reshape((len(cols), len(cols)))
        else:
            base_cov = libalgos.nancorr(mat, cov=True, minp=min_periods)
        result = self._constructor(base_cov, index=idx, columns=cols, copy=False)
        return result.__finalize__(self, method='cov')

    def corrwith(self, other: Any, axis: int=0, drop: bool=False, method: typing.Text='pearson', numeric_only: bool=False, min_periods: None=None):
        """
        Compute pairwise correlation.

        Pairwise correlation is computed between rows or columns of
        DataFrame with rows or columns of Series or DataFrame. DataFrames
        are first aligned along both axes before computing the
        correlations.

        Parameters
        ----------
        other : DataFrame, Series
            Object with which to compute correlations.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to use. 0 or 'index' to compute row-wise, 1 or 'columns' for
            column-wise.
        drop : bool, default False
            Drop missing indices from result.
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method of correlation:

            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * callable: callable with input two 1d ndarrays
                and returning a float.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

        min_periods : int, optional
            Minimum number of observations needed to have a valid result.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.

        Returns
        -------
        Series
            Pairwise correlations.

        See Also
        --------
        DataFrame.corr : Compute pairwise correlation of columns.

        Examples
        --------
        >>> index = ["a", "b", "c", "d", "e"]
        >>> columns = ["one", "two", "three", "four"]
        >>> df1 = pd.DataFrame(
        ...     np.arange(20).reshape(5, 4), index=index, columns=columns
        ... )
        >>> df2 = pd.DataFrame(
        ...     np.arange(16).reshape(4, 4), index=index[:4], columns=columns
        ... )
        >>> df1.corrwith(df2)
        one      1.0
        two      1.0
        three    1.0
        four     1.0
        dtype: float64

        >>> df2.corrwith(df1, axis=1)
        a    1.0
        b    1.0
        c    1.0
        d    1.0
        e    NaN
        dtype: float64
        """
        axis = self._get_axis_number(axis)
        this = self._get_numeric_data() if numeric_only else self
        if isinstance(other, Series):
            return this.apply(lambda x: other.corr(x, method=method, min_periods=min_periods), axis=axis)
        if numeric_only:
            other = other._get_numeric_data()
        left, right = this.align(other, join='inner')
        if axis == 1:
            left = left.T
            right = right.T
        if method == 'pearson':
            left = left + right * 0
            right = right + left * 0
            ldem = left - left.mean(numeric_only=numeric_only)
            rdem = right - right.mean(numeric_only=numeric_only)
            num = (ldem * rdem).sum()
            dom = (left.count() - 1) * left.std(numeric_only=numeric_only) * right.std(numeric_only=numeric_only)
            correl = num / dom
        elif method in ['kendall', 'spearman'] or callable(method):

            def c(x: Any):
                return nanops.nancorr(x[0], x[1], method=method)
            correl = self._constructor_sliced(map(c, zip(left.values.T, right.values.T)), index=left.columns, copy=False)
        else:
            raise ValueError(f"Invalid method {method} was passed, valid methods are: 'pearson', 'kendall', 'spearman', or callable")
        if not drop:
            raxis = 1 if axis == 0 else 0
            result_index = this._get_axis(raxis).union(other._get_axis(raxis))
            idx_diff = result_index.difference(correl.index)
            if len(idx_diff) > 0:
                correl = correl._append(Series([np.nan] * len(idx_diff), index=idx_diff))
        return correl

    def count(self, axis: int=0, numeric_only: bool=False):
        """
        Count non-NA cells for each column or row.

        The values `None`, `NaN`, `NaT`, ``pandas.NA`` are considered NA.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            If 0 or 'index' counts are generated for each column.
            If 1 or 'columns' counts are generated for each row.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

        Returns
        -------
        Series
            For each column/row the number of non-NA/null entries.

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.value_counts: Count unique combinations of columns.
        DataFrame.shape: Number of DataFrame rows and columns (including NA
            elements).
        DataFrame.isna: Boolean same-sized DataFrame showing places of NA
            elements.

        Examples
        --------
        Constructing DataFrame from a dictionary:

        >>> df = pd.DataFrame(
        ...     {
        ...         "Person": ["John", "Myla", "Lewis", "John", "Myla"],
        ...         "Age": [24.0, np.nan, 21.0, 33, 26],
        ...         "Single": [False, True, True, True, False],
        ...     }
        ... )
        >>> df
           Person   Age  Single
        0    John  24.0   False
        1    Myla   NaN    True
        2   Lewis  21.0    True
        3    John  33.0    True
        4    Myla  26.0   False

        Notice the uncounted NA values:

        >>> df.count()
        Person    5
        Age       4
        Single    5
        dtype: int64

        Counts for each **row**:

        >>> df.count(axis="columns")
        0    3
        1    2
        2    3
        3    3
        4    3
        dtype: int64
        """
        axis = self._get_axis_number(axis)
        if numeric_only:
            frame = self._get_numeric_data()
        else:
            frame = self
        if len(frame._get_axis(axis)) == 0:
            result = self._constructor_sliced(0, index=frame._get_agg_axis(axis))
        else:
            result = notna(frame).sum(axis=axis)
        return result.astype('int64').__finalize__(self, method='count')

    def _reduce(self, op: Any, name: Any, *, axis: int=0, skipna: bool=True, numeric_only: bool=False, filter_type: None=None, **kwds):
        assert filter_type is None or filter_type == 'bool', filter_type
        out_dtype = 'bool' if filter_type == 'bool' else None
        if axis is not None:
            axis = self._get_axis_number(axis)

        def func(values: Any):
            return op(values, axis=axis, skipna=skipna, **kwds)

        def blk_func(values: Any, axis: int=1):
            if isinstance(values, ExtensionArray):
                if not is_1d_only_ea_dtype(values.dtype):
                    return values._reduce(name, axis=1, skipna=skipna, **kwds)
                return values._reduce(name, skipna=skipna, keepdims=True, **kwds)
            else:
                return op(values, axis=axis, skipna=skipna, **kwds)

        def _get_data():
            if filter_type is None:
                data = self._get_numeric_data()
            else:
                assert filter_type == 'bool'
                data = self._get_bool_data()
            return data
        df = self
        if numeric_only:
            df = _get_data()
        if axis is None:
            dtype = find_common_type([block.values.dtype for block in df._mgr.blocks])
            if isinstance(dtype, ExtensionDtype):
                df = df.astype(dtype)
                arr = concat_compat(list(df._iter_column_arrays()))
                return arr._reduce(name, skipna=skipna, keepdims=False, **kwds)
            return func(df.values)
        elif axis == 1:
            if len(df.index) == 0:
                result = df._reduce(op, name, axis=0, skipna=skipna, numeric_only=False, filter_type=filter_type, **kwds).iloc[:0]
                result.index = df.index
                return result
            if df.shape[1] and name != 'kurt':
                dtype = find_common_type([block.values.dtype for block in df._mgr.blocks])
                if isinstance(dtype, ExtensionDtype):
                    name = {'argmax': 'idxmax', 'argmin': 'idxmin'}.get(name, name)
                    df = df.astype(dtype)
                    arr = concat_compat(list(df._iter_column_arrays()))
                    nrows, ncols = df.shape
                    row_index = np.tile(np.arange(nrows), ncols)
                    col_index = np.repeat(np.arange(ncols), nrows)
                    ser = Series(arr, index=col_index, copy=False)
                    with rewrite_warning(target_message=f'The behavior of SeriesGroupBy.{name} with all-NA values', target_category=FutureWarning, new_message=f'The behavior of {type(self).__name__}.{name} with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError'):
                        result = ser.groupby(row_index).agg(name, **kwds)
                    result.index = df.index
                    if not skipna and name not in ('any', 'all'):
                        mask = df.isna().to_numpy(dtype=np.bool_).any(axis=1)
                        other = -1 if name in ('idxmax', 'idxmin') else lib.no_default
                        result = result.mask(mask, other)
                    return result
            df = df.T
        res = df._mgr.reduce(blk_func)
        out = df._constructor_from_mgr(res, axes=res.axes).iloc[0]
        if out_dtype is not None and out.dtype != 'boolean':
            out = out.astype(out_dtype)
        elif (df._mgr.get_dtypes() == object).any() and name not in ['any', 'all']:
            out = out.astype(object)
        elif len(self) == 0 and out.dtype == object and (name in ('sum', 'prod')):
            out = out.astype(np.float64)
        return out

    def _reduce_axis1(self, name: Any, func: Any, skipna: Any):
        """
        Special case for _reduce to try to avoid a potentially-expensive transpose.

        Apply the reduction block-wise along axis=1 and then reduce the resulting
        1D arrays.
        """
        if name == 'all':
            result = np.ones(len(self), dtype=bool)
            ufunc = np.logical_and
        elif name == 'any':
            result = np.zeros(len(self), dtype=bool)
            ufunc = np.logical_or
        else:
            raise NotImplementedError(name)
        for blocks in self._mgr.blocks:
            middle = func(blocks.values, axis=0, skipna=skipna)
            result = ufunc(result, middle)
        res_ser = self._constructor_sliced(result, index=self.index, copy=False)
        return res_ser

    @overload
    def any(self, *, axis: Any=..., bool_only: Any=..., skipna: Any=..., **kwargs) -> None:
        ...

    @overload
    def any(self, *, axis: Any, bool_only: Any=..., skipna: Any=..., **kwargs) -> None:
        ...

    @overload
    def any(self, *, axis: Any, bool_only: Any=..., skipna: Any=..., **kwargs) -> None:
        ...

    @doc(make_doc('any', ndim=1))
    def any(self, *, axis: Any=0, bool_only: Any=False, skipna: Any=True, **kwargs) -> None:
        result = self._logical_func('any', nanops.nanany, axis, bool_only, skipna, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='any')
        return result

    @overload
    def all(self, *, axis: Any=..., bool_only: Any=..., skipna: Any=..., **kwargs) -> None:
        ...

    @overload
    def all(self, *, axis: Any, bool_only: Any=..., skipna: Any=..., **kwargs) -> None:
        ...

    @overload
    def all(self, *, axis: Any, bool_only: Any=..., skipna: Any=..., **kwargs) -> None:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='all')
    @doc(make_doc('all', ndim=1))
    def all(self, axis: Any=0, bool_only: Any=False, skipna: Any=True, **kwargs) -> None:
        result = self._logical_func('all', nanops.nanall, axis, bool_only, skipna, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='all')
        return result

    @overload
    def min(self, *, axis: Any=..., skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def min(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def min(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='min')
    @doc(make_doc('min', ndim=2))
    def min(self, axis: Any=0, skipna: Any=True, numeric_only: Any=False, **kwargs) -> None:
        result = super().min(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='min')
        return result

    @overload
    def max(self, *, axis: Any=..., skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def max(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def max(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='max')
    @doc(make_doc('max', ndim=2))
    def max(self, axis: Any=0, skipna: Any=True, numeric_only: Any=False, **kwargs) -> None:
        result = super().max(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='max')
        return result

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='sum')
    def sum(self, axis: int=0, skipna: bool=True, numeric_only: bool=False, min_count: int=0, **kwargs):
        """
        Return the sum of the values over the requested axis.

        This is equivalent to the method ``numpy.sum``.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.sum with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.
        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer than
            ``min_count`` non-NA values are present the result will be NA.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or scalar
            Sum over requested axis.

        See Also
        --------
        Series.sum : Return the sum over Series values.
        DataFrame.mean : Return the mean of the values over the requested axis.
        DataFrame.median : Return the median of the values over the requested axis.
        DataFrame.mode : Get the mode(s) of each element along the requested axis.
        DataFrame.std : Return the standard deviation of the values over the
            requested axis.

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
        Name: legs, dtype: int64

        >>> s.sum()
        14

        By default, the sum of an empty or all-NA Series is ``0``.

        >>> pd.Series([], dtype="float64").sum()  # min_count=0 is the default
        0.0

        This can be controlled with the ``min_count`` parameter. For example, if
        you'd like the sum of an empty series to be NaN, pass ``min_count=1``.

        >>> pd.Series([], dtype="float64").sum(min_count=1)
        nan

        Thanks to the ``skipna`` parameter, ``min_count`` handles all-NA and
        empty series identically.

        >>> pd.Series([np.nan]).sum()
        0.0

        >>> pd.Series([np.nan]).sum(min_count=1)
        nan
        """
        result = super().sum(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='sum')
        return result

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='prod')
    def prod(self, axis: int=0, skipna: bool=True, numeric_only: bool=False, min_count: int=0, **kwargs):
        """
        Return the product of the values over the requested axis.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.prod with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer than
            ``min_count`` non-NA values are present the result will be NA.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or scalar
            The product of the values over the requested axis.

        See Also
        --------
        Series.sum : Return the sum.
        Series.min : Return the minimum.
        Series.max : Return the maximum.
        Series.idxmin : Return the index of the minimum.
        Series.idxmax : Return the index of the maximum.
        DataFrame.sum : Return the sum over the requested axis.
        DataFrame.min : Return the minimum over the requested axis.
        DataFrame.max : Return the maximum over the requested axis.
        DataFrame.idxmin : Return the index of the minimum over the requested axis.
        DataFrame.idxmax : Return the index of the maximum over the requested axis.

        Examples
        --------
        By default, the product of an empty or all-NA Series is ``1``

        >>> pd.Series([], dtype="float64").prod()
        1.0

        This can be controlled with the ``min_count`` parameter

        >>> pd.Series([], dtype="float64").prod(min_count=1)
        nan

        Thanks to the ``skipna`` parameter, ``min_count`` handles all-NA and
        empty series identically.

        >>> pd.Series([np.nan]).prod()
        1.0

        >>> pd.Series([np.nan]).prod(min_count=1)
        nan
        """
        result = super().prod(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='prod')
        return result

    @overload
    def mean(self, *, axis: Any=..., skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def mean(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def mean(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='mean')
    @doc(make_doc('mean', ndim=2))
    def mean(self, axis: Any=0, skipna: Any=True, numeric_only: Any=False, **kwargs) -> None:
        result = super().mean(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='mean')
        return result

    @overload
    def median(self, *, axis: Any=..., skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def median(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def median(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='median')
    @doc(make_doc('median', ndim=2))
    def median(self, axis: Any=0, skipna: Any=True, numeric_only: Any=False, **kwargs) -> None:
        result = super().median(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='median')
        return result

    @overload
    def sem(self, *, axis: Any=..., skipna: Any=..., ddof: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def sem(self, *, axis: Any, skipna: Any=..., ddof: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def sem(self, *, axis: Any, skipna: Any=..., ddof: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='sem')
    def sem(self, axis: Any=0, skipna: Any=True, ddof: Any=1, numeric_only: Any=False, **kwargs) -> None:
        """
        Return unbiased standard error of the mean over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Parameters
        ----------
        axis : {index (0), columns (1)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.sem with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.
        **kwargs :
            Additional keywords passed.

        Returns
        -------
        Series or DataFrame (if level specified)
            Unbiased standard error of the mean over requested axis.

        See Also
        --------
        DataFrame.var : Return unbiased variance over requested axis.
        DataFrame.std : Returns sample standard deviation over requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.sem().round(6)
        0.57735

        With a DataFrame

        >>> df = pd.DataFrame({"a": [1, 2], "b": [2, 3]}, index=["tiger", "zebra"])
        >>> df
               a   b
        tiger  1   2
        zebra  2   3
        >>> df.sem()
        a   0.5
        b   0.5
        dtype: float64

        Using axis=1

        >>> df.sem(axis=1)
        tiger   0.5
        zebra   0.5
        dtype: float64

        In this case, `numeric_only` should be set to `True`
        to avoid getting an error.

        >>> df = pd.DataFrame({"a": [1, 2], "b": ["T", "Z"]}, index=["tiger", "zebra"])
        >>> df.sem(numeric_only=True)
        a   0.5
        dtype: float64
        """
        result = super().sem(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='sem')
        return result

    @overload
    def var(self, *, axis: Any=..., skipna: Any=..., ddof: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def var(self, *, axis: Any, skipna: Any=..., ddof: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def var(self, *, axis: Any, skipna: Any=..., ddof: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='var')
    def var(self, axis: Any=0, skipna: Any=True, ddof: Any=1, numeric_only: Any=False, **kwargs) -> None:
        """
        Return unbiased variance over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.var with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.
        **kwargs :
            Additional keywords passed.

        Returns
        -------
        Series or scalaer
            Unbiased variance over requested axis.

        See Also
        --------
        numpy.var : Equivalent function in NumPy.
        Series.var : Return unbiased variance over Series values.
        Series.std : Return standard deviation over Series values.
        DataFrame.std : Return standard deviation of the values over
            the requested axis.

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
        result = super().var(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='var')
        return result

    @overload
    def std(self, *, axis: Any=..., skipna: Any=..., ddof: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def std(self, *, axis: Any, skipna: Any=..., ddof: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def std(self, *, axis: Any, skipna: Any=..., ddof: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='std')
    def std(self, axis: Any=0, skipna: Any=True, ddof: Any=1, numeric_only: Any=False, **kwargs) -> None:
        """
        Return sample standard deviation over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            For `Series` this parameter is unused and defaults to 0.

            .. warning::

                The behavior of DataFrame.std with ``axis=None`` is deprecated,
                in a future version this will reduce over both axes and return a scalar
                To retain the old behavior, pass axis=0 (or do not pass axis).

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.
        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.
        **kwargs : dict
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or scalar
            Standard deviation over requested axis.

        See Also
        --------
        Series.std : Return standard deviation over Series values.
        DataFrame.mean : Return the mean of the values over the requested axis.
        DataFrame.median : Return the median of the values over the requested axis.
        DataFrame.mode : Get the mode(s) of each element along the requested axis.
        DataFrame.sum : Return the sum of the values over the requested axis.

        Notes
        -----
        To have the same behaviour as `numpy.std`, use `ddof=0` (instead of the
        default `ddof=1`)

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

        The standard deviation of the columns can be found as follows:

        >>> df.std()
        age       18.786076
        height     0.237417
        dtype: float64

        Alternatively, `ddof=0` can be set to normalize by N instead of N-1:

        >>> df.std(ddof=0)
        age       16.269219
        height     0.205609
        dtype: float64
        """
        result = super().std(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='std')
        return result

    @overload
    def skew(self, *, axis: Any=..., skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def skew(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def skew(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='skew')
    def skew(self, axis: Any=0, skipna: Any=True, numeric_only: Any=False, **kwargs) -> None:
        """
        Return unbiased skew over requested axis.

        Normalized by N-1.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or scalar
            Unbiased skew over requested axis.

        See Also
        --------
        Dataframe.kurt : Returns unbiased kurtosis over requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.skew()
        0.0

        With a DataFrame

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2, 3], "b": [2, 3, 4], "c": [1, 3, 5]},
        ...     index=["tiger", "zebra", "cow"],
        ... )
        >>> df
                a   b   c
        tiger   1   2   1
        zebra   2   3   3
        cow     3   4   5
        >>> df.skew()
        a   0.0
        b   0.0
        c   0.0
        dtype: float64

        Using axis=1

        >>> df.skew(axis=1)
        tiger   1.732051
        zebra  -1.732051
        cow     0.000000
        dtype: float64

        In this case, `numeric_only` should be set to `True` to avoid
        getting an error.

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2, 3], "b": ["T", "Z", "X"]}, index=["tiger", "zebra", "cow"]
        ... )
        >>> df.skew(numeric_only=True)
        a   0.0
        dtype: float64
        """
        result = super().skew(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='skew')
        return result

    @overload
    def kurt(self, *, axis: Any=..., skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def kurt(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @overload
    def kurt(self, *, axis: Any, skipna: Any=..., numeric_only: Any=..., **kwargs) -> None:
        ...

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='kurt')
    def kurt(self, axis: Any=0, skipna: Any=True, numeric_only: Any=False, **kwargs) -> None:
        """
        Return unbiased kurtosis over requested axis.

        Kurtosis obtained using Fisher's definition of
        kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
            For `Series` this parameter is unused and defaults to 0.

            For DataFrames, specifying ``axis=None`` will apply the aggregation
            across both axes.

            .. versionadded:: 2.0.0

        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            Include only float, int, boolean columns.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or scalar
            Unbiased kurtosis over requested axis.

        See Also
        --------
        Dataframe.kurtosis : Returns unbiased kurtosis over requested axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 2, 3], index=["cat", "dog", "dog", "mouse"])
        >>> s
        cat    1
        dog    2
        dog    2
        mouse  3
        dtype: int64
        >>> s.kurt()
        1.5

        With a DataFrame

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2, 2, 3], "b": [3, 4, 4, 4]},
        ...     index=["cat", "dog", "dog", "mouse"],
        ... )
        >>> df
               a   b
          cat  1   3
          dog  2   4
          dog  2   4
        mouse  3   4
        >>> df.kurt()
        a   1.5
        b   4.0
        dtype: float64

        With axis=None

        >>> df.kurt(axis=None).round(6)
        -0.988693

        Using axis=1

        >>> df = pd.DataFrame(
        ...     {"a": [1, 2], "b": [3, 4], "c": [3, 4], "d": [1, 2]},
        ...     index=["cat", "dog"],
        ... )
        >>> df.kurt(axis=1)
        cat   -6.0
        dog   -6.0
        dtype: float64
        """
        result = super().kurt(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='kurt')
        return result
    kurtosis = kurt
    product = prod

    @doc(make_doc('cummin', ndim=2))
    def cummin(self, axis: int=0, skipna: bool=True, numeric_only: bool=False, *args, **kwargs):
        data = self._get_numeric_data() if numeric_only else self
        return NDFrame.cummin(data, axis, skipna, *args, **kwargs)

    @doc(make_doc('cummax', ndim=2))
    def cummax(self, axis: int=0, skipna: bool=True, numeric_only: bool=False, *args, **kwargs):
        data = self._get_numeric_data() if numeric_only else self
        return NDFrame.cummax(data, axis, skipna, *args, **kwargs)

    @doc(make_doc('cumsum', ndim=2))
    def cumsum(self, axis: int=0, skipna: bool=True, numeric_only: bool=False, *args, **kwargs):
        data = self._get_numeric_data() if numeric_only else self
        return NDFrame.cumsum(data, axis, skipna, *args, **kwargs)

    @doc(make_doc('cumprod', 2))
    def cumprod(self, axis: int=0, skipna: bool=True, numeric_only: bool=False, *args, **kwargs):
        data = self._get_numeric_data() if numeric_only else self
        return NDFrame.cumprod(data, axis, skipna, *args, **kwargs)

    def nunique(self, axis: int=0, dropna: bool=True):
        """
        Count number of distinct elements in specified axis.

        Return Series with number of distinct elements. Can ignore NaN
        values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for
            column-wise.
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        Series
            Series with counts of unique values per row or column, depending on `axis`.

        See Also
        --------
        Series.nunique: Method nunique for Series.
        DataFrame.count: Count non-NA cells for each column or row.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [4, 5, 6], "B": [4, 1, 1]})
        >>> df.nunique()
        A    3
        B    2
        dtype: int64

        >>> df.nunique(axis=1)
        0    1
        1    2
        2    2
        dtype: int64
        """
        return self.apply(Series.nunique, axis=axis, dropna=dropna)

    def idxmin(self, axis: int=0, skipna: bool=True, numeric_only: bool=False):
        """
        Return index of first occurrence of minimum over requested axis.

        NA/null values are excluded.

        Parameters
        ----------
        axis : {{0 or 'index', 1 or 'columns'}}, default 0
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
        skipna : bool, default True
            Exclude NA/null values. If the entire DataFrame is NA,
            or if ``skipna=False`` and there is an NA value, this method
            will raise a ``ValueError``.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series
            Indexes of minima along the specified axis.

        Raises
        ------
        ValueError
            * If the row/column is empty

        See Also
        --------
        Series.idxmin : Return index of the minimum element.

        Notes
        -----
        This method is the DataFrame version of ``ndarray.argmin``.

        Examples
        --------
        Consider a dataset containing food consumption in Argentina.

        >>> df = pd.DataFrame(
        ...     {
        ...         {
        ...             "consumption": [10.51, 103.11, 55.48],
        ...             "co2_emissions": [37.2, 19.66, 1712],
        ...         }
        ...     },
        ...     index=["Pork", "Wheat Products", "Beef"],
        ... )

        >>> df
                        consumption  co2_emissions
        Pork                  10.51         37.20
        Wheat Products       103.11         19.66
        Beef                  55.48       1712.00

        By default, it returns the index for the minimum value in each column.

        >>> df.idxmin()
        consumption                Pork
        co2_emissions    Wheat Products
        dtype: object

        To return the index for the minimum value in each row, use ``axis="columns"``.

        >>> df.idxmin(axis="columns")
        Pork                consumption
        Wheat Products    co2_emissions
        Beef                consumption
        dtype: object
        """
        axis = self._get_axis_number(axis)
        if self.empty and len(self.axes[axis]):
            axis_dtype = self.axes[axis].dtype
            return self._constructor_sliced(dtype=axis_dtype)
        if numeric_only:
            data = self._get_numeric_data()
        else:
            data = self
        res = data._reduce(nanops.nanargmin, 'argmin', axis=axis, skipna=skipna, numeric_only=False)
        indices = res._values
        if (indices == -1).any():
            warnings.warn(f'The behavior of {type(self).__name__}.idxmin with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError', FutureWarning, stacklevel=find_stack_level())
        index = data._get_axis(axis)
        result = algorithms.take(index._values, indices, allow_fill=True, fill_value=index._na_value)
        final_result = data._constructor_sliced(result, index=data._get_agg_axis(axis))
        return final_result.__finalize__(self, method='idxmin')

    def idxmax(self, axis: int=0, skipna: bool=True, numeric_only: bool=False):
        """
        Return index of first occurrence of maximum over requested axis.

        NA/null values are excluded.

        Parameters
        ----------
        axis : {{0 or 'index', 1 or 'columns'}}, default 0
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
        skipna : bool, default True
            Exclude NA/null values. If the entire DataFrame is NA,
            or if ``skipna=False`` and there is an NA value, this method
            will raise a ``ValueError``.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series
            Indexes of maxima along the specified axis.

        Raises
        ------
        ValueError
            * If the row/column is empty

        See Also
        --------
        Series.idxmax : Return index of the maximum element.

        Notes
        -----
        This method is the DataFrame version of ``ndarray.argmax``.

        Examples
        --------
        Consider a dataset containing food consumption in Argentina.

        >>> df = pd.DataFrame(
        ...     {
        ...         {
        ...             "consumption": [10.51, 103.11, 55.48],
        ...             "co2_emissions": [37.2, 19.66, 1712],
        ...         }
        ...     },
        ...     index=["Pork", "Wheat Products", "Beef"],
        ... )

        >>> df
                        consumption  co2_emissions
        Pork                  10.51         37.20
        Wheat Products       103.11         19.66
        Beef                  55.48       1712.00

        By default, it returns the index for the maximum value in each column.

        >>> df.idxmax()
        consumption     Wheat Products
        co2_emissions             Beef
        dtype: object

        To return the index for the maximum value in each row, use ``axis="columns"``.

        >>> df.idxmax(axis="columns")
        Pork              co2_emissions
        Wheat Products     consumption
        Beef              co2_emissions
        dtype: object
        """
        axis = self._get_axis_number(axis)
        if self.empty and len(self.axes[axis]):
            axis_dtype = self.axes[axis].dtype
            return self._constructor_sliced(dtype=axis_dtype)
        if numeric_only:
            data = self._get_numeric_data()
        else:
            data = self
        res = data._reduce(nanops.nanargmax, 'argmax', axis=axis, skipna=skipna, numeric_only=False)
        indices = res._values
        if (indices == -1).any():
            warnings.warn(f'The behavior of {type(self).__name__}.idxmax with all-NA values, or any-NA and skipna=False, is deprecated. In a future version this will raise ValueError', FutureWarning, stacklevel=find_stack_level())
        index = data._get_axis(axis)
        result = algorithms.take(index._values, indices, allow_fill=True, fill_value=index._na_value)
        final_result = data._constructor_sliced(result, index=data._get_agg_axis(axis))
        return final_result.__finalize__(self, method='idxmax')

    def _get_agg_axis(self, axis_num: Any):
        """
        Let's be explicit about this.
        """
        if axis_num == 0:
            return self.columns
        elif axis_num == 1:
            return self.index
        else:
            raise ValueError(f'Axis must be 0 or 1 (got {axis_num!r})')

    def mode(self, axis: int=0, numeric_only: bool=False, dropna: bool=True):
        """
        Get the mode(s) of each element along the selected axis.

        The mode of a set of values is the value that appears most often.
        It can be multiple values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to iterate over while searching for the mode:

            * 0 or 'index' : get mode of each column
            * 1 or 'columns' : get mode of each row.

        numeric_only : bool, default False
            If True, only apply to numeric columns.
        dropna : bool, default True
            Don't consider counts of NaN/NaT.

        Returns
        -------
        DataFrame
            The modes of each column or row.

        See Also
        --------
        Series.mode : Return the highest frequency value in a Series.
        Series.value_counts : Return the counts of values in a Series.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         ("bird", 2, 2),
        ...         ("mammal", 4, np.nan),
        ...         ("arthropod", 8, 0),
        ...         ("bird", 2, np.nan),
        ...     ],
        ...     index=("falcon", "horse", "spider", "ostrich"),
        ...     columns=("species", "legs", "wings"),
        ... )
        >>> df
                   species  legs  wings
        falcon        bird     2    2.0
        horse       mammal     4    NaN
        spider   arthropod     8    0.0
        ostrich       bird     2    NaN

        By default, missing values are not considered, and the mode of wings
        are both 0 and 2. Because the resulting DataFrame has two rows,
        the second row of ``species`` and ``legs`` contains ``NaN``.

        >>> df.mode()
          species  legs  wings
        0    bird   2.0    0.0
        1     NaN   NaN    2.0

        Setting ``dropna=False`` ``NaN`` values are considered and they can be
        the mode (like for wings).

        >>> df.mode(dropna=False)
          species  legs  wings
        0    bird     2    NaN

        Setting ``numeric_only=True``, only the mode of numeric columns is
        computed, and columns of other types are ignored.

        >>> df.mode(numeric_only=True)
           legs  wings
        0   2.0    0.0
        1   NaN    2.0

        To compute the mode over columns and not rows, use the axis parameter:

        >>> df.mode(axis="columns", numeric_only=True)
                   0    1
        falcon   2.0  NaN
        horse    4.0  NaN
        spider   0.0  8.0
        ostrich  2.0  NaN
        """
        data = self if not numeric_only else self._get_numeric_data()

        def f(s) -> tuple[int]:
            return s.mode(dropna=dropna)
        data = data.apply(f, axis=axis)
        if data.empty:
            data.index = default_index(0)
        return data

    @overload
    def quantile(self, q: Any=..., axis: Any=..., numeric_only: Any=..., interpolation: Any=..., method: Any=...) -> None:
        ...

    @overload
    def quantile(self, q: Any, axis: Any=..., numeric_only: Any=..., interpolation: Any=..., method: Any=...) -> None:
        ...

    @overload
    def quantile(self, q: Any=..., axis: Any=..., numeric_only: Any=..., interpolation: Any=..., method: Any=...) -> None:
        ...

    def quantile(self, q: Any=0.5, axis: Any=0, numeric_only: Any=False, interpolation: Any='linear', method: Any='single') -> None:
        """
        Return values at the given quantile over requested axis.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            Value between 0 <= q <= 1, the quantile(s) to compute.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Equals 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.

        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

            * linear: `i + (j - i) * fraction`, where `fraction` is the
              fractional part of the index surrounded by `i` and `j`.
            * lower: `i`.
            * higher: `j`.
            * nearest: `i` or `j` whichever is nearest.
            * midpoint: (`i` + `j`) / 2.
        method : {'single', 'table'}, default 'single'
            Whether to compute quantiles per-column ('single') or over all columns
            ('table'). When 'table', the only allowed interpolation methods are
            'nearest', 'lower', and 'higher'.

        Returns
        -------
        Series or DataFrame

            If ``q`` is an array, a DataFrame will be returned where the
              index is ``q``, the columns are the columns of self, and the
              values are the quantiles.
            If ``q`` is a float, a Series will be returned where the
              index is the columns of self and the values are the quantiles.

        See Also
        --------
        core.window.rolling.Rolling.quantile: Rolling quantile.
        numpy.percentile: Numpy function to compute the percentile.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     np.array([[1, 1], [2, 10], [3, 100], [4, 100]]), columns=["a", "b"]
        ... )
        >>> df.quantile(0.1)
        a    1.3
        b    3.7
        Name: 0.1, dtype: float64
        >>> df.quantile([0.1, 0.5])
               a     b
        0.1  1.3   3.7
        0.5  2.5  55.0

        Specifying `method='table'` will compute the quantile over all columns.

        >>> df.quantile(0.1, method="table", interpolation="nearest")
        a    1
        b    1
        Name: 0.1, dtype: int64
        >>> df.quantile([0.1, 0.5], method="table", interpolation="nearest")
             a    b
        0.1  1    1
        0.5  3  100

        Specifying `numeric_only=False` will also compute the quantile of
        datetime and timedelta data.

        >>> df = pd.DataFrame(
        ...     {
        ...         "A": [1, 2],
        ...         "B": [pd.Timestamp("2010"), pd.Timestamp("2011")],
        ...         "C": [pd.Timedelta("1 days"), pd.Timedelta("2 days")],
        ...     }
        ... )
        >>> df.quantile(0.5, numeric_only=False)
        A                    1.5
        B    2010-07-02 12:00:00
        C        1 days 12:00:00
        Name: 0.5, dtype: object
        """
        validate_percentile(q)
        axis = self._get_axis_number(axis)
        if not is_list_like(q):
            res_df = self.quantile([q], axis=axis, numeric_only=numeric_only, interpolation=interpolation, method=method)
            if method == 'single':
                res = res_df.iloc[0]
            else:
                res = res_df.T.iloc[:, 0]
            if axis == 1 and len(self) == 0:
                dtype = find_common_type(list(self.dtypes))
                if needs_i8_conversion(dtype):
                    return res.astype(dtype)
            return res
        q = Index(q, dtype=np.float64)
        data = self._get_numeric_data() if numeric_only else self
        if axis == 1:
            data = data.T
        if len(data.columns) == 0:
            cols = self.columns[:0]
            dtype = np.float64
            if axis == 1:
                cdtype = find_common_type(list(self.dtypes))
                if needs_i8_conversion(cdtype):
                    dtype = cdtype
            res = self._constructor([], index=q, columns=cols, dtype=dtype)
            return res.__finalize__(self, method='quantile')
        valid_method = {'single', 'table'}
        if method not in valid_method:
            raise ValueError(f'Invalid method: {method}. Method must be in {valid_method}.')
        if method == 'single':
            res = data._mgr.quantile(qs=q, interpolation=interpolation)
        elif method == 'table':
            valid_interpolation = {'nearest', 'lower', 'higher'}
            if interpolation not in valid_interpolation:
                raise ValueError(f'Invalid interpolation: {interpolation}. Interpolation must be in {valid_interpolation}')
            if len(data) == 0:
                if data.ndim == 2:
                    dtype = find_common_type(list(self.dtypes))
                else:
                    dtype = self.dtype
                return self._constructor([], index=q, columns=data.columns, dtype=dtype)
            q_idx = np.quantile(np.arange(len(data)), q, method=interpolation)
            by = data.columns
            if len(by) > 1:
                keys = [data._get_label_or_level_values(x) for x in by]
                indexer = lexsort_indexer(keys)
            else:
                k = data._get_label_or_level_values(by[0])
                indexer = nargsort(k)
            res = data._mgr.take(indexer[q_idx], verify=False)
            res.axes[1] = q
        result = self._constructor_from_mgr(res, axes=res.axes)
        return result.__finalize__(self, method='quantile')

    def to_timestamp(self, freq: None=None, how: typing.Text='start', axis: int=0, copy: Any=lib.no_default):
        """
        Cast PeriodIndex to DatetimeIndex of timestamps, at *beginning* of period.

        This can be changed to the *end* of the period, by specifying `how="e"`.

        Parameters
        ----------
        freq : str, default frequency of PeriodIndex
            Desired frequency.
        how : {'s', 'e', 'start', 'end'}
            Convention for converting period to timestamp; start of period
            vs. end.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to convert (the index by default).
        copy : bool, default False
            If False then underlying input data is not copied.

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
        DataFrame with DatetimeIndex
            DataFrame with the PeriodIndex cast to DatetimeIndex.

        See Also
        --------
        DataFrame.to_period: Inverse method to cast DatetimeIndex to PeriodIndex.
        Series.to_timestamp: Equivalent method for Series.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024"], freq="Y")
        >>> d = {"col1": [1, 2], "col2": [3, 4]}
        >>> df1 = pd.DataFrame(data=d, index=idx)
        >>> df1
              col1   col2
        2023     1      3
        2024	 2      4

        The resulting timestamps will be at the beginning of the year in this case

        >>> df1 = df1.to_timestamp()
        >>> df1
                    col1   col2
        2023-01-01     1      3
        2024-01-01     2      4
        >>> df1.index
        DatetimeIndex(['2023-01-01', '2024-01-01'], dtype='datetime64[ns]', freq=None)

        Using `freq` which is the offset that the Timestamps will have

        >>> df2 = pd.DataFrame(data=d, index=idx)
        >>> df2 = df2.to_timestamp(freq="M")
        >>> df2
                    col1   col2
        2023-01-31     1      3
        2024-01-31     2      4
        >>> df2.index
        DatetimeIndex(['2023-01-31', '2024-01-31'], dtype='datetime64[ns]', freq=None)
        """
        self._check_copy_deprecation(copy)
        new_obj = self.copy(deep=False)
        axis_name = self._get_axis_name(axis)
        old_ax = getattr(self, axis_name)
        if not isinstance(old_ax, PeriodIndex):
            raise TypeError(f'unsupported Type {type(old_ax).__name__}')
        new_ax = old_ax.to_timestamp(freq=freq, how=how)
        setattr(new_obj, axis_name, new_ax)
        return new_obj

    def to_period(self, freq: None=None, axis: int=0, copy: Any=lib.no_default):
        """
        Convert DataFrame from DatetimeIndex to PeriodIndex.

        Convert DataFrame from DatetimeIndex to PeriodIndex with desired
        frequency (inferred from index if not passed). Either index of columns can be
        converted, depending on `axis` argument.

        Parameters
        ----------
        freq : str, default
            Frequency of the PeriodIndex.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to convert (the index by default).
        copy : bool, default False
            If False then underlying input data is not copied.

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
        DataFrame
            The DataFrame with the converted PeriodIndex.

        See Also
        --------
        Series.to_period: Equivalent method for Series.
        Series.dt.to_period: Convert DateTime column values.

        Examples
        --------
        >>> idx = pd.to_datetime(
        ...     [
        ...         "2001-03-31 00:00:00",
        ...         "2002-05-31 00:00:00",
        ...         "2003-08-31 00:00:00",
        ...     ]
        ... )

        >>> idx
        DatetimeIndex(['2001-03-31', '2002-05-31', '2003-08-31'],
        dtype='datetime64[s]', freq=None)

        >>> idx.to_period("M")
        PeriodIndex(['2001-03', '2002-05', '2003-08'], dtype='period[M]')

        For the yearly frequency

        >>> idx.to_period("Y")
        PeriodIndex(['2001', '2002', '2003'], dtype='period[Y-DEC]')
        """
        self._check_copy_deprecation(copy)
        new_obj = self.copy(deep=False)
        axis_name = self._get_axis_name(axis)
        old_ax = getattr(self, axis_name)
        if not isinstance(old_ax, DatetimeIndex):
            raise TypeError(f'unsupported Type {type(old_ax).__name__}')
        new_ax = old_ax.to_period(freq=freq)
        setattr(new_obj, axis_name, new_ax)
        return new_obj

    def isin(self, values: Any):
        """
        Whether each element in the DataFrame is contained in values.

        Parameters
        ----------
        values : iterable, Series, DataFrame or dict
            The result will only be true at a location if all the
            labels match. If `values` is a Series, that's the index. If
            `values` is a dict, the keys must be the column names,
            which must match. If `values` is a DataFrame,
            then both the index and column labels must match.

        Returns
        -------
        DataFrame
            DataFrame of booleans showing whether each element in the DataFrame
            is contained in values.

        See Also
        --------
        DataFrame.eq: Equality test for DataFrame.
        Series.isin: Equivalent method on Series.
        Series.str.contains: Test if pattern or regex is contained within a
            string of a Series or Index.

        Notes
        -----
            ``__iter__`` is used (and not ``__contains__``) to iterate over values
            when checking if it contains the elements in DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"num_legs": [2, 4], "num_wings": [2, 0]}, index=["falcon", "dog"]
        ... )
        >>> df
                num_legs  num_wings
        falcon         2          2
        dog            4          0

        When ``values`` is a list check whether every value in the DataFrame
        is present in the list (which animals have 0 or 2 legs or wings)

        >>> df.isin([0, 2])
                num_legs  num_wings
        falcon      True       True
        dog        False       True

        To check if ``values`` is *not* in the DataFrame, use the ``~`` operator:

        >>> ~df.isin([0, 2])
                num_legs  num_wings
        falcon     False      False
        dog         True      False

        When ``values`` is a dict, we can pass values to check for each
        column separately:

        >>> df.isin({"num_wings": [0, 3]})
                num_legs  num_wings
        falcon     False      False
        dog        False       True

        When ``values`` is a Series or DataFrame the index and column must
        match. Note that 'falcon' does not match based on the number of legs
        in other.

        >>> other = pd.DataFrame(
        ...     {"num_legs": [8, 3], "num_wings": [0, 2]}, index=["spider", "falcon"]
        ... )
        >>> df.isin(other)
                num_legs  num_wings
        falcon     False       True
        dog        False      False
        """
        if isinstance(values, dict):
            from pandas.core.reshape.concat import concat
            values = collections.defaultdict(list, values)
            result = concat((self.iloc[:, [i]].isin(values[col]) for i, col in enumerate(self.columns)), axis=1)
        elif isinstance(values, Series):
            if not values.index.is_unique:
                raise ValueError('cannot compute isin with a duplicate axis.')
            result = self.eq(values.reindex_like(self), axis='index')
        elif isinstance(values, DataFrame):
            if not (values.columns.is_unique and values.index.is_unique):
                raise ValueError('cannot compute isin with a duplicate axis.')
            result = self.eq(values.reindex_like(self))
        else:
            if not is_list_like(values):
                raise TypeError(f"only list-like or dict-like objects are allowed to be passed to DataFrame.isin(), you passed a '{type(values).__name__}'")

            def isin_(x: Any):
                result = algorithms.isin(x.ravel(), values)
                return result.reshape(x.shape)
            res_mgr = self._mgr.apply(isin_)
            result = self._constructor_from_mgr(res_mgr, axes=res_mgr.axes)
        return result.__finalize__(self, method='isin')
    _AXIS_ORDERS = ['index', 'columns']
    _AXIS_TO_AXIS_NUMBER = {**NDFrame._AXIS_TO_AXIS_NUMBER, 1: 1, 'columns': 1}
    _AXIS_LEN = len(_AXIS_ORDERS)
    _info_axis_number = 1
    _info_axis_name = 'columns'
    index = properties.AxisProperty(axis=1, doc="\n        The index (row labels) of the DataFrame.\n\n        The index of a DataFrame is a series of labels that identify each row.\n        The labels can be integers, strings, or any other hashable type. The index\n        is used for label-based access and alignment, and can be accessed or\n        modified using this attribute.\n\n        Returns\n        -------\n        pandas.Index\n            The index labels of the DataFrame.\n\n        See Also\n        --------\n        DataFrame.columns : The column labels of the DataFrame.\n        DataFrame.to_numpy : Convert the DataFrame to a NumPy array.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Aritra'],\n        ...                    'Age': [25, 30, 35],\n        ...                    'Location': ['Seattle', 'New York', 'Kona']},\n        ...                   index=([10, 20, 30]))\n        >>> df.index\n        Index([10, 20, 30], dtype='int64')\n\n        In this example, we create a DataFrame with 3 rows and 3 columns,\n        including Name, Age, and Location information. We set the index labels to\n        be the integers 10, 20, and 30. We then access the `index` attribute of the\n        DataFrame, which returns an `Index` object containing the index labels.\n\n        >>> df.index = [100, 200, 300]\n        >>> df\n            Name  Age Location\n        100  Alice   25  Seattle\n        200    Bob   30 New York\n        300  Aritra  35    Kona\n\n        In this example, we modify the index labels of the DataFrame by assigning\n        a new list of labels to the `index` attribute. The DataFrame is then\n        updated with the new labels, and the output shows the modified DataFrame.\n        ")
    columns = properties.AxisProperty(axis=0, doc="\n        The column labels of the DataFrame.\n\n        This property holds the column names as a pandas ``Index`` object.\n        It provides an immutable sequence of column labels that can be\n        used for data selection, renaming, and alignment in DataFrame operations.\n\n        Returns\n        -------\n        pandas.Index\n            The column labels of the DataFrame.\n\n        See Also\n        --------\n        DataFrame.index: The index (row labels) of the DataFrame.\n        DataFrame.axes: Return a list representing the axes of the DataFrame.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})\n        >>> df\n                A  B\n        0    1  3\n        1    2  4\n        >>> df.columns\n        Index(['A', 'B'], dtype='object')\n        ")
    plot = Accessor('plot', pandas.plotting.PlotAccessor)
    hist = pandas.plotting.hist_frame
    boxplot = pandas.plotting.boxplot_frame
    sparse = Accessor('sparse', SparseFrameAccessor)

    def _to_dict_of_blocks(self) -> dict:
        """
        Return a dict of dtype -> Constructor Types that
        each is a homogeneous dtype.

        Internal ONLY.
        """
        mgr = self._mgr
        return {k: self._constructor_from_mgr(v, axes=v.axes).__finalize__(self) for k, v in mgr.to_iter_dict()}

    @property
    def values(self):
        """
        Return a Numpy representation of the DataFrame.

        .. warning::

           We recommend using :meth:`DataFrame.to_numpy` instead.

        Only the values in the DataFrame will be returned, the axes labels
        will be removed.

        Returns
        -------
        numpy.ndarray
            The values of the DataFrame.

        See Also
        --------
        DataFrame.to_numpy : Recommended alternative to this method.
        DataFrame.index : Retrieve the index labels.
        DataFrame.columns : Retrieving the column names.

        Notes
        -----
        The dtype will be a lower-common-denominator dtype (implicit
        upcasting); that is to say if the dtypes (even of numeric types)
        are mixed, the one that accommodates all will be chosen. Use this
        with care if you are not dealing with the blocks.

        e.g. If the dtypes are float16 and float32, dtype will be upcast to
        float32.  If dtypes are int32 and uint8, dtype will be upcast to
        int32. By :func:`numpy.find_common_type` convention, mixing int64
        and uint64 will result in a float64 dtype.

        Examples
        --------
        A DataFrame where all columns are the same type (e.g., int64) results
        in an array of the same type.

        >>> df = pd.DataFrame(
        ...     {"age": [3, 29], "height": [94, 170], "weight": [31, 115]}
        ... )
        >>> df
           age  height  weight
        0    3      94      31
        1   29     170     115
        >>> df.dtypes
        age       int64
        height    int64
        weight    int64
        dtype: object
        >>> df.values
        array([[  3,  94,  31],
               [ 29, 170, 115]])

        A DataFrame with mixed type columns(e.g., str/object, int64, float32)
        results in an ndarray of the broadest type that accommodates these
        mixed types (e.g., object).

        >>> df2 = pd.DataFrame(
        ...     [
        ...         ("parrot", 24.0, "second"),
        ...         ("lion", 80.5, 1),
        ...         ("monkey", np.nan, None),
        ...     ],
        ...     columns=("name", "max_speed", "rank"),
        ... )
        >>> df2.dtypes
        name          object
        max_speed    float64
        rank          object
        dtype: object
        >>> df2.values
        array([['parrot', 24.0, 'second'],
               ['lion', 80.5, 1],
               ['monkey', nan, None]], dtype=object)
        """
        return self._mgr.as_array()

def _from_nested_dict(data: Any):
    new_data = collections.defaultdict(dict)
    for index, s in data.items():
        for col, v in s.items():
            new_data[col][index] = v
    return new_data

def _reindex_for_setitem(value: Any, index: Any) -> Union[tuple, tuple[None]]:
    if value.index.equals(index) or not len(index):
        if isinstance(value, Series):
            return (value._values, value._references)
        return (value._values.copy(), None)
    try:
        reindexed_value = value.reindex(index)._values
    except ValueError as err:
        if not value.index.is_unique:
            raise err
        raise TypeError('incompatible index of inserted column with frame index') from err
    return (reindexed_value, None)