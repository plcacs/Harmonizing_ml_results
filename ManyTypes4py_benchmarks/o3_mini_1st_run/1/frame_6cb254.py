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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
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
_merge_doc = '\nMerge DataFrame or named Series objects with a database-style join.\n\nA named Series object is treated as a DataFrame with a single named column.\n\nThe join is done on columns or indexes. If joining columns on\ncolumns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes\non indexes or indexes on a column or columns, the index will be passed on.\nWhen performing a cross merge, no column specifications to merge on are\nallowed.\n\n.. warning::\n\n    If both key columns contain rows where the key is a null value, those\n    rows will be matched against each other. This is different from usual SQL\n    join behaviour and can lead to unexpected results.\n\nParameters\n----------%s\nright : DataFrame or named Series\n    Object to merge with.\nhow : {\'left\', \'right\', \'outer\', \'inner\', \'cross\', \'left_anti\', \'right_anti\'},\n    default \'inner\'\n    Type of merge to be performed.\n\n    * left: use only keys from left frame, similar to a SQL left outer join;\n      preserve key order.\n    * right: use only keys from right frame, similar to a SQL right outer join;\n      preserve key order.\n    * outer: use union of keys from both frames, similar to a SQL full outer\n      join; sort keys lexicographically.\n    * inner: use intersection of keys from both frames, similar to a SQL inner\n      join; preserve the order of the left keys.\n    * cross: creates the cartesian product from both frames, preserves the order\n      of the left keys.\n    * left_anti: use only keys from left frame that are not in right frame, similar\n      to SQL left anti join; preserve key order.\n    * right_anti: use only keys from right frame that are not in left frame, similar\n      to SQL right anti join; preserve key order.\non : label or list\n    Column or index level names to join on. These must be found in both\n    DataFrames. If `on` is None and not merging on indexes then this defaults\n    to the intersection of the columns in both DataFrames.\nleft_on : label or list, or array-like\n    Column or index level names to join on in the left DataFrame. Can also\n    be an array or list of arrays of the length of the left DataFrame.\n    These arrays are treated as if they are columns.\nright_on : label or list, or array-like\n    Column or index level names to join on in the right DataFrame. Can also\n    be an array or list of arrays of the length of the right DataFrame.\n    These arrays are treated as if they are columns.\nleft_index : bool, default False\n    Use the index from the left DataFrame as the join key(s). If it is a\n    MultiIndex, the number of keys in the other DataFrame (either the index\n    or a number of columns) must match the number of levels.\nright_index : bool, default False\n    Use the index from the right DataFrame as the join key. Same caveats as\n    left_index.\nsort : bool, default False\n    Sort the join keys lexicographically in the result DataFrame. If False,\n    the order of the join keys depends on the join type (how keyword).\nsuffixes : list-like, default is ("_x", "_y")\n    A length-2 sequence where each element is optionally a string\n    indicating the suffix to add to overlapping column names in\n    `left` and `right` respectively. Pass a value of `None` instead\n    of a string to indicate that the column name from `left` or\n    `right` should be left as-is, with no suffix. At least one of the\n    values must not be None.\ncopy : bool, default False\n    If False, avoid copy if possible.\n\n    .. note::\n        The `copy` keyword will change behavior in pandas 3.0.\n        `Copy-on-Write\n        <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__\n        will be enabled by default, which means that all methods with a\n        `copy` keyword will use a lazy copy mechanism to defer the copy and\n        ignore the `copy` keyword. The `copy` keyword will be removed in a\n        future version of pandas.\n\n    .. deprecated:: 3.0.0\nindicator : bool or str, default False\n    If True, adds a column to the output DataFrame called "_merge" with\n    information on the source of each row. The column can be given a different\n    name by providing a string argument. The column will have a Categorical\n    type with the value of "left_only" for observations whose merge key only\n    appears in the left DataFrame, "right_only" for observations\n    whose merge key only appears in the right DataFrame, and "both"\n    if the observation\'s merge key is found in both DataFrames.\n\nvalidate : str, optional\n    If specified, checks if merge is of specified type.\n\n    * "one_to_one" or "1:1": check if merge keys are unique in both\n      left and right datasets.\n    * "one_to_many" or "1:m": check if merge keys are unique in left\n      dataset.\n    * "many_to_one" or "m:1": check if merge keys are unique in right\n      dataset.\n    * "many_to_many" or "m:m": allowed, but does not result in checks.\n\nReturns\n-------\nDataFrame\n    A DataFrame of the two merged objects.\n\nSee Also\n--------\nmerge_ordered : Merge with optional filling/interpolation.\nmerge_asof : Merge on nearest keys.\nDataFrame.join : Similar method using indices.\n\nExamples\n--------\n>>> df1 = pd.DataFrame({\'lkey\': [\'foo\', \'bar\', \'baz\', \'foo\'],\n...                     \'value\': [1, 2, 3, 5]})\n>>> df2 = pd.DataFrame({\'rkey\': [\'foo\', \'bar\', \'baz\', \'foo\'],\n...                     \'value\': [5, 6, 7, 8]})\n>>> df1\n    lkey value\n0   foo      1\n1   bar      2\n2   baz      3\n3   foo      5\n>>> df2\n    rkey value\n0   foo      5\n1   bar      6\n2   baz      7\n3   foo      8\n\nMerge df1 and df2 on the lkey and rkey columns. The value columns have\nthe default suffixes, _x and _y, appended.\n\n>>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\')\n  lkey  value_x rkey  value_y\n0  foo        1  foo        5\n1  foo        1  foo        8\n2  bar        2  bar        6\n3  baz        3  baz        7\n4  foo        5  foo        5\n5  foo        5  foo        8\n\nMerge DataFrames df1 and df2 with specified left and right suffixes\nappended to any overlapping columns.\n\n>>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\',\n...           suffixes=(\'_left\', \'_right\'))\n  lkey  value_left rkey  value_right\n0  foo           1  foo            5\n1  foo           1  foo            8\n2  bar           2  bar            6\n3  baz           3  baz            7\n4  foo           5  foo            5\n5  foo           5  foo            8\n\nMerge DataFrames df1 and df2, but raise an exception if the DataFrames have\nany overlapping columns.\n\n>>> df1.merge(df2, left_on=\'lkey\', right_on=\'rkey\', suffixes=(False, False))\nTraceback (most recent call last):\n...\nValueError: columns overlap but no suffix specified:\n    Index([\'value\'], dtype=\'object\')\n\n>>> df1 = pd.DataFrame({\'a\': [\'foo\', \'bar\'], \'b\': [1, 2]})\n>>> df2 = pd.DataFrame({\'a\': [\'foo\', \'baz\'], \'c\': [3, 4]})\n>>> df1\n      a  b\n0   foo  1\n1   bar  2\n>>> df2\n      a  c\n0   foo  3\n1   baz  4\n\n>>> df1.merge(df2, how=\'inner\', on=\'a\')\n      a  b  c\n0   foo  1  3\n\n>>> df1.merge(df2, how=\'left\', on=\'a\')\n      a  b  c\n0   foo  1  3.0\n1   bar  2  NaN\n\n>>> df1 = pd.DataFrame({\'left\': [\'foo\', \'bar\']})\n>>> df2 = pd.DataFrame({\'right\': [7, 8]})\n>>> df1\n    left\n0   foo\n1   bar\n>>> df2\n    right\n0   7\n1   8\n\n>>> df1.merge(df2, how=\'cross\')\n   left  right\n0   foo      7\n1   foo      8\n2   bar      7\n3   bar      8\n'
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
    def _constructor(self) -> type[DataFrame]:
        return DataFrame

    def _constructor_from_mgr(self, mgr: BlockManager, axes: dict[str, Index]) -> DataFrame:
        df = DataFrame._from_mgr(mgr, axes=axes)
        if type(self) is DataFrame:
            return df
        elif type(self).__name__ == 'GeoDataFrame':
            return self._constructor(mgr)
        return self._constructor(df)

    _constructor_sliced = Series

    def _constructor_sliced_from_mgr(self, mgr: BlockManager, axes: dict[str, Index]) -> Series:
        ser = Series._from_mgr(mgr, axes)
        ser._name = None
        if type(self) is DataFrame:
            return ser
        return self._constructor_sliced(ser)

    def __init__(self, data: Any = None, index: Any = None, columns: Any = None, dtype: Any = None, copy: Any = None) -> None:
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

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> Any:
        from pandas.core.interchange.dataframe import PandasDataFrameXchg
        return PandasDataFrameXchg(self, allow_copy=allow_copy)

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        pa = import_optional_dependency('pyarrow', min_version='14.0.0')
        if requested_schema is not None:
            requested_schema = pa.Schema._import_from_c_capsule(requested_schema)
        table = pa.Table.from_pandas(self, schema=requested_schema)
        return table.__arrow_c_stream__()

    @property
    def axes(self) -> List[Index]:
        return [self.index, self.columns]

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.index), len(self.columns))

    @property
    def _is_homogeneous_type(self) -> bool:
        return len({block.values.dtype for block in self._mgr.blocks}) <= 1

    @property
    def _can_fast_transpose(self) -> bool:
        blocks = self._mgr.blocks
        if len(blocks) != 1:
            return False
        dtype = blocks[0].dtype
        return not is_1d_only_ea_dtype(dtype)

    @property
    def _values(self) -> Any:
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
        max_rows = get_option('display.max_rows')
        return len(self) <= max_rows

    def _repr_fits_horizontal_(self) -> bool:
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
        info_repr_option = get_option('display.large_repr') == 'info'
        return info_repr_option and (not (self._repr_fits_horizontal_() and self._repr_fits_vertical_()))

    def __repr__(self) -> str:
        if self._info_repr():
            buf = StringIO()
            self.info(buf=buf)
            return buf.getvalue()
        repr_params = fmt.get_dataframe_repr_params()
        return self.to_string(**repr_params)

    def _repr_html_(self) -> Optional[str]:
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
    def to_string(self, buf: Any = ..., *, columns: Any = ..., col_space: Any = ..., header: Any = ..., index: Any = ..., na_rep: Any = ..., formatters: Any = ..., float_format: Any = ..., sparsify: Any = ..., index_names: Any = ..., justify: Any = ..., max_rows: Any = ..., max_cols: Any = ..., show_dimensions: Any = ..., decimal: Any = ..., line_width: Any = ..., min_rows: Any = ..., max_colwidth: Any = ..., encoding: Any = ...) -> Any:
        ...

    @overload
    def to_string(self, buf: Any, *, columns: Any = ..., col_space: Any = ..., header: Any = ..., index: Any = ..., na_rep: Any = ..., formatters: Any = ..., float_format: Any = ..., sparsify: Any = ..., index_names: Any = ..., justify: Any = ..., max_rows: Any = ..., max_cols: Any = ..., show_dimensions: Any = ..., decimal: Any = ..., line_width: Any = ..., min_rows: Any = ..., max_colwidth: Any = ..., encoding: Any = ...) -> Any:
        ...

    @Substitution(header_type='bool or list of str', header='Write out the column names. If a list of columns is given, it is assumed to be aliases for the column names', col_space_type='int, list or dict of int', col_space='The minimum width of each column. If a list of ints is given every integers corresponds with one column. If a dict is given, the key references the column, while the value defines the space to use.')
    @Substitution(shared_params=fmt.common_docstring, returns=fmt.return_docstring)
    def to_string(self, buf: Optional[Any] = None, *, columns: Any = None, col_space: Any = None, header: bool = True, index: bool = True, na_rep: str = 'NaN', formatters: Any = None, float_format: Any = None, sparsify: Any = None, index_names: bool = True, justify: Any = None, max_rows: Any = None, max_cols: Any = None, show_dimensions: bool = False, decimal: str = '.', line_width: Any = None, min_rows: Any = None, max_colwidth: Any = None, encoding: Any = None) -> Any:
        from pandas import option_context
        with option_context('display.max_colwidth', max_colwidth):
            formatter = fmt.DataFrameFormatter(self, columns=columns, col_space=col_space, na_rep=na_rep, formatters=formatters, float_format=float_format, sparsify=sparsify, justify=justify, index_names=index_names, header=header, index=index, min_rows=min_rows, max_rows=max_rows, max_cols=max_cols, show_dimensions=show_dimensions, decimal=decimal)
            return fmt.DataFrameRenderer(formatter).to_string(buf=buf, encoding=encoding, line_width=line_width)

    def _get_values_for_csv(self, *, float_format: Any, date_format: Any, decimal: Any, na_rep: Any, quoting: Any) -> DataFrame:
        mgr = self._mgr.get_values_for_csv(float_format=float_format, date_format=date_format, decimal=decimal, na_rep=na_rep, quoting=quoting)
        return self._constructor_from_mgr(mgr, axes=mgr.axes)

    @property
    def style(self) -> Styler:
        has_jinja2 = import_optional_dependency('jinja2', errors='ignore')
        if not has_jinja2:
            raise AttributeError("The '.style' accessor requires jinja2")
        from pandas.io.formats.style import Styler
        return Styler(self)
    _shared_docs['items'] = "\n        Iterate over (column name, Series) pairs.\n\n        Iterates over the DataFrame columns, returning a tuple with\n        the column name and the content as a Series.\n\n        Yields\n        ------\n        label : object\n            The column names for the DataFrame being iterated over.\n        content : Series\n            The column entries belonging to each label, as a Series.\n\n        See Also\n        --------\n        DataFrame.iterrows : Iterate over DataFrame rows as\n            (index, Series) pairs.\n        DataFrame.itertuples : Iterate over DataFrame rows as namedtuples\n            of the values.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'species': ['bear', 'bear', 'marsupial'],\n        ...                   'population': [1864, 22000, 80000]},\n        ...                   index=['panda', 'polar', 'koala'])\n        >>> df\n                species   population\n        panda   bear      1864\n        polar   bear      22000\n        koala   marsupial 80000\n        >>> for label, content in df.items():\n        ...     print(f'label: {label}')\n        ...     print(f'content: {content}', sep='\\n')\n        ...\n        label: species\n        content:\n        panda         bear\n        polar         bear\n        koala    marsupial\n        Name: species, dtype: object\n        label: population\n        content:\n        panda     1864\n        polar    22000\n        koala    80000\n        Name: population, dtype: int64\n        "

    @Appender(_shared_docs['items'])
    def items(self) -> Iterator[Tuple[Any, Series]]:
        for i, k in enumerate(self.columns):
            yield (k, self._ixs(i, axis=1))

    def iterrows(self) -> Iterator[Tuple[Any, Series]]:
        columns = self.columns
        klass = self._constructor_sliced
        for k, v in zip(self.index, self.values):
            s = klass(v, index=columns, name=k).__finalize__(self)
            if self._mgr.is_single_block:
                s._mgr.add_references(self._mgr)
            yield (k, s)

    def itertuples(self, index: bool = True, name: Union[str, None] = 'Pandas') -> Iterator[Any]:
        arrays = []
        fields: List[Any] = list(self.columns)
        if index:
            arrays.append(self.index)
            fields.insert(0, 'Index')
        arrays.extend((self.iloc[:, k] for k in range(len(self.columns))))
        if name is not None:
            itertuple = collections.namedtuple(name, fields, rename=True)
            return map(itertuple._make, zip(*arrays))
        return zip(*arrays)

    def __len__(self) -> int:
        return len(self.index)

    @overload
    def dot(self, other: Any) -> Union[Series, DataFrame]:
        ...

    @overload
    def dot(self, other: Any) -> Union[Series, DataFrame]:
        ...

    def dot(self, other: Any) -> Union[Series, DataFrame]:
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
    def __matmul__(self, other: Any) -> Union[Series, DataFrame]:
        ...

    @overload
    def __matmul__(self, other: Any) -> Union[Series, DataFrame]:
        ...

    def __matmul__(self, other: Any) -> Union[Series, DataFrame]:
        return self.dot(other)

    def __rmatmul__(self, other: Any) -> Union[Series, DataFrame]:
        try:
            return self.T.dot(np.transpose(other)).T
        except ValueError as err:
            if 'shape mismatch' not in str(err):
                raise
            msg = f'shapes {np.shape(other)} and {self.shape} not aligned'
            raise ValueError(msg) from err

    @classmethod
    def from_dict(cls, data: dict[Any, Any], orient: str = 'columns', dtype: Any = None, columns: Any = None) -> DataFrame:
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
            def create_index(indexlist: Sequence[Any], namelist: Sequence[Any]) -> Index:
                if len(namelist) > 1:
                    index = MultiIndex.from_tuples(indexlist, names=namelist)
                else:
                    index = Index(indexlist, name=namelist[0])
                return index
            index = create_index(data['index'], data['index_names'])
            columns = create_index(data['columns'], data['column_names'])
            return cls(realdata, index=index, columns=columns, dtype=dtype)

    def to_numpy(self, dtype: Optional[Union[str, np.dtype]] = None, copy: bool = False, na_value: Any = lib.no_default) -> np.ndarray:
        if dtype is not None:
            dtype = np.dtype(dtype)
        result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
        if result.dtype is not dtype:
            result = np.asarray(result, dtype=dtype)
        return result

    @overload
    def to_dict(self, orient: Any = ..., *, into: type[MutableMapping] = dict, index: Any = ...) -> Union[dict, list]:
        ...

    @overload
    def to_dict(self, orient: Any, *, into: type[MutableMapping] = dict, index: Any = ...) -> Union[dict, list]:
        ...

    @overload
    def to_dict(self, orient: Any = ..., *, into: Any, index: Any = ...) -> Union[dict, list]:
        ...

    @overload
    def to_dict(self, orient: Any, *, into: Any, index: Any = ...) -> Union[dict, list]:
        ...

    def to_dict(self, orient: str = 'dict', *, into: type[MutableMapping] = dict, index: bool = True) -> Union[dict, list, MutableMapping]:
        from pandas.core.methods.to_dict import to_dict
        return to_dict(self, orient, into=into, index=index)

    @classmethod
    def from_records(cls, data: Any, index: Any = None, exclude: Optional[Sequence[Any]] = None, columns: Any = None, coerce_float: bool = False, nrows: Optional[int] = None) -> DataFrame:
        if isinstance(data, DataFrame):
            raise TypeError('Passing a DataFrame to DataFrame.from_records is not supported. Use set_index and/or drop to modify the DataFrame instead.')
        result_index = None
        if columns is not None:
            columns = ensure_index(columns)
        def maybe_reorder(arrays: List[Any], arr_columns: Index, columns: Index, index: Any) -> Tuple[List[Any], Index, Optional[Index]]:
            if len(arrays):
                length = len(arrays[0])
            else:
                length = 0
            result_index: Optional[Index] = None
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

    def to_records(self, index: bool = True, column_dtypes: Optional[Any] = None, index_dtypes: Optional[Any] = None) -> np.rec.recarray:
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
        formats: List[Any] = []
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
    def _from_arrays(cls, arrays: Sequence[Any], columns: Any, index: Any, dtype: Any = None, verify_integrity: bool = True) -> DataFrame:
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        columns = ensure_index(columns)
        if len(columns) != len(arrays):
            raise ValueError('len(columns) must match len(arrays)')
        mgr = arrays_to_mgr(arrays, columns, index, dtype=dtype, verify_integrity=verify_integrity)
        return cls._from_mgr(mgr, axes=mgr.axes)

    @doc(storage_options=_shared_docs['storage_options'], compression_options=_shared_docs['compression_options'] % 'path')
    def to_stata(self, path: Union[str, Any], *, convert_dates: Any = None, write_index: bool = True, byteorder: Any = None, time_stamp: Any = None, data_label: Optional[str] = None, variable_labels: Any = None, version: Optional[Union[int, None]] = 114, convert_strl: Any = None, compression: str = 'infer', storage_options: Any = None, value_labels: Any = None) -> None:
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
        kwargs: Dict[str, Any] = {}
        if version is None or version >= 117:
            kwargs['convert_strl'] = convert_strl
        if version is None or version >= 118:
            kwargs['version'] = version
        writer = statawriter(path, self, convert_dates=convert_dates, byteorder=byteorder, time_stamp=time_stamp, data_label=data_label, write_index=write_index, variable_labels=variable_labels, compression=compression, storage_options=storage_options, value_labels=value_labels, **kwargs)
        writer.write_file()

    def to_feather(self, path: Union[str, Any], **kwargs: Any) -> None:
        from pandas.io.feather_format import to_feather
        to_feather(self, path, **kwargs)

    @overload
    def to_markdown(self, buf: Any = ..., *, mode: str = ..., index: bool = ..., storage_options: Any = ..., **kwargs: Any) -> Union[str, None]:
        ...

    @overload
    def to_markdown(self, buf: Any, *, mode: str = ..., index: bool = ..., storage_options: Any = ..., **kwargs: Any) -> Union[str, None]:
        ...

    @overload
    def to_markdown(self, buf: Any, *, mode: str = ..., index: bool = ..., storage_options: Any = ..., **kwargs: Any) -> Union[str, None]:
        ...

    def to_markdown(self, buf: Optional[Union[str, Any]] = None, *, mode: str = 'wt', index: bool = True, storage_options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Union[str, None]:
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
    def to_parquet(self, path: Any = ..., *, engine: str = ..., compression: Any = ..., index: Optional[bool] = ..., partition_cols: Optional[List[Any]] = ..., storage_options: Optional[Dict[str, Any]] = ..., **kwargs: Any) -> Union[bytes, None]:
        ...

    @overload
    def to_parquet(self, path: Any, *, engine: str = ..., compression: Any = ..., index: Optional[bool] = ..., partition_cols: Optional[List[Any]] = ..., storage_options: Optional[Dict[str, Any]] = ..., **kwargs: Any) -> Union[bytes, None]:
        ...

    @doc(storage_options=_shared_docs['storage_options'])
    def to_parquet(self, path: Optional[Any] = None, *, engine: str = 'auto', compression: Optional[str] = 'snappy', index: Optional[bool] = None, partition_cols: Optional[List[Any]] = None, storage_options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Union[bytes, None]:
        from pandas.io.parquet import to_parquet
        return to_parquet(self, path, engine, compression=compression, index=index, partition_cols=partition_cols, storage_options=storage_options, **kwargs)

    @overload
    def to_orc(self, path: Any = ..., *, engine: str = ..., index: Optional[bool] = ..., engine_kwargs: Optional[Mapping[str, Any]] = ...) -> Union[bytes, None]:
        ...

    @overload
    def to_orc(self, path: Any, *, engine: str = ..., index: Optional[bool] = ..., engine_kwargs: Optional[Mapping[str, Any]] = ...) -> Union[bytes, None]:
        ...

    def to_orc(self, path: Optional[Any] = None, *, engine: str = 'pyarrow', index: Optional[bool] = None, engine_kwargs: Optional[Mapping[str, Any]] = None) -> Union[bytes, None]:
        from pandas.io.orc import to_orc
        return to_orc(self, path, engine=engine, index=index, engine_kwargs=engine_kwargs)

    @overload
    def to_html(self, buf: Any, *, columns: Any = ..., col_space: Any = ..., header: Any = ..., index: Any = ..., na_rep: Any = ..., formatters: Any = ..., float_format: Any = ..., sparsify: Any = ..., index_names: Any = ..., justify: Any = ..., max_rows: Any = ..., max_cols: Any = ..., show_dimensions: Any = ..., decimal: Any = ..., bold_rows: Any = ..., classes: Any = ..., escape: Any = ..., notebook: Any = ..., border: Any = ..., table_id: Any = ..., render_links: Any = ..., encoding: Any = ...) -> Any:
        ...

    @overload
    def to_html(self, buf: Optional[Any] = ..., *, columns: Any = ..., col_space: Any = ..., header: Any = ..., index: Any = ..., na_rep: Any = ..., formatters: Any = ..., float_format: Any = ..., sparsify: Any = ..., index_names: Any = ..., justify: Any = ..., max_rows: Any = ..., max_cols: Any = ..., show_dimensions: Any = ..., decimal: Any = ..., bold_rows: Any = ..., classes: Any = ..., escape: Any = ..., notebook: Any = ..., border: Any = ..., table_id: Any = ..., render_links: Any = ..., encoding: Any = ...) -> Any:
        ...

    @Substitution(header_type='bool', header='Whether to print column labels, default True', col_space_type='str or int, list or dict of int or str', col_space='The minimum width of each column in CSS length units.  An int is assumed to be px units.')
    @Substitution(shared_params=fmt.common_docstring, returns=fmt.return_docstring)
    def to_html(self, buf: Optional[Any] = None, *, columns: Any = None, col_space: Any = None, header: bool = True, index: bool = True, na_rep: str = 'NaN', formatters: Any = None, float_format: Any = None, sparsify: Any = None, index_names: bool = True, justify: Any = None, max_rows: Any = None, max_cols: Any = None, show_dimensions: bool = False, decimal: str = '.', bold_rows: bool = True, classes: Any = None, escape: bool = True, notebook: bool = False, border: Any = None, table_id: Any = None, render_links: bool = False, encoding: Optional[str] = None) -> Any:
        formatter = fmt.DataFrameFormatter(self, columns=columns, col_space=col_space, na_rep=na_rep, header=header, index=index, formatters=formatters, float_format=float_format, bold_rows=bold_rows, sparsify=sparsify, justify=justify, index_names=index_names, escape=escape, decimal=decimal, max_rows=max_rows, max_cols=max_cols, show_dimensions=show_dimensions)
        return fmt.DataFrameRenderer(formatter).to_html(buf=buf, classes=classes, notebook=notebook, border=border, encoding=encoding, table_id=table_id, render_links=render_links)

    @overload
    def to_xml(self, path_or_buffer: Any = ..., *, index: bool = ..., root_name: str = ..., row_name: str = ..., na_rep: Any = ..., attr_cols: Any = ..., elem_cols: Any = ..., namespaces: Any = ..., prefix: Any = ..., encoding: str = ..., xml_declaration: bool = ..., pretty_print: bool = ..., parser: Any = ..., stylesheet: Any = ..., compression: Any = ..., storage_options: Any = ...) -> Any:
        ...

    @overload
    def to_xml(self, path_or_buffer: Any, *, index: bool = ..., root_name: str = ..., row_name: str = ..., na_rep: Any = ..., attr_cols: Any = ..., elem_cols: Any = ..., namespaces: Any = ..., prefix: Any = ..., encoding: str = ..., xml_declaration: bool = ..., pretty_print: bool = ..., parser: Any = ..., stylesheet: Any = ..., compression: Any = ..., storage_options: Any = ...) -> Any:
        ...

    @doc(storage_options=_shared_docs['storage_options'], compression_options=_shared_docs['compression_options'] % 'path_or_buffer')
    def to_xml(self, path_or_buffer: Optional[Any] = None, *, index: bool = True, root_name: str = 'data', row_name: str = 'row', na_rep: Optional[str] = None, attr_cols: Optional[Sequence[Any]] = None, elem_cols: Optional[Sequence[Any]] = None, namespaces: Optional[Mapping[str, Any]] = None, prefix: Optional[str] = None, encoding: str = 'utf-8', xml_declaration: bool = True, pretty_print: bool = True, parser: str = 'lxml', stylesheet: Optional[Any] = None, compression: str = 'infer', storage_options: Optional[Dict[str, Any]] = None) -> Any:
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
    def info(self, verbose: Optional[Any] = None, buf: Optional[StringIO] = None, max_cols: Optional[Any] = None, memory_usage: Any = None, show_counts: Any = None) -> None:
        info = DataFrameInfo(data=self, memory_usage=memory_usage)
        info.render(buf=buf, max_cols=max_cols, verbose=verbose, show_counts=show_counts)

    def memory_usage(self, index: bool = True, deep: bool = False) -> Series:
        result = self._constructor_sliced([c.memory_usage(index=False, deep=deep) for col, c in self.items()], index=self.columns, dtype=np.intp)
        if index:
            index_memory_usage = self._constructor_sliced(self.index.memory_usage(deep=deep), index=['Index'])
            result = index_memory_usage._append(result)
        return result

    def transpose(self, *args: Any, copy: Any = lib.no_default) -> DataFrame:
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
    def T(self) -> DataFrame:
        return self.transpose()

    def _ixs(self, i: int, axis: int = 0) -> Series:
        if axis == 0:
            new_mgr = self._mgr.fast_xs(i)
            result = self._constructor_sliced_from_mgr(new_mgr, axes=new_mgr.axes)
            result._name = self.index[i]
            return result.__finalize__(self)
        else:
            col_mgr = self._mgr.iget(i)
            return self._box_col_values(col_mgr, i)

    def _get_column_array(self, i: int) -> Any:
        return self._mgr.iget_values(i)

    def _iter_column_arrays(self) -> Iterator[Any]:
        for i in range(len(self.columns)):
            yield self._get_column_array(i)

    def __getitem__(self, key: Any) -> Any:
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

    def _getitem_bool_array(self, key: Any) -> DataFrame:
        if isinstance(key, Series) and (not key.index.equals(self.index)):
            warnings.warn('Boolean Series key will be reindexed to match DataFrame index.', UserWarning, stacklevel=find_stack_level())
        elif len(key) != len(self.index):
            raise ValueError(f'Item wrong length {len(key)} instead of {len(self.index)}.')
        key = check_bool_indexer(self.index, key)
        if key.all():
            return self.copy(deep=False)
        indexer = key.nonzero()[0]
        return self.take(indexer, axis=0)

    def _getitem_multilevel(self, key: Any) -> Any:
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

    def _get_value(self, index: Any, col: Any, takeable: bool = False) -> Any:
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

    def __setitem__(self, key: Any, value: Any) -> None:
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

    def _setitem_slice(self, key: slice, value: Any) -> None:
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
        def igetitem(obj: Any, i: int) -> Any:
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

    def _iset_item_mgr(self, loc: Any, value: Any, inplace: bool = True, refs: Optional[Any] = None) -> None:
        self._mgr.iset(loc, value, inplace=inplace, refs=refs)

    def _set_item_mgr(self, key: Any, value: Any, refs: Optional[Any] = None) -> None:
        try:
            loc = self._info_axis.get_loc(key)
        except KeyError:
            self._mgr.insert(len(self._info_axis), key, value, refs)
        else:
            self._iset_item_mgr(loc, value, refs=refs)

    def _iset_item(self, loc: Any, value: DataFrame, inplace: bool = True) -> None:
        self._iset_item_mgr(loc, value._values, inplace=inplace, refs=value._references)

    def _set_item(self, key: Any, value: Any) -> None:
        value, refs = self._sanitize_column(value)
        if key in self.columns and value.ndim == 1 and (not isinstance(value.dtype, ExtensionDtype)):
            if not self.columns.is_unique or isinstance(self.columns, MultiIndex):
                existing_piece = self[key]
                if isinstance(existing_piece, DataFrame):
                    value = np.tile(value, (len(existing_piece.columns), 1)).T
                    refs = None
        self._set_item_mgr(key, value, refs)

    def _set_value(self, index: Any, col: Any, value: Any, takeable: bool = False) -> None:
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

    def _box_col_values(self, values: Any, loc: int) -> Series:
        name = self.columns[loc]
        obj = self._constructor_sliced_from_mgr(values, axes=values.axes)
        obj._name = name
        return obj.__finalize__(self)

    def _get_item(self, item: Any) -> Any:
        loc = self.columns.get_loc(item)
        return self._ixs(loc, axis=1)

    @overload
    def query(self, expr: str, *, inplace: bool = ..., **kwargs: Any) -> Union[DataFrame, None]:
        ...

    @overload
    def query(self, expr: str, *, inplace: bool, **kwargs: Any) -> Union[DataFrame, None]:
        ...

    @overload
    def query(self, expr: str, *, inplace: bool = ..., **kwargs: Any) -> Union[DataFrame, None]:
        ...

    def query(self, expr: str, *, inplace: bool = False, **kwargs: Any) -> Union[DataFrame, None]:
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
    def eval(self, expr: str, *, inplace: bool = ..., **kwargs: Any) -> Any:
        ...

    @overload
    def eval(self, expr: str, *, inplace: bool, **kwargs: Any) -> Any:
        ...

    def eval(self, expr: str, *, inplace: bool = False, **kwargs: Any) -> Any:
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

    def select_dtypes(self, include: Optional[Any] = None, exclude: Optional[Any] = None) -> DataFrame:
        if not is_list_like(include):
            include = (include,) if include is not None else ()
        if not is_list_like(exclude):
            exclude = (exclude,) if exclude is not None else ()
        selection = (frozenset(include), frozenset(exclude))
        if not any(selection):
            raise ValueError('at least one of include or exclude must be nonempty')
        def check_int_infer_dtype(dtypes: Any) -> frozenset:
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
        def dtype_predicate(dtype: Any, dtypes_set: frozenset) -> bool:
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

    def insert(self, loc: int, column: Any, value: Any, allow_duplicates: Any = lib.no_default) -> None:
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

    def assign(self, **kwargs: Any) -> DataFrame:
        data = self.copy(deep=False)
        for k, v in kwargs.items():
            data[k] = com.apply_if_callable(v, data)
        return data

    def _sanitize_column(self, value: Any) -> Tuple[np.ndarray, Optional[Any]]:
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
    def _series(self) -> Dict[Any, Series]:
        return {item: self._ixs(idx, axis=1) for idx, item in enumerate(self.columns)}

    def _reindex_multi(self, axes: dict[str, Any], fill_value: Any) -> DataFrame:
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
    def set_axis(self, labels: Sequence[Any], *, axis: Union[int, str] = 0, copy: Any = lib.no_default) -> DataFrame:
        return super().set_axis(labels, axis=axis, copy=copy)

    @doc(NDFrame.reindex, klass=_shared_doc_kwargs['klass'], optional_reindex=_shared_doc_kwargs['optional_reindex'])
    def reindex(self, labels: Optional[Any] = None, *, index: Any = None, columns: Any = None, axis: Optional[Union[int, str]] = None, method: Any = None, copy: Any = lib.no_default, level: Any = None, fill_value: Any = np.nan, limit: Any = None, tolerance: Any = None) -> DataFrame:
        return super().reindex(labels=labels, index=index, columns=columns, axis=axis, method=method, level=level, fill_value=fill_value, limit=limit, tolerance=tolerance, copy=copy)

    @overload
    def drop(self, labels: Any = None, *, axis: Any = ..., index: Any = ..., columns: Any = ..., level: Any = ..., inplace: bool, errors: Any = ...) -> None:
        ...

    @overload
    def drop(self, labels: Any = None, *, axis: Any = ..., index: Any = ..., columns: Any = ..., level: Any = ..., inplace: bool = ..., errors: Any = ...) -> DataFrame:
        ...

    def drop(self, labels: Any = None, *, axis: Union[int, str] = 0, index: Any = None, columns: Any = None, level: Any = None, inplace: bool = False, errors: str = 'raise') -> Optional[DataFrame]:
        return super().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)

    @overload
    def rename(self, mapper: Any = None, *, index: Any = ..., columns: Any = ..., axis: Any = ..., copy: Any = lib.no_default, inplace: bool, level: Any = ..., errors: Any = ...) -> None:
        ...

    @overload
    def rename(self, mapper: Any = None, *, index: Any = ..., columns: Any = ..., axis: Any = ..., copy: Any = lib.no_default, inplace: bool = ..., level: Any = ..., errors: Any = ...) -> DataFrame:
        ...

    def rename(self, mapper: Any = None, *, index: Any = None, columns: Any = None, axis: Any = None, copy: Any = lib.no_default, inplace: bool = False, level: Any = None, errors: str = 'ignore') -> Optional[DataFrame]:
        self._check_copy_deprecation(copy)
        return super()._rename(mapper=mapper, index=index, columns=columns, axis=axis, inplace=inplace, level=level, errors=errors)

    def pop(self, item: Any) -> Series:
        return super().pop(item=item)

    @overload
    def _replace_columnwise(self, mapping: dict[Any, Tuple[Any, Any]], inplace: bool, regex: Any) -> Union[DataFrame, None]:
        ...

    @overload
    def _replace_columnwise(self, mapping: dict[Any, Tuple[Any, Any]], inplace: bool, regex: Any) -> Union[DataFrame, None]:
        ...

    def _replace_columnwise(self, mapping: dict[Any, Tuple[Any, Any]], inplace: bool, regex: Any) -> Optional[DataFrame]:
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
        return res.__finalize__(self, method='combine')

    @doc(make_doc('any', ndim=1))
    def any(self, *, axis: Union[int, str] = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> Any:
        result = self._logical_func('any', nanops.nanany, axis, bool_only, skipna, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='any')
        return result

    @doc(make_doc('all', ndim=1))
    def all(self, axis: Union[int, str] = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> Any:
        result = self._logical_func('all', nanops.nanall, axis, bool_only, skipna, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='all')
        return result

    @overload
    def min(self, *, axis: Union[int, str] = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @overload
    def min(self, *, axis: Union[int, str], skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    def min(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        result = super().min(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='min')
        return result

    @overload
    def max(self, *, axis: Union[int, str] = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @overload
    def max(self, *, axis: Union[int, str], skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    def max(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        result = super().max(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='max')
        return result

    @deprecate_nonkeyword_arguments(version='3.0.0', allowed_args=['self'], name='sum')
    def sum(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> Any:
        result = super().sum(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='sum')
        return result

    @deprecate_nonkeyword_arguments(version='3.0.0', allowed_args=['self'], name='prod')
    def prod(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> Any:
        result = super().prod(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='prod')
        return result
    product = prod

    @overload
    def mean(self, *, axis: Union[int, str] = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @overload
    def mean(self, *, axis: Union[int, str], skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='3.0.0', allowed_args=['self'], name='mean')
    def mean(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        result = super().mean(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='mean')
        return result

    @overload
    def median(self, *, axis: Union[int, str] = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @overload
    def median(self, *, axis: Union[int, str], skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='3.0.0', allowed_args=['self'], name='median')
    def median(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        result = super().median(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='median')
        return result

    @overload
    def sem(self, *, axis: Union[int, str] = ..., skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @overload
    def sem(self, *, axis: Union[int, str], skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='3.0.0', allowed_args=['self'], name='sem')
    def sem(self, axis: Union[int, str] = 0, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        result = super().sem(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='sem')
        return result

    @overload
    def var(self, *, axis: Union[int, str] = ..., skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @overload
    def var(self, *, axis: Union[int, str], skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='3.0.0', allowed_args=['self'], name='var')
    def var(self, axis: Union[int, str] = 0, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        result = super().var(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='var')
        return result

    @overload
    def std(self, *, axis: Union[int, str] = ..., skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @overload
    def std(self, *, axis: Union[int, str], skipna: bool = ..., ddof: int = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='3.0.0', allowed_args=['self'], name='std')
    def std(self, axis: Union[int, str] = 0, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        result = super().std(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='std')
        return result

    @overload
    def skew(self, *, axis: Union[int, str] = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @overload
    def skew(self, *, axis: Union[int, str], skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='3.0.0', allowed_args=['self'], name='skew')
    def skew(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        result = super().skew(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='skew')
        return result

    @overload
    def kurt(self, *, axis: Union[int, str] = ..., skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @overload
    def kurt(self, *, axis: Union[int, str], skipna: bool = ..., numeric_only: bool = ..., **kwargs: Any) -> Any:
        ...

    @deprecate_nonkeyword_arguments(version='3.0.0', allowed_args=['self'], name='kurt')
    def kurt(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        result = super().kurt(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
        if isinstance(result, Series):
            result = result.__finalize__(self, method='kurt')
        return result
    kurtosis = kurt

    @deprecate_nonkeyword_arguments(version='3.0.0', allowed_args=['self'], name='cummin')
    def cummin(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, *args: Any, **kwargs: Any) -> Any:
        data = self._get_numeric_data() if numeric_only else self
        return NDFrame.cummin(data, axis, skipna, *args, **kwargs)

    @doc(make_doc('cummax', ndim=2))
    def cummax(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, *args: Any, **kwargs: Any) -> Any:
        data = self._get_numeric_data() if numeric_only else self
        return NDFrame.cummax(data, axis, skipna, *args, **kwargs)

    @doc(make_doc('cumsum', ndim=2))
    def cumsum(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, *args: Any, **kwargs: Any) -> Any:
        data = self._get_numeric_data() if numeric_only else self
        return NDFrame.cumsum(data, axis, skipna, *args, **kwargs)

    @doc(make_doc('cumprod', 2))
    def cumprod(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False, *args: Any, **kwargs: Any) -> Any:
        data = self._get_numeric_data() if numeric_only else self
        return NDFrame.cumprod(data, axis, skipna, *args, **kwargs)

    def nunique(self, axis: Union[int, str] = 0, dropna: bool = True) -> Series:
        return self.apply(Series.nunique, axis=axis, dropna=dropna)

    def idxmin(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
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

    def idxmax(self, axis: Union[int, str] = 0, skipna: bool = True, numeric_only: bool = False) -> Series:
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

    def _get_agg_axis(self, axis_num: int) -> Index:
        if axis_num == 0:
            return self.columns
        elif axis_num == 1:
            return self.index
        else:
            raise ValueError(f'Axis must be 0 or 1 (got {axis_num!r})')

    def mode(self, axis: Union[int, str] = 0, numeric_only: bool = False, dropna: bool = True) -> DataFrame:
        data = self if not numeric_only else self._get_numeric_data()
        def f(s: Series) -> Series:
            return s.mode(dropna=dropna)
        data = data.apply(f, axis=axis)
        if data.empty:
            data.index = default_index(0)
        return data

    @overload
    def quantile(self, q: Union[float, Sequence[float]] = ..., axis: Union[int, str] = ..., numeric_only: bool = ..., interpolation: str = ..., method: str = ...) -> Union[Series, DataFrame]:
        ...

    @overload
    def quantile(self, q: Union[float, Sequence[float]], axis: Union[int, str] = ..., numeric_only: bool = ..., interpolation: str = ..., method: str = ...) -> Union[Series, DataFrame]:
        ...

    def quantile(self, q: Union[float, Sequence[float]] = 0.5, axis: Union[int, str] = 0, numeric_only: bool = False, interpolation: str = 'linear', method: str = 'single') -> Union[Series, DataFrame]:
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
            dtype: Any = np.float64
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
                indexer = nargsort(k, kind='quicksort')
            res = data._mgr.take(indexer[q_idx], verify=False)
            res.axes[1] = q
        result = self._constructor_from_mgr(res, axes=res.axes)
        return result.__finalize__(self, method='quantile')

    def to_timestamp(self, freq: Optional[str] = None, how: str = 'start', axis: Union[int, str] = 0, copy: Any = lib.no_default) -> DataFrame:
        self._check_copy_deprecation(copy)
        new_obj = self.copy(deep=False)
        axis_name = self._get_axis_name(axis)
        old_ax = getattr(self, axis_name)
        if not isinstance(old_ax, PeriodIndex):
            raise TypeError(f'unsupported Type {type(old_ax).__name__}')
        new_ax = old_ax.to_timestamp(freq=freq, how=how)
        setattr(new_obj, axis_name, new_ax)
        return new_obj

    def to_period(self, freq: Optional[str] = None, axis: Union[int, str] = 0, copy: Any = lib.no_default) -> DataFrame:
        self._check_copy_deprecation(copy)
        new_obj = self.copy(deep=False)
        axis_name = self._get_axis_name(axis)
        old_ax = getattr(self, axis_name)
        if not isinstance(old_ax, DatetimeIndex):
            raise TypeError(f'unsupported Type {type(old_ax).__name__}')
        new_ax = old_ax.to_period(freq=freq)
        setattr(new_obj, axis_name, new_ax)
        return new_obj

    def isin(self, values: Any) -> DataFrame:
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
            def isin_(x: np.ndarray) -> np.ndarray:
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
    index = properties.AxisProperty(axis=1, doc="\n        The index (row labels) of the DataFrame.\n\n        The index of a DataFrame is a series of labels that identify each row.\n        The labels can be integers, strings, or any other hashable type. The index\n        is used for label-based access and alignment, and can be accessed or\n        modified using this attribute.\n\n        Returns\n        -------\n        pandas.Index\n            The index labels of the DataFrame.\n\n        See Also\n        --------\n        DataFrame.columns: The column labels of the DataFrame.\n        DataFrame.to_numpy: Convert the DataFrame to a NumPy array.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Aritra'],\n        ...                    'Age': [25, 30, 35],\n        ...                    'Location': ['Seattle', 'New York', 'Kona']},\n        ...                   index=([10, 20, 30]))\n        >>> df.index\n        Index([10, 20, 30], dtype='int64')\n\n        In this example, we create a DataFrame with 3 rows and 3 columns,\n        including Name, Age, and Location information. We set the index labels to\n        be the integers 10, 20, and 30. We then access the `index` attribute of the\n        DataFrame, which returns an `Index` object containing the index labels.\n\n        >>> df.index = [100, 200, 300]\n        >>> df\n            Name  Age Location\n        100  Alice   25  Seattle\n        200    Bob   30 New York\n        300  Aritra  35    Kona\n\n        In this example, we modify the index labels of the DataFrame by assigning\n        a new list of labels to the `index` attribute. The DataFrame is then\n        updated with the new labels, and the output shows the modified DataFrame.\n        ")
    columns = properties.AxisProperty(axis=0, doc="\n        The column labels of the DataFrame.\n\n        This property holds the column names as a pandas ``Index`` object.\n        It provides an immutable sequence of column labels that can be\n        used for data selection, renaming, and alignment in DataFrame operations.\n\n        Returns\n        -------\n        pandas.Index\n            The column labels of the DataFrame.\n\n        See Also\n        --------\n        DataFrame.index: The index (row labels) of the DataFrame.\n        DataFrame.axes: Return a list representing the axes of the DataFrame.\n\n        Examples\n        --------\n        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})\n        >>> df\n                A  B\n        0    1  3\n        1    2  4\n        >>> df.columns\n        Index(['A', 'B'], dtype='object')\n        ")

    plot = Accessor('plot', pandas.plotting.PlotAccessor)
    hist = pandas.plotting.hist_frame
    boxplot = pandas.plotting.boxplot_frame
    sparse = Accessor('sparse', SparseFrameAccessor)

    def _to_dict_of_blocks(self) -> dict[Any, DataFrame]:
        mgr = self._mgr
        return {k: self._constructor_from_mgr(v, axes=v.axes).__finalize__(self) for k, v in mgr.to_iter_dict()}

    @property
    def values(self) -> np.ndarray:
        return self._mgr.as_array()

# Free helper functions with type annotations

def _from_nested_dict(data: Mapping[Any, Mapping[Any, Any]]) -> Dict[Any, Dict[Any, Any]]:
    new_data: Dict[Any, Dict[Any, Any]] = collections.defaultdict(dict)
    for index, s in data.items():
        for col, v in s.items():
            new_data[col][index] = v
    return new_data

def _reindex_for_setitem(value: Any, index: Index) -> Tuple[np.ndarray, Optional[Any]]:
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