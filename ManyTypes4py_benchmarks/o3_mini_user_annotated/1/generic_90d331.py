from __future__ import annotations
import datetime as dt
import numpy as np
from pandas._libs.tslibs.timedeltas import TimedeltaConvertibleTypes
from pandas._libs.tslibs.timestamps import TimestampConvertibleTypes
from pandas._typing import (
    AlignJoin,
    ArrayLike,
    DtypeBackend,
    DtypeObj,
    FilePath,
    FillnaOptions,
    FloatFormatType,
    FormattersType,
    Frequency,
    IgnoreRaise,
    IndexLabel,
    Level,
    ListLike,
    Manager,
    NaPosition,
    NDFrameT,
    OpenFileErrors,
    RandomState,
    ReindexMethod,
    Renamer,
    Scalar,
    Self,
    Suffixes,
    T,
    TimeAmbiguous,
    TimeNonexistent,
    TimeUnit,
    ValueKeyFunc,
    WriteBuffer,
)
from pandas._libs.lib import is_list_like, is_scalar, is_bool_dtype, is_number
from pandas.core.arrays import ExtensionArray
from pandas.core.indexes.datetimes import DatetimeIndex, PeriodIndex
from pandas.core.dtypes.common import pandas_dtype, is_dict_like, is_re_compilable
from pandas.core.missing import notna, isna
from pandas.core.nanops import nanops
from pandas.core.window.expanding import Expanding
from pandas.core.window.ewm import ExponentialMovingWindow
from pandas.core.window.rolling import Rolling
from pandas.core.window.window import Window

_num_doc: str = """\
{desc}

Parameters
----------
axis : {axis_descr}
    Axis for the function to be applied on.
    For `Series` this parameter is unused and defaults to 0.

    For DataFrames, specifying ``axis=None`` will apply the aggregation
    across both axes.

    .. versionadded:: 2.0.0

skipna : bool, default True
    Exclude NA/null values when computing the result.
numeric_only : bool, default False
    Include only float, int, boolean columns.

{min_count}\
**kwargs
    Additional keyword arguments to be passed to the function.

Returns
-------
{name1} or scalar\

    Value containing the calculation referenced in the description.\
{see_also}\
{examples}
"""

_sum_prod_doc: str = """\
{desc}

Parameters
----------
axis : {axis_descr}
    Axis for the function to be applied on.
    For `Series` this parameter is unused and defaults to 0.

    .. warning::

        The behavior of DataFrame.{name} with ``axis=None`` is deprecated,
        in a future version this will reduce over both axes and return a scalar
        To retain the old behavior, pass axis=0 (or do not pass axis).

skipna : bool, default True
    Exclude NA/null values when computing the result.
numeric_only : bool, default False
    Include only float, int, boolean columns.
{min_count}\
**kwargs
    Additional keyword arguments to be passed to the function.

Returns
-------
{name1} or scalar\

    Value containing the calculation referenced in the description.\
{see_also}\
{examples}
"""

_num_ddof_doc: str = """\
{desc}

Parameters
----------
axis : {axis_descr}
    For `Series` this parameter is unused and defaults to 0.

    .. warning::

        The behavior of DataFrame.{name} with ``axis=None`` is deprecated,
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
    Additional keywords have no effect but might be accepted
    for compatibility with NumPy.

Returns
-------
{name1} or {name2} (if level specified)
    {return_desc}

See Also
--------
{name2}.sum : Return the sum.
{name2}.min : Return the minimum.
{name2}.max : Return the maximum.
{name2}.idxmin : Return the index of the minimum.
{name2}.idxmax : Return the index of the maximum.
{name2}.mean : Return the mean.
{name2}.median : Return the median.
{name2}.mode : Return the mode(s).
{name2}.std : Return unbiased standard deviation over requested axis.
{name2}.sem : Return unbiased standard error of the mean over requested axis.
{name2}.var : Return unbiased variance over requested axis.
{see_also}\
{notes}\
{examples}
"""

_sem_see_also: str = """\
scipy.stats.sem : Compute standard error of the mean.
{name2}.std : Return sample standard deviation over requested axis.
{name2}.var : Return unbiased variance over requested axis.
{name2}.mean : Return the mean of the values over the requested axis.
{name2}.median : Return the median of the values over the requested axis.
{name2}.mode : Return the mode(s) of the Series."""
_std_see_also: str = """\
numpy.std : Compute the standard deviation along the specified axis.
{name2}.var : Return unbiased variance over requested axis.
{name2}.sem : Return unbiased standard error of the mean over requested axis.
{name2}.mean : Return the mean of the values over the requested axis.
{name2}.median : Return the median of the values over the requested axis.
{name2}.mode : Return the mode(s) of the Series."""
_std_return_desc: str = """\
Standard deviation over requested axis."""
_std_notes: str = """

Notes
-----
To have the same behaviour as `numpy.std`, use `ddof=0` (instead of the
default `ddof=1`)"""
_std_examples: str = """

Examples
--------
>>> df = pd.DataFrame({'person_id': [0, 1, 2, 3],
...                    'age': [21, 25, 62, 43],
...                    'height': [1.61, 1.87, 1.49, 2.01]}
...                   ).set_index('person_id')
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
dtype: float64"""

_var_examples: str = """

Examples
--------
>>> df = pd.DataFrame({'person_id': [0, 1, 2, 3],
...                    'age': [21, 25, 62, 43],
...                    'height': [1.61, 1.87, 1.49, 2.01]}
...                   ).set_index('person_id')
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
dtype: float64"""

_bool_doc: str = """\
{desc}

Parameters
----------
axis : {{0 or 'index', 1 or 'columns', None}}, default 0
    Indicate which axis or axes should be reduced. For `Series`
    this parameter is unused and defaults to 0.

skipna : bool, default True
    Exclude NA/null values when computing the result.
bool_only : bool, default False
    Include only boolean columns. Not implemented for Series.
**kwargs
    Additional keyword arguments which are ignored but accepted for compatibility.

Returns
-------
{name2} or {name1}
    If axis=None, then a scalar boolean is returned.
    Otherwise a Series is returned with index matching the axis argument.

{see_also}
{examples}"""

_all_desc: str = """\
Return whether all elements are True, potentially over an axis.

Returns True unless there at least one element within a series or
along a DataFrame axis that is False or equivalent (e.g. zero or
empty)."""
_all_examples: str = """\
Examples
--------
**Series**

>>> pd.Series([True, True]).all()
True
>>> pd.Series([True, False]).all()
False
>>> pd.Series([], dtype="float64").all()
True
>>> pd.Series([np.nan]).all()
True
>>> pd.Series([np.nan]).all(skipna=False)
True

**DataFrames**

Create a DataFrame from a dictionary.

>>> df = pd.DataFrame({'col1': [True, True], 'col2': [True, False]})
>>> df
       col1   col2
0      True   True
1      True  False

Default behaviour checks if values in each column all return True.

>>> df.all()
col1     True
col2    False
dtype: bool

Specify ``axis='columns'`` to check if values in each row all return True.

>>> df.all(axis='columns')
0     True
1    False
dtype: bool

Or ``axis=None`` for whether every value is True.

>>> df.all(axis=None)
False
"""

_all_see_also: str = """\
See Also
--------
Series.all : Return True if all elements are True.
DataFrame.any : Return True if one (or more) elements are True.
"""

_cnum_pd_doc: str = """\
Return cumulative {desc} over a DataFrame or Series axis.

Returns a DataFrame or Series of the same size containing the cumulative
{desc}.

Parameters
----------
axis : {{0 or 'index', 1 or 'columns'}}, default 0
    The index or the name of the axis. For `Series` this parameter
    is unused and defaults to 0.
skipna : bool, default True
    Exclude NA/null values. If an entire row/column is NA, the result
    will be NA.
numeric_only : bool, default False
    Include only float, int, boolean columns.
*args, **kwargs
    Additional keyword arguments are accepted but ignored.

Returns
-------
{name1} or {name2}
    Return cumulative {desc} of {name1} or {name2}.

See Also
--------
core.window.expanding.Expanding.{accum_func_name} : Similar functionality
    but ignores ``NaN`` values.
{name2}.{accum_func_name} : Return the cumulative {desc} over
    {name2} axis.
{name2}.cummax : Return cumulative maximum over {name2} axis.
{name2}.cummin : Return cumulative minimum over {name2} axis.
{name2}.cumsum : Return cumulative sum over {name2} axis.
{name2}.cumprod : Return cumulative product over {name2} axis.

{examples}"""

_cnum_series_doc: str = """\
Return cumulative {desc} over a DataFrame or Series axis.

Returns a {name1} of the same size containing the cumulative
{desc}.

Parameters
----------
axis : {{0 or 'index', 1 or 'columns'}}, default 0
    The index or the name of the axis. For `Series` this parameter
    is unused and defaults to 0.
skipna : bool, default True
    Exclude NA/null values.
*args, **kwargs
    Additional keyword arguments are accepted but ignored.

Returns
-------
{name1} or {name2}
    The cumulative {desc} of {name1} or {name2}.

See Also
--------
core.window.expanding.Expanding.{accum_func_name} : Similar functionality
    but ignores ``NaN`` values.
{name2}.{accum_func_name} : Return the cumulative {desc} over
    {name2} axis.
{name2}.cummax : Return cumulative maximum over {name2} axis.
{name2}.cummin : Return cumulative minimum over {name2} axis.
{name2}.cumsum : Return cumulative sum over {name2} axis.
{name2}.cumprod : Return cumulative product over {name2} axis.

{examples}"""

_cummin_examples: str = """\
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

By default, NA values are ignored.

>>> s.cummin()
0    2.0
1    NaN
2    2.0
3   -1.0
4   -1.0
dtype: float64

To include NA values, use skipna=False:

>>> s.cummin(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0

Default (axis=0):

>>> df.cummin()
     A    B
0  2.0  1.0
1  2.0  NaN
2  1.0  0.0

Axis=1:

>>> df.cummin(axis=1)
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0
"""

_cumsum_examples: str = """\
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

Default, NA values ignored:

>>> s.cumsum()
0    2.0
1    NaN
2    7.0
3    6.0
4    6.0
dtype: float64

With skipna=False:

>>> s.cumsum(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0

Default (axis=0):

>>> df.cumsum()
     A    B
0  2.0  1.0
1  5.0  NaN
2  6.0  1.0

Axis=1:

>>> df.cumsum(axis=1)
     A    B
0  2.0  3.0
1  3.0  NaN
2  1.0  1.0
"""

_cumprod_examples: str = """\
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

Default:

>>> s.cumprod()
0     2.0
1     NaN
2    10.0
3   -10.0
4    -0.0
dtype: float64

With skipna=False:

>>> s.cumprod(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0

Default (axis=0):

>>> df.cumprod()
     A    B
0  2.0  1.0
1  6.0  NaN
2  6.0  0.0

Axis=1:

>>> df.cumprod(axis=1)
     A    B
0  2.0  2.0
1  3.0  NaN
2  1.0  0.0
"""

_cummax_examples: str = """\
Examples
--------
**Series**

>>> s = pd.Series([2, np.nan, 5, -1, 0])
>>> s
0    2.0
1    NaN
2    5.0
3   -1.0
4    0.0
dtype: float64

Default (skipna=True):

>>> s.cummax()
0    2.0
1    NaN
2    5.0
3    5.0
4    5.0
dtype: float64

With skipna=False:

>>> s.cummax(skipna=False)
0    2.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

**DataFrame**

>>> df = pd.DataFrame([[2.0, 1.0],
...                    [3.0, np.nan],
...                    [1.0, 0.0]],
...                   columns=list('AB'))
>>> df
     A    B
0  2.0  1.0
1  3.0  NaN
2  1.0  0.0

Default (axis=0):

>>> df.cummax()
     A    B
0  2.0  1.0
1  3.0  NaN
2  3.0  1.0

Axis=1:

>>> df.cummax(axis=1)
     A    B
0  2.0  2.0
1  3.0  NaN
2  1.0  1.0
"""

_any_examples: str = """\
Examples
--------
**Series**

>>> pd.Series([False, False]).any()
False
>>> pd.Series([True, False]).any()
True
>>> pd.Series([], dtype="float64").any()
False
>>> pd.Series([np.nan]).any()
False
>>> pd.Series([np.nan]).any(skipna=False)
True

**DataFrame**

>>> df = pd.DataFrame({"A": [True, True], "B": [1, 2]})
>>> df.any(axis='columns')
0    True
1    True
dtype: bool
>>> df = pd.DataFrame({"A": [True, False], "B": [1, 0]})
>>> df.any(axis='columns')
0    True
1    False
dtype: bool
>>> df.any(axis=None)
True
"""

_shared_docs: dict[str, str] = {}
_shared_docs["stat_func_example"] = """
Examples
--------
>>> idx = pd.MultiIndex.from_arrays([
...     ['warm', 'warm', 'cold', 'cold'],
...     ['dog', 'falcon', 'fish', 'spider']],
...     names=['blooded', 'animal'])
>>> s = pd.Series([4, 2, 0, 8], name='legs', index=idx)
>>> s
blooded  animal
warm     dog       4
         falcon    2
cold     fish      0
         spider    8
Name: legs, dtype: int64

>>> s.{stat_func}()
{default_output}"""

_sum_examples: str = _shared_docs["stat_func_example"].format(
    stat_func="sum", default_output="14"
)

_max_examples: str = _shared_docs["stat_func_example"].format(
    stat_func="max", default_output="8"
)

_min_examples: str = _shared_docs["stat_func_example"].format(
    stat_func="min", default_output="0"
)

_prod_examples: str = _shared_docs["stat_func_example"].format(
    stat_func="prod", default_output="0"
)

_min_count_stub: str = """\
min_count : int, default 0
    The required number of valid values to perform the operation. If fewer than
    ``min_count`` non-NA values are present the result will be NA.
"""

_all_see_also: str = _all_see_also  # already defined
_any_see_also: str = _any_see_also  # already defined

def make_doc(name: str, ndim: int) -> str:
    """
    Generate the docstring for a Series/DataFrame reduction.

    Parameters
    ----------
    name : str
        Name of the statistic function.
    ndim : int
        Number of dimensions (1 for Series, 2 for DataFrame).

    Returns
    -------
    str
        The formatted docstring.
    """
    if ndim == 1:
        name1: str = "scalar"
        name2: str = "Series"
        axis_descr: str = "{index (0)}"
    else:
        name1 = "Series"
        name2 = "DataFrame"
        axis_descr = "{index (0), columns (1)}"

    if name == "any":
        base_doc: str = _bool_doc
        desc: str = _any_desc
        see_also: str = _any_see_also
        examples: str = _any_examples
        kwargs: dict[str, str] = {"empty_value": "False"}
    elif name == "all":
        base_doc = _bool_doc
        desc = _all_desc
        see_also = _all_see_also
        examples = _all_examples
        kwargs = {"empty_value": "True"}
    elif name == "min":
        base_doc = _num_doc
        desc = (
            "Return the minimum of the values over the requested axis.\n\n"
            "If you want the *index* of the minimum, use ``idxmin``. This is "
            "the equivalent of the ``numpy.ndarray`` method ``argmin``."
        )
        see_also = _stat_func_see_also
        examples = _min_examples
        kwargs = {"min_count": ""}
    elif name == "max":
        base_doc = _num_doc
        desc = (
            "Return the maximum of the values over the requested axis.\n\n"
            "If you want the *index* of the maximum, use ``idxmax``. This is "
            "the equivalent of the ``numpy.ndarray`` method ``argmax``."
        )
        see_also = _stat_func_see_also
        examples = _max_examples
        kwargs = {"min_count": ""}
    elif name == "sum":
        base_doc = _sum_prod_doc
        desc = (
            "Return the sum of the values over the requested axis.\n\n"
            "This is equivalent to the method ``numpy.sum``."
        )
        see_also = _stat_func_see_also
        examples = _sum_examples
        kwargs = {"min_count": _min_count_stub}
    elif name == "prod":
        base_doc = _sum_prod_doc
        desc = "Return the product of the values over the requested axis."
        see_also = _stat_func_see_also
        examples = _prod_examples
        kwargs = {"min_count": _min_count_stub}
    elif name == "median":
        base_doc = _num_doc
        desc = "Return the median of the values over the requested axis."
        see_also = _stat_func_see_also
        examples = """
Examples
--------
>>> s = pd.Series([1, 2, 3])
>>> s.median()
2.0

With a DataFrame

>>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
>>> df.median()
a   1.5
b   2.5
dtype: float64

Using axis=1

>>> df.median(axis=1)
tiger   1.5
zebra   2.5
dtype: float64

For non-numeric columns, set numeric_only=True:

>>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']}, index=['tiger', 'zebra'])
>>> df.median(numeric_only=True)
a   1.5
dtype: float64
"""
        kwargs = {"min_count": ""}
    elif name == "mean":
        base_doc = _num_doc
        desc = "Return the mean of the values over the requested axis."
        see_also = _stat_func_see_also
        examples = """
Examples
--------
>>> s = pd.Series([1, 2, 3])
>>> s.mean()
2.0

With a DataFrame

>>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
>>> df.mean()
a   1.5
b   2.5
dtype: float64

Using axis=1

>>> df.mean(axis=1)
tiger   1.5
zebra   2.5
dtype: float64

For non-numeric columns, set numeric_only=True:

>>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']}, index=['tiger', 'zebra'])
>>> df.mean(numeric_only=True)
a   1.5
dtype: float64
"""
        kwargs = {"min_count": ""}
    elif name == "var":
        base_doc = _num_ddof_doc
        desc = (
            "Return unbiased variance over requested axis.\n\nNormalized by "
            "N-1 by default. This can be changed using the ddof argument."
        )
        examples = _var_examples
        see_also = ""
        kwargs = {"notes": ""}
    elif name == "std":
        base_doc = _num_ddof_doc
        desc = (
            "Return sample standard deviation over requested axis.\n\n"
            "Normalized by N-1 by default. This can be changed using the "
            "ddof argument."
        )
        examples = _std_examples
        see_also = _std_see_also.format(name2=name2)
        kwargs = {"notes": "", "return_desc": _std_return_desc}
    elif name == "sem":
        base_doc = _num_ddof_doc
        desc = (
            "Return unbiased standard error of the mean over requested axis.\n\n"
            "Normalized by N-1 by default. This can be changed using the ddof argument."
        )
        examples = """
Examples
--------
>>> s = pd.Series([1, 2, 3])
>>> s.sem().round(6)
0.57735

With a DataFrame

>>> df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]}, index=['tiger', 'zebra'])
>>> df.sem()
a   0.5
b   0.5
dtype: float64

Using axis=1

>>> df.sem(axis=1)
tiger   0.5
zebra   0.5
dtype: float64

For non-numeric columns, set numeric_only=True:

>>> df = pd.DataFrame({'a': [1, 2], 'b': ['T', 'Z']}, index=['tiger', 'zebra'])
>>> df.sem(numeric_only=True)
a   0.5
dtype: float64
"""
        see_also = _sem_see_also.format(name2=name2)
        kwargs = {"notes": "", "return_desc": "Unbiased standard error of the mean over requested axis."}
    elif name == "skew":
        base_doc = _num_doc
        desc = "Return unbiased skew over requested axis.\n\nNormalized by N-1."
        see_also = _skew_see_also
        examples = """
Examples
--------
>>> s = pd.Series([1, 2, 3])
>>> s.skew()
0.0

With a DataFrame

>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [1, 3, 5]},
...                   index=['tiger', 'zebra', 'cow'])
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

For non-numeric columns, set numeric_only=True:

>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['T', 'Z', 'X']},
...                   index=['tiger', 'zebra', 'cow'])
>>> df.skew(numeric_only=True)
a   0.0
dtype: float64
"""
        kwargs = {"min_count": ""}
    elif name == "kurt":
        base_doc = _num_doc
        desc = (
            "Return unbiased kurtosis over requested axis.\n\n"
            "Kurtosis using Fisher's definition (normal==0.0). Normalized by "
            "N-1."
        )
        see_also = ""
        examples = """
Examples
--------
>>> s = pd.Series([1, 2, 2, 3])
>>> s.kurt()
1.5

With a DataFrame

>>> df = pd.DataFrame({'a': [1, 2, 2, 3], 'b': [3, 4, 4, 4]},
...                   index=['cat', 'dog', 'dog', 'mouse'])
>>> df.kurt()
a   1.5
b   4.0
dtype: float64

Using axis=None

>>> df.kurt(axis=None).round(6)
-0.988693

Using axis=1

>>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [3, 4], 'd': [1, 2]},
...                   index=['cat', 'dog'])
>>> df.kurt(axis=1)
cat   -6.0
dog   -6.0
dtype: float64
"""
        kwargs = {"min_count": ""}
    elif name == "cumsum":
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        desc = "sum"
        see_also = ""
        examples = _cumsum_examples
        kwargs = {"accum_func_name": "sum"}
    elif name == "cumprod":
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        desc = "product"
        see_also = ""
        examples = _cumprod_examples
        kwargs = {"accum_func_name": "prod"}
    elif name == "cummin":
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        desc = "minimum"
        see_also = ""
        examples = _cummin_examples
        kwargs = {"accum_func_name": "min"}
    elif name == "cummax":
        if ndim == 1:
            base_doc = _cnum_series_doc
        else:
            base_doc = _cnum_pd_doc
        desc = "maximum"
        see_also = ""
        examples = _cummax_examples
        kwargs = {"accum_func_name": "max"}
    else:
        raise NotImplementedError(f"make_doc for {name!r} is not implemented")

    docstr: str = base_doc.format(
        desc=desc,
        name=name,
        name1=name1,
        name2=name2,
        axis_descr=axis_descr,
        see_also=see_also,
        examples=examples,
        **kwargs,
    )
    return docstr