"""
Constructor functions intended to be shared by pd.array, Series.__init__,
and Index.__new__.

These should not depend on core.internals.
"""
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
    Optional,
    cast,
    overload,
    Sequence,
    TypeVar,
    List,
    Tuple,
    Dict,
    Set,
    FrozenSet,
    Iterable,
    Iterator,
    Generator,
    Callable,
    Mapping,
    MutableMapping,
    Type,
    Generic,
    NoReturn,
    Awaitable,
    Coroutine,
    AsyncIterable,
    AsyncIterator,
    AsyncGenerator,
    ClassVar,
    Final,
    Literal,
    Protocol,
    runtime_checkable,
    TypedDict,
    NamedTuple,
    NewType,
    AnyStr,
    IO,
    TextIO,
    BinaryIO,
    Pattern,
    Match,
    Hashable,
    Sized,
    Container,
    Collection,
    Reversible,
    SupportsInt,
    SupportsFloat,
    SupportsComplex,
    SupportsAbs,
    SupportsRound,
    SupportsBytes,
    SupportsIndex,
    SupportsFloat,
    SupportsComplex,
    SupportsRound,
    SupportsBytes,
    SupportsIndex,
    SupportsFloat,
    SupportsComplex,
    SupportsRound,
    SupportsBytes,
    SupportsIndex,
    AbstractSet,
    MutableSet,
    MutableSequence,
    MutableMapping,
    MappingView,
    KeysView,
    ItemsView,
    ValuesView,
    ContextManager,
    AsyncContextManager,
    runtime_checkable,
    get_type_hints,
    get_origin,
    get_args,
    ForwardRef,
    _SpecialForm,
    _GenericAlias,
    _SpecialGenericAlias,
    _VariadicGenericAlias,
    _UnionGenericAlias,
    _TupleGenericAlias,
    _CallableGenericAlias,
    _LiteralGenericAlias,
    _AnnotatedAlias,
    _TypeAlias,
    _TypeVarLike,
    _SpecialForm,
    _GenericAlias,
    _SpecialGenericAlias,
    _VariadicGenericAlias,
    _UnionGenericAlias,
    _TupleGenericAlias,
    _CallableGenericAlias,
    _LiteralGenericAlias,
    _AnnotatedAlias,
    _TypeAlias,
    _TypeVarLike,
)
import numpy as np
from numpy import ma
from pandas._config import using_string_dtype
from pandas._libs import lib
from pandas._libs.tslibs import get_supported_dtype, is_supported_dtype
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import (
    construct_1d_arraylike_from_scalar,
    construct_1d_object_array_from_listlike,
    maybe_cast_to_datetime,
    maybe_cast_to_integer_array,
    maybe_convert_platform,
    maybe_infer_to_datetimelike,
    maybe_promote,
)
from pandas.core.dtypes.common import (
    ensure_object,
    is_list_like,
    is_object_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCExtensionArray,
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna
import pandas.core.common as com

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pandas._typing import (
        AnyArrayLike,
        ArrayLike,
        Dtype,
        DtypeObj,
        T,
    )
    from pandas import Index, Series
    from pandas.core.arrays import (
        DatetimeArray,
        ExtensionArray,
        TimedeltaArray,
    )

T = TypeVar("T")

def array(
    data: Sequence[object] | AnyArrayLike,
    dtype: Optional[Dtype] = None,
    copy: bool = True,
) -> ExtensionArray:
    """
    Create an array.

    This method constructs an array using pandas extension types when possible.
    If `dtype` is specified, it determines the type of array returned. Otherwise,
    pandas attempts to infer the appropriate dtype based on `data`.

    Parameters
    ----------
    data : Sequence of objects
        The scalars inside `data` should be instances of the
        scalar type for `dtype`. It's expected that `data`
        represents a 1-dimensional array of data.

        When `data` is an Index or Series, the underlying array
        will be extracted from `data`.

    dtype : str, np.dtype, or ExtensionDtype, optional
        The dtype to use for the array. This may be a NumPy
        dtype or an extension type registered with pandas using
        :meth:`pandas.api.extensions.register_extension_dtype`.

        If not specified, there are two possibilities:

        1. When `data` is a :class:`Series`, :class:`Index`, or
           :class:`ExtensionArray`, the `dtype` will be taken
           from the data.
        2. Otherwise, pandas will attempt to infer the `dtype`
           from the data.

        Note that when `data` is a NumPy array, ``data.dtype`` is
        *not* used for inferring the array type. This is because
        NumPy cannot represent all the types of data that can be
        held in extension arrays.

        Currently, pandas will infer an extension dtype for sequences of

        ============================== =======================================
        Scalar Type                    Array Type
        ============================== =======================================
        :class:`pandas.Interval`       :class:`pandas.arrays.IntervalArray`
        :class:`pandas.Period`         :class:`pandas.arrays.PeriodArray`
        :class:`datetime.datetime`     :class:`pandas.arrays.DatetimeArray`
        :class:`datetime.timedelta`    :class:`pandas.arrays.TimedeltaArray`
        :class:`int`                   :class:`pandas.arrays.IntegerArray`
        :class:`float`                 :class:`pandas.arrays.FloatingArray`
        :class:`str`                   :class:`pandas.arrays.StringArray` or
                                       :class:`pandas.arrays.ArrowStringArray`
        :class:`bool`                  :class:`pandas.arrays.BooleanArray`
        ============================== =======================================

        The ExtensionArray created when the scalar type is :class:`str` is determined by
        ``pd.options.mode.string_storage`` if the dtype is not explicitly given.

        For all other cases, NumPy's usual inference rules will be used.
    copy : bool, default True
        Whether to copy the data, even if not necessary. Depending
        on the type of `data`, creating the new array may require
        copying data, even if ``copy=False``.

    Returns
    -------
    ExtensionArray
        The newly created array.

    Raises
    ------
    ValueError
        When `data` is not 1-dimensional.

    See Also
    --------
    numpy.array : Construct a NumPy array.
    Series : Construct a pandas Series.
    Index : Construct a pandas Index.
    arrays.NumpyExtensionArray : ExtensionArray wrapping a NumPy array.
    Series.array : Extract the array stored within a Series.

    Notes
    -----
    Omitting the `dtype` argument means pandas will attempt to infer the
    best array type from the values in the data. As new array types are
    added by pandas and 3rd party libraries, the "best" array type may
    change. We recommend specifying `dtype` to ensure that

    1. the correct array type for the data is returned
    2. the returned array type doesn't change as new extension types
       are added by pandas and third-party libraries

    Additionally, if the underlying memory representation of the returned
    array matters, we recommend specifying the `dtype` as a concrete object
    rather than a string alias or allowing it to be inferred. For example,
    a future version of pandas or a 3rd-party library may include a
    dedicated ExtensionArray for string data. In this event, the following
    would no longer return a :class:`arrays.NumpyExtensionArray` backed by a
    NumPy array.

    >>> pd.array(["a", "b"], dtype=str)
    <NumpyExtensionArray>
    ['a', 'b']
    Length: 2, dtype: str32

    This would instead return the new ExtensionArray dedicated for string
    data. If you really need the new array to be backed by a  NumPy array,
    specify that in the dtype.

    >>> pd.array(["a", "b"], dtype=np.dtype("<U1"))
    <NumpyExtensionArray>
    ['a', 'b']
    Length: 2, dtype: str32

    Finally, Pandas has arrays that mostly overlap with NumPy

      * :class:`arrays.DatetimeArray`
      * :class:`arrays.TimedeltaArray`

    When data with a ``datetime64[ns]`` or ``timedelta64[ns]`` dtype is
    passed, pandas will always return a ``DatetimeArray`` or ``TimedeltaArray``
    rather than a ``NumpyExtensionArray``. This is for symmetry with the case of
    timezone-aware data, which NumPy does not natively support.

    >>> pd.array(["2015", "2016"], dtype="datetime64[ns]")
    <DatetimeArray>
    ['2015-01-01 00:00:00', '2016-01-01 00:00:00']
    Length: 2, dtype: datetime64[ns]

    >>> pd.array(["1h", "2h"], dtype="timedelta64[ns]")
    <TimedeltaArray>
    ['0 days 01:00:00', '0 days 02:00:00']
    Length: 2, dtype: timedelta64[ns]

    Examples
    --------
    If a dtype is not specified, pandas will infer the best dtype from the values.
    See the description of `dtype` for the types pandas infers for.

    >>> pd.array([1, 2])
    <IntegerArray>
    [1, 2]
    Length: 2, dtype: Int64

    >>> pd.array([1, 2, np.nan])
    <IntegerArray>
    [1, 2, <NA>]
    Length: 3, dtype: Int64

    >>> pd.array([1.1, 2.2])
    <FloatingArray>
    [1.1, 2.2]
    Length: 2, dtype: Float64

    >>> pd.array(["a", None, "c"])
    <StringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

    >>> with pd.option_context("string_storage", "pyarrow"):
    ...     arr = pd.array(["a", None, "c"])
    >>> arr
    <ArrowStringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

    >>> pd.array([pd.Period("2000", freq="D"), pd.Period("2000", freq="D")])
    <PeriodArray>
    ['2000-01-01', '2000-01-01']
    Length: 2, dtype: period[D]

    You can use the string alias for `dtype`

    >>> pd.array(["a", "b", "a"], dtype="category")
    ['a', 'b', 'a']
    Categories (2, object): ['a', 'b']

    Or specify the actual dtype

    >>> pd.array(
    ...     ["a", "b", "a"], dtype=pd.CategoricalDtype(["a", "b", "c"], ordered=True)
    ... )
    ['a', 'b', 'a']
    Categories (3, object): ['a' < 'b' < 'c']

    If pandas does not infer a dedicated extension type a
    :class:`arrays.NumpyExtensionArray` is returned.

    >>> pd.array([1 + 1j, 3 + 2j])
    <NumpyExtensionArray>
    [(1+1j), (3+2j)]
    Length: 2, dtype: complex128

    As mentioned in the "Notes" section, new extension types may be added
    in the future (by pandas or 3rd party libraries), causing the return
    value to no longer be a :class:`arrays.NumpyExtensionArray`. Specify the
    `dtype` as a NumPy dtype if you need to ensure there's no future change in
    behavior.

    >>> pd.array([1, 2], dtype=np.dtype("int32"))
    <NumpyExtensionArray>
    [1, 2]
    Length: 2, dtype: int32

    `data` must be 1-dimensional. A ValueError is raised when the input
    has the wrong dimensionality.

    >>> pd.array(1)
    Traceback (most recent call last):
      ...
    ValueError: Cannot pass scalar '1' to 'pandas.array'.
    """
    from pandas.core.arrays import (
        BooleanArray,
        DatetimeArray,
        ExtensionArray,
        FloatingArray,
        IntegerArray,
        NumpyExtensionArray,
        TimedeltaArray,
    )
    from pandas.core.arrays.string_ import StringDtype

    if lib.is_scalar(data):
        msg = f"Cannot pass scalar '{data}' to 'pandas.array'."
        raise ValueError(msg)
    elif isinstance(data, ABCDataFrame):
        raise TypeError("Cannot pass DataFrame to 'pandas.array'")

    if dtype is None and isinstance(data, (ABCSeries, ABCIndex, ExtensionArray)):
        dtype = data.dtype

    data = extract_array(data, extract_numpy=True)

    if dtype is not None:
        dtype = pandas_dtype(dtype)

    if isinstance(data, ExtensionArray) and (dtype is None or data.dtype == dtype):
        if copy:
            return data.copy()
        return data

    if isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        return cls._from_sequence(data, dtype=dtype, copy=copy)

    if dtype is None:
        was_ndarray = isinstance(data, np.ndarray)
        if not was_ndarray or data.dtype == object:
            result = lib.maybe_convert_objects(
                ensure_object(data),
                convert_non_numeric=True,
                convert_to_nullable_dtype=True,
                dtype_if_all_nat=None,
            )
            result = ensure_wrapped_if_datetimelike(result)
            if isinstance(result, np.ndarray):
                if len(result) == 0 and (not was_ndarray):
                    return FloatingArray._from_sequence(data, dtype="Float64")
                return NumpyExtensionArray._from_sequence(data, dtype=result.dtype, copy=copy)
            if result is data and copy:
                return result.copy()
            return result

        data = cast(np.ndarray, data)
        result = ensure_wrapped_if_datetimelike(data)
        if result is not data:
            result = cast("DatetimeArray | TimedeltaArray", result)
            if copy and result.dtype == data.dtype:
                return result.copy()
            return result

        if data.dtype.kind in "SU":
            dtype = StringDtype()
            cls = dtype.construct_array_type()
            return cls._from_sequence(data, dtype=dtype, copy=copy)
        elif data.dtype.kind in "iu":
            dtype = IntegerArray._dtype_cls._get_dtype_mapping()[data.dtype]
            return IntegerArray._from_sequence(data, dtype=dtype, copy=copy)
        elif data.dtype.kind == "f":
            if data.dtype == np.float16:
                return NumpyExtensionArray._from_sequence(data, dtype=data.dtype, copy=copy)
            dtype = FloatingArray._dtype_cls._get_dtype_mapping()[data.dtype]
            return FloatingArray._from_sequence(data, dtype=dtype, copy=copy)
        elif data.dtype.kind == "b":
            return BooleanArray._from_sequence(data, dtype="boolean", copy=copy)
        else:
            return NumpyExtensionArray._from_sequence(data, dtype=data.dtype, copy=copy)

    if lib.is_np_dtype(dtype, "M") and is_supported_dtype(dtype):
        return DatetimeArray._from_sequence(data, dtype=dtype, copy=copy)

    if lib.is_np_dtype(dtype, "m") and is_supported_dtype(dtype):
        return TimedeltaArray._from_sequence(data, dtype=dtype, copy=copy)
    elif lib.is_np_dtype(dtype, "mM"):
        raise ValueError(
            "datetime64 and timedelta64 dtype resolutions other than 's', 'ms', 'us', and 'ns' are no longer supported."
        )

    return NumpyExtensionArray._from_sequence(data, dtype=dtype, copy=copy)

_typs = frozenset(
    {
        "index",
        "rangeindex",
        "multiindex",
        "datetimeindex",
        "timedeltaindex",
        "periodindex",
        "categoricalindex",
        "intervalindex",
        "series",
    }
)

@overload
def extract_array(
    obj: T,
    extract_numpy: bool = ...,
    extract_range: bool = ...,
) -> T:
    ...

@overload
def extract_array(
    obj: Union[ABCSeries, ABCIndex],
    extract_numpy: bool = ...,
    extract_range: bool = ...,
) -> Union[np.ndarray, ExtensionArray]:
    ...

def extract_array(
    obj: Any,
    extract_numpy: bool = False,
    extract_range: bool = False,
) -> Any:
    """
    Extract the ndarray or ExtensionArray from a Series or Index.

    For all other types, `obj` is just returned as is.

    Parameters
    ----------
    obj : object
        For Series / Index, the underlying ExtensionArray is unboxed.

    extract_numpy : bool, default False
        Whether to extract the ndarray from a NumpyExtensionArray.

    extract_range : bool, default False
        If we have a RangeIndex, return range._values if True
        (which is a materialized integer ndarray), otherwise return unchanged.

    Returns
    -------
    arr : object

    Examples
    --------
    >>> extract_array(pd.Series(["a", "b", "c"], dtype="category"))
    ['a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Other objects like lists, arrays, and DataFrames are just passed through.

    >>> extract_array([1, 2, 3])
    [