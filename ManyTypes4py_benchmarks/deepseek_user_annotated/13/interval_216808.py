from __future__ import annotations

import operator
from operator import (
    le,
    lt,
)
import textwrap
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
    cast,
)

import numpy as np
import numpy.typing as npt

from pandas._libs import lib
from pandas._libs.interval import (
    VALID_CLOSED,
    Interval,
    IntervalMixin,
    intervals_to_interval_bounds,
)
from pandas._libs.missing import NA
from pandas._typing import (
    ArrayLike,
    AxisInt,
    Dtype,
    IntervalClosedType,
    NpDtype,
    PositionalIndexer,
    ScalarIndexer,
    Self,
    SequenceIndexer,
    SortKind,
    TimeArrayLike,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import IntCastingNaNError
from pandas.util._decorators import Appender

from pandas.core.dtypes.cast import (
    LossySetitemError,
    maybe_upcast_numeric_to_64bit,
)
from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    IntervalDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCDatetimeIndex,
    ABCIntervalIndex,
    ABCPeriodIndex,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    notna,
)

from pandas.core.algorithms import (
    isin,
    take,
    unique,
    value_counts_internal as value_counts,
)
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import (
    ExtensionArray,
    _extension_array_shared_docs,
)
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import (
    invalid_comparison,
    unpack_zerodim_and_defer,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Sequence,
    )

    from pandas import (
        Index,
        Series,
    )


IntervalSide = Union[TimeArrayLike, np.ndarray]
IntervalOrNA = Union[Interval, float]

_interval_shared_docs: dict[str, str] = {}

_shared_docs_kwargs = {
    "klass": "IntervalArray",
    "qualname": "arrays.IntervalArray",
    "name": "",
}


_interval_shared_docs["class"] = """
%(summary)s

Parameters
----------
data : array-like (1-dimensional)
    Array-like (ndarray, :class:`DateTimeArray`, :class:`TimeDeltaArray`) containing
    Interval objects from which to build the %(klass)s.
closed : {'left', 'right', 'both', 'neither'}, default 'right'
    Whether the intervals are closed on the left-side, right-side, both or
    neither.
dtype : dtype or None, default None
    If None, dtype will be inferred.
copy : bool, default False
    Copy the input data.
%(name)s\
verify_integrity : bool, default True
    Verify that the %(klass)s is valid.

Attributes
----------
left
right
closed
mid
length
is_empty
is_non_overlapping_monotonic
%(extra_attributes)s\

Methods
-------
from_arrays
from_tuples
from_breaks
contains
overlaps
set_closed
to_tuples
%(extra_methods)s\

See Also
--------
Index : The base pandas Index type.
Interval : A bounded slice-like interval; the elements of an %(klass)s.
interval_range : Function to create a fixed frequency IntervalIndex.
cut : Bin values into discrete Intervals.
qcut : Bin values into equal-sized Intervals based on rank or sample quantiles.

Notes
-----
See the `user guide
<https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#intervalindex>`__
for more.

%(examples)s\
"""


@Appender(
    _interval_shared_docs["class"]
    % {
        "klass": "IntervalArray",
        "summary": "Pandas array for interval data that are closed on the same side.",
        "name": "",
        "extra_attributes": "",
        "extra_methods": "",
        "examples": textwrap.dedent(
            """\
    Examples
    --------
    A new ``IntervalArray`` can be constructed directly from an array-like of
    ``Interval`` objects:

    >>> pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
    <IntervalArray>
    [(0, 1], (1, 5]]
    Length: 2, dtype: interval[int64, right]

    It may also be constructed using one of the constructor
    methods: :meth:`IntervalArray.from_arrays`,
    :meth:`IntervalArray.from_breaks`, and :meth:`IntervalArray.from_tuples`.
    """
        ),
    }
)
class IntervalArray(IntervalMixin, ExtensionArray):
    can_hold_na = True
    _na_value = _fill_value = np.nan

    @property
    def ndim(self) -> Literal[1]:
        return 1

    # To make mypy recognize the fields
    _left: IntervalSide
    _right: IntervalSide
    _dtype: IntervalDtype

    # ---------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data: ArrayLike,
        closed: IntervalClosedType | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        verify_integrity: bool = True,
    ) -> Self:
        data = extract_array(data, extract_numpy=True)

        if isinstance(data, cls):
            left: IntervalSide = data._left
            right: IntervalSide = data._right
            closed = closed or data.closed
            dtype = IntervalDtype(left.dtype, closed=closed)
        else:
            # don't allow scalars
            if is_scalar(data):
                msg = (
                    f"{cls.__name__}(...) must be called with a collection "
                    f"of some kind, {data} was passed"
                )
                raise TypeError(msg)

            # might need to convert empty or purely na data
            data = _maybe_convert_platform_interval(data)
            left, right, infer_closed = intervals_to_interval_bounds(
                data, validate_closed=closed is None
            )
            if left.dtype == object:
                left = lib.maybe_convert_objects(left)
                right = lib.maybe_convert_objects(right)
            closed = closed or infer_closed

            left, right, dtype = cls._ensure_simple_new_inputs(
                left,
                right,
                closed=closed,
                copy=copy,
                dtype=dtype,
            )

        if verify_integrity:
            cls._validate(left, right, dtype=dtype)

        return cls._simple_new(
            left,
            right,
            dtype=dtype,
        )

    @classmethod
    def _simple_new(
        cls,
        left: IntervalSide,
        right: IntervalSide,
        dtype: IntervalDtype,
    ) -> Self:
        result = IntervalMixin.__new__(cls)
        result._left = left
        result._right = right
        result._dtype = dtype

        return result

    @classmethod
    def _ensure_simple_new_inputs(
        cls,
        left: ArrayLike,
        right: ArrayLike,
        closed: IntervalClosedType | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> tuple[IntervalSide, IntervalSide, IntervalDtype]:
        """Ensure correctness of input parameters for cls._simple_new."""
        from pandas.core.indexes.base import ensure_index

        left = ensure_index(left, copy=copy)
        left = maybe_upcast_numeric_to_64bit(left)

        right = ensure_index(right, copy=copy)
        right = maybe_upcast_numeric_to_64bit(right)

        if closed is None and isinstance(dtype, IntervalDtype):
            closed = dtype.closed

        closed = closed or "right"

        if dtype is not None:
            # GH 19262: dtype must be an IntervalDtype to override inferred
            dtype = pandas_dtype(dtype)
            if isinstance(dtype, IntervalDtype):
                if dtype.subtype is not None:
                    left = left.astype(dtype.subtype)
                    right = right.astype(dtype.subtype)
            else:
                msg = f"dtype must be an IntervalDtype, got {dtype}"
                raise TypeError(msg)

            if dtype.closed is None:
                # possibly loading an old pickle
                dtype = IntervalDtype(dtype.subtype, closed)
            elif closed != dtype.closed:
                raise ValueError("closed keyword does not match dtype.closed")

        # coerce dtypes to match if needed
        if is_float_dtype(left.dtype) and is_integer_dtype(right.dtype):
            right = right.astype(left.dtype)
        elif is_float_dtype(right.dtype) and is_integer_dtype(left.dtype):
            left = left.astype(right.dtype)

        if type(left) != type(right):
            msg = (
                f"must not have differing left [{type(left).__name__}] and "
                f"right [{type(right).__name__}] types"
            )
            raise ValueError(msg)
        if isinstance(left.dtype, CategoricalDtype) or is_string_dtype(left.dtype):
            # GH 19016
            msg = (
                "category, object, and string subtypes are not supported "
                "for IntervalArray"
            )
            raise TypeError(msg)
        if isinstance(left, ABCPeriodIndex):
            msg = "Period dtypes are not supported, use a PeriodIndex instead"
            raise ValueError(msg)
        if isinstance(left, ABCDatetimeIndex) and str(left.tz) != str(right.tz):
            msg = (
                "left and right must have the same time zone, got "
                f"'{left.tz}' and '{right.tz}'"
            )
            raise ValueError(msg)
        elif needs_i8_conversion(left.dtype) and left.unit != right.unit:
            # e.g. m8[s] vs m8[ms], try to cast to a common dtype GH#55714
            left_arr, right_arr = left._data._ensure_matching_resos(right._data)
            left = ensure_index(left_arr)
            right = ensure_index(right_arr)

        # For dt64/td64 we want DatetimeArray/TimedeltaArray instead of ndarray
        left = ensure_wrapped_if_datetimelike(left)
        left = extract_array(left, extract_numpy=True)
        right = ensure_wrapped_if_datetimelike(right)
        right = extract_array(right, extract_numpy=True)

        if isinstance(left, ArrowExtensionArray) or isinstance(
            right, ArrowExtensionArray
        ):
            pass
        else:
            lbase = getattr(left, "_ndarray", left)
            lbase = getattr(lbase, "_data", lbase).base
            rbase = getattr(right, "_ndarray", right)
            rbase = getattr(rbase, "_data", rbase).base
            if lbase is not None and lbase is rbase:
                # If these share data, then setitem could corrupt our IA
                right = right.copy()

        dtype = IntervalDtype(left.dtype, closed=closed)

        return left, right, dtype

    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Interval],
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> Self:
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: IntervalArray) -> Self:
        return cls._from_sequence(values, dtype=original.dtype)

    _interval_shared_docs["from_breaks"] = textwrap.dedent(
        """
        Construct an %(klass)s from an array of splits.

        Parameters
        ----------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            Copy the data.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_arrays : Construct from a left and right array.
        %(klass)s.from_tuples : Construct from a sequence of tuples.

        %(examples)s\
        """
    )

    @classmethod
    @Appender(
        _interval_shared_docs["from_breaks"]
        % {
            "klass": "IntervalArray",
            "name": "",
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_breaks(
        cls,
        breaks: ArrayLike,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        breaks = _maybe_convert_platform_interval(breaks)

        return cls.from_arrays(breaks[:-1], breaks[1:], closed, copy=copy, dtype=dtype)

    _interval_shared_docs["from_arrays"] = textwrap.dedent(
        """
        Construct from two arrays defining the left and right bounds.

        Parameters
        ----------
        left : array-like (1-dimensional)
            Left bounds for each interval.
        right : array-like (1-dimensional)
            Right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            Copy the data.
        dtype : dtype, optional
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        Raises
        ------
        ValueError
            When a value is missing in only one of `left` or `right`.
            When a value in `left` is greater than the corresponding value
            in `right`.

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_breaks : Construct an %(klass)s from an array of
            splits.
        %(klass)s.from_tuples : Construct an %(klass)s from an
            array-like of tuples.

        Notes
        -----
        Each element of `left` must be less than or equal to the `right`
        element at the same position. If an element is missing, it must be
        missing in both `left` and `right`. A TypeError is raised when
        using an unsupported type for `left` or `right`. At the moment,
        'category', 'object', and 'string' subtypes are not supported.

        %(examples)s\
        """
    )

    @classmethod
    @Appender(
        _interval_shared_docs["from_arrays"]
        % {
            "klass": "IntervalArray",
            "name": "",
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.arrays.IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_arrays(
        cls,
        left: ArrayLike,
        right: ArrayLike,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        left = _maybe_convert_platform_interval(left)
        right = _maybe_convert_platform_interval(right)

        left, right, dtype = cls._ensure_simple_new_inputs(
            left,
            right,
            closed=closed,
            copy=copy,
            dtype=dtype,
        )
        cls._validate(left, right, dtype=dtype)

        return cls._simple_new(left, right, dtype=dtype)

    _interval_shared_docs["from_tuples"] = textwrap.dedent(
        """
        Construct an %(klass)s from an array-like of tuples.

        Parameters
        ----------
        data : array-like (1-dimensional)
            Array of tuples.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            By-default copy the data, this is compat only and ignored.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_arrays : Construct an %(klass)s from a left and
                                    right array.
        %(klass)s.from_breaks : Construct an %(kl