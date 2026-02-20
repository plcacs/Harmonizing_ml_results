"""define the IntervalIndex"""

from __future__ import annotations

from operator import (
    le,
    lt,
)
import textwrap
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.interval import (
    Interval,
    IntervalMixin,
    IntervalTree,
)
from pandas._libs.tslibs import (
    BaseOffset,
    Period,
    Timedelta,
    Timestamp,
    to_offset,
)
from pandas.errors import InvalidIndexError
from pandas.util._decorators import (
    Appender,
    cache_readonly,
    set_module,
)
from pandas.util._exceptions import rewrite_exception

from pandas.core.dtypes.cast import (
    find_common_type,
    infer_dtype_from_scalar,
    maybe_box_datetimelike,
    maybe_downcast_numeric,
    maybe_upcast_numeric_to_64bit,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_number,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
)
from pandas.core.dtypes.missing import is_valid_na_for_dtype

from pandas.core.algorithms import unique
from pandas.core.arrays.datetimelike import validate_periods
from pandas.core.arrays.interval import (
    IntervalArray,
    _interval_shared_docs,
)
import pandas.core.common as com
from pandas.core.indexers import is_valid_positional_slice
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    _index_shared_docs,
    ensure_index,
    maybe_extract_name,
)
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    date_range,
)
from pandas.core.indexes.extension import (
    ExtensionIndex,
    inherit_names,
)
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.timedeltas import (
    TimedeltaIndex,
    timedelta_range,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        IntervalClosedType,
        Self,
        npt,
    )
_index_doc_kwargs = dict(ibase._index_doc_kwargs)

_index_doc_kwargs.update(
    {
        "klass": "IntervalIndex",
        "qualname": "IntervalIndex",
        "target_klass": "IntervalIndex or list of Intervals",
        "name": textwrap.dedent(
            """\
         name : object, optional
              Name to be stored in the index.
         """
        ),
    }
)


def _get_next_label(label: Any) -> Any:
    # see test_slice_locs_with_ints_and_floats_succeeds
    dtype = getattr(label, "dtype", type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = "datetime64[ns]"
    dtype = pandas_dtype(dtype)

    if lib.is_np_dtype(dtype, "mM") or isinstance(dtype, DatetimeTZDtype):
        return label + np.timedelta64(1, "ns")
    elif is_integer_dtype(dtype):
        return label + 1
    elif is_float_dtype(dtype):
        return np.nextafter(label, np.inf)
    else:
        raise TypeError(f"cannot determine next label for type {type(label)!r}")


def _get_prev_label(label: Any) -> Any:
    # see test_slice_locs_with_ints_and_floats_succeeds
    dtype = getattr(label, "dtype", type(label))
    if isinstance(label, (Timestamp, Timedelta)):
        dtype = "datetime64[ns]"
    dtype = pandas_dtype(dtype)

    if lib.is_np_dtype(dtype, "mM") or isinstance(dtype, DatetimeTZDtype):
        return label - np.timedelta64(1, "ns")
    elif is_integer_dtype(dtype):
        return label - 1
    elif is_float_dtype(dtype):
        return np.nextafter(label, -np.inf)
    else:
        raise TypeError(f"cannot determine next label for type {type(label)!r}")


def _new_IntervalIndex(cls: type[IntervalIndex], d: dict[str, Any]) -> IntervalIndex:
    """
    This is called upon unpickling, rather than the default which doesn't have
    arguments and breaks __new__.
    """
    return cls.from_arrays(**d)


@Appender(
    _interval_shared_docs["class"]
    % {
        "klass": "IntervalIndex",
        "summary": "Immutable index of intervals that are closed on the same side.",
        "name": _index_doc_kwargs["name"],
        "extra_attributes": "is_overlapping\nvalues\n",
        "extra_methods": "",
        "examples": textwrap.dedent(
            """\
    Examples
    --------
    A new ``IntervalIndex`` is typically constructed using
    :func:`interval_range`:

    >>> pd.interval_range(start=0, end=5)
    IntervalIndex([(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]],
                  dtype='interval[int64, right]')

    It may also be constructed using one of the constructor
    methods: :meth:`IntervalIndex.from_arrays`,
    :meth:`IntervalIndex.from_breaks`, and :meth:`IntervalIndex.from_tuples`.

    See further examples in the doc strings of ``interval_range`` and the
    mentioned constructor methods.
    """
        ),
    }
)
@inherit_names(["set_closed", "to_tuples"], IntervalArray, wrap=True)
@inherit_names(
    [
        "__array__",
        "overlaps",
        "contains",
        "closed_left",
        "closed_right",
        "open_left",
        "open_right",
        "is_empty",
    ],
    IntervalArray,
)
@inherit_names(["is_non_overlapping_monotonic", "closed"], IntervalArray, cache=True)
@set_module("pandas")
class IntervalIndex(ExtensionIndex):
    _typ = "intervalindex"

    # annotate properties pinned via inherit_names
    closed: IntervalClosedType
    is_non_overlapping_monotonic: bool
    closed_left: bool
    closed_right: bool
    open_left: bool
    open_right: bool

    _data: IntervalArray
    _values: IntervalArray
    _can_hold_strings = False
    _data_cls = IntervalArray

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data: Any,
        closed: IntervalClosedType | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
        verify_integrity: bool = True,
    ) -> Self:
        name = maybe_extract_name(name, data, cls)

        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray(
                data,
                closed=closed,
                copy=copy,
                dtype=dtype,
                verify_integrity=verify_integrity,
            )

        return cls._simple_new(array, name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_breaks"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
             name : str, optional
                  Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.IntervalIndex.from_breaks([0, 1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_breaks(
        cls,
        breaks: Any,
        closed: IntervalClosedType | None = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray.from_breaks(
                breaks, closed=closed, copy=copy, dtype=dtype
            )
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_arrays"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
             name : str, optional
                  Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.IntervalIndex.from_arrays([0, 1, 2], [1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                      dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_arrays(
        cls,
        left: Any,
        right: Any,
        closed: IntervalClosedType = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        with rewrite_exception("IntervalArray", cls.__name__):
            array = IntervalArray.from_arrays(
                left, right, closed, copy=copy, dtype=dtype
            )
        return cls._simple_new(array, name=name)

    @classmethod
    @Appender(
        _interval_shared_docs["from_tuples"]
        % {
            "klass": "IntervalIndex",
            "name": textwrap.dedent(
                """
             name : str, optional
                  Name of the resulting IntervalIndex."""
            ),
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.IntervalIndex.from_tuples([(0, 1), (1, 2)])
        IntervalIndex([(0, 1], (1, 2]],
                       dtype='interval[int64, right]')
        """
            ),
        }
    )
    def from_tuples(
        cls,
        data: Any,
        closed: IntervalClosedType = "right",
        name: Hashable | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> IntervalIndex:
        with rewrite_exception("IntervalArray", cls.__name__):
            arr = IntervalArray.from_tuples(data, closed=closed, copy=copy, dtype=dtype)
        return cls._simple_new(arr, name=name)

    # --------------------------------------------------------------------
    # error: Return type "IntervalTree" of "_engine" incompatible with return type
    # "Union[IndexEngine, ExtensionEngine]" in supertype "Index"
    @cache_readonly
    def _engine(self) -> IntervalTree:  # type: ignore[override]
        # IntervalTree does not supports numpy array unless they are 64 bit
        left = self._maybe_convert_i8(self.left)
        left = maybe_upcast_numeric_to_64bit(left)
        right = self._maybe_convert_i8(self.right)
        right = maybe_upcast_numeric_to_64bit(right)
        return IntervalTree(left, right, closed=self.closed)

    def __contains__(self, key: Any) -> bool:
        """
        return a boolean if this key is IN the index
        We *only* accept an Interval

        Parameters
        ----------
        key : Interval

        Returns
        -------
        bool
        """
        hash(key)
        if not isinstance(key, Interval):
            if is_valid_na_for_dtype(key, self.dtype):
                return self.hasnans
            return False

        try:
            self.get_loc(key)
            return True
        except KeyError:
            return False

    def _getitem_slice(self, slobj: slice) -> IntervalIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._data[slobj]
        return type(self)._simple_new(res, name=self._name)

    @cache_readonly
    def _multiindex(self) -> MultiIndex:
        return MultiIndex.from_arrays([self.left, self.right], names=["left", "right"])

    def __reduce__(self) -> tuple[Any, ...]:
        d = {
            "left": self.left,
            "right": self.right,
            "closed": self.closed,
            "name": self.name,
        }
        return _new_IntervalIndex, (type(self), d), None

    @property
    def inferred_type(self) -> str:
        """Return a string of the type inferred from the values"""
        return "interval"

    # Cannot determine type of "memory_usage"
    @Appender(Index.memory_usage.__doc__)  # type: ignore[has-type]
    def memory_usage(self, deep: bool = False) -> int:
        # we don't use an explicit engine
        # so return the bytes here
        return self.left.memory_usage(deep=deep) + self.right.memory_usage(deep=deep)

    # IntervalTree doesn't have a is_monotonic_decreasing, so have to override
    #  the Index implementation
    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        """
        Return True if the IntervalIndex is monotonic decreasing (only equal or
        decreasing values), else False
        """
        return self[::-1].is_monotonic_increasing

    @cache_readonly
    def is_unique(self) -> bool:
        """
        Return True if the IntervalIndex contains unique elements, else False.
        """
        left = self.left
        right = self.right

        if self.isna().sum() > 1:
            return False

        if left.is_unique or right.is_unique:
            return True

        seen_pairs = set()
        check_idx = np.where(left.duplicated(keep=False))[0]
        for idx in check_idx:
            pair = (left[idx], right[idx])
            if pair in seen_pairs:
                return False
            seen_pairs.add(pair)

        return True

    @property
    def is_overlapping(self) -> bool:
        """
        Return True if the IntervalIndex has overlapping intervals, else False.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Returns
        -------
        bool
            Boolean indicating if the IntervalIndex has overlapping intervals.

        See Also
        --------
        Interval.overlaps : Check whether two Interval objects overlap.
        IntervalIndex.overlaps : Check an IntervalIndex elementwise for
            overlaps.

        Examples
        --------
        >>> index = pd.IntervalIndex.from_tuples([(0, 2), (1, 3), (4, 5)])
        >>> index
        IntervalIndex([(0, 2], (1, 3], (4, 5]],
              dtype='interval[int64, right]')
        >>> index.is_overlapping
        True

        Intervals that share closed endpoints overlap:

        >>> index = pd.interval_range(0, 3, closed="both")
        >>> index
        IntervalIndex([[0, 1], [1, 2], [2, 3]],
              dtype='interval[int64, both]')
        >>> index.is_overlapping
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> index = pd.interval_range(0, 3, closed="left")
        >>> index
        IntervalIndex([[0, 1), [1, 2), [2, 3)],
              dtype='interval[int64, left]')
        >>> index.is_overlapping
        False
        """
        # GH 23309
        return self._engine.is_overlapping

    def _needs_i8_conversion(self, key: Any) -> bool:
        """
        Check if a given key needs i8 conversion. Conversion is necessary for
        Timestamp, Timedelta, DatetimeIndex, and TimedeltaIndex keys. An
        Interval-like requires conversion if its endpoints are one of the
        aforementioned types.

        Assumes that any list-like data has already been cast to an Index.

        Parameters
        ----------
        key : scalar or Index-like
            The key that should be checked for i8 conversion

        Returns
        -------
        bool
        """
        key_dtype = getattr(key, "dtype", None)
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            return self._needs_i8_conversion(key.left)

        i8_types = (Timestamp, Timedelta, DatetimeIndex, TimedeltaIndex)
        return isinstance(key, i8_types)

    def _maybe_convert_i8(self, key: Any) -> Any:
        """
        Maybe convert a given key to its equivalent i8 value(s). Used as a
        preprocessing step prior to IntervalTree queries (self._engine), which
        expects numeric data.

        Parameters
        ----------
        key : scalar or list-like
            The key that should maybe be converted to i8.

        Returns
        -------
        scalar or list-like
            The original key if no conversion occurred, int if converted scalar,
            Index with an int64 dtype if converted list-like.
        """
        if is_list_like(key):
            key = ensure_index(key)
            key = maybe_upcast_numeric_to_64bit(key)

        if not self._needs_i8_conversion(key):
            return key

        scalar = is_scalar(key)
        key_dtype = getattr(key, "dtype", None)
        if isinstance(key_dtype, IntervalDtype) or isinstance(key, Interval):
            # convert left/right and reconstruct
            left = self._maybe_convert_i8(key.left)
            right = self._maybe_convert_i8(key.right)
            constructor = Interval if scalar else IntervalIndex.from_arrays
            return constructor(left, right, closed=self.closed)

        if scalar:
            # Timestamp/Timedelta
            key_dtype, key_i8 = infer_dtype_from_scalar(key)
            if isinstance(key, Period):
                key_i8 = key.ordinal
            elif isinstance(key_i8,