from __future__ import annotations
from collections.abc import Callable, Hashable, Iterator
from datetime import timedelta
import operator
from sys import getsizeof
from typing import TYPE_CHECKING, Any, Literal, cast, overload, Optional, Tuple, Union, Sequence
import numpy as np
from pandas._libs import index as libindex, lib
from pandas._libs.lib import no_default
from pandas.compat.numpy import function as nv
from pandas.util._decorators import cache_readonly, doc, set_module
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    ensure_platform_int,
    ensure_python_int,
    is_float,
    is_integer,
    is_scalar,
    is_signed_integer_dtype,
)
from pandas.core.dtypes.generic import ABCTimedeltaIndex
from pandas.core import ops
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import Index, maybe_extract_name
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        Dtype,
        JoinHow,
        NaPosition,
        NumpySorter,
        Self,
        npt,
    )
    from pandas import Series

_empty_range: range = range(0)
_dtype_int64: np.dtype = np.dtype(np.int64)


def func_xv7l2ds2(start: int, step: int, lower_limit: int) -> int:
    """Returns the smallest element greater than or equal to the limit"""
    no_steps = -(-(lower_limit - start) // abs(step))
    return start + abs(step) * no_steps


@set_module("pandas")
class RangeIndex(Index):
    """
    Immutable Index implementing a monotonic integer range.

    RangeIndex is a memory-saving special case of an Index limited to representing
    monotonic ranges with a 64-bit dtype. Using RangeIndex may in some instances
    improve computing speed.

    This is the default index type used
    by DataFrame and Series when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), range, or other RangeIndex instance
        If int and "stop" is not given, interpreted as "stop" instead.
    stop : int (default: 0)
        The end value of the range (exclusive).
    step : int (default: 1)
        The step size of the range.
    dtype : np.int64
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    from_range

    See Also
    --------
    Index : The base pandas Index type.

    Examples
    --------
    >>> list(pd.RangeIndex(5))
    [0, 1, 2, 3, 4]

    >>> list(pd.RangeIndex(-2, 4))
    [-2, -1, 0, 1, 2, 3]

    >>> list(pd.RangeIndex(0, 10, 2))
    [0, 2, 4, 6, 8]

    >>> list(pd.RangeIndex(2, -10, -3))
    [2, -1, -4, -7]

    >>> list(pd.RangeIndex(0))
    []

    >>> list(pd.RangeIndex(1, 0))
    []
    """

    _typ: str = "rangeindex"
    _dtype_validation_metadata: Tuple[Callable[[Any], bool], str] = (
        is_signed_integer_dtype,
        "signed integer",
    )

    @property
    def func_qo2p6n3q(self) -> Any:
        return libindex.Int64Engine

    def __new__(
        cls,
        start: Optional[int | range | RangeIndex] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        dtype: Optional[Dtype] = None,
        copy: bool = False,
        name: Any = None,
    ) -> RangeIndex:
        cls._validate_dtype(dtype)
        name = maybe_extract_name(name, start, cls)
        if isinstance(start, cls):
            return cast(RangeIndex, start.copy(name=name))
        elif isinstance(start, range):
            return cls._simple_new(start, name=name)
        if com.all_none(start, stop, step):
            raise TypeError("RangeIndex(...) must be called with integers")
        start = ensure_python_int(start) if start is not None else 0
        if stop is None:
            start, stop = 0, start
        else:
            stop = ensure_python_int(stop)
        step = ensure_python_int(step) if step is not None else 1
        if step == 0:
            raise ValueError("Step must not be zero")
        rng = range(start, stop, step)
        return cls._simple_new(rng, name=name)

    @classmethod
    def func_b9w0cc54(
        cls, data: range, name: Optional[str] = None, dtype: Optional[Dtype] = None
    ) -> RangeIndex:
        """
        Create :class:`pandas.RangeIndex` from a ``range`` object.

        This method provides a way to create a :class:`pandas.RangeIndex` directly
        from a Python ``range`` object. The resulting :class:`RangeIndex` will have
        the same start, stop, and step values as the input ``range`` object.
        It is particularly useful for constructing indices in an efficient and
        memory-friendly manner.

        Parameters
        ----------
        data : range
            The range object to be converted into a RangeIndex.
        name : str, default None
            Name to be stored in the index.
        dtype : Dtype or None
            Data type for the RangeIndex. If None, the default integer type will
            be used.

        Returns
        -------
        RangeIndex

        See Also
        --------
        RangeIndex : Immutable Index implementing a monotonic integer range.
        Index : Immutable sequence used for indexing and alignment.

        Examples
        --------
        >>> pd.RangeIndex.from_range(range(5))
        RangeIndex(start=0, stop=5, step=1)

        >>> pd.RangeIndex.from_range(range(2, -10, -3))
        RangeIndex(start=2, stop=-10, step=-3)
        """
        if not isinstance(data, range):
            raise TypeError(
                f"{cls.__name__}(...) must be called with object coercible to a range, {data!r} was passed"
            )
        cls._validate_dtype(dtype)
        return cls._simple_new(data, name=name)

    @classmethod
    def func_v4cy5urz(cls, values: range, name: Optional[Any] = None) -> RangeIndex:
        result = object.__new__(cls)
        assert isinstance(values, range)
        result._range: range = values
        result._name: Optional[Any] = name
        result._cache: dict[str, Any] = {}
        result._reset_identity()
        result._references: Optional[Any] = None
        return result

    @classmethod
    def func_ozaxs0wh(cls, dtype: Optional[Dtype]) -> None:
        if dtype is None:
            return
        validation_func, expected = cls._dtype_validation_metadata
        if not validation_func(dtype):
            raise ValueError(
                f"Incorrect `dtype` passed: expected {expected}, received {dtype}"
            )

    @cache_readonly
    def func_k6xrb8l3(self) -> type[Index]:
        """return the class to use for construction"""
        return Index

    @cache_readonly
    def func_lnwqokb9(self) -> np.ndarray:
        """
        An int array that for performance reasons is created only when needed.

        The constructed array is saved in ``_cache``.
        """
        return np.arange(self.start, self.stop, self.step, dtype=np.int64)

    def func_zkz8x158(self) -> list[tuple[str, int]]:
        """return a list of tuples of start, stop, step"""
        rng = self._range
        return [("start", rng.start), ("stop", rng.stop), ("step", rng.step)]

    def __reduce__(
        self,
    ) -> tuple[type, tuple[Any, ...], Any]:
        d: dict[str, Any] = {"name": self._name}
        d.update(dict(self._get_data_as_items()))
        return ibase._new_Index, (type(self), d), None

    def func_2lva6885(self) -> list[tuple[str, Union[str, int]]]:
        """
        Return a list of tuples of the (attr, formatted_value)
        """
        attrs = cast(
            "list[tuple[str, str | int]]", self._get_data_as_items()
        )
        if self._name is not None:
            attrs.append(("name", ibase.default_pprint(self._name)))
        return attrs

    def func_b66ad9ja(
        self, *, header: list[str], na_rep: Any
    ) -> list[str]:
        if not len(self._range):
            return header
        first_val_str = str(self._range[0])
        last_val_str = str(self._range[-1])
        max_length = max(len(first_val_str), len(last_val_str))
        return header + [f"{x:<{max_length}}" for x in self._range]

    @property
    def func_vmfi9eb6(self) -> int:
        """
        The value of the `start` parameter (``0`` if this was not supplied).

        This property returns the starting value of the `RangeIndex`. If the `start`
        value is not explicitly provided during the creation of the `RangeIndex`,
        it defaults to 0.

        See Also
        --------
        RangeIndex : Immutable index implementing a range-based index.
        RangeIndex.stop : Returns the stop value of the `RangeIndex`.
        RangeIndex.step : Returns the step value of the `RangeIndex`.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.start
        0

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.start
        2
        """
        return self._range.start

    @property
    def func_kof44mk0(self) -> int:
        """
        The value of the `stop` parameter.

        This property returns the `stop` value of the RangeIndex, which defines the
        upper (or lower, in case of negative steps) bound of the index range. The
        `stop` value is exclusive, meaning the RangeIndex includes values up to but
        not including this value.

        See Also
        --------
        RangeIndex : Immutable index representing a range of integers.
        RangeIndex.start : The start value of the RangeIndex.
        RangeIndex.step : The step value of the RangeIndex.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.stop
        5

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.stop
        -10
        """
        return self._range.stop

    @property
    def func_nqsvyud5(self) -> int:
        """
        The value of the `step` parameter (``1`` if this was not supplied).

        The ``step`` parameter determines the increment (or decrement in the case
        of negative values) between consecutive elements in the ``RangeIndex``.

        See Also
        --------
        RangeIndex : Immutable index implementing a range-based index.
        RangeIndex.stop : Returns the stop value of the RangeIndex.
        RangeIndex.start : Returns the start value of the RangeIndex.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.step
        1

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.step
        -3

        Even if :class:`pandas.RangeIndex` is empty, ``step`` is still ``1`` if
        not supplied.

        >>> idx = pd.RangeIndex(1, 0)
        >>> idx.step
        1
        """
        return self._range.step

    @cache_readonly
    def func_ck4bp72u(self) -> int:
        """
        Return the number of bytes in the underlying data.
        """
        rng = self._range
        return getsizeof(rng) + sum(
            getsizeof(getattr(rng, attr_name))
            for attr_name in ["start", "stop", "step"]
        )

    def func_tpzitmk1(self, deep: bool = False) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
        return self.nbytes

    @property
    def func_xxoouu58(self) -> np.dtype:
        return _dtype_int64

    @property
    def func_8aykflsd(self) -> bool:
        """return if the index has unique values"""
        return True

    @cache_readonly
    def func_zfa7gsm6(self) -> bool:
        return self._range.step > 0 or len(self) <= 1

    @cache_readonly
    def func_4aj3ej2w(self) -> bool:
        return self._range.step < 0 or len(self) <= 1

    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            key = ensure_python_int(key)
        except (TypeError, OverflowError):
            return False
        return key in self._range

    @property
    def func_tc5ienvr(self) -> str:
        return "integer"

    @doc(Index.get_loc)
    def func_ijz992sg(self, key: Any) -> int:
        if is_integer(key) or (is_float(key) and key.is_integer()):
            new_key = int(key)
            try:
                return self._range.index(new_key)
            except ValueError as err:
                raise KeyError(key) from err
        if isinstance(key, Hashable):
            raise KeyError(key)
        self._check_indexing_error(key)
        raise KeyError(key)

    def func_h9bta2fp(
        self,
        target: Any,
        method: Optional[str] = None,
        limit: Optional[int] = None,
        tolerance: Optional[Any] = None,
    ) -> np.ndarray:
        if com.any_not_none(method, tolerance, limit):
            return super()._get_indexer(
                target,
                method=method,
                tolerance=tolerance,
                limit=limit,
            )
        if self.step > 0:
            start, stop, step = self.start, self.stop, self.step
        else:
            reverse = self._range[::-1]
            start, stop, step = reverse.start, reverse.stop, reverse.step
        target_array = np.asarray(target)
        locs = target_array - start
        valid = (locs % step == 0) & (locs >= 0) & (target_array < stop)
        locs[~valid] = -1
        locs[valid] = locs[valid] / step
        if step != self.step:
            locs[valid] = len(self) - 1 - locs[valid]
        return ensure_platform_int(locs)

    @cache_readonly
    def func_9j6r72kq(self) -> bool:
        """
        Should an integer key be treated as positional?
        """
        return False

    def func_eotijzic(self) -> list[int]:
        return list(self._range)

    @doc(Index.__iter__)
    def __iter__(self) -> Iterator[int]:
        yield from self._range

    @doc(Index._shallow_copy)
    def func_191kqb9e(
        self, values: Any, name: Any = no_default
    ) -> Index:
        name = self._name if name is no_default else name
        if values.dtype.kind == "f":
            return Index(values, name=name, dtype=np.float64)
        if values.dtype.kind == "i" and values.ndim == 1:
            if len(values) == 1:
                start = values[0]
                new_range = range(start, start + self.step, self.step)
                return type(self)._simple_new(new_range, name=name)
            maybe_range = ibase.maybe_sequence_to_range(values)
            if isinstance(maybe_range, range):
                return type(self)._simple_new(maybe_range, name=name)
        return self._constructor._simple_new(values, name=name)

    def func_7ho7ns1c(self) -> RangeIndex:
        result = type(self)._simple_new(self._range, name=self._name)
        result._cache = self._cache
        return result

    def func_p1naumwt(
        self, target: Any, indexer: Any, preserve_names: bool
    ) -> Any:
        if not isinstance(target, type(self)) and target.dtype.kind == "i":
            target = self._shallow_copy(target._values, name=target.name)
        return super()._wrap_reindex_result(target, indexer, preserve_names)

    @doc(Index.copy)
    def func_4yt6obqw(
        self, name: Optional[Any] = None, deep: bool = False
    ) -> RangeIndex:
        name = self._validate_names(name=name, deep=deep)[0]
        new_index = self._rename(name=name)
        return new_index

    def func_nq81ezbn(self, meth: str) -> Union[int, float]:
        no_steps = len(self) - 1
        if no_steps == -1:
            return np.nan
        elif (meth == "min" and self.step > 0) or (meth == "max" and self.step < 0):
            return self.start
        return self.start + self.step * no_steps

    def min(
        self,
        axis: Optional[int] = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """The minimum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_min(args, kwargs)
        return self._minmax("min")

    def max(
        self,
        axis: Optional[int] = None,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """The maximum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_max(args, kwargs)
        return self._minmax("max")

    def func_0zhepest(
        self, meth: str, axis: Optional[int] = None, skipna: bool = True
    ) -> Union[int, float]:
        nv.validate_minmax_axis(axis)
        if len(self) == 0:
            return getattr(super(), f"arg{meth}")(
                axis=axis, skipna=skipna
            )
        elif meth == "min":
            if self.step > 0:
                return self.start
            else:
                return len(self) - 1
        elif meth == "max":
            if self.step > 0:
                return len(self) - 1
            else:
                return 0
        else:
            raise ValueError(f'meth={meth!r} must be max or min')

    def func_581pjgpk(
        self, axis: Optional[int] = None, skipna: bool = True, *args: Any, **kwargs: Any
    ) -> int:
        nv.validate_argmin(args, kwargs)
        return self._argminmax("min", axis=axis, skipna=skipna)

    def func_ijo8sk1e(
        self, axis: Optional[int] = None, skipna: bool = True, *args: Any, **kwargs: Any
    ) -> int:
        nv.validate_argmax(args, kwargs)
        return self._argminmax("max", axis=axis, skipna=skipna)

    def func_opsle51q(
        self, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Returns the indices that would sort the index and its
        underlying data.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort
        """
        ascending: bool = kwargs.pop("ascending", True)
        kwargs.pop("kind", None)
        nv.validate_argsort(args, kwargs)
        start: Optional[int] = None
        stop: Optional[int] = None
        step: Optional[int] = None
        if self._range.step > 0:
            if ascending:
                start = len(self)
            else:
                start, stop, step = len(self) - 1, -1, -1
        elif ascending:
            start, stop, step = len(self) - 1, -1, -1
        else:
            start = len(self)
        return np.arange(start, stop, step, dtype=np.intp)

    def func_3nryq5ip(
        self, sort: bool = False, use_na_sentinel: bool = True
    ) -> Tuple[np.ndarray, RangeIndex]:
        if sort and self.step < 0:
            codes = np.arange(len(self) - 1, -1, -1, dtype=np.intp)
            uniques = self[::-1]
        else:
            codes = np.arange(len(self), dtype=np.intp)
            uniques = self
        return codes, uniques

    def func_xytt2h8o(self, other: Any) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        if isinstance(other, RangeIndex):
            return self._range == other._range
        return super().equals(other)

    @overload
    def func_goql6ajk(
        self,
        *,
        return_indexer: Literal[True],
        ascending: bool = ...,
        na_position: Literal["last"],
        key: Any = ...,
    ) -> Tuple[RangeIndex, RangeIndex]:
        ...

    @overload
    def func_goql6ajk(
        self,
        *,
        return_indexer: Literal[False],
        ascending: bool = ...,
        na_position: Literal["last"],
        key: Any = ...,
    ) -> RangeIndex:
        ...

    @overload
    def func_goql6ajk(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = ...,
        na_position: Literal["last"],
        key: Any = ...,
    ) -> Union[RangeIndex, Tuple[RangeIndex, RangeIndex]]:
        ...

    def func_goql6ajk(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: Literal["last"] = "last",
        key: Any = None,
    ) -> Union[RangeIndex, Tuple[RangeIndex, RangeIndex]]:
        if key is not None:
            return super().sort_values(
                return_indexer=return_indexer,
                ascending=ascending,
                na_position=na_position,
                key=key,
            )
        else:
            sorted_index: RangeIndex = self
            inverse_indexer: bool = False
            if ascending:
                if self.step < 0:
                    sorted_index = self[::-1]
                    inverse_indexer = True
            elif self.step > 0:
                sorted_index = self[::-1]
                inverse_indexer = True
        if return_indexer:
            if inverse_indexer:
                rng = range(len(self) - 1, -1, -1)
            else:
                rng = range(len(self))
            return sorted_index, RangeIndex(rng)
        else:
            return sorted_index

    def func_ffvyyiec(
        self, other: RangeIndex, sort: bool = False
    ) -> RangeIndex:
        if not isinstance(other, RangeIndex):
            return super()._intersection(other, sort=sort)
        first = self._range[::-1] if self.step < 0 else self._range
        second = other._range[::-1] if other.step < 0 else other._range
        int_low = max(first.start, second.start)
        int_high = min(first.stop, second.stop)
        if int_high <= int_low:
            return self._simple_new(_empty_range)
        gcd, s, _ = self._extended_gcd(first.step, second.step)
        if (first.start - second.start) % gcd:
            return self._simple_new(_empty_range)
        tmp_start = first.start + (second.start - first.start) * first.step // gcd * s
        new_step = first.step * second.step // gcd
        new_start = func_xv7l2ds2(tmp_start, new_step, int_low)
        new_range = range(new_start, int_high, new_step)
        if (self.step < 0 and other.step < 0) is not (new_range.step < 0):
            new_range = new_range[::-1]
        return self._simple_new(new_range)

    def func_99u294vg(
        self, a: int, b: int
    ) -> Tuple[int, int, int]:
        """
        Extended Euclidean algorithms to solve Bezout's identity:
           a*x + b*y = gcd(x, y)
        Finds one particular solution for x, y: s, t
        Returns: gcd, s, t
        """
        s, old_s = 0, 1
        t, old_t = 1, 0
        r, old_r = b, a
        while r:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t
        return old_r, old_s, old_t

    def func_6es4okil(self, other: RangeIndex) -> bool:
        """Check if other range is contained in self"""
        if not other:
            return True
        if not self._range:
            return False
        if len(other) > 1 and other.step % self._range.step:
            return False
        return other.start in self._range and other[-1] in self._range

    def func_n7fruewf(self, other: Any, sort: bool) -> Index:
        """
        Form the union of two Index objects and sorts if possible

        Parameters
        ----------
        other : Index or array-like

        sort : bool or None, default None
            Whether to sort (monotonically increasing) the resulting index.
            ``sort=None|True`` returns a ``RangeIndex`` if possible or a sorted
            ``Index`` with a int64 dtype if not.
            ``sort=False`` can return a ``RangeIndex`` if self is monotonically
            increasing and other is fully contained in self. Otherwise, returns
            an unsorted ``Index`` with a int64 dtype.

        Returns
        -------
        union : Index
        """
        if isinstance(other, RangeIndex):
            if sort in (None, True) or (
                sort is False and self.step > 0 and self._range_in_self(other._range)
            ):
                start_s, step_s = self.start, self.step
                end_s = self.start + self.step * (len(self) - 1)
                start_o, step_o = other.start, other.step
                end_o = other.start + other.step * (len(other) - 1)
                if self.step < 0:
                    start_s, step_s, end_s = end_s, -step_s, start_s
                if other.step < 0:
                    start_o, step_o, end_o = end_o, -step_o, start_o
                if len(self) == 1 and len(other) == 1:
                    step_s = step_o = abs(self.start - other.start)
                elif len(self) == 1:
                    step_s = step_o
                elif len(other) == 1:
                    step_o = step_s
                start_r = min(start_s, start_o)
                end_r = max(end_s, end_o)
                if step_o == step_s:
                    if (
                        (start_s - start_o) % step_s == 0
                        and start_s - end_o <= step_s
                        and start_o - end_s <= step_s
                    ):
                        return type(self)(start_r, end_r + step_s, step_s)
                    if (
                        step_s % 2 == 0
                        and abs(start_s - start_o) == step_s / 2
                        and abs(end_s - end_o) == step_s / 2
                    ):
                        return type(self)(start_r, end_r + step_s / 2, step_s / 2)
                elif step_o % step_s == 0:
                    if (
                        (start_o - start_s) % step_s == 0
                        and start_o + step_s >= start_s
                        and end_o - step_s <= end_s
                    ):
                        return type(self)(start_r, end_r + step_s, step_s)
                elif step_s % step_o == 0:
                    if (
                        (start_s - start_o) % step_o == 0
                        and start_s + step_o >= start_o
                        and end_s - step_o <= end_o
                    ):
                        return type(self)(start_r, end_r + step_o, step_o)
        return super()._union(other, sort=sort)

    def func_0q34dh0f(
        self, other: Any, sort: Optional[bool] = None
    ) -> Index:
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)
        if not isinstance(other, RangeIndex):
            return super()._difference(other, sort=sort)
        if sort is not False and self.step < 0:
            return self[::-1]._difference(other)
        res_name: Any = ops.get_op_result_name(self, other)
        first = self._range[::-1] if self.step < 0 else self._range
        overlap = self.intersection(other)
        if overlap.step < 0:
            overlap = overlap[::-1]
        if len(overlap) == 0:
            return self.rename(name=res_name)
        if len(overlap) == len(self):
            return self[:0].rename(res_name)
        if len(overlap) == 1:
            if overlap[0] == self[0]:
                return self[1:]
            elif overlap[0] == self[-1]:
                return self[:-1]
            elif len(self) == 3 and overlap[0] == self[1]:
                return self[::2]
            else:
                return super()._difference(other, sort=sort)
        elif len(overlap) == 2 and overlap[0] == first.start and overlap[-1] == first.stop:
            return self[1:-1]
        if overlap.step == first.step:
            if overlap[0] == first.start:
                new_rng = range(overlap[-1] + first.step, first.stop, first.step)
            elif overlap[-1] == first[-1]:
                new_rng = range(first.start, overlap[0], first.step)
            elif overlap._range == first[1:-1]:
                step = len(first) - 1
                new_rng = first[::step]
            else:
                return super()._difference(other, sort=sort)
        else:
            assert len(self) > 1
            if overlap.step == first.step * 2:
                if overlap[0] == first[0] and overlap[-1] in (first[-1], first[-2]):
                    new_rng = first[1::2]
                elif overlap[0] == first[1] and overlap[-1] in (first[-1], first[-2]):
                    new_rng = first[::2]
                else:
                    return super()._difference(other, sort=sort)
            else:
                return super()._difference(other, sort=sort)
        if first is not self._range:
            new_rng = new_rng[::-1]
        new_index: RangeIndex = type(self)._simple_new(new_rng, name=res_name)
        return new_index

    def func_wg1jgb9j(
        self, other: Any, result_name: Optional[Any] = None, sort: Optional[bool] = None
    ) -> Index:
        if not isinstance(other, RangeIndex) or sort is not None:
            return super().symmetric_difference(other, result_name, sort)
        left = self.difference(other)
        right = other.difference(self)
        result = left.union(right)
        if result_name is not None:
            result = result.rename(result_name)
        return result

    def func_38t4lgkt(
        self, other: Any, how: str, sort: bool
    ) -> Tuple[Index, Optional[np.ndarray], Optional[np.ndarray]]:
        if not isinstance(other, RangeIndex) and other.dtype.kind == "i":
            other = self._shallow_copy(other._values, name=other.name)
        return super()._join_empty(other, how=how, sort=sort)

    def func_zmflgm6x(
        self,
        other: Any,
        how: Literal["left", "right", "inner", "outer"] = "left",
    ) -> Tuple[Index, Optional[np.ndarray], Optional[np.ndarray]]:
        if not isinstance(other, type(self)):
            maybe_ri = self._shallow_copy(other._values, name=other.name)
            if not isinstance(maybe_ri, type(self)):
                return super()._join_monotonic(other, how=how)
            other = maybe_ri
        if self.equals(other):
            ret_index: RangeIndex = other if how == "right" else self
            return ret_index, None, None
        if how == "left":
            join_index: RangeIndex = self
            lidx: Optional[np.ndarray] = None
            ridx: Optional[np.ndarray] = other.get_indexer(join_index)
        elif how == "right":
            join_index = other
            lidx = self.get_indexer(join_index)
            ridx = None
        elif how == "inner":
            join_index = self.intersection(other)
            lidx = self.get_indexer(join_index)
            ridx = other.get_indexer(join_index)
        elif how == "outer":
            join_index = self.union(other)
            lidx = self.get_indexer(join_index)
            ridx = other.get_indexer(join_index)
        lidx = None if lidx is None else ensure_platform_int(lidx)
        ridx = None if ridx is None else ensure_platform_int(ridx)
        return join_index, lidx, ridx

    def func_pntes61t(self, loc: Any) -> Index:
        if is_integer(loc):
            if loc in (0, -len(self)):
                return self[1:]
            if loc in (-1, len(self) - 1):
                return self[:-1]
            if len(self) == 3 and loc in (1, -2):
                return self[::2]
        elif lib.is_list_like(loc):
            slc = lib.maybe_indices_to_slice(
                np.asarray(loc, dtype=np.intp), len(self)
            )
            if isinstance(slc, slice):
                other = self[slc]
                return self.difference(other, sort=False)
        return super().delete(loc)

    def func_tc9j951o(
        self, loc: int, item: Union[int, float]
    ) -> Index:
        if is_integer(item) or is_float(item):
            if len(self) == 0 and loc == 0 and is_integer(item):
                new_rng = range(item, item + self.step, self.step)
                return type(self)._simple_new(new_rng, name=self._name)
            elif len(self):
                rng = self._range
                if loc == 0 and item == self[0] - self.step:
                    new_rng = range(rng.start - rng.step, rng.stop, rng.step)
                    return type(self)._simple_new(new_rng, name=self._name)
                elif loc == len(self) and item == self[-1] + self.step:
                    new_rng = range(rng.start, rng.stop + rng.step, rng.step)
                    return type(self)._simple_new(new_rng, name=self._name)
                elif len(self) == 2 and item == self[0] + self.step / 2:
                    step = int(self.step / 2)
                    new_rng = range(self.start, self.stop, step)
                    return type(self)._simple_new(new_rng, name=self._name)
        return super().insert(loc, item)

    def func_l6nyyx2l(
        self, indexes: list[Index], name: Any
    ) -> Index:
        """
        Overriding parent method for the case of all RangeIndex instances.

        When all members of "indexes" are of type RangeIndex: result will be
        RangeIndex if possible, Index with a int64 dtype otherwise. E.g.:
        indexes = [RangeIndex(3), RangeIndex(3, 6)] -> RangeIndex(6)
        indexes = [RangeIndex(3), RangeIndex(4, 6)] -> Index([0,1,2,4,5], dtype='int64')
        """
        if not all(isinstance(x, RangeIndex) for x in indexes):
            result = super()._concat(indexes, name)
            if result.dtype.kind == "i":
                return self._shallow_copy(result._values)
            return result
        elif len(indexes) == 1:
            return indexes[0]
        rng_indexes: list[RangeIndex] = cast(list[RangeIndex], indexes)
        start: Optional[int] = None
        step: Optional[int] = None
        next_: Optional[int] = None
        non_empty_indexes: list[RangeIndex] = []
        all_same_index: bool = True
        prev: Optional[RangeIndex] = None
        for obj in rng_indexes:
            if len(obj):
                non_empty_indexes.append(obj)
                if all_same_index:
                    if prev is not None:
                        all_same_index = prev.equals(obj)
                    else:
                        prev = obj
        for obj in non_empty_indexes:
            rng = obj._range
            if start is None:
                start = rng.start
                if step is None and len(rng) > 1:
                    step = rng.step
            elif step is None:
                if rng.start == start:
                    if all_same_index:
                        values = np.tile(
                            non_empty_indexes[0]._values, len(non_empty_indexes)
                        )
                    else:
                        values = np.concatenate([x._values for x in rng_indexes])
                    result = self._constructor(values)
                    return result.rename(name)
                step = rng.start - start
            non_consecutive = (
                step != rng.step and len(rng) > 1
            ) or (next_ is not None and rng.start != next_)
            if non_consecutive:
                if all_same_index:
                    values = np.tile(
                        non_empty_indexes[0]._values, len(non_empty_indexes)
                    )
                else:
                    values = np.concatenate([x._values for x in rng_indexes])
                result = self._constructor(values)
                return result.rename(name)
            if step is not None:
                next_ = rng[-1] + step
        if non_empty_indexes:
            stop: Optional[int] = (
                non_empty_indexes[-1].stop if next_ is None else next_
            )
            if len(non_empty_indexes) == 1:
                step = non_empty_indexes[0].step
            return RangeIndex(start, stop, step, name=name)
        return RangeIndex(_empty_range, name=name)

    def __len__(self) -> int:
        """
        return the length of the RangeIndex
        """
        return len(self._range)

    @property
    def func_hull69j8(self) -> int:
        return len(self)

    def __getitem__(
        self,
        key: Union[int, slice, Sequence[int], Sequence[bool], Any],
    ) -> Union[int, RangeIndex, Index]:
        """
        Conserve RangeIndex type for scalar and slice keys.
        """
        if key is Ellipsis:
            key = slice(None)
        if isinstance(key, slice):
            return self._getitem_slice(key)
        elif is_integer(key):
            new_key = int(key)
            try:
                return self._range[new_key]
            except IndexError as err:
                raise IndexError(
                    f"index {key} is out of bounds for axis 0 with size {len(self)}"
                ) from err
        elif is_scalar(key):
            raise IndexError(
                "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"
            )
        elif com.is_bool_indexer(key):
            if isinstance(getattr(key, "dtype", None), ExtensionDtype):
                key = key.to_numpy(dtype=bool, na_value=False)
            else:
                key = np.asarray(key, dtype=bool)
            check_array_indexer(self._range, key)
            key = np.flatnonzero(key)
        try:
            return self.take(key)
        except (TypeError, ValueError):
            return super().__getitem__(key)

    def func_e9hh7s9k(self, slobj: slice) -> RangeIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._range[slobj]
        return type(self)._simple_new(res, name=self._name)

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other: int) -> RangeIndex | Any:
        if is_integer(other) and other != 0:
            if (
                len(self) == 0
                or (self.start % other == 0 and self.step % other == 0)
            ):
                start = self.start // other
                step = self.step // other
                stop = start + len(self) * step
                new_range = range(start, stop, step or 1)
                return self._simple_new(new_range, name=self._name)
            if len(self) == 1:
                start = self.start // other
                new_range = range(start, start + 1, 1)
                return self._simple_new(new_range, name=self._name)
        return super().__floordiv__(other)

    def all(self, *args: Any, **kwargs: Any) -> bool:
        return 0 not in self._range

    def any(self, *args: Any, **kwargs: Any) -> bool:
        return any(self._range)

    def func_p0e0z24z(
        self, decimals: int = 0
    ) -> Union[Index, RangeIndex]:
        """
        Round each value in the Index to the given number of decimals.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point
            e.g. ``round(11.0, -1) == 10.0``.

        Returns
        -------
        Index or RangeIndex
            A new Index with the rounded values.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.RangeIndex(10, 30, 10)
        >>> idx.round(decimals=-1)
        RangeIndex(start=10, stop=30, step=10)
        >>> idx = pd.RangeIndex(10, 15, 1)
        >>> idx.round(decimals=-1)
        Index([10, 10, 10, 10, 10], dtype='int64')
        """
        if decimals >= 0:
            return self.copy()
        elif (
            self.start % 10**-decimals == 0
            and self.step % 10**-decimals == 0
        ):
            return self.copy()
        else:
            return super().round(decimals=decimals)

    def func_az2sa5zv(
        self, other: Any, op: Callable[[Any, Any], Any]
    ) -> Any:
        if isinstance(other, RangeIndex) and self._range == other._range:
            return super()._cmp_method(self, op)
        return super()._cmp_method(other, op)

    def func_wwpgorqj(
        self, other: Any, op: Callable[[Any, Any], Any]
    ) -> Any:
        """
        Parameters
        ----------
        other : Any
        op : callable that accepts 2 params
            perform the binary op
        """
        if isinstance(other, ABCTimedeltaIndex):
            return NotImplemented
        elif isinstance(other, (timedelta, np.timedelta64)):
            return super()._arith_method(other, op)
        elif lib.is_np_dtype(getattr(other, "dtype", None), "m"):
            return super()._arith_method(other, op)
        if op in [
            operator.pow,
            ops.rpow,
            operator.mod,
            ops.rmod,
            operator.floordiv,
            ops.rfloordiv,
            divmod,
            ops.rdivmod,
        ]:
            return super()._arith_method(other, op)
        step: Optional[Callable[[int, Any], Any]] = None
        if op in [
            operator.mul,
            ops.rmul,
            operator.truediv,
            ops.rtruediv,
        ]:
            step = op
        right = extract_array(other, extract_numpy=True, extract_range=True)
        left = self
        try:
            if step:
                with np.errstate(all="ignore"):
                    rstep = func_nqsvyud5  # This should be a method or property
                if not is_integer(rstep) or not rstep:
                    raise ValueError
            else:
                rstep = -left.step if op == ops.rsub else left.step
            with np.errstate(all="ignore"):
                rstart = op(left.start, right)
                rstop = op(left.stop, right)
            res_name = ops.get_op_result_name(self, other)
            result = type(self)(rstart, rstop, rstep, name=res_name)
            if not all(is_integer(x) for x in [rstart, rstop, rstep]):
                result = result.astype("float64")
            return result
        except (ValueError, TypeError, ZeroDivisionError):
            return super()._arith_method(other, op)

    def __abs__(self) -> RangeIndex | Index:
        if len(self) == 0 or self.min() >= 0:
            return self.copy()
        elif self.max() <= 0:
            return -self
        else:
            return super().__abs__()

    def __neg__(self) -> RangeIndex:
        rng = range(-self.start, -self.stop, -self.step)
        return self._simple_new(rng, name=self.name)

    def __pos__(self) -> RangeIndex:
        return self.copy()

    def __invert__(self) -> RangeIndex:
        if len(self) == 0:
            return self.copy()
        rng = range(~self.start, ~self.stop, -self.step)
        return self._simple_new(rng, name=self.name)

    def func_gj0vtaf0(
        self,
        indices: Any,
        axis: int = 0,
        allow_fill: bool = True,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> RangeIndex:
        if kwargs:
            nv.validate_take((), kwargs)
        if is_scalar(indices):
            raise TypeError("Expected indices to be array-like")
        indices = ensure_platform_int(indices)
        self._maybe_disallow_fill(allow_fill, fill_value, indices)
        if len(indices) == 0:
            return type(self)(_empty_range, name=self.name)
        else:
            ind_max = indices.max()
            if ind_max >= len(self):
                raise IndexError(
                    f"index {ind_max} is out of bounds for axis 0 with size {len(self)}"
                )
            ind_min = indices.min()
            if ind_min < -len(self):
                raise IndexError(
                    f"index {ind_min} is out of bounds for axis 0 with size {len(self)}"
                )
            taken = indices.astype(self.dtype, casting="safe")
            if ind_min < 0:
                taken %= len(self)
            if self.step != 1:
                taken *= self.step
            if self.start != 0:
                taken += self.start
        return self._shallow_copy(taken, name=self.name)

    def func_u6ftddfv(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins: Optional[int] = None,
        dropna: bool = True,
    ) -> Series:
        from pandas import Series
        if bins is not None:
            return super().value_counts(
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
                dropna=dropna,
            )
        name: str = "proportion" if normalize else "count"
        data: np.ndarray = np.ones(len(self), dtype=np.int64)
        if normalize:
            data = data / len(self)
        return Series(data, index=self.copy(), name=name)

    def func_kok4zb6r(
        self,
        value: int | np.integer | float,
        side: Literal["left", "right"] = "left",
        sorter: Optional[int] = None,
    ) -> int | np.ndarray:
        if side not in {"left", "right"} or sorter is not None:
            return super().searchsorted(value=value, side=side, sorter=sorter)
        was_scalar: bool = False
        if is_scalar(value):
            was_scalar = True
            array_value: np.ndarray = np.array([value])
        else:
            array_value = np.asarray(value)
        if array_value.dtype.kind not in "iu":
            return super().searchsorted(value=value, side=side, sorter=sorter)
        if (flip := self.step < 0):
            rng = self._range[::-1]
            start = rng.start
            step = rng.step
            shift = side == "right"
        else:
            start = self.start
            step = self.step
            shift = side == "left"
        result = (array_value - start - int(shift)) // step + 1
        if flip:
            result = len(self) - result
        result = np.maximum(np.minimum(result, len(self)), 0)
        if was_scalar:
            return int(result.item())
        return result.astype(np.intp, copy=False)
