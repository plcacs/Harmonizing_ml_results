from __future__ import annotations
from collections.abc import Callable, Hashable, Iterator
from datetime import timedelta
import operator
from sys import getsizeof
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union, overload
import numpy as np
from pandas._libs import index as libindex, lib
from pandas._libs.lib import no_default
from pandas.compat.numpy import function as nv
from pandas.util._decorators import cache_readonly, doc, set_module
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import ensure_platform_int, ensure_python_int, is_float, is_integer, is_scalar, is_signed_integer_dtype
from pandas.core.dtypes.generic import ABCTimedeltaIndex
from pandas.core import ops
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import Index, maybe_extract_name
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING:
    from pandas._typing import Axis, Dtype, JoinHow, NaPosition, NumpySorter, Self, npt
    from pandas import Series

_empty_range: range = range(0)
_dtype_int64: np.dtype = np.dtype(np.int64)


def min_fitting_element(start: int, step: int, lower_limit: int) -> int:
    """Returns the smallest element greater than or equal to the limit"""
    no_steps: int = - (-(lower_limit - start) // abs(step))
    return start + abs(step) * no_steps


@set_module("pandas")
class RangeIndex(Index):
    """
    Immutable Index implementing a monotonic integer range.
    """
    _typ: str = "rangeindex"
    _dtype_validation_metadata: Tuple[Callable[[Any], bool], str] = (is_signed_integer_dtype, "signed integer")

    @property
    def _engine_type(self) -> type:
        return libindex.Int64Engine

    def __new__(
        cls: type[RangeIndex],
        start: Optional[Union[int, range, RangeIndex]] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        dtype: Optional[Any] = None,
        copy: bool = False,
        name: Any = None,
    ) -> RangeIndex:
        cls._validate_dtype(dtype)
        name = maybe_extract_name(name, start, cls)
        if isinstance(start, cls):
            return start.copy(name=name)
        elif isinstance(start, range):
            return cls._simple_new(start, name=name)
        if com.all_none(start, stop, step):
            raise TypeError("RangeIndex(...) must be called with integers")
        start_int: int = ensure_python_int(start) if start is not None else 0
        if stop is None:
            start_int, stop = (0, start_int)
        else:
            stop = ensure_python_int(stop)
        step_int: int = ensure_python_int(step) if step is not None else 1
        if step_int == 0:
            raise ValueError("Step must not be zero")
        rng: range = range(start_int, stop, step_int)
        return cls._simple_new(rng, name=name)

    @classmethod
    def from_range(
        cls: type[RangeIndex],
        data: range,
        name: Any = None,
        dtype: Optional[Any] = None,
    ) -> RangeIndex:
        """
        Create :class:`pandas.RangeIndex` from a ``range`` object.
        """
        if not isinstance(data, range):
            raise TypeError(f"{cls.__name__}(...) must be called with object coercible to a range, {data!r} was passed")
        cls._validate_dtype(dtype)
        return cls._simple_new(data, name=name)

    @classmethod
    def _simple_new(cls: type[RangeIndex], values: range, name: Any = None) -> RangeIndex:
        result: RangeIndex = object.__new__(cls)
        assert isinstance(values, range)
        result._range: range = values
        result._name = name
        result._cache: dict[Any, Any] = {}
        result._reset_identity()
        result._references = None
        return result

    @classmethod
    def _validate_dtype(cls, dtype: Optional[Any]) -> None:
        if dtype is None:
            return
        validation_func, expected = cls._dtype_validation_metadata
        if not validation_func(dtype):
            raise ValueError(f"Incorrect `dtype` passed: expected {expected}, received {dtype}")

    @cache_readonly
    def _constructor(self) -> type[Index]:
        """return the class to use for construction"""
        return Index

    @cache_readonly
    def _data(self) -> np.ndarray:
        """
        An int array that for performance reasons is created only when needed.
        """
        return np.arange(self.start, self.stop, self.step, dtype=np.int64)

    def _get_data_as_items(self) -> list[Tuple[str, Union[str, int]]]:
        """return a list of tuples of start, stop, step"""
        rng: range = self._range
        return [("start", rng.start), ("stop", rng.stop), ("step", rng.step)]

    def __reduce__(self) -> Tuple[Any, Tuple[Any, ...], None]:
        d: dict[str, Any] = {"name": self._name}
        d.update(dict(self._get_data_as_items()))
        return (ibase._new_Index, (type(self), d), None)

    def _format_attrs(self) -> list[Tuple[str, Union[str, int]]]:
        """
        Return a list of tuples of the (attr, formatted_value)
        """
        attrs: list[Tuple[str, Union[str, int]]] = self._get_data_as_items()  # type: ignore
        if self._name is not None:
            attrs.append(("name", ibase.default_pprint(self._name)))
        return attrs

    def _format_with_header(self, *, header: list[str], na_rep: str) -> list[str]:
        if not len(self._range):
            return header
        first_val_str: str = str(self._range[0])
        last_val_str: str = str(self._range[-1])
        max_length: int = max(len(first_val_str), len(last_val_str))
        return header + [f"{x:<{max_length}}" for x in self._range]

    @property
    def start(self) -> int:
        """
        The value of the `start` parameter.
        """
        return self._range.start

    @property
    def stop(self) -> int:
        """
        The value of the `stop` parameter.
        """
        return self._range.stop

    @property
    def step(self) -> int:
        """
        The value of the `step` parameter.
        """
        return self._range.step

    @cache_readonly
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.
        """
        rng: range = self._range
        return getsizeof(rng) + sum(getsizeof(getattr(rng, attr_name)) for attr_name in ["start", "stop", "step"])

    def memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply
        Returns
        -------
        bytes used
        """
        return self.nbytes

    @property
    def dtype(self) -> np.dtype:
        return _dtype_int64

    @property
    def is_unique(self) -> bool:
        """return if the index has unique values"""
        return True

    @cache_readonly
    def is_monotonic_increasing(self) -> bool:
        return self._range.step > 0 or len(self) <= 1

    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        return self._range.step < 0 or len(self) <= 1

    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            key_int: int = ensure_python_int(key)
        except (TypeError, OverflowError):
            return False
        return key_int in self._range

    @property
    def inferred_type(self) -> str:
        return "integer"

    @doc(Index.get_loc)
    def get_loc(self, key: Any) -> int:
        if is_integer(key) or (is_float(key) and key.is_integer()):
            new_key: int = int(key)  # type: ignore
            try:
                return self._range.index(new_key)
            except ValueError as err:
                raise KeyError(key) from err
        if isinstance(key, Hashable):
            raise KeyError(key)
        self._check_indexing_error(key)
        raise KeyError(key)

    def _get_indexer(
        self,
        target: Any,
        method: Optional[Any] = None,
        limit: Optional[Any] = None,
        tolerance: Optional[Any] = None,
    ) -> np.ndarray:
        if com.any_not_none(method, tolerance, limit):
            return super()._get_indexer(target, method=method, tolerance=tolerance, limit=limit)  # type: ignore
        if self.step > 0:
            start_val, stop_val, step_val = (self.start, self.stop, self.step)
        else:
            reverse: range = self._range[::-1]
            start_val, stop_val, step_val = (reverse.start, reverse.stop, reverse.step)
        target_array: np.ndarray = np.asarray(target)
        locs: np.ndarray = target_array - start_val
        valid: np.ndarray = (locs % step_val == 0) & (locs >= 0) & (target_array < stop_val)
        locs[~valid] = -1
        locs[valid] = locs[valid] / step_val
        if step_val != self.step:
            locs[valid] = len(self) - 1 - locs[valid]
        return ensure_platform_int(locs)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        """
        Should an integer key be treated as positional?
        """
        return False

    def tolist(self) -> list[int]:
        return list(self._range)

    @doc(Index.__iter__)
    def __iter__(self) -> Iterator[int]:
        yield from self._range

    @doc(Index._shallow_copy)
    def _shallow_copy(self, values: np.ndarray, name: Any = no_default) -> Index:
        name = self._name if name is no_default else name
        if values.dtype.kind == "f":
            return Index(values, name=name, dtype=np.float64)
        if values.dtype.kind == "i" and values.ndim == 1:
            if len(values) == 1:
                start_val = values[0]
                new_range: range = range(start_val, start_val + self.step, self.step)
                return type(self)._simple_new(new_range, name=name)
            maybe_range = ibase.maybe_sequence_to_range(values)
            if isinstance(maybe_range, range):
                return type(self)._simple_new(maybe_range, name=name)
        return self._constructor._simple_new(values, name=name)

    def _view(self) -> RangeIndex:
        result: RangeIndex = type(self)._simple_new(self._range, name=self._name)
        result._cache = self._cache
        return result

    def _wrap_reindex_result(self, target: Index, indexer: Any, preserve_names: bool) -> Index:
        if not isinstance(target, type(self)) and target.dtype.kind == "i":
            target = self._shallow_copy(target._values, name=target.name)
        return super()._wrap_reindex_result(target, indexer, preserve_names)

    @doc(Index.copy)
    def copy(self, name: Any = None, deep: bool = False) -> RangeIndex:
        name = self._validate_names(name=name, deep=deep)[0]
        new_index: RangeIndex = self._rename(name=name)
        return new_index

    def _minmax(self, meth: str) -> Union[int, float]:
        no_steps: int = len(self) - 1
        if no_steps == -1:
            return np.nan
        elif (meth == "min" and self.step > 0) or (meth == "max" and self.step < 0):
            return self.start
        return self.start + self.step * no_steps

    def min(self, axis: Optional[Axis] = None, skipna: bool = True, *args: Any, **kwargs: Any) -> Union[int, float]:
        """The minimum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_min(args, kwargs)
        return self._minmax("min")

    def max(self, axis: Optional[Axis] = None, skipna: bool = True, *args: Any, **kwargs: Any) -> Union[int, float]:
        """The maximum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_max(args, kwargs)
        return self._minmax("max")

    def _argminmax(self, meth: str, axis: Optional[Axis] = None, skipna: bool = True) -> int:
        nv.validate_minmax_axis(axis)
        if len(self) == 0:
            return getattr(super(), f"arg{meth}")(axis=axis, skipna=skipna)
        elif meth == "min":
            if self.step > 0:
                return 0
            else:
                return len(self) - 1
        elif meth == "max":
            if self.step > 0:
                return len(self) - 1
            else:
                return 0
        else:
            raise ValueError(f"meth={meth!r} must be max or min")

    def argmin(self, axis: Optional[Axis] = None, skipna: bool = True, *args: Any, **kwargs: Any) -> int:
        nv.validate_argmin(args, kwargs)
        return self._argminmax("min", axis=axis, skipna=skipna)

    def argmax(self, axis: Optional[Axis] = None, skipna: bool = True, *args: Any, **kwargs: Any) -> int:
        nv.validate_argmax(args, kwargs)
        return self._argminmax("max", axis=axis, skipna=skipna)

    def argsort(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Returns the indices that would sort the index.
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
                start, stop, step = (len(self) - 1, -1, -1)
        elif ascending:
            start, stop, step = (len(self) - 1, -1, -1)
        else:
            start = len(self)
        return np.arange(start, stop, step, dtype=np.intp)  # type: ignore

    def factorize(self, sort: bool = False, use_na_sentinel: bool = True) -> Tuple[np.ndarray, RangeIndex]:
        if sort and self.step < 0:
            codes: np.ndarray = np.arange(len(self) - 1, -1, -1, dtype=np.intp)
            uniques: RangeIndex = self[::-1]
        else:
            codes = np.arange(len(self), dtype=np.intp)
            uniques = self
        return (codes, uniques)

    def equals(self, other: Any) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        if isinstance(other, RangeIndex):
            return self._range == other._range
        return super().equals(other)

    @overload
    def sort_values(
        self, *, return_indexer: Literal[True], ascending: bool, na_position: NaPosition, key: Callable[[Any], Any]
    ) -> Tuple[RangeIndex, RangeIndex]:
        ...

    @overload
    def sort_values(
        self, *, return_indexer: Literal[False], ascending: bool, na_position: NaPosition, key: Callable[[Any], Any]
    ) -> RangeIndex:
        ...

    @overload
    def sort_values(
        self, *, return_indexer: bool, ascending: bool, na_position: NaPosition, key: Optional[Callable[[Any], Any]] = None
    ) -> Union[RangeIndex, Tuple[RangeIndex, RangeIndex]]:
        ...

    def sort_values(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: str = "last",
        key: Optional[Callable[[Any], Any]] = None,
    ) -> Union[RangeIndex, Tuple[RangeIndex, RangeIndex]]:
        if key is not None:
            return super().sort_values(return_indexer=return_indexer, ascending=ascending, na_position=na_position, key=key)  # type: ignore
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
                rng: range = range(len(self) - 1, -1, -1)
            else:
                rng = range(len(self))
            return (sorted_index, RangeIndex(rng))
        else:
            return sorted_index

    def _intersection(self, other: Index, sort: bool = False) -> Index:
        if not isinstance(other, RangeIndex):
            return super()._intersection(other, sort=sort)  # type: ignore
        first: range = self._range[::-1] if self.step < 0 else self._range
        second: range = other._range[::-1] if other.step < 0 else other._range  # type: ignore
        int_low: int = max(first.start, second.start)
        int_high: int = min(first.stop, second.stop)
        if int_high <= int_low:
            return self._simple_new(_empty_range)
        gcd, s, _ = self._extended_gcd(first.step, second.step)
        if (first.start - second.start) % gcd:
            return self._simple_new(_empty_range)
        tmp_start: int = first.start + (second.start - first.start) * first.step // gcd
        new_step: int = first.step * second.step // gcd
        new_start: int = min_fitting_element(tmp_start, new_step, int_low)
        new_range: range = range(new_start, int_high, new_step)
        if (self.step < 0 and other.step < 0) is not (new_range.step < 0):
            new_range = new_range[::-1]
        return self._simple_new(new_range)

    def _extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean algorithms to solve Bezout's identity:
           a*x + b*y = gcd(x, y)
        Returns: gcd, s, t
        """
        s: int = 0
        old_s: int = 1
        t: int = 1
        old_t: int = 0
        r: int = b
        old_r: int = a
        while r:
            quotient: int = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t
        return (old_r, old_s, old_t)

    def _range_in_self(self, other: range) -> bool:
        """Check if other range is contained in self"""
        if not other:
            return True
        if not self._range:
            return False
        if len(other) > 1 and other.step % self._range.step:
            return False
        return other.start in self._range and other[-1] in self._range

    def _union(self, other: Index, sort: Union[bool, None] = None) -> Index:
        """
        Form the union of two Index objects and sorts if possible.
        """
        if isinstance(other, RangeIndex):
            if sort in (None, True) or (sort is False and self.step > 0 and self._range_in_self(other._range)):
                start_s: int = self.start
                step_s: int = self.step
                end_s: int = self.start + self.step * (len(self) - 1)
                start_o: int = other.start  # type: ignore
                step_o: int = other.step  # type: ignore
                end_o: int = other.start + other.step * (len(other) - 1)  # type: ignore
                if self.step < 0:
                    start_s, step_s, end_s = (end_s, -step_s, self.start)
                if other.step < 0:  # type: ignore
                    start_o, step_o, end_o = (end_o, -step_o, other.start)  # type: ignore
                if len(self) == 1 and len(other) == 1:
                    step_s = step_o = abs(self.start - other.start)  # type: ignore
                elif len(self) == 1:
                    step_s = step_o
                elif len(other) == 1:
                    step_o = step_s
                start_r: int = min(start_s, start_o)
                end_r: int = max(end_s, end_o)
                if step_o == step_s:
                    if (start_s - start_o) % step_s == 0 and start_s - end_o <= step_s and (start_o - end_s <= step_s):
                        return type(self)(start_r, end_r + step_s, step_s)
                    if step_s % 2 == 0 and abs(start_s - start_o) == step_s / 2 and (abs(end_s - end_o) == step_s / 2):
                        return type(self)(start_r, end_r + int(step_s / 2), int(step_s / 2))
                elif step_o % step_s == 0:
                    if (start_o - start_s) % step_s == 0 and start_o + step_s >= start_s and (end_o - step_s <= end_s):
                        return type(self)(start_r, end_r + step_s, step_s)
                elif step_s % step_o == 0:
                    if (start_s - start_o) % step_o == 0 and start_s + step_o >= start_o and (end_s - step_o <= end_o):
                        return type(self)(start_r, end_r + step_o, step_o)
        return super()._union(other, sort=sort)  # type: ignore

    def _difference(self, other: Index, sort: Optional[bool] = None) -> Index:
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)
        if not isinstance(other, RangeIndex):
            return super()._difference(other, sort=sort)  # type: ignore
        if sort is not False and self.step < 0:
            return self[::-1]._difference(other)
        res_name: Any = ops.get_op_result_name(self, other)
        first: range = self._range[::-1] if self.step < 0 else self._range
        overlap: Index = self.intersection(other)
        if overlap.step < 0:  # type: ignore
            overlap = overlap[::-1]  # type: ignore
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
                return super()._difference(other, sort=sort)  # type: ignore
        elif len(overlap) == 2 and overlap[0] == first[0] and (overlap[-1] == first[-1]):
            return self[1:-1]
        if overlap.step == first.step:  # type: ignore
            if overlap[0] == first.start:
                new_rng: range = range(overlap[-1] + first.step, first.stop, first.step)
            elif overlap[-1] == first[-1]:
                new_rng = range(first.start, overlap[0], first.step)
            elif overlap._range == first[1:-1]:  # type: ignore
                step_val: int = len(first) - 1
                new_rng = first[::step_val]
            else:
                return super()._difference(other, sort=sort)  # type: ignore
        else:
            assert len(self) > 1
            if overlap.step == first.step * 2:  # type: ignore
                if overlap[0] == first[0] and overlap[-1] in (first[-1], first[-2]):
                    new_rng = first[1::2]
                elif overlap[0] == first[1] and overlap[-1] in (first[-1], first[-2]):
                    new_rng = first[::2]
                else:
                    return super()._difference(other, sort=sort)  # type: ignore
            else:
                return super()._difference(other, sort=sort)  # type: ignore
        if first is not self._range:
            new_rng = new_rng[::-1]
        new_index: RangeIndex = type(self)._simple_new(new_rng, name=res_name)
        return new_index

    def symmetric_difference(self, other: Index, result_name: Optional[Any] = None, sort: Optional[bool] = None) -> Index:
        if not isinstance(other, RangeIndex) or sort is not None:
            return super().symmetric_difference(other, result_name, sort)  # type: ignore
        left: Index = self.difference(other)
        right: Index = other.difference(self)  # type: ignore
        result: Index = left.union(right)
        if result_name is not None:
            result = result.rename(result_name)
        return result

    def _join_empty(self, other: Index, how: Any, sort: Optional[bool]) -> Index:
        if not isinstance(other, RangeIndex) and other.dtype.kind == "i":
            other = self._shallow_copy(other._values, name=other.name)
        return super()._join_empty(other, how=how, sort=sort)  # type: ignore

    def _join_monotonic(self, other: Index, how: str = "left") -> Tuple[Index, Optional[np.ndarray], Optional[np.ndarray]]:
        if not isinstance(other, type(self)):
            maybe_ri: Index = self._shallow_copy(other._values, name=other.name)
            if not isinstance(maybe_ri, type(self)):
                return super()._join_monotonic(other, how=how)  # type: ignore
            other = maybe_ri  # type: ignore
        if self.equals(other):
            ret_index: Index = other if how == "right" else self
            return (ret_index, None, None)
        if how == "left":
            join_index: Index = self
            lidx: Optional[np.ndarray] = None
            ridx: Optional[np.ndarray] = other.get_indexer(join_index)
        elif how == "right":
            join_index = other
            lidx = self.get_indexer(join_index)
            ridx = None
        elif how == "inner":
            join_index = self.intersection(other)
            lidx = self.get_indexer(join_index)
            ridx = other.get_indexer(join_index)  # type: ignore
        elif how == "outer":
            join_index = self.union(other)
            lidx = self.get_indexer(join_index)
            ridx = other.get_indexer(join_index)  # type: ignore
        lidx = None if lidx is None else ensure_platform_int(lidx)
        ridx = None if ridx is None else ensure_platform_int(ridx)
        return (join_index, lidx, ridx)

    def delete(self, loc: Any) -> Index:
        if is_integer(loc):
            if loc in (0, -len(self)):
                return self[1:]
            if loc in (-1, len(self) - 1):
                return self[:-1]
            if len(self) == 3 and loc in (1, -2):
                return self[::2]
        elif lib.is_list_like(loc):
            slc = lib.maybe_indices_to_slice(np.asarray(loc, dtype=np.intp), len(self))
            if isinstance(slc, slice):
                other: Index = self[slc]
                return self.difference(other, sort=False)
        return super().delete(loc)  # type: ignore

    def insert(self, loc: int, item: Any) -> Index:
        if is_integer(item) or is_float(item):
            if len(self) == 0 and loc == 0 and is_integer(item):
                new_rng: range = range(item, item + self.step, self.step)
                return type(self)._simple_new(new_rng, name=self._name)
            elif len(self):
                rng: range = self._range
                if loc == 0 and item == self[0] - self.step:
                    new_rng = range(rng.start - rng.step, rng.stop, rng.step)
                    return type(self)._simple_new(new_rng, name=self._name)
                elif loc == len(self) and item == self[-1] + self.step:
                    new_rng = range(rng.start, rng.stop + rng.step, rng.step)
                    return type(self)._simple_new(new_rng, name=self._name)
                elif len(self) == 2 and item == self[0] + self.step / 2:
                    step_val: int = int(self.step / 2)
                    new_rng = range(self.start, self.stop, step_val)
                    return type(self)._simple_new(new_rng, name=self._name)
        return super().insert(loc, item)  # type: ignore

    def _concat(self, indexes: list[Index], name: Any) -> Index:
        """
        Overriding parent method for the case of all RangeIndex instances.
        """
        if not all((isinstance(x, RangeIndex) for x in indexes)):
            result = super()._concat(indexes, name)
            if result.dtype.kind == "i":
                return self._shallow_copy(result._values)
            return result
        elif len(indexes) == 1:
            return indexes[0]
        rng_indexes: list[RangeIndex] = [cast(RangeIndex, x) for x in indexes]
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
            rng: range = obj._range
            if start is None:
                start = rng.start
                if step is None and len(rng) > 1:
                    step = rng.step
            elif step is None:
                if rng.start == start:
                    if all_same_index:
                        values = np.tile(non_empty_indexes[0]._values, len(non_empty_indexes))
                    else:
                        values = np.concatenate([x._values for x in rng_indexes])
                    result = self._constructor(values)
                    return result.rename(name)
                step = rng.start - start
            non_consecutive: bool = (step != rng.step and len(rng) > 1) or (next_ is not None and rng.start != next_)
            if non_consecutive:
                if all_same_index:
                    values = np.tile(non_empty_indexes[0]._values, len(non_empty_indexes))
                else:
                    values = np.concatenate([x._values for x in rng_indexes])
                result = self._constructor(values)
                return result.rename(name)
            if step is not None:
                next_ = rng[-1] + step
        if non_empty_indexes:
            stop: int = non_empty_indexes[-1].stop if next_ is None else next_
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
    def size(self) -> int:
        return len(self)

    def __getitem__(self, key: Any) -> Union[int, RangeIndex, Any]:
        """
        Conserve RangeIndex type for scalar and slice keys.
        """
        if key is Ellipsis:
            key = slice(None)
        if isinstance(key, slice):
            return self._getitem_slice(key)
        elif is_integer(key):
            new_key: int = int(key)
            try:
                return self._range[new_key]
            except IndexError as err:
                raise IndexError(f"index {key} is out of bounds for axis 0 with size {len(self)}") from err
        elif is_scalar(key):
            raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices")
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

    def _getitem_slice(self, slobj: slice) -> RangeIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res: range = self._range[slobj]
        return type(self)._simple_new(res, name=self._name)

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other: Any) -> Union[RangeIndex, Any]:
        if is_integer(other) and other != 0:
            if len(self) == 0 or (self.start % other == 0 and self.step % other == 0):
                start_val: int = self.start // other
                step_val: int = self.step // other
                stop_val: int = start_val + len(self) * step_val
                new_range: range = range(start_val, stop_val, step_val or 1)
                return self._simple_new(new_range, name=self._name)
            if len(self) == 1:
                start_val = self.start // other
                new_range = range(start_val, start_val + 1, 1)
                return self._simple_new(new_range, name=self._name)
        return super().__floordiv__(other)

    def all(self, *args: Any, **kwargs: Any) -> bool:
        return 0 not in self._range

    def any(self, *args: Any, **kwargs: Any) -> bool:
        return any(self._range)

    def round(self, decimals: int = 0) -> Union[RangeIndex, Index]:
        """
        Round each value in the Index to the given number of decimals.
        """
        if decimals >= 0:
            return self.copy()
        elif self.start % 10 ** (-decimals) == 0 and self.step % 10 ** (-decimals) == 0:
            return self.copy()
        else:
            return super().round(decimals=decimals)  # type: ignore

    def _cmp_method(self, other: Any, op: Any) -> Any:
        if isinstance(other, RangeIndex) and self._range == other._range:
            return super()._cmp_method(self, op)
        return super()._cmp_method(other, op)

    def _arith_method(self, other: Any, op: Any) -> Any:
        """
        Parameters
        ----------
        other : Any
        op : callable that accepts 2 params
        """
        if isinstance(other, ABCTimedeltaIndex):
            return NotImplemented
        elif isinstance(other, (timedelta, np.timedelta64)):
            return super()._arith_method(other, op)
        elif lib.is_np_dtype(getattr(other, "dtype", None), "m"):
            return super()._arith_method(other, op)
        if op in [operator.pow, ops.rpow, operator.mod, ops.rmod, operator.floordiv, ops.rfloordiv, divmod, ops.rdivmod]:
            return super()._arith_method(other, op)
        step_op: Optional[Any] = None
        if op in [operator.mul, ops.rmul, operator.truediv, ops.rtruediv]:
            step_op = op
        right = extract_array(other, extract_numpy=True, extract_range=True)
        left = self
        try:
            if step_op:
                with np.errstate(all="ignore"):
                    rstep = step_op(left.step, right)
                if not is_integer(rstep) or not rstep:
                    raise ValueError
            else:
                rstep = -left.step if op == ops.rsub else left.step
            with np.errstate(all="ignore"):
                rstart = op(left.start, right)
                rstop = op(left.stop, right)
            res_name = ops.get_op_result_name(self, other)
            result = type(self)(rstart, rstop, rstep, name=res_name)
            if not all((is_integer(x) for x in [rstart, rstop, rstep])):
                result = result.astype("float64")
            return result
        except (ValueError, TypeError, ZeroDivisionError):
            return super()._arith_method(other, op)

    def __abs__(self) -> RangeIndex:
        if len(self) == 0 or self.min() >= 0:
            return self.copy()
        elif self.max() <= 0:
            return -self
        else:
            return super().__abs__()  # type: ignore

    def __neg__(self) -> RangeIndex:
        rng: range = range(-self.start, -self.stop, -self.step)
        return self._simple_new(rng, name=self.name)

    def __pos__(self) -> RangeIndex:
        return self.copy()

    def __invert__(self) -> RangeIndex:
        if len(self) == 0:
            return self.copy()
        rng: range = range(~self.start, ~self.stop, -self.step)
        return self._simple_new(rng, name=self.name)

    def take(
        self,
        indices: Any,
        axis: int = 0,
        allow_fill: bool = True,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Index:
        if kwargs:
            nv.validate_take((), kwargs)
        if is_scalar(indices):
            raise TypeError("Expected indices to be array-like")
        indices = ensure_platform_int(indices)
        self._maybe_disallow_fill(allow_fill, fill_value, indices)
        if len(indices) == 0:
            return type(self)(_empty_range, name=self.name)
        else:
            ind_max: int = int(indices.max())
            if ind_max >= len(self):
                raise IndexError(f"index {ind_max} is out of bounds for axis 0 with size {len(self)}")
            ind_min: int = int(indices.min())
            if ind_min < -len(self):
                raise IndexError(f"index {ind_min} is out of bounds for axis 0 with size {len(self)}")
            taken = indices.astype(self.dtype, casting="safe")
            if ind_min < 0:
                taken %= len(self)
            if self.step != 1:
                taken *= self.step
            if self.start != 0:
                taken += self.start
        return self._shallow_copy(taken, name=self.name)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins: Optional[int] = None,
        dropna: bool = True,
    ) -> Series:
        from pandas import Series
        if bins is not None:
            return super().value_counts(normalize=normalize, sort=sort, ascending=ascending, bins=bins, dropna=dropna)  # type: ignore
        name: str = "proportion" if normalize else "count"
        data: np.ndarray = np.ones(len(self), dtype=np.int64)
        if normalize:
            data = data / len(self)
        return Series(data, index=self.copy(), name=name)

    def searchsorted(self, value: Any, side: str = "left", sorter: Optional[Any] = None) -> Union[int, np.ndarray]:
        if side not in {"left", "right"} or sorter is not None:
            return super().searchsorted(value=value, side=side, sorter=sorter)  # type: ignore
        was_scalar: bool = False
        if is_scalar(value):
            was_scalar = True
            array_value: np.ndarray = np.array([value])
        else:
            array_value = np.asarray(value)
        if array_value.dtype.kind not in "iu":
            return super().searchsorted(value=value, side=side, sorter=sorter)  # type: ignore
        if (flip := (self.step < 0)):
            rng: range = self._range[::-1]
            start_val: int = rng.start
            step_val: int = rng.step
            shift: bool = side == "right"
        else:
            start_val = self.start
            step_val = self.step
            shift = side == "left"
        result = (array_value - start_val - int(shift)) // step_val + 1
        if flip:
            result = len(self) - result
        result = np.maximum(np.minimum(result, len(self)), 0)
        if was_scalar:
            return np.intp(result.item())
        return result.astype(np.intp, copy=False)

# End of RangeIndex class with type annotations.
