from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Tuple, Union, Sequence, cast, final
import numpy as np
from datetime import datetime
from pandas._libs import NaT, Timedelta, lib
from pandas._libs.tslibs import BaseOffset, Resolution, Tick, parsing, to_offset
from pandas.compat.numpy import function as nv
from pandas.errors import InvalidIndexError, NullFrequencyError
from pandas.util._decorators import Appender, cache_readonly, doc
from pandas.core.dtypes.common import is_integer, is_list_like
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype, PeriodDtype
from pandas.core.arrays import DatetimeArray, ExtensionArray, PeriodArray, TimedeltaArray
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import Index, _index_shared_docs
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.tools.timedeltas import to_timedelta

if TYPE_CHECKING:
    from collections.abc import Sequence as AbcSequence
    from pandas._typing import Axis, JoinHow, Self, npt

_index_doc_kwargs = dict(ibase._index_doc_kwargs)


class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    _can_hold_strings: bool = False

    @doc(DatetimeLikeArrayMixin.mean)
    def mean(self, *, skipna: bool = True, axis: int = 0) -> Any:
        return self._data.mean(skipna=skipna, axis=axis)

    @property
    def freq(self) -> Optional[Any]:
        return self._data.freq

    @freq.setter
    def freq(self, value: Any) -> None:
        self._data.freq = value

    @property
    def asi8(self) -> np.ndarray:
        return self._data.asi8

    @property
    @doc(DatetimeLikeArrayMixin.freqstr)
    def freqstr(self) -> Optional[str]:
        from pandas import PeriodIndex
        if self._data.freqstr is not None and isinstance(self._data, (PeriodArray, PeriodIndex)):
            freq: Any = PeriodDtype(self._data.freq)._freqstr
            return freq
        else:
            return self._data.freqstr

    @cache_readonly
    @abstractmethod
    def _resolution_obj(self) -> Any:
        ...

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.resolution)
    def resolution(self) -> Any:
        return self._data.resolution

    @cache_readonly
    def hasnans(self) -> bool:
        return self._data._hasna

    def equals(self, other: Any) -> bool:
        if self.is_(other):
            return True
        if not isinstance(other, Index):
            return False
        elif other.dtype.kind in "iufc":
            return False
        elif not isinstance(other, type(self)):
            should_try: bool = False
            inferable: Any = self._data._infer_matches
            if other.dtype == object:
                should_try = other.inferred_type in inferable
            elif isinstance(other.dtype, CategoricalDtype):
                other = cast("CategoricalIndex", other)
                should_try = other.categories.inferred_type in inferable
            if should_try:
                try:
                    other = type(self)(other)
                except (ValueError, TypeError, OverflowError):
                    return False
        if self.dtype != other.dtype:
            return False
        return np.array_equal(self.asi8, other.asi8)

    @Appender(Index.__contains__.__doc__)
    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            self.get_loc(key)
        except (KeyError, TypeError, ValueError, InvalidIndexError):
            return False
        return True

    def _convert_tolerance(self, tolerance: Any, target: Any) -> Any:
        tol_arr: np.ndarray = np.asarray(to_timedelta(tolerance).to_numpy())
        return super()._convert_tolerance(tol_arr, target)

    _default_na_rep: str = "NaT"

    def _format_with_header(self, *, header: List[str], na_rep: str, date_format: Optional[str] = None) -> List[str]:
        return header + list(self._get_values_for_csv(na_rep=na_rep, date_format=date_format))

    @property
    def _formatter_func(self) -> Any:
        return self._data._formatter()

    def _format_attrs(self) -> List[Tuple[str, Any]]:
        attrs: List[Tuple[str, Any]] = super()._format_attrs()
        for attrib in self._attributes:
            if attrib == "freq":
                freq_val: Optional[str] = self.freqstr
                if freq_val is not None:
                    freq_val = repr(freq_val)
                attrs.append(("freq", freq_val))
        return attrs

    @Appender(Index._summary.__doc__)
    def _summary(self, name: Optional[str] = None) -> str:
        result: str = super()._summary(name=name)
        if self.freq:
            result += f"\nFreq: {self.freqstr}"
        return result

    @final
    def _can_partial_date_slice(self, reso: Resolution) -> bool:
        return reso > self._resolution_obj

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime) -> Tuple[datetime, datetime]:
        raise NotImplementedError

    def _parse_with_reso(self, label: Any) -> Tuple[datetime, Resolution]:
        try:
            if self.freq is None or hasattr(self.freq, "rule_code"):
                freq: Any = self.freq
        except NotImplementedError:
            freq = getattr(self, "freqstr", getattr(self, "inferred_freq", None))
        if freq is not None and (not isinstance(freq, str)):
            freqstr: Any = freq.rule_code
        else:
            freqstr = freq
        if isinstance(label, np.str_):
            label = str(label)
        parsed, reso_str = parsing.parse_datetime_string_with_reso(label, freqstr)
        reso_obj: Resolution = Resolution.from_attrname(reso_str)
        return (parsed, reso_obj)

    def _get_string_slice(self, key: str) -> Union[slice, np.ndarray]:
        parsed, reso = self._parse_with_reso(key)
        try:
            return self._partial_date_slice(reso, parsed)
        except KeyError as err:
            raise KeyError(key) from err

    @final
    def _partial_date_slice(self, reso: Resolution, parsed: datetime) -> Union[slice, np.ndarray]:
        if not self._can_partial_date_slice(reso):
            raise ValueError
        t1, t2 = self._parsed_string_to_bounds(reso, parsed)
        vals: Any = self._data._ndarray
        unbox = self._data._unbox
        if self.is_monotonic_increasing:
            if len(self) and ((t1 < self[0] and t2 < self[0]) or (t1 > self[-1] and t2 > self[-1])):
                raise KeyError
            left: int = vals.searchsorted(unbox(t1), side="left")
            right: int = vals.searchsorted(unbox(t2), side="right")
            return slice(left, right)
        else:
            lhs_mask: np.ndarray = vals >= unbox(t1)
            rhs_mask: np.ndarray = vals <= unbox(t2)
            return (lhs_mask & rhs_mask).nonzero()[0]

    def _maybe_cast_slice_bound(self, label: Any, side: str) -> Any:
        if isinstance(label, str):
            try:
                parsed, reso = self._parse_with_reso(label)
            except ValueError as err:
                self._raise_invalid_indexer("slice", label, err)
            lower, upper = self._parsed_string_to_bounds(reso, parsed)
            return lower if side == "left" else upper
        elif not isinstance(label, self._data._recognized_scalars):
            self._raise_invalid_indexer("slice", label)
        return label

    def shift(self, periods: int = 1, freq: Optional[Any] = None) -> Self:
        raise NotImplementedError

    @doc(Index._maybe_cast_listlike_indexer)
    def _maybe_cast_listlike_indexer(self, keyarr: Any) -> Index:
        try:
            res = self._data._validate_listlike(keyarr, allow_object=True)
        except (ValueError, TypeError):
            if not isinstance(keyarr, ExtensionArray):
                res = com.asarray_tuplesafe(keyarr)
            else:
                res = keyarr
        return Index(res, dtype=res.dtype)


class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin, ABC):
    _comparables: List[str] = ["name", "freq"]
    _attributes: List[str] = ["name", "freq"]
    _is_monotonic_increasing = Index.is_monotonic_increasing
    _is_monotonic_decreasing = Index.is_monotonic_decreasing
    _is_unique = Index.is_unique

    @property
    def unit(self) -> Any:
        return self._data.unit

    def as_unit(self, unit: str) -> Self:
        arr = self._data.as_unit(unit)
        return type(self)._simple_new(arr, name=self.name)

    def _with_freq(self, freq: Any) -> Self:
        arr = self._data._with_freq(freq)
        return type(self)._simple_new(arr, name=self._name)

    @property
    def values(self) -> np.ndarray:
        data = self._data._ndarray
        data = data.view()
        data.flags.writeable = False
        return data

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq: Optional[Any] = None) -> Self:
        if freq is not None and freq != self.freq:
            if isinstance(freq, str):
                freq = to_offset(freq)
            offset = periods * freq
            return self + offset
        if periods == 0 or len(self) == 0:
            return self.copy()
        if self.freq is None:
            raise NullFrequencyError("Cannot shift with no freq")
        start = self[0] + periods * self.freq
        end = self[-1] + periods * self.freq
        result = self._data._generate_range(start=start, end=end, periods=None, freq=self.freq, unit=self.unit)
        return type(self)._simple_new(result, name=self.name)

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.inferred_freq)
    def inferred_freq(self) -> Optional[str]:
        return self._data.inferred_freq

    @cache_readonly
    def _as_range_index(self) -> RangeIndex:
        freq_cast: Tick = cast(Tick, self.freq)
        tick: Any = Timedelta(freq_cast).as_unit(self.unit)._value
        rng = range(self[0]._value, self[-1]._value + tick, tick)
        return RangeIndex(rng)

    def _can_range_setop(self, other: Any) -> bool:
        return isinstance(self.freq, Tick) and isinstance(other.freq, Tick)

    def _wrap_range_setop(self, other: Any, res_i8: Any) -> Self:
        new_freq: Optional[Any] = None
        if not len(res_i8):
            new_freq = self.freq
        elif isinstance(res_i8, RangeIndex):
            new_freq = to_offset(Timedelta(res_i8.step, unit=self.unit).as_unit(self.unit))
        res_values = res_i8.values.view(self._data._ndarray.dtype)
        result = type(self._data)._simple_new(res_values, dtype=self.dtype, freq=new_freq)
        return cast("Self", self._wrap_setop_result(other, result))

    def _range_intersect(self, other: Self, sort: bool) -> Self:
        left: RangeIndex = self._as_range_index
        right: RangeIndex = other._as_range_index
        res_i8 = left.intersection(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _range_union(self, other: Self, sort: bool) -> Self:
        left: RangeIndex = self._as_range_index
        right: RangeIndex = other._as_range_index
        res_i8 = left.union(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _intersection(self, other: Self, sort: bool = False) -> Self:
        other = cast("DatetimeTimedeltaMixin", other)
        if self._can_range_setop(other):
            return self._range_intersect(other, sort=sort)
        if not self._can_fast_intersect(other):
            result = Index._intersection(self, other, sort=sort)
            result = self._wrap_setop_result(other, result)
            return result._with_freq(None)._with_freq("infer")
        else:
            return self._fast_intersect(other, sort)

    def _fast_intersect(self, other: Self, sort: bool) -> Any:
        if self[0] <= other[0]:
            left, right = (self, other)
        else:
            left, right = (other, self)
        end = min(left[-1], right[-1])
        start = right[0]
        if end < start:
            result = self[:0]
        else:
            lslice = slice(*left.slice_locs(start, end))
            result = left._values[lslice]
        return result

    def _can_fast_intersect(self, other: Self) -> bool:
        if self.freq is None:
            return False
        elif other.freq != self.freq:
            return False
        elif not self.is_monotonic_increasing:
            return False
        return self.freq.n == 1

    def _can_fast_union(self, other: Self) -> bool:
        freq = self.freq
        if freq is None or freq != other.freq:
            return False
        if not self.is_monotonic_increasing:
            return False
        if len(self) == 0 or len(other) == 0:
            return True
        if self[0] <= other[0]:
            left, right = (self, other)
        else:
            left, right = (other, self)
        right_start = right[0]
        left_end = left[-1]
        return right_start == left_end + freq or right_start in left

    def _fast_union(self, other: Self, sort: Optional[bool] = None) -> Self:
        if self[0] <= other[0]:
            left, right = (self, other)
        elif sort is False:
            left, right = (self, other)
            left_start = left[0]
            loc = right.searchsorted(left_start, side="left")
            right_chunk = right._values[:loc]
            dates = concat_compat((left._values, right_chunk))
            result = type(self)._simple_new(dates, name=self.name)
            return result
        else:
            left, right = (other, self)
        left_end = left[-1]
        right_end = right[-1]
        if left_end < right_end:
            loc = right.searchsorted(left_end, side="right")
            right_chunk = right._values[loc:]
            dates = concat_compat([left._values, right_chunk])
            assert isinstance(dates, type(self._data))
            assert dates._freq == self.freq
            result = type(self)._simple_new(dates)
            return result
        else:
            return left

    def _union(self, other: Self, sort: bool) -> Self:
        assert isinstance(other, type(self))
        assert self.dtype == other.dtype
        if self._can_range_setop(other):
            return self._range_union(other, sort=sort)
        if self._can_fast_union(other):
            result = self._fast_union(other, sort=sort)
            return result
        else:
            return super()._union(other, sort)._with_freq("infer")

    def _get_join_freq(self, other: Self) -> Optional[Any]:
        freq: Optional[Any] = None
        if self._can_fast_union(other):
            freq = self.freq
        return freq

    def _wrap_join_result(
        self, joined: Any, other: Self, lidx: Any, ridx: Any, how: Any
    ) -> Tuple[Self, Any, Any]:
        assert other.dtype == self.dtype, (other.dtype, self.dtype)
        join_index, lidx, ridx = super()._wrap_join_result(joined, other, lidx, ridx, how)
        join_index._data._freq = self._get_join_freq(other)
        return (join_index, lidx, ridx)

    def _get_engine_target(self) -> np.ndarray:
        return self._data._ndarray.view("i8")

    def _from_join_target(self, result: np.ndarray) -> Any:
        result = result.view(self._data._ndarray.dtype)
        return self._data._from_backing_data(result)

    def _get_delete_freq(self, loc: Union[int, slice, Sequence[int]]) -> Optional[Any]:
        freq: Optional[Any] = None
        if self.freq is not None:
            if is_integer(loc):
                if loc in (0, -len(self), -1, len(self) - 1):
                    freq = self.freq
            else:
                if is_list_like(loc):
                    loc = lib.maybe_indices_to_slice(np.asarray(loc, dtype=np.intp), len(self))
                if isinstance(loc, slice) and loc.step in (1, None):
                    if loc.start in (0, None) or loc.stop in (len(self), None):
                        freq = self.freq
        return freq

    def _get_insert_freq(self, loc: int, item: Any) -> Optional[Any]:
        value: Any = self._data._validate_scalar(item)
        item_boxed: Any = self._data._box_func(value)
        freq: Optional[Any] = None
        if self.freq is not None:
            if self.size:
                if item_boxed is NaT:
                    pass
                elif loc in (0, -len(self)) and item_boxed + self.freq == self[0]:
                    freq = self.freq
                elif loc == len(self) and item_boxed - self.freq == self[-1]:
                    freq = self.freq
            elif isinstance(self.freq, Tick):
                freq = self.freq
            elif self.freq.is_on_offset(item_boxed):
                freq = self.freq
        return freq

    @doc(NDArrayBackedExtensionIndex.delete)
    def delete(self, loc: Union[int, slice, Sequence[int]]) -> Self:
        result = super().delete(loc)
        result._data._freq = self._get_delete_freq(loc)
        return result

    @doc(NDArrayBackedExtensionIndex.insert)
    def insert(self, loc: int, item: Any) -> Self:
        result = super().insert(loc, item)
        if isinstance(result, type(self)):
            result._data._freq = self._get_insert_freq(loc, item)
        return result

    @Appender(_index_shared_docs["take"] % _index_doc_kwargs)
    def take(
        self,
        indices: Union[np.ndarray, Sequence[int]],
        axis: int = 0,
        allow_fill: bool = True,
        fill_value: Any = None,
        **kwargs: Any,
    ) -> Self:
        nv.validate_take((), kwargs)
        indices_arr: np.ndarray = np.asarray(indices, dtype=np.intp)
        result = NDArrayBackedExtensionIndex.take(self, indices_arr, axis, allow_fill, fill_value, **kwargs)
        maybe_slice = lib.maybe_indices_to_slice(indices_arr, len(self))
        if isinstance(maybe_slice, slice):
            freq = self._data._get_getitem_freq(maybe_slice)
            result._data._freq = freq
        return result
