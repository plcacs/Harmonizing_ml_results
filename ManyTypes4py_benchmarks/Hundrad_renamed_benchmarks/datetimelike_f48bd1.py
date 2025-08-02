"""
Base and utility classes for tseries type pandas objects.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast, final
import numpy as np
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
    from collections.abc import Sequence
    from datetime import datetime
    from pandas._typing import Axis, JoinHow, Self, npt
    from pandas import CategoricalIndex
_index_doc_kwargs = dict(ibase._index_doc_kwargs)


class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    """
    Common ops mixin to support a unified interface datetimelike Index.
    """
    _can_hold_strings = False

    @doc(DatetimeLikeArrayMixin.mean)
    def func_idll450j(self, *, skipna=True, axis=0):
        return self._data.mean(skipna=skipna, axis=axis)

    @property
    def func_75jd9kta(self):
        """
        Return the frequency object if it is set, otherwise None.

        To learn more about the frequency strings, please see
        :ref:`this link<timeseries.offset_aliases>`.

        See Also
        --------
        DatetimeIndex.freq : Return the frequency object if it is set, otherwise None.
        PeriodIndex.freq : Return the frequency object if it is set, otherwise None.

        Examples
        --------
        >>> datetimeindex = pd.date_range(
        ...     "2022-02-22 02:22:22", periods=10, tz="America/Chicago", freq="h"
        ... )
        >>> datetimeindex
        DatetimeIndex(['2022-02-22 02:22:22-06:00', '2022-02-22 03:22:22-06:00',
                       '2022-02-22 04:22:22-06:00', '2022-02-22 05:22:22-06:00',
                       '2022-02-22 06:22:22-06:00', '2022-02-22 07:22:22-06:00',
                       '2022-02-22 08:22:22-06:00', '2022-02-22 09:22:22-06:00',
                       '2022-02-22 10:22:22-06:00', '2022-02-22 11:22:22-06:00'],
                      dtype='datetime64[ns, America/Chicago]', freq='h')
        >>> datetimeindex.freq
        <Hour>
        """
        return self._data.freq

    @freq.setter
    def func_75jd9kta(self, value):
        self._data.freq = value

    @property
    def func_8dckaila(self):
        return self._data.asi8

    @property
    @doc(DatetimeLikeArrayMixin.freqstr)
    def func_ensq9t2s(self):
        from pandas import PeriodIndex
        if self._data.freqstr is not None and isinstance(self._data, (
            PeriodArray, PeriodIndex)):
            freq = PeriodDtype(self._data.freq)._freqstr
            return freq
        else:
            return self._data.freqstr

    @cache_readonly
    @abstractmethod
    def func_auz1hokv(self):
        ...

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.resolution)
    def func_w4fzqk7i(self):
        return self._data.resolution

    @cache_readonly
    def func_afvll1rx(self):
        return self._data._hasna

    def func_p5ne8ph5(self, other):
        """
        Determines if two Index objects contain the same elements.
        """
        if self.is_(other):
            return True
        if not isinstance(other, Index):
            return False
        elif other.dtype.kind in 'iufc':
            return False
        elif not isinstance(other, type(self)):
            should_try = False
            inferable = self._data._infer_matches
            if other.dtype == object:
                should_try = other.inferred_type in inferable
            elif isinstance(other.dtype, CategoricalDtype):
                other = cast('CategoricalIndex', other)
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
    def __contains__(self, key):
        hash(key)
        try:
            self.get_loc(key)
        except (KeyError, TypeError, ValueError, InvalidIndexError):
            return False
        return True

    def func_eep2byn8(self, tolerance, target):
        tolerance = np.asarray(to_timedelta(tolerance).to_numpy())
        return super()._convert_tolerance(tolerance, target)
    _default_na_rep = 'NaT'

    def func_jop5boz4(self, *, header, na_rep, date_format=None):
        return header + list(self._get_values_for_csv(na_rep=na_rep,
            date_format=date_format))

    @property
    def func_7arrn258(self):
        return self._data._formatter()

    def func_2u1flifr(self):
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        attrs = super()._format_attrs()
        for attrib in self._attributes:
            if attrib == 'freq':
                freq = self.freqstr
                if freq is not None:
                    freq = repr(freq)
                attrs.append(('freq', freq))
        return attrs

    @Appender(Index._summary.__doc__)
    def func_tpu2fxaq(self, name=None):
        result = super()._summary(name=name)
        if self.freq:
            result += f'\nFreq: {self.freqstr}'
        return result

    @final
    def func_7t7d69wl(self, reso):
        return reso > self._resolution_obj

    def func_lrk8z7vq(self, reso, parsed):
        raise NotImplementedError

    def func_ufsg14v7(self, label):
        try:
            if self.freq is None or hasattr(self.freq, 'rule_code'):
                freq = self.freq
        except NotImplementedError:
            freq = getattr(self, 'freqstr', getattr(self, 'inferred_freq',
                None))
        if freq is not None and not isinstance(freq, str):
            freqstr = freq.rule_code
        else:
            freqstr = freq
        if isinstance(label, np.str_):
            label = str(label)
        parsed, reso_str = parsing.parse_datetime_string_with_reso(label,
            freqstr)
        reso = Resolution.from_attrname(reso_str)
        return parsed, reso

    def func_q395sznu(self, key):
        parsed, reso = self._parse_with_reso(key)
        try:
            return self._partial_date_slice(reso, parsed)
        except KeyError as err:
            raise KeyError(key) from err

    @final
    def func_ep8kfp6z(self, reso, parsed):
        """
        Parameters
        ----------
        reso : Resolution
        parsed : datetime

        Returns
        -------
        slice or ndarray[intp]
        """
        if not self._can_partial_date_slice(reso):
            raise ValueError
        t1, t2 = self._parsed_string_to_bounds(reso, parsed)
        vals = self._data._ndarray
        unbox = self._data._unbox
        if self.is_monotonic_increasing:
            if len(self) and (t1 < self[0] and t2 < self[0] or t1 > self[-1
                ] and t2 > self[-1]):
                raise KeyError
            left = vals.searchsorted(unbox(t1), side='left')
            right = vals.searchsorted(unbox(t2), side='right')
            return slice(left, right)
        else:
            lhs_mask = vals >= unbox(t1)
            rhs_mask = vals <= unbox(t2)
            return (lhs_mask & rhs_mask).nonzero()[0]

    def func_qg3qy0zi(self, label, side):
        """
        If label is a string, cast it to scalar type according to resolution.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        label : object

        Notes
        -----
        Value of `side` parameter should be validated in caller.
        """
        if isinstance(label, str):
            try:
                parsed, reso = self._parse_with_reso(label)
            except ValueError as err:
                self._raise_invalid_indexer('slice', label, err)
            lower, upper = self._parsed_string_to_bounds(reso, parsed)
            return lower if side == 'left' else upper
        elif not isinstance(label, self._data._recognized_scalars):
            self._raise_invalid_indexer('slice', label)
        return label

    def func_10ikyt66(self, periods=1, freq=None):
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
        raise NotImplementedError

    @doc(Index._maybe_cast_listlike_indexer)
    def func_ejqk4pft(self, keyarr):
        try:
            res = self._data._validate_listlike(keyarr, allow_object=True)
        except (ValueError, TypeError):
            if not isinstance(keyarr, ExtensionArray):
                res = com.asarray_tuplesafe(keyarr)
            else:
                res = keyarr
        return Index(res, dtype=res.dtype)


class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin, ABC):
    """
    Mixin class for methods shared by DatetimeIndex and TimedeltaIndex,
    but not PeriodIndex
    """
    _comparables = ['name', 'freq']
    _attributes = ['name', 'freq']
    _is_monotonic_increasing = Index.is_monotonic_increasing
    _is_monotonic_decreasing = Index.is_monotonic_decreasing
    _is_unique = Index.is_unique

    @property
    def func_p95d409q(self):
        return self._data.unit

    def func_63um197m(self, unit):
        """
        Convert to a dtype with the given unit resolution.

        This method is for converting the dtype of a ``DatetimeIndex`` or
        ``TimedeltaIndex`` to a new dtype with the given unit
        resolution/precision.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}

        Returns
        -------
        same type as self
            Converted to the specified unit.

        See Also
        --------
        Timestamp.as_unit : Convert to the given unit.
        Timedelta.as_unit : Convert to the given unit.
        DatetimeIndex.as_unit : Convert to the given unit.
        TimedeltaIndex.as_unit : Convert to the given unit.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.DatetimeIndex(["2020-01-02 01:02:03.004005006"])
        >>> idx
        DatetimeIndex(['2020-01-02 01:02:03.004005006'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.as_unit("s")
        DatetimeIndex(['2020-01-02 01:02:03'], dtype='datetime64[s]', freq=None)

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta(["1 day 3 min 2 us 42 ns"])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.as_unit("s")
        TimedeltaIndex(['1 days 00:03:00'], dtype='timedelta64[s]', freq=None)
        """
        arr = self._data.as_unit(unit)
        return type(self)._simple_new(arr, name=self.name)

    def func_l4wzq7p7(self, freq):
        arr = self._data._with_freq(freq)
        return type(self)._simple_new(arr, name=self._name)

    @property
    def func_9fhwoipd(self):
        data = self._data._ndarray
        data = data.view()
        data.flags.writeable = False
        return data

    @doc(DatetimeIndexOpsMixin.shift)
    def func_10ikyt66(self, periods=1, freq=None):
        if freq is not None and freq != self.freq:
            if isinstance(freq, str):
                freq = to_offset(freq)
            offset = periods * freq
            return self + offset
        if periods == 0 or len(self) == 0:
            return self.copy()
        if self.freq is None:
            raise NullFrequencyError('Cannot shift with no freq')
        start = self[0] + periods * self.freq
        end = self[-1] + periods * self.freq
        result = self._data._generate_range(start=start, end=end, periods=
            None, freq=self.freq, unit=self.unit)
        return type(self)._simple_new(result, name=self.name)

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.inferred_freq)
    def func_8kj61b54(self):
        return self._data.inferred_freq

    @cache_readonly
    def func_jq2kpgwb(self):
        freq = cast(Tick, self.freq)
        tick = Timedelta(freq).as_unit(self.unit)._value
        rng = range(self[0]._value, self[-1]._value + tick, tick)
        return RangeIndex(rng)

    def func_bgdzb268(self, other):
        return isinstance(self.freq, Tick) and isinstance(other.freq, Tick)

    def func_btftole7(self, other, res_i8):
        new_freq = None
        if not len(res_i8):
            new_freq = self.freq
        elif isinstance(res_i8, RangeIndex):
            new_freq = to_offset(Timedelta(res_i8.step, unit=self.unit).
                as_unit(self.unit))
        res_values = res_i8.values.view(self._data._ndarray.dtype)
        result = type(self._data)._simple_new(res_values, dtype=self.dtype,
            freq=new_freq)
        return cast('Self', self._wrap_setop_result(other, result))

    def func_8djavsxg(self, other, sort):
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.intersection(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def func_ey5on58y(self, other, sort):
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.union(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def func_ci0xk3no(self, other, sort=False):
        """
        intersection specialized to the case with matching dtypes and both non-empty.
        """
        other = cast('DatetimeTimedeltaMixin', other)
        if self._can_range_setop(other):
            return self._range_intersect(other, sort=sort)
        if not self._can_fast_intersect(other):
            result = Index._intersection(self, other, sort=sort)
            result = self._wrap_setop_result(other, result)
            return result._with_freq(None)._with_freq('infer')
        else:
            return self._fast_intersect(other, sort)

    def func_knh3xs3b(self, other, sort):
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self
        end = min(left[-1], right[-1])
        start = right[0]
        if end < start:
            result = self[:0]
        else:
            lslice = slice(*left.slice_locs(start, end))
            result = left._values[lslice]
        return result

    def func_r1tyaq1k(self, other):
        if self.freq is None:
            return False
        elif other.freq != self.freq:
            return False
        elif not self.is_monotonic_increasing:
            return False
        return self.freq.n == 1

    def func_0yy8w4og(self, other):
        freq = self.freq
        if freq is None or freq != other.freq:
            return False
        if not self.is_monotonic_increasing:
            return False
        if len(self) == 0 or len(other) == 0:
            return True
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self
        right_start = right[0]
        left_end = left[-1]
        return right_start == left_end + freq or right_start in left

    def func_rx8iwfqx(self, other, sort=None):
        if self[0] <= other[0]:
            left, right = self, other
        elif sort is False:
            left, right = self, other
            left_start = left[0]
            loc = right.searchsorted(left_start, side='left')
            right_chunk = right._values[:loc]
            dates = concat_compat((left._values, right_chunk))
            result = type(self)._simple_new(dates, name=self.name)
            return result
        else:
            left, right = other, self
        left_end = left[-1]
        right_end = right[-1]
        if left_end < right_end:
            loc = right.searchsorted(left_end, side='right')
            right_chunk = right._values[loc:]
            dates = concat_compat([left._values, right_chunk])
            assert isinstance(dates, type(self._data))
            assert dates._freq == self.freq
            result = type(self)._simple_new(dates)
            return result
        else:
            return left

    def func_jab5egcm(self, other, sort):
        assert isinstance(other, type(self))
        assert self.dtype == other.dtype
        if self._can_range_setop(other):
            return self._range_union(other, sort=sort)
        if self._can_fast_union(other):
            result = self._fast_union(other, sort=sort)
            return result
        else:
            return super()._union(other, sort)._with_freq('infer')

    def func_orxlkoh7(self, other):
        """
        Get the freq to attach to the result of a join operation.
        """
        freq = None
        if self._can_fast_union(other):
            freq = self.freq
        return freq

    def func_a2updnf0(self, joined, other, lidx, ridx, how):
        assert other.dtype == self.dtype, (other.dtype, self.dtype)
        join_index, lidx, ridx = super()._wrap_join_result(joined, other,
            lidx, ridx, how)
        join_index._data._freq = self._get_join_freq(other)
        return join_index, lidx, ridx

    def func_c8beu2am(self):
        return self._data._ndarray.view('i8')

    def func_6qc2rmdx(self, result):
        result = result.view(self._data._ndarray.dtype)
        return self._data._from_backing_data(result)

    def func_zjmzmcpt(self, loc):
        """
        Find the `freq` for self.delete(loc).
        """
        freq = None
        if self.freq is not None:
            if is_integer(loc):
                if loc in (0, -len(self), -1, len(self) - 1):
                    freq = self.freq
            else:
                if is_list_like(loc):
                    loc = lib.maybe_indices_to_slice(np.asarray(loc, dtype=
                        np.intp), len(self))
                if isinstance(loc, slice) and loc.step in (1, None):
                    if loc.start in (0, None) or loc.stop in (len(self), None):
                        freq = self.freq
        return freq

    def func_35c0wddc(self, loc, item):
        """
        Find the `freq` for self.insert(loc, item).
        """
        value = self._data._validate_scalar(item)
        item = self._data._box_func(value)
        freq = None
        if self.freq is not None:
            if self.size:
                if item is NaT:
                    pass
                elif loc in (0, -len(self)) and item + self.freq == self[0]:
                    freq = self.freq
                elif loc == len(self) and item - self.freq == self[-1]:
                    freq = self.freq
            elif isinstance(self.freq, Tick):
                freq = self.freq
            elif self.freq.is_on_offset(item):
                freq = self.freq
        return freq

    @doc(NDArrayBackedExtensionIndex.delete)
    def func_ko365jtz(self, loc):
        result = super().delete(loc)
        result._data._freq = self._get_delete_freq(loc)
        return result

    @doc(NDArrayBackedExtensionIndex.insert)
    def func_fbmyaqn7(self, loc, item):
        result = super().insert(loc, item)
        if isinstance(result, type(self)):
            result._data._freq = self._get_insert_freq(loc, item)
        return result

    @Appender(_index_shared_docs['take'] % _index_doc_kwargs)
    def func_x11apqcv(self, indices, axis=0, allow_fill=True, fill_value=
        None, **kwargs):
        nv.validate_take((), kwargs)
        indices = np.asarray(indices, dtype=np.intp)
        result = NDArrayBackedExtensionIndex.take(self, indices, axis,
            allow_fill, fill_value, **kwargs)
        maybe_slice = lib.maybe_indices_to_slice(indices, len(self))
        if isinstance(maybe_slice, slice):
            freq = self._data._get_getitem_freq(maybe_slice)
            result._data._freq = freq
        return result
