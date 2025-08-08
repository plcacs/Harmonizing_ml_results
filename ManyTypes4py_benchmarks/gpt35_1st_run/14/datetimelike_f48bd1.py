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
_index_doc_kwargs: dict = dict(ibase._index_doc_kwargs)

class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    _can_hold_strings: bool = False

    @doc(DatetimeLikeArrayMixin.mean)
    def mean(self, *, skipna: bool = True, axis: int = 0) -> Any:
        return self._data.mean(skipna=skipna, axis=axis)

    @property
    def freq(self) -> Any:
        ...

    @freq.setter
    def freq(self, value: Any) -> None:
        ...

    @property
    def asi8(self) -> Any:
        ...

    @property
    @doc(DatetimeLikeArrayMixin.freqstr)
    def freqstr(self) -> Any:
        ...

    @cache_readonly
    @abstractmethod
    def _resolution_obj(self) -> Any:
        ...

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.resolution)
    def resolution(self) -> Any:
        ...

    @cache_readonly
    def hasnans(self) -> Any:
        ...

    def equals(self, other: Any) -> bool:
        ...

    @Appender(Index.__contains__.__doc__)
    def __contains__(self, key: Any) -> bool:
        ...

    def _convert_tolerance(self, tolerance: Any, target: Any) -> Any:
        ...

    _default_na_rep: str = 'NaT'

    def _format_with_header(self, *, header: Any, na_rep: Any, date_format: Any = None) -> Any:
        ...

    @property
    def _formatter_func(self) -> Any:
        ...

    def _format_attrs(self) -> Any:
        ...

    @Appender(Index._summary.__doc__)
    def _summary(self, name: Any = None) -> Any:
        ...

    @final
    def _can_partial_date_slice(self, reso: Any) -> Any:
        ...

    def _parsed_string_to_bounds(self, reso: Any, parsed: Any) -> Any:
        ...

    def _parse_with_reso(self, label: Any) -> Any:
        ...

    def _get_string_slice(self, key: Any) -> Any:
        ...

    @final
    def _partial_date_slice(self, reso: Any, parsed: Any) -> Any:
        ...

    def _maybe_cast_slice_bound(self, label: Any, side: str) -> Any:
        ...

    def shift(self, periods: int = 1, freq: Any = None) -> Any:
        ...

    @doc(Index._maybe_cast_listlike_indexer)
    def _maybe_cast_listlike_indexer(self, keyarr: Any) -> Any:
        ...

class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin, ABC):
    _comparables: list[str] = ['name', 'freq']
    _attributes: list[str] = ['name', 'freq']
    _is_monotonic_increasing: bool = Index.is_monotonic_increasing
    _is_monotonic_decreasing: bool = Index.is_monotonic_decreasing
    _is_unique: bool = Index.is_unique

    @property
    def unit(self) -> Any:
        ...

    def as_unit(self, unit: str) -> Any:
        ...

    def _with_freq(self, freq: Any) -> Any:
        ...

    @property
    def values(self) -> Any:
        ...

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq: Any = None) -> Any:
        ...

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.inferred_freq)
    def inferred_freq(self) -> Any:
        ...

    @cache_readonly
    def _as_range_index(self) -> Any:
        ...

    def _can_range_setop(self, other: Any) -> Any:
        ...

    def _wrap_range_setop(self, other: Any, res_i8: Any) -> Any:
        ...

    def _range_intersect(self, other: Any, sort: Any) -> Any:
        ...

    def _range_union(self, other: Any, sort: Any) -> Any:
        ...

    def _intersection(self, other: Any, sort: bool = False) -> Any:
        ...

    def _fast_intersect(self, other: Any, sort: Any) -> Any:
        ...

    def _can_fast_intersect(self, other: Any) -> Any:
        ...

    def _can_fast_union(self, other: Any) -> Any:
        ...

    def _fast_union(self, other: Any, sort: Any) -> Any:
        ...

    def _union(self, other: Any, sort: Any) -> Any:
        ...

    def _get_join_freq(self, other: Any) -> Any:
        ...

    def _wrap_join_result(self, joined: Any, other: Any, lidx: Any, ridx: Any, how: Any) -> Any:
        ...

    def _get_engine_target(self) -> Any:
        ...

    def _from_join_target(self, result: Any) -> Any:
        ...

    def _get_delete_freq(self, loc: Any) -> Any:
        ...

    def _get_insert_freq(self, loc: Any, item: Any) -> Any:
        ...

    @doc(NDArrayBackedExtensionIndex.delete)
    def delete(self, loc: Any) -> Any:
        ...

    @doc(NDArrayBackedExtensionIndex.insert)
    def insert(self, loc: Any, item: Any) -> Any:
        ...

    @Appender(_index_shared_docs['take'] % _index_doc_kwargs)
    def take(self, indices: Any, axis: int = 0, allow_fill: bool = True, fill_value: Any = None, **kwargs: Any) -> Any:
        ...
