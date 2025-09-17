#!/usr/bin/env python3
"""
Define extension dtypes.
"""
from __future__ import annotations
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import re
import warnings
import zoneinfo
from typing import Any, Optional, Dict, Union, Sequence, Type, cast
import numpy as np
from pandas._config.config import get_option
from pandas._libs import lib, missing as libmissing
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import BaseOffset, NaT, NaTType, Period, Timedelta, Timestamp, timezones, to_offset, tz_compare
from pandas._libs.tslibs.dtypes import PeriodDtypeBase, abbrev_to_npy_unit
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._decorators import set_module
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype, StorageExtensionDtype, register_extension_dtype
from pandas.core.dtypes.generic import ABCCategoricalIndex, ABCIndex, ABCRangeIndex
from pandas.core.dtypes.inference import is_bool, is_list_like
if not pa_version_under10p1:
    import pyarrow as pa
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from datetime import tzinfo
    import pyarrow as pa
    from pandas._typing import Dtype, DtypeObj, IntervalClosedType, Ordered, Scalar, Self, npt, type_t
    from pandas import Categorical, CategoricalIndex, DatetimeIndex, Index, IntervalIndex, PeriodIndex
    from pandas.core.arrays import BaseMaskedArray, DatetimeArray, IntervalArray, NumpyExtensionArray, PeriodArray, SparseArray
    from pandas.core.arrays.arrow import ArrowExtensionArray
str_type = str


class PandasExtensionDtype(ExtensionDtype):
    """
    A np.dtype duck-typed class, suitable for holding a custom dtype.

    THIS IS NOT A REAL NUMPY DTYPE
    """
    subdtype = None
    num = 100
    shape = ()
    itemsize = 8
    base = None
    isbuiltin = 0
    isnative = 0
    _cache_dtypes: Dict[Any, Any] = {}

    def __repr__(self) -> str:
        """
        Return a string representation for a particular object.
        """
        return str(self)

    def __hash__(self) -> int:
        raise NotImplementedError('sub-classes should implement an __hash__ method')

    def __getstate__(self) -> Dict[str, Any]:
        return {k: getattr(self, k, None) for k in self._metadata}

    @classmethod
    def reset_cache(cls) -> None:
        """clear the cache"""
        cls._cache_dtypes = {}


class CategoricalDtypeType(type):
    """
    the type of CategoricalDtype, this metaclass determines subclass ability
    """
    ...


@register_extension_dtype
@set_module('pandas')
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype):
    """
    Type for categorical data with the categories and orderedness.
    """
    name: str = 'category'
    type = CategoricalDtypeType
    kind: str = 'O'
    str: str = '|O08'
    base = np.dtype('O')
    _metadata: tuple[str, ...] = ('categories', 'ordered')
    _cache_dtypes: Dict[Any, Any] = {}
    _supports_2d: bool = False
    _can_fast_transpose: bool = False

    def __init__(self, categories: Optional[Any] = None, ordered: Optional[bool] = False) -> None:
        self._finalize(categories, ordered, fastpath=False)

    @classmethod
    def _from_fastpath(cls, categories: Optional[Any] = None, ordered: Optional[bool] = None) -> CategoricalDtype:
        self = cls.__new__(cls)
        self._finalize(categories, ordered, fastpath=True)
        return self

    @classmethod
    def _from_categorical_dtype(cls, dtype: CategoricalDtype, categories: Optional[Any] = None, ordered: Optional[bool] = None) -> CategoricalDtype:
        if categories is ordered is None:
            return dtype
        if categories is None:
            categories = dtype.categories
        if ordered is None:
            ordered = dtype.ordered
        return cls(categories, ordered)

    @classmethod
    def _from_values_or_dtype(
        cls, 
        values: Optional[Any] = None, 
        categories: Optional[Any] = None, 
        ordered: Optional[bool] = None, 
        dtype: Optional[Union[CategoricalDtype, str]] = None
    ) -> CategoricalDtype:
        if dtype is not None:
            if isinstance(dtype, str):
                if dtype == 'category':
                    if ordered is None and cls.is_dtype(values):
                        ordered = values.dtype.ordered  # type: ignore[attr-defined]
                    dtype = CategoricalDtype(categories, ordered)
                else:
                    raise ValueError(f'Unknown dtype {dtype!r}')
            elif categories is not None or ordered is not None:
                raise ValueError('Cannot specify `categories` or `ordered` together with `dtype`.')
            elif not isinstance(dtype, CategoricalDtype):
                raise ValueError(f'Cannot not construct CategoricalDtype from {dtype}')
        elif cls.is_dtype(values):
            dtype = values.dtype._from_categorical_dtype(values.dtype, categories, ordered)  # type: ignore[attr-defined]
        else:
            dtype = CategoricalDtype(categories, ordered)
        return cast(CategoricalDtype, dtype)

    @classmethod
    def construct_from_string(cls, string: str) -> CategoricalDtype:
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if string != cls.name:
            raise TypeError(f"Cannot construct a 'CategoricalDtype' from '{string}'")
        return cls(ordered=None)

    def _finalize(self, categories: Any, ordered: Optional[bool], fastpath: bool) -> None:
        if ordered is not None:
            self.validate_ordered(ordered)
        if categories is not None:
            categories = self.validate_categories(categories, fastpath=fastpath)
        self._categories = categories
        self._ordered = ordered

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._categories = state.pop('categories', None)
        self._ordered = state.pop('ordered', False)

    def __hash__(self) -> int:
        if self.categories is None:
            if self.ordered:
                return -1
            else:
                return -2
        return int(self._hash_categories)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not (hasattr(other, 'ordered') and hasattr(other, 'categories')):
            return False
        elif self.categories is None or other.categories is None:
            return self.categories is other.categories
        elif self.ordered or other.ordered:
            return self.ordered == other.ordered and self.categories.equals(other.categories)
        else:
            left = self.categories
            right = other.categories
            if not left.dtype == right.dtype:
                return False
            if len(left) != len(right):
                return False
            if self.categories.equals(other.categories):
                return True
            if left.dtype != object:
                indexer = left.get_indexer(right)
                return bool((indexer != -1).all())
            return set(left) == set(right)

    def __repr__(self) -> str:
        if self.categories is None:
            data = 'None'
            dtype = 'None'
        else:
            data = self.categories._format_data(name=type(self).__name__)
            if isinstance(self.categories, ABCRangeIndex):
                data = str(self.categories._range)
            data = data.rstrip(', ')
            dtype = self.categories.dtype
        return f'CategoricalDtype(categories={data}, ordered={self.ordered}, categories_dtype={dtype})'

    @cache_readonly
    def _hash_categories(self) -> np.ndarray:
        from pandas.core.util.hashing import combine_hash_arrays, hash_array, hash_tuples
        categories = self.categories
        ordered = self.ordered
        if len(categories) and isinstance(categories[0], tuple):
            cat_list = list(categories)
            cat_array = hash_tuples(cat_list)
        else:
            if categories.dtype == 'O' and len({type(x) for x in categories}) != 1:
                hashed = hash((tuple(categories), ordered))
                return np.array(hashed)
            from pandas.core.dtypes.dtypes import DatetimeTZDtype
            if DatetimeTZDtype.is_dtype(categories.dtype):  # type: ignore[attr-defined]
                categories = categories.view('datetime64[ns]')
            cat_array = hash_array(np.asarray(categories), categorize=False)
        if ordered:
            cat_array = np.vstack([cat_array, np.arange(len(cat_array), dtype=cat_array.dtype)])
        else:
            cat_array = cat_array.reshape(1, len(cat_array))
        combined_hashed = combine_hash_arrays(iter(cat_array), num_items=len(cat_array))
        return np.bitwise_xor.reduce(combined_hashed)

    @classmethod
    def construct_array_type(cls) -> Type[Any]:
        from pandas import Categorical
        return Categorical

    @staticmethod
    def validate_ordered(ordered: Any) -> None:
        if not is_bool(ordered):
            raise TypeError("'ordered' must either be 'True' or 'False'")

    @staticmethod
    def validate_categories(categories: Any, fastpath: bool = False) -> Any:
        from pandas.core.indexes.base import Index
        if not fastpath and (not is_list_like(categories)):
            raise TypeError(f"Parameter 'categories' must be list-like, was {categories!r}")
        from pandas.core.dtypes.generic import ABCIndex, ABCCategoricalIndex
        if not isinstance(categories, ABCIndex):
            categories = Index._with_infer(categories, tupleize_cols=False)
        if not fastpath:
            if categories.hasnans:
                raise ValueError('Categorical categories cannot be null')
            if not categories.is_unique:
                raise ValueError('Categorical categories must be unique')
        if isinstance(categories, ABCCategoricalIndex):
            categories = categories.categories
        return categories

    def update_dtype(self, dtype: Union[CategoricalDtype, str]) -> CategoricalDtype:
        if isinstance(dtype, str) and dtype == 'category':
            return self
        elif not self.is_dtype(dtype):
            raise ValueError(f'a CategoricalDtype must be passed to perform an update, got {dtype!r}')
        else:
            dtype = cast(CategoricalDtype, dtype)
        if isinstance(dtype, CategoricalDtype) and dtype.categories is not None and (dtype.ordered is not None):
            return dtype
        new_categories = dtype.categories if dtype.categories is not None else self.categories
        new_ordered = dtype.ordered if dtype.ordered is not None else self.ordered
        return CategoricalDtype(new_categories, new_ordered)

    @property
    def categories(self) -> Any:
        return self._categories

    @property
    def ordered(self) -> Optional[bool]:
        return self._ordered

    @property
    def _is_boolean(self) -> bool:
        from pandas.core.dtypes.common import is_bool_dtype
        return is_bool_dtype(self.categories)

    def _get_common_dtype(self, dtypes: Sequence[Any]) -> Optional[Any]:
        if all((isinstance(x, CategoricalDtype) for x in dtypes)):
            first = dtypes[0]
            if all((first == other for other in dtypes[1:])):
                return first
        non_init_cats = [isinstance(x, CategoricalDtype) and x.categories is None for x in dtypes]
        if all(non_init_cats):
            return self
        elif any(non_init_cats):
            return None
        subtypes = (x.subtype if hasattr(x, 'subtype') and isinstance(x, SparseDtype) else x for x in dtypes)
        non_cat_dtypes = [x.categories.dtype if isinstance(x, CategoricalDtype) else x for x in subtypes]
        from pandas.core.dtypes.cast import find_common_type
        return find_common_type(non_cat_dtypes)

    @cache_readonly
    def index_class(self) -> Type[Any]:
        from pandas import CategoricalIndex
        return CategoricalIndex


@register_extension_dtype
@set_module('pandas')
class DatetimeTZDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for timezone-aware datetime data.
    """
    type = Timestamp
    kind: str = 'M'
    num: int = 101
    _metadata: tuple[str, ...] = ('unit', 'tz')
    _match = re.compile('(datetime64|M8)\\[(?P<unit>.+), (?P<tz>.+)\\]')
    _cache_dtypes: Dict[Any, Any] = {}
    _supports_2d: bool = True
    _can_fast_transpose: bool = True

    @property
    def na_value(self) -> Any:
        return NaT

    @cache_readonly
    def base(self) -> np.dtype:
        return np.dtype(f'M8[{self.unit}]')

    @cache_readonly
    def str(self) -> str:
        return f'|M8[{self.unit}]'

    def __init__(self, unit: str = 'ns', tz: Optional[Union[str, int, tzinfo]] = None) -> None:
        if isinstance(unit, DatetimeTZDtype):
            unit, tz = (unit.unit, unit.tz)
        if unit != 'ns':
            if isinstance(unit, str) and tz is None:
                result = type(self).construct_from_string(unit)
                unit = result.unit
                tz = result.tz
                msg = f"Passing a dtype alias like 'datetime64[ns, {tz}]' to DatetimeTZDtype is no longer supported. Use 'DatetimeTZDtype.construct_from_string()' instead."
                raise ValueError(msg)
            if unit not in ['s', 'ms', 'us', 'ns']:
                raise ValueError('DatetimeTZDtype only supports s, ms, us, ns units')
        if tz:
            tz = timezones.maybe_get_tz(tz)
            tz = timezones.tz_standardize(tz)
        elif tz is not None:
            raise zoneinfo.ZoneInfoNotFoundError(tz)
        if tz is None:
            raise TypeError("A 'tz' is required.")
        self._unit = unit
        self._tz = tz

    @cache_readonly
    def _creso(self) -> Any:
        return abbrev_to_npy_unit(self.unit)

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def tz(self) -> Any:
        return self._tz

    @classmethod
    def construct_array_type(cls) -> Type[Any]:
        from pandas.core.arrays import DatetimeArray
        return DatetimeArray

    @classmethod
    def construct_from_string(cls, string: str) -> DatetimeTZDtype:
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        match = cls._match.match(string)
        if match:
            d = match.groupdict()
            try:
                return cls(unit=d['unit'], tz=d['tz'])
            except (KeyError, TypeError, ValueError) as err:
                raise TypeError(msg) from err
        raise TypeError(msg)

    def __str__(self) -> str:
        return f'datetime64[{self.unit}, {self.tz}]'

    @property
    def name(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            if other.startswith('M8['):
                other = f'datetime64[{other[3:]}'
            return other == self.name
        return isinstance(other, DatetimeTZDtype) and self.unit == other.unit and tz_compare(self.tz, other.tz)

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]) -> Any:
        import pyarrow
        from pandas.core.arrays import DatetimeArray
        array = array.cast(pyarrow.timestamp(unit=self._unit), safe=True)  # type: ignore[union-attr]
        if isinstance(array, pyarrow.Array):
            np_arr = array.to_numpy(zero_copy_only=False)
        else:
            np_arr = array.to_numpy()
        return DatetimeArray._simple_new(np_arr, dtype=self)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._tz = state['tz']
        self._unit = state['unit']

    def _get_common_dtype(self, dtypes: Sequence[Any]) -> Optional[Any]:
        if all((isinstance(t, DatetimeTZDtype) and t.tz == self.tz for t in dtypes)):
            np_dtype = np.max([cast(DatetimeTZDtype, t).base for t in [self, *dtypes]])
            unit = np.datetime_data(np_dtype)[0]
            return type(self)(unit=unit, tz=self.tz)
        return super()._get_common_dtype(dtypes)

    @cache_readonly
    def index_class(self) -> Type[Any]:
        from pandas import DatetimeIndex
        return DatetimeIndex


@register_extension_dtype
@set_module('pandas')
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype):
    """
    An ExtensionDtype for Period data.
    """
    type = Period
    kind: str = 'O'
    str: str = '|O08'
    base = np.dtype('O')
    num: int = 102
    _metadata: tuple[str, ...] = ('freq',)
    _match = re.compile('(P|p)eriod\\[(?P<freq>.+)\\]')
    _cache_dtypes: Dict[Any, Any] = {}
    __hash__ = PeriodDtypeBase.__hash__  # type: ignore[attr-defined]
    _supports_2d: bool = True
    _can_fast_transpose: bool = True

    def __new__(cls, freq: Union[str, BaseOffset, PeriodDtype]) -> PeriodDtype:
        if isinstance(freq, PeriodDtype):
            return freq
        if not isinstance(freq, BaseOffset):
            freq = cls._parse_dtype_strict(freq)
        if isinstance(freq, BDay):
            warnings.warn("PeriodDtype[B] is deprecated and will be removed in a future version. Use a DatetimeIndex with freq='B' instead", FutureWarning, stacklevel=find_stack_level())
        try:
            dtype_code = cls._cache_dtypes[freq]
        except KeyError:
            dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]
            cls._cache_dtypes[freq] = dtype_code
        u = PeriodDtypeBase.__new__(cls, dtype_code, freq.n)
        u._freq = freq
        return u

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.name,))

    @property
    def freq(self) -> BaseOffset:
        return self._freq

    @classmethod
    def _parse_dtype_strict(cls, freq: Any) -> BaseOffset:
        if isinstance(freq, str):
            if freq.startswith(('Period[', 'period[')):
                m = cls._match.search(freq)
                if m is not None:
                    freq = m.group('freq')
            from pandas._libs.tslibs.offsets import to_offset
            freq_offset = to_offset(freq, is_period=True)
            if freq_offset is not None:
                return freq_offset
        raise TypeError(f'PeriodDtype argument should be string or BaseOffset, got {type(freq).__name__}')

    @classmethod
    def construct_from_string(cls, string: str) -> PeriodDtype:
        if isinstance(string, str) and (string.startswith('period[') or string.startswith('Period[')) or isinstance(string, BaseOffset):
            try:
                return cls(freq=string)
            except ValueError:
                pass
        if isinstance(string, str):
            msg = f"Cannot construct a 'PeriodDtype' from '{string}'"
        else:
            msg = f"'construct_from_string' expects a string, got {type(string)}"
        raise TypeError(msg)

    def __str__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return f'period[{self._freqstr}]'  # type: ignore[attr-defined]

    @property
    def na_value(self) -> Any:
        return NaT

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return other[:1].lower() + other[1:] == self.name
        return super().__eq__(other)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @classmethod
    def is_dtype(cls, dtype: Any) -> bool:
        if isinstance(dtype, str):
            if dtype.startswith(('period[', 'Period[')):
                try:
                    return cls._parse_dtype_strict(dtype) is not None
                except ValueError:
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    @classmethod
    def construct_array_type(cls) -> Type[Any]:
        from pandas.core.arrays import PeriodArray
        return PeriodArray

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]) -> Any:
        import pyarrow
        from pandas.core.arrays import PeriodArray
        from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks
        results = []
        for arr in chunks:
            data, mask = pyarrow_array_to_numpy_and_mask(arr, dtype=np.dtype(np.int64))
            parr = PeriodArray(data.copy(), dtype=self, copy=False)
            parr[~mask] = NaT
            results.append(parr)
        if not results:
            return PeriodArray(np.array([], dtype='int64'), dtype=self, copy=False)
        return PeriodArray._concat_same_type(results)

    @cache_readonly
    def index_class(self) -> Type[Any]:
        from pandas import PeriodIndex
        return PeriodIndex


@register_extension_dtype
@set_module('pandas')
class IntervalDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for Interval data.
    """
    name: str = 'interval'
    kind: str = 'O'
    str: str = '|O08'
    base = np.dtype('O')
    num: int = 103
    _metadata: tuple[str, ...] = ('subtype', 'closed')
    _match = re.compile('(I|i)nterval\\[(?P<subtype>[^,]+(\\[.+\\])?)(, (?P<closed>(right|left|both|neither)))?\\]')
    _cache_dtypes: Dict[Any, Any] = {}

    def __init__(self, subtype: Optional[Any] = None, closed: Optional[str] = None) -> None:
        from pandas.core.dtypes.common import is_string_dtype, pandas_dtype
        if closed is not None and closed not in {'right', 'left', 'both', 'neither'}:
            raise ValueError("closed must be one of 'right', 'left', 'both', 'neither'")
        if isinstance(subtype, IntervalDtype):
            if closed is not None and closed != subtype.closed:
                raise ValueError("dtype.closed and 'closed' do not match. Try IntervalDtype(dtype.subtype, closed) instead.")
            self._subtype = subtype._subtype
            self._closed = subtype._closed
        elif subtype is None:
            self._subtype = None
            self._closed = closed
        elif isinstance(subtype, str) and subtype.lower() == 'interval':
            self._subtype = None
            self._closed = closed
        else:
            if isinstance(subtype, str):
                m = IntervalDtype._match.search(subtype)
                if m is not None:
                    gd = m.groupdict()
                    subtype = gd['subtype']
                    if gd.get('closed', None) is not None:
                        if closed is not None:
                            if closed != gd['closed']:
                                raise ValueError("'closed' keyword does not match value specified in dtype string")
                        closed = gd['closed']
            from pandas.core.dtypes.common import pandas_dtype
            try:
                subtype = pandas_dtype(subtype)
            except TypeError as err:
                raise TypeError('could not construct IntervalDtype') from err
            from pandas.core.dtypes.generic import ABCCategoricalIndex
            from pandas.core.dtypes.common import is_string_dtype
            if CategoricalDtype.is_dtype(subtype) or is_string_dtype(subtype):
                msg = 'category, object, and string subtypes are not supported for IntervalDtype'
                raise TypeError(msg)
            self._subtype = subtype
            self._closed = closed

    @cache_readonly
    def _can_hold_na(self) -> bool:
        subtype = self._subtype
        if subtype is None:
            raise NotImplementedError('_can_hold_na is not defined for partially-initialized IntervalDtype')
        if subtype.kind in 'iu':
            return False
        return True

    @property
    def closed(self) -> Optional[str]:
        return self._closed

    @property
    def subtype(self) -> Any:
        return self._subtype

    @classmethod
    def construct_array_type(cls) -> Type[Any]:
        from pandas.core.arrays import IntervalArray
        return IntervalArray

    @classmethod
    def construct_from_string(cls, string: str) -> IntervalDtype:
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if string.lower() == 'interval' or cls._match.search(string) is not None:
            return cls(string)
        msg = f"Cannot construct a 'IntervalDtype' from '{string}'.\n\nIncorrectly formatted string passed to constructor. Valid formats include Interval or Interval[dtype] where dtype is numeric, datetime, or timedelta"
        raise TypeError(msg)

    @property
    def type(self) -> Type[Interval]:
        return Interval

    def __str__(self) -> str:
        if self.subtype is None:
            return 'interval'
        if self.closed is None:
            return f'interval[{self.subtype}]'
        return f'interval[{self.subtype}, {self.closed}]'

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return other.lower() in (self.name.lower(), str(self).lower())
        elif not isinstance(other, IntervalDtype):
            return False
        elif self.subtype is None or other.subtype is None:
            return True
        elif self.closed != other.closed:
            return False
        else:
            return self.subtype == other.subtype

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._subtype = state['subtype']
        self._closed = state.pop('closed', None)

    @classmethod
    def is_dtype(cls, dtype: Any) -> bool:
        dtype = getattr(dtype, 'dtype', dtype)
        if isinstance(dtype, str):
            if dtype.lower().startswith('interval'):
                try:
                    return cls.construct_from_string(dtype) is not None
                except (ValueError, TypeError):
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]) -> Any:
        import pyarrow
        from pandas.core.arrays import IntervalArray
        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks
        results = []
        for arr in chunks:
            if isinstance(arr, pyarrow.ExtensionArray):
                arr = arr.storage
            left = np.asarray(arr.field('left'), dtype=self.subtype)  # type: ignore
            right = np.asarray(arr.field('right'), dtype=self.subtype)  # type: ignore
            iarr = IntervalArray.from_arrays(left, right, closed=self.closed)
            results.append(iarr)
        if not results:
            return IntervalArray.from_arrays(np.array([], dtype=self.subtype), np.array([], dtype=self.subtype), closed=self.closed)
        return IntervalArray._concat_same_type(results)

    def _get_common_dtype(self, dtypes: Sequence[Any]) -> Optional[Any]:
        if not all((isinstance(x, IntervalDtype) for x in dtypes)):
            return None
        closed = cast(IntervalDtype, dtypes[0]).closed
        if not all((cast(IntervalDtype, x).closed == closed for x in dtypes)):
            return np.dtype(object)
        from pandas.core.dtypes.cast import find_common_type
        common = find_common_type([cast(IntervalDtype, x).subtype for x in dtypes])
        if common == object:
            return np.dtype(object)
        return IntervalDtype(common, closed=closed)

    @cache_readonly
    def index_class(self) -> Type[Any]:
        from pandas import IntervalIndex
        return IntervalIndex


class NumpyEADtype(ExtensionDtype):
    """
    A Pandas ExtensionDtype for NumPy dtypes.
    """
    _metadata: tuple[str, ...] = ('_dtype',)
    _supports_2d: bool = False
    _can_fast_transpose: bool = False

    def __init__(self, dtype: Any) -> None:
        if isinstance(dtype, NumpyEADtype):
            dtype = dtype.numpy_dtype
        self._dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        return f'NumpyEADtype({self.name!r})'

    @property
    def numpy_dtype(self) -> np.dtype:
        return self._dtype

    @property
    def name(self) -> str:
        return self._dtype.name

    @property
    def type(self) -> Any:
        return self._dtype.type

    @property
    def _is_numeric(self) -> bool:
        return self.kind in set('biufcMOSUV')

    @property
    def _is_boolean(self) -> bool:
        return self.kind == 'b'

    @classmethod
    def construct_from_string(cls, string: str) -> NumpyEADtype:
        try:
            dtype = np.dtype(string)
        except TypeError as err:
            if not isinstance(string, str):
                msg = f"'construct_from_string' expects a string, got {type(string)}"
            else:
                msg = f"Cannot construct a 'NumpyEADtype' from '{string}'"
            raise TypeError(msg) from err
        return cls(dtype)

    @classmethod
    def construct_array_type(cls) -> Type[Any]:
        from pandas.core.arrays import NumpyExtensionArray
        return NumpyExtensionArray

    @property
    def kind(self) -> str:
        return self._dtype.kind

    @property
    def itemsize(self) -> int:
        return self._dtype.itemsize


class BaseMaskedDtype(ExtensionDtype):
    """
    Base class for dtypes for BaseMaskedArray subclasses.
    """
    base = None

    @property
    def _truthy_value(self) -> Any:
        if self.kind == 'f':
            return 1.0
        if self.kind in 'iu':
            return 1
        return True

    @property
    def _falsey_value(self) -> Any:
        if self.kind == 'f':
            return 0.0
        if self.kind in 'iu':
            return 0
        return False

    @property
    def na_value(self) -> Any:
        return libmissing.NA

    @cache_readonly
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.type)

    @cache_readonly
    def kind(self) -> str:
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self) -> int:
        return self.numpy_dtype.itemsize

    @classmethod
    def construct_array_type(cls) -> Type[Any]:
        raise NotImplementedError

    @classmethod
    def from_numpy_dtype(cls, dtype: np.dtype) -> BaseMaskedDtype:
        if dtype.kind == 'b':
            from pandas.core.arrays.boolean import BooleanDtype
            return BooleanDtype()
        elif dtype.kind in 'iu':
            from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
            return NUMPY_INT_TO_DTYPE[dtype]
        elif dtype.kind == 'f':
            from pandas.core.arrays.floating import NUMPY_FLOAT_TO_DTYPE
            return NUMPY_FLOAT_TO_DTYPE[dtype]
        else:
            raise NotImplementedError(dtype)

    def _get_common_dtype(self, dtypes: Sequence[Any]) -> Optional[Any]:
        from pandas.core.dtypes.cast import find_common_type
        new_dtype = find_common_type([dtype.numpy_dtype if isinstance(dtype, BaseMaskedDtype) else dtype for dtype in dtypes])
        if not isinstance(new_dtype, np.dtype):
            return None
        try:
            return type(self).from_numpy_dtype(new_dtype)
        except (KeyError, NotImplementedError):
            return None


@register_extension_dtype
@set_module('pandas')
class SparseDtype(ExtensionDtype):
    """
    Dtype for data stored in SparseArray.
    """
    _is_immutable: bool = True
    _metadata: tuple[str, ...] = ('_dtype', '_fill_value', '_is_na_fill_value')

    def __init__(self, dtype: Any = np.float64, fill_value: Optional[Any] = None) -> None:
        if isinstance(dtype, type(self)):
            if fill_value is None:
                fill_value = dtype.fill_value
            dtype = dtype.subtype
        from pandas.core.dtypes.common import is_string_dtype, pandas_dtype
        from pandas.core.dtypes.missing import na_value_for_dtype
        dtype = pandas_dtype(dtype)
        if is_string_dtype(dtype):
            dtype = np.dtype('object')
        if not isinstance(dtype, np.dtype):
            raise TypeError('SparseDtype subtype must be a numpy dtype')
        if fill_value is None:
            fill_value = na_value_for_dtype(dtype)
        self._dtype = dtype
        self._fill_value = fill_value
        self._check_fill_value()

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False
        if isinstance(other, type(self)):
            subtype = self.subtype == other.subtype
            if self._is_na_fill_value or other._is_na_fill_value:
                fill_value = isinstance(self.fill_value, type(other.fill_value)) or isinstance(other.fill_value, type(self.fill_value))
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'elementwise comparison failed', category=DeprecationWarning)
                    fill_value = self.fill_value == other.fill_value
            return subtype and fill_value
        return False

    @property
    def fill_value(self) -> Any:
        return self._fill_value

    def _check_fill_value(self) -> None:
        if not lib.is_scalar(self._fill_value):
            raise ValueError(f'fill_value must be a scalar. Got {self._fill_value} instead')
        from pandas.core.dtypes.cast import can_hold_element
        from pandas.core.dtypes.missing import is_valid_na_for_dtype, isna
        from pandas.core.construction import ensure_wrapped_if_datetimelike
        val = self._fill_value
        if isna(val):
            if not is_valid_na_for_dtype(val, self.subtype):
                raise ValueError('fill_value must be a valid value for the SparseDtype.subtype')
        else:
            dummy = np.empty(0, dtype=self.subtype)
            dummy = ensure_wrapped_if_datetimelike(dummy)
            if not can_hold_element(dummy, val):
                raise ValueError('fill_value must be a valid value for the SparseDtype.subtype')

    @property
    def _is_na_fill_value(self) -> bool:
        from pandas import isna
        return isna(self.fill_value)

    @property
    def _is_numeric(self) -> bool:
        return not self.subtype == object

    @property
    def _is_boolean(self) -> bool:
        return self.subtype.kind == 'b'

    @property
    def kind(self) -> str:
        return self.subtype.kind

    @property
    def type(self) -> Any:
        return self.subtype.type

    @property
    def subtype(self) -> np.dtype:
        return self._dtype

    @property
    def name(self) -> str:
        return f'Sparse[{self.subtype.name}, {self.fill_value!r}]'

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def construct_array_type(cls) -> Type[Any]:
        from pandas.core.arrays.sparse.array import SparseArray
        return SparseArray

    @classmethod
    def construct_from_string(cls, string: str) -> SparseDtype:
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        msg = f"Cannot construct a 'SparseDtype' from '{string}'"
        if string.startswith('Sparse'):
            try:
                sub_type, has_fill_value = cls._parse_subtype(string)
            except ValueError as err:
                raise TypeError(msg) from err
            else:
                result = SparseDtype(sub_type)
                msg = f"Cannot construct a 'SparseDtype' from '{string}'.\n\nIt looks like the fill_value in the string is not the default for the dtype. Non-default fill_values are not supported. Use the 'SparseDtype()' constructor instead."
                if has_fill_value and str(result) != string:
                    raise TypeError(msg)
                return result
        else:
            raise TypeError(msg)

    @staticmethod
    def _parse_subtype(dtype: str) -> tuple[str, bool]:
        xpr = re.compile('Sparse\\[(?P<subtype>[^,]*)(, )?(?P<fill_value>.*?)?\\]$')
        m = xpr.match(dtype)
        has_fill_value = False
        if m:
            subtype = m.groupdict()['subtype']
            has_fill_value = bool(m.groupdict()['fill_value'])
        elif dtype == 'Sparse':
            subtype = 'float64'
        else:
            raise ValueError(f'Cannot parse {dtype}')
        return (subtype, has_fill_value)

    @classmethod
    def is_dtype(cls, dtype: Any) -> bool:
        dtype = getattr(dtype, 'dtype', dtype)
        if isinstance(dtype, str) and dtype.startswith('Sparse'):
            sub_type, _ = cls._parse_subtype(dtype)
            dtype = np.dtype(sub_type)
        elif isinstance(dtype, cls):
            return True
        return isinstance(dtype, np.dtype) or dtype == 'Sparse'

    def update_dtype(self, dtype: Union[str, np.dtype, SparseDtype]) -> SparseDtype:
        from pandas.core.dtypes.astype import astype_array
        from pandas.core.dtypes.common import pandas_dtype
        cls = type(self)
        dtype = pandas_dtype(dtype)
        if not isinstance(dtype, cls):
            if not isinstance(dtype, np.dtype):
                raise TypeError('sparse arrays of extension dtypes not supported')
            fv_asarray = np.atleast_1d(np.array(self.fill_value))
            fvarr = astype_array(fv_asarray, dtype)
            fill_value = fvarr[0]
            dtype = cls(dtype, fill_value=fill_value)
        return dtype

    @property
    def _subtype_with_str(self) -> Any:
        if isinstance(self.fill_value, str):
            return type(self.fill_value)
        return self.subtype

    def _get_common_dtype(self, dtypes: Sequence[Any]) -> Optional[Any]:
        from pandas.core.dtypes.cast import np_find_common_type
        if any((isinstance(x, ExtensionDtype) and (not isinstance(x, SparseDtype)) for x in dtypes)):
            return None
        fill_values = [x.fill_value for x in dtypes if isinstance(x, SparseDtype)]
        fill_value = fill_values[0]
        from pandas import isna
        if get_option('performance_warnings') and (not (len(set(fill_values)) == 1 or isna(fill_values).all())):
            warnings.warn(f"Concatenating sparse arrays with multiple fill values: '{fill_values}'. Picking the first and converting the rest.", PerformanceWarning, stacklevel=find_stack_level())
        np_dtypes = (x.subtype if isinstance(x, SparseDtype) else x for x in dtypes)
        return SparseDtype(np_find_common_type(*np_dtypes), fill_value=fill_value)

@register_extension_dtype
@set_module('pandas')
class ArrowDtype(StorageExtensionDtype):
    """
    An ExtensionDtype for PyArrow data types.
    """
    _metadata: tuple[str, ...] = ('storage', 'pyarrow_dtype')

    def __init__(self, pyarrow_dtype: pa.DataType) -> None:
        super().__init__('pyarrow')
        if pa_version_under10p1:
            raise ImportError('pyarrow>=10.0.1 is required for ArrowDtype')
        if not isinstance(pyarrow_dtype, pa.DataType):
            raise ValueError(f'pyarrow_dtype ({pyarrow_dtype}) must be an instance of a pyarrow.DataType. Got {type(pyarrow_dtype)} instead.')
        self.pyarrow_dtype = pyarrow_dtype

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return super().__eq__(other)
        return self.pyarrow_dtype == other.pyarrow_dtype

    @property
    def type(self) -> Any:
        pa_type = self.pyarrow_dtype
        if pa.types.is_integer(pa_type):
            return int
        elif pa.types.is_floating(pa_type):
            return float
        elif pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
            return str
        elif pa.types.is_binary(pa_type) or pa.types.is_fixed_size_binary(pa_type) or pa.types.is_large_binary(pa_type):
            return bytes
        elif pa.types.is_boolean(pa_type):
            return bool
        elif pa.types.is_duration(pa_type):
            if pa_type.unit == 'ns':
                return Timedelta
            else:
                return timedelta
        elif pa.types.is_timestamp(pa_type):
            if pa_type.unit == 'ns':
                return Timestamp
            else:
                return datetime
        elif pa.types.is_date(pa_type):
            return date
        elif pa.types.is_time(pa_type):
            return time
        elif pa.types.is_decimal(pa_type):
            return Decimal
        elif pa.types.is_dictionary(pa_type):
            return CategoricalDtypeType
        elif pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
            return list
        elif pa.types.is_fixed_size_list(pa_type):
            return list
        elif pa.types.is_map(pa_type):
            return list
        elif pa.types.is_struct(pa_type):
            return dict
        elif pa.types.is_null(pa_type):
            return type(pa_type)
        elif isinstance(pa_type, pa.ExtensionType):
            return type(self)(pa_type.storage_type).type
        raise NotImplementedError(pa_type)

    @property
    def name(self) -> str:
        return f'{self.pyarrow_dtype!s}[{self.storage}]'

    @cache_readonly
    def numpy_dtype(self) -> np.dtype:
        if pa.types.is_timestamp(self.pyarrow_dtype):
            return np.dtype(f'datetime64[{self.pyarrow_dtype.unit}]')
        if pa.types.is_duration(self.pyarrow_dtype):
            return np.dtype(f'timedelta64[{self.pyarrow_dtype.unit}]')
        if pa.types.is_string(self.pyarrow_dtype) or pa.types.is_large_string(self.pyarrow_dtype):
            return np.dtype(str)
        try:
            return np.dtype(self.pyarrow_dtype.to_pandas_dtype())
        except (NotImplementedError, TypeError):
            return np.dtype(object)

    @cache_readonly
    def kind(self) -> str:
        if pa.types.is_timestamp(self.pyarrow_dtype):
            return 'M'
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self) -> int:
        return self.numpy_dtype.itemsize

    @classmethod
    def construct_array_type(cls) -> Type[Any]:
        from pandas.core.arrays.arrow import ArrowExtensionArray
        return ArrowExtensionArray

    @classmethod
    def construct_from_string(cls, string: str) -> ArrowDtype:
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if not string.endswith('[pyarrow]'):
            raise TypeError(f"'{string}' must end with '[pyarrow]'")
        if string in ('string[pyarrow]', 'str[pyarrow]'):
            raise TypeError('string[pyarrow] should be constructed by StringDtype')
        if pa_version_under10p1:
            raise ImportError('pyarrow>=10.0.1 is required for ArrowDtype')
        base_type = string[:-9]
        try:
            pa_dtype = pa.type_for_alias(base_type)
        except ValueError as err:
            has_parameters = re.search('[\\[\\(].*[\\]\\)]', base_type)
            if has_parameters:
                try:
                    return cls._parse_temporal_dtype_string(base_type)
                except (NotImplementedError, ValueError):
                    pass
                raise NotImplementedError(f'Passing pyarrow type specific parameters ({has_parameters.group()}) in the string is not supported. Please construct an ArrowDtype object with a pyarrow_dtype instance with specific parameters.') from err
            raise TypeError(f"'{base_type}' is not a valid pyarrow data type.") from err
        return cls(pa_dtype)

    @classmethod
    def _parse_temporal_dtype_string(cls, string: str) -> ArrowDtype:
        head, tail = string.split('[', 1)
        if not tail.endswith(']'):
            raise ValueError
        tail = tail[:-1]
        if head == 'timestamp':
            assert ',' in tail
            unit, tz = tail.split(',', 1)
            unit = unit.strip()
            tz = tz.strip()
            if tz.startswith('tz='):
                tz = tz[3:]
            pa_type = pa.timestamp(unit, tz=tz)
            dtype = cls(pa_type)
            return dtype
        raise NotImplementedError(string)

    @property
    def _is_numeric(self) -> bool:
        return pa.types.is_integer(self.pyarrow_dtype) or pa.types.is_floating(self.pyarrow_dtype) or pa.types.is_decimal(self.pyarrow_dtype)

    @property
    def _is_boolean(self) -> bool:
        return pa.types.is_boolean(self.pyarrow_dtype)

    def _get_common_dtype(self, dtypes: Sequence[Any]) -> Optional[Any]:
        from pandas.core.dtypes.cast import find_common_type
        null_dtype = type(self)(pa.null())
        new_dtype = find_common_type([dtype.numpy_dtype if isinstance(dtype, ArrowDtype) else dtype for dtype in dtypes if dtype != null_dtype])
        if not isinstance(new_dtype, np.dtype):
            return None
        try:
            pa_dtype = pa.from_numpy_dtype(new_dtype)
            return type(self)(pa_dtype)
        except NotImplementedError:
            return None

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]) -> Any:
        array_class = self.construct_array_type()
        arr = array.cast(self.pyarrow_dtype, safe=True)  # type: ignore
        return array_class(arr)