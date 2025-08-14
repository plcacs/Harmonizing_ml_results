from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterator,
)
from datetime import timedelta
import operator
from sys import getsizeof
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
    Optional,
    Union,
    Tuple,
    List,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray

from pandas._libs import (
    index as libindex,
    lib,
)
from pandas._libs.lib import no_default
from pandas.compat.numpy import function as nv
from pandas.util._decorators import (
    cache_readonly,
    doc,
    set_module,
)

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
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
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

_empty_range = range(0)
_dtype_int64 = np.dtype(np.int64)

Self = TypeVar('Self', bound='RangeIndex')

def min_fitting_element(start: int, step: int, lower_limit: int) -> int:
    """Returns the smallest element greater than or equal to the limit"""
    no_steps = -(-(lower_limit - start) // abs(step))
    return start + abs(step) * no_steps


@set_module("pandas")
class RangeIndex(Index):
    _typ: str = "rangeindex"
    _dtype_validation_metadata: Tuple[Callable[[Any], bool], str] = (is_signed_integer_dtype, "signed integer")
    _range: range
    _values: np.ndarray

    @property
    def _engine_type(self) -> type[libindex.Int64Engine]:
        return libindex.Int64Engine

    def __new__(
        cls,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        dtype: Optional[Dtype] = None,
        copy: bool = False,
        name: Optional[Hashable] = None,
    ) -> Self:
        cls._validate_dtype(dtype)
        name = maybe_extract_name(name, start, cls)

        if isinstance(start, cls):
            return start.copy(name=name)
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
    def from_range(cls, data: range, name: Optional[Hashable] = None, dtype: Optional[Dtype] = None) -> Self:
        if not isinstance(data, range):
            raise TypeError(
                f"{cls.__name__}(...) must be called with object coercible to a "
                f"range, {data!r} was passed"
            )
        cls._validate_dtype(dtype)
        return cls._simple_new(data, name=name)

    @classmethod
    def _simple_new(cls, values: range, name: Optional[Hashable] = None) -> Self:
        result = object.__new__(cls)
        assert isinstance(values, range)
        result._range = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        result._references = None
        return result

    @classmethod
    def _validate_dtype(cls, dtype: Optional[Dtype]) -> None:
        if dtype is None:
            return
        validation_func, expected = cls._dtype_validation_metadata
        if not validation_func(dtype):
            raise ValueError(
                f"Incorrect `dtype` passed: expected {expected}, received {dtype}"
            )

    @cache_readonly
    def _constructor(self) -> type[Index]:
        return Index

    @cache_readonly
    def _data(self) -> np.ndarray:
        return np.arange(self.start, self.stop, self.step, dtype=np.int64)

    def _get_data_as_items(self) -> List[Tuple[str, int]]:
        rng = self._range
        return [("start", rng.start), ("stop", rng.stop), ("step", rng.step)]

    def __reduce__(self) -> Tuple[Any, Any, Any]:
        d = {"name": self._name}
        d.update(dict(self._get_data_as_items()))
        return ibase._new_Index, (type(self), d), None

    def _format_attrs(self) -> List[Tuple[str, Union[str, int]]]:
        attrs = cast(List[Tuple[str, Union[str, int]]], self._get_data_as_items())
        if self._name is not None:
            attrs.append(("name", ibase.default_pprint(self._name)))
        return attrs

    def _format_with_header(self, *, header: List[str], na_rep: str) -> List[str]:
        if not len(self._range):
            return header
        first_val_str = str(self._range[0])
        last_val_str = str(self._range[-1])
        max_length = max(len(first_val_str), len(last_val_str))
        return header + [f"{x:<{max_length}}" for x in self._range]

    @property
    def start(self) -> int:
        return self._range.start

    @property
    def stop(self) -> int:
        return self._range.stop

    @property
    def step(self) -> int:
        return self._range.step

    @cache_readonly
    def nbytes(self) -> int:
        rng = self._range
        return getsizeof(rng) + sum(
            getsizeof(getattr(rng, attr_name))
            for attr_name in ["start", "stop", "step"]
        )

    def memory_usage(self, deep: bool = False) -> int:
        return self.nbytes

    @property
    def dtype(self) -> np.dtype:
        return _dtype_int64

    @property
    def is_unique(self) -> bool:
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
            key = ensure_python_int(key)
        except (TypeError, OverflowError):
            return False
        return key in self._range

    @property
    def inferred_type(self) -> str:
        return "integer"

    def get_loc(self, key: Any) -> int:
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

    def _get_indexer(
        self,
        target: Index,
        method: Optional[str] = None,
        limit: Optional[int] = None,
        tolerance: Any = None,
    ) -> NDArray[np.intp]:
        if com.any_not_none(method, tolerance, limit):
            return super()._get_indexer(
                target, method=method, tolerance=tolerance, limit=limit
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
    def _should_fallback_to_positional(self) -> bool:
        return False

    def tolist(self) -> List[int]:
        return list(self._range)

    def __iter__(self) -> Iterator[int]:
        yield from self._range

    def _shallow_copy(self, values: Any, name: Hashable = no_default) -> Index:
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

    def _view(self) -> Self:
        result = type(self)._simple_new(self._range, name=self._name)
        result._cache = self._cache
        return result

    def _wrap_reindex_result(self, target: Index, indexer: Any, preserve_names: bool) -> Index:
        if not isinstance(target, type(self)) and target.dtype.kind == "i":
            target = self._shallow_copy(target._values, name=target.name)
        return super()._wrap_reindex_result(target, indexer, preserve_names)

    def copy(self, name: Optional[Hashable] = None, deep: bool = False) -> Self:
        name = self._validate_names(name=name, deep=deep)[0]
        new_index = self._rename(name=name)
        return new_index

    def _minmax(self, meth: Literal["min", "max"]) -> Union[int, float]:
        no_steps = len(self) - 1
        if no_steps == -1:
            return np.nan
        elif (meth == "min" and self.step > 0) or (meth == "max" and self.step < 0):
            return self.start
        return self.start + self.step * no_steps

    def min(self, axis: Any = None, skipna: bool = True, *args: Any, **kwargs: Any) -> Union[int, float]:
        nv.validate_minmax_axis(axis)
        nv.validate_min(args, kwargs)
        return self._minmax("min")

    def max(self, axis: Any = None, skipna: bool = True, *args: Any, **kwargs: Any) -> Union[int, float]:
        nv.validate_minmax_axis(axis)
        nv.validate_max(args, kwargs)
        return self._minmax("max")

    def _argminmax(
        self,
        meth: Literal["min", "max"],
        axis: Any = None,
        skipna: bool = True,
    ) -> int:
        nv.validate_minmax_axis(axis)
        if len(self) == 0:
            return getattr(super(), f"arg{meth}")(
                axis=axis,
                skipna=skipna,
            )
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
            raise ValueError(f"{meth=} must be max or min")

    def argmin(self, axis: Any = None, skipna: bool = True, *args: Any, **kwargs: Any) -> int:
        nv.validate_argmin(args, kwargs)
        return self._argminmax("min", axis=axis, skipna=skipna)

    def argmax(self, axis: Any = None, skipna: bool = True, *args: Any, **kwargs: Any) -> int:
        nv.validate_argmax(args, kwargs)
        return self._argminmax("max", axis=axis, skipna=skipna)

    def argsort(self, *args: Any, **kwargs: Any) -> NDArray[np.intp]:
        ascending = kwargs.pop("ascending", True)
        kwargs.pop("kind", None)
        nv.validate_argsort(args, kwargs)

        start, stop, step = None, None, None
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

    def factorize(
        self,
        sort: bool = False,
        use_na_sentinel: bool = True,
    ) -> Tuple[NDArray[np.intp], Self]:
        if sort and self.step < 0:
            codes = np.arange(len(self) - 1, -1, -1, dtype=np.intp)
            uniques = self[::-1]
        else:
            codes = np.arange(len(self), dtype=np.intp)
            uniques = self
        return codes, uniques

    def equals(self, other: object) -> bool:
        if isinstance(other, RangeIndex):
            return self._range == other._range
        return super().equals(other)

    @overload
    def sort_values(
        self,
        *,
        return_indexer: Literal[False] = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Optional[Callable] = ...,
    ) -> Self: ...

    @overload
    def sort_values(
        self,
        *,
        return_indexer: Literal[True],
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Optional[Callable] = ...,
    ) -> Tuple[Self, Union[NDArray[np.intp], Self]]: ...

    @overload
    def sort_values(
        self,
        *,
        return_indexer: bool = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Optional[Callable] = ...,
    ) -> Union[Self, Tuple[Self, Union[NDArray[np.intp], Self]]]: ...

    def sort_values(
        self,
        *,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: NaPosition = "last",
        key: Optional[Callable] = None,
    ) -> Union[Self, Tuple[Self, Union[NDArray[np.intp], Self]]:
        if key is not None:
            return super().sort_values(
                return_indexer=return_indexer,
                ascending=ascending,
                na_position=na_position,
                key=key,
            )
        else:
            sorted_index = self
            inverse_indexer = False
            if ascending:
                if self.step < 0:
                    sorted_index = self[::-1]
                    inverse_indexer = True
            else:
                if self.step > 0:
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

    def _intersection(self, other: Index, sort: bool = False) -> Index:
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

        new_start = min_fitting_element(tmp_start, new_step, int_low)
        new_range = range(new_start, int_high, new_step)

        if (self.step < 0 and other.step < 0) is not (new_range.step < 0):
            new_range = new_range[::-1]

        return self._simple_new(new_range)

    def _extended_gcd(self, a: int, b: