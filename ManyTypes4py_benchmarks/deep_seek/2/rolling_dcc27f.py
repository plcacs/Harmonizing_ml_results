from __future__ import annotations
import copy
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    final,
    overload,
    Optional,
    Union,
    Dict,
    List,
    Tuple,
    Callable,
    TypeVar,
    Generic,
    cast,
)
import numpy as np
from pandas._libs.tslibs import BaseOffset, Timedelta, to_offset
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import Appender, Substitution, doc
from pandas.core.dtypes.common import (
    ensure_float64,
    is_bool,
    is_integer,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import notna
from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import ResamplerWindowApply, reconstruct_func
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import (
    BaseIndexer,
    FixedWindowIndexer,
    GroupbyIndexer,
    VariableWindowIndexer,
)
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    TimedeltaIndex,
)
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import (
    get_jit_arguments,
    maybe_use_numba,
    prepare_function_arguments,
)
from pandas.core.window.common import flex_binary_moment, zsqrt
from pandas.core.window.doc import (
    _shared_docs,
    create_section_header,
    kwargs_numeric_only,
    kwargs_scipy,
    numba_notes,
    template_header,
    template_pipe,
    template_returns,
    template_see_also,
    window_agg_numba_parameters,
    window_apply_parameters,
)
from pandas.core.window.numba_ import (
    generate_manual_numpy_nan_agg_with_axis,
    generate_numba_apply_func,
    generate_numba_table_func,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator, Sized
    from pandas._typing import (
        ArrayLike,
        NDFrameT,
        QuantileInterpolation,
        WindowingRankType,
        npt,
    )
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame
    from pandas.core.groupby.ops import BaseGrouper

T = TypeVar("T", bound="NDFrame")
NDFrameT = TypeVar("NDFrameT", bound="NDFrame")
Self = TypeVar("Self", bound="BaseWindow")

class BaseWindow(SelectionMixin, Generic[NDFrameT]):
    _attributes: List[str] = []
    exclusions: frozenset = frozenset()

    def __init__(
        self,
        obj: NDFrameT,
        window: Optional[Union[int, BaseIndexer, str, BaseOffset, timedelta]] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
        win_type: Optional[str] = None,
        on: Optional[Union[str, Index]] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        method: str = "single",
        *,
        selection: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self.obj = obj
        self.on = on
        self.closed = closed
        self.step = step
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self.win_type = win_type
        self.method = method
        self._win_freq_i8: Optional[int] = None
        if self.on is None:
            self._on = self.obj.index
        elif isinstance(self.on, Index):
            self._on = self.on
        elif isinstance(self.obj, ABCDataFrame) and self.on in self.obj.columns:
            self._on = Index(self.obj[self.on])
        else:
            raise ValueError(
                f"invalid on specified as {self.on}, must be a column (of DataFrame), an Index or None"
            )
        self._selection = selection
        self._validate()

    def _validate(self) -> None:
        if self.center is not None and (not is_bool(self.center)):
            raise ValueError("center must be a boolean")
        if self.min_periods is not None:
            if not is_integer(self.min_periods):
                raise ValueError("min_periods must be an integer")
            if self.min_periods < 0:
                raise ValueError("min_periods must be >= 0")
            if is_integer(self.window) and self.min_periods > self.window:
                raise ValueError(
                    f"min_periods {self.min_periods} must be <= window {self.window}"
                )
        if self.closed is not None and self.closed not in [
            "right",
            "both",
            "left",
            "neither",
        ]:
            raise ValueError("closed must be 'right', 'left', 'both' or 'neither'")
        if not isinstance(self.obj, (ABCSeries, ABCDataFrame)):
            raise TypeError(f"invalid type: {type(self)}")
        if isinstance(self.window, BaseIndexer):
            get_window_bounds_signature = inspect.signature(
                self.window.get_window_bounds
            ).parameters.keys()
            expected_signature = inspect.signature(
                BaseIndexer().get_window_bounds
            ).parameters.keys()
            if get_window_bounds_signature != expected_signature:
                raise ValueError(
                    f"{type(self.window).__name__} does not implement the correct signature for get_window_bounds"
                )
        if self.method not in ["table", "single"]:
            raise ValueError("method must be 'table' or 'single")
        if self.step is not None:
            if not is_integer(self.step):
                raise ValueError("step must be an integer")
            if self.step < 0:
                raise ValueError("step must be >= 0")

    def _check_window_bounds(
        self, start: np.ndarray, end: np.ndarray, num_vals: int
    ) -> None:
        if len(start) != len(end):
            raise ValueError(
                f"start ({len(start)}) and end ({len(end)}) bounds must be the same length"
            )
        if len(start) != (num_vals + (self.step or 1) - 1) // (self.step or 1):
            raise ValueError(
                f"start and end bounds ({len(start)}) must be the same length as the object ({num_vals}) divided by the step ({self.step}) if given and rounded up"
            )

    def _slice_axis_for_step(
        self, index: Index, result: Optional[np.ndarray] = None
    ) -> Index:
        return index if result is None or len(result) == len(index) else index[:: self.step]

    def _validate_numeric_only(self, name: str, numeric_only: bool) -> None:
        if self._selected_obj.ndim == 1 and numeric_only and (
            not is_numeric_dtype(self._selected_obj.dtype)
        ):
            raise NotImplementedError(
                f"{type(self).__name__}.{name} does not implement numeric_only"
            )

    def _make_numeric_only(self, obj: DataFrame) -> DataFrame:
        result = obj.select_dtypes(include=["number"], exclude=["timedelta"])
        return result

    def _create_data(
        self, obj: NDFrameT, numeric_only: bool = False
    ) -> Union[Series, DataFrame]:
        if self.on is not None and (not isinstance(self.on, Index)) and (obj.ndim == 2):
            obj = obj.reindex(columns=obj.columns.difference([self.on], sort=False))
        if obj.ndim > 1 and numeric_only:
            obj = self._make_numeric_only(obj)
        return obj

    def _gotitem(
        self,
        key: Union[str, List[str]],
        ndim: int,
        subset: Optional[NDFrameT] = None,
    ) -> BaseWindow:
        if subset is None:
            subset = self.obj
        kwargs = {attr: getattr(self, attr) for attr in self._attributes}
        selection = self._infer_selection(key, subset)
        new_win = type(self)(subset, selection=selection, **kwargs)
        return new_win

    def __getattr__(self, attr: str) -> Any:
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self.obj:
            return self[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def _dir_additions(self) -> set[str]:
        return self.obj._dir_additions()

    def __repr__(self) -> str:
        attrs_list = (
            f"{attr_name}={getattr(self, attr_name)}"
            for attr_name in self._attributes
            if getattr(self, attr_name, None) is not None and attr_name[0] != "_"
        )
        attrs = ",".join(attrs_list)
        return f"{type(self).__name__} [{attrs}]"

    def __iter__(self) -> Iterator[NDFrameT]:
        obj = self._selected_obj.set_axis(self._on)
        obj = self._create_data(obj)
        indexer = self._get_window_indexer()
        start, end = indexer.get_window_bounds(
            num_values=len(obj),
            min_periods=self.min_periods,
            center=self.center,
            closed=self.closed,
            step=self.step,
        )
        self._check_window_bounds(start, end, len(obj))
        for s, e in zip(start, end):
            result = obj.iloc[slice(s, e)]
            yield result

    def _prep_values(self, values: ArrayLike) -> np.ndarray:
        if needs_i8_conversion(values.dtype):
            raise NotImplementedError(
                f"ops for {type(self).__name__} for this dtype {values.dtype} are not implemented"
            )
        try:
            if isinstance(values, ExtensionArray):
                values = values.to_numpy(np.float64, na_value=np.nan)
            else:
                values = ensure_float64(values)
        except (ValueError, TypeError) as err:
            raise TypeError(f"cannot handle this type -> {values.dtype}") from err
        inf = np.isinf(values)
        if inf.any():
            values = np.where(inf, np.nan, values)
        return values

    def _insert_on_column(self, result: DataFrame, obj: NDFrameT) -> None:
        from pandas import Series

        if self.on is not None and (not self._on.equals(obj.index)):
            name = self._on.name
            extra_col = Series(self._on, index=self.obj.index, name=name, copy=False)
            if name in result.columns:
                result[name] = extra_col
            elif name in result.index.names:
                pass
            elif name in self._selected_obj.columns:
                old_cols = self._selected_obj.columns
                new_cols = result.columns
                old_loc = old_cols.get_loc(name)
                overlap = new_cols.intersection(old_cols[:old_loc])
                new_loc = len(overlap)
                result.insert(new_loc, name, extra_col)
            else:
                result[name] = extra_col

    @property
    def _index_array(self) -> Optional[np.ndarray]:
        if isinstance(self._on, (PeriodIndex, DatetimeIndex, TimedeltaIndex)):
            return self._on.asi8
        elif isinstance(self._on.dtype, ArrowDtype) and self._on.dtype.kind in "mM":
            return self._on.to_numpy(dtype=np.int64)
        return None

    def _resolve_output(self, out: DataFrame, obj: NDFrameT) -> DataFrame:
        if out.shape[1] == 0 and obj.shape[1] > 0:
            raise DataError("No numeric types to aggregate")
        if out.shape[1] == 0:
            return obj.astype("float64")
        self._insert_on_column(out, obj)
        return out

    def _get_window_indexer(self) -> BaseIndexer:
        if isinstance(self.window, BaseIndexer):
            return self.window
        if self._win_freq_i8 is not None:
            return VariableWindowIndexer(
                index_array=self._index_array,
                window_size=self._win_freq_i8,
                center=self.center,
            )
        return FixedWindowIndexer(window_size=self.window)

    def _apply_series(
        self, homogeneous_func: Callable[[np.ndarray], np.ndarray], name: Optional[str] = None
    ) -> Series:
        obj = self._create_data(self._selected_obj)
        if name == "count":
            obj = notna(obj).astype(int)
        try:
            values = self._prep_values(obj._values)
        except (TypeError, NotImplementedError) as err:
            raise DataError("No numeric types to aggregate") from err
        result = homogeneous_func(values)
        index = self._slice_axis_for_step(obj.index, result)
        return obj._constructor(result, index=index, name=obj.name)

    def _apply_columnwise(
        self,
        homogeneous_func: Callable[[np.ndarray], np.ndarray],
        name: str,
        numeric_only: bool = False,
    ) -> Union[Series, DataFrame]:
        self._validate_numeric_only(name, numeric_only)
        if self._selected_obj.ndim == 1:
            return self._apply_series(homogeneous_func, name)
        obj = self._create_data(self._selected_obj, numeric_only)
        if name == "count":
            obj = notna(obj).astype(int)
            obj._mgr = obj._mgr.consolidate()
        taker = []
        res_values = []
        for i, arr in enumerate(obj._iter_column_arrays()):
            try:
                arr = self._prep_values(arr)
            except (TypeError, NotImplementedError) as err:
                raise DataError(f"Cannot aggregate non-numeric type: {arr.dtype}") from err
            res = homogeneous_func(arr)
            res_values.append(res)
            taker.append(i)
        index = self._slice_axis_for_step(
            obj.index, res_values[0] if len(res_values) > 0 else None
        )
        df = type(obj)._from_arrays(
            res_values,
            index=index,
            columns=obj.columns.take(taker),
            verify_integrity=False,
        )
        return self._resolve_output(df, obj)

    def _apply_tablewise(
        self,
        homogeneous_func: Callable[[np.ndarray], np.ndarray],
        name: Optional[str] = None,
        numeric_only: bool = False,
    ) -> DataFrame:
        if self._selected_obj.ndim == 1:
            raise ValueError("method='table' not applicable for Series objects.")
        obj = self._create_data(self._selected_obj, numeric_only)
        values = self._prep_values(obj.to_numpy())
        result = homogeneous_func(values)
        index = self._slice_axis_for_step(obj.index, result)
        columns = (
            obj.columns
            if result.shape[1] == len(obj.columns)
            else obj.columns[:: self.step]
        )
        out = obj._constructor(result, index=index, columns=columns)
        return self._resolve_output(out, obj)

    def _apply_pairwise(
        self,
        target: NDFrameT,
        other: Optional[NDFrameT],
        pairwise: Optional[bool],
        func: Callable[..., Any],
        numeric_only: bool,
    ) -> Union[Series, DataFrame]:
        target = self._create_data(target, numeric_only)
        if other is None:
            other = target
            pairwise = True if pairwise is None else pairwise
        elif not isinstance(other, (ABCDataFrame, ABCSeries)):
            raise ValueError("other must be a DataFrame or Series")
        elif other.ndim == 2 and numeric_only:
            other = self._make_numeric_only(other)
        return flex_binary_moment(target, other, func, pairwise=bool(pairwise))

    def _apply(
        self,
        func: Callable[..., Any],
        name: str,
        numeric_only: bool = False,
        numba_args: Tuple[Any, ...] = (),
        **kwargs: Any,
    ) -> Union[Series, DataFrame]:
        window_indexer = self._get_window_indexer()
        min_periods = (
            self.min_periods
            if self.min_periods is not None
            else window_indexer.window_size
        )

        def homogeneous_func(values: np.ndarray) -> np.ndarray:
            if values.size == 0:
                return values.copy()

            def calc(x: np.ndarray) -> np.ndarray:
                start, end = window_indexer.get_window_bounds(
                    num_values=len(x),
                    min_periods=min_periods,
                    center=self.center,
                    closed=self.closed,
                    step=self.step,
                )
                self._check_window_bounds(start, end, len(x))
                return func(x, start, end, min_periods, *numba_args)

            with np.errstate(all="ignore"):
                result = calc(values)
            return result

        if self.method == "single":
            return self._apply_columnwise(homogeneous_func, name, numeric_only)
        else:
            return self._apply_tablewise(homogeneous_func, name, numeric_only)

    def _numba_apply(
        self,
        func: Callable[..., Any],
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **func_kwargs: Any,
    ) -> Union[Series, DataFrame]:
        window_indexer = self._get_window_indexer()
        min_periods = (
            self.min_periods
            if self.min_periods