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
    Hashable,
    Iterator,
    Sized,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from pandas._libs.tslibs import (
    BaseOffset,
    Timedelta,
    to_offset,
)
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import (
    Appender,
    Substitution,
    doc,
)

from pandas.core.dtypes.common import (
    ensure_float64,
    is_bool,
    is_integer,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import notna

from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import (
    ResamplerWindowApply,
    reconstruct_func,
)
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
from pandas.core.window.common import (
    flex_binary_moment,
    zsqrt,
)
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
    from collections.abc import Callable
    from collections.abc import (
        Hashable,
        Iterator,
        Sized,
    )

    from pandas._typing import (
        ArrayLike,
        Concatenate,
        NDFrameT,
        QuantileInterpolation,
        P,
        Self,
        T,
        WindowingRankType,
        npt,
    )

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame
    from pandas.core.groupby.ops import BaseGrouper

from pandas.core.arrays.datetimelike import dtype_to_unit


class BaseWindow(SelectionMixin):
    """Provides utilities for performing windowing operations."""

    _attributes: List[str] = []
    exclusions: frozenset[Hashable] = frozenset()
    _on: Index

    def __init__(
        self,
        obj: NDFrame,
        window: Optional[Union[int, BaseIndexer, str, timedelta, BaseOffset]] = None,
        min_periods: Optional[int] = None,
        center: Optional[bool] = False,
        win_type: Optional[str] = None,
        on: Optional[Union[str, Index]] = None,
        closed: Optional[str] = None,
        step: Optional[int] = None,
        method: str = "single",
        *,
        selection=None,
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
                f"invalid on specified as {self.on}, "
                "must be a column (of DataFrame), an Index or None"
            )

        self._selection = selection
        self._validate()

    def _validate(self) -> None:
        if self.center is not None and not is_bool(self.center):
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
            # Validate that the passed BaseIndexer subclass has
            # a get_window_bounds with the correct signature.
            get_window_bounds_signature = inspect.signature(
                self.window.get_window_bounds
            ).parameters.keys()
            expected_signature = inspect.signature(
                BaseIndexer().get_window_bounds
            ).parameters.keys()
            if get_window_bounds_signature != expected_signature:
                raise ValueError(
                    f"{type(self.window).__name__} does not implement "
                    f"the correct signature for get_window_bounds"
                )
        if self.method not in ["table", "single"]:
            raise ValueError("method must be 'table' or 'single'")
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
                f"start ({len(start)}) and end ({len(end)}) bounds must be the "
                f"same length"
            )
        if len(start) != (num_vals + (self.step or 1) - 1) // (self.step or 1):
            raise ValueError(
                f"start and end bounds ({len(start)}) must be the same length "
                f"as the object ({num_vals}) divided by the step ({self.step}) "
                f"if given and rounded up"
            )

    def _slice_axis_for_step(self, index: Index, result: Optional[Sized] = None) -> Index:
        """
        Slices the index for a given result and the preset step.
        """
        return (
            index
            if result is None or len(result) == len(index)
            else index[:: self.step]
        )

    def _validate_numeric_only(self, name: str, numeric_only: bool) -> None:
        """
        Validate numeric_only argument, raising if invalid for the input.

        Parameters
        ----------
        name : str
            Name of the operator (kernel).
        numeric_only : bool
            Value passed by user.
        """
        if (
            self._selected_obj.ndim == 1
            and numeric_only
            and not is_numeric_dtype(self._selected_obj.dtype)
        ):
            raise NotImplementedError(
                f"{type(self).__name__}.{name} does not implement numeric_only"
            )

    def _make_numeric_only(self, obj: NDFrameT) -> NDFrameT:
        """Subset DataFrame to numeric columns.

        Parameters
        ----------
        obj : DataFrame

        Returns
        -------
        obj subset to numeric-only columns.
        """
        result = obj.select_dtypes(include=["number"], exclude=["timedelta"])
        return result

    def _create_data(self, obj: NDFrameT, numeric_only: bool = False) -> NDFrameT:
        """
        Split data into blocks & return conformed data.
        """
        # filter out the on from the object
        if self.on is not None and not isinstance(self.on, Index) and obj.ndim == 2:
            obj = obj.reindex(columns=obj.columns.difference([self.on], sort=False))
        if obj.ndim > 1 and numeric_only:
            obj = self._make_numeric_only(obj)
        return obj

    def _gotitem(self, key, ndim, subset=None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : str / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        # create a new object to prevent aliasing
        if subset is None:
            subset = self.obj

        # we need to make a shallow copy of ourselves
        # with the same groupby
        kwargs = {attr: getattr(self, attr) for attr in self._attributes}

        selection = self._infer_selection(key, subset)
        new_win = type(self)(subset, selection=selection, **kwargs)
        return new_win

    def __getattr__(self, attr: str) -> Any:
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self.obj:
            return self[attr]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def _dir_additions(self) -> List[str]:
        return self.obj._dir_additions()

    def __repr__(self) -> str:
        """
        Provide a nice str repr of our rolling object.
        """
        attrs_list = (
            f"{attr_name}={getattr(self, attr_name)}"
            for attr_name in self._attributes
            if getattr(self, attr_name, None) is not None and attr_name[0] != "_"
        )
        attrs = ",".join(attrs_list)
        return f"{type(self).__name__} [{attrs}]"

    def __iter__(self) -> Iterator[NDFrame]:
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
        """Convert input to numpy arrays for Cython routines"""
        if needs_i8_conversion(values.dtype):
            raise NotImplementedError(
                f"ops for {type(self).__name__} for this "
                f"dtype {values.dtype} are not implemented"
            )
        # GH #12373 : rolling functions error on float32 data
        # make sure the data is coerced to float64
        try:
            if isinstance(values, ExtensionArray):
                values = values.to_numpy(np.float64, na_value=np.nan)
            else:
                values = ensure_float64(values)
        except (ValueError, TypeError) as err:
            raise TypeError(f"cannot handle this type -> {values.dtype}") from err

        # Convert inf to nan for C funcs
        inf = np.isinf(values)
        if inf.any():
            values = np.where(inf, np.nan, values)

        return values

    def _insert_on_column(self, result: DataFrame, obj: DataFrame) -> None:
        # if we have an 'on' column we want to put it back into
        # the results in the same location
        from pandas import Series

        if self.on is not None and not self._on.equals(obj.index):
            name = self._on.name
            extra_col = Series(self._on, index=self.obj.index, name=name, copy=False)
            if name in result.columns:
                # TODO: sure we want to overwrite results?
                result[name] = extra_col
            elif name in result.index.names:
                pass
            elif name in self._selected_obj.columns:
                # insert in the same location as we had in _selected_obj
                old_cols = self._selected_obj.columns
                new_cols = result.columns
                old_loc = old_cols.get_loc(name)
                overlap = new_cols.intersection(old_cols[:old_loc])
                new_loc = len(overlap)
                result.insert(new_loc, name, extra_col)
            else:
                # insert at the end
                result[name] = extra_col

    @property
    def _index_array(self) -> Optional[NDArray[np.int64]]:
        # TODO: why do we get here with e.g. MultiIndex?
        if isinstance(self._on, (PeriodIndex, DatetimeIndex, TimedeltaIndex)):
            return self._on.asi8
        elif isinstance(self._on.dtype, ArrowDtype) and self._on.dtype.kind in "mM":
            return self._on.to_numpy(dtype=np.int64)
        return None

    def _resolve_output(self, out: DataFrame, obj: DataFrame) -> DataFrame:
        """Validate and finalize result."""
        if out.shape[1] == 0 and obj.shape[1] > 0:
            raise DataError("No numeric types to aggregate")
        if out.shape[1] == 0:
            return obj.astype("float64")

        self._insert_on_column(out, obj)
        return out

    def _get_window_indexer(self) -> BaseIndexer:
        """
        Return an indexer class that will compute the window start and end bounds
        """
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
        self, homogeneous_func: Callable[..., ArrayLike], name: Optional[str] = None
    ) -> Series:
        """
        Series version of _apply_columnwise
        """
        obj = self._create_data(self._selected_obj)

        if name == "count":
            # GH 12541: Special case for count where we support date-like types
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
        homogeneous_func: Callable[..., ArrayLike],
        name: str,
        numeric_only: bool = False,
    ) -> Union[DataFrame, Series]:
        """
        Apply the given function to the DataFrame broken down into homogeneous
        sub-frames.
        """
        self._validate_numeric_only(name, numeric_only)
        if self._selected_obj.ndim == 1:
            return self._apply_series(homogeneous_func, name)

        obj = self._create_data(self._selected_obj, numeric_only)
        if name == "count":
            # GH 12541: Special case for count where we support date-like types
            obj = notna(obj).astype(int)
            obj._mgr = obj._mgr.consolidate()

        taker = []
        res_values = []
        for i, arr in enumerate(obj._iter_column_arrays()):
            # GH#42736 operate column-wise instead of block-wise
            # As of 2.0, hfunc will raise for nuisance columns
            try:
                arr = self._prep_values(arr)
            except (TypeError, NotImplementedError) as err:
                raise DataError(
                    f"Cannot aggregate non-numeric type: {arr.dtype}"
                ) from err
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
        homogeneous_func: Callable[..., ArrayLike],
        name: Optional[str] = None,
        numeric_only: bool = False,
    ) -> Union[DataFrame, Series]:
        """
        Apply the given function to the DataFrame across the entire object
        """
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
        out