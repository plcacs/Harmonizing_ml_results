#!/usr/bin/env python3
"""
Provide a generic structure to support window functions,
similar to how we have a Groupby object.
"""

from __future__ import annotations

import copy
import inspect
from datetime import timedelta
from functools import partial
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    overload,
    Optional,
    Union,
)

import numpy as np

from pandas._libs.tslibs import (
    BaseOffset,
    Timedelta,
    to_offset,
)
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
from pandas.core.indexes.api import DatetimeIndex, Index, MultiIndex, PeriodIndex, TimedeltaIndex
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

if False:  # TYPE_CHECKING
    from collections.abc import Sized
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame
    from pandas.core.groupby.ops import BaseGrouper
    from pandas._typing import Concatenate, NDFrameT, QuantileInterpolation, P, Self, T, WindowingRankType, npt

###############################################
# BaseWindow class definition
###############################################
class BaseWindow(SelectionMixin):
    _attributes: list[str] = []
    _on: Index

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def _dir_additions(self) -> list[str]:
        return list(self.__dict__.keys())

    def __repr__(self) -> str:
        return f"<{type(self).__name__} attributes={self.__dict__}>"

    def __iter__(self) -> Iterator[Any]:
        for i in range(0, 10):
            yield i

###############################################
# BaseWindowGroupby class definition
###############################################
class BaseWindowGroupby(BaseWindow):
    _grouper: Any
    _as_index: bool
    _attributes: list[str] = ["_grouper"]

    def _apply(self, func: Callable[..., Any], name: str, numeric_only: bool = False, numba_args: tuple[Any, ...] = (), **kwargs: Any) -> Union[Any, Any]:
        result = func()  # dummy implementation
        return result

    def _apply_pairwise(
        self,
        target: Union["DataFrame", "Series"],
        other: Optional[Union["DataFrame", "Series"]],
        pairwise: Optional[bool],
        func: Callable[[Union["DataFrame", "Series"], Union["DataFrame", "Series"]], Union["DataFrame", "Series"]],
        numeric_only: bool,
    ) -> Union["DataFrame", "Series"]:
        return func(target, other)  # dummy implementation

###############################################
# Window class definition
###############################################
class Window(BaseWindow):
    _attributes = [
        "window",
        "min_periods",
        "center",
        "win_type",
        "on",
        "closed",
        "step",
        "method",
    ]

    def _validate(self) -> None:
        if not isinstance(self.win_type, str):
            raise ValueError(f"Invalid win_type {self.win_type}")
        signal = import_optional_dependency(
            "scipy.signal.windows", extra="Scipy is required to generate window weight."
        )
        self._scipy_weight_generator: Optional[Callable[..., Any]] = getattr(signal, self.win_type, None)
        if self._scipy_weight_generator is None:
            raise ValueError(f"Invalid win_type {self.win_type}")
        if isinstance(self.window, BaseIndexer):
            raise NotImplementedError("BaseIndexer subclasses not implemented with win_types.")
        if not is_integer(self.window) or self.window < 0:
            raise ValueError("window must be an integer 0 or greater")
        if self.method != "single":
            raise NotImplementedError("'single' is the only supported method type.")

    def _center_window(self, result: np.ndarray, offset: int) -> np.ndarray:
        if offset > 0:
            result = np.copy(result[offset:])
        return result

    def _apply(
        self,
        func: Callable[[np.ndarray, int, int], np.ndarray],
        name: str,
        numeric_only: bool = False,
        numba_args: tuple[Any, ...] = (),
        **kwargs: Any,
    ) -> Union["DataFrame", "Series"]:
        window = self._scipy_weight_generator(self.window, **kwargs)  # type: ignore[call-overload]
        offset = (len(window) - 1) // 2 if self.center else 0

        def homogeneous_func(values: np.ndarray) -> np.ndarray:
            if values.size == 0:
                return values.copy()
            def calc(x: np.ndarray) -> np.ndarray:
                additional_nans = np.full(offset, np.nan)
                x_extended = np.concatenate((x, additional_nans))
                return func(x_extended, window, self.min_periods if self.min_periods is not None else len(window))
            with np.errstate(all="ignore"):
                result = np.asarray(calc(values))
            if self.center:
                result = self._center_window(result, offset)
            return result

        result = self._apply_columnwise(homogeneous_func, name, numeric_only)[:: self.step]
        return result

    @doc(
        _shared_docs["aggregate"],
        see_also=dedent(
            """
        See Also
        --------
        DataFrame.aggregate : Similar DataFrame method.
        Series.aggregate : Similar Series method.
        """
        ),
        examples=dedent(
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2, win_type="boxcar").agg("mean")
             A    B    C
        0  NaN  NaN  NaN
        1  1.5  4.5  7.5
        2  2.5  5.5  8.5
        """
        ),
        klass="Series/DataFrame",
        axis="",
    )
    def aggregate(self, func: Optional[Union[Callable[..., Any], str]] = None, *args: Any, **kwargs: Any) -> Union["DataFrame", "Series"]:
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            return self.apply(func, raw=False, args=args, kwargs=kwargs)  # type: ignore[arg-type]
        return result

    agg = aggregate

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method
        (`sum` in this case):

        >>> ser.rolling(2, win_type='gaussian').sum(std=3)
        0         NaN
        1    0.986207
        2    5.917243
        3    6.903450
        4    9.862071
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="weighted window sum",
        agg_method="sum",
    )
    def sum(self, numeric_only: bool = False, **kwargs: Any) -> Union["DataFrame", "Series"]:
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_weighted_sum  # type: ignore
        return self._apply(window_func, name="sum", numeric_only=numeric_only, **kwargs)

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type='gaussian').mean(std=3)
        0    NaN
        1    0.5
        2    3.0
        3    3.5
        4    5.0
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="weighted window mean",
        agg_method="mean",
    )
    def mean(self, numeric_only: bool = False, **kwargs: Any) -> Union["DataFrame", "Series"]:
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_weighted_mean  # type: ignore
        return self._apply(window_func, name="mean", numeric_only=numeric_only, **kwargs)

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type='gaussian').var(std=3)
        0     NaN
        1     0.5
        2     8.0
        3     4.5
        4    18.0
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="weighted window variance",
        agg_method="var",
    )
    def var(self, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Union["DataFrame", "Series"]:
        window_func = partial(window_aggregations.roll_weighted_var, ddof=ddof)
        kwargs.pop("name", None)
        return self._apply(window_func, name="var", numeric_only=numeric_only, **kwargs)

    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type='gaussian').std(std=3)
        0         NaN
        1    0.707107
        2    2.828427
        3    2.121320
        4    4.242641
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="weighted window standard deviation",
        agg_method="std",
    )
    def std(self, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Union["DataFrame", "Series"]:
        return zsqrt(self.var(ddof=ddof, name="std", numeric_only=numeric_only, **kwargs))

###############################################
# RollingAndExpandingMixin class definition
###############################################
class RollingAndExpandingMixin(BaseWindow):
    def count(self, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_sum  # type: ignore
        return self._apply(window_func, name="count", numeric_only=numeric_only)

    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Union["DataFrame", "Series"]:
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if not is_bool(raw):
            raise ValueError("raw parameter must be `True` or `False`")
        numba_args: tuple[Any, ...] = ()
        if maybe_use_numba(engine):
            if raw is False:
                raise ValueError("raw must be `True` when using the numba engine")
            numba_args, kwargs = prepare_function_arguments(func, args, kwargs, num_required_args=1)
            if self.method == "single":
                apply_func = generate_numba_apply_func(func, **get_jit_arguments(engine_kwargs))
            else:
                apply_func = generate_numba_table_func(func, **get_jit_arguments(engine_kwargs))
        elif engine in ("cython", None):
            if engine_kwargs is not None:
                raise ValueError("cython engine does not accept engine_kwargs")
            apply_func = self._generate_cython_apply_func(args, kwargs, raw, func)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")
        return self._apply(apply_func, name="apply", numba_args=numba_args)

    def _generate_cython_apply_func(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        raw: bool,
        function: Callable[..., Any],
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]:
        window_func = partial(
            window_aggregations.roll_apply,
            args=args,
            kwargs=kwargs,
            raw=bool(raw),
            function=function,
        )
        def apply_func(values: np.ndarray, begin: np.ndarray, end: np.ndarray, min_periods: int, raw_param: bool = raw) -> np.ndarray:
            from pandas import Series
            if not raw_param:
                values = Series(values, index=self._on, copy=False)
            return window_func(values, begin, end, min_periods)
        return apply_func

    @overload
    def pipe(
        self,
        func: Callable[[Any], Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...
    @overload
    def pipe(
        self,
        func: tuple[Callable[..., Any], str],
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...
    def pipe(
        self,
        func: Union[Callable[[Any], Any], tuple[Callable[..., Any], str]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return com.pipe(self, func, *args, **kwargs)

    def sum(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nansum)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_sum
                return self._numba_apply(sliding_sum, engine_kwargs)
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_sum  # type: ignore
        return self._apply(window_func, name="sum", numeric_only=numeric_only)

    def max(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmax)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_min_max
                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=True)
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_max  # type: ignore
        return self._apply(window_func, name="max", numeric_only=numeric_only)

    def min(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmin)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_min_max
                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=False)
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_min  # type: ignore
        return self._apply(window_func, name="min", numeric_only=numeric_only)

    def mean(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmean)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_mean
                return self._numba_apply(sliding_mean, engine_kwargs)
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_mean  # type: ignore
        return self._apply(window_func, name="mean", numeric_only=numeric_only)

    def median(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        if maybe_use_numba(engine):
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmedian)
            else:
                func = np.nanmedian
            return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_median_c  # type: ignore
        return self._apply(window_func, name="median", numeric_only=numeric_only)

    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        if maybe_use_numba(engine):
            if self.method == "table":
                raise NotImplementedError("std not supported with method='table'")
            from pandas.core._numba.kernels import sliding_var
            return zsqrt(self._numba_apply(sliding_var, engine_kwargs, ddof=ddof))
        window_func = window_aggregations.roll_var
        def zsqrt_func(values, begin, end, min_periods):
            return zsqrt(window_func(values, begin, end, min_periods, ddof=ddof))
        return self._apply(zsqrt_func, name="std", numeric_only=numeric_only)

    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        if maybe_use_numba(engine):
            if self.method == "table":
                raise NotImplementedError("var not supported with method='table'")
            from pandas.core._numba.kernels import sliding_var
            return self._numba_apply(sliding_var, engine_kwargs, ddof=ddof)
        window_func = partial(window_aggregations.roll_var, ddof=ddof)
        return self._apply(window_func, name="var", numeric_only=numeric_only)

    def skew(self, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_skew  # type: ignore
        return self._apply(window_func, name="skew", numeric_only=numeric_only)

    def sem(self, ddof: int = 1, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        self._validate_numeric_only("sem", numeric_only)
        return self.std(numeric_only=numeric_only) / (self.count(numeric_only=numeric_only) - ddof).pow(0.5)

    def kurt(self, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_kurt  # type: ignore
        return self._apply(window_func, name="kurt", numeric_only=numeric_only)

    def first(self, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_first  # type: ignore
        return self._apply(window_func, name="first", numeric_only=numeric_only)

    def last(self, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        window_func: Callable[..., np.ndarray] = window_aggregations.roll_last  # type: ignore
        return self._apply(window_func, name="last", numeric_only=numeric_only)

    def quantile(
        self,
        q: float,
        interpolation: Any = "linear",
        numeric_only: bool = False,
    ) -> Union["DataFrame", "Series"]:
        if q == 1.0:
            window_func = window_aggregations.roll_max
        elif q == 0.0:
            window_func = window_aggregations.roll_min
        else:
            window_func = partial(window_aggregations.roll_quantile, quantile=q, interpolation=interpolation)
        return self._apply(window_func, name="quantile", numeric_only=numeric_only)

    def rank(
        self,
        method: Union[str, Any] = "average",
        ascending: bool = True,
        pct: bool = False,
        numeric_only: bool = False,
    ) -> Union["DataFrame", "Series"]:
        window_func = partial(window_aggregations.roll_rank, method=method, ascending=ascending, percentile=pct)
        return self._apply(window_func, name="rank", numeric_only=numeric_only)

    def cov(
        self,
        other: Optional[Union["DataFrame", "Series"]] = None,
        pairwise: Optional[bool] = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Union["DataFrame", "Series"]:
        self._validate_numeric_only("cov", numeric_only)
        from pandas import Series
        def cov_func(x, y):
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size
            start, end = window_indexer.get_window_bounds(num_values=len(x_array),
                                                          min_periods=min_periods,
                                                          center=self.center,
                                                          closed=self.closed,
                                                          step=self.step)
            self._check_window_bounds(start, end, len(x_array))
            with np.errstate(all="ignore"):
                mean_x_y = window_aggregations.roll_mean(x_array * y_array, start, end, min_periods)
                mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
                mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
                count_x_y = window_aggregations.roll_sum(notna(x_array + y_array).astype(np.float64), start, end, 0)
                result = (mean_x_y - mean_x * mean_y) * (count_x_y / (count_x_y - ddof))
            return Series(result, index=x.index, name=x.name, copy=False)
        return self._apply_pairwise(self._selected_obj, other, pairwise, cov_func, numeric_only)

    def corr(
        self,
        other: Optional[Union["DataFrame", "Series"]] = None,
        pairwise: Optional[bool] = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Union["DataFrame", "Series"]:
        self._validate_numeric_only("corr", numeric_only)
        from pandas import Series
        def corr_func(x, y):
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size
            start, end = window_indexer.get_window_bounds(num_values=len(x_array),
                                                          min_periods=min_periods,
                                                          center=self.center,
                                                          closed=self.closed,
                                                          step=self.step)
            self._check_window_bounds(start, end, len(x_array))
            with np.errstate(all="ignore"):
                mean_x_y = window_aggregations.roll_mean(x_array * y_array, start, end, min_periods)
                mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
                mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
                count_x_y = window_aggregations.roll_sum(notna(x_array + y_array).astype(np.float64), start, end, 0)
                x_var = window_aggregations.roll_var(x_array, start, end, min_periods, ddof)
                y_var = window_aggregations.roll_var(y_array, start, end, min_periods, ddof)
                numerator = (mean_x_y - mean_x * mean_y) * (count_x_y / (count_x_y - ddof))
                denominator = (x_var * y_var) ** 0.5
                result = numerator / denominator
            return Series(result, index=x.index, name=x.name, copy=False)
        return self._apply_pairwise(self._selected_obj, other, pairwise, corr_func, numeric_only)


###############################################
# Rolling class definition
###############################################
class Rolling(RollingAndExpandingMixin):
    _attributes: list[str] = [
        "window",
        "min_periods",
        "center",
        "win_type",
        "on",
        "closed",
        "step",
        "method",
    ]

    def _validate(self) -> None:
        super()._validate()
        if (self.obj.empty or isinstance(self._on, (DatetimeIndex, TimedeltaIndex, PeriodIndex)) 
            or (hasattr(self._on, "dtype") and isinstance(self._on.dtype, ArrowDtype) and self._on.dtype.kind in "mM")) and isinstance(self.window, (str, BaseOffset, timedelta)):
            self._validate_datetimelike_monotonic()
            try:
                freq = to_offset(self.window)
            except (TypeError, ValueError) as err:
                raise ValueError(f"passed window {self.window} is not compatible with a datetimelike index") from err
            if isinstance(self._on, PeriodIndex):
                self._win_freq_i8 = freq.nanos / (self._on.freq.nanos / self._on.freq.n)
            else:
                try:
                    from pandas.core.arrays.datetimelike import dtype_to_unit
                    unit = dtype_to_unit(self._on.dtype)
                except TypeError:
                    unit = "ns"
                self._win_freq_i8 = Timedelta(freq.nanos).as_unit(unit)._value
            if self.min_periods is None:
                self.min_periods = 1
            if self.step is not None:
                raise NotImplementedError("step is not supported with frequency windows")
        elif isinstance(self.window, BaseIndexer):
            pass
        elif not is_integer(self.window) or self.window < 0:
            raise ValueError("window must be an integer 0 or greater")

    def _validate_datetimelike_monotonic(self) -> None:
        if self._on.hasnans:
            self._raise_monotonic_error("values must not have NaT")
        if not (self._on.is_monotonic_increasing or self._on.is_monotonic_decreasing):
            self._raise_monotonic_error("values must be monotonic")

    def _raise_monotonic_error(self, msg: str) -> None:
        on = self.on if self.on is not None else "index"
        raise ValueError(f"{on} {msg}")

    @doc(
        _shared_docs["aggregate"],
        see_also=dedent(
            """
        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrame data.
        """
        ),
        examples=dedent(
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2).sum()
             A     B     C
        0  NaN   NaN   NaN
        1  3.0   9.0  15.0
        2  5.0  11.0  17.0
        """
        ),
        klass="Series/Dataframe",
        axis="",
    )
    def aggregate(self, func: Optional[Union[Callable[..., Any], str]] = None, *args: Any, **kwargs: Any) -> Union["DataFrame", "Series"]:
        return super().aggregate(func, *args, **kwargs)
    agg = aggregate

    @doc(template_pipe)
    def pipe(
        self,
        func: Union[Callable[[Rolling, Any], Any], tuple[Callable[..., Any], str]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return super().pipe(func, *args, **kwargs)

    def count(self, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        return super().count(numeric_only)

    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Union["DataFrame", "Series"]:
        return super().apply(func, raw=raw, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)

    def sum(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        return super().sum(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    def max(
        self,
        numeric_only: bool = False,
        *args: Any,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
        **kwargs: Any,
    ) -> Union["DataFrame", "Series"]:
        return super().max(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    def min(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        return super().min(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    def mean(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        return super().mean(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    def median(
        self,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        return super().median(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        return super().std(ddof=ddof, numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Optional[Literal["cython", "numba"]] = None,
        engine_kwargs: Optional[dict[str, bool]] = None,
    ) -> Union["DataFrame", "Series"]:
        return super().var(ddof=ddof, numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    def skew(self, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        return super().skew(numeric_only=numeric_only)

    def sem(self, ddof: int = 1, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        return super().sem(ddof=ddof, numeric_only=numeric_only)

    def kurt(self, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        return super().kurt(numeric_only=numeric_only)

    def first(self, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        return super().first(numeric_only=numeric_only)

    def last(self, numeric_only: bool = False) -> Union["DataFrame", "Series"]:
        return super().last(numeric_only=numeric_only)

    def quantile(
        self,
        q: float,
        interpolation: Any = "linear",
        numeric_only: bool = False,
    ) -> Union["DataFrame", "Series"]:
        return super().quantile(q=q, interpolation=interpolation, numeric_only=numeric_only)

    def rank(
        self,
        method: Union[str, Any] = "average",
        ascending: bool = True,
        pct: bool = False,
        numeric_only: bool = False,
    ) -> Union["DataFrame", "Series"]:
        return super().rank(method=method, ascending=ascending, pct=pct, numeric_only=numeric_only)

    def cov(
        self,
        other: Optional[Union["DataFrame", "Series"]] = None,
        pairwise: Optional[bool] = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Union["DataFrame", "Series"]:
        return super().cov(other=other, pairwise=pairwise, ddof=ddof, numeric_only=numeric_only)

    def corr(
        self,
        other: Optional[Union["DataFrame", "Series"]] = None,
        pairwise: Optional[bool] = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Union["DataFrame", "Series"]:
        return super().corr(other=other, pairwise=pairwise, ddof=ddof, numeric_only=numeric_only)


Rolling.__doc__ = Window.__doc__

###############################################
# RollingGroupby class definition
###############################################
class RollingGroupby(BaseWindowGroupby, Rolling):
    _attributes: list[str] = Rolling._attributes + BaseWindowGroupby._attributes

    def _get_window_indexer(self) -> GroupbyIndexer:
        rolling_indexer: type[BaseIndexer]
        indexer_kwargs: Optional[dict[str, Any]] = None
        index_array: Optional[np.ndarray] = self._index_array  # type: ignore
        if isinstance(self.window, BaseIndexer):
            rolling_indexer = type(self.window)
            indexer_kwargs = self.window.__dict__.copy()  # type: ignore
            indexer_kwargs.pop("index_array", None)
            window = self.window
        elif hasattr(self, "_win_freq_i8") and self._win_freq_i8 is not None:
            rolling_indexer = VariableWindowIndexer
            window = self._win_freq_i8  # type: ignore
        else:
            rolling_indexer = FixedWindowIndexer
            window = self.window
        window_indexer = GroupbyIndexer(
            index_array=index_array,
            window_size=window,
            groupby_indices=self._grouper.indices,
            window_indexer=rolling_indexer,
            indexer_kwargs=indexer_kwargs,
        )
        return window_indexer

    def _validate_datetimelike_monotonic(self) -> None:
        if self._on.hasnans:
            self._raise_monotonic_error("values must not have NaT")
        for group_indices in self._grouper.indices.values():
            group_on = self._on.take(group_indices)
            if not (group_on.is_monotonic_increasing or group_on.is_monotonic_decreasing):
                on = "index" if self.on is None else self.on
                raise ValueError(f"Each group within {on} must be monotonic. Sort the values in {on} first.")
