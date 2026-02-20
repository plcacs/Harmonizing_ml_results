#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from collections.abc import Iterable
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pandas._libs import Interval
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import is_numeric_dtype, is_list_like, is_scalar
from pandas.core.indexes.api import Index, MultiIndex, default_index
from pandas.core.internals.blocks import BlockManager
from pandas.core.reshape.concat import concat
from pandas.plotting import boxplot_frame_groupby

from pandas.core.frame import DataFrame
from pandas.core.groupby.groupby import GroupBy
from pandas.core.series import Series
from pandas._typing import TakeIndexer
from pandas.core.groupby.plot import GroupByPlot


class DataFrameGroupBy(GroupBy[DataFrame]):
    _agg_examples_doc: str = (
        "Examples\n"
        "--------\n"
        ">>> data = {\"A\": [1, 1, 2, 2],\n"
        ">>>         \"B\": [1, 2, 3, 4],\n"
        ">>>         \"C\": [0.362838, 0.227877, 1.267767, -0.562860]}\n"
        ">>> df = pd.DataFrame(data)\n"
        ">>> df\n"
        "   A  B         C\n"
        "0  1  1  0.362838\n"
        "1  1  2  0.227877\n"
        "2  2  3  1.267767\n"
        "3  2  4 -0.562860\n\n"
        "The aggregation is for each column.\n\n"
        ">>> df.groupby('A').agg('min')\n"
        "   B         C\n"
        "A\n"
        "1  1  0.227877\n"
        "2  3 -0.562860\n\n"
        "Multiple aggregations\n\n"
        ">>> df.groupby('A').agg(['min', 'max'])\n"
        "    B             C\n"
        "  min max       min       max\n"
        "A\n"
        "1   1   2  0.227877  0.362838\n"
        "2   3   4 -0.562860  1.267767\n\n"
        "Select a column for aggregation\n\n"
        ">>> df.groupby('A').B.agg(['min', 'max'])\n"
        "   min  max\n"
        "A\n"
        "1    1    2\n"
        "2    3    4\n\n"
        "User-defined function for aggregation\n\n"
        ">>> df.groupby('A').agg(lambda x: sum(x) + 2)\n"
        "    B          C\n"
        "A\n"
        "1       5       2.590715\n"
        "2       9       2.704907\n\n"
        "Different aggregations per column\n\n"
        ">>> df.groupby('A').agg({\"B\": [\"min\", \"max\"], \"C\": \"sum\"})\n"
        "    B             C\n"
        "  min max       sum\n"
        "A\n"
        "1   1   2  0.590715\n"
        "2   3   4  0.704907\n\n"
        "To control the output names with different aggregations per column,\n"
        "pandas supports \"named aggregation\"\n\n"
        ">>> df.groupby(\"A\").agg(\n"
        ">>>     b_min=pd.NamedAgg(column=\"B\", aggfunc=\"min\"),\n"
        ">>>     c_sum=pd.NamedAgg(column=\"C\", aggfunc=\"sum\")\n"
        ">>> )\n"
        "   b_min     c_sum\n"
        "A\n"
        "1      1  0.590715\n"
        "2      3  0.704907\n"
    )

    def aggregate(
        self,
        func: Optional[Union[Callable[..., Any], str, Iterable[Any], Dict[Any, Any]]] = None,
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DataFrame:
        relabeling, func, columns, order = self._reconstruct_func(func, **kwargs)
        func = self._maybe_mangle_lambdas(func)

        if self._maybe_use_numba(engine):
            kwargs["engine"] = engine
            kwargs["engine_kwargs"] = engine_kwargs

        op = self._groupby_apply(func, args=args, kwargs=kwargs)
        result: Optional[DataFrame] = op.agg()
        if not isinstance(func, dict) and result is not None:
            if not self.as_index and is_list_like(func):
                return result.reset_index()
            else:
                return result
        elif relabeling:
            result = result  # type: ignore[assignment]
            result = result.iloc[:, order]
            result.columns = columns  # type: ignore[assignment]
        if result is None:
            if "engine" in kwargs:
                del kwargs["engine"]
                del kwargs["engine_kwargs"]
            if self._maybe_use_numba(engine):
                return self._aggregate_with_numba(func, *args, engine_kwargs=engine_kwargs, **kwargs)
            if self._grouper.nkeys > 1:
                return self._python_agg_general(func, *args, **kwargs)
            elif args or kwargs:
                result = self._aggregate_frame(func, *args, **kwargs)
            else:
                gba = self._groupby_apply([func], args=(), kwargs={})
                try:
                    result = gba.agg()
                except ValueError as err:
                    if "No objects to concatenate" not in str(err):
                        raise
                    result = self._aggregate_frame(func)
                else:
                    result = result  # type: ignore[assignment]
                    result.columns = self._obj_with_exclusions.columns.copy(deep=False)
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    agg = aggregate

    def _python_agg_general(
        self, func: Callable[[Any], Any], *args: Any, **kwargs: Any
    ) -> DataFrame:
        f = lambda x: func(x, *args, **kwargs)
        if self.ngroups == 0:
            return self._python_apply_general(f, self._selected_obj, is_agg=True)
        obj: DataFrame = self._obj_with_exclusions
        if not len(obj.columns):
            return self._python_apply_general(f, self._selected_obj)
        output: Dict[int, Any] = {}
        for idx, (name, ser) in enumerate(obj.items()):
            result = self._grouper.agg_series(ser, f)
            output[idx] = result
        res: DataFrame = self.obj._constructor(output)
        res.columns = obj.columns.copy(deep=False)
        return self._wrap_aggregated_output(res)

    def _aggregate_frame(self, func: Callable[[DataFrame], Any], *args: Any, **kwargs: Any) -> DataFrame:
        if self._grouper.nkeys != 1:
            raise AssertionError("Number of keys must be 1")
        obj: DataFrame = self._obj_with_exclusions
        result: Dict[Hashable, Union[DataFrame, np.ndarray, Series]] = {}
        for name, grp_df in self._grouper.get_iterator(obj):
            fres = func(grp_df, *args, **kwargs)
            result[name] = fres
        result_index = self._grouper.result_index
        out: DataFrame = self.obj._constructor(result, index=obj.columns, columns=result_index)
        out = out.T
        return out

    def _wrap_applied_output(
        self,
        data: DataFrame,
        values: List[Any],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> Union[DataFrame, Series]:
        if len(values) == 0:
            if is_transform:
                res_index = data.index
            elif not self.group_keys:
                res_index = None
            else:
                res_index = self._grouper.result_index
            result: DataFrame = self.obj._constructor(index=res_index, columns=data.columns)
            result = result.astype(data.dtypes)
            return result
        first_not_none: Optional[Any] = next((x for x in values if x is not None), None)
        if first_not_none is None:
            result = self.obj._constructor(columns=data.columns)
            result = result.astype(data.dtypes)
            return result
        elif isinstance(first_not_none, DataFrame):
            return self._concat_objects(values, not_indexed_same=not_indexed_same, is_transform=is_transform)
        key_index: Optional[Index] = self._grouper.result_index if self.as_index else None
        if isinstance(first_not_none, (np.ndarray, Index)):
            if not self._is_hashable(self._selection):
                name = tuple(self._selection)  # type: ignore
            else:
                name = self._selection  # type: ignore
            return self.obj._constructor_sliced(values, index=key_index, name=name)
        elif not isinstance(first_not_none, Series):
            if self.as_index:
                return self.obj._constructor_sliced(values, index=key_index)
            else:
                result = self.obj._constructor(values, columns=[self._selection])  # type: ignore
                result = self._insert_inaxis_grouper(result)
                return result
        else:
            return self._wrap_applied_output_series(values, not_indexed_same, first_not_none, key_index, is_transform)

    def _wrap_applied_output_series(
        self,
        values: List[Series],
        not_indexed_same: bool,
        first_not_none: Series,
        key_index: Optional[Index],
        is_transform: bool,
    ) -> Union[DataFrame, Series]:
        kwargs: Dict[str, Any] = first_not_none._construct_axes_dict()
        backup: Series = Series(**kwargs)
        values = [x if (x is not None) else backup for x in values]
        all_indexed_same = self._all_indexes_same([x.index for x in values])
        if not all_indexed_same:
            return self._concat_objects(values, not_indexed_same=True, is_transform=is_transform)
        stacked_values = np.vstack([np.asarray(v) for v in values])
        index: Optional[Index] = key_index
        columns: Index = first_not_none.index.copy()
        if columns.name is None:
            names = {v.name for v in values}
            if len(names) == 1:
                columns.name = next(iter(names))
        if stacked_values.dtype == object:
            stacked_values = stacked_values.tolist()
        result: DataFrame = self.obj._constructor(stacked_values, index=index, columns=columns)
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
        return result

    def _cython_transform(
        self,
        how: str,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        mgr: BlockManager = self._get_data_to_aggregate(numeric_only=numeric_only, name=how)
        def arr_func(bvalues: Any) -> Any:
            return self._grouper._cython_operation("transform", bvalues, how, 1, **kwargs)
        res_mgr = mgr.apply(arr_func)
        res_df: DataFrame = self.obj._constructor_from_mgr(res_mgr, axes=res_mgr.axes)
        return res_df

    def _transform_general(
        self, func: Callable[..., Any], engine: Optional[str], engine_kwargs: Optional[Dict[str, Any]], *args: Any, **kwargs: Any
    ) -> DataFrame:
        if self._maybe_use_numba(engine):
            return self._transform_with_numba(func, *args, engine_kwargs=engine_kwargs, **kwargs)
        applied: List[DataFrame] = []
        obj: DataFrame = self._obj_with_exclusions
        gen = self._grouper.get_iterator(obj)
        fast_path, slow_path = self._define_paths(func, *args, **kwargs)
        try:
            name, group = next(gen)
        except StopIteration:
            pass
        else:
            object.__setattr__(group, "name", name)
            try:
                path, res = self._choose_path(fast_path, slow_path, group)
            except ValueError as err:
                msg = "transform must return a scalar value for each group"
                raise ValueError(msg) from err
            if group.size > 0:
                res = _wrap_transform_general_frame(self.obj, group, res)
                applied.append(res)
        for name, group in gen:
            if group.size == 0:
                continue
            object.__setattr__(group, "name", name)
            res = fast_path(group)
            res = _wrap_transform_general_frame(self.obj, group, res)
            applied.append(res)
        concat_index: Index = obj.columns
        concatenated = concat(applied, axis=0, verify_integrity=False, ignore_index=True)
        concatenated = concatenated.reindex(concat_index, axis=1)
        return self._set_result_index_ordered(concatenated)

    def transform(
        self, func: Union[str, Callable[..., Any]], *args: Any, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> DataFrame:
        return self._transform(func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs)

    def _define_paths(
        self, func: Union[str, Callable[..., Any]], *args: Any, **kwargs: Any
    ) -> Tuple[Callable[[DataFrame], Any], Callable[[DataFrame], Any]]:
        if isinstance(func, str):
            fast_path = lambda group: getattr(group, func)(*args, **kwargs)
            slow_path = lambda group: group.apply(lambda x: getattr(x, func)(*args, **kwargs), axis=0)
        else:
            fast_path = lambda group: func(group, *args, **kwargs)
            slow_path = lambda group: group.apply(lambda x: func(x, *args, **kwargs), axis=0)
        return fast_path, slow_path

    def _choose_path(
        self,
        fast_path: Callable[[DataFrame], Any],
        slow_path: Callable[[DataFrame], Any],
        group: DataFrame,
    ) -> Tuple[Callable[[DataFrame], Any], Any]:
        path: Callable[[DataFrame], Any] = slow_path
        res = slow_path(group)
        if self.ngroups == 1:
            return path, res
        try:
            res_fast = fast_path(group)
        except AssertionError:
            raise
        except Exception:
            return path, res
        if isinstance(res_fast, DataFrame):
            if not res_fast.columns.equals(group.columns):
                return path, res
        elif isinstance(res_fast, Series):
            if not res_fast.index.equals(group.columns):
                return path, res
        else:
            return path, res
        if res_fast.equals(res):
            path = fast_path
        return path, res

    def filter(
        self, func: Callable[[DataFrame, Any], Any], dropna: bool = True, *args: Any, **kwargs: Any
    ) -> DataFrame:
        indices: List[Any] = []
        obj: DataFrame = self._selected_obj
        gen = self._grouper.get_iterator(obj)
        for name, group in gen:
            object.__setattr__(group, "name", name)
            res = func(group, *args, **kwargs)
            try:
                res = res.squeeze()
            except AttributeError:
                pass
            if isinstance(res, (bool, np.bool_)) or (is_scalar(res) and np.isnan(res)):
                if not np.isnan(res) and res:
                    indices.append(self._get_index(name))
            else:
                raise TypeError(
                    f"filter function returned a {type(res).__name__}, but expected a scalar bool"
                )
        return self._apply_filter(indices, dropna)

    def __getitem__(self, key: Any) -> Union[DataFrameGroupBy, SeriesGroupBy]:
        if isinstance(key, tuple) and len(key) > 1:
            raise ValueError(
                "Cannot subset columns with a tuple with more than one element. Use a list instead."
            )
        return super().__getitem__(key)

    def _gotitem(self, key: Any, ndim: int, subset: Optional[DataFrame] = None) -> Union[DataFrameGroupBy, SeriesGroupBy]:
        if ndim == 2:
            if subset is None:
                subset = self.obj
            return DataFrameGroupBy(
                subset,
                self.keys,
                level=self.level,
                grouper=self._grouper,
                exclusions=self.exclusions,
                selection=key,
                as_index=self.as_index,
                sort=self.sort,
                group_keys=self.group_keys,
                observed=self.observed,
                dropna=self.dropna,
            )
        elif ndim == 1:
            if subset is None:
                subset = self.obj[key]
            return SeriesGroupBy(
                subset,
                self.keys,
                level=self.level,
                grouper=self._grouper,
                exclusions=self.exclusions,
                selection=key,
                as_index=self.as_index,
                sort=self.sort,
                group_keys=self.group_keys,
                observed=self.observed,
                dropna=self.dropna,
            )
        raise AssertionError("invalid ndim for _gotitem")

    def _get_data_to_aggregate(self, *, numeric_only: bool = False, name: Optional[str] = None) -> BlockManager:
        obj: DataFrame = self._obj_with_exclusions
        mgr: BlockManager = obj._mgr
        if numeric_only:
            mgr = mgr.get_numeric_data()
        return mgr

    def _wrap_agged_manager(self, mgr: BlockManager) -> DataFrame:
        return self.obj._constructor_from_mgr(mgr, axes=mgr.axes)

    def _apply_to_column_groupbys(self, func: Callable[[SeriesGroupBy], Any]) -> DataFrame:
        obj: DataFrame = self._obj_with_exclusions
        columns: Index = obj.columns
        sgbs = (
            SeriesGroupBy(
                obj.iloc[:, i],
                selection=colname,
                grouper=self._grouper,
                exclusions=self.exclusions,
                observed=self.observed,
            )
            for i, colname in enumerate(obj.columns)
        )
        results: List[Any] = [func(sgb) for sgb in sgbs]
        if not len(results):
            res_df = DataFrame([], columns=columns, index=self._grouper.result_index)
        else:
            res_df = concat(results, keys=columns, axis=1)
        if not self.as_index:
            res_df.index = default_index(len(res_df))
            res_df = self._insert_inaxis_grouper(res_df)
        return res_df

    def nunique(self, dropna: bool = True) -> DataFrame:
        return self._apply_to_column_groupbys(lambda sgb: sgb.nunique(dropna))

    def idxmax(self, skipna: bool = True, numeric_only: bool = False) -> DataFrame:
        return self._idxmax_idxmin("idxmax", numeric_only=numeric_only, skipna=skipna)

    def idxmin(self, skipna: bool = True, numeric_only: bool = False) -> DataFrame:
        return self._idxmax_idxmin("idxmin", numeric_only=numeric_only, skipna=skipna)

    boxplot = boxplot_frame_groupby

    def value_counts(
        self,
        subset: Optional[Sequence[Hashable]] = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> Union[DataFrame, Series]:
        return self._value_counts(subset, normalize, sort, ascending, dropna)

    def take(self, indices: TakeIndexer, **kwargs: Any) -> DataFrame:
        result: DataFrame = self._op_via_apply("take", indices=indices, **kwargs)
        return result

    def skew(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        def alt(obj: Any) -> Any:
            raise TypeError(f"'skew' is not supported for dtype={obj.dtype}")
        return self._cython_agg_general("skew", alt=alt, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def kurt(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        return self._cython_agg_general("kurt", alt=None, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @property
    def plot(self) -> GroupByPlot:
        result: GroupByPlot = GroupByPlot(self)
        return result

    def corr(
        self,
        method: str = "pearson",
        min_periods: int = 1,
        numeric_only: bool = False,
    ) -> DataFrame:
        result: DataFrame = self._op_via_apply("corr", method=method, min_periods=min_periods, numeric_only=numeric_only)
        return result

    def cov(
        self,
        min_periods: Optional[int] = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> DataFrame:
        result: DataFrame = self._op_via_apply("cov", min_periods=min_periods, ddof=ddof, numeric_only=numeric_only)
        return result

    def hist(
        self,
        column: Any = None,
        by: Any = None,
        grid: bool = True,
        xlabelsize: Optional[Any] = None,
        xrot: Optional[Any] = None,
        ylabelsize: Optional[Any] = None,
        yrot: Optional[Any] = None,
        ax: Any = None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: Optional[Any] = None,
        layout: Optional[Any] = None,
        bins: Union[int, Sequence[int]] = 10,
        backend: Any = None,
        legend: bool = False,
        **kwargs: Any,
    ) -> Any:
        result: Any = self._op_via_apply(
            "hist",
            column=column,
            by=by,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
            bins=bins,
            backend=backend,
            legend=legend,
            **kwargs,
        )
        return result

    def corrwith(
        self,
        other: Union[DataFrame, Series],
        drop: bool = False,
        method: Union[str, Callable[..., Any]] = "pearson",
        numeric_only: bool = False,
    ) -> DataFrame:
        warnings.warn(
            "DataFrameGroupBy.corrwith is deprecated",
            FutureWarning,
            stacklevel=self._find_stack_level(),
        )
        result: DataFrame = self._op_via_apply("corrwith", other=other, drop=drop, method=method, numeric_only=numeric_only)
        return result


def _wrap_transform_general_frame(obj: DataFrame, group: DataFrame, res: Any) -> DataFrame:
    if isinstance(res, Series):
        if res.index.is_(obj.index):
            res_frame: DataFrame = concat([res] * len(group.columns), axis=1, ignore_index=True)
            res_frame.columns = group.columns
            res_frame.index = group.index
        else:
            res_frame = obj._constructor(
                np.tile(res.values, (len(group.index), 1)),
                columns=group.columns,
                index=group.index,
            )
        return res_frame
    elif isinstance(res, DataFrame) and not res.index.is_(group.index):
        return res._align_frame(group)[0]
    else:
        return res

# Note: Functions such as _reconstruct_func, _maybe_mangle_lambdas, _maybe_use_numba, 
# _groupby_apply, _python_apply_general, _wrap_aggregated_output, _insert_inaxis_grouper, 
# _concat_objects, _is_hashable, _all_indexes_same, _op_via_apply, _idxmax_idxmin, 
# _cython_agg_general, _transform, _transform_with_numba, _set_result_index_ordered, 
# _apply_filter, _find_stack_level, and attribute _selected_obj are assumed to be defined 
# elsewhere in the class hierarchy or mixins.
//
// Similarly, SeriesGroupBy is assumed to be defined with appropriate type annotations.
