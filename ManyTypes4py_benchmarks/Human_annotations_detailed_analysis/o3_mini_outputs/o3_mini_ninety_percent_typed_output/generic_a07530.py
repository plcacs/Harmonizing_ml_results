from __future__ import annotations
from collections.abc import Iterable, Callable
from typing import Any, Optional, Union, Dict, List, Sequence, cast
import numpy as np
import warnings

from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    ArrayLike,
    BlockManager,
    CorrelationMethod,
    IndexLabel,
    Manager,
    SingleBlockManager,
    TakeIndexer,
)
from pandas.core.dtypes.common import is_list_like, is_bool, is_scalar, isna, notna
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby.groupby import GroupBy
from pandas.core.series import Series
from pandas.core.indexes.base import Index, MultiIndex, default_index
from pandas.util._decorators import Appender, Substitution
from pandas.plotting import boxplot_frame_groupby

# NOTE: Other necessary imports for types are assumed to be available.

class SeriesGroupBy(GroupBy[Series]):
    def aggregate(
        self,
        func: Optional[Union[str, Callable[..., Any], List[Any], Dict[Any, Any]]] = None,
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Series:
        relabeling = func is None
        columns: Optional[Any] = None
        if relabeling:
            columns, func = validate_func_kwargs(kwargs)
            kwargs = {}
        if isinstance(func, str):
            if maybe_use_numba(engine) and engine is not None:
                kwargs["engine"] = engine
            if engine_kwargs is not None:
                kwargs["engine_kwargs"] = engine_kwargs
            return getattr(self, func)(*args, **kwargs)
        elif isinstance(func, Iterable) and not isinstance(func, (str, bytes)):
            func = maybe_mangle_lambdas(func)
            kwargs["engine"] = engine
            kwargs["engine_kwargs"] = engine_kwargs
            ret = self._aggregate_multiple_funcs(func, *args, **kwargs)
            if relabeling:
                assert columns is not None
                ret.columns = columns
            if not self.as_index:
                ret = ret.reset_index()
            return ret
        else:
            if maybe_use_numba(engine):
                return self._aggregate_with_numba(func, *args, engine_kwargs=engine_kwargs, **kwargs)
            if self.ngroups == 0:
                obj = self._obj_with_exclusions
                return self.obj._constructor([], name=self.obj.name, index=self._grouper.result_index, dtype=obj.dtype)
            return self._python_agg_general(func, *args, **kwargs)

    agg = aggregate

    def _python_agg_general(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Series:
        f: Callable[[Series], Any] = lambda x: func(x, *args, **kwargs)
        obj: Series = self._obj_with_exclusions
        result: Any = self._grouper.agg_series(obj, f)
        res: Series = obj._constructor(result, name=obj.name)
        return self._wrap_aggregated_output(res)

    def _aggregate_multiple_funcs(self, arg: Any, *args: Any, **kwargs: Any) -> DataFrame:
        if isinstance(arg, dict):
            raise SpecificationError("nested renamer is not supported")
        if any(isinstance(x, (tuple, list)) for x in arg):
            arg = ((x, x) if not isinstance(x, (tuple, list)) else x for x in arg)
        else:
            columns = (com.get_callable_name(f) or f for f in arg)
            arg = zip(columns, arg)
        results: Dict[base.OutputKey, Union[DataFrame, Series]] = {}
        with com.temp_setattr(self, "as_index", True):
            for idx, (name, func) in enumerate(arg):
                key = base.OutputKey(label=name, position=idx)
                results[key] = self.aggregate(func, *args, **kwargs)
        if any(isinstance(x, DataFrame) for x in results.values()):
            from pandas import concat
            res_df: DataFrame = concat(list(results.values()), axis=1, keys=[key.label for key in results])
            return res_df
        indexed_output: Dict[int, Series] = {key.position: val for key, val in results.items()}  # type: ignore
        output: DataFrame = self.obj._constructor_expanddim(indexed_output, index=None)
        output.columns = Index(key.label for key in results)
        return output

    def _wrap_applied_output(
        self,
        data: Series,
        values: List[Any],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> Union[DataFrame, Series]:
        if len(values) == 0:
            if is_transform:
                res_index: Optional[Index] = data.index
            elif not self.group_keys:
                res_index = None
            else:
                res_index = self._grouper.result_index
            return self.obj._constructor([], name=self.obj.name, index=res_index, dtype=data.dtype)
        assert values is not None
        if isinstance(values[0], dict):
            index: Index = self._grouper.result_index
            res_df: DataFrame = self.obj._constructor_expanddim(values, index=index)
            res_ser: Series = res_df.stack()
            res_ser.name = self.obj.name
            return res_ser
        elif isinstance(values[0], (Series, DataFrame)):
            result = self._concat_objects(
                values,
                not_indexed_same=not_indexed_same,
                is_transform=is_transform,
            )
            if isinstance(result, Series):
                result.name = self.obj.name
            if not self.as_index and not_indexed_same:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result
        else:
            result = self.obj._constructor(data=values, index=self._grouper.result_index, name=self.obj.name)
            if not self.as_index:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result

    @Substitution(klass="Series", example=__examples_series_doc)
    @Appender(_transform_template)
    def transform(self, func: Union[str, Callable[..., Any]], *args: Any, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Series:
        return self._transform(func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs)

    def _cython_transform(self, how: str, numeric_only: bool = False, **kwargs: Any) -> Series:
        obj: Series = self._obj_with_exclusions
        try:
            result = self._grouper._cython_operation("transform", obj._values, how, 0, **kwargs)
        except NotImplementedError as err:
            raise TypeError(f"{how} is not supported for {obj.dtype} dtype") from err
        return obj._constructor(result, index=self.obj.index, name=obj.name)

    def _transform_general(self, func: Callable[..., Any], engine: Any, engine_kwargs: Any, *args: Any, **kwargs: Any) -> Series:
        if maybe_use_numba(engine):
            return self._transform_with_numba(func, *args, engine_kwargs=engine_kwargs, **kwargs)
        assert callable(func)
        klass = type(self.obj)
        results: List[Series] = []
        for name, group in self._grouper.get_iterator(self._obj_with_exclusions):
            object.__setattr__(group, "name", name)
            res = func(group, *args, **kwargs)
            results.append(klass(res, index=group.index))
        if results:
            from pandas.core.reshape.concat import concat
            concatenated: Series = concat(results, ignore_index=True)
            result = self._set_result_index_ordered(concatenated)
        else:
            result = self.obj._constructor(dtype=np.float64)
        result.name = self.obj.name
        return result

    def filter(self, func: Callable[..., Any], dropna: bool = True, *args: Any, **kwargs: Any) -> Series:
        if isinstance(func, str):
            wrapper: Callable[[Any], Any] = lambda x: getattr(x, func)(*args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)
        def true_and_notna(x: Any) -> bool:
            b = wrapper(x)
            return notna(b) and b
        try:
            indices: List[Any] = [self._get_index(name)
                                    for name, group in self._grouper.get_iterator(self._obj_with_exclusions)
                                    if true_and_notna(group)]
        except (ValueError, TypeError) as err:
            raise TypeError("the filter must return a boolean result") from err
        filtered: Series = self._apply_filter(indices, dropna)
        return filtered

    def nunique(self, dropna: bool = True) -> Union[Series, DataFrame]:
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        val = self.obj._values
        codes, uniques = algorithms.factorize(val, use_na_sentinel=dropna, sort=False)
        if self._grouper.has_dropped_na:
            mask = ids >= 0
            ids = ids[mask]
            codes = codes[mask]
        group_index = get_group_index(
            labels=[ids, codes],
            shape=(ngroups, len(uniques)),
            sort=False,
            xnull=dropna,
        )
        if dropna:
            mask = group_index >= 0
            if (~mask).any():
                ids = ids[mask]
                group_index = group_index[mask]
        mask = duplicated(group_index, "first")
        res = np.bincount(ids[~mask], minlength=ngroups)
        res = ensure_int64(res)
        ri: Index = self._grouper.result_index
        result: Union[Series, DataFrame] = self.obj._constructor(res, index=ri, name=self.obj.name)
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    @doc(Series.describe)
    def describe(self, percentiles: Optional[Sequence[float]] = None, include: Any = None, exclude: Any = None) -> Series:
        return super().describe(percentiles=percentiles, include=include, exclude=exclude)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins: Optional[Union[int, Sequence[int]]] = None,
        dropna: bool = True,
    ) -> Union[Series, DataFrame]:
        name: str = "proportion" if normalize else "count"
        if bins is None:
            result: Series = self._value_counts(normalize=normalize, sort=sort, ascending=ascending, dropna=dropna)
            result.name = name
            return result
        from pandas.core.reshape.merge import get_join_indexers
        from pandas.core.reshape.tile import cut
        ids = self._grouper.ids
        val = self.obj._values
        index_names: List[Any] = self._grouper.names + [self.obj.name]
        if isinstance(val.dtype, CategoricalDtype) or (bins is not None and not np.iterable(bins)):
            ser: Series = self.apply(Series.value_counts, normalize=normalize, sort=sort, ascending=ascending, bins=bins)
            ser.name = name
            ser.index.names = index_names
            return ser
        mask = ids != -1
        ids, val = ids[mask], val[mask]
        lab: Union[Index, np.ndarray]
        if bins is None:
            lab, lev = algorithms.factorize(val, sort=True)
            llab = lambda lab, inc: lab[inc]
        else:
            cat_ser: Series = cut(Series(val, copy=False), bins, include_lowest=True)
            cat_obj = cast("Categorical", cat_ser._values)
            lev = cat_obj.categories
            lab = lev.take(cat_obj.codes, allow_fill=True, fill_value=lev._na_value)
            llab = lambda lab, inc: lab[inc]._multiindex.codes[-1]
        if isinstance(lab.dtype, IntervalDtype):
            lab_interval = cast(Interval, lab)
            sorter = np.lexsort((lab_interval.left, lab_interval.right, ids))
        else:
            sorter = np.lexsort((lab, ids))
        ids, lab = ids[sorter], lab[sorter]
        idchanges = 1 + np.nonzero(ids[1:] != ids[:-1])[0]
        idx = np.r_[0, idchanges]
        if not len(ids):
            idx = idchanges
        lchanges = llab(lab, slice(1, None)) != llab(lab, slice(None, -1))
        inc = np.r_[True, lchanges]
        if not len(val):
            inc = lchanges
        inc[idx] = True
        out = np.diff(np.nonzero(np.r_[inc, True])[0])
        from functools import partial
        rep = partial(np.repeat, repeats=np.add.reduceat(inc, idx))
        if isinstance(self._grouper.result_index, MultiIndex):
            codes = list(self._grouper.result_index.codes)
        else:
            codes = [algorithms.factorize(self._grouper.result_index, sort=self._grouper._sort, use_na_sentinel=self._grouper.dropna)[0]]
        codes = [rep(level_codes) for level_codes in codes] + [llab(lab, inc)]
        levels = self._grouper.levels + [lev]
        if dropna:
            mask = codes[-1] != -1
            if mask.all():
                dropna = False
            else:
                out, codes = out[mask], [level_codes[mask] for level_codes in codes]
        if normalize:
            out = out.astype("float")
            d = np.diff(np.r_[idx, len(ids)])
            if dropna:
                m = ids[lab == -1]
                np.add.at(d, m, -1)
                acc = rep(d)[mask]
            else:
                acc = rep(d)
            out /= acc
        if sort and bins is None:
            cat = ids[inc][mask] if dropna else ids[inc]
            sorter = np.lexsort((out if ascending else -out, cat))
            out, codes[-1] = out[sorter], codes[-1][sorter]
        if bins is not None:
            diff = np.zeros(len(out), dtype="bool")
            for level_codes in codes[:-1]:
                diff |= np.r_[True, level_codes[1:] != level_codes[:-1]]
            ncat, nbin = diff.sum(), len(levels[-1])
            left = [np.repeat(np.arange(ncat), nbin), np.tile(np.arange(nbin), ncat)]
            right = [diff.cumsum() - 1, codes[-1]]
            _, idx_join = get_join_indexers(left, right, sort=False, how="left")  # type: ignore
            if idx_join is not None:
                out = np.where(idx_join != -1, out[idx_join], 0)
            if sort:
                sorter = np.lexsort((out if ascending else -out, left[0]))
                out, left[-1] = out[sorter], left[-1][sorter]
            def build_codes(lev_codes: np.ndarray) -> np.ndarray:
                return np.repeat(lev_codes[diff], nbin)
            codes = [build_codes(lev_codes) for lev_codes in codes[:-1]]
            codes.append(left[-1])
        mi = MultiIndex(levels=levels, codes=codes, names=index_names, verify_integrity=False)
        if is_integer_dtype(out.dtype):
            out = ensure_int64(out)
        result = self.obj._constructor(out, index=mi, name=name)
        if not self.as_index:
            result = result.reset_index()
        return result

    def take(
        self,
        indices: TakeIndexer,
        **kwargs: Any,
    ) -> Series:
        result: Series = self._op_via_apply("take", indices=indices, **kwargs)
        return result

    def skew(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Series:
        return self._cython_agg_general("skew", alt=None, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def kurt(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Series:
        def alt(obj: Series) -> Any:
            raise TypeError(f"'kurt' is not supported for dtype={obj.dtype}")
        return self._cython_agg_general("kurt", alt=alt, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @property
    @doc(Series.plot.__doc__)
    def plot(self) -> GroupByPlot:
        result = GroupByPlot(self)
        return result

    @doc(Series.nlargest.__doc__)
    def nlargest(
        self,
        n: int = 5,
        keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        f: Callable[..., Any] = partial(Series.nlargest, n=n, keep=keep)
        data: Series = self._obj_with_exclusions
        result: Series = self._python_apply_general(f, data, not_indexed_same=True)
        return result

    @doc(Series.nsmallest.__doc__)
    def nsmallest(
        self,
        n: int = 5,
        keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        f: Callable[..., Any] = partial(Series.nsmallest, n=n, keep=keep)
        data: Series = self._obj_with_exclusions
        result: Series = self._python_apply_general(f, data, not_indexed_same=True)
        return result

    def idxmin(self, skipna: bool = True) -> Series:
        return self._idxmax_idxmin("idxmin", skipna=skipna)

    def idxmax(self, skipna: bool = True) -> Series:
        return self._idxmax_idxmin("idxmax", skipna=skipna)

    @doc(Series.corr.__doc__)
    def corr(
        self,
        other: Series,
        method: CorrelationMethod = "pearson",
        min_periods: Optional[int] = None,
    ) -> Series:
        result: Series = self._op_via_apply("corr", other=other, method=method, min_periods=min_periods)
        return result

    @doc(Series.cov.__doc__)
    def cov(
        self,
        other: Series,
        min_periods: Optional[int] = None,
        ddof: Optional[int] = 1
    ) -> Series:
        result: Series = self._op_via_apply("cov", other=other, min_periods=min_periods, ddof=ddof)
        return result

    @property
    def is_monotonic_increasing(self) -> Series:
        return self.apply(lambda ser: ser.is_monotonic_increasing)

    @property
    def is_monotonic_decreasing(self) -> Series:
        return self.apply(lambda ser: ser.is_monotonic_decreasing)

    @doc(Series.hist.__doc__)
    def hist(
        self,
        by: Any = None,
        ax: Any = None,
        grid: bool = True,
        xlabelsize: Optional[int] = None,
        xrot: Any = None,
        ylabelsize: Optional[int] = None,
        yrot: Optional[float] = None,
        figsize: Optional[tuple[float, float]] = None,
        bins: Union[int, Sequence[int]] = 10,
        backend: Optional[str] = None,
        legend: bool = False,
        **kwargs: Any,
    ):
        result = self._op_via_apply(
            "hist",
            by=by,
            ax=ax,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            figsize=figsize,
            bins=bins,
            backend=backend,
            legend=legend,
            **kwargs,
        )
        return result

    @property
    @doc(Series.dtype.__doc__)
    def dtype(self) -> Series:
        return self.apply(lambda ser: ser.dtype)

    def unique(self) -> Series:
        result: Series = self._op_via_apply("unique")
        return result


class DataFrameGroupBy(GroupBy[DataFrame]):
    _agg_examples_doc = __examples_dataframe_doc

    def aggregate(
        self,
        func: Optional[Union[str, Callable[..., Any], List[Any], Dict[Any, Any]]] = None,
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> DataFrame:
        relabeling, func, columns, order = reconstruct_func(func, **kwargs)
        func = maybe_mangle_lambdas(func)
        if maybe_use_numba(engine):
            kwargs["engine"] = engine
            kwargs["engine_kwargs"] = engine_kwargs
        op = GroupByApply(self, func, args=args, kwargs=kwargs)
        result: Optional[Union[DataFrame, Series]] = op.agg()
        if not is_dict_like(func) and result is not None:
            if not self.as_index and is_list_like(func):
                return cast(DataFrame, result).reset_index()
            else:
                return cast(DataFrame, result)
        elif relabeling:
            result = cast(DataFrame, result)
            result = result.iloc[:, order]
            result.columns = columns  # type: ignore[assignment]
        if result is None:
            if "engine" in kwargs:
                del kwargs["engine"]
                del kwargs["engine_kwargs"]
            if maybe_use_numba(engine):
                return self._aggregate_with_numba(func, *args, engine_kwargs=engine_kwargs, **kwargs)
            if self._grouper.nkeys > 1:
                return self._python_agg_general(func, *args, **kwargs)
            elif args or kwargs:
                result = self._aggregate_frame(func, *args, **kwargs)
            else:
                gba = GroupByApply(self, [func], args=(), kwargs={})
                try:
                    result = gba.agg()
                except ValueError as err:
                    if "No objects to concatenate" not in str(err):
                        raise
                    result = self._aggregate_frame(func)
                else:
                    result = cast(DataFrame, result)
                    result.columns = self._obj_with_exclusions.columns.copy(deep=False)
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    agg = aggregate

    def _python_agg_general(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> DataFrame:
        f: Callable[[Any], Any] = lambda x: func(x, *args, **kwargs)
        if self.ngroups == 0:
            return self._python_apply_general(f, self._selected_obj, is_agg=True)
        obj: DataFrame = self._obj_with_exclusions
        if not len(obj.columns):
            return self._python_apply_general(f, self._selected_obj)
        output: Dict[int, ArrayLike] = {}
        for idx, (name, ser) in enumerate(obj.items()):
            result = self._grouper.agg_series(ser, f)
            output[idx] = result
        res: DataFrame = self.obj._constructor(output)
        res.columns = obj.columns.copy(deep=False)
        return self._wrap_aggregated_output(res)

    def _aggregate_frame(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> DataFrame:
        if self._grouper.nkeys != 1:
            raise AssertionError("Number of keys must be 1")
        obj: DataFrame = self._obj_with_exclusions
        result: Dict[Any, Union[NDFrame, np.ndarray]] = {}
        for name, grp_df in self._grouper.get_iterator(obj):
            fres = func(grp_df, *args, **kwargs)
            result[name] = fres
        result_index: Any = self._grouper.result_index
        out: DataFrame = self.obj._constructor(result, index=obj.columns, columns=result_index)
        out = out.T
        return out

    def _wrap_applied_output(
        self,
        data: DataFrame,
        values: List[Any],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ):
        if len(values) == 0:
            if is_transform:
                res_index: Index = data.index
            elif not self.group_keys:
                res_index = None
            else:
                res_index = self._grouper.result_index
            result: DataFrame = self.obj._constructor(index=res_index, columns=data.columns)
            result = result.astype(data.dtypes)
            return result
        first_not_none: Optional[Any] = next(com.not_none(*values), None)
        if first_not_none is None:
            result = self.obj._constructor(columns=data.columns)
            result = result.astype(data.dtypes)
            return result
        elif isinstance(first_not_none, DataFrame):
            return self._concat_objects(values, not_indexed_same=not_indexed_same, is_transform=is_transform)
        key_index: Optional[Index] = self._grouper.result_index if self.as_index else None
        if isinstance(first_not_none, (np.ndarray, Index)):
            if not is_hashable(self._selection):
                name = tuple(self._selection)  # type: ignore
            else:
                name = self._selection  # type: ignore
            return self.obj._constructor_sliced(values, index=key_index, name=name)
        elif not isinstance(first_not_none, Series):
            if self.as_index:
                return self.obj._constructor_sliced(values, index=key_index)
            else:
                result = self.obj._constructor(values, columns=[self._selection])
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
        kwargs = first_not_none._construct_axes_dict()
        backup = Series(**kwargs)
        values = [x if (x is not None) else backup for x in values]
        all_indexed_same = all_indexes_same(x.index for x in values)
        if not all_indexed_same:
            return self._concat_objects(values, not_indexed_same=True, is_transform=is_transform)
        stacked_values = np.vstack([np.asarray(v) for v in values])
        index = key_index
        columns = first_not_none.index.copy()
        if columns.name is None:
            names = {v.name for v in values}
            if len(names) == 1:
                columns.name = next(iter(names))
        if stacked_values.dtype == object:
            stacked_values = stacked_values.tolist()
        result = self.obj._constructor(stacked_values, index=index, columns=columns)
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
        def arr_func(bvalues: Any) -> ArrayLike:
            return self._grouper._cython_operation("transform", bvalues, how, 1, **kwargs)
        res_mgr = mgr.apply(arr_func)
        res_df: DataFrame = self.obj._constructor_from_mgr(res_mgr, axes=res_mgr.axes)
        return res_df

    def _transform_general(self, func: Callable[..., Any], engine: Any, engine_kwargs: Any, *args: Any, **kwargs: Any) -> DataFrame:
        if maybe_use_numba(engine):
            return self._transform_with_numba(func, *args, engine_kwargs=engine_kwargs, **kwargs)
        from pandas.core.reshape.concat import concat
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
            res = path(group)
            res = _wrap_transform_general_frame(self.obj, group, res)
            applied.append(res)
        concat_index = obj.columns
        concatenated: DataFrame = concat(applied, axis=0, verify_integrity=False, ignore_index=True)
        concatenated = concatenated.reindex(concat_index, axis=1)
        return self._set_result_index_ordered(concatenated)

    @Substitution(klass="DataFrame", example=__examples_dataframe_doc)
    @Appender(_transform_template)
    def transform(self, func: Union[str, Callable[..., Any]], *args: Any, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> DataFrame:
        return self._transform(func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs)

    def _define_paths(self, func: Union[str, Callable[..., Any]], *args: Any, **kwargs: Any) -> tuple[Callable[[DataFrame], Any], Callable[[DataFrame], Any]]:
        if isinstance(func, str):
            fast_path = lambda group: getattr(group, func)(*args, **kwargs)
            slow_path = lambda group: group.apply(lambda x: getattr(x, func)(*args, **kwargs), axis=0)
        else:
            fast_path = lambda group: func(group, *args, **kwargs)
            slow_path = lambda group: group.apply(lambda x: func(x, *args, **kwargs), axis=0)
        return fast_path, slow_path

    def _choose_path(self, fast_path: Callable[[DataFrame], Any], slow_path: Callable[[DataFrame], Any], group: DataFrame) -> tuple[Callable[[DataFrame], Any], Any]:
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

    def filter(self, func: Callable[..., Any], dropna: bool = True, *args: Any, **kwargs: Any) -> DataFrame:
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
            if is_bool(res) or (is_scalar(res) and isna(res)):
                if notna(res) and res:
                    indices.append(self._get_index(name))
            else:
                raise TypeError(f"filter function returned a {type(res).__name__}, but expected a scalar bool")
        return self._apply_filter(indices, dropna)

    def __getitem__(self, key: Any) -> Union[DataFrameGroupBy, SeriesGroupBy]:
        if isinstance(key, tuple) and len(key) > 1:
            raise ValueError("Cannot subset columns with a tuple with more than one element. Use a list instead.")
        return super().__getitem__(key)

    def _gotitem(self, key: Any, ndim: int, subset: Optional[Any] = None) -> Union[DataFrameGroupBy, SeriesGroupBy]:
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
        from pandas.core.reshape.concat import concat
        obj: DataFrame = self._obj_with_exclusions
        columns: Index = obj.columns
        sgbs = (SeriesGroupBy(
                    obj.iloc[:, i],
                    selection=colname,
                    grouper=self._grouper,
                    exclusions=self.exclusions,
                    observed=self.observed,
                ) for i, colname in enumerate(obj.columns))
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

    def idxmax(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame:
        return self._idxmax_idxmin("idxmax", numeric_only=numeric_only, skipna=skipna)

    def idxmin(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> DataFrame:
        return self._idxmax_idxmin("idxmin", numeric_only=numeric_only, skipna=skipna)

    boxplot = boxplot_frame_groupby

    def value_counts(
        self,
        subset: Optional[Sequence[Any]] = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> Union[DataFrame, Series]:
        return self._value_counts(subset, normalize, sort, ascending, dropna)

    def take(
        self,
        indices: TakeIndexer,
        **kwargs: Any,
    ) -> DataFrame:
        result: DataFrame = self._op_via_apply("take", indices=indices, **kwargs)
        return result

    def skew(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        def alt(obj: DataFrame) -> Any:
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
    @doc(DataFrame.plot.__doc__)
    def plot(self) -> GroupByPlot:
        result = GroupByPlot(self)
        return result

    @doc(DataFrame.corr.__doc__)
    def corr(
        self,
        method: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "pearson",
        min_periods: int = 1,
        numeric_only: bool = False,
    ) -> DataFrame:
        result: DataFrame = self._op_via_apply("corr", method=method, min_periods=min_periods, numeric_only=numeric_only)
        return result

    @doc(DataFrame.cov.__doc__)
    def cov(
        self,
        min_periods: Optional[int] = None,
        ddof: Optional[int] = 1,
        numeric_only: bool = False,
    ) -> DataFrame:
        result: DataFrame = self._op_via_apply("cov", min_periods=min_periods, ddof=ddof, numeric_only=numeric_only)
        return result

    def hist(
        self,
        column: Optional[IndexLabel] = None,
        by: Any = None,
        grid: bool = True,
        xlabelsize: Optional[int] = None,
        xrot: Optional[float] = None,
        ylabelsize: Any = None,
        yrot: Optional[float] = None,
        ax: Any = None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: Optional[tuple[float, float]] = None,
        layout: Optional[tuple[int, int]] = None,
        bins: Union[int, Sequence[int]] = 10,
        backend: Optional[str] = None,
        legend: bool = False,
        **kwargs: Any,
    ):
        result = self._op_via_apply(
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
        method: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "pearson",
        numeric_only: bool = False,
    ) -> DataFrame:
        warnings.warn("DataFrameGroupBy.corrwith is deprecated", FutureWarning, stacklevel=find_stack_level())
        result: DataFrame = self._op_via_apply("corrwith", other=other, drop=drop, method=method, numeric_only=numeric_only)
        return result

def _wrap_transform_general_frame(
    obj: DataFrame,
    group: DataFrame,
    res: Union[DataFrame, Series]
) -> DataFrame:
    from pandas import concat
    if isinstance(res, Series):
        if res.index.is_(obj.index):
            res_frame: DataFrame = concat([res] * len(group.columns), axis=1, ignore_index=True)
            res_frame.columns = group.columns
            res_frame.index = group.index
        else:
            res_frame = obj._constructor(np.tile(res.values, (len(group.index), 1)), columns=group.columns, index=group.index)
        assert isinstance(res_frame, DataFrame)
        return res_frame
    elif isinstance(res, DataFrame) and not res.index.is_(group.index):
        return res._align_frame(group)[0]
    else:
        return res