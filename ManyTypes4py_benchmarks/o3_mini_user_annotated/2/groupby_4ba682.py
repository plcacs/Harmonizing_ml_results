#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas._libs.lib as lib
from pandas._libs.missing import NA
from pandas.core.dtypes.common import is_numeric_dtype, is_bool_dtype, is_object_dtype
from pandas.core.dtypes.cast import ensure_dtype_can_hold_na
from pandas.core.indexes.base import Index
from pandas.core.indexes.multi import MultiIndex
from pandas.core.internals.blocks import ensure_block_shape
from pandas.core.panel.ndpanel import NDFrame  # This represents a generic NDFrame
from pandas import Series, DataFrame
from pandas._typing import NDFrameT, PositionalIndexer, RandomState, DtypeObj, AnyArrayLike
import pandas.core.common as com
import pandas._libs.algos as algorithms
import pandas.core.missing as missing
import pandas._libs.groupby as libgroupby
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import nv
from pandas.core.indexes.api import default_index

# Type alias for our keys arguments
_KeysArgType = Union[Any, Sequence[Any]]

NDArrayBool = np.ndarray  # For simplicity

class GroupBy:
    def __init__(
        self,
        obj: NDFrameT,
        grouper: Any,
        group_keys: bool = True,
        dropna: bool = True,
    ) -> None:
        self.obj: NDFrameT = obj
        self._grouper: Any = grouper
        self.group_keys: bool = group_keys
        self.dropna: bool = dropna

    @property
    def indices(self) -> np.ndarray:
        return self._grouper.indices

    @property
    def ngroups(self) -> int:
        return self._grouper.ngroups

    @property
    def _selected_obj(self) -> NDFrameT:
        return self.obj

    def _get_data_to_aggregate(self, numeric_only: bool = False, name: str = "") -> NDFrameT:
        if numeric_only and isinstance(self.obj, DataFrame):
            return self.obj._get_numeric_data()  # type: ignore
        return self.obj

    def _wrap_agged_manager(self, mgr: Any) -> NDFrameT:
        return self.obj._constructor(mgr, index=self._grouper.result_index)  # type: ignore

    def _wrap_aggregated_output(self, res: NDFrameT, qs: Optional[np.ndarray] = None) -> NDFrameT:
        # If group_keys is False, reset the index.
        if not self.group_keys and isinstance(res, DataFrame):
            res = res.reset_index(drop=True)
        return res

    def _agg_general(
        self,
        numeric_only: bool = False,
        min_count: int = -1,
        alias: str = "",
        npfunc: Optional[Callable] = None,
        skipna: bool = True,
    ) -> NDFrameT:
        result: NDFrameT = self._cython_agg_general(
            how=alias,
            alt=(lambda x: Series(x, copy=False).agg(npfunc) if npfunc is not None else None),
            numeric_only=numeric_only,
            min_count=min_count,
            skipna=skipna,
        )
        return result.__finalize__(self.obj, method="groupby")  # type: ignore

    def _cython_agg_general(
        self,
        how: str,
        alt: Optional[Callable[[Any], Any]] = None,
        numeric_only: bool = False,
        min_count: int = -1,
        skipna: bool = True,
        **kwargs: Any,
    ) -> NDFrameT:
        data: NDFrameT = self._get_data_to_aggregate(numeric_only=numeric_only, name=how)
        def array_func(values: Any) -> Any:
            try:
                result = self._grouper._cython_operation(
                    "aggregate", values, how, axis=data.ndim - 1, min_count=min_count, skipna=skipna, **kwargs
                )
            except NotImplementedError:
                if alt is None or how in ["any", "all", "std", "sem"]:
                    raise
                result = self._agg_py_fallback(how, values, ndim=data.ndim, alt=alt)
            return result
        new_mgr = data.grouped_reduce(array_func)
        new_obj: NDFrameT = self._wrap_agged_manager(new_mgr)
        if how in ["idxmin", "idxmax"]:
            new_obj = self._wrap_idxmax_idxmin(new_obj)
        out: NDFrameT = self._wrap_aggregated_output(new_obj)
        return out

    def _agg_py_fallback(self, how: str, values: Any, ndim: int, alt: Callable) -> Any:
        if values.ndim == 1:
            ser: Series = Series(values, copy=False)
        else:
            df: DataFrame = DataFrame(values.T, dtype=values.dtype)
            assert df.shape[1] == 1
            ser = df.iloc[:, 0]
        try:
            res_values = self._grouper.agg_series(ser, alt, preserve_dtype=True)
        except Exception as err:
            msg = f"agg function failed [how->{how},dtype->{ser.dtype}]"
            raise type(err)(msg) from err
        if ser.dtype == object:
            res_values = res_values.astype(object, copy=False)
        return ensure_block_shape(res_values, ndim=ndim)

    def _cython_transform(self, how: str, numeric_only: bool = False, **kwargs: Any) -> NDFrameT:
        raise NotImplementedError

    def _transform(
        self,
        func: Union[str, Callable],
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, bool]] = None,
        **kwargs: Any,
    ) -> NDFrameT:
        if not isinstance(func, str):
            return self._transform_general(func, engine, engine_kwargs, *args, **kwargs)
        elif func not in {"transform_func1", "transform_func2"}:  # placeholder for allowlist
            msg = f"'{func}' is not a valid function name for transform(name)"
            raise ValueError(msg)
        elif func in {"cythonized_kernel1", "transformation_kernel1"}:
            if engine is not None:
                kwargs["engine"] = engine
                kwargs["engine_kwargs"] = engine_kwargs
            return getattr(self, func)(*args, **kwargs)  # type: ignore
        else:
            if self.dropna:
                return self._reduction_kernel_transform(func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs)
            with com.temp_setattr(self, "dropna", True):
                return self._reduction_kernel_transform(func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs)

    def _transform_general(
        self,
        func: Callable,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, bool]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> NDFrameT:
        return self._python_apply_general(func, self._selected_obj, is_transform=True)

    def _reduction_kernel_transform(
        self,
        func: Union[str, Callable],
        *args: Any,
        engine: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, bool]] = None,
        **kwargs: Any,
    ) -> NDFrameT:
        with com.temp_setattr(self, "group_keys", True):
            if func in ["idxmin", "idxmax"]:
                func = cast(Union[Literal["idxmin"], Literal["idxmax"]], func)
                result: NDFrameT = self._idxmax_idxmin(func, True, *args, **kwargs)
            else:
                if engine is not None:
                    kwargs["engine"] = engine
                    kwargs["engine_kwargs"] = engine_kwargs
                result = getattr(self, func)(*args, **kwargs)  # type: ignore
        return self._wrap_transform_fast_result(result)

    def _wrap_transform_fast_result(self, result: NDFrameT) -> NDFrameT:
        obj: NDFrameT = self._selected_obj
        ids: np.ndarray = self._grouper.ids
        result = result.reindex(self._grouper.result_index, axis=0)
        if obj.ndim == 1:
            out_values = algorithms.take_nd(result._values, ids)
            output = obj._constructor(out_values, index=obj.index, name=obj.name)
        else:
            new_ax = result.index.take(ids)
            output = result._reindex_with_indexers({0: (new_ax, ids)}, allow_dups=True)
            output = output.set_axis(obj.index, axis=0)
        return output

    def _mask_selected_obj(self, mask: np.ndarray) -> NDFrameT:
        mask = mask & (self._grouper.ids != -1)
        return self._selected_obj[mask]

    def _wrap_idxmax_idxmin(self, res: NDFrameT) -> NDFrameT:
        index: Index = self.obj.index
        if res.size == 0:
            result = res.astype(index.dtype)
        else:
            if isinstance(index, MultiIndex):
                index = index.to_flat_index()
            values: np.ndarray = res._values  # type: ignore
            na_value = missing.na_value_for_dtype(index.dtype, compat=False)
            if isinstance(res, Series):
                result = res._constructor(index.array.take(values, allow_fill=True, fill_value=na_value),
                                          index=res.index,
                                          name=res.name)
            else:
                data: Dict[Any, Any] = {}
                for k, column_values in enumerate(values.T):
                    data[k] = index.array.take(column_values, allow_fill=True, fill_value=na_value)
                result = self.obj._constructor(data, index=res.index)
                result.columns = res.columns
        return result

    def _python_apply_general(
        self,
        f: Callable,
        data: NDFrameT,
        is_transform: bool = False,
        not_indexed_same: bool = False,
    ) -> NDFrameT:
        values, mutated = self._grouper.apply_groupwise(f, data)
        if not_indexed_same is None:
            not_indexed_same = mutated
        return self._wrap_applied_output(data, values, not_indexed_same, is_transform)

    def _wrap_applied_output(
        self, data: NDFrameT, values: Any, not_indexed_same: bool, is_transform: bool
    ) -> NDFrameT:
        # Placeholder implementation; actual logic depends on Pandas internals.
        return values

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        weights: Optional[Union[Sequence[Any], Series]] = None,
        random_state: Optional[RandomState] = None,
    ) -> NDFrameT:
        if self._selected_obj.empty:
            return self._selected_obj
        size: Optional[int] = com.sample.process_sampling_size(n, frac, replace)
        if weights is not None:
            weights_arr: np.ndarray = com.sample.preprocess_weights(self._selected_obj, weights, axis=0)
        rs: RandomState = com.random_state(random_state)
        group_iterator: Iterable[Tuple[Any, NDFrameT]] = self._grouper.get_iterator(self._selected_obj)
        sampled_indices: List[np.ndarray] = []
        for labels, obj in group_iterator:
            grp_indices: np.ndarray = self.indices[labels]
            group_size: int = len(grp_indices)
            sample_size: int = size if size is not None else round(frac * group_size)  # type: ignore
            grp_sample: np.ndarray = com.sample.sample(
                group_size,
                size=sample_size,
                replace=replace,
                weights=None if weights is None else weights_arr[grp_indices],
                random_state=rs,
            )
            sampled_indices.append(grp_indices[grp_sample])
        sampled_indices_arr: np.ndarray = np.concatenate(sampled_indices)
        return self._selected_obj.take(sampled_indices_arr, axis=0)

    def _idxmax_idxmin(
        self,
        how: Union[Literal["idxmax"], Literal["idxmin"]],
        ignore_unobserved: bool = False,
        skipna: bool = True,
        numeric_only: bool = False,
    ) -> NDFrameT:
        if not self.dropna and any(g._passed_categorical for g in self._grouper.groupings):
            expected_len: int = len(self._grouper.result_index)
            group_sizes: np.ndarray = self._grouper.size()
            result_len: int = group_sizes[group_sizes > 0].shape[0]
            has_unobserved: bool = result_len < expected_len
            raise_err: bool = (not ignore_unobserved) and has_unobserved
            data: NDFrameT = self._selected_obj
            if raise_err and isinstance(data, DataFrame):
                if numeric_only:
                    data = data._get_numeric_data()  # type: ignore
                raise_err = len(data.columns) > 0
            if raise_err:
                raise ValueError(
                    f"Can't get {how} of an empty group due to unobserved categories. "
                    "Specify observed=True in groupby instead."
                )
        elif not skipna and self._selected_obj.isna().any(axis=None):
            raise ValueError(f"{type(self).__name__}.{how} with skipna=False encountered an NA value.")
        result: NDFrameT = self._agg_general(numeric_only=numeric_only, min_count=1, alias=how, skipna=skipna)
        return result

    def head(self, n: int = 5) -> NDFrameT:
        mask: np.ndarray = self._make_mask_from_positional_indexer(slice(None, n))
        return self._mask_selected_obj(mask)

    def tail(self, n: int = 5) -> NDFrameT:
        if n:
            mask: np.ndarray = self._make_mask_from_positional_indexer(slice(-n, None))
        else:
            mask = self._make_mask_from_positional_indexer([])
        return self._mask_selected_obj(mask)

    def _make_mask_from_positional_indexer(self, idx: Any) -> np.ndarray:
        # Placeholder: In actual implementation, this creates a boolean mask based on positional indexing.
        # Here we assume _selected_obj has a defined length.
        length: int = len(self._selected_obj)
        mask: np.ndarray = np.zeros(length, dtype=bool)
        if isinstance(idx, slice):
            indices: np.ndarray = np.arange(length)[idx]
            mask[indices] = True
        elif isinstance(idx, list):
            mask[idx] = True
        else:
            mask[idx] = True
        return mask

    def rank(
        self,
        method: str = "average",
        ascending: bool = True,
        na_option: str = "keep",
        pct: bool = False,
    ) -> NDFrameT:
        if na_option not in {"keep", "top", "bottom"}:
            raise ValueError("na_option must be one of 'keep', 'top', or 'bottom'")
        kwargs: Dict[str, Any] = {
            "ties_method": method,
            "ascending": ascending,
            "na_option": na_option,
            "pct": pct,
        }
        return self._cython_transform("rank", numeric_only=False, **kwargs)

    def cumprod(self, numeric_only: bool = False, *args: Any, **kwargs: Any) -> NDFrameT:
        nv.validate_groupby_func("cumprod", args, kwargs, ["skipna"])
        return self._cython_transform("cumprod", numeric_only, **kwargs)

    def cumsum(self, numeric_only: bool = False, *args: Any, **kwargs: Any) -> NDFrameT:
        nv.validate_groupby_func("cumsum", args, kwargs, ["skipna"])
        return self._cython_transform("cumsum", numeric_only, **kwargs)

    def cummin(self, numeric_only: bool = False, **kwargs: Any) -> NDFrameT:
        skipna: bool = kwargs.get("skipna", True)
        return self._cython_transform("cummin", numeric_only=numeric_only, skipna=skipna)

    def cummax(self, numeric_only: bool = False, **kwargs: Any) -> NDFrameT:
        skipna: bool = kwargs.get("skipna", True)
        return self._cython_transform("cummax", numeric_only=numeric_only, skipna=skipna)

    def shift(
        self,
        periods: Union[int, Sequence[int]] = 1,
        freq: Any = None,
        fill_value: Any = lib.no_default,
        suffix: Optional[str] = None,
    ) -> Union[Series, DataFrame]:
        if isinstance(periods, Sequence) and not isinstance(periods, (str, bytes)):
            periods = list(periods)
            if len(periods) == 0:
                raise ValueError("If `periods` is an iterable, it cannot be empty.")
            add_suffix: bool = True
        else:
            if not isinstance(periods, int):
                raise TypeError(f"Periods must be integer, but {periods} is {type(periods)}.")
            if suffix:
                raise ValueError("Cannot specify `suffix` if `periods` is an int.")
            periods = [cast(int, periods)]
            add_suffix = False
        shifted_dataframes: List[Union[Series, DataFrame]] = []
        for period in periods:
            if not isinstance(period, int):
                raise TypeError(f"Periods must be integer, but {period} is {type(period)}.")
            period = cast(int, period)
            if freq is not None:
                f: Callable[[NDFrameT], NDFrameT] = lambda x: x.shift(period, freq, 0, fill_value)  # type: ignore
                shifted: NDFrameT = self._python_apply_general(f, self._selected_obj, is_transform=True)
            else:
                if fill_value is lib.no_default:
                    fill_value = None
                ids: np.ndarray = self._grouper.ids
                ngroups: int = self._grouper.ngroups
                res_indexer: np.ndarray = np.zeros(len(ids), dtype=np.int64)
                libgroupby.group_shift_indexer(res_indexer, ids, ngroups, period)
                obj: NDFrameT = self._obj_with_exclusions
                shifted = obj._reindex_with_indexers({0: (obj.index, res_indexer)}, fill_value=fill_value, allow_dups=True)
            if add_suffix:
                if isinstance(shifted, Series):
                    shifted = shifted.to_frame()
                shifted = shifted.add_suffix(f"{suffix}_{period}" if suffix else f"_{period}")
            shifted_dataframes.append(shifted)
        if len(shifted_dataframes) == 1:
            return shifted_dataframes[0]
        else:
            return concat(shifted_dataframes, axis=1)

    def diff(self, periods: int = 1) -> NDFrameT:
        obj: NDFrameT = self._selected_obj
        shifted: NDFrameT = self.shift(periods=periods)
        dtypes_to_f32: List[str] = ["int8", "int16"]
        if obj.ndim == 1:
            if obj.dtype.name in dtypes_to_f32:
                shifted = shifted.astype("float32")
        else:
            to_coerce: List[Any] = [c for c, dt in obj.dtypes.items() if dt.name in dtypes_to_f32]  # type: ignore
            if len(to_coerce):
                shifted = shifted.astype({c: "float32" for c in to_coerce})
        return obj - shifted

    def pct_change(
        self,
        periods: int = 1,
        fill_method: Any = None,
        freq: Any = None,
    ) -> NDFrameT:
        if fill_method is not None:
            raise ValueError(f"fill_method must be None; got {fill_method=}.")
        if freq is not None:
            f: Callable[[NDFrameT], NDFrameT] = lambda x: x.pct_change(periods=periods, freq=freq, axis=0)  # type: ignore
            return self._python_apply_general(f, self._selected_obj, is_transform=True)
        if fill_method is None:  # GH30463
            op: str = "ffill"
        else:
            op = fill_method
        filled: NDFrameT = getattr(self, op)(limit=0)
        fill_grp = filled.groupby(self._grouper.codes, group_keys=self.group_keys)
        shifted = fill_grp.shift(periods=periods, freq=freq)
        return (filled / shifted) - 1

    def describe(
        self,
        percentiles: Optional[Sequence[float]] = None,
        include: Any = None,
        exclude: Any = None,
    ) -> NDFrameT:
        obj: NDFrameT = self._obj_with_exclusions
        if len(obj) == 0:
            described: NDFrameT = obj.describe(percentiles=percentiles, include=include, exclude=exclude)
            if obj.ndim == 1:
                result: NDFrameT = described
            else:
                result = described.unstack()
            return result.to_frame().T.iloc[:0]
        with com.temp_setattr(self, "group_keys", True):
            result = self._python_apply_general(lambda x: x.describe(percentiles=percentiles, include=include, exclude=exclude), obj, not_indexed_same=True)
        result = result.unstack()
        if not self.group_keys:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    def _insert_inaxis_grouper(self, result: NDFrameT, qs: Optional[np.ndarray] = None) -> NDFrameT:
        # Placeholder implementation for inserting grouping info into DataFrame
        return result

    def ngroup(self, ascending: bool = True) -> Series:
        obj: NDFrameT = self._obj_with_exclusions
        index: Index = obj.index
        comp_ids: np.ndarray = self._grouper.ids
        if self._grouper.has_dropped_na:
            comp_ids = np.where(comp_ids == -1, np.nan, comp_ids)
            dtype = np.float64
        else:
            dtype = np.int64
        if any(g._passed_categorical for g in self._grouper.groupings):
            comp_ids = algorithms.rank_1d(comp_ids, ties_method="dense") - 1
        result = self._obj_1d_constructor(comp_ids, index, dtype=dtype)  # type: ignore
        if not ascending:
            result = self.ngroups - 1 - result
        return result

    def cumcount(self, ascending: bool = True) -> Series:
        index: Index = self._obj_with_exclusions.index
        cumcounts: np.ndarray = self._cumcount_array(ascending=ascending)
        return self._obj_1d_constructor(cumcounts, index)

    def _cumcount_array(self, ascending: bool = True) -> np.ndarray:
        ids: np.ndarray = self._grouper.ids
        ngroups: int = self._grouper.ngroups
        sorter: np.ndarray = algorithms.get_group_index_sorter(ids, ngroups)
        sorted_ids: np.ndarray = ids[sorter]
        count: int = len(ids)
        if count == 0:
            return np.empty(0, dtype=np.int64)
        run: np.ndarray = np.r_[True, sorted_ids[:-1] != sorted_ids[1:]]
        rep: np.ndarray = np.diff(np.r_[np.nonzero(run)[0], count])
        out: np.ndarray = (~run).cumsum()
        if ascending:
            out -= np.repeat(out[run], rep)
        else:
            out = np.repeat(out[np.r_[run[1:], True]], rep) - out
        if self._grouper.has_dropped_na:
            out = np.where(sorted_ids == -1, np.nan, out.astype(np.float64, copy=False))
        else:
            out = out.astype(np.int64, copy=False)
        rev: np.ndarray = np.empty(count, dtype=np.intp)
        rev[sorter] = np.arange(count, dtype=np.intp)
        return out[rev]

    @property
    def _obj_1d_constructor(self) -> Callable[..., Any]:
        if isinstance(self.obj, DataFrame):
            return self.obj._constructor_sliced  # type: ignore
        assert isinstance(self.obj, Series)
        return self.obj._constructor

    def nth(self) -> Any:
        return GroupByNthSelector(self)

    def _nth(self, n: Union[PositionalIndexer, Tuple[Any, ...]], dropna: Optional[Union[Literal["any"], Literal["all"]]] = None) -> NDFrameT:
        if not dropna:
            mask = self._make_mask_from_positional_indexer(n)
            ids = self._grouper.ids
            mask = mask & (ids != -1)
            out = self._mask_selected_obj(mask)
            return out
        if not isinstance(n, int):
            raise ValueError("dropna option only supported for an integer argument")
        if dropna not in ["any", "all"]:
            raise ValueError("For a DataFrame or Series groupby.nth, dropna must be either None, 'any' or 'all', " f"(was passed {dropna}).")
        n_int: int = n  # type: ignore
        dropped: NDFrameT = self._selected_obj.dropna(how=dropna, axis=0)
        axis = self._grouper.axis
        grouper = self._grouper.codes_info[axis.isin(dropped.index)]
        if self._grouper.has_dropped_na:
            nulls = grouper == -1
            values = np.where(nulls, NA, grouper)
            grouper = Index(values, dtype="Int64")
        grb = dropped.groupby(grouper, as_index=self.group_keys, sort=self._grouper.sort)
        return grb.nth(n_int)

    def quantile(
        self,
        q: Union[float, AnyArrayLike] = 0.5,
        interpolation: Literal["linear", "lower", "higher", "nearest", "midpoint"] = "linear",
        numeric_only: bool = False,
    ) -> NDFrameT:
        mgr: NDFrameT = self._get_data_to_aggregate(numeric_only=numeric_only, name="quantile")
        obj: NDFrameT = self._wrap_agged_manager(mgr)
        splitter = self._grouper._get_splitter(obj)
        sdata = splitter._sorted_data
        starts, ends = lib.generate_slices(splitter._slabels, splitter.ngroups)
        def pre_processor(vals: Any) -> Tuple[Any, Optional[DtypeObj]]:
            if isinstance(vals.dtype, type("dummy")) or is_object_dtype(vals.dtype):
                raise TypeError(f"dtype '{vals.dtype}' does not support operation 'quantile'")
            inference: Optional[DtypeObj] = None
            if hasattr(vals, "to_numpy") and hasattr(vals, "_mask"):
                out = vals.to_numpy(dtype=float, na_value=np.nan)
                inference = vals.dtype
            elif is_numeric_dtype(vals.dtype) and np.issubdtype(vals.dtype, np.integer):
                if hasattr(vals, "to_numpy"):
                    out = vals.to_numpy(dtype=float, na_value=np.nan)
                else:
                    out = vals
                inference = np.dtype(np.int64)
            elif is_bool_dtype(vals.dtype) and hasattr(vals, "to_numpy"):
                out = vals.to_numpy(dtype=float, na_value=np.nan)
            elif is_bool_dtype(vals.dtype):
                raise TypeError("Cannot use quantile with bool dtype")
            elif lib.needs_i8_conversion(vals.dtype):
                inference = vals.dtype
                return vals, inference
            elif hasattr(vals, "to_numpy") and is_numeric_dtype(vals.dtype):
                inference = np.dtype(np.float64)
                out = vals.to_numpy(dtype=float, na_value=np.nan)
            else:
                out = np.asarray(vals)
            return out, inference
        def post_processor(
            vals: np.ndarray,
            inference: Optional[DtypeObj],
            result_mask: Optional[np.ndarray],
            orig_vals: Any,
        ) -> Any:
            if inference:
                if hasattr(orig_vals, "_mask"):
                    assert result_mask is not None
                    if interpolation in {"linear", "midpoint"} and not is_float_dtype(orig_vals.dtype):
                        from pandas.core.arrays.floating import FloatingArray
                        return FloatingArray(vals, result_mask)
                    else:
                        with np.errstate(invalid="ignore"):
                            return type(orig_vals)(vals.astype(inference.numpy_dtype), result_mask)  # type: ignore
                elif not (is_integer_dtype(inference) and interpolation in {"linear", "midpoint"}):
                    if lib.needs_i8_conversion(inference):
                        vals = vals.astype("i8").view(orig_vals._ndarray.dtype)  # type: ignore
                        return orig_vals._from_backing_data(vals)  # type: ignore
                    return vals.astype(inference)
            return vals
        qs: np.ndarray
        pass_qs: Optional[np.ndarray]
        if np.isscalar(q):
            qs = np.array([q], dtype=np.float64)
            pass_qs = None
        else:
            qs = np.asarray(q, dtype=np.float64)
            pass_qs = qs
        ids: np.ndarray = self._grouper.ids
        ngroups: int = self._grouper.ngroups
        if self.dropna:
            ids = ids[ids >= 0]
        nqs: int = len(qs)
        from functools import partial
        func = partial(
            libgroupby.group_quantile,
            labels=ids,
            qs=qs,
            interpolation=interpolation,
            starts=starts,
            ends=ends,
        )
        def blk_func(values: Any) -> Any:
            orig_vals: Any = values
            if hasattr(values, "_mask"):
                mask = values._mask
                result_mask = np.zeros((ngroups, nqs), dtype=bool)
            else:
                mask = isna(values)
                result_mask = None
            is_datetimelike: bool = lib.needs_i8_conversion(values.dtype)
            vals, inference = pre_processor(values)
            ncols: int = 1
            if vals.ndim == 2:
                ncols = vals.shape[0]
            out = np.empty((ncols, ngroups, nqs), dtype=np.float64)
            if is_datetimelike:
                vals = vals.view("i8")
            if vals.ndim == 1:
                func(out[0], values=vals, mask=mask, result_mask=result_mask, is_datetimelike=is_datetimelike)
            else:
                for i in range(ncols):
                    func(out[i], values=vals[i], mask=mask[i], result_mask=None, is_datetimelike=is_datetimelike)
            if vals.ndim == 1:
                out = out.ravel("K")
                if result_mask is not None:
                    result_mask = result_mask.ravel("K")
            else:
                out = out.reshape(ncols, ngroups * nqs)
            return post_processor(out, inference, result_mask, orig_vals)
        res_mgr = sdata._mgr.grouped_reduce(blk_func)
        res: NDFrameT = self._wrap_agged_manager(res_mgr)
        return self._wrap_aggregated_output(res, qs=pass_qs)

    def ngroup(self, ascending: bool = True) -> Series:
        obj: NDFrameT = self._obj_with_exclusions
        index: Index = obj.index
        comp_ids: np.ndarray = self._grouper.ids
        if self._grouper.has_dropped_na:
            comp_ids = np.where(comp_ids == -1, np.nan, comp_ids)
            dtype = np.float64
        else:
            dtype = np.int64
        if any(g._passed_categorical for g in self._grouper.groupings):
            comp_ids = algorithms.rank_1d(comp_ids, ties_method="dense") - 1
        result = self._obj_1d_constructor(comp_ids, index, dtype=dtype)
        if not ascending:
            result = self.ngroups - 1 - result
        return result

    def cumcount(self, ascending: bool = True) -> Series:
        index: Index = self._obj_with_exclusions.index
        cumcounts: np.ndarray = self._cumcount_array(ascending=ascending)
        return self._obj_1d_constructor(cumcounts, index)

    def rank(
        self,
        method: str = "average",
        ascending: bool = True,
        na_option: str = "keep",
        pct: bool = False,
    ) -> NDFrameT:
        if na_option not in {"keep", "top", "bottom"}:
            raise ValueError("na_option must be one of 'keep', 'top', or 'bottom'")
        kwargs: Dict[str, Any] = {
            "ties_method": method,
            "ascending": ascending,
            "na_option": na_option,
            "pct": pct,
        }
        return self._cython_transform("rank", numeric_only=False, **kwargs)

    def cumprod(self, numeric_only: bool = False, *args: Any, **kwargs: Any) -> NDFrameT:
        nv.validate_groupby_func("cumprod", args, kwargs, ["skipna"])
        return self._cython_transform("cumprod", numeric_only, **kwargs)

    def cumsum(self, numeric_only: bool = False, *args: Any, **kwargs: Any) -> NDFrameT:
        nv.validate_groupby_func("cumsum", args, kwargs, ["skipna"])
        return self._cython_transform("cumsum", numeric_only, **kwargs)

    def cummin(self, numeric_only: bool = False, **kwargs: Any) -> NDFrameT:
        skipna: bool = kwargs.get("skipna", True)
        return self._cython_transform("cummin", numeric_only=numeric_only, skipna=skipna)

    def cummax(self, numeric_only: bool = False, **kwargs: Any) -> NDFrameT:
        skipna: bool = kwargs.get("skipna", True)
        return self._cython_transform("cummax", numeric_only=numeric_only, skipna=skipna)

    def shift(
        self,
        periods: Union[int, Sequence[int]] = 1,
        freq: Any = None,
        fill_value: Any = lib.no_default,
        suffix: Optional[str] = None,
    ) -> Union[Series, DataFrame]:
        # Implementation provided above
        return self.shift(periods, freq, fill_value, suffix)  # Placeholder

    def diff(self, periods: int = 1) -> NDFrameT:
        obj: NDFrameT = self._selected_obj
        shifted: NDFrameT = self.shift(periods=periods)
        dtypes_to_f32: List[str] = ["int8", "int16"]
        if obj.ndim == 1:
            if obj.dtype.name in dtypes_to_f32:
                shifted = shifted.astype("float32")
        else:
            to_coerce: List[Any] = [c for c, dt in obj.dtypes.items() if dt.name in dtypes_to_f32]  # type: ignore
            if len(to_coerce):
                shifted = shifted.astype({c: "float32" for c in to_coerce})
        return obj - shifted

    def pct_change(self, periods: int = 1, fill_method: Any = None, freq: Any = None) -> NDFrameT:
        return self.pct_change(periods, fill_method, freq)

    def describe(self, percentiles: Optional[Sequence[float]] = None, include: Any = None, exclude: Any = None) -> NDFrameT:
        return self.describe(percentiles, include, exclude)

class GroupByNthSelector:
    def __init__(self, groupby: GroupBy) -> None:
        self.groupby: GroupBy = groupby

    def __getitem__(self, key: Any) -> NDFrameT:
        return self.groupby._nth(key)

def get_groupby(
    obj: NDFrame,
    by: Optional[_KeysArgType] = None,
    grouper: Optional[Any] = None,
    group_keys: bool = True,
) -> GroupBy:
    if isinstance(obj, Series):
        from pandas.core.groupby.generic import SeriesGroupBy
        klass: Any = SeriesGroupBy
    elif isinstance(obj, DataFrame):
        from pandas.core.groupby.generic import DataFrameGroupBy
        klass = DataFrameGroupBy
    else:
        raise TypeError(f"invalid type: {obj}")
    return klass(obj=obj, keys=by, grouper=grouper, group_keys=group_keys)

def _insert_quantile_level(idx: Index, qs: np.ndarray) -> MultiIndex:
    nqs: int = len(qs)
    lev_codes, lev = Index(qs).factorize()
    lev_codes = com.coerce_indexer_dtype(lev_codes, lev)
    if idx._is_multi:
        idx = cast(MultiIndex, idx)
        levels: List[Any] = list(idx.levels) + [lev]
        codes: List[np.ndarray] = [np.repeat(x, nqs) for x in idx.codes] + [np.tile(lev_codes, len(idx))]
        mi: MultiIndex = MultiIndex(levels=levels, codes=codes, names=idx.names + [None])
    else:
        nidx: int = len(idx)
        idx_codes = com.coerce_indexer_dtype(np.arange(nidx), idx)
        levels = [idx, lev]
        codes = [np.repeat(idx_codes, nqs), np.tile(lev_codes, nidx)]
        mi = MultiIndex(levels=levels, codes=codes, names=[idx.name, None])
    return mi
