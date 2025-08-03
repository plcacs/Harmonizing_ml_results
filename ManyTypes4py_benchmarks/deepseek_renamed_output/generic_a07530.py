from __future__ import annotations
from collections import abc
from collections.abc import Callable, Hashable, Sequence
from functools import partial
from textwrap import dedent
from typing import (
    TYPE_CHECKING, Any, Dict, List, Literal, NamedTuple, Optional, Tuple, TypeVar, Union, cast
)
import warnings
import numpy as np
from pandas._libs import Interval
from pandas._libs.hashtable import duplicated
from pandas.errors import SpecificationError
from pandas.util._decorators import Appender, Substitution, doc, set_module
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
    ensure_int64, is_bool, is_dict_like, is_integer_dtype, is_list_like, 
    is_numeric_dtype, is_scalar
)
from pandas.core.dtypes.dtypes import CategoricalDtype, IntervalDtype
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import isna, notna
from pandas.core import algorithms
from pandas.core.apply import (
    GroupByApply, maybe_mangle_lambdas, reconstruct_func, validate_func_kwargs
)
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import base
from pandas.core.groupby.groupby import GroupBy, GroupByPlot, _transform_template
from pandas.core.indexes.api import Index, MultiIndex, all_indexes_same, default_index
from pandas.core.series import Series
from pandas.core.sorting import get_group_index
from pandas.core.util.numba_ import maybe_use_numba
from pandas.plotting import boxplot_frame_groupby

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike, BlockManager, CorrelationMethod, IndexLabel, Manager, 
        SingleBlockManager, TakeIndexer
    )
    from pandas import Categorical
    from pandas.core.generic import NDFrame

AggScalar = Union[str, Callable[..., Any]]
ScalarResult = TypeVar('ScalarResult')

@set_module('pandas')
class NamedAgg(NamedTuple):
    column: Hashable
    aggfunc: Union[str, Callable[..., Any]]

@set_module('pandas.api.typing')
class SeriesGroupBy(GroupBy[Series]):
    def func_i1f1jtgv(self, mgr: Manager) -> Series:
        out = self.obj._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self.obj.name
        return out

    def func_ptnbe4ip(self, *, numeric_only: bool = False, name: Optional[str] = None) -> SingleBlockManager:
        ser = self._obj_with_exclusions
        single = ser._mgr
        if numeric_only and not is_numeric_dtype(ser.dtype):
            kwd_name = 'numeric_only'
            raise TypeError(
                f'Cannot use {kwd_name}=True with {type(self).__name__}.{name} and non-numeric dtypes.'
            )
        return single

    def func_qfuul1kv(self, func: Callable, *args: Any, **kwargs: Any) -> Union[Series, DataFrame]:
        return super().apply(func, *args, **kwargs)

    def func_mcq0s4ku(
        self, 
        func: Optional[Union[str, Callable, List[Union[str, Callable]]] = None, 
        *args: Any, 
        engine: Optional[str] = None, 
        engine_kwargs: Optional[Dict[str, bool]] = None, 
        **kwargs: Any
    ) -> Union[Series, DataFrame]:
        relabeling = func is None
        columns = None
        if relabeling:
            columns, func = validate_func_kwargs(kwargs)
            kwargs = {}
        if isinstance(func, str):
            if maybe_use_numba(engine) and engine is not None:
                kwargs['engine'] = engine
            if engine_kwargs is not None:
                kwargs['engine_kwargs'] = engine_kwargs
            return getattr(self, func)(*args, **kwargs)
        elif isinstance(func, abc.Iterable):
            func = maybe_mangle_lambdas(func)
            kwargs['engine'] = engine
            kwargs['engine_kwargs'] = engine_kwargs
            ret = self._aggregate_multiple_funcs(func, *args, **kwargs)
            if relabeling:
                assert columns is not None
                ret.columns = columns
            if not self.as_index:
                ret = ret.reset_index()
            return ret
        else:
            if maybe_use_numba(engine):
                return self._aggregate_with_numba(
                    func, *args, engine_kwargs=engine_kwargs, **kwargs
                )
            if self.ngroups == 0:
                obj = self._obj_with_exclusions
                return self.obj._constructor(
                    [], name=self.obj.name, index=self._grouper.result_index, dtype=obj.dtype
                )
            return self._python_agg_general(func, *args, **kwargs)
    
    agg = aggregate

    def func_s61x43hm(self, func: Callable, *args: Any, **kwargs: Any) -> Series:
        f = lambda x: func(x, *args, **kwargs)
        obj = self._obj_with_exclusions
        result = self._grouper.agg_series(obj, f)
        res = obj._constructor(result, name=obj.name)
        return self._wrap_aggregated_output(res)

    def func_6w400jyf(self, arg: Any, *args: Any, **kwargs: Any) -> Union[Series, DataFrame]:
        if isinstance(arg, dict):
            raise SpecificationError('nested renamer is not supported')
        if any(isinstance(x, (tuple, list)) for x in arg):
            arg = ((x, x) if not isinstance(x, (tuple, list)) else x for x in arg)
        else:
            columns = (com.get_callable_name(f) or f for f in arg)
            arg = zip(columns, arg)
        results = {}
        with com.temp_setattr(self, 'as_index', True):
            for idx, (name, func) in enumerate(arg):
                key = base.OutputKey(label=name, position=idx)
                results[key] = self.aggregate(func, *args, **kwargs)
        if any(isinstance(x, DataFrame) for x in results.values()):
            from pandas import concat
            res_df = concat(results.values(), axis=1, keys=[key.label for key in results])
            return res_df
        indexed_output = {key.position: val for key, val in results.items()}
        output = self.obj._constructor_expanddim(indexed_output, index=None)
        output.columns = Index(key.label for key in results)
        return output

    def func_u7w42znx(
        self, 
        data: Series, 
        values: List[Any], 
        not_indexed_same: bool = False,
        is_transform: bool = False
    ) -> Union[Series, DataFrame]:
        if len(values) == 0:
            if is_transform:
                res_index = data.index
            elif not self.group_keys:
                res_index = None
            else:
                res_index = self._grouper.result_index
            return self.obj._constructor([], name=self.obj.name, index=res_index, dtype=data.dtype)
        assert values is not None
        if isinstance(values[0], dict):
            index = self._grouper.result_index
            res_df = self.obj._constructor_expanddim(values, index=index)
            res_ser = res_df.stack()
            res_ser.name = self.obj.name
            return res_ser
        elif isinstance(values[0], (Series, DataFrame)):
            result = self._concat_objects(
                values, not_indexed_same=not_indexed_same, is_transform=is_transform
            )
            if isinstance(result, Series):
                result.name = self.obj.name
            if not self.as_index and not_indexed_same:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result
        else:
            result = self.obj._constructor(
                data=values, index=self._grouper.result_index, name=self.obj.name
            )
            if not self.as_index:
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result

    def func_wfojs7y1(
        self, 
        func: Callable, 
        *args: Any, 
        engine: Optional[str] = None, 
        engine_kwargs: Optional[Dict[str, bool]] = None,
        **kwargs: Any
    ) -> Series:
        return self._transform(func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs)

    def func_cuazy428(self, how: str, numeric_only: bool = False, **kwargs: Any) -> Series:
        obj = self._obj_with_exclusions
        try:
            result = self._grouper._cython_operation('transform', obj._values, how, 0, **kwargs)
        except NotImplementedError as err:
            raise TypeError(f'{how} is not supported for {obj.dtype} dtype') from err
        return obj._constructor(result, index=self.obj.index, name=obj.name)

    def func_pev7cii3(
        self, 
        func: Callable, 
        engine: Optional[str], 
        engine_kwargs: Optional[Dict[str, bool]], 
        *args: Any, 
        **kwargs: Any
    ) -> Series:
        if maybe_use_numba(engine):
            return self._transform_with_numba(func, *args, engine_kwargs=engine_kwargs, **kwargs)
        assert callable(func)
        klass = type(self.obj)
        results = []
        for name, group in self._grouper.get_iterator(self._obj_with_exclusions):
            object.__setattr__(group, 'name', name)
            res = func(group, *args, **kwargs)
            results.append(klass(res, index=group.index))
        if results:
            from pandas.core.reshape.concat import concat
            concatenated = concat(results, ignore_index=True)
            result = self._set_result_index_ordered(concatenated)
        else:
            result = self.obj._constructor(dtype=np.float64)
        result.name = self.obj.name
        return result

    def filter(self, func: Callable, dropna: bool = True, *args: Any, **kwargs: Any) -> Series:
        if isinstance(func, str):
            wrapper = lambda x: getattr(x, func)(*args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)

        def func_qw7jpydn(x: Any) -> bool:
            b = wrapper(x)
            return notna(b) and b
        try:
            indices = [self._get_index(name) for name, group in self._grouper.get_iterator(self._obj_with_exclusions) if func_qw7jpydn(group)]
        except (ValueError, TypeError) as err:
            raise TypeError('the filter must return a boolean result') from err
        filtered = self._apply_filter(indices, dropna)
        return filtered

    def func_rkq5fflu(self, dropna: bool = True) -> Series:
        ids = self._grouper.ids
        ngroups = self._grouper.ngroups
        val = self.obj._values
        codes, uniques = algorithms.factorize(val, use_na_sentinel=dropna, sort=False)
        if self._grouper.has_dropped_na:
            mask = ids >= 0
            ids = ids[mask]
            codes = codes[mask]
        group_index = get_group_index(
            labels=[ids, codes], shape=(ngroups, len(uniques)), sort=False, xnull=dropna
        )
        if dropna:
            mask = group_index >= 0
            if (~mask).any():
                ids = ids[mask]
                group_index = group_index[mask]
        mask = duplicated(group_index, 'first')
        res = np.bincount(ids[~mask], minlength=ngroups)
        res = ensure_int64(res)
        ri = self._grouper.result_index
        result = self.obj._constructor(res, index=ri, name=self.obj.name)
        if not self.as_index:
            result = self._insert_inaxis_grouper(result)
            result.index = default_index(len(result))
        return result

    def func_awj8s9e1(
        self, 
        percentiles: Optional[List[float]] = None, 
        include: Optional[Any] = None, 
        exclude: Optional[Any] = None
    ) -> DataFrame:
        return super().describe(percentiles=percentiles, include=include, exclude=exclude)

    def func_e58sx644(
        self, 
        normalize: bool = False, 
        sort: bool = True, 
        ascending: bool = False,
        bins: Optional[Union[int, List[int]]] = None, 
        dropna: bool = True
    ) -> Union[Series, DataFrame]:
        name = 'proportion' if normalize else 'count'
        if bins is None:
            result = self._value_counts(normalize=normalize, sort=sort, ascending=ascending, dropna=dropna)
            result.name = name
            return result
        from pandas.core.reshape.merge import get_join_indexers
        from pandas.core.reshape.tile import cut
        ids = self._grouper.ids
        val = self.obj._values
        index_names = self._grouper.names + [self.obj.name]
        if isinstance(val.dtype, CategoricalDtype) or bins is not None and not np.iterable(bins):
            ser = self.apply(Series.value_counts, normalize=normalize, sort=sort, ascending=ascending, bins=bins)
            ser.name = name
            ser.index.names = index_names
            return ser
        mask = ids != -1
        ids, val = ids[mask], val[mask]
        if bins is None:
            lab, lev = algorithms.factorize(val, sort=True)
            llab = lambda lab, inc: lab[inc]
        else:
            cat_ser = cut(Series(val, copy=False), bins, include_lowest=True)
            cat_obj = cast('Categorical', cat_ser._values)
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
        rep = partial(np.repeat, repeats=np.add.reduceat(inc, idx))
        if isinstance(self._grouper.result_index, MultiIndex):
            codes = list(self._grouper.result_index.codes)
        else:
            codes = [algorithms.factorize(
                self._grouper.result_index, 
                sort=self._grouper._sort, 
                use_na_sentinel=self._grouper.dropna
            )[0]]
        codes = [rep(level_codes) for level_codes in codes] + [llab(lab, inc)]
        levels = self._grouper.levels + [lev]
        if dropna:
            mask = codes[-1] != -1
            if mask.all():
                dropna = False
            else:
                out, codes = out[mask], [level_codes[mask] for level_codes in codes]
        if normalize:
            out = out.astype('float')
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
            diff = np.zeros(len(out), dtype='bool')
            for level_codes in codes[:-1]:
                diff |= np.r_[True, level_codes[1:] != level_codes[:-1]]
            ncat, nbin = diff.sum(), len(levels[-1])
            left = [np.repeat(np.arange(ncat), nbin), np.tile(np.arange(nbin), ncat)]
            right = [diff.cumsum() - 1, codes[-1]]
            _, idx = get_join_indexers(left, right, sort=False, how='left')
            if idx is not None:
                out = np.where(idx != -1, out[idx], 0)
            if sort:
                sorter = np.lexsort((out if ascending else -out, left[0]))
                out, left[-1] = out[sorter], left[-1][sorter]

            def func_nvuywkzz(lev_codes: Any) -> Any:
                return np.repeat(lev_codes[diff], nbin)
            codes = [func_nvuywkzz(lev_codes) for lev_codes in codes[:-1]]
            codes.append(left[-1])
        mi