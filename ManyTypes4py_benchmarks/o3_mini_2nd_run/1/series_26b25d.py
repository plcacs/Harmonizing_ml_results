from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping, Sequence
import functools
import operator
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# ... (other imports and definitions)

class Series(NDFrame):
    # ... (other attributes and methods)

    def any(self, *, axis: int = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> Any:
        nv.validate_logical_func((), kwargs, fname='any')
        validate_bool_kwarg(skipna, 'skipna', none_allowed=False)
        return self._reduce(nanops.nanany, name='any', axis=axis, numeric_only=bool_only, skipna=skipna, filter_type='bool')

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='all')
    def all(self, axis: int = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> Any:
        nv.validate_logical_func((), kwargs, fname='all')
        validate_bool_kwarg(skipna, 'skipna', none_allowed=False)
        return self._reduce(nanops.nanall, name='all', axis=axis, numeric_only=bool_only, skipna=skipna, filter_type='bool')

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='min')
    def min(self, axis: Optional[int] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.min(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='max')
    def max(self, axis: Optional[int] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.max(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='sum')
    def sum(self, axis: Optional[int] = 0, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> Any:
        return NDFrame.sum(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='prod')
    def prod(self, axis: Optional[int] = 0, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> Any:
        return NDFrame.prod(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='mean')
    def mean(self, axis: Optional[int] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.mean(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='median')
    def median(self, axis: Optional[int] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.median(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='sem')
    def sem(self, axis: Optional[int] = 0, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.sem(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='var')
    def var(self, axis: Optional[int] = 0, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.var(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='std')
    def std(self, axis: Optional[int] = 0, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.std(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='skew')
    def skew(self, axis: Optional[int] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.skew(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='kurt')
    def kurt(self, axis: Optional[int] = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.kurt(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
    kurtosis = kurt
    product = prod

    @doc(make_doc('cummin', ndim=1))
    def cummin(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        return NDFrame.cummin(self, axis, skipna, *args, **kwargs)

    @doc(make_doc('cummax', ndim=1))
    def cummax(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        return NDFrame.cummax(self, axis, skipna, *args, **kwargs)

    @doc(make_doc('cumsum', ndim=1))
    def cumsum(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        return NDFrame.cumsum(self, axis, skipna, *args, **kwargs)

    @doc(make_doc('cumprod', ndim=1))
    def cumprod(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        return NDFrame.cumprod(self, axis, skipna, *args, **kwargs)

    def dropna(self, *, axis: int = 0, inplace: bool = False, how: Optional[str] = None, ignore_index: bool = False) -> Optional[Series]:
        inplace = validate_bool_kwarg(inplace, 'inplace')
        ignore_index = validate_bool_kwarg(ignore_index, 'ignore_index')
        self._get_axis_number(axis or 0)
        if self._can_hold_na:
            result = remove_na_arraylike(self)
        elif not inplace:
            result = self.copy(deep=False)
        else:
            result = self
        if ignore_index:
            result.index = default_index(len(result))
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def to_timestamp(self, freq: Optional[str] = None, how: str = 'start', copy: Any = lib.no_default) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, PeriodIndex):
            raise TypeError(f'unsupported Type {type(self.index).__name__}')
        new_obj = self.copy(deep=False)
        new_index = self.index.to_timestamp(freq=freq, how=how)
        setattr(new_obj, 'index', new_index)
        return new_obj

    def to_period(self, freq: Optional[str] = None, copy: Any = lib.no_default) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError(f'unsupported Type {type(self.index).__name__}')
        new_obj = self.copy(deep=False)
        new_index = self.index.to_period(freq=freq)
        setattr(new_obj, 'index', new_index)
        return new_obj

    def any_other_method(self, arg: Any) -> Any:
        # Placeholder for other methods that need type annotations.
        pass

    def eq(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.eq, level=level, fill_value=fill_value, axis=axis)

    def ne(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.ne, level=level, fill_value=fill_value, axis=axis)

    def le(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.le, level=level, fill_value=fill_value, axis=axis)

    def lt(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.lt, level=level, fill_value=fill_value, axis=axis)

    def ge(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.ge, level=level, fill_value=fill_value, axis=axis)

    def gt(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.gt, level=level, fill_value=fill_value, axis=axis)

    def add(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.add, level=level, fill_value=fill_value, axis=axis)

    def radd(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.radd, level=level, fill_value=fill_value, axis=axis)

    def sub(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.sub, level=level, fill_value=fill_value, axis=axis)
    subtract = sub

    def rsub(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rsub, level=level, fill_value=fill_value, axis=axis)

    def mul(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.mul, level=level, fill_value=fill_value, axis=axis)
    multiply = mul

    def rmul(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rmul, level=level, fill_value=fill_value, axis=axis)

    def truediv(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.truediv, level=level, fill_value=fill_value, axis=axis)
    div = truediv
    divide = truediv

    def rtruediv(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rtruediv, level=level, fill_value=fill_value, axis=axis)
    rdiv = rtruediv

    def floordiv(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.floordiv, level=level, fill_value=fill_value, axis=axis)

    def rfloordiv(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rfloordiv, level=level, fill_value=fill_value, axis=axis)

    def mod(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.mod, level=level, fill_value=fill_value, axis=axis)

    def rmod(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rmod, level=level, fill_value=fill_value, axis=axis)

    def pow(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.pow, level=level, fill_value=fill_value, axis=axis)

    def rpow(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rpow, level=level, fill_value=fill_value, axis=axis)

    def divmod(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, divmod, level=level, fill_value=fill_value, axis=axis)

    def rdivmod(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Any = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rdivmod, level=level, fill_value=fill_value, axis=axis)

    def _reduce(self, op: Callable[..., Any], name: str, *, axis: Optional[int] = 0, skipna: bool = True, numeric_only: bool = False, filter_type: Optional[str] = None, **kwds: Any) -> Any:
        delegate = self._values
        if axis is not None:
            self._get_axis_number(axis)
        if isinstance(delegate, ExtensionArray):
            return delegate._reduce(name, skipna=skipna, **kwds)
        else:
            if numeric_only and self.dtype.kind not in 'iufcb':
                kwd_name = 'numeric_only'
                if name in ['any', 'all']:
                    kwd_name = 'bool_only'
                raise TypeError(f'Series.{name} does not allow {kwd_name}={numeric_only} with non-numeric dtypes.')
            return op(delegate, skipna=skipna, **kwds)

    def argsort(self, axis: int = 0, kind: str = 'quicksort', order: Any = None, stable: Any = None) -> Series:
        if axis != -1:
            self._get_axis_number(axis)
        result = self.array.argsort(kind=kind)
        res = self._constructor(result, index=self.index, name=self.name, dtype=np.intp, copy=False)
        return res.__finalize__(self, method='argsort')

    def nlargest(self, n: int = 5, keep: Union[str, bool] = 'first') -> Series:
        return selectn.SelectNSeries(self, n=n, keep=keep).nlargest()

    def nsmallest(self, n: int = 5, keep: Union[str, bool] = 'first') -> Series:
        return selectn.SelectNSeries(self, n=n, keep=keep).nsmallest()

    def swaplevel(self, i: Union[int, str] = -2, j: Union[int, str] = -1, copy: Any = lib.no_default) -> Series:
        self._check_copy_deprecation(copy)
        assert isinstance(self.index, MultiIndex)
        result: Series = self.copy(deep=False)
        result.index = self.index.swaplevel(i, j)
        return result

    def reorder_levels(self, order: List[int]) -> Series:
        if not isinstance(self.index, MultiIndex):
            raise Exception('Can only reorder levels on a hierarchical axis.')
        result: Series = self.copy(deep=False)
        assert isinstance(result.index, MultiIndex)
        result.index = result.index.reorder_levels(order)
        return result

    def explode(self, ignore_index: bool = False) -> Series:
        if isinstance(self.dtype, ExtensionDtype):
            values, counts = self._values._explode()
        elif len(self) and is_object_dtype(self.dtype):
            values, counts = reshape.explode(np.asarray(self._values))
        else:
            result: Series = self.copy()
            return result.reset_index(drop=True) if ignore_index else result
        if ignore_index:
            index = default_index(len(values))
        else:
            index = self.index.repeat(counts)
        return self._constructor(values, index=index, name=self.name, copy=False)

    def unstack(self, level: Union[int, str, List[Union[int, str]]] = -1, fill_value: Any = None, sort: bool = True) -> DataFrame:
        from pandas.core.reshape.reshape import unstack
        return unstack(self, level, fill_value, sort)

    def map(self, arg: Union[Callable[[Any], Any], Mapping[Any, Any], Series], na_action: Optional[str] = None, **kwargs: Any) -> Series:
        if callable(arg):
            arg = functools.partial(arg, **kwargs)
        new_values = self._map_values(arg, na_action=na_action)
        return self._constructor(new_values, index=self.index, copy=False).__finalize__(self, method='map')

    def _gotitem(self, key: Any, ndim: int, subset: Optional[Any] = None) -> Any:
        return self

    def aggregate(self, func: Optional[Any] = None, axis: int = 0, *args: Any, **kwargs: Any) -> Any:
        self._get_axis_number(axis)
        if func is None:
            func = dict(kwargs.items())
        op = SeriesApply(self, func, args=args, kwargs=kwargs)
        result = op.agg()
        return result
    agg = aggregate

    def transform(self, func: Any, axis: int = 0, *args: Any, **kwargs: Any) -> Any:
        self._get_axis_number(axis)
        ser: Series = self.copy(deep=False)
        result = SeriesApply(ser, func=func, args=args, kwargs=kwargs).transform()
        return result

    def apply(self, func: Any, args: Tuple[Any, ...] = (), *, by_row: Union[bool, str] = 'compat', **kwargs: Any) -> Any:
        return SeriesApply(self, func, by_row=by_row, args=args, kwargs=kwargs).apply()

    def _reindex_indexer(self, new_index: Any, indexer: Optional[Any]) -> Series:
        if indexer is None and (new_index is None or new_index.names == self.index.names):
            return self.copy(deep=False)
        new_values = algorithms.take_nd(self._values, indexer, allow_fill=True, fill_value=None)
        return self._constructor(new_values, index=new_index, copy=False)

    def _needs_reindex_multi(self, axes: Any, method: Any, level: Any) -> bool:
        return False

    @overload
    def rename(self, index: Any = None, *, axis: Optional[Union[int, str]] = None, copy: Any, inplace: bool, level: Any, errors: str) -> Series: ...
    @overload
    def rename(self, index: Any = None, *, axis: Optional[Union[int, str]] = None, copy: Any, inplace: bool = False, level: Any, errors: str) -> Series: ...
    def rename(self, index: Any = None, *, axis: Optional[Union[int, str]] = None, copy: Any = lib.no_default, inplace: bool = False, level: Any = None, errors: str = 'ignore') -> Optional[Series]:
        self._check_copy_deprecation(copy)
        if axis is not None:
            axis = self._get_axis_number(axis)
        if callable(index) or is_dict_like(index):
            return super()._rename(index, inplace=inplace, level=level, errors=errors)
        else:
            return self._set_name(index, inplace=inplace)

    def set_axis(self, labels: Iterable[Any], *, axis: Optional[Union[int, str]] = 0, copy: Any = lib.no_default) -> Any:
        return super().set_axis(labels, axis=axis, copy=copy)

    def reindex(self, index: Optional[Any] = None, *, axis: Optional[Union[int, str]] = None, method: Optional[str] = None, copy: Any = lib.no_default, level: Any = None, fill_value: Any = None, limit: Optional[int] = None, tolerance: Any = None) -> Any:
        return super().reindex(index=index, method=method, level=level, fill_value=fill_value, limit=limit, tolerance=tolerance, copy=copy)

    @overload
    def rename_axis(self, mapper: Any = lib.no_default, *, index: Any = lib.no_default, axis: Optional[Union[int, str]] = 0, copy: Any = lib.no_default, inplace: bool) -> Series: ...
    @overload
    def rename_axis(self, mapper: Any = lib.no_default, *, index: Any = lib.no_default, axis: Optional[Union[int, str]] = 0, copy: Any = lib.no_default, inplace: bool = False) -> Series: ...
    def rename_axis(self, mapper: Any = lib.no_default, *, index: Any = lib.no_default, axis: Optional[Union[int, str]] = 0, copy: Any = lib.no_default, inplace: bool = False) -> Series:
        return super().rename_axis(mapper=mapper, index=index, axis=axis, inplace=inplace, copy=copy)

    @overload
    def drop(self, labels: Any = None, *, axis: Optional[Union[int, str]] = 0, index: Any = None, columns: Any = None, level: Any = None, inplace: bool, errors: str) -> Series: ...
    @overload
    def drop(self, labels: Any = None, *, axis: Optional[Union[int, str]] = 0, index: Any = None, columns: Any = None, level: Any = None, inplace: bool = False, errors: str = 'raise') -> Series: ...
    def drop(self, labels: Any = None, *, axis: Optional[Union[int, str]] = 0, index: Any = None, columns: Any = None, level: Any = None, inplace: bool = False, errors: str = 'raise') -> Optional[Series]:
        return super().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)

    def pop(self, item: Any) -> Any:
        return super().pop(item=item)

    def info(self, verbose: Optional[bool] = None, buf: Optional[Any] = None, max_cols: Optional[int] = None, memory_usage: Optional[Union[bool, str]] = None, show_counts: bool = True) -> Any:
        return SeriesInfo(self, memory_usage).render(buf=buf, max_cols=max_cols, verbose=verbose, show_counts=show_counts)

    def memory_usage(self, index: bool = True, deep: bool = False) -> int:
        v: int = self._memory_usage(deep=deep)
        if index:
            v += self.index.memory_usage(deep=deep)
        return v

    def isin(self, values: Iterable[Any]) -> Series:
        result = algorithms.isin(self._values, values)
        return self._constructor(result, index=self.index, copy=False).__finalize__(self, method='isin')

    def between(self, left: Any, right: Any, inclusive: str = 'both') -> Series:
        if inclusive == 'both':
            lmask = self >= left
            rmask = self <= right
        elif inclusive == 'left':
            lmask = self >= left
            rmask = self < right
        elif inclusive == 'right':
            lmask = self > left
            rmask = self <= right
        elif inclusive == 'neither':
            lmask = self > left
            rmask = self < right
        else:
            raise ValueError("Inclusive has to be either string of 'both','left', 'right', or 'neither'.")
        return lmask & rmask

    def case_when(self, caselist: List[Tuple[Any, Any]]) -> Series:
        if not isinstance(caselist, list):
            raise TypeError(f'The caselist argument should be a list; instead got {type(caselist)}')
        if not caselist:
            raise ValueError('provide at least one boolean condition, with a corresponding replacement.')
        for num, entry in enumerate(caselist):
            if not isinstance(entry, tuple):
                raise TypeError(f'Argument {num} must be a tuple; instead got {type(entry)}.')
            if len(entry) != 2:
                raise ValueError(f'Argument {num} must have length 2; a condition and replacement; instead got length {len(entry)}.')
        caselist = [(com.apply_if_callable(condition, self), com.apply_if_callable(replacement, self)) for condition, replacement in caselist]
        default: Series = self.copy(deep=False)
        conditions, replacements = zip(*caselist)
        common_dtypes = [infer_dtype_from(arg)[0] for arg in [*replacements, default]]
        if len(set(common_dtypes)) > 1:
            common_dtype = find_common_type(common_dtypes)
            updated_replacements: List[Any] = []
            for condition, replacement in zip(conditions, replacements):
                if is_scalar(replacement):
                    replacement = construct_1d_arraylike_from_scalar(value=replacement, length=len(condition), dtype=common_dtype)
                elif isinstance(replacement, ABCSeries):
                    replacement = replacement.astype(common_dtype)
                else:
                    replacement = pd_array(replacement, dtype=common_dtype)
                updated_replacements.append(replacement)
            replacements = updated_replacements
            default = default.astype(common_dtype)
        counter = range(len(conditions) - 1, -1, -1)
        for position, condition, replacement in zip(counter, reversed(conditions), reversed(replacements)):
            try:
                default = default.mask(condition, other=replacement, axis=0, inplace=False, level=None)
            except Exception as error:
                raise ValueError(f'Failed to apply condition{position} and replacement{position}.') from error
        return default

    def isna(self) -> Any:
        return NDFrame.isna(self)

    def isnull(self) -> Any:
        return super().isnull()

    def notna(self) -> Any:
        return super().notna()

    def notnull(self) -> Any:
        return super().notnull()

    @overload
    def drop_duplicates(self, *, keep: Union[str, bool] = 'first', inplace: bool, ignore_index: bool) -> Series: ...
    @overload
    def drop_duplicates(self, *, keep: Union[str, bool] = 'first', inplace: bool = False, ignore_index: bool = False) -> Series: ...
    def drop_duplicates(self, *, keep: Union[str, bool] = 'first', inplace: bool = False, ignore_index: bool = False) -> Optional[Series]:
        inplace = validate_bool_kwarg(inplace, 'inplace')
        result = super().drop_duplicates(keep=keep)
        if ignore_index:
            result.index = default_index(len(result))
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def duplicated(self, keep: Union[str, bool] = 'first') -> Series:
        res = self._duplicated(keep=keep)
        result: Series = self._constructor(res, index=self.index, copy=False)
        return result.__finalize__(self, method='duplicated')

    def idxmin(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        axis = self._get_axis_number(axis)
        iloc = self.argmin(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def idxmax(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        axis = self._get_axis_number(axis)
        iloc = self.argmax(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def autocorr(self, lag: int = 1) -> float:
        return self.corr(cast(Series, self.shift(lag)))
    
    def dot(self, other: Any) -> Union[Any, Series, np.ndarray]:
        if isinstance(other, (Series, ABCDataFrame)):
            common = self.index.union(other.index)
            if len(common) > len(self.index) or len(common) > len(other.index):
                raise ValueError('matrices are not aligned')
            left: Series = self.reindex(index=common)
            right: Any = other.reindex(index=common)
            lvals = left.values
            rvals = right.values
        else:
            lvals = self.values
            rvals = np.asarray(other)
            if lvals.shape[0] != rvals.shape[0]:
                raise Exception(f'Dot product shape mismatch, {lvals.shape} vs {rvals.shape}')
        if isinstance(other, ABCDataFrame):
            return self._constructor(np.dot(lvals, rvals), index=other.columns, copy=False).__finalize__(self, method='dot')
        elif isinstance(other, Series):
            return np.dot(lvals, rvals)
        elif isinstance(rvals, np.ndarray):
            return np.dot(lvals, rvals)
        else:
            raise TypeError(f'unsupported type: {type(other)}')

    def __matmul__(self, other: Any) -> Any:
        return self.dot(other)

    def __rmatmul__(self, other: Any) -> Any:
        return self.dot(np.transpose(other))

    def searchsorted(self, value: Any, side: str = 'left', sorter: Optional[Any] = None) -> Any:
        return base.IndexOpsMixin.searchsorted(self, value, side=side, sorter=sorter)

    def _append(self, to_append: Any, ignore_index: bool = False, verify_integrity: bool = False) -> Any:
        from pandas.core.reshape.concat import concat
        if isinstance(to_append, (list, tuple)):
            to_concat: List[Any] = [self]
            to_concat.extend(to_append)
        else:
            to_concat = [self, to_append]
        if any((isinstance(x, (ABCDataFrame,)) for x in to_concat[1:])):
            msg = 'to_append should be a Series or list/tuple of Series, got DataFrame'
            raise TypeError(msg)
        return concat(to_concat, ignore_index=ignore_index, verify_integrity=verify_integrity)

    def compare(self, other: Any, align_axis: int = 1, keep_shape: bool = False, keep_equal: bool = False, result_names: Tuple[str, str] = ('self', 'other')) -> Any:
        return super().compare(other=other, align_axis=align_axis, keep_shape=keep_shape, keep_equal=keep_equal, result_names=result_names)

    def combine(self, other: Any, func: Callable[[Any, Any], Any], fill_value: Optional[Any] = None) -> Series:
        if fill_value is None:
            fill_value = na_value_for_dtype(self.dtype, compat=False)
        if isinstance(other, Series):
            new_index = self.index.union(other.index)
            new_name = ops.get_op_result_name(self, other)
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all='ignore'):
                for i, idx in enumerate(new_index):
                    lv = self.get(idx, fill_value)
                    rv = other.get(idx, fill_value)
                    new_values[i] = func(lv, rv)
        else:
            new_index = self.index
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all='ignore'):
                new_values[:] = [func(lv, other) for lv in self._values]
            new_name = self.name
        npvalues = lib.maybe_convert_objects(new_values, try_float=False)
        same_dtype = isinstance(self.dtype, (StringDtype, CategoricalDtype))
        res_values = maybe_cast_pointwise_result(npvalues, self.dtype, same_dtype=same_dtype)
        return self._constructor(res_values, index=new_index, name=new_name, copy=False)

    def combine_first(self, other: Series) -> Series:
        from pandas.core.reshape.concat import concat
        if self.dtype == other.dtype:
            if self.index.equals(other.index):
                return self.mask(self.isna(), other)
            elif self._can_hold_na and (not isinstance(self.dtype, SparseDtype)):
                this, other = self.align(other, join='outer')
                return this.mask(this.isna(), other)
        new_index = self.index.union(other.index)
        this: Series = self
        keep_other = other.index.difference(this.index[notna(this)])
        keep_this = this.index.difference(keep_other)
        this = this.reindex(keep_this)
        other = other.reindex(keep_other)
        if this.dtype.kind == 'M' and other.dtype.kind != 'M':
            other = to_datetime(other)
        combined: Series = concat([this, other])
        combined = combined.reindex(new_index)
        return combined.__finalize__(self, method='combine_first')

    def update(self, other: Any) -> None:
        if not PYPY:
            if sys.getrefcount(self) <= REF_COUNT:
                warnings.warn(_chained_assignment_method_msg, ChainedAssignmentError, stacklevel=2)
        if not isinstance(other, Series):
            other = Series(other)
        other = other.reindex_like(self)
        mask = notna(other)
        self._mgr = self._mgr.putmask(mask=mask, new=other)

    def sort_values(self, *, axis: Optional[Union[int, str]] = 0, ascending: Union[bool, List[bool]] = True, inplace: bool = False, kind: str = 'quicksort', na_position: str = 'last', ignore_index: bool = False, key: Optional[Callable[[Series], Any]] = None) -> Optional[Series]:
        inplace = validate_bool_kwarg(inplace, 'inplace')
        self._get_axis_number(axis)
        if is_list_like(ascending):
            ascending = cast(Sequence[bool], ascending)
            if len(ascending) != 1:
                raise ValueError(f'Length of ascending ({len(ascending)}) must be 1 for Series')
            ascending = ascending[0]
        ascending = validate_ascending(ascending)
        if na_position not in ['first', 'last']:
            raise ValueError(f'invalid na_position: {na_position}')
        if key:
            values_to_sort = cast(Series, ensure_key_mapped(self, key))._values
        else:
            values_to_sort = self._values
        sorted_index = nargsort(values_to_sort, kind, bool(ascending), na_position)
        if is_range_indexer(sorted_index, len(sorted_index)):
            if inplace:
                return self._update_inplace(self)
            return self.copy(deep=False)
        result: Series = self._constructor(self._values[sorted_index], index=self.index[sorted_index], copy=False)
        if ignore_index:
            result.index = default_index(len(sorted_index))
        if not inplace:
            return result.__finalize__(self, method='sort_values')
        self._update_inplace(result)
        return None

    def sort_index(self, *, axis: Optional[Union[int, str]] = 0, level: Optional[Union[int, str, List[Union[int, str]]]] = None, ascending: Union[bool, List[bool]] = True, inplace: bool = False, kind: str = 'quicksort', na_position: str = 'last', sort_remaining: bool = True, ignore_index: bool = False, key: Optional[Callable[[Any], Any]] = None) -> Optional[Series]:
        return super().sort_index(axis=axis, level=level, ascending=ascending, inplace=inplace, kind=kind, na_position=na_position, sort_remaining=sort_remaining, ignore_index=ignore_index, key=key)

    def argsort(self, axis: int = 0, kind: str = 'quicksort', order: Any = None, stable: Any = None) -> Series:
        if axis != -1:
            self._get_axis_number(axis)
        result = self.array.argsort(kind=kind)
        res: Series = self._constructor(result, index=self.index, name=self.name, dtype=np.intp, copy=False)
        return res.__finalize__(self, method='argsort')

    def nlargest(self, n: int = 5, keep: Union[str, bool] = 'first') -> Series:
        return selectn.SelectNSeries(self, n=n, keep=keep).nlargest()

    def nsmallest(self, n: int = 5, keep: Union[str, bool] = 'first') -> Series:
        return selectn.SelectNSeries(self, n=n, keep=keep).nsmallest()

    # ... (other methods and properties)
