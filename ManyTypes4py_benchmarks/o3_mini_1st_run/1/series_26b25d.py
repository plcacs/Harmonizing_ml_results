#!/usr/bin/env python3
"""
This module provides a Series class.
"""

from __future__ import annotations
import operator
import functools
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from pandas._libs.lib import maybe_box_native
from pandas._libs import lib
from pandas.core.indexes.api import Index, MultiIndex, default_index
from pandas.core.dtypes.common import is_list_like, is_scalar, isna, notna, is_object_dtype
from pandas.core.dtypes.cast import maybe_cast_pointwise_result, construct_1d_arraylike_from_scalar
from pandas.core.array_algos import algorithms
from pandas.core.missing import remove_na_arraylike
from pandas.core.base import NDFrame
from pandas.core.reshape.reshape import unstack
from pandas.core.indexing import extract_array
from pandas.core.plotting import hist_series
from pandas.core.strings import StringMethods
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.datetimelike import CombinedDatetimelikeProperties
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.datetimes import DatetimeIndex

# The Series class
class Series(NDFrame):
    # Class attributes, properties, etc.
    index: Index
    name: Optional[Any]

    # Accessors
    str: StringMethods
    dt: CombinedDatetimelikeProperties
    cat: CategoricalAccessor
    plot: Any  # Typically PlotAccessor type
    sparse: SparseAccessor

    hist: Any = hist_series

    def any(self, *, axis: int = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> Any:
        """
        Return whether any element is True.
        """
        nv_args: Tuple[Any, ...] = ()
        # Validate logical function arguments, then reduce using nanops.nanany.
        return self._reduce(algorithms.nanany, name="any", axis=axis, numeric_only=bool_only, skipna=skipna, **kwargs)

    def all(self, *, axis: int = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> Any:
        """
        Return whether all elements are True.
        """
        nv_args: Tuple[Any, ...] = ()
        return self._reduce(algorithms.nanall, name="all", axis=axis, numeric_only=bool_only, skipna=skipna, **kwargs)

    def min(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        """
        Return the minimum of the Series.
        """
        return NDFrame.min(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def max(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        """
        Return the maximum of the Series.
        """
        return NDFrame.max(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def sum(self, axis: Optional[int] = 0, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> Any:
        """
        Return the sum of the Series.
        """
        return NDFrame.sum(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    def prod(self, axis: Optional[int] = 0, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> Any:
        """
        Return the product of the Series.
        """
        return NDFrame.prod(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    product = prod

    def mean(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        """
        Return the mean of the Series.
        """
        return NDFrame.mean(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def median(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        """
        Return the median of the Series.
        """
        return NDFrame.median(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def sem(self, axis: Optional[int] = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        """
        Return the standard error of the mean of the Series.
        """
        return NDFrame.sem(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    def var(self, axis: Optional[int] = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        """
        Return the variance of the Series.
        """
        return NDFrame.var(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    def std(self, axis: Optional[int] = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        """
        Return the standard deviation of the Series.
        """
        return NDFrame.std(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    def skew(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        """
        Return the skew of the Series.
        """
        return NDFrame.skew(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def kurt(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        """
        Return the kurtosis of the Series (Fisher's definition).
        """
        return NDFrame.kurt(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
    kurtosis = kurt

    @property
    def product(self) -> Any:
        return self.prod()

    def cummin(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Series:
        """
        Return cumulative minimum of the Series.
        """
        result = NDFrame.cummin(self, axis, skipna, *args, **kwargs)
        return result  # type: ignore

    def cummax(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Series:
        """
        Return cumulative maximum of the Series.
        """
        result = NDFrame.cummax(self, axis, skipna, *args, **kwargs)
        return result  # type: ignore

    def cumsum(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Series:
        """
        Return cumulative sum of the Series.
        """
        result = NDFrame.cumsum(self, axis, skipna, *args, **kwargs)
        return result  # type: ignore

    def cumprod(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Series:
        """
        Return cumulative product of the Series.
        """
        result = NDFrame.cumprod(self, axis, skipna, *args, **kwargs)
        return result  # type: ignore

    def any_(self, *, axis: int = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> Any:
        # Backward compatibility alias, if needed.
        return self.any(axis=axis, bool_only=bool_only, skipna=skipna, **kwargs)

    def eq(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise equality comparison.
        """
        res_name = ops.get_op_result_name(self, other)
        if isinstance(other, Series) and (not self._indexed_same(other)):
            raise ValueError('Can only compare identically-labeled Series objects')
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        res_values = ops.comparison_op(lvalues, rvalues, operator.eq)
        return self._construct_result(res_values, name=res_name)

    def ne(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise inequality comparison.
        """
        return self._flex_method(other, operator.ne, level=level, fill_value=fill_value, axis=axis)

    def le(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise less-than-or-equal-to comparison.
        """
        return self._flex_method(other, operator.le, level=level, fill_value=fill_value, axis=axis)

    def lt(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise less-than comparison.
        """
        return self._flex_method(other, operator.lt, level=level, fill_value=fill_value, axis=axis)

    def ge(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise greater-than-or-equal-to comparison.
        """
        return self._flex_method(other, operator.ge, level=level, fill_value=fill_value, axis=axis)

    def gt(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise greater-than comparison.
        """
        return self._flex_method(other, operator.gt, level=level, fill_value=fill_value, axis=axis)

    def add(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise addition.
        """
        return self._flex_method(other, operator.add, level=level, fill_value=fill_value, axis=axis)

    def radd(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return reflected elementwise addition.
        """
        return self._flex_method(other, roperator.radd, level=level, fill_value=fill_value, axis=axis)

    def sub(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise subtraction.
        """
        return self._flex_method(other, operator.sub, level=level, fill_value=fill_value, axis=axis)
    subtract = sub

    def rsub(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return reflected elementwise subtraction.
        """
        return self._flex_method(other, roperator.rsub, level=level, fill_value=fill_value, axis=axis)

    def mul(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise multiplication.
        """
        return self._flex_method(other, operator.mul, level=level, fill_value=fill_value, axis=axis)
    multiply = mul

    def rmul(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return reflected elementwise multiplication.
        """
        return self._flex_method(other, roperator.rmul, level=level, fill_value=fill_value, axis=axis)

    def truediv(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise true division.
        """
        return self._flex_method(other, operator.truediv, level=level, fill_value=fill_value, axis=axis)
    div = truediv
    divide = truediv

    def rtruediv(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return reflected elementwise true division.
        """
        return self._flex_method(other, roperator.rtruediv, level=level, fill_value=fill_value, axis=axis)
    rdiv = rtruediv

    def floordiv(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise floor division.
        """
        return self._flex_method(other, operator.floordiv, level=level, fill_value=fill_value, axis=axis)

    def rfloordiv(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return reflected elementwise floor division.
        """
        return self._flex_method(other, roperator.rfloordiv, level=level, fill_value=fill_value, axis=axis)

    def mod(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise modulo.
        """
        return self._flex_method(other, operator.mod, level=level, fill_value=fill_value, axis=axis)

    def rmod(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return reflected elementwise modulo.
        """
        return self._flex_method(other, roperator.rmod, level=level, fill_value=fill_value, axis=axis)

    def pow(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return elementwise exponentiation.
        """
        return self._flex_method(other, operator.pow, level=level, fill_value=fill_value, axis=axis)

    def rpow(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        """
        Return reflected elementwise exponentiation.
        """
        return self._flex_method(other, roperator.rpow, level=level, fill_value=fill_value, axis=axis)

    def divmod(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Any:
        """
        Return elementwise divmod.
        """
        return self._flex_method(other, divmod, level=level, fill_value=fill_value, axis=axis)

    def rdivmod(self, other: Any, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Any:
        """
        Return reflected elementwise divmod.
        """
        return self._flex_method(other, roperator.rdivmod, level=level, fill_value=fill_value, axis=axis)

    def _reduce(self, op: Callable[..., Any], name: str, *, axis: int = 0, skipna: bool = True, numeric_only: bool = False, filter_type: Optional[str] = None, **kwds: Any) -> Any:
        """
        Perform a reduction operation.
        """
        delegate = self._values
        if axis is not None:
            self._get_axis_number(axis)
        if hasattr(delegate, '_reduce'):
            return delegate._reduce(name, skipna=skipna, **kwds)
        else:
            if numeric_only and self.dtype.kind not in 'iufcb':
                kwd_name = 'numeric_only'
                if name in ['any', 'all']:
                    kwd_name = 'bool_only'
                raise TypeError(f'Series.{name} does not allow {kwd_name}={numeric_only} with non-numeric dtypes.')
            return op(delegate, skipna=skipna, **kwds)

    def cummin(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Series:
        result = NDFrame.cummin(self, axis, skipna, *args, **kwargs)
        return self._constructor(result, index=self.index, copy=False)  # type: ignore

    def _construct_result(self, result: Any, name: Any) -> Series:
        """
        Construct an appropriately-labelled Series from the result.
        """
        if isinstance(result, tuple):
            res1: Series = self._construct_result(result[0], name=name)
            res2: Series = self._construct_result(result[1], name=name)
            return (res1, res2)  # type: ignore
        dtype = getattr(result, 'dtype', None)
        out: Series = self._constructor(result, index=self.index, dtype=dtype, copy=False)
        out = out.__finalize__(self)
        out.name = name
        return out

    def _flex_method(self, other: Any, op: Callable[..., Any], *, level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None, axis: Optional[int] = 0) -> Series:
        if axis is not None:
            self._get_axis_number(axis)
        res_name = ops.get_op_result_name(self, other)
        if isinstance(other, Series):
            return self._binop(other, op, level=level, fill_value=fill_value)
        elif isinstance(other, (np.ndarray, list, tuple)):
            if len(other) != len(self):
                raise ValueError('Lengths must be equal')
            other = self._constructor(other, self.index, copy=False)
            result = self._binop(other, op, level=level, fill_value=fill_value)
            result._name = res_name
            return result
        else:
            if fill_value is not None:
                if isna(other):
                    return op(self, fill_value)
                self = self.fillna(fill_value)
            return op(self, other)

    def _binop(self, other: Series, func: Callable[..., Any], level: Optional[Union[int,str]] = None, fill_value: Optional[Any] = None) -> Series:
        this: Series = self
        if not self.index.equals(other.index):
            this, other = self.align(other, level=level, join='outer')
        this_vals, other_vals = ops.fill_binop(this._values, other._values, fill_value)
        with np.errstate(all='ignore'):
            result = func(this_vals, other_vals)
        name = ops.get_op_result_name(self, other)
        out: Series = this._construct_result(result, name)
        return out

    def map(self, arg: Union[Callable[[Any], Any], Mapping[Any, Any]], na_action: Optional[str] = None, **kwargs: Any) -> Series:
        """
        Map values of Series according to an input mapping or function.
        """
        if callable(arg):
            arg = functools.partial(arg, **kwargs)
        new_values = self._map_values(arg, na_action=na_action)
        return self._constructor(new_values, index=self.index, copy=False).__finalize__(self, method='map')

    def explode(self, ignore_index: bool = False) -> Series:
        """
        Transform each list-like element to a row.
        """
        if isinstance(self.dtype, ExtensionDtype):
            values, counts = self._values._explode()
        elif len(self) and is_object_dtype(self.dtype):
            values, counts = reshape.explode(np.asarray(self._values))
        else:
            result = self.copy()
            return result.reset_index(drop=True) if ignore_index else result
        index = default_index(len(values)) if ignore_index else self.index.repeat(counts)
        return self._constructor(values, index=index, name=self.name, copy=False)

    def unstack(self, level: Union[int, str] = -1, fill_value: Optional[Any] = None, sort: bool = True) -> Any:
        """
        Unstack Series with MultiIndex to produce DataFrame.
        """
        return unstack(self, level, fill_value, sort)

    def reorder_levels(self, order: List[int]) -> Series:
        """
        Rearrange index levels using input order.
        """
        if not isinstance(self.index, MultiIndex):
            raise Exception('Can only reorder levels on a hierarchical axis.')
        result: Series = self.copy(deep=False)
        result.index = result.index.reorder_levels(order)
        return result

    def rename(self, index: Any = None, *, axis: Optional[Union[int, str]] = None, copy: Any = lib.no_default, inplace: bool = False, level: Optional[Union[int, str]] = None, errors: str = 'ignore') -> Optional[Series]:
        """
        Alter Series index labels or name.
        """
        self._check_copy_deprecation(copy)
        if axis is not None:
            axis = self._get_axis_number(axis)
        if callable(index) or isinstance(index, Mapping):
            return super()._rename(index, inplace=inplace, level=level, errors=errors)
        else:
            return self._set_name(index, inplace=inplace)

    def set_axis(self, labels: Iterable[Any], *, axis: int = 0, copy: Any = lib.no_default) -> Series:
        """
        Set new index labels.
        """
        return super().set_axis(labels, axis=axis, copy=copy)

    def reindex(self, index: Optional[Iterable[Any]] = None, *, axis: Optional[Union[int, str]] = None, method: Optional[str] = None, copy: Any = lib.no_default, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, limit: Optional[int] = None, tolerance: Optional[Any] = None) -> Any:
        """
        Conform the Series to new index.
        """
        return super().reindex(index=index, method=method, level=level, fill_value=fill_value, limit=limit, tolerance=tolerance, copy=copy)

    def rename_axis(self, mapper: Any = lib.no_default, *, index: Any = lib.no_default, axis: int = 0, copy: Any = lib.no_default, inplace: bool = False) -> Optional[Series]:
        """
        Set the name of the axis for the index.
        """
        return super().rename_axis(mapper=mapper, index=index, axis=axis, inplace=inplace, copy=copy)

    def drop(self, labels: Optional[Union[Any, Iterable[Any]]] = None, *, axis: int = 0, index: Optional[Union[Any, Iterable[Any]]] = None, columns: Optional[Union[Any, Iterable[Any]]] = None, level: Optional[Union[int, str]] = None, inplace: bool = False, errors: str = 'raise') -> Optional[Series]:
        """
        Return Series with specified index labels removed.
        """
        return super().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)

    def pop(self, item: Any) -> Any:
        """
        Return item and drop it from Series.
        """
        return super().pop(item=item)

    def info(self, verbose: Optional[bool] = None, buf: Optional[Any] = None, max_cols: Optional[int] = None, memory_usage: Optional[bool] = None, show_counts: bool = True) -> Any:
        """
        Print a concise summary of the Series.
        """
        from pandas.io.formats.info import SeriesInfo, series_sub_kwargs, INFO_DOCSTRING
        return SeriesInfo(self, memory_usage).render(buf=buf, max_cols=max_cols, verbose=verbose, show_counts=show_counts)

    def memory_usage(self, index: bool = True, deep: bool = False) -> int:
        """
        Return the memory usage of the Series.
        """
        v: int = self._memory_usage(deep=deep)
        if index:
            v += self.index.memory_usage(deep=deep)
        return v

    def isin(self, values: Union[Iterable[Any], set]) -> Series:
        """
        Check whether each element is contained in values.
        """
        result = algorithms.isin(self._values, values)
        return self._constructor(result, index=self.index, copy=False).__finalize__(self, method='isin')

    def between(self, left: Any, right: Any, inclusive: str = 'both') -> Series:
        """
        Return boolean Series indicating if values are between left and right.
        """
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
            raise ValueError("inclusive has to be either 'both','left', 'right', or 'neither'.")
        return lmask & rmask

    def case_when(self, caselist: List[Tuple[Any, Any]]) -> Series:
        """
        Replace values where the conditions are True.
        """
        if not isinstance(caselist, list):
            raise TypeError(f'The caselist argument should be a list; instead got {type(caselist)}')
        if not caselist:
            raise ValueError('provide at least one condition and replacement.')
        for num, entry in enumerate(caselist):
            if not isinstance(entry, tuple):
                raise TypeError(f'Argument {num} must be a tuple; instead got {type(entry)}.')
            if len(entry) != 2:
                raise ValueError(f'Argument {num} must have length 2; got {len(entry)}.')
        caselist = [(com.apply_if_callable(condition, self), com.apply_if_callable(replacement, self)) for condition, replacement in caselist]  # type: ignore
        default: Series = self.copy(deep=False)
        conditions, replacements = zip(*caselist)
        common_dtypes = [infer_dtype_from(arg)[0] for arg in [*replacements, default]]
        if len(set(common_dtypes)) > 1:
            common_dtype = find_common_type(common_dtypes)
            updated_replacements: List[Any] = []
            for condition, replacement in zip(conditions, replacements):
                if is_scalar(replacement):
                    replacement = construct_1d_arraylike_from_scalar(value=replacement, length=len(condition), dtype=common_dtype)
                elif isinstance(replacement, Series):
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
                raise ValueError(f'Failed to apply condition {position} and replacement {position}.') from error
        return default

    def isna(self) -> Series:
        """
        Return boolean Series indicating if values are NA.
        """
        return NDFrame.isna(self)

    def isnull(self) -> Series:
        """
        Alias for isna.
        """
        return self.isna()

    def notna(self) -> Series:
        """
        Return boolean Series indicating if values are not NA.
        """
        return NDFrame.notna(self)

    def notnull(self) -> Series:
        """
        Alias for notna.
        """
        return self.notna()

    def dropna(self, *, axis: int = 0, inplace: bool = False, how: Optional[str] = None, ignore_index: bool = False) -> Optional[Series]:
        """
        Return a new Series with missing values removed.
        """
        inplace = bool(inplace)
        ignore_index = bool(ignore_index)
        self._get_axis_number(axis or 0)
        if self._can_hold_na:
            result: Series = remove_na_arraylike(self)
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
        """
        Cast Series from PeriodIndex to DatetimeIndex.
        """
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, PeriodIndex):
            raise TypeError(f'unsupported type {type(self.index).__name__}')
        new_obj = self.copy(deep=False)
        new_index = self.index.to_timestamp(freq=freq, how=how)
        setattr(new_obj, 'index', new_index)
        return new_obj

    def to_period(self, freq: Optional[str] = None, copy: Any = lib.no_default) -> Series:
        """
        Convert Series from DatetimeIndex to PeriodIndex.
        """
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError(f'unsupported type {type(self.index).__name__}')
        new_obj = self.copy(deep=False)
        new_index = self.index.to_period(freq=freq)
        setattr(new_obj, 'index', new_index)
        return new_obj

    # Axis properties
    _AXIS_ORDERS: List[str] = ['index']
    _AXIS_LEN: int = len(_AXIS_ORDERS)
    _info_axis_number: int = 0
    _info_axis_name: str = 'index'
    index = property(lambda self: self.__dict__.get("index"), doc="""
        The index (axis labels) of the Series.
    """)

    # End of Series class.
    
# Note: This code assumes that auxiliary functions and classes such as
#       _map_values, _update_inplace, _indexed_same, _constructor, and others are
#       defined elsewhere in the pandas code base.
