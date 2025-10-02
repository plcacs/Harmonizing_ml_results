from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Any, Tuple, Optional, Union
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import is_supported_dtype
from pandas.compat.numpy import function as nv
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import isna
from pandas.core import arraylike, missing, nanops, ops
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.strings.object_array import ObjectStringArrayMixin

if TYPE_CHECKING:
    from pandas._typing import (
        AxisInt,
        Dtype,
        FillnaOptions,
        InterpolateOptions,
        NpDtype,
        Scalar,
        Self,
        npt,
    )
    from pandas import Index


class NumpyExtensionArray(OpsMixin, NDArrayBackedExtensionArray, ObjectStringArrayMixin):
    """
    A pandas ExtensionArray for NumPy data.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    values : ndarray
        The NumPy ndarray to wrap. Must be 1-dimensional.
    copy : bool, default False
        Whether to copy `values`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    array : Create an array.
    Series.to_numpy : Convert a Series to a NumPy array.

    Examples
    --------
    >>> pd.arrays.NumpyExtensionArray(np.array([0, 1, 2, 3]))
    <NumpyExtensionArray>
    [0, 1, 2, 3]
    Length: 4, dtype: int64
    """
    _typ: str = 'npy_extension'
    __array_priority__: int = 1000
    _internal_fill_value: float = np.nan

    def __init__(self, values: np.ndarray, copy: bool = False) -> None:
        if isinstance(values, type(self)):
            values = values._ndarray
        if not isinstance(values, np.ndarray):
            raise ValueError(f"'values' must be a NumPy array, not {type(values).__name__}")
        if values.ndim == 0:
            raise ValueError('NumpyExtensionArray must be 1-dimensional.')
        if copy:
            values = values.copy()
        dtype: NumpyEADtype = NumpyEADtype(values.dtype)
        super().__init__(values, dtype)

    @classmethod
    def _from_sequence(
        cls,
        scalars: npt.ArrayLike,
        *,
        dtype: Optional[Dtype] = None,
        copy: bool = False
    ) -> NumpyExtensionArray:
        if isinstance(dtype, NumpyEADtype):
            dtype = dtype._dtype
        result: np.ndarray = np.asarray(scalars, dtype=dtype)
        if result.ndim > 1 and (not hasattr(scalars, 'dtype')) and (dtype is None or dtype == object):
            result = construct_1d_object_array_from_listlike(scalars)
        if copy and result is scalars:
            result = result.copy()
        return cls(result)

    @property
    def dtype(self) -> NumpyEADtype:
        return self._dtype

    def __array__(self, dtype: Optional[Any] = None, copy: Optional[bool] = None) -> np.ndarray:
        if copy is not None:
            return np.array(self._ndarray, dtype=dtype, copy=copy)
        return np.asarray(self._ndarray, dtype=dtype)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any
    ) -> Any:
        out = kwargs.get('out', ())
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result
        if 'out' in kwargs:
            return arraylike.dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        if method == 'reduce':
            result = arraylike.dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
            if result is not NotImplemented:
                return result
        inputs_converted: Tuple[Any, ...] = tuple(
            (x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in inputs)
        )
        if out:
            kwargs['out'] = tuple(
                (x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in out)
            )
        result = getattr(ufunc, method)(*inputs_converted, **kwargs)
        if ufunc.nout > 1:
            return tuple((type(self)(x) for x in result))
        elif method == 'at':
            return None
        elif method == 'reduce':
            if isinstance(result, np.ndarray):
                return type(self)(result)
            return result
        else:
            return type(self)(result)

    def astype(self, dtype: Any, copy: bool = True) -> NumpyExtensionArray | Any:
        dtype = pandas_dtype(dtype)
        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        result = astype_array(self._ndarray, dtype=dtype, copy=copy)
        return result

    def isna(self) -> np.ndarray:
        return isna(self._ndarray)

    def _validate_scalar(self, fill_value: Any) -> Any:
        if fill_value is None:
            fill_value = self.dtype.na_value
        return fill_value

    def _values_for_factorize(self) -> Tuple[np.ndarray, Any]:
        if self.dtype.kind in 'iub':
            fv: Any = None
        else:
            fv = np.nan
        return (self._ndarray, fv)

    def _pad_or_backfill(
        self,
        *,
        method: str,
        limit: Optional[int] = None,
        limit_area: Optional[str] = None,
        copy: bool = True
    ) -> NumpyExtensionArray:
        """
        ffill or bfill along axis=0.
        """
        if copy:
            out_data: np.ndarray = self._ndarray.copy()
        else:
            out_data = self._ndarray
        meth: str = missing.clean_fill_method(method)
        missing.pad_or_backfill_inplace(
            out_data.T,
            method=meth,
            axis=0,
            limit=limit,
            limit_area=limit_area
        )
        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

    def interpolate(
        self,
        *,
        method: str,
        axis: Optional[int] = 0,
        index: Optional[Index] = None,
        limit: Optional[int] = None,
        limit_direction: Optional[str] = None,
        limit_area: Optional[str] = None,
        copy: bool = True,
        **kwargs: Any
    ) -> NumpyExtensionArray:
        """
        See NDFrame.interpolate.__doc__.
        """
        if not self.dtype._is_numeric:
            raise TypeError(f'Cannot interpolate with {self.dtype} dtype')
        if not copy:
            out_data: np.ndarray = self._ndarray
        else:
            out_data = self._ndarray.copy()
        missing.interpolate_2d_inplace(
            out_data,
            method=method,
            axis=axis,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            **kwargs
        )
        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

    def any(
        self,
        *,
        axis: Optional[AxisInt] = None,
        out: Optional[npt.NDArray[Any]] = None,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Any:
        nv.validate_any((), {'out': out, 'keepdims': keepdims})
        result: Any = nanops.nanany(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def all(
        self,
        *,
        axis: Optional[AxisInt] = None,
        out: Optional[npt.NDArray[Any]] = None,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Any:
        nv.validate_all((), {'out': out, 'keepdims': keepdims})
        result: Any = nanops.nanall(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def min(
        self,
        *,
        axis: Optional[AxisInt] = None,
        skipna: bool = True,
        **kwargs: Any
    ) -> Any:
        nv.validate_min((), kwargs)
        result: Any = nanops.nanmin(values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def max(
        self,
        *,
        axis: Optional[AxisInt] = None,
        skipna: bool = True,
        **kwargs: Any
    ) -> Any:
        nv.validate_max((), kwargs)
        result: Any = nanops.nanmax(values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def sum(
        self,
        *,
        axis: Optional[AxisInt] = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any
    ) -> Any:
        nv.validate_sum((), kwargs)
        result: Any = nanops.nansum(self._ndarray, axis=axis, skipna=skipna, min_count=min_count)
        return self._wrap_reduction_result(axis, result)

    def prod(
        self,
        *,
        axis: Optional[AxisInt] = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any
    ) -> Any:
        nv.validate_prod((), kwargs)
        result: Any = nanops.nanprod(self._ndarray, axis=axis, skipna=skipna, min_count=min_count)
        return self._wrap_reduction_result(axis, result)

    def mean(
        self,
        *,
        axis: Optional[AxisInt] = None,
        dtype: Optional[Dtype] = None,
        out: Optional[Any] = None,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Any:
        nv.validate_mean((), {'dtype': dtype, 'out': out, 'keepdims': keepdims})
        result: Any = nanops.nanmean(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def median(
        self,
        *,
        axis: Optional[AxisInt] = None,
        out: Optional[Any] = None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Any:
        nv.validate_median((), {'out': out, 'overwrite_input': overwrite_input, 'keepdims': keepdims})
        result: Any = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def std(
        self,
        *,
        axis: Optional[AxisInt] = None,
        dtype: Optional[Dtype] = None,
        out: Optional[Any] = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Any:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='std')
        result: Any = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def var(
        self,
        *,
        axis: Optional[AxisInt] = None,
        dtype: Optional[Dtype] = None,
        out: Optional[Any] = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Any:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='var')
        result: Any = nanops.nanvar(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def sem(
        self,
        *,
        axis: Optional[AxisInt] = None,
        dtype: Optional[Dtype] = None,
        out: Optional[Any] = None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Any:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='sem')
        result: Any = nanops.nansem(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def kurt(
        self,
        *,
        axis: Optional[AxisInt] = None,
        dtype: Optional[Dtype] = None,
        out: Optional[Any] = None,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Any:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='kurt')
        result: Any = nanops.nankurt(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def skew(
        self,
        *,
        axis: Optional[AxisInt] = None,
        dtype: Optional[Dtype] = None,
        out: Optional[Any] = None,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Any:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='skew')
        result: Any = nanops.nanskew(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def to_numpy(
        self,
        dtype: Optional[NpDtype] = None,
        copy: bool = False,
        na_value: Any = lib.no_default
    ) -> np.ndarray:
        mask: np.ndarray = self.isna()
        if na_value is not lib.no_default and mask.any():
            result: np.ndarray = self._ndarray.copy()
            result[mask] = na_value
        else:
            result = self._ndarray
        result = np.asarray(result, dtype=dtype)
        if copy and result is self._ndarray:
            result = result.copy()
        return result

    def __invert__(self) -> NumpyExtensionArray:
        return type(self)(~self._ndarray)

    def __neg__(self) -> NumpyExtensionArray:
        return type(self)(-self._ndarray)

    def __pos__(self) -> NumpyExtensionArray:
        return type(self)(+self._ndarray)

    def __abs__(self) -> NumpyExtensionArray:
        return type(self)(abs(self._ndarray))

    def _cmp_method(self, other: Any, op: Any) -> Any:
        if isinstance(other, NumpyExtensionArray):
            other = other._ndarray
        other_prepared: Any = ops.maybe_prepare_scalar_for_op(other, (len(self),))
        pd_op = ops.get_array_op(op)
        other_prepared = ensure_wrapped_if_datetimelike(other_prepared)
        result: Any = pd_op(self._ndarray, other_prepared)
        if op in {divmod, ops.rdivmod}:
            a, b = result
            if isinstance(a, np.ndarray):
                return (self._wrap_ndarray_result(a), self._wrap_ndarray_result(b))
            return (a, b)
        if isinstance(result, np.ndarray):
            return self._wrap_ndarray_result(result)
        return result

    _arith_method = _cmp_method

    def _wrap_ndarray_result(self, result: np.ndarray) -> NumpyExtensionArray | Any:
        if result.dtype.kind == 'm' and is_supported_dtype(result.dtype):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray._simple_new(result, dtype=result.dtype)
        return type(self)(result)
