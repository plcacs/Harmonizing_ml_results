from __future__ import annotations
from typing import Any, Callable, Optional, Tuple, TYPE_CHECKING
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
    from pandas._typing import AxisInt, Dtype, Scalar, Self, npt
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
    _internal_fill_value: Any = np.nan

    def __init__(self, values: np.ndarray | NumpyExtensionArray, copy: bool = False) -> None:
        if isinstance(values, type(self)):
            values = values._ndarray
        if not isinstance(values, np.ndarray):
            raise ValueError(f"'values' must be a NumPy array, not {type(values).__name__}")
        if values.ndim == 0:
            raise ValueError('NumpyExtensionArray must be 1-dimensional.')
        if copy:
            values = values.copy()
        dtype = NumpyEADtype(values.dtype)
        super().__init__(values, dtype)

    @classmethod
    def _from_sequence(cls, scalars: Any, *, dtype: Optional[Dtype] = None, copy: bool = False) -> NumpyExtensionArray:
        if isinstance(dtype, NumpyEADtype):
            dtype = dtype._dtype
        result = np.asarray(scalars, dtype=dtype)
        if result.ndim > 1 and (not hasattr(scalars, 'dtype')) and (dtype is None or dtype == object):
            result = construct_1d_object_array_from_listlike(scalars)
        if copy and result is scalars:
            result = result.copy()
        return cls(result)

    @property
    def dtype(self) -> NumpyEADtype:
        return self._dtype

    def __array__(self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        if copy is not None:
            return np.array(self._ndarray, dtype=dtype, copy=copy)
        return np.asarray(self._ndarray, dtype=dtype)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
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
        inputs = tuple((x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in inputs))
        if out:
            kwargs['out'] = tuple((x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in out))
        result = getattr(ufunc, method)(*inputs, **kwargs)
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

    def astype(self, dtype: Dtype, copy: bool = True) -> Any:
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
            fv = None
        else:
            fv = np.nan
        return (self._ndarray, fv)

    def _pad_or_backfill(
        self, *, method: str, limit: Optional[int] = None, limit_area: Optional[str] = None, copy: bool = True
    ) -> NumpyExtensionArray:
        """
        ffill or bfill along axis=0.
        """
        if copy:
            out_data = self._ndarray.copy()
        else:
            out_data = self._ndarray
        meth = missing.clean_fill_method(method)
        missing.pad_or_backfill_inplace(out_data.T, method=meth, axis=0, limit=limit, limit_area=limit_area)
        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

    def interpolate(
        self,
        *,
        method: str,
        axis: int,
        index: Index,
        limit: Optional[int],
        limit_direction: str,
        limit_area: Optional[str],
        copy: bool,
        **kwargs: Any
    ) -> NumpyExtensionArray:
        """
        See NDFrame.interpolate.__doc__.
        """
        if not self.dtype._is_numeric:
            raise TypeError(f'Cannot interpolate with {self.dtype} dtype')
        if not copy:
            out_data = self._ndarray
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
        self, *, axis: Optional[int] = None, out: Optional[Any] = None, keepdims: bool = False, skipna: bool = True
    ) -> Any:
        nv.validate_any((), {'out': out, 'keepdims': keepdims})
        result = nanops.nanany(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def all(
        self, *, axis: Optional[int] = None, out: Optional[Any] = None, keepdims: bool = False, skipna: bool = True
    ) -> Any:
        nv.validate_all((), {'out': out, 'keepdims': keepdims})
        result = nanops.nanall(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def min(self, *, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any) -> Any:
        nv.validate_min((), kwargs)
        result = nanops.nanmin(values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def max(self, *, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any) -> Any:
        nv.validate_max((), kwargs)
        result = nanops.nanmax(values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def sum(
        self, *, axis: Optional[int] = None, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> Any:
        nv.validate_sum((), kwargs)
        result = nanops.nansum(self._ndarray, axis=axis, skipna=skipna, min_count=min_count)
        return self._wrap_reduction_result(axis, result)

    def prod(
        self, *, axis: Optional[int] = None, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> Any:
        nv.validate_prod((), kwargs)
        result = nanops.nanprod(self._ndarray, axis=axis, skipna=skipna, min_count=min_count)
        return self._wrap_reduction_result(axis, result)

    def mean(
        self,
        *,
        axis: Optional[int] = None,
        dtype: Optional[Any] = None,
        out: Optional[Any] = None,
        keepdims: bool = False,
        skipna: bool = True
    ) -> Any:
        nv.validate_mean((), {'dtype': dtype, 'out': out, 'keepdims': keepdims})
        result = nanops.nanmean(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def median(
        self, *, axis: Optional[int] = None, out: Optional[Any] = None, overwrite_input: bool = False, keepdims: bool = False, skipna: bool = True
    ) -> Any:
        nv.validate_median((), {'out': out, 'overwrite_input': overwrite_input, 'keepdims': keepdims})
        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def std(
        self, *, axis: Optional[int] = None, dtype: Optional[Any] = None, out: Optional[Any] = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True
    ) -> Any:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='std')
        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def var(
        self, *, axis: Optional[int] = None, dtype: Optional[Any] = None, out: Optional[Any] = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True
    ) -> Any:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='var')
        result = nanops.nanvar(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def sem(
        self, *, axis: Optional[int] = None, dtype: Optional[Any] = None, out: Optional[Any] = None, ddof: int = 1, keepdims: bool = False, skipna: bool = True
    ) -> Any:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='sem')
        result = nanops.nansem(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def kurt(
        self, *, axis: Optional[int] = None, dtype: Optional[Any] = None, out: Optional[Any] = None, keepdims: bool = False, skipna: bool = True
    ) -> Any:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='kurt')
        result = nanops.nankurt(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def skew(
        self, *, axis: Optional[int] = None, dtype: Optional[Any] = None, out: Optional[Any] = None, keepdims: bool = False, skipna: bool = True
    ) -> Any:
        nv.validate_stat_ddof_func((), {'dtype': dtype, 'out': out, 'keepdims': keepdims}, fname='skew')
        result = nanops.nanskew(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def to_numpy(
        self, dtype: Optional[np.dtype] = None, copy: bool = False, na_value: Any = lib.no_default
    ) -> np.ndarray:
        mask = self.isna()
        if na_value is not lib.no_default and mask.any():
            result = self._ndarray.copy()
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

    def _cmp_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Any:
        if isinstance(other, NumpyExtensionArray):
            other = other._ndarray
        other = ops.maybe_prepare_scalar_for_op(other, (len(self),))
        pd_op = ops.get_array_op(op)
        other = ensure_wrapped_if_datetimelike(other)
        result = pd_op(self._ndarray, other)
        if op is divmod or op is ops.rdivmod:
            a, b = result
            if isinstance(a, np.ndarray):
                return (self._wrap_ndarray_result(a), self._wrap_ndarray_result(b))
            return (a, b)
        if isinstance(result, np.ndarray):
            return self._wrap_ndarray_result(result)
        return result

    _arith_method = _cmp_method

    def _wrap_ndarray_result(self, result: np.ndarray) -> NumpyExtensionArray:
        if result.dtype.kind == 'm' and is_supported_dtype(result.dtype):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray._simple_new(result, dtype=result.dtype)
        return type(self)(result)