from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, cast, overload, Sequence, Tuple, Optional, Union, List
import warnings
import numpy as np
from numpy.typing import NDArray
from pandas._libs import lib, missing as libmissing
from pandas._libs.tslibs import is_supported_dtype
from pandas.compat import IS64, is_platform_windows
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import is_bool, is_integer_dtype, is_list_like, is_scalar, is_string_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import BaseMaskedDtype
from pandas.core.dtypes.missing import array_equivalent, is_valid_na_for_dtype, isna, notna
from pandas.core import algorithms as algos, arraylike, missing, nanops, ops
from pandas.core.algorithms import factorize_array, isin, map_array, mode, take
from pandas.core.array_algos import masked_accumulations, masked_reductions
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import ExtensionArray
from pandas.core.construction import array as pd_array, ensure_wrapped_if_datetimelike, extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import invalid_comparison
from pandas.core.util.hashing import hash_array
if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pandas import Series
    from pandas.core.arrays import BooleanArray, FloatingArray, IntegerArray
    from pandas._typing import NumpySorter, NumpyValueArrayLike, ArrayLike, AstypeArg, AxisInt, DtypeObj, FillnaOptions, InterpolateOptions, NpDtype, PositionalIndexer, Scalar, ScalarIndexer, Self, SequenceIndexer, Shape, npt
    from pandas._libs.missing import NAType
from pandas.compat.numpy import function as nv

class BaseMaskedArray(OpsMixin, ExtensionArray):
    """
    Base class for masked arrays (which use _data and _mask to store the data).

    numpy based
    """
    _data: np.ndarray
    _mask: NDArray[np.bool_]

    @classmethod
    def _simple_new(cls, values, mask):
        result = BaseMaskedArray.__new__(cls)
        result._data = values
        result._mask = mask
        return result

    def __init__(self, values, mask, copy=False):
        if not (isinstance(mask, np.ndarray) and mask.dtype == np.bool_):
            raise TypeError("mask should be boolean numpy array. Use the 'pd.array' function instead")
        if values.shape != mask.shape:
            raise ValueError('values.shape must match mask.shape')
        if copy:
            values = values.copy()
            mask = mask.copy()
        self._data = values
        self._mask = mask

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Optional[DtypeObj]=None, copy: bool=False):
        values, mask = cls._coerce_to_array(scalars, dtype=dtype, copy=copy)
        return cls(values, mask)

    @classmethod
    @doc(ExtensionArray._empty)
    def _empty(cls, shape, dtype):
        dtype = cast(BaseMaskedDtype, dtype)
        values: np.ndarray = np.empty(shape, dtype=dtype.type)
        values.fill(dtype._internal_fill_value)
        mask = np.ones(shape, dtype=bool)
        result = cls(values, mask)
        if not isinstance(result, cls) or dtype != result.dtype:
            raise NotImplementedError(f"Default 'empty' implementation is invalid for dtype='{dtype}'")
        return result

    def _formatter(self, boxed=False):
        return str

    @property
    def dtype(self):
        raise AbstractMethodError(self)

    @overload
    def __getitem__(self, item):
        ...

    @overload
    def __getitem__(self, item):
        ...

    def __getitem__(self, item):
        item = check_array_indexer(self, item)
        newmask = self._mask[item]
        if is_bool(newmask):
            if newmask:
                return self.dtype.na_value
            return self._data[item]
        return self._simple_new(self._data[item], newmask)

    def _pad_or_backfill(self, *, method: FillnaOptions, limit: Optional[int]=None, limit_area: Optional[Literal['inside', 'outside']]=None, copy: bool=True):
        mask = self._mask
        if mask.any():
            func = missing.get_fill_func(method, ndim=self.ndim)
            npvalues = self._data.T
            new_mask = mask.T
            if copy:
                npvalues = npvalues.copy()
                new_mask = new_mask.copy()
            elif limit_area is not None:
                mask = mask.copy()
            func(npvalues, limit=limit, mask=new_mask)
            if limit_area is not None and (not mask.all()):
                mask = mask.T
                neg_mask = ~mask
                first = neg_mask.argmax()
                last = len(neg_mask) - neg_mask[::-1].argmax() - 1
                if limit_area == 'inside':
                    new_mask[:first] |= mask[:first]
                    new_mask[last + 1:] |= mask[last + 1:]
                elif limit_area == 'outside':
                    new_mask[first + 1:last] |= mask[first + 1:last]
            if copy:
                return self._simple_new(npvalues.T, new_mask.T)
            else:
                return self
        elif copy:
            new_values = self.copy()
        else:
            new_values = self
        return new_values

    @doc(ExtensionArray.fillna)
    def fillna(self, value, limit=None, copy=True):
        mask = self._mask
        if limit is not None and limit < len(self):
            modify = mask.cumsum() > limit
            if modify.any():
                mask = mask.copy()
                mask[modify] = False
        value = missing.check_value_size(value, mask, len(self))
        if mask.any():
            if copy:
                new_values = self.copy()
            else:
                new_values = self[:]
            new_values[mask] = value
        elif copy:
            new_values = self.copy()
        else:
            new_values = self[:]
        return new_values

    @classmethod
    def _coerce_to_array(cls, values, *, dtype: DtypeObj, copy: bool=False):
        raise AbstractMethodError(cls)

    def _validate_setitem_value(self, value):
        """
        Check if we have a scalar that we can cast losslessly.

        Raises
        ------
        TypeError
        """
        kind = self.dtype.kind
        if kind == 'b':
            if lib.is_bool(value):
                return value
        elif kind == 'f':
            if lib.is_integer(value) or lib.is_float(value):
                return value
        elif lib.is_integer(value) or (lib.is_float(value) and value.is_integer()):
            return value
        raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")

    def __setitem__(self, key, value):
        key = check_array_indexer(self, key)
        if is_scalar(value):
            if is_valid_na_for_dtype(value, self.dtype):
                self._mask[key] = True
            else:
                value = self._validate_setitem_value(value)
                self._data[key] = value
                self._mask[key] = False
            return
        value, mask = self._coerce_to_array(value, dtype=self.dtype)
        self._data[key] = value
        self._mask[key] = mask

    def __contains__(self, key):
        if isna(key) and key is not self.dtype.na_value:
            if self._data.dtype.kind == 'f' and lib.is_float(key):
                return bool((np.isnan(self._data) & ~self._mask).any())
        return bool(super().__contains__(key))

    def __iter__(self):
        if self.ndim == 1:
            if not self._hasna:
                for val in self._data:
                    yield val
            else:
                na_value = self.dtype.na_value
                for isna_, val in zip(self._mask, self._data):
                    if isna_:
                        yield na_value
                    else:
                        yield val
        else:
            for i in range(len(self)):
                yield self[i]

    def __len__(self):
        return len(self._data)

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    def swapaxes(self, axis1, axis2):
        data = self._data.swapaxes(axis1, axis2)
        mask = self._mask.swapaxes(axis1, axis2)
        return self._simple_new(data, mask)

    def delete(self, loc, axis=0):
        data = np.delete(self._data, loc, axis=axis)
        mask = np.delete(self._mask, loc, axis=axis)
        return self._simple_new(data, mask)

    def reshape(self, *args: Any, **kwargs: Any):
        data = self._data.reshape(*args, **kwargs)
        mask = self._mask.reshape(*args, **kwargs)
        return self._simple_new(data, mask)

    def ravel(self, *args: Any, **kwargs: Any):
        data = self._data.ravel(*args, **kwargs)
        mask = self._mask.ravel(*args, **kwargs)
        return type(self)(data, mask)

    @property
    def T(self):
        return self._simple_new(self._data.T, self._mask.T)

    def round(self, decimals=0, *args: Any, **kwargs: Any):
        """
        Round each value in the array a to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect but might be
            accepted for compatibility with NumPy.

        Returns
        -------
        NumericArray
            Rounded values of the NumericArray.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.
        Series.round : Round values of a Series.
        """
        if self.dtype.kind == 'b':
            return self
        nv.validate_round(args, kwargs)
        values = np.round(self._data, decimals=decimals, **kwargs)
        return self._maybe_mask_result(values, self._mask.copy())

    def __invert__(self):
        return self._simple_new(~self._data, self._mask.copy())

    def __neg__(self):
        return self._simple_new(-self._data, self._mask.copy())

    def __pos__(self):
        return self.copy()

    def __abs__(self):
        return self._simple_new(abs(self._data), self._mask.copy())

    def _values_for_json(self):
        return np.asarray(self, dtype=object)

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default):
        """
        Convert to a NumPy Array.

        By default converts to an object-dtype NumPy array. Specify the `dtype` and
        `na_value` keywords to customize the conversion.

        Parameters
        ----------
        dtype : dtype, default object
            The numpy dtype to convert to.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            the array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary. This is typically
            only possible when no missing values are present and `dtype`
            is the equivalent numpy dtype.
        na_value : scalar, optional
             Scalar missing value indicator to use in numpy array. Defaults
             to the native missing value indicator of this array (pd.NA).

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        An object-dtype is the default result

        >>> a = pd.array([True, False, pd.NA], dtype="boolean")
        >>> a.to_numpy()
        array([True, False, <NA>], dtype=object)

        When no missing values are present, an equivalent dtype can be used.

        >>> pd.array([True, False], dtype="boolean").to_numpy(dtype="bool")
        array([ True, False])
        >>> pd.array([1, 2], dtype="Int64").to_numpy("int64")
        array([1, 2])

        However, requesting such dtype will raise a ValueError if
        missing values are present and the default missing value :attr:`NA`
        is used.

        >>> a = pd.array([True, False, pd.NA], dtype="boolean")
        >>> a
        <BooleanArray>
        [True, False, <NA>]
        Length: 3, dtype: boolean

        >>> a.to_numpy(dtype="bool")
        Traceback (most recent call last):
        ...
        ValueError: cannot convert to bool numpy array in presence of missing values

        Specify a valid `na_value` instead

        >>> a.to_numpy(dtype="bool", na_value=False)
        array([ True, False, False])
        """
        hasna = self._hasna
        dtype, na_value = to_numpy_dtype_inference(self, dtype, na_value, hasna)
        if dtype is None:
            dtype = object
        if hasna:
            if dtype != object and (not is_string_dtype(dtype)) and (na_value is libmissing.NA):
                raise ValueError(f"cannot convert to '{dtype}'-dtype NumPy array with missing values. Specify an appropriate 'na_value' for this dtype.")
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                data = self._data.astype(dtype)
            data[self._mask] = na_value
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                data = self._data.astype(dtype, copy=copy)
        return data

    @doc(ExtensionArray.tolist)
    def tolist(self):
        if self.ndim > 1:
            return [x.tolist() for x in self]
        dtype = None if self._hasna else self._data.dtype
        return self.to_numpy(dtype=dtype, na_value=libmissing.NA).tolist()

    @overload
    def astype(self, dtype, copy=...):
        ...

    @overload
    def astype(self, dtype, copy=...):
        ...

    @overload
    def astype(self, dtype, copy=...):
        ...

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        if isinstance(dtype, BaseMaskedDtype):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                data = self._data.astype(dtype.numpy_dtype, copy=copy)
            mask = self._mask if data is self._data else self._mask.copy()
            cls = dtype.construct_array_type()
            return cls(data, mask, copy=False)
        if isinstance(dtype, ExtensionDtype):
            eacls = dtype.construct_array_type()
            return eacls._from_sequence(self, dtype=dtype, copy=copy)
        na_value: Union[float, np.datetime64, lib.NoDefault]
        if dtype.kind == 'f':
            na_value = np.nan
        elif dtype.kind == 'M':
            na_value = np.datetime64('NaT')
        else:
            na_value = lib.no_default
        if dtype.kind in 'iu' and self._hasna:
            raise ValueError('cannot convert NA to integer')
        if dtype.kind == 'b' and self._hasna:
            raise ValueError('cannot convert float NaN to bool')
        data = self.to_numpy(dtype=dtype, na_value=na_value, copy=copy)
        return data
    __array_priority__ = 1000

    def __array__(self, dtype=None, copy=None):
        """
        the array interface, return my values
        We return an object array here to preserve our scalar values
        """
        if copy is False:
            if not self._hasna:
                return np.array(self._data, dtype=dtype, copy=copy)
            raise ValueError('Unable to avoid copy while creating an array as requested.')
        if copy is None:
            copy = False
        return self.to_numpy(dtype=dtype, copy=copy)
    _HANDLED_TYPES: Tuple[type, ...]

    def __array_ufunc__(self, ufunc, method, *inputs: Any, **kwargs: Any):
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (BaseMaskedArray,)):
                return NotImplemented
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result
        if 'out' in kwargs:
            return arraylike.dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        if method == 'reduce':
            result = arraylike.dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
            if result is not NotImplemented:
                return result
        mask = np.zeros(len(self), dtype=bool)
        inputs2 = []
        for x in inputs:
            if isinstance(x, BaseMaskedArray):
                mask |= x._mask
                inputs2.append(x._data)
            else:
                inputs2.append(x)

        def reconstruct(x):
            from pandas.core.arrays import BooleanArray, FloatingArray, IntegerArray
            if x.dtype.kind == 'b':
                m = mask.copy()
                return BooleanArray(x, m)
            elif x.dtype.kind in 'iu':
                m = mask.copy()
                return IntegerArray(x, m)
            elif x.dtype.kind == 'f':
                m = mask.copy()
                if x.dtype == np.float16:
                    x = x.astype(np.float32)
                return FloatingArray(x, m)
            else:
                x[mask] = np.nan
            return x
        result = getattr(ufunc, method)(*inputs2, **kwargs)
        if ufunc.nout > 1:
            return tuple((reconstruct(x) for x in result))
        elif method == 'reduce':
            if self._mask.any():
                return self._na_value
            return result
        else:
            return reconstruct(result)

    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow as pa
        return pa.array(self._data, mask=self._mask, type=type)

    @property
    def _hasna(self):
        return self._mask.any()

    def _propagate_mask(self, mask, other):
        if mask is None:
            mask = self._mask.copy()
            if other is libmissing.NA:
                mask = mask | True
            elif is_list_like(other) and len(other) == len(mask):
                mask = mask | isna(other)
        else:
            mask = self._mask | mask
        return mask

    def _arith_method(self, other, op):
        op_name = op.__name__
        omask = None
        if not hasattr(other, 'dtype') and is_list_like(other) and (len(other) == len(self)):
            other = pd_array(other)
            other = extract_array(other, extract_numpy=True)
        if isinstance(other, BaseMaskedArray):
            other, omask = (other._data, other._mask)
        elif is_list_like(other):
            if not isinstance(other, ExtensionArray):
                other = np.asarray(other)
            if other.ndim > 1:
                raise NotImplementedError('can only perform ops with 1-d structures')
        other = ops.maybe_prepare_scalar_for_op(other, (len(self),))
        pd_op = ops.get_array_op(op)
        other = ensure_wrapped_if_datetimelike(other)
        if op_name in {'pow', 'rpow'} and isinstance(other, np.bool_):
            other = bool(other)
        mask = self._propagate_mask(omask, other)
        if other is libmissing.NA:
            result = np.ones_like(self._data)
            if self.dtype.kind == 'b':
                if op_name in {'floordiv', 'rfloordiv', 'pow', 'rpow', 'truediv', 'rtruediv'}:
                    raise NotImplementedError(f"operator '{op_name}' not implemented for bool dtypes")
                if op_name in {'mod', 'rmod'}:
                    dtype = 'int8'
                else:
                    dtype = 'bool'
                result = result.astype(dtype)
            elif 'truediv' in op_name and self.dtype.kind != 'f':
                result = result.astype(np.float64)
        else:
            if self.dtype.kind in 'iu' and op_name in ['floordiv', 'mod']:
                pd_op = op
            with np.errstate(all='ignore'):
                result = pd_op(self._data, other)
        if op_name == 'pow':
            mask = np.where((self._data == 1) & ~self._mask, False, mask)
            if omask is not None:
                mask = np.where((other == 0) & ~omask, False, mask)
            elif other is not libmissing.NA:
                mask = np.where(other == 0, False, mask)
        elif op_name == 'rpow':
            if omask is not None:
                mask = np.where((other == 1) & ~omask, False, mask)
            elif other is not libmissing.NA:
                mask = np.where(other == 1, False, mask)
            mask = np.where((self._data == 0) & ~self._mask, False, mask)
        return self._maybe_mask_result(result, mask)
    _logical_method = _arith_method

    def _cmp_method(self, other, op):
        from pandas.core.arrays import BooleanArray
        mask = None
        if isinstance(other, BaseMaskedArray):
            other, mask = (other._data, other._mask)
        elif is_list_like(other):
            other = np.asarray(other)
            if other.ndim > 1:
                raise NotImplementedError('can only perform ops with 1-d structures')
            if len(self) != len(other):
                raise ValueError('Lengths must match to compare')
        if other is libmissing.NA:
            result = np.zeros(self._data.shape, dtype='bool')
            mask = np.ones(self._data.shape, dtype='bool')
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'elementwise', FutureWarning)
                warnings.filterwarnings('ignore', 'elementwise', DeprecationWarning)
                method = getattr(self._data, f'__{op.__name__}__')
                result = method(other)
                if result is NotImplemented:
                    result = invalid_comparison(self._data, other, op)
        mask = self._propagate_mask(mask, other)
        return BooleanArray(result, mask, copy=False)

    def _maybe_mask_result(self, result, mask):
        """
        Parameters
        ----------
        result : array-like or tuple[array-like]
        mask : array-like bool
        """
        if isinstance(result, tuple):
            div, mod = result
            return (self._maybe_mask_result(div, mask), self._maybe_mask_result(mod, mask))
        if result.dtype.kind == 'f':
            from pandas.core.arrays import FloatingArray
            return FloatingArray(result, mask, copy=False)
        elif result.dtype.kind == 'b':
            from pandas.core.arrays import BooleanArray
            return BooleanArray(result, mask, copy=False)
        elif lib.is_np_dtype(result.dtype, 'm') and is_supported_dtype(result.dtype):
            from pandas.core.arrays import TimedeltaArray
            result[mask] = result.dtype.type('NaT')
            if not isinstance(result, TimedeltaArray):
                return TimedeltaArray._simple_new(result, dtype=result.dtype)
            return result
        elif result.dtype.kind in 'iu':
            from pandas.core.arrays import IntegerArray
            return IntegerArray(result, mask, copy=False)
        else:
            result[mask] = np.nan
            return result

    def isna(self):
        return self._mask.copy()

    @property
    def _na_value(self):
        return self.dtype.na_value

    @property
    def nbytes(self):
        return self._data.nbytes + self._mask.nbytes

    @classmethod
    def _concat_same_type(cls, to_concat, axis=0):
        data = np.concatenate([x._data for x in to_concat], axis=axis)
        mask = np.concatenate([x._mask for x in to_concat], axis=axis)
        return cls(data, mask)

    def _hash_pandas_object(self, *, encoding: str, hash_key: str, categorize: bool):
        hashed_array = hash_array(self._data, encoding=encoding, hash_key=hash_key, categorize=categorize)
        hashed_array[self.isna()] = hash(self.dtype.na_value)
        return hashed_array

    def take(self, indexer, *, allow_fill: bool=False, fill_value: Optional[Scalar]=None, axis: AxisInt=0):
        data_fill_value = self.dtype._internal_fill_value if isna(fill_value) else fill_value
        result = take(self._data, indexer, fill_value=data_fill_value, allow_fill=allow_fill, axis=axis)
        mask = take(self._mask, indexer, fill_value=True, allow_fill=allow_fill, axis=axis)
        if allow_fill and notna(fill_value):
            fill_mask = np.asarray(indexer) == -1
            result[fill_mask] = fill_value
            mask = mask ^ fill_mask
        return self._simple_new(result, mask)

    def isin(self, values):
        from pandas.core.arrays import BooleanArray
        values_arr = np.asarray(values)
        result = isin(self._data, values_arr)
        if self._hasna:
            values_have_NA = values_arr.dtype == object and any((val is self.dtype.na_value for val in values_arr))
            result[self._mask] = values_have_NA
        mask = np.zeros(self._data.shape, dtype=bool)
        return BooleanArray(result, mask, copy=False)