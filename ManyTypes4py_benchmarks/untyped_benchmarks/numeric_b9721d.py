from __future__ import annotations
import numbers
from typing import TYPE_CHECKING, Any
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import is_integer_dtype, is_string_dtype, pandas_dtype
from pandas.core.arrays.masked import BaseMaskedArray, BaseMaskedDtype
if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    import pyarrow
    from pandas._typing import DtypeObj, Self, npt
    from pandas.core.dtypes.dtypes import ExtensionDtype

class NumericDtype(BaseMaskedDtype):

    def __repr__(self):
        return f'{self.name}Dtype()'

    @cache_readonly
    def is_signed_integer(self):
        return self.kind == 'i'

    @cache_readonly
    def is_unsigned_integer(self):
        return self.kind == 'u'

    @property
    def _is_numeric(self):
        return True

    def __from_arrow__(self, array):
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow
        from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
        array_class = self.construct_array_type()
        pyarrow_type = pyarrow.from_numpy_dtype(self.type)
        if not array.type.equals(pyarrow_type) and (not pyarrow.types.is_null(array.type)):
            rt_dtype = pandas_dtype(array.type.to_pandas_dtype())
            if rt_dtype.kind not in 'iuf':
                raise TypeError(f'Expected array of {self} type, got {array.type} instead')
            array = array.cast(pyarrow_type)
        if isinstance(array, pyarrow.ChunkedArray):
            if array.num_chunks == 0:
                array = pyarrow.array([], type=array.type)
            else:
                array = array.combine_chunks()
        data, mask = pyarrow_array_to_numpy_and_mask(array, dtype=self.numpy_dtype)
        return array_class(data.copy(), ~mask, copy=False)

    @classmethod
    def _get_dtype_mapping(cls):
        raise AbstractMethodError(cls)

    @classmethod
    def _standardize_dtype(cls, dtype):
        """
        Convert a string representation or a numpy dtype to NumericDtype.
        """
        if isinstance(dtype, str) and dtype.startswith(('Int', 'UInt', 'Float')):
            dtype = dtype.lower()
        if not isinstance(dtype, NumericDtype):
            mapping = cls._get_dtype_mapping()
            try:
                dtype = mapping[np.dtype(dtype)]
            except KeyError as err:
                raise ValueError(f'invalid dtype specified {dtype}') from err
        return dtype

    @classmethod
    def _safe_cast(cls, values, dtype, copy):
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        """
        raise AbstractMethodError(cls)

def _coerce_to_data_and_mask(values, dtype, copy, dtype_cls, default_dtype):
    checker = dtype_cls._checker
    mask = None
    inferred_type = None
    if dtype is None and hasattr(values, 'dtype'):
        if checker(values.dtype):
            dtype = values.dtype
    if dtype is not None:
        dtype = dtype_cls._standardize_dtype(dtype)
    cls = dtype_cls.construct_array_type()
    if isinstance(values, cls):
        values, mask = (values._data, values._mask)
        if dtype is not None:
            values = values.astype(dtype.numpy_dtype, copy=False)
        if copy:
            values = values.copy()
            mask = mask.copy()
        return (values, mask, dtype, inferred_type)
    original = values
    if not copy:
        values = np.asarray(values)
    else:
        values = np.array(values, copy=copy)
    inferred_type = None
    if values.dtype == object or is_string_dtype(values.dtype):
        inferred_type = lib.infer_dtype(values, skipna=True)
        if inferred_type == 'boolean' and dtype is None:
            name = dtype_cls.__name__.strip('_')
            raise TypeError(f'{values.dtype} cannot be converted to {name}')
    elif values.dtype.kind == 'b' and checker(dtype):
        mask = np.zeros(len(values), dtype=np.bool_)
        if not copy:
            values = np.asarray(values, dtype=default_dtype)
        else:
            values = np.array(values, dtype=default_dtype, copy=copy)
    elif values.dtype.kind not in 'iuf':
        name = dtype_cls.__name__.strip('_')
        raise TypeError(f'{values.dtype} cannot be converted to {name}')
    if values.ndim != 1:
        raise TypeError('values must be a 1D list-like')
    if mask is None:
        if values.dtype.kind in 'iu':
            mask = np.zeros(len(values), dtype=np.bool_)
        elif values.dtype.kind == 'f':
            mask = np.isnan(values)
        else:
            mask = libmissing.is_numeric_na(values)
    else:
        assert len(mask) == len(values)
    if mask.ndim != 1:
        raise TypeError('mask must be a 1D list-like')
    if dtype is None:
        dtype = default_dtype
    else:
        dtype = dtype.numpy_dtype
    if is_integer_dtype(dtype) and values.dtype.kind == 'f' and (len(values) > 0):
        if mask.all():
            values = np.ones(values.shape, dtype=dtype)
        else:
            idx = np.nanargmax(values)
            if int(values[idx]) != original[idx]:
                inferred_type = lib.infer_dtype(original, skipna=True)
                if inferred_type not in ['floating', 'mixed-integer-float'] and (not mask.any()):
                    values = np.asarray(original, dtype=dtype)
                else:
                    values = np.asarray(original, dtype='object')
    if mask.any():
        values = values.copy()
        values[mask] = dtype_cls._internal_fill_value
    if inferred_type in ('string', 'unicode'):
        values = values.astype(dtype, copy=copy)
    else:
        values = dtype_cls._safe_cast(values, dtype, copy=False)
    return (values, mask, dtype, inferred_type)

class NumericArray(BaseMaskedArray):
    """
    Base class for IntegerArray and FloatingArray.
    """

    def __init__(self, values, mask, copy=False):
        checker = self._dtype_cls._checker
        if not (isinstance(values, np.ndarray) and checker(values.dtype)):
            descr = 'floating' if self._dtype_cls.kind == 'f' else 'integer'
            raise TypeError(f"values should be {descr} numpy array. Use the 'pd.array' function instead")
        if values.dtype == np.float16:
            raise TypeError('FloatingArray does not support np.float16 dtype.')
        super().__init__(values, mask, copy=copy)

    @cache_readonly
    def dtype(self):
        mapping = self._dtype_cls._get_dtype_mapping()
        return mapping[self._data.dtype]

    @classmethod
    def _coerce_to_array(cls, value, *, dtype, copy=False):
        dtype_cls = cls._dtype_cls
        default_dtype = dtype_cls._default_np_dtype
        values, mask, _, _ = _coerce_to_data_and_mask(value, dtype, copy, dtype_cls, default_dtype)
        return (values, mask)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype, copy=False):
        from pandas.core.tools.numeric import to_numeric
        scalars = to_numeric(strings, errors='raise', dtype_backend='numpy_nullable')
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)
    _HANDLED_TYPES = (np.ndarray, numbers.Number)