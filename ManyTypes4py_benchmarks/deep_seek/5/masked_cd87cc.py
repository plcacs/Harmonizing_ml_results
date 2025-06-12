from __future__ import annotations
from typing import (
    TYPE_CHECKING, Any, Literal, cast, overload, Union, Optional, Tuple, List, 
    Sequence, Dict, TypeVar, Generic, Callable, Iterator, Type, Set
)
import warnings
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas._libs.tslibs import is_supported_dtype
from pandas.compat import IS64, is_platform_windows
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    is_bool, is_integer_dtype, is_list_like, is_scalar, is_string_dtype, pandas_dtype
)
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
    from collections.abc import Callable, Iterator, Sequence
    from pandas import Series
    from pandas.core.arrays import BooleanArray, FloatingArray, IntegerArray
    from pandas._typing import (
        NumpySorter, NumpyValueArrayLike, ArrayLike, AstypeArg, AxisInt, DtypeObj, 
        FillnaOptions, InterpolateOptions, NpDtype, PositionalIndexer, Scalar, 
        ScalarIndexer, Self, SequenceIndexer, Shape, npt
    )
    from pandas._libs.missing import NAType
    from pandas.core.arrays import FloatingArray
    import pyarrow as pa

T = TypeVar('T', bound='BaseMaskedArray')
T_Dtype = TypeVar('T_Dtype', bound=BaseMaskedDtype)

class BaseMaskedArray(OpsMixin, ExtensionArray):
    """
    Base class for masked arrays (which use _data and _mask to store the data).
    numpy based
    """

    _data: np.ndarray
    _mask: np.ndarray
    _HANDLED_TYPES: Tuple[Type[Any], ...] = (np.ndarray,)

    @classmethod
    def _simple_new(
        cls: Type[T], 
        values: np.ndarray, 
        mask: np.ndarray
    ) -> T:
        result = BaseMaskedArray.__new__(cls)
        result._data = values
        result._mask = mask
        return result

    def __init__(
        self, 
        values: np.ndarray, 
        mask: np.ndarray, 
        copy: bool = False
    ) -> None:
        if not (isinstance(mask, np.ndarray) and mask.dtype == np.bool_:
            raise TypeError(
                "mask should be boolean numpy array. Use the 'pd.array' function instead"
            )
        if values.shape != mask.shape:
            raise ValueError('values.shape must match mask.shape')
        if copy:
            values = values.copy()
            mask = mask.copy()
        self._data = values
        self._mask = mask

    @classmethod
    def _from_sequence(
        cls: Type[T], 
        scalars: Sequence[Any], 
        *, 
        dtype: Optional[DtypeObj] = None, 
        copy: bool = False
    ) -> T:
        values, mask = cls._coerce_to_array(scalars, dtype=dtype, copy=copy)
        return cls(values, mask)

    @classmethod
    @doc(ExtensionArray._empty)
    def _empty(
        cls: Type[T], 
        shape: Shape, 
        dtype: DtypeObj
    ) -> T:
        dtype = cast(BaseMaskedDtype, dtype)
        values = np.empty(shape, dtype=dtype.type)
        values.fill(dtype._internal_fill_value)
        mask = np.ones(shape, dtype=bool)
        result = cls(values, mask)
        if not isinstance(result, cls) or dtype != result.dtype:
            raise NotImplementedError(
                f"Default 'empty' implementation is invalid for dtype='{dtype}'"
            )
        return result

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str]:
        return str

    @property
    def dtype(self) -> BaseMaskedDtype:
        raise AbstractMethodError(self)

    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any:
        ...

    @overload
    def __getitem__(self, item: SequenceIndexer) -> T:
        ...

    def __getitem__(
        self, 
        item: Union[ScalarIndexer, SequenceIndexer]
    ) -> Union[T, Any]:
        item = check_array_indexer(self, item)
        newmask = self._mask[item]
        if is_bool(newmask):
            if newmask:
                return self.dtype.na_value
            return self._data[item]
        return self._simple_new(self._data[item], newmask)

    def _pad_or_backfill(
        self, 
        *, 
        method: str, 
        limit: Optional[int] = None, 
        limit_area: Optional[str] = None, 
        copy: bool = True
    ) -> T:
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
    def fillna(
        self, 
        value: Any, 
        limit: Optional[int] = None, 
        copy: bool = True
    ) -> T:
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
    def _coerce_to_array(
        cls, 
        values: Sequence[Any], 
        *, 
        dtype: Optional[DtypeObj], 
        copy: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise AbstractMethodError(cls)

    def _validate_setitem_value(self, value: Any) -> Any:
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

    def __setitem__(
        self, 
        key: Union[ScalarIndexer, SequenceIndexer], 
        value: Any
    ) -> None:
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

    def __contains__(self, key: Any) -> bool:
        if isna(key) and key is not self.dtype.na_value:
            if self._data.dtype.kind == 'f' and lib.is_float(key):
                return bool((np.isnan(self._data) & ~self._mask).any())
        return bool(super().__contains__(key))

    def __iter__(self) -> Iterator[Any]:
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

    def __len__(self) -> int:
        return len(self._data)

    @property
    def shape(self) -> Shape:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    def swapaxes(self, axis1: int, axis2: int) -> T:
        data = self._data.swapaxes(axis1, axis2)
        mask = self._mask.swapaxes(axis1, axis2)
        return self._simple_new(data, mask)

    def delete(self, loc: Union[int, Sequence[int]], axis: int = 0) -> T:
        data = np.delete(self._data, loc, axis=axis)
        mask = np.delete(self._mask, loc, axis=axis)
        return self._simple_new(data, mask)

    def reshape(self, *args: Any, **kwargs: Any) -> T:
        data = self._data.reshape(*args, **kwargs)
        mask = self._mask.reshape(*args, **kwargs)
        return self._simple_new(data, mask)

    def ravel(self, *args: Any, **kwargs: Any) -> T:
        data = self._data.ravel(*args, **kwargs)
        mask = self._mask.ravel(*args, **kwargs)
        return type(self)(data, mask)

    @property
    def T(self) -> T:
        return self._simple_new(self._data.T, self._mask.T)

    def round(self, decimals: int = 0, *args: Any, **kwargs: Any) -> T:
        """
        Round each value in the array a to the given number of decimals.
        """
        if self.dtype.kind == 'b':
            return self
        nv.validate_round(args, kwargs)
        values = np.round(self._data, decimals=decimals, **kwargs)
        return self._maybe_mask_result(values, self._mask.copy())

    def __invert__(self) -> T:
        return self._simple_new(~self._data, self._mask.copy())

    def __neg__(self) -> T:
        return self._simple_new(-self._data, self._mask.copy())

    def __pos__(self) -> T:
        return self.copy()

    def __abs__(self) -> T:
        return self._simple_new(abs(self._data), self._mask.copy())

    def _values_for_json(self) -> np.ndarray:
        return np.asarray(self, dtype=object)

    def to_numpy(
        self,
        dtype: Optional[npt.DTypeLike] = None,
        copy: bool = False,
        na_value: Any = lib.no_default,
    ) -> np.ndarray:
        hasna = self._hasna
        dtype, na_value = to_numpy_dtype_inference(self, dtype, na_value, hasna)
        if dtype is None:
            dtype = object
        if hasna:
            if dtype != object and (not is_string_dtype(dtype)) and (na_value is libmissing.NA):
                raise ValueError(
                    f"cannot convert to '{dtype}'-dtype NumPy array with missing values. "
                    "Specify an appropriate 'na_value' for this dtype."
                )
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
    def tolist(self) -> List[Any]:
        if self.ndim > 1:
            return [x.tolist() for x in self]
        dtype = None if self._hasna else self._data.dtype
        return self.to_numpy(dtype=dtype, na_value=libmissing.NA).tolist()

    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray:
        ...

    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray:
        ...

    @overload
    def astype(self, dtype: BaseMaskedDtype, copy: bool = ...) -> T:
        ...

    def astype(
        self, 
        dtype: Union[npt.DTypeLike, DtypeObj], 
        copy: bool = True
    ) -> Union[np.ndarray, ExtensionArray, T]:
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

    def __array__(
        self, 
        dtype: Optional[npt.DTypeLike] = None, 
        copy: Optional[bool] = None
    ) -> np.ndarray:
        if copy is False:
            if not self._hasna:
                return np.array(self._data, dtype=dtype, copy=copy)
            raise ValueError('Unable to avoid copy while creating an array as requested.')
        if copy is None:
            copy = False
        return self.to_numpy(dtype=dtype, copy=copy)

    def __array_ufunc__(
        self, 
        ufunc: np.ufunc, 
        method: str, 
        *inputs: Any, 
        **kwargs: Any
    ) -> Any:
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (BaseMaskedArray,)):
                return NotImplemented
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result
        if 'out' in kwargs:
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )
        if method == 'reduce':
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
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

        def reconstruct(x: np.ndarray) -> Any:
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
                    x = x.