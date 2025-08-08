from __future__ import annotations
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, cast, overload, Union
import numpy as np
from pandas._libs import lib
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import is_supported_dtype
from pandas._typing import ArrayLike, AxisInt, Dtype, F, FillnaOptions, PositionalIndexer2D, PositionalIndexerTuple, ScalarIndexer, Self, SequenceIndexer, Shape, TakeIndexer, npt
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._validators import validate_bool_kwarg, validate_insert_loc
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype, ExtensionDtype, PeriodDtype
from pandas.core.dtypes.missing import array_equivalent
from pandas.core import missing
from pandas.core.algorithms import take, unique, value_counts_internal as value_counts
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays.base import ExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.sorting import nargminmax
if TYPE_CHECKING:
    from collections.abc import Sequence
    from pandas._typing import NumpySorter, NumpyValueArrayLike
    from pandas import Series

def ravel_compat(meth: F) -> F:
    @wraps(meth)
    def method(self, *args, **kwargs) -> Any:
        if self.ndim == 1:
            return meth(self, *args, **kwargs)
        flags = self._ndarray.flags
        flat = self.ravel('K')
        result = meth(flat, *args, **kwargs)
        order = 'F' if flags.f_contiguous else 'C'
        return result.reshape(self.shape, order=order)
    return cast(F, method)

class NDArrayBackedExtensionArray(NDArrayBacked, ExtensionArray):
    def _box_func(self, x: Any) -> Any:
        return x

    def _validate_scalar(self, value: Any) -> None:
        raise AbstractMethodError(self)

    def view(self, dtype: Union[type, Dtype, None] = None) -> ExtensionArray:
        ...

    def take(self, indices: ArrayLike, *, allow_fill: bool = False, fill_value: Any = None, axis: AxisInt = 0) -> ExtensionArray:
        ...

    def equals(self, other: Any) -> bool:
        ...

    @classmethod
    def _from_factorized(cls, values: NDArrayBacked, original: NDArrayBacked) -> NDArrayBacked:
        ...

    def _values_for_argsort(self) -> NDArrayBacked:
        ...

    def _values_for_factorize(self) -> Tuple[NDArrayBacked, Any]:
        ...

    def _hash_pandas_object(self, *, encoding: str, hash_key: Any, categorize: bool) -> Any:
        ...

    def argmin(self, axis: AxisInt = 0, skipna: bool = True) -> int:
        ...

    def argmax(self, axis: AxisInt = 0, skipna: bool = True) -> int:
        ...

    def unique(self) -> NDArrayBacked:
        ...

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[NDArrayBacked], axis: AxisInt = 0) -> NDArrayBacked:
        ...

    def searchsorted(self, value: Any, side: Literal['left', 'right'] = 'left', sorter: NumpySorter = None) -> NDArrayBacked:
        ...

    def shift(self, periods: int = 1, fill_value: Any = None) -> NDArrayBacked:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def fillna(self, value: Any, limit: int = None, copy: bool = True) -> NDArrayBacked:
        ...

    def _wrap_reduction_result(self, axis: AxisInt, result: Any) -> Any:
        ...

    def _putmask(self, mask: np.ndarray[bool], value: Any) -> None:
        ...

    def _where(self, mask: np.ndarray[bool], value: Any) -> NDArrayBacked:
        ...

    def insert(self, loc: int, item: Any) -> NDArrayBacked:
        ...

    def value_counts(self, dropna: bool = True) -> Series:
        ...

    def _quantile(self, qs: ArrayLike, interpolation: str) -> NDArrayBacked:
        ...

    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype) -> NDArrayBacked:
        ...
