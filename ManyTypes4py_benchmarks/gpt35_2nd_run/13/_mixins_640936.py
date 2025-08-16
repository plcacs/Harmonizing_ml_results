from __future__ import annotations
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, cast, overload
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
    ...

class NDArrayBackedExtensionArray(NDArrayBacked, ExtensionArray):
    ...

    def _box_func(self, x: Any) -> Any:
        ...

    def _validate_scalar(self, value: Any) -> None:
        ...

    def view(self, dtype: Dtype = None) -> NDArrayBackedExtensionArray:
        ...

    def take(self, indices: ArrayLike, *, allow_fill: bool = False, fill_value: Any = None, axis: AxisInt = 0) -> NDArrayBackedExtensionArray:
        ...

    def equals(self, other: NDArrayBackedExtensionArray) -> bool:
        ...

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: NDArrayBackedExtensionArray) -> NDArrayBackedExtensionArray:
        ...

    def _values_for_argsort(self) -> np.ndarray:
        ...

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        ...

    def _hash_pandas_object(self, *, encoding: str, hash_key: Any, categorize: bool) -> Any:
        ...

    def argmin(self, axis: AxisInt = 0, skipna: bool = True) -> int:
        ...

    def argmax(self, axis: AxisInt = 0, skipna: bool = True) -> int:
        ...

    def unique(self) -> NDArrayBackedExtensionArray:
        ...

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[NDArrayBackedExtensionArray], axis: AxisInt = 0) -> NDArrayBackedExtensionArray:
        ...

    def searchsorted(self, value: Any, side: Literal['left', 'right'] = 'left', sorter: NumpySorter = None) -> np.ndarray:
        ...

    def shift(self, periods: int = 1, fill_value: Any = None) -> NDArrayBackedExtensionArray:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def _pad_or_backfill(self, *, method: str, limit: int = None, limit_area: Any = None, copy: bool = True) -> NDArrayBackedExtensionArray:
        ...

    def fillna(self, value: Any, limit: int = None, copy: bool = True) -> NDArrayBackedExtensionArray:
        ...

    def _wrap_reduction_result(self, axis: AxisInt, result: Any) -> Any:
        ...

    def _putmask(self, mask: np.ndarray, value: Any) -> None:
        ...

    def _where(self, mask: np.ndarray, value: Any) -> NDArrayBackedExtensionArray:
        ...

    def insert(self, loc: int, item: Any) -> NDArrayBackedExtensionArray:
        ...

    def value_counts(self, dropna: bool = True) -> Series:
        ...

    def _quantile(self, qs: ArrayLike, interpolation: str) -> NDArrayBackedExtensionArray:
        ...

    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype) -> NDArrayBackedExtensionArray:
        ...
