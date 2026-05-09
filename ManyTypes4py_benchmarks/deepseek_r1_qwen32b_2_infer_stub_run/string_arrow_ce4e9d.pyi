from __future__ import annotations
import numpy as np
from typing import Any, Callable, List, Optional, Sequence, Union
from pandas._typing import (
    ArrayLike,
    Dtype,
    ExtensionArray,
    ExtensionDtype,
    NpDtype,
    Self,
    npt,
)
from pyarrow import Array as pa_Array, ChunkedArray as pa_ChunkedArray
from pandas.core.dtypes.missing import NAType
from pandas.core.arrays.string_ import StringDtype
from pandas.core.arrays.arrow import ArrowExtensionArray

class ArrowStringArray:
    _storage: str
    _na_value: NAType
    dtype: StringDtype
    _pa_array: pa_Array

    def __init__(self, values: Union[pa_Array, pa_ChunkedArray]) -> None: ...

    @classmethod
    def _box_pa_scalar(cls, value: Any, pa_type: Optional[Any] = None) -> Any: ...

    @classmethod
    def _box_pa_array(cls, value: Any, pa_type: Optional[Any] = None, copy: bool = False) -> Any: ...

    def __len__(self) -> int: ...

    @classmethod
    def _from_sequence(cls, scalars: ArrayLike, dtype: Optional[Dtype] = None, copy: bool = False) -> Self: ...

    @classmethod
    def _from_sequence_of_strings(cls, strings: ArrayLike, dtype: Dtype, copy: bool = False) -> Self: ...

    def insert(self, loc: int, item: Union[str, NAType]) -> Self: ...

    def _convert_bool_result(self, values: Any, na: Any = ..., method_name: Optional[str] = None) -> npt.NDArray[np.bool_]: ...

    def _maybe_convert_setitem_value(self, value: Any) -> Any: ...

    def isin(self, values: Any) -> npt.NDArray[np.bool_]: ...

    def astype(self, dtype: Union[str, ExtensionDtype, Dtype, NpDtype], copy: bool = True) -> Union[Self, ExtensionArray]: ...

    def isna(self) -> npt.NDArray[np.bool_]: ...

    def notna(self) -> npt.NDArray[np.bool_]: ...

    @property
    def str(self) -> Any: ...

    def _reduce(self, name: str, skipna: bool = True, keepdims: bool = False, **kwargs: Any) -> Union[npt.NDArray[np.bool_], npt.NDArray[np.int64], npt.NDArray[np.float64]]: ...

    def value_counts(self, dropna: bool = True) -> Any: ...

    def _cmp_method(self, other: Any, op: Any) -> Any: ...

    def __pos__(self) -> None: ...

class ArrowStringArrayNumpySemantics(ArrowStringArray):
    _na_value: NAType