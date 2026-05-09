from __future__ import annotations
import re
from typing import Union, Optional, overload, Any, Sequence, Tuple, overload
import numpy as np
import pyarrow as pa
from pandas._libs import missing as libmissing
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.arrays.string_ import BaseStringArray, StringDtype
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas._typing import Dtype, npt

ArrowStringScalarOrNAT = Union[str, libmissing.NAType]

def _chk_pyarrow_available() -> None: ...

def _is_string_view(typ: Any) -> bool: ...

class ArrowStringArray(ObjectStringArrayMixin, ArrowExtensionArray, BaseStringArray):
    _storage: str
    _na_value: Any

    def __init__(self, values: Union[pa.Array, pa.ChunkedArray]) -> None: ...

    @classmethod
    def _box_pa_scalar(cls, value: Any, pa_type: Optional[pa.DataType] = None) -> pa.Scalar: ...

    @classmethod
    def _box_pa_array(cls, value: Any, pa_type: Optional[pa.DataType] = None, copy: bool = False) -> pa.Array: ...

    def __len__(self) -> int: ...

    @classmethod
    def _from_sequence(
        cls, 
        scalars: Any, 
        *, 
        dtype: Optional[Union[str, Dtype]] = None, 
        copy: bool = False
    ) -> ArrowStringArray: ...

    @classmethod
    def _from_sequence_of_strings(
        cls, 
        strings: Sequence[str], 
        *, 
        dtype: Dtype, 
        copy: bool = False
    ) -> ArrowStringArray: ...

    @property
    def dtype(self) -> StringDtype: ...

    def insert(self, loc: int, item: ArrowStringScalarOrNAT) -> ArrowStringArray: ...

    def _convert_bool_result(
        self, 
        values: pa.Array, 
        na: Any = ..., 
        method_name: Optional[str] = None
    ) -> Union[npt.NDArray[np.bool_], BooleanDtype]: ...

    def _maybe_convert_setitem_value(self, value: Any) -> Any: ...

    def isin(self, values: Sequence[Any]) -> npt.NDArray[np.bool_]: ...

    def astype(self, dtype: Dtype, copy: bool = True) -> Any: ...

    def _str_contains(
        self, 
        pat: str, 
        case: bool = True, 
        flags: int = 0, 
        na: Any = ..., 
        regex: bool = True
    ) -> Any: ...

    def _str_replace(
        self, 
        pat: str, 
        repl: Union[str, callable], 
        n: int = -1, 
        case: bool = True, 
        flags: int = 0, 
        regex: bool = True
    ) -> Any: ...

    def _str_repeat(self, repeats: Union[int, npt.NDArray[np.int64]]) -> Any: ...

    def _str_removeprefix(self, prefix: str) -> Any: ...

    def _str_count(self, pat: str, flags: int = 0) -> Any: ...

    def _str_find(self, sub: str, start: int = 0, end: Optional[int] = None) -> Any: ...

    def _str_get_dummies(self, sep: str = '|', dtype: Any = None) -> Tuple[npt.NDArray[Any], Sequence[str]]: ...

    def _convert_int_result(self, result: pa.Array) -> Union[npt.NDArray[np.int64], Int64Dtype]: ...

    def _convert_rank_result(self, result: pa.Array) -> Union[npt.NDArray[np.float64], Float64Dtype]: ...

    def _reduce(
        self, 
        name: str, 
        *, 
        skipna: bool = True, 
        keepdims: bool = False, 
        **kwargs: Any
    ) -> Any: ...

    def value_counts(self, dropna: bool = True) -> Any: ...

    def _cmp_method(self, other: Any, op: Any) -> Any: ...

    def __pos__(self) -> Any: ...

class ArrowStringArrayNumpySemantics(ArrowStringArray):
    _na_value: float