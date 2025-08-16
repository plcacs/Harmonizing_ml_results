from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Any, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.string_ import BaseStringArray
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.dtypes.common import is_scalar, pandas_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import isna
from pandas.compat import pa_version_under10p1, pa_version_under13p0, pa_version_under16p0
from pandas.util._exceptions import find_stack_level
from pandas._typing import ArrayLike, Dtype, NpDtype, Self, npt
from pandas import Series

ArrowStringScalarOrNAT = Union[str, pa.lib.NAType]

def _chk_pyarrow_available() -> None:
    ...

def _is_string_view(typ: Any) -> bool:
    ...

class ArrowStringArray(ObjectStringArrayMixin, ArrowExtensionArray, BaseStringArray):
    def __init__(self, values: Union[pa.Array, pa.ChunkedArray]) -> None:
        ...

    @classmethod
    def _box_pa_scalar(cls, value: Any, pa_type: Any = None) -> Any:
        ...

    @classmethod
    def _box_pa_array(cls, value: Any, pa_type: Any = None, copy: bool = False) -> Any:
        ...

    def __len__(self) -> int:
        ...

    @classmethod
    def _from_sequence(cls, scalars: Any, *, dtype: Any = None, copy: bool = False) -> ArrowStringArray:
        ...

    @classmethod
    def _from_sequence_of_strings(cls, strings: Any, *, dtype: Any, copy: bool = False) -> ArrowStringArray:
        ...

    @property
    def dtype(self) -> ExtensionDtype:
        ...

    def insert(self, loc: int, item: Any) -> ArrowStringArray:
        ...

    def _convert_bool_result(self, values: Any, na: Any = lib.no_default, method_name: str = None) -> np.ndarray:
        ...

    def _maybe_convert_setitem_value(self, value: Any) -> Any:
        ...

    def isin(self, values: List[Any]) -> np.ndarray:
        ...

    def astype(self, dtype: Any, copy: bool = True) -> Any:
        ...

    def _str_contains(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default, regex: bool = True) -> np.ndarray:
        ...

    def _str_replace(self, pat: str, repl: str, n: int = -1, case: bool = True, flags: int = 0, regex: bool = True) -> Any:
        ...

    def _str_repeat(self, repeats: int) -> Any:
        ...

    def _str_removeprefix(self, prefix: str) -> Any:
        ...

    def _str_count(self, pat: str, flags: int = 0) -> Any:
        ...

    def _str_find(self, sub: str, start: int = 0, end: int = None) -> Any:
        ...

    def _str_get_dummies(self, sep: str = '|', dtype: Any = None) -> Tuple[np.ndarray, List[str]]:
        ...

    def _convert_int_result(self, result: Any) -> Any:
        ...

    def _convert_rank_result(self, result: Any) -> Any:
        ...

    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs: Any) -> Any:
        ...

    def value_counts(self, dropna: bool = True) -> Any:
        ...

    def _cmp_method(self, other: Any, op: Any) -> Any:
        ...

    def __pos__(self) -> None:
        ...

class ArrowStringArrayNumpySemantics(ArrowStringArray):
    ...
