from __future__ import annotations
import operator
import re
from typing import TYPE_CHECKING, Union, Any, Callable, Literal, Sequence, overload, cast
import warnings
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas.compat import pa_version_under10p1, pa_version_under13p0, pa_version_under16p0
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_scalar, pandas_dtype
from pandas.core.dtypes.missing import isna
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.arrays.string_ import BaseStringArray, StringDtype
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.core.arrays.masked import BaseMaskedArray
import pyarrow as pa
import pyarrow.compute as pc
from pandas._typing import ArrayLike, Dtype, NpDtype, Self, npt
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas import Series
from collections.abc import Callable as CallableABC

ArrowStringScalarOrNAT = Union[str, libmissing.NAType]

def _chk_pyarrow_available() -> None: ...

def _is_string_view(typ: Any) -> bool: ...

class ArrowStringArray(ObjectStringArrayMixin, ArrowExtensionArray, BaseStringArray):
    _storage: str
    _na_value: Any
    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None: ...
    @classmethod
    def _box_pa_scalar(cls, value: Any, pa_type: pa.DataType | None = ...) -> pa.Scalar: ...
    @classmethod
    def _box_pa_array(cls, value: Any, pa_type: pa.DataType | None = ..., copy: bool = ...) -> pa.Array: ...
    def __len__(self) -> int: ...
    @classmethod
    def _from_sequence(cls, scalars: Sequence[str | None] | ArrayLike | BaseMaskedArray | pa.Array | pa.ChunkedArray, *, dtype: Dtype | None = ..., copy: bool = ...) -> Self: ...
    @classmethod
    def _from_sequence_of_strings(cls, strings: Sequence[str | None] | ArrayLike, *, dtype: Dtype, copy: bool = ...) -> Self: ...
    @property
    def dtype(self) -> StringDtype: ...
    def insert(self, loc: int, item: str | libmissing.NAType) -> Self: ...
    def _convert_bool_result(self, values: pa.Array, na: Any = ..., method_name: str | None = ...) -> np.ndarray | BooleanDtype: ...
    def _maybe_convert_setitem_value(self, value: Any) -> Any: ...
    def isin(self, values: Sequence[Any]) -> np.ndarray: ...
    def astype(self, dtype: Dtype, copy: bool = ...) -> Any: ...
    _str_isalnum: Any
    _str_isalpha: Any
    _str_isdecimal: Any
    _str_isdigit: Any
    _str_islower: Any
    _str_isnumeric: Any
    _str_isspace: Any
    _str_istitle: Any
    _str_isupper: Any
    _str_map: Any
    _str_startswith: Any
    _str_endswith: Any
    _str_pad: Any
    _str_match: Any
    _str_fullmatch: Any
    _str_lower: Any
    _str_upper: Any
    _str_strip: Any
    _str_lstrip: Any
    _str_rstrip: Any
    _str_removesuffix: Any
    _str_get: Any
    _str_capitalize: Any
    _str_title: Any
    _str_swapcase: Any
    _str_slice_replace: Any
    _str_len: Any
    _str_slice: Any
    def _str_contains(self, pat: str | re.Pattern, case: bool = ..., flags: int = ..., na: Any = ..., regex: bool = ...) -> BooleanDtype | np.ndarray: ...
    def _str_replace(self, pat: str | re.Pattern, repl: str | Callable, n: int = ..., case: bool = ..., flags: int = ..., regex: bool = ...) -> Self: ...
    def _str_repeat(self, repeats: int | Sequence[int]) -> Self: ...
    def _str_removeprefix(self, prefix: str) -> Self: ...
    def _str_count(self, pat: str, flags: int = ...) -> np.ndarray | Int64Dtype: ...
    def _str_find(self, sub: str, start: int = ..., end: int | None = ...) -> np.ndarray | Int64Dtype: ...
    def _str_get_dummies(self, sep: str = ..., dtype: Dtype | None = ...) -> tuple[np.ndarray, list[str]]: ...
    def _convert_int_result(self, result: pa.Array) -> np.ndarray | Int64Dtype: ...
    def _convert_rank_result(self, result: pa.Array) -> np.ndarray | Float64Dtype: ...
    def _reduce(self, name: str, *, skipna: bool = ..., keepdims: bool = ..., **kwargs: Any) -> Any: ...
    def value_counts(self, dropna: bool = ...) -> Series: ...
    def _cmp_method(self, other: Any, op: Callable) -> BooleanDtype | np.ndarray: ...
    def __pos__(self) -> Any: ...

class ArrowStringArrayNumpySemantics(ArrowStringArray):
    _na_value: Any