from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Sequence,
    Union,
    overload,
    cast
)
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import re
from pandas._libs import lib, missing as libmissing
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.arrays.string_ import BaseStringArray, StringDtype
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas._typing import ArrayLike, Dtype, NpDtype, Self, npt
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas import Series
from collections.abc import Callable as CallableABC

if TYPE_CHECKING:
    from pandas.core.arrays.masked import BaseMaskedArray

ArrowStringScalarOrNAT = Union[str, libmissing.NAType]

def _chk_pyarrow_available() -> None: ...

def _is_string_view(typ: Any) -> bool: ...

class ArrowStringArray(ObjectStringArrayMixin, ArrowExtensionArray, BaseStringArray):
    _storage: str
    _na_value: libmissing.NAType
    _dtype: StringDtype
    
    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None: ...
    
    @classmethod
    def _box_pa_scalar(cls, value: Any, pa_type: pa.DataType | None = None) -> pa.Scalar: ...
    
    @classmethod
    def _box_pa_array(cls, value: Any, pa_type: pa.DataType | None = None, copy: bool = False) -> pa.Array: ...
    
    def __len__(self) -> int: ...
    
    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[str | None] | ArrayLike | BaseMaskedArray | pa.Array | pa.ChunkedArray,
        *,
        dtype: Dtype | None = None,
        copy: bool = False
    ) -> Self: ...
    
    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: Sequence[str | None] | ArrayLike,
        *,
        dtype: Dtype,
        copy: bool = False
    ) -> Self: ...
    
    @property
    def dtype(self) -> StringDtype: ...
    
    def insert(self, loc: int, item: str | libmissing.NAType) -> Self: ...
    
    def _convert_bool_result(
        self,
        values: pa.Array,
        na: Any = ...,
        method_name: str | None = None
    ) -> np.ndarray | BooleanDtype: ...
    
    def _maybe_convert_setitem_value(self, value: Any) -> Any: ...
    
    def isin(self, values: Sequence[Any]) -> np.ndarray: ...
    
    def astype(self, dtype: Dtype, copy: bool = True) -> Any: ...
    
    _str_isalnum: Callable[..., BooleanDtype | np.ndarray]
    _str_isalpha: Callable[..., BooleanDtype | np.ndarray]
    _str_isdecimal: Callable[..., BooleanDtype | np.ndarray]
    _str_isdigit: Callable[..., BooleanDtype | np.ndarray]
    _str_islower: Callable[..., BooleanDtype | np.ndarray]
    _str_isnumeric: Callable[..., BooleanDtype | np.ndarray]
    _str_isspace: Callable[..., BooleanDtype | np.ndarray]
    _str_istitle: Callable[..., BooleanDtype | np.ndarray]
    _str_isupper: Callable[..., BooleanDtype | np.ndarray]
    _str_map: Callable[..., Self]
    _str_startswith: Callable[..., BooleanDtype | np.ndarray]
    _str_endswith: Callable[..., BooleanDtype | np.ndarray]
    _str_pad: Callable[..., Self]
    _str_match: Callable[..., BooleanDtype | np.ndarray]
    _str_fullmatch: Callable[..., BooleanDtype | np.ndarray]
    _str_lower: Callable[..., Self]
    _str_upper: Callable[..., Self]
    _str_strip: Callable[..., Self]
    _str_lstrip: Callable[..., Self]
    _str_rstrip: Callable[..., Self]
    _str_removesuffix: Callable[..., Self]
    _str_get: Callable[..., Self]
    _str_capitalize: Callable[..., Self]
    _str_title: Callable[..., Self]
    _str_swapcase: Callable[..., Self]
    _str_slice_replace: Callable[..., Self]
    _str_len: Callable[..., np.ndarray | Int64Dtype]
    _str_slice: Callable[..., Self]
    
    def _str_contains(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Any = ...,
        regex: bool = True
    ) -> BooleanDtype | np.ndarray: ...
    
    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True
    ) -> Self: ...
    
    def _str_repeat(self, repeats: int | Sequence[int]) -> Self: ...
    
    def _str_removeprefix(self, prefix: str) -> Self: ...
    
    def _str_count(self, pat: str, flags: int = 0) -> np.ndarray | Int64Dtype: ...
    
    def _str_find(self, sub: str, start: int = 0, end: int | None = None) -> np.ndarray | Int64Dtype: ...
    
    def _str_get_dummies(
        self,
        sep: str = "|",
        dtype: Dtype | None = None
    ) -> tuple[np.ndarray, list[str]]: ...
    
    def _convert_int_result(self, result: pa.Array) -> np.ndarray | Int64Dtype: ...
    
    def _convert_rank_result(self, result: pa.Array) -> np.ndarray | Float64Dtype: ...
    
    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        **kwargs: Any
    ) -> Any: ...
    
    def value_counts(self, dropna: bool = True) -> Series: ...
    
    def _cmp_method(self, other: Any, op: Callable) -> BooleanDtype | np.ndarray: ...
    
    def __pos__(self) -> Any: ...

class ArrowStringArrayNumpySemantics(ArrowStringArray):
    _na_value: float