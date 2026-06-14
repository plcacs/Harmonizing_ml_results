from __future__ import annotations

import re
from typing import TYPE_CHECKING, Union

import numpy as np
import pyarrow as pa

from pandas._libs import lib, missing as libmissing

from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.arrays.string_ import BaseStringArray, StringDtype
from pandas.core.strings.object_array import ObjectStringArrayMixin

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pandas._typing import ArrayLike, Dtype, NpDtype, Self, npt
    from pandas.core.dtypes.dtypes import ExtensionDtype
    from pandas import Series

ArrowStringScalarOrNAT = Union[str, libmissing.NAType]

def _chk_pyarrow_available() -> None: ...
def _is_string_view(typ: pa.DataType) -> bool: ...

class ArrowStringArray(ObjectStringArrayMixin, ArrowExtensionArray, BaseStringArray):
    _storage: str
    _na_value: libmissing.NAType
    _dtype: StringDtype

    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None: ...

    @classmethod
    def _box_pa_scalar(cls, value: object, pa_type: pa.DataType | None = ...) -> pa.Scalar: ...

    @classmethod
    def _box_pa_array(
        cls, value: object, pa_type: pa.DataType | None = ..., copy: bool = ...
    ) -> pa.Array | pa.ChunkedArray: ...

    def __len__(self) -> int: ...

    @classmethod
    def _from_sequence(
        cls, scalars: Sequence[object], *, dtype: Dtype | None = ..., copy: bool = ...
    ) -> ArrowStringArray: ...

    @classmethod
    def _from_sequence_of_strings(
        cls, strings: Sequence[str], *, dtype: Dtype | None = ..., copy: bool = ...
    ) -> ArrowStringArray: ...

    @property
    def dtype(self) -> StringDtype: ...

    def insert(self, loc: int, item: object) -> ArrowStringArray: ...

    def _convert_bool_result(
        self,
        values: pa.Array | pa.ChunkedArray,
        na: object = ...,
        method_name: str | None = ...,
    ) -> np.ndarray | BooleanDtype: ...

    def _maybe_convert_setitem_value(self, value: object) -> object: ...

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]: ...

    def astype(self, dtype: Dtype, copy: bool = ...) -> ArrayLike: ...

    _str_isalnum: Callable[..., object]
    _str_isalpha: Callable[..., object]
    _str_isdecimal: Callable[..., object]
    _str_isdigit: Callable[..., object]
    _str_islower: Callable[..., object]
    _str_isnumeric: Callable[..., object]
    _str_isspace: Callable[..., object]
    _str_istitle: Callable[..., object]
    _str_isupper: Callable[..., object]
    _str_map: Callable[..., object]
    _str_startswith: Callable[..., object]
    _str_endswith: Callable[..., object]
    _str_pad: Callable[..., object]
    _str_match: Callable[..., object]
    _str_fullmatch: Callable[..., object]
    _str_lower: Callable[..., object]
    _str_upper: Callable[..., object]
    _str_strip: Callable[..., object]
    _str_lstrip: Callable[..., object]
    _str_rstrip: Callable[..., object]
    _str_removesuffix: Callable[..., object]
    _str_get: Callable[..., object]
    _str_capitalize: Callable[..., object]
    _str_title: Callable[..., object]
    _str_swapcase: Callable[..., object]
    _str_slice_replace: Callable[..., object]
    _str_len: Callable[..., object]
    _str_slice: Callable[..., object]

    def _str_contains(
        self,
        pat: str | re.Pattern,
        case: bool = ...,
        flags: int = ...,
        na: object = ...,
        regex: bool = ...,
    ) -> object: ...

    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = ...,
        case: bool = ...,
        flags: int = ...,
        regex: bool = ...,
    ) -> object: ...

    def _str_repeat(self, repeats: int | Sequence[int]) -> object: ...

    def _str_removeprefix(self, prefix: str) -> object: ...

    def _str_count(self, pat: str, flags: int = ...) -> object: ...

    def _str_find(self, sub: str, start: int = ..., end: int | None = ...) -> object: ...

    def _str_get_dummies(
        self, sep: str = ..., dtype: NpDtype | ExtensionDtype | None = ...
    ) -> tuple[np.ndarray, npt.NDArray[np.object_]]: ...

    def _convert_int_result(self, result: pa.Array | pa.ChunkedArray) -> object: ...

    def _convert_rank_result(self, result: pa.Array | pa.ChunkedArray) -> object: ...

    def _reduce(
        self, name: str, *, skipna: bool = ..., keepdims: bool = ..., **kwargs: object
    ) -> object: ...

    def value_counts(self, dropna: bool = ...) -> Series: ...

    def _cmp_method(self, other: object, op: Callable[..., object]) -> object: ...

    def __pos__(self) -> None: ...

class ArrowStringArrayNumpySemantics(ArrowStringArray):
    _na_value: float  # np.nan