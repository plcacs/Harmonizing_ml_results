from __future__ import annotations
import numpy as np
import pyarrow as pa
from pandas._libs import lib, missing as libmissing
from pandas.core.arrays import ArrowExtensionArray, BaseStringArray
from pandas.core.arrays.string_ import StringDtype
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

ArrowStringScalarOrNAT = Union[str, libmissing.NAType]

class ArrowStringArray(BaseStringArray, ArrowExtensionArray):
    _storage: ClassVar[str] = 'pyarrow'
    _na_value: ClassVar = libmissing.NA

    def __init__(self, values: Union[pa.Array, pa.ChunkedArray]) -> None:
        ...

    @classmethod
    def _box_pa_scalar(cls, value: Any, pa_type: Optional[pa.DataType] = None) -> pa.Scalar:
        ...

    @classmethod
    def _box_pa_array(cls, value: Any, pa_type: Optional[pa.DataType] = None, copy: bool = False) -> pa.Array:
        ...

    def __len__(self) -> int:
        ...

    @classmethod
    def _from_sequence(cls, scalars: List[Union[str, libmissing.NAType]], dtype: Optional[Any] = None, copy: bool = False) -> Self:
        ...

    @classmethod
    def _from_sequence_of_strings(cls, strings: List[str], dtype: Any, copy: bool = False) -> Self:
        ...

    @property
    def dtype(self) -> StringDtype:
        ...

    def insert(self, loc: int, item: Union[str, libmissing.NAType]) -> Self:
        ...

    def _convert_bool_result(self, values: pa.Array, na: Any = lib.no_default, method_name: Optional[str] = None) -> np.ndarray:
        ...

    def _maybe_convert_setitem_value(self, value: Any) -> Union[str, libmissing.NAType, np.ndarray]:
        ...

    def isin(self, values: List[str]) -> np.ndarray:
        ...

    def astype(self, dtype: Any, copy: bool = True) -> ArrowExtensionArray:
        ...

    def _str_contains(self, pat: str, case: bool = True, flags: int = 0, na: Any = lib.no_default, regex: bool = True) -> np.ndarray:
        ...

    def _str_replace(self, pat: str, repl: str, n: int = -1, case: bool = True, flags: int = 0, regex: bool = True) -> Union[Self, np.ndarray]:
        ...

    def _str_repeat(self, repeats: int) -> Union[Self, np.ndarray]:
        ...

    def _str_removeprefix(self, prefix: str) -> Self:
        ...

    def _str_count(self, pat: str, flags: int = 0) -> np.ndarray:
        ...

    def _str_find(self, sub: str, start: int = 0, end: Optional[int] = None) -> np.ndarray:
        ...

    def _str_get_dummies(self, sep: str = '|', dtype: Optional[Type] = None) -> Tuple[np.ndarray, List[str]]:
        ...

    def _convert_int_result(self, result: pa.Array) -> np.ndarray:
        ...

    def _convert_rank_result(self, result: pa.Array) -> np.ndarray:
        ...

    def _reduce(self, name: str, skipna: bool = True, keepdims: bool = False, **kwargs: Any) -> Union[int, float, Self, np.ndarray]:
        ...

    def value_counts(self, dropna: bool = True) -> Any:
        ...

    def _cmp_method(self, other: Union[str, Self], op: Callable) -> np.ndarray:
        ...

    def __pos__(self) -> None:
        ...

class ArrowStringArrayNumpySemantics(ArrowStringArray):
    _na_value: ClassVar = np.nan
    ...