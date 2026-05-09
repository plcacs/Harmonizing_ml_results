from __future__ import annotations
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Union
import numpy as np
import pyarrow
from pandas._libs import lib, missing as libmissing
from pandas._libs.arrays import NDArrayBacked
from pandas.core.dtypes.base import ExtensionDtype, StorageExtensionDtype
from pandas.core.arrays import BooleanArray, FloatingArray, IntegerArray
from pandas.core.arrays.floating import FloatingDtype
from pandas.core.arrays.integer import IntegerDtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.string_arrow import ArrowStringArray, ArrowStringArrayNumpySemantics
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.missing import isna
from pandas.io.formats import printing

@set_module('pandas')
@register_extension_dtype
class StringDtype(StorageExtensionDtype):
    _metadata: ClassVar[List[str]] = ['storage', '_na_value']
    storage: str
    _na_value: Union[libmissing.NA, np.nan]

    def __init__(self, storage: Literal['python', 'pyarrow'] = ..., na_value: Union[libmissing.NA, np.nan] = ...):
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def na_value(self) -> Union[libmissing.NA, np.nan]:
        ...

    @property
    def type(self) -> type:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __reduce__(self) -> tuple:
        ...

    @classmethod
    def construct_from_string(cls, string: str) -> StringDtype:
        ...

    def construct_array_type(self) -> type:
        ...

    def _get_common_dtype(self, dtypes: List[ExtensionDtype]) -> Optional[StringDtype]:
        ...

    def __from_arrow__(self, array: pyarrow.Array) -> Union[StringArray, ArrowStringArray]:
        ...

class BaseStringArray(ExtensionArray):
    def tolist(self) -> List[Optional[str]]:
        ...

    @classmethod
    def _from_scalars(cls, scalars: Any, dtype: Any) -> BaseStringArray:
        ...

    def _formatter(self, boxed: bool = ...) -> Callable:
        ...

    def _str_map(self, f: Callable, na_value: Any = ..., dtype: Any = ..., convert: bool = ...) -> Union[BooleanArray, ExtensionArray]:
        ...

    def _str_map_str_or_object(self, dtype: Any, na_value: Any, arr: Any, f: Callable, mask: Any) -> Union[ExtensionArray, Any]:
        ...

    def _str_map_nan_semantics(self, f: Callable, na_value: Any = ..., dtype: Any = ...) -> Union[ndarray, ExtensionArray]:
        ...

    def view(self, dtype: Any = ...) -> ndarray:
        ...

class StringArray(BaseStringArray, NumpyExtensionArray):
    _typ: str
    _storage: str
    _na_value: Union[libmissing.NA, np.nan]

    def __init__(self, values: Any, copy: bool = ...) -> None:
        ...

    def _validate(self) -> None:
        ...

    @classmethod
    def _from_sequence(cls, scalars: Any, dtype: Optional[StringDtype] = ..., copy: bool = ...) -> StringArray:
        ...

    @classmethod
    def _from_sequence_of_strings(cls, strings: Any, dtype: StringDtype, copy: bool = ...) -> StringArray:
        ...

    @classmethod
    def _empty(cls, shape: Any, dtype: StringDtype) -> StringArray:
        ...

    def __arrow_array__(self, type: Any = ...) -> pyarrow.Array:
        ...

    def _values_for_factorize(self) -> tuple[ndarray, Union[libmissing.NA, np.nan]]:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def _putmask(self, mask: Any, value: Any) -> None:
        ...

    def _where(self, mask: Any, value: Any) -> ExtensionArray:
        ...

    def isin(self, values: Any) -> ndarray:
        ...

    def astype(self, dtype: Any, copy: bool = ...) -> Union[IntegerArray, FloatingArray, ndarray, ExtensionArray]:
        ...

    def _reduce(self, name: str, skipna: bool = ..., keepdims: bool = ..., axis: int = ..., **kwargs: Any) -> Any:
        ...

    def _wrap_reduction_result(self, axis: Any, result: Any) -> Any:
        ...

    def min(self, axis: Any = ..., skipna: bool = ..., **kwargs: Any) -> Any:
        ...

    def max(self, axis: Any = ..., skipna: bool = ..., **kwargs: Any) -> Any:
        ...

    def sum(self, axis: Any = ..., skipna: bool = ..., min_count: int = ..., **kwargs: Any) -> Any:
        ...

    def value_counts(self, dropna: bool = ...) -> Any:
        ...

    def memory_usage(self, deep: bool = ...) -> int:
        ...

    def searchsorted(self, value: Any, side: Literal['left', 'right'] = ..., sorter: Any = ...) -> ndarray:
        ...

    def _cmp_method(self, other: Any, op: Callable) -> ExtensionArray:
        ...

    _arith_method: Callable

class StringArrayNumpySemantics(StringArray):
    _storage: str
    _na_value: np.nan

    @classmethod
    def _from_sequence(cls, scalars: Any, dtype: Optional[StringDtype] = ..., copy: bool = ...) -> StringArrayNumpySemantics:
        ...