from __future__ import annotations
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    overload,
)
import numpy as np
import pandas as pd
from pandas._libs import lib, missing as libmissing
from pandas._libs.arrays import NDArrayBacked
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import ExtensionDtype, StorageExtensionDtype
from pandas.core.dtypes.common import Dtype, DtypeObj
from pandas.core.indexers import check_array_indexer
from pandas.core.missing import isna

@set_module('pandas')
@register_extension_dtype
class StringDtype(StorageExtensionDtype):
    _metadata: ClassVar[tuple[str, ...]] = ('storage', '_na_value')
    storage: str
    _na_value: Union[np.nan, libmissing.NA]

    def __init__(self, storage: Optional[str] = None, na_value: Union[np.nan, libmissing.NA] = libmissing.NA) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __reduce__(self) -> tuple:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def na_value(self) -> Union[np.nan, libmissing.NA]:
        ...

    @property
    def type(self) -> type:
        ...

    @classmethod
    def construct_from_string(cls, string: str) -> StringDtype:
        ...

    def construct_array_type(self) -> type[ExtensionArray]:
        ...

    def _get_common_dtype(self, dtypes: list[Dtype]) -> Optional[StringDtype]:
        ...

    def __from_arrow__(self, array: Any) -> ExtensionArray:
        ...

class BaseStringArray(ExtensionArray):
    def tolist(self) -> list[Union[str, libmissing.NA]]:
        ...

    @classmethod
    def _from_scalars(cls, scalars: list[Any], dtype: Dtype) -> BaseStringArray:
        ...

    def _formatter(self, boxed: bool = False) -> Callable:
        ...

    def _str_map(self, f: Callable, na_value: Any = lib.no_default, dtype: Optional[Dtype] = None, convert: bool = True) -> Union[ExtensionArray, np.ndarray]:
        ...

    def _str_map_str_or_object(self, dtype: Dtype, na_value: Any, arr: np.ndarray, f: Callable, mask: np.ndarray) -> ExtensionArray:
        ...

    def _str_map_nan_semantics(self, f: Callable, na_value: Any = lib.no_default, dtype: Optional[Dtype] = None) -> Union[ExtensionArray, np.ndarray]:
        ...

    def view(self, dtype: Optional[Dtype] = None) -> ExtensionArray:
        ...

class StringArray(BaseStringArray, NumpyExtensionArray):
    _typ: str
    _storage: str
    _na_value: Union[np.nan, libmissing.NA]

    def __init__(self, values: ArrayLike, copy: bool = False) -> None:
        ...

    def _validate(self) -> None:
        ...

    @classmethod
    def _from_sequence(cls, scalars: ArrayLike, dtype: Optional[Dtype] = None, copy: bool = False) -> StringArray:
        ...

    @classmethod
    def _from_sequence_of_strings(cls, strings: list[str], dtype: Dtype, copy: bool = False) -> StringArray:
        ...

    @classmethod
    def _empty(cls, shape: tuple[int, ...], dtype: Dtype) -> StringArray:
        ...

    def __arrow_array__(self, type: Any = None) -> Any:
        ...

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        ...

    def _maybe_convert_setitem_value(self, value: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def _putmask(self, mask: np.ndarray, value: Any) -> None:
        ...

    def _where(self, mask: np.ndarray, value: Any) -> ExtensionArray:
        ...

    def isin(self, values: Any) -> np.ndarray:
        ...

    def astype(self, dtype: Dtype, copy: bool = True) -> ExtensionArray:
        ...

    def _reduce(self, name: str, skipna: bool = True, keepdims: bool = False, axis: int = 0, **kwargs: Any) -> Any:
        ...

    def _wrap_reduction_result(self, axis: int, result: Any) -> Any:
        ...

    def min(self, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any) -> Any:
        ...

    def max(self, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any) -> Any:
        ...

    def sum(self, axis: Optional[int] = None, skipna: bool = True, min_count: int = 0, **kwargs: Any) -> Any:
        ...

    def value_counts(self, dropna: bool = True) -> pd.Series:
        ...

    def memory_usage(self, deep: bool = False) -> int:
        ...

    def searchsorted(self, value: Any, side: Literal['left', 'right'] = 'left', sorter: Optional[np.ndarray] = None) -> np.ndarray:
        ...

    def _cmp_method(self, other: Any, op: Callable) -> ExtensionArray:
        ...

    _arith_method = _cmp_method

class StringArrayNumpySemantics(StringArray):
    _storage: str
    _na_value: np.nan

    def _validate(self) -> None:
        ...

    @classmethod
    def _from_sequence(cls, scalars: ArrayLike, dtype: Optional[Dtype] = None, copy: bool = False) -> StringArrayNumpySemantics:
        ...