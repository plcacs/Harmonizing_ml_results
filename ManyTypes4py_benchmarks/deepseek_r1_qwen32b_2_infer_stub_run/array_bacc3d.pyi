from __future__ import annotations
import decimal
import numbers
from typing import Any, AnyStr, Callable, ClassVar, Iterable, List, Optional, Sequence, Tuple, Union, type
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype, ExtensionScalarOpsMixin
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import is_dtype_equal, is_float, is_integer, pandas_dtype
from pandas.core.arrays import ExtensionArray
from pandas.core.indexers import check_array_indexer

@register_extension_dtype
class DecimalDtype(ExtensionDtype):
    type: ClassVar[type] = decimal.Decimal
    name: ClassVar[str] = 'decimal'
    na_value: ClassVar[decimal.Decimal] = decimal.Decimal('NaN')
    _metadata: ClassVar[Tuple[str]] = ('context',)

    def __init__(self, context: Optional[decimal.Context] = None):
        ...

    def __repr__(self) -> str:
        ...

    @classmethod
    def construct_array_type(cls) -> type[DecimalArray]:
        ...

    @property
    def _is_numeric(self) -> bool:
        ...

class DecimalArray(OpsMixin, ExtensionScalarOpsMixin, ExtensionArray):
    __array_priority__: ClassVar[int] = 1000
    _HANDLED_TYPES: ClassVar[Tuple[type]] = (decimal.Decimal, numbers.Number, np.ndarray)

    def __init__(self, values: Union[List[decimal.Decimal], np.ndarray, Iterable[Union[int, float, decimal.Decimal]]], dtype: Optional[DecimalDtype] = None, copy: bool = False, context: Optional[decimal.Context] = None):
        ...

    @property
    def dtype(self) -> DecimalDtype:
        ...

    @classmethod
    def _from_sequence(cls, scalars: Union[List[decimal.Decimal], np.ndarray, Iterable[Union[int, float]]], dtype: Optional[DecimalDtype] = None, copy: bool = False) -> DecimalArray:
        ...

    @classmethod
    def _from_sequence_of_strings(cls, strings: List[str], dtype: DecimalDtype, copy: bool = False) -> DecimalArray:
        ...

    @classmethod
    def _from_factorized(cls, values: Any, original: Any) -> DecimalArray:
        ...

    def to_numpy(self, dtype: Optional[np.dtype] = None, copy: bool = False, na_value: Any = no_default, decimals: Optional[int] = None) -> np.ndarray:
        ...

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        ...

    def __getitem__(self, item: Union[int, np.ndarray, List[int]]) -> Union[decimal.Decimal, DecimalArray]:
        ...

    def take(self, indexer: Union[np.ndarray, List[int]], allow_fill: bool = False, fill_value: Optional[decimal.Decimal] = None) -> DecimalArray:
        ...

    def copy(self) -> DecimalArray:
        ...

    def astype(self, dtype: Union[str, ExtensionDtype], copy: bool = True) -> ExtensionArray:
        ...

    def __setitem__(self, key: Union[int, np.ndarray, List[int]], value: Union[decimal.Decimal, List[decimal.Decimal], int, float]) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __contains__(self, item: decimal.Decimal) -> bool:
        ...

    @property
    def nbytes(self) -> int:
        ...

    def isna(self) -> np.ndarray:
        ...

    @property
    def _na_value(self) -> decimal.Decimal:
        ...

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str]:
        ...

    @classmethod
    def _concat_same_type(cls, to_concat: List[DecimalArray]) -> DecimalArray:
        ...

    def _reduce(self, name: str, skipna: bool = True, keepdims: bool = False, **kwargs: Any) -> Union[decimal.Decimal, DecimalArray]:
        ...

    def _cmp_method(self, other: Union[DecimalArray, np.ndarray, List[Union[int, float, decimal.Decimal]]], op: Callable[[Any, Any], bool]) -> np.ndarray:
        ...

    def value_counts(self, dropna: bool = True) -> pd.Series:
        ...

    def fillna(self, value: Optional[decimal.Decimal] = None, limit: Optional[int] = None) -> DecimalArray:
        ...

def to_decimal(values: Union[List[Union[int, float, decimal.Decimal]], np.ndarray], context: Optional[decimal.Context] = None) -> DecimalArray:
    ...

def make_data() -> List[decimal.Decimal]:
    ...