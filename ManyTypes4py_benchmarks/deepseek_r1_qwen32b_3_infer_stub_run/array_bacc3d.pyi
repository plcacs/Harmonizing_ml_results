from __future__ import annotations
import decimal
from typing import Any, Iterable, List, Optional, Tuple, Union
import numpy as np
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.api.types import type_t

class DecimalDtype(ExtensionDtype):
    type: type[decimal.Decimal] = decimal.Decimal
    name: str = 'decimal'
    na_value: decimal.Decimal = decimal.Decimal('NaN')
    _metadata: Tuple[str, ...] = ('context',)

    def __init__(self, context: Optional[decimal.Context] = None) -> None:
        ...

    def __repr__(self) -> str:
        ...

    @classmethod
    def construct_array_type(cls) -> type[DecimalArray]:
        ...

    @property
    def _is_numeric(self) -> bool:
        ...

class DecimalArray(ExtensionArray):
    _HANDLED_TYPES: Tuple[type, ...] = (decimal.Decimal, numbers.Number, np.ndarray)

    def __init__(self, values: Iterable[Union[decimal.Decimal, float, int]], dtype: Optional[DecimalDtype] = None, copy: bool = False, context: Optional[decimal.Context] = None) -> None:
        ...

    @classmethod
    def _from_sequence(cls, scalars: Iterable[Union[decimal.Decimal, float, int]], dtype: Optional[DecimalDtype] = None, copy: bool = False) -> DecimalArray:
        ...

    @classmethod
    def _from_sequence_of_strings(cls, strings: Iterable[str], dtype: DecimalDtype, copy: bool = False) -> DecimalArray:
        ...

    @classmethod
    def _from_factorized(cls, values: Iterable[decimal.Decimal], original: DecimalArray) -> DecimalArray:
        ...

    def to_numpy(self, dtype: Optional[np.dtype] = None, copy: bool = False, na_value: Any = no_default, decimals: Optional[int] = None) -> np.ndarray:
        ...

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
        ...

    def __getitem__(self, item: Union[int, np.ndarray]) -> Union[decimal.Decimal, DecimalArray]:
        ...

    def take(self, indexer: Union[np.ndarray, list[int]], allow_fill: bool = False, fill_value: Optional[decimal.Decimal] = None) -> DecimalArray:
        ...

    def copy(self) -> DecimalArray:
        ...

    def astype(self, dtype: Union[DecimalDtype, type], copy: bool = True) -> DecimalArray:
        ...

    def __setitem__(self, key: Union[int, np.ndarray], value: Union[decimal.Decimal, Iterable[decimal.Decimal]]) -> None:
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

    def _formatter(self, boxed: bool = False) -> Any:
        ...

    @classmethod
    def _concat_same_type(cls, to_concat: Iterable[DecimalArray]) -> DecimalArray:
        ...

    def _reduce(self, name: str, skipna: bool = True, keepdims: bool = False, **kwargs: dict[str, Any]) -> Union[decimal.Decimal, DecimalArray]:
        ...

    def _cmp_method(self, other: Union[DecimalArray, Iterable[decimal.Decimal], decimal.Decimal], op: Any) -> np.ndarray:
        ...

    def value_counts(self, dropna: bool = True) -> pd.core.series.Series:
        ...

    def fillna(self, value: Optional[decimal.Decimal] = None, limit: Optional[int] = None) -> DecimalArray:
        ...

def to_decimal(values: Iterable[Union[decimal.Decimal, float, int]], context: Optional[decimal.Context] = None) -> DecimalArray:
    ...

def make_data() -> List[decimal.Decimal]:
    ...