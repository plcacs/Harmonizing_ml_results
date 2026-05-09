from __future__ import annotations
import decimal
import numbers
import sys
from typing import Any, Callable, Iterable, Optional, Union, overload, Type, TypeVar
import numpy as np
import pandas as pd
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.arrays import ExtensionArray, OpsMixin, ExtensionScalarOpsMixin
from pandas._typing import type_t

T = TypeVar("T", bound="DecimalArray")

class DecimalDtype(ExtensionDtype):
    type: Type[decimal.Decimal]
    name: str
    na_value: decimal.Decimal
    _metadata: tuple[str, ...]

    def __init__(self, context: Optional[decimal.Context] = None) -> None: ...
    def __repr__(self) -> str: ...
    @classmethod
    def construct_array_type(cls) -> Type[DecimalArray]: ...
    @property
    def _is_numeric(self) -> bool: ...

class DecimalArray(OpsMixin, ExtensionScalarOpsMixin, ExtensionArray):
    __array_priority__: int
    _HANDLED_TYPES: tuple[Type[decimal.Decimal], Type[numbers.Number], Type[np.ndarray]]

    def __init__(
        self, 
        values: Iterable[Any], 
        dtype: Optional[Union[str, ExtensionDtype]] = None, 
        copy: bool = False, 
        context: Optional[decimal.Context] = None
    ) -> None: ...

    @property
    def dtype(self) -> DecimalDtype: ...

    @classmethod
    def _from_sequence(cls, scalars: Iterable[Any], *, dtype: Any = None, copy: bool = False) -> DecimalArray: ...

    @classmethod
    def _from_sequence_of_strings(cls, strings: Iterable[str], *, dtype: Any, copy: bool = False) -> DecimalArray: ...

    @classmethod
    def _from_factorized(cls, values: Iterable[Any], original: Any) -> DecimalArray: ...

    def to_numpy(
        self, 
        dtype: Optional[Any] = None, 
        copy: bool = False, 
        na_value: Any = ..., 
        decimals: Optional[int] = None
    ) -> np.ndarray: ...

    def __array_ufunc__(
        self, 
        ufunc: Any, 
        method: str, 
        *inputs: Any, 
        **kwargs: Any
    ) -> Any: ...

    @overload
    def __getitem__(self, item: numbers.Integral) -> decimal.Decimal: ...
    @overload
    def __getitem__(self, item: Any) -> DecimalArray: ...
    def __getitem__(self, item: Any) -> Union[decimal.Decimal, DecimalArray]: ...

    def take(self, indexer: Any, allow_fill: bool = False, fill_value: Any = None) -> DecimalArray: ...

    def copy(self) -> DecimalArray: ...

    def astype(self, dtype: Any, copy: bool = True) -> Union[DecimalArray, ExtensionArray]: ...

    def __setitem__(self, key: Any, value: Any) -> None: ...

    def __len__(self) -> int: ...

    def __contains__(self, item: object) -> bool: ...

    @property
    def nbytes(self) -> int: ...

    def isna(self) -> np.ndarray: ...

    @property
    def _na_value(self) -> decimal.Decimal: ...

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str]: ...

    @classmethod
    def _concat_same_type(cls, to_concat: Iterable[DecimalArray]) -> DecimalArray: ...

    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs: Any) -> Union[decimal.Decimal, DecimalArray]: ...

    def _cmp_method(self, other: Any, op: Callable[[Any, Any], bool]) -> np.ndarray: ...

    def value_counts(self, dropna: bool = True) -> pd.Series: ...

    def fillna(self, value: Any = None, limit: Optional[int] = None) -> DecimalArray: ...

def to_decimal(values: Iterable[Any], context: Optional[decimal.Context] = None) -> DecimalArray: ...

def make_data() -> list[decimal.Decimal]: ...