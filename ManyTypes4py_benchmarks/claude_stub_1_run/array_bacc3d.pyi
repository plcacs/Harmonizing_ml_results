```pyi
from __future__ import annotations

import decimal
import numbers
import sys
from typing import Any, ClassVar, overload

import numpy as np
from pandas.api.extensions import ExtensionArray, ExtensionDtype, ExtensionScalarOpsMixin, no_default
from pandas.core.arraylike import OpsMixin

class DecimalDtype(ExtensionDtype):
    type: type[decimal.Decimal]
    name: str
    na_value: decimal.Decimal
    _metadata: tuple[str, ...]
    context: decimal.Context

    def __init__(self, context: decimal.Context | None = None) -> None: ...
    def __repr__(self) -> str: ...
    @classmethod
    def construct_array_type(cls) -> type[DecimalArray]: ...
    @property
    def _is_numeric(self) -> bool: ...

class DecimalArray(OpsMixin, ExtensionScalarOpsMixin, ExtensionArray):
    __array_priority__: ClassVar[int]
    _HANDLED_TYPES: ClassVar[tuple[type, ...]]
    _data: np.ndarray
    _items: np.ndarray
    data: np.ndarray
    _dtype: DecimalDtype

    def __init__(
        self,
        values: Any,
        dtype: DecimalDtype | None = None,
        copy: bool = False,
        context: decimal.Context | None = None,
    ) -> None: ...
    @property
    def dtype(self) -> DecimalDtype: ...
    @classmethod
    def _from_sequence(
        cls,
        scalars: Any,
        *,
        dtype: DecimalDtype | None = None,
        copy: bool = False,
    ) -> DecimalArray: ...
    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: Any,
        *,
        dtype: DecimalDtype,
        copy: bool = False,
    ) -> DecimalArray: ...
    @classmethod
    def _from_factorized(cls, values: Any, original: Any) -> DecimalArray: ...
    def to_numpy(
        self,
        dtype: Any = None,
        copy: bool = False,
        na_value: Any = no_default,
        decimals: int | None = None,
    ) -> np.ndarray: ...
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...
    @overload
    def __getitem__(self, item: numbers.Integral) -> decimal.Decimal: ...
    @overload
    def __getitem__(self, item: Any) -> DecimalArray: ...
    def take(
        self,
        indexer: Any,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> DecimalArray: ...
    def copy(self) -> DecimalArray: ...
    def astype(self, dtype: Any, copy: bool = True) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __len__(self) -> int: ...
    def __contains__(self, item: Any) -> bool: ...
    @property
    def nbytes(self) -> int: ...
    def isna(self) -> np.ndarray: ...
    @property
    def _na_value(self) -> decimal.Decimal: ...
    def _formatter(self, boxed: bool = False) -> Any: ...
    @classmethod
    def _concat_same_type(cls, to_concat: Any) -> DecimalArray: ...
    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> Any: ...
    def _cmp_method(self, other: Any, op: Any) -> np.ndarray: ...
    def value_counts(self, dropna: bool = True) -> Any: ...
    def fillna(self, value: Any = None, limit: int | None = None) -> DecimalArray: ...

def to_decimal(values: Any, context: decimal.Context | None = None) -> DecimalArray: ...
def make_data() -> list[decimal.Decimal]: ...
```