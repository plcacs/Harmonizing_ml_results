from __future__ import annotations

import decimal
import numbers
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray, ExtensionScalarOpsMixin

if TYPE_CHECKING:
    from pandas._typing import type_t


@register_extension_dtype
class DecimalDtype(ExtensionDtype):
    type: type[decimal.Decimal]
    name: str
    na_value: decimal.Decimal
    _metadata: tuple[str, ...]
    context: decimal.Context

    def __init__(self, context: decimal.Context | None = None) -> None: ...
    def __repr__(self) -> str: ...

    @classmethod
    def construct_array_type(cls) -> type_t[DecimalArray]: ...

    @property
    def _is_numeric(self) -> bool: ...


class DecimalArray(OpsMixin, ExtensionScalarOpsMixin, ExtensionArray):
    __array_priority__: int
    _data: np.ndarray
    _items: np.ndarray
    data: np.ndarray
    _dtype: DecimalDtype
    _HANDLED_TYPES: tuple[type, ...]

    def __init__(
        self,
        values: Sequence[Any] | np.ndarray,
        dtype: DecimalDtype | None = None,
        copy: bool = False,
        context: decimal.Context | None = None,
    ) -> None: ...

    @property
    def dtype(self) -> DecimalDtype: ...

    @classmethod
    def _from_sequence(
        cls, scalars: Sequence[Any], *, dtype: DecimalDtype | None = None, copy: bool = False
    ) -> DecimalArray: ...

    @classmethod
    def _from_sequence_of_strings(
        cls, strings: Sequence[str], *, dtype: DecimalDtype | None = None, copy: bool = False
    ) -> DecimalArray: ...

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: DecimalArray) -> DecimalArray: ...

    def to_numpy(
        self,
        dtype: np.dtype[Any] | type | None = ...,
        copy: bool = False,
        na_value: Any = ...,
        decimals: int | None = None,
    ) -> np.ndarray: ...

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Any: ...

    def __getitem__(self, item: Any) -> decimal.Decimal | DecimalArray: ...

    def take(
        self,
        indexer: Sequence[int] | np.ndarray,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> DecimalArray: ...

    def copy(self) -> DecimalArray: ...

    def astype(self, dtype: Any, copy: bool = True) -> ExtensionArray | np.ndarray: ...

    def __setitem__(self, key: Any, value: Any) -> None: ...

    def __len__(self) -> int: ...

    def __contains__(self, item: object) -> bool: ...

    @property
    def nbytes(self) -> int: ...

    def isna(self) -> NDArray[np.bool_]: ...

    @property
    def _na_value(self) -> decimal.Decimal: ...

    def _formatter(self, boxed: bool = False) -> Callable[..., str]: ...

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[DecimalArray]) -> DecimalArray: ...

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs: Any
    ) -> Any: ...

    def _cmp_method(self, other: Any, op: Callable[..., Any]) -> np.ndarray: ...

    def value_counts(self, dropna: bool = True) -> Any: ...

    def fillna(self, value: Any = None, limit: int | None = None) -> DecimalArray: ...

    @classmethod
    def _add_arithmetic_ops(cls) -> None: ...


def to_decimal(
    values: Sequence[Any], context: decimal.Context | None = None
) -> DecimalArray: ...

def make_data() -> list[decimal.Decimal]: ...