from __future__ import annotations
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
    TYPE_CHECKING
)
import decimal
import numbers
import sys
import numpy as np
import pandas as pd
from pandas.api.extensions import no_default
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.arrays import ExtensionArray
from pandas.core.arraylike import OpsMixin
from pandas.core.indexers import check_array_indexer

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Axis,
        Dtype,
        DtypeObj,
        FillnaOptions,
        NpDtype,
        PositionalIndexer,
        Scalar,
        Self,
        TakeIndexer,
        type_t,
        npt
    )
    import numpy.typing as npt

_T = TypeVar("_T")

@register_extension_dtype
class DecimalDtype(ExtensionDtype):
    type: ClassVar[Type[decimal.Decimal]] = decimal.Decimal
    name: ClassVar[str] = "decimal"
    na_value: ClassVar[decimal.Decimal] = decimal.Decimal("NaN")
    _metadata: ClassVar[Tuple[str, ...]] = ("context",)
    context: decimal.Context

    def __init__(self, context: Optional[decimal.Context] = None) -> None: ...
    def __repr__(self) -> str: ...
    @classmethod
    def construct_array_type(cls) -> Type[DecimalArray]: ...
    @property
    def _is_numeric(self) -> bool: ...

class DecimalArray(OpsMixin, ExtensionScalarOpsMixin, ExtensionArray):
    __array_priority__: ClassVar[int] = 1000
    _HANDLED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (
        decimal.Decimal,
        numbers.Number,
        np.ndarray,
    )
    _data: np.ndarray
    _items: np.ndarray
    data: np.ndarray
    _dtype: DecimalDtype

    def __init__(
        self,
        values: Sequence[Union[decimal.Decimal, float, int]],
        dtype: Optional[DecimalDtype] = None,
        copy: bool = False,
        context: Optional[decimal.Context] = None,
    ) -> None: ...
    @property
    def dtype(self) -> DecimalDtype: ...
    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Union[decimal.Decimal, float, int, str]],
        *,
        dtype: Optional[DecimalDtype] = None,
        copy: bool = False,
    ) -> Self: ...
    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: Sequence[str],
        *,
        dtype: DecimalDtype,
        copy: bool = False,
    ) -> Self: ...
    @classmethod
    def _from_factorized(
        cls, values: np.ndarray, original: DecimalArray
    ) -> Self: ...
    def to_numpy(
        self,
        dtype: Optional[Union[npt.DTypeLike, Type[Any]]] = None,
        copy: bool = False,
        na_value: Any = no_default,
        decimals: Optional[int] = None,
    ) -> np.ndarray: ...
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Union[Self, Tuple[Self, ...], np.ndarray, Any]: ...
    @overload
    def __getitem__(self, item: int) -> decimal.Decimal: ...
    @overload
    def __getitem__(self, item: Union[slice, Sequence[int], np.ndarray]) -> Self: ...
    def __getitem__(
        self, item: Union[int, slice, Sequence[int], np.ndarray]
    ) -> Union[decimal.Decimal, Self]: ...
    def take(
        self,
        indexer: TakeIndexer,
        allow_fill: bool = False,
        fill_value: Optional[Any] = None,
    ) -> Self: ...
    def copy(self) -> Self: ...
    def astype(self, dtype: DtypeObj, copy: bool = True) -> ArrayLike: ...
    def __setitem__(
        self,
        key: Union[int, slice, Sequence[int], np.ndarray],
        value: Union[decimal.Decimal, float, int, str, Sequence[Union[decimal.Decimal, float, int, str]]],
    ) -> None: ...
    def __len__(self) -> int: ...
    def __contains__(self, item: Any) -> bool: ...
    @property
    def nbytes(self) -> int: ...
    def isna(self) -> np.ndarray: ...
    @property
    def _na_value(self) -> decimal.Decimal: ...
    def _formatter(self, boxed: bool = False) -> Any: ...
    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self: ...
    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> Union[Self, decimal.Decimal]: ...
    def _cmp_method(self, other: Any, op: Any) -> np.ndarray: ...
    def value_counts(self, dropna: bool = True) -> pd.Series: ...
    def fillna(
        self,
        value: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> Self: ...

def to_decimal(
    values: Sequence[Union[str, float, int, decimal.Decimal]],
    context: Optional[decimal.Context] = None,
) -> DecimalArray: ...
def make_data() -> list[decimal.Decimal]: ...