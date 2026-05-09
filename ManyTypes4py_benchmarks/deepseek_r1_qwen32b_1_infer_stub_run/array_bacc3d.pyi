from decimal import Decimal, Context
from typing import (
    Any,
    ClassVar,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Type,
    overload,
)
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import type_t

class DecimalDtype(ExtensionDtype):
    type: ClassVar[type] = Decimal
    name: ClassVar[str] = 'decimal'
    na_value: ClassVar[Decimal] = Decimal('NaN')
    _metadata: ClassVar[Tuple[str]] = ('context',)

    def __init__(self, context: Optional[Context] = None):
        ...

    def __repr__(self) -> str:
        ...

    @classmethod
    def construct_array_type(cls) -> Type[ExtensionArray]:
        ...

    @property
    def _is_numeric(self) -> bool:
        ...

class DecimalArray(ExtensionArray):
    _HANDLED_TYPES: ClassVar[Tuple[type]] = (Decimal, numbers.Number, np.ndarray)

    def __init__(
        self,
        values: Union[List[Decimal], np.ndarray],
        dtype: Optional[DecimalDtype] = None,
        copy: bool = False,
        context: Optional[Context] = None,
    ):
        ...

    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Union[Decimal, numbers.Number]],
        dtype: Optional[DecimalDtype] = None,
        copy: bool = False,
    ) -> 'DecimalArray':
        ...

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: Sequence[str],
        dtype: Optional[DecimalDtype] = None,
        copy: bool = False,
    ) -> 'DecimalArray':
        ...

    @classmethod
    def _from_factorized(
        cls,
        values: np.ndarray,
        original: 'DecimalArray',
    ) -> 'DecimalArray':
        ...

    def to_numpy(
        self,
        dtype: Optional[np.dtype] = None,
        copy: bool = False,
        na_value: Any = no_default,
        decimals: Optional[int] = None,
    ) -> np.ndarray:
        ...

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Union['DecimalArray', Decimal, numbers.Number, np.ndarray],
        **kwargs: Any,
    ) -> Union['DecimalArray', Decimal, np.ndarray]:
        ...

    def __getitem__(self, item: Union[int, np.ndarray]) -> Union[Decimal, 'DecimalArray']:
        ...

    def take(
        self,
        indexer: Union[int, np.ndarray],
        allow_fill: bool = False,
        fill_value: Optional[Decimal] = None,
    ) -> 'DecimalArray':
        ...

    def copy(self) -> 'DecimalArray':
        ...

    def astype(
        self,
        dtype: type_t,
        copy: bool = True,
    ) -> Union['DecimalArray', ExtensionArray]:
        ...

    def __setitem__(
        self,
        key: Union[int, np.ndarray],
        value: Union[Decimal, Sequence[Decimal], numbers.Number, Sequence[numbers.Number]],
    ) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __contains__(self, item: Decimal) -> bool:
        ...

    @property
    def nbytes(self) -> int:
        ...

    def isna(self) -> np.ndarray:
        ...

    @property
    def _na_value(self) -> Decimal:
        ...

    def _formatter(self, boxed: bool = False) -> callable:
        ...

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence['DecimalArray']) -> 'DecimalArray':
        ...

    def _reduce(
        self,
        name: str,
        skipna: bool = True,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> Union[Decimal, 'DecimalArray']:
        ...

    def _cmp_method(
        self,
        other: Union['DecimalArray', Decimal, numbers.Number, Sequence[Union['DecimalArray', Decimal, numbers.Number]]],
        op: callable,
    ) -> np.ndarray:
        ...

    def value_counts(self, dropna: bool = True) -> pd.Series:
        ...

    def fillna(
        self,
        value: Optional[Union[Decimal, str]] = None,
        limit: Optional[int] = None,
    ) -> 'DecimalArray':
        ...

def to_decimal(
    values: Union[List[Union[Decimal, str, numbers.Number]], np.ndarray],
    context: Optional[Context] = None,
) -> DecimalArray:
    ...

def make_data() -> List[Decimal]:
    ...