from __future__ import annotations
import numbers
from typing import TYPE_CHECKING, Any, Callable, Mapping
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.common import is_integer_dtype, is_string_dtype, pandas_dtype
from pandas.core.arrays.masked import BaseMaskedArray, BaseMaskedDtype
if TYPE_CHECKING:
    import pyarrow
    from pandas._typing import DtypeObj, Self, npt
    from pandas.core.dtypes.dtypes import ExtensionDtype

class NumericDtype(BaseMaskedDtype):

    def __repr__(self) -> str:
        return f'{self.name}Dtype()'

    @cache_readonly
    def is_signed_integer(self) -> bool:
        return self.kind == 'i'

    @cache_readonly
    def is_unsigned_integer(self) -> bool:
        return self.kind == 'u'

    @property
    def _is_numeric(self) -> bool:
        return True

    def __from_arrow__(self, array: pyarrow.Array) -> BaseMaskedArray:
        ...

    @classmethod
    def _get_dtype_mapping(cls) -> Mapping[np.dtype, NumericDtype]:
        raise AbstractMethodError(cls)

    @classmethod
    def _standardize_dtype(cls, dtype: Any) -> NumericDtype:
        ...

    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: NumericDtype, copy: bool) -> Any:
        ...

def _coerce_to_data_and_mask(values: Any, dtype: Any, copy: bool, dtype_cls: type, default_dtype: np.dtype) -> tuple:
    ...

class NumericArray(BaseMaskedArray):

    def __init__(self, values: np.ndarray, mask: np.ndarray, copy: bool = False):
        ...

    @cache_readonly
    def dtype(self) -> NumericDtype:
        ...

    @classmethod
    def _coerce_to_array(cls, value: Any, *, dtype: Any, copy: bool = False) -> tuple:
        ...

    @classmethod
    def _from_sequence_of_strings(cls, strings: list[str], *, dtype: Any, copy: bool = False) -> NumericArray:
        ...
