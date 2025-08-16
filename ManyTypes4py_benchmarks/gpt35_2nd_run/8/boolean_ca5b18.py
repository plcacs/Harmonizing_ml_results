from __future__ import annotations
import numbers
from typing import TYPE_CHECKING, ClassVar, cast, Tuple
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import register_extension_dtype
from pandas.core.dtypes.missing import isna
from pandas.core import ops
from pandas.core.array_algos import masked_accumulations
from pandas.core.arrays.masked import BaseMaskedArray, BaseMaskedDtype
if TYPE_CHECKING:
    import pyarrow
    from pandas._typing import DtypeObj, Self, npt, type_t
    from pandas.core.dtypes.dtypes import ExtensionDtype

@register_extension_dtype
class BooleanDtype(BaseMaskedDtype):
    name: ClassVar[str] = 'boolean'
    _internal_fill_value: ClassVar[bool] = False

    @property
    def type(self) -> np.dtype:
        return np.bool_

    @property
    def kind(self) -> str:
        return 'b'

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype('bool')

    @classmethod
    def construct_array_type(cls) -> type:
        return BooleanArray

    def __repr__(self) -> str:
        return 'BooleanDtype'

    @property
    def _is_boolean(self) -> bool:
        return True

    @property
    def _is_numeric(self) -> bool:
        return True

    def __from_arrow__(self, array: pyarrow.Array) -> BooleanArray:
        ...
        
def coerce_to_array(values, mask=None, copy=False) -> Tuple[np.ndarray, np.ndarray]:
    ...

class BooleanArray(BaseMaskedArray):
    _TRUE_VALUES: ClassVar[set] = {'True', 'TRUE', 'true', '1', '1.0'}
    _FALSE_VALUES: ClassVar[set] = {'False', 'FALSE', 'false', '0', '0.0'}

    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: np.ndarray) -> BooleanArray:
        ...

    def __init__(self, values: np.ndarray, mask: np.ndarray, copy: bool = False) -> None:
        ...

    @property
    def dtype(self) -> BooleanDtype:
        return self._dtype

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype, copy=False, true_values=None, false_values=None, none_values=None) -> BooleanArray:
        ...

    @classmethod
    def _coerce_to_array(cls, value, *, dtype, copy=False) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def _logical_method(self, other, op) -> BooleanArray:
        ...

    def _accumulate(self, name, *, skipna=True, **kwargs) -> IntegerArray:
        ...
