from __future__ import annotations
from functools import partial
import operator
from typing import TYPE_CHECKING, Any, Callable, cast, List, Literal, Optional, Tuple, Type, Union
import warnings
import numpy as np
from pandas._config import get_option, using_string_dtype
from pandas._libs import lib, missing as libmissing
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.lib import ensure_string_array
from pandas.compat import HAS_PYARROW, pa_version_under10p1
from pandas.compat.numpy import function as nv
from pandas.util._decorators import doc, set_module
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype, StorageExtensionDtype, register_extension_dtype
from pandas.core.dtypes.common import is_array_like, is_bool_dtype, is_integer_dtype, is_object_dtype, is_string_dtype, pandas_dtype
from pandas.core import nanops, ops
from pandas.core.algorithms import isin
from pandas.core.array_algos import masked_reductions
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.floating import FloatingArray, FloatingDtype
from pandas.core.arrays.integer import IntegerArray, IntegerDtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.missing import isna
from pandas.io.formats import printing

if TYPE_CHECKING:
    import pyarrow
    from pandas._typing import ArrayLike, AxisInt, Dtype, DtypeObj, NumpySorter, NumpyValueArrayLike, Scalar, Self, npt, type_t
    from pandas import Series

@set_module('pandas')
@register_extension_dtype
class StringDtype(StorageExtensionDtype):
    @property
    def name(self) -> str:
        ...

    @property
    def na_value(self) -> Union[np.float64, libmissing.NAType]:
        ...

    _metadata = ('storage', '_na_value')

    def __init__(self, storage: str | None = None, na_value: np.float64 | libmissing.NAType = libmissing.NA):
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, other: str | Self) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __reduce__(self) -> tuple:
        ...

    @property
    def type(self) -> type:
        ...

    @classmethod
    def construct_from_string(cls, string: str) -> Self:
        ...

    def construct_array_type(self) -> type[ExtensionArray]:
        ...

    def _get_common_dtype(self, dtypes: List[DtypeObj]) -> Self | None:
        ...

    def __from_arrow__(self, array: pyarrow.Array | pyarrow.ChunkedArray) -> StringArray:
        ...

class BaseStringArray(ExtensionArray):
    def tolist(self) -> List[str | libmissing.NAType]:
        ...

    @classmethod
    def _from_scalars(cls, scalars: ArrayLike, dtype: DtypeObj) -> Self:
        ...

    def _formatter(self, boxed: bool = False) -> Callable:
        ...

    def _str_map(self, f: Callable, na_value: Any = lib.no_default, dtype: DtypeObj | None = None, convert: bool = True) -> ExtensionArray:
        ...

    def _str_map_str_or_object(self, dtype: DtypeObj, na_value: Any, arr: np.ndarray, f: Callable, mask: np.ndarray) -> ExtensionArray:
        ...

    def _str_map_nan_semantics(self, f: Callable, na_value: Any = lib.no_default, dtype: DtypeObj | None = None) -> ExtensionArray:
        ...

    def view(self, dtype: Dtype | None = None) -> Self:
        ...

class StringArray(BaseStringArray, NumpyExtensionArray):
    def __init__(self, values: ArrayLike, copy: bool = False):
        ...

    def _validate(self) -> None:
        ...

    def _validate_scalar(self, value: Any) -> str | libmissing.NAType:
        ...

    @classmethod
    def _from_sequence(cls, scalars: ArrayLike, dtype: DtypeObj | None = None, copy: bool = False) -> Self:
        ...

    @classmethod
    def _from_sequence_of_strings(cls, strings: ArrayLike, dtype: DtypeObj, copy: bool = False) -> Self:
        ...

    @classmethod
    def _empty(cls, shape: tuple[int, ...], dtype: DtypeObj) -> Self:
        ...

    def __arrow_array__(self, type: pyarrow.DataType | None = None) -> pyarrow.Array:
        ...

    def _values_for_factorize(self) -> tuple[np.ndarray, libmissing.NAType]:
        ...

    def _maybe_convert_setitem_value(self, value: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def _putmask(self, mask: np.ndarray, value: Any) -> None:
        ...

    def _where(self, mask: np.ndarray, value: Any) -> Self:
        ...

    def isin(self, values: ArrayLike) -> np.ndarray:
        ...

    def astype(self, dtype: DtypeObj, copy: bool = True) -> ExtensionArray:
        ...

    def _reduce(self, name: str, skipna: bool = True, keepdims: bool = False, axis: int = 0, **kwargs: Any) -> Any:
        ...

    def _wrap_reduction_result(self, axis: int, result: Any) -> Any:
        ...

    def min(self, axis: int = None, skipna: bool = True, **kwargs: Any) -> Any:
        ...

    def max(self, axis: int = None, skipna: bool = True, **kwargs: Any) -> Any:
        ...

    def sum(self, axis: int = None, skipna: bool = True, min_count: int = 0, **kwargs: Any) -> Any:
        ...

    def value_counts(self, dropna: bool = True) -> Series:
        ...

    def memory_usage(self, deep: bool = False) -> int:
        ...

    def searchsorted(self, value: Any, side: Literal['left', 'right'] = 'left', sorter: NumpySorter | None = None) -> np.ndarray:
        ...

    def _cmp_method(self, other: Any, op: Callable) -> BooleanArray:
        ...

    _arith_method = _cmp_method

class StringArrayNumpySemantics(StringArray):
    def _validate(self) -> None:
        ...

    @classmethod
    def _from_sequence(cls, scalars: ArrayLike, dtype: DtypeObj | None = None, copy: bool = False) -> Self:
        ...