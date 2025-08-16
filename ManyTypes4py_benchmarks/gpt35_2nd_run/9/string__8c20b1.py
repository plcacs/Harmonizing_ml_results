from __future__ import annotations
from functools import partial
import operator
from typing import TYPE_CHECKING, Any, Literal, cast
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
    name: str

    def __init__(self, storage: str = None, na_value: Any = libmissing.NA) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __reduce__(self) -> tuple:
        ...

    @property
    def type(self) -> type:
        ...

    @classmethod
    def construct_from_string(cls, string: str) -> StringDtype:
        ...

    def construct_array_type(self) -> type:
        ...

    def _get_common_dtype(self, dtypes: list) -> StringDtype:
        ...

    def __from_arrow__(self, array: Any) -> ExtensionArray:
        ...

class BaseStringArray(ExtensionArray):
    def tolist(self) -> list:
        ...

    @classmethod
    def _from_scalars(cls, scalars: Any, dtype: Dtype) -> ExtensionArray:
        ...

    def _formatter(self, boxed: bool = False) -> partial:
        ...

    def _str_map(self, f: Any, na_value: Any = lib.no_default, dtype: Dtype = None, convert: bool = True) -> ExtensionArray:
        ...

    def _str_map_str_or_object(self, dtype: Dtype, na_value: Any, arr: Any, f: Any, mask: Any) -> ExtensionArray:
        ...

    def _str_map_nan_semantics(self, f: Any, na_value: Any = lib.no_default, dtype: Dtype = None) -> ExtensionArray:
        ...

    def view(self, dtype: Dtype = None) -> ExtensionArray:
        ...

class StringArray(BaseStringArray, NumpyExtensionArray):
    def __init__(self, values: Any, copy: bool = False) -> None:
        ...

    def _validate(self) -> None:
        ...

    def _validate_scalar(self, value: Any) -> Any:
        ...

    @classmethod
    def _from_sequence(cls, scalars: Any, dtype: Dtype = None, copy: bool = False) -> ExtensionArray:
        ...

    @classmethod
    def _from_sequence_of_strings(cls, strings: Any, dtype: Dtype, copy: bool = False) -> ExtensionArray:
        ...

    @classmethod
    def _empty(cls, shape: Any, dtype: Dtype) -> ExtensionArray:
        ...

    def __arrow_array__(self, type: Any = None) -> Any:
        ...

    def _values_for_factorize(self) -> tuple:
        ...

    def _maybe_convert_setitem_value(self, value: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def _putmask(self, mask: Any, value: Any) -> None:
        ...

    def _where(self, mask: Any, value: Any) -> Any:
        ...

    def isin(self, values: Any) -> Any:
        ...

    def astype(self, dtype: Dtype, copy: bool = True) -> ExtensionArray:
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

    def value_counts(self, dropna: bool = True) -> Any:
        ...

    def memory_usage(self, deep: bool = False) -> int:
        ...

    def searchsorted(self, value: Any, side: str = 'left', sorter: Any = None) -> Any:
        ...

    def _cmp_method(self, other: Any, op: Any) -> Any:
        ...

    def _arith_method(self, other: Any, op: Any) -> Any:
        ...

class StringArrayNumpySemantics(StringArray):
    def _validate(self) -> None:
        ...

    @classmethod
    def _from_sequence(cls, scalars: Any, dtype: Dtype = None, copy: bool = False) -> ExtensionArray:
        ...
