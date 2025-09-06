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
    _na_value: Any
    storage: str

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
    def func_g1nrxlv8(cls, string: str) -> StringDtype:
        ...

    def func_fv1bit4g(self) -> type:
        ...

    def func_u484j2ay(self, dtypes: list) -> StringDtype:
        ...

    def __from_arrow__(self, array: Any) -> ExtensionArray:
        ...


class BaseStringArray(ExtensionArray):
    def func_lb63c54t(self) -> list:
        ...

    @classmethod
    def func_wcw6hsvr(cls, scalars: Any, dtype: Dtype) -> ExtensionArray:
        ...

    def func_aimtgotk(self, boxed: bool) -> partial:
        ...

    def func_gs30s0pc(self, f: Any, na_value: Any, dtype: Dtype, convert: bool) -> ExtensionArray:
        ...

    def func_pi82padx(self, dtype: Dtype, na_value: Any, arr: Any, f: Any, mask: Any) -> Any:
        ...

    def func_vhn2mm7o(self, f: Any, na_value: Any, dtype: Dtype) -> Any:
        ...

    def func_tmnstwzt(self, dtype: Dtype) -> Any:
        ...


class StringArray(BaseStringArray, NumpyExtensionArray):
    _typ: str
    _storage: str
    _na_value: Any

    def __init__(self, values: Any, copy: bool = False) -> None:
        ...

    def func_vs7ihzys(self) -> None:
        ...

    @classmethod
    def func_xe1hkosf(cls, scalars: Any, dtype: Dtype, copy: bool = False) -> ExtensionArray:
        ...

    @classmethod
    def func_yj66rau8(cls, strings: Any, dtype: Dtype, copy: bool = False) -> ExtensionArray:
        ...

    @classmethod
    def func_qpxvnl1g(cls, shape: tuple, dtype: Dtype) -> ExtensionArray:
        ...

    def __arrow_array__(self, type: Any = None) -> Any:
        ...

    def func_siwsw71h(self) -> tuple:
        ...

    def func_s9qv65ft(self, value: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def func_889hi52d(self, mask: Any, value: Any) -> None:
        ...

    def func_8xejz36i(self, mask: Any, value: Any) -> Any:
        ...

    def func_7wapm4ld(self, values: Any) -> Any:
        ...

    def func_9kfhxs09(self, dtype: Dtype, copy: bool = True) -> ExtensionArray:
        ...

    def func_g5r9cx9n(self, name: str, skipna: bool = True, keepdims: bool = False, axis: int = 0, **kwargs: Any) -> Any:
        ...

    def func_izwhmw06(self, axis: int, result: Any) -> Any:
        ...

    def min(self, axis: int = None, skipna: bool = True, **kwargs: Any) -> Any:
        ...

    def max(self, axis: int = None, skipna: bool = True, **kwargs: Any) -> Any:
        ...

    def sum(self, *, axis: int = None, skipna: bool = True, min_count: int = 0, **kwargs: Any) -> Any:
        ...

    def func_vweef6vs(self, dropna: bool = True) -> Any:
        ...

    def func_i5vmk6yb(self, deep: bool = False) -> Any:
        ...

    @doc(ExtensionArray.searchsorted)
    def func_yy8x09fo(self, value: Any, side: str = 'left', sorter: Any = None) -> Any:
        ...

    def func_0avfooxh(self, other: Any, op: Any) -> Any:
        ...


class StringArrayNumpySemantics(StringArray):
    _storage: str
    _na_value: Any

    def func_vs7ihzys(self) -> None:
        ...

    @classmethod
    def func_xe1hkosf(cls, scalars: Any, dtype: Dtype, copy: bool = False) -> ExtensionArray:
        ...
