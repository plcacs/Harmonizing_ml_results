from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, ClassVar, Union, Optional, Sequence, Callable, TypeVar
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas._typing import ArrayLike, AxisInt, Dtype, DtypeObj, NumpySorter, NumpyValueArrayLike, Scalar, Self, npt, type_t
from pandas import Series
from pandas.core.dtypes.base import ExtensionDtype, StorageExtensionDtype, register_extension_dtype
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.floating import FloatingArray, FloatingDtype
from pandas.core.arrays.integer import IntegerArray, IntegerDtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.string_arrow import ArrowStringArray, ArrowStringArrayNumpySemantics
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
import pyarrow

_T = TypeVar('_T')

@register_extension_dtype
class StringDtype(StorageExtensionDtype):
    _metadata = ('storage', '_na_value')
    def __init__(
        self,
        storage: Optional[str] = None,
        na_value: Union[libmissing.NAType, float] = libmissing.NA,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def type(self) -> type: ...
    @property
    def na_value(self) -> Union[libmissing.NAType, float]: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __reduce__(self) -> tuple[type[StringDtype], tuple[str, 'Union[libmissing.NAType, float]']]: ...
    @classmethod
    def construct_from_string(cls, string: str) -> StringDtype: ...
    def construct_array_type(self) -> type[Union[StringArray, StringArrayNumpySemantics, ArrowStringArray, ArrowStringArrayNumpySemantics]]: ...
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> Optional[StringDtype]: ...
    def __from_arrow__(self, array: Union[pyarrow.Array, pyarrow.ChunkedArray]) -> Union[StringArray, ArrowStringArray, ArrowStringArrayNumpySemantics]: ...

class BaseStringArray(ExtensionArray):
    @classmethod
    def _from_scalars(cls, scalars: Sequence, dtype: Optional[Dtype]) -> BaseStringArray: ...
    def tolist(self) -> list: ...
    def _formatter(self, boxed: bool = False) -> Callable[[Any], str]: ...
    def _str_map(
        self,
        f: Callable[[str], Any],
        na_value: Any = ...,
        dtype: Optional[Dtype] = None,
        convert: bool = True,
    ) -> Union[ExtensionArray, np.ndarray]: ...
    def _str_map_str_or_object(
        self,
        dtype: Dtype,
        na_value: Any,
        arr: np.ndarray,
        f: Callable[[str], Any],
        mask: np.ndarray,
    ) -> Union[ExtensionArray, np.ndarray]: ...
    def _str_map_nan_semantics(
        self,
        f: Callable[[str], Any],
        na_value: Any = ...,
        dtype: Optional[Dtype] = None,
    ) -> Union[ExtensionArray, np.ndarray]: ...
    def view(self, dtype: Optional[Dtype] = None) -> ExtensionArray: ...

class StringArray(BaseStringArray, NumpyExtensionArray):
    _typ = 'extension'
    _storage = 'python'
    _na_value = libmissing.NA
    def __init__(self, values: ArrayLike, copy: bool = False) -> None: ...
    def _validate(self) -> None: ...
    def _validate_scalar(self, value: Any) -> Union[str, libmissing.NAType]: ...
    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Any],
        *,
        dtype: Optional[Dtype] = None,
        copy: bool = False,
    ) -> StringArray: ...
    @classmethod
    def _from_sequence_of_strings(
        cls, strings: Sequence[str], *, dtype: Dtype, copy: bool = False
    ) -> StringArray: ...
    @classmethod
    def _empty(cls, shape: tuple[int, ...], dtype: Dtype) -> StringArray: ...
    def __arrow_array__(self, type: Optional[pyarrow.DataType] = None) -> pyarrow.Array: ...
    def _values_for_factorize(self) -> tuple[np.ndarray, libmissing.NAType]: ...
    def _maybe_convert_setitem_value(self, value: Any) -> Union[str, np.ndarray, libmissing.NAType]: ...
    def __setitem__(self, key: Union[int, slice, np.ndarray], value: Any) -> None: ...
    def _putmask(self, mask: np.ndarray, value: Any) -> None: ...
    def _where(self, mask: np.ndarray, value: Any) -> StringArray: ...
    def isin(self, values: Union[ArrayLike, ExtensionArray]) -> np.ndarray: ...
    def astype(self, dtype: Dtype, copy: bool = True) -> Union[ExtensionArray, np.ndarray]: ...
    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        axis: AxisInt = 0,
        **kwargs: Any,
    ) -> Union[Scalar, ExtensionArray]: ...
    def _wrap_reduction_result(self, axis: Optional[AxisInt], result: Any) -> Any: ...
    def min(
        self, axis: Optional[AxisInt] = None, skipna: bool = True, **kwargs: Any
    ) -> Union[str, libmissing.NAType]: ...
    def max(
        self, axis: Optional[AxisInt] = None, skipna: bool = True, **kwargs: Any
    ) -> Union[str, libmissing.NAType]: ...
    def sum(
        self,
        *,
        axis: Optional[AxisInt] = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Union[str, libmissing.NAType]: ...
    def value_counts(self, dropna: bool = True) -> Series: ...
    def memory_usage(self, deep: bool = False) -> int: ...
    def searchsorted(
        self,
        value: Union[Scalar, ArrayLike],
        side: Literal['left', 'right'] = 'left',
        sorter: Optional[NumpySorter] = None,
    ) -> Union[np.ndarray, int]: ...
    def _cmp_method(
        self, other: Any, op: Callable[[Any, Any], Any]
    ) -> Union[ExtensionArray, np.ndarray]: ...

class StringArrayNumpySemantics(StringArray):
    _storage = 'python'
    _na_value = np.nan
    def _validate(self) -> None: ...
    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[Any],
        *,
        dtype: Optional[Dtype] = None,
        copy: bool = False,
    ) -> StringArrayNumpySemantics: ...