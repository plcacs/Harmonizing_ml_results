from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
    TYPE_CHECKING,
    overload,
)
from pandas._libs import lib, missing as libmissing
from pandas._libs.arrays import NDArrayBacked
from pandas.core.dtypes.base import ExtensionDtype, StorageExtensionDtype
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.floating import FloatingArray, FloatingDtype
from pandas.core.arrays.integer import IntegerArray, IntegerDtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas._typing import (
    ArrayLike,
    Dtype,
    DtypeObj,
    Scalar,
    Self,
    type_t,
)

if TYPE_CHECKING:
    from pandas import Series

class StringDtype(StorageExtensionDtype):
    _metadata: tuple[str, str] = ('storage', '_na_value')
    storage: str
    _na_value: Union[libmissing.NA, np.nan]
    
    def __init__(self, storage: Optional[str] = None, na_value: Optional[Union[libmissing.NA, np.nan]] = libmissing.NA):
        ...
    
    @property
    def name(self) -> str:
        ...
    
    @property
    def na_value(self) -> Union[libmissing.NA, np.nan]:
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
    
    def construct_array_type(self) -> type[ExtensionArray]:
        ...
    
    def _get_common_dtype(self, dtypes: List[Dtype]) -> Optional[StringDtype]:
        ...
    
    def __from_arrow__(self, array: Any) -> ExtensionArray:
        ...

class BaseStringArray(ExtensionArray):
    def tolist(self) -> list:
        ...
    
    @classmethod
    def _from_scalars(cls, scalars: ArrayLike, dtype: DtypeObj) -> Self:
        ...
    
    def _formatter(self, boxed: bool = False) -> Any:
        ...
    
    def _str_map(self, f: Any, na_value: Any = lib.no_default, dtype: Optional[Dtype] = None, convert: bool = True) -> ExtensionArray:
        ...
    
    def _str_map_str_or_object(self, dtype: Dtype, na_value: Any, arr: np.ndarray, f: Any, mask: np.ndarray) -> ExtensionArray:
        ...
    
    def _str_map_nan_semantics(self, f: Any, na_value: Any = lib.no_default, dtype: Optional[Dtype] = None) -> Union[np.ndarray, ExtensionArray]:
        ...
    
    def view(self, dtype: Optional[Dtype] = None) -> ExtensionArray:
        ...
    
    def _validate(self) -> None:
        ...
    
    def _validate_scalar(self, value: Scalar) -> str:
        ...
    
    @classmethod
    def _from_sequence(cls, scalars: ArrayLike, dtype: Optional[Dtype] = None, copy: bool = False) -> Self:
        ...
    
    @classmethod
    def _from_sequence_of_strings(cls, strings: ArrayLike, dtype: Dtype, copy: bool = False) -> Self:
        ...
    
    @classmethod
    def _empty(cls, shape: tuple[int, ...], dtype: Dtype) -> Self:
        ...
    
    def __arrow_array__(self, type: Any = None) -> Any:
        ...
    
    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        ...
    
    def _maybe_convert_setitem_value(self, value: Any) -> Any:
        ...
    
    def __setitem__(self, key: Any, value: Any) -> None:
        ...
    
    def _putmask(self, mask: np.ndarray, value: Any) -> None:
        ...
    
    def _where(self, mask: np.ndarray, value: Any) -> ExtensionArray:
        ...
    
    def isin(self, values: Any) -> np.ndarray:
        ...
    
    def astype(self, dtype: Union[str, Dtype], copy: bool = True) -> ExtensionArray:
        ...
    
    def _reduce(self, name: str, skipna: bool = True, keepdims: bool = False, axis: int = 0, **kwargs: Any) -> Any:
        ...
    
    def _wrap_reduction_result(self, axis: int, result: Any) -> Any:
        ...
    
    def min(self, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any) -> Any:
        ...
    
    def max(self, axis: Optional[int] = None, skipna: bool = True, **kwargs: Any) -> Any:
        ...
    
    def sum(self, axis: Optional[int] = None, skipna: bool = True, min_count: int = 0, **kwargs: Any) -> Any:
        ...
    
    def value_counts(self, dropna: bool = True) -> Series:
        ...
    
    def memory_usage(self, deep: bool = False) -> int:
        ...
    
    def searchsorted(self, value: Any, side: Literal['left', 'right'] = 'left', sorter: Optional[np.ndarray] = None) -> np.ndarray:
        ...
    
    def _cmp_method(self, other: Any, op: Any) -> ExtensionArray:
        ...
    _arith_method = _cmp_method

class StringArray(BaseStringArray, NumpyExtensionArray):
    _typ: str = 'extension'
    _storage: str = 'python'
    _na_value: Union[libmissing.NA, np.nan] = libmissing.NA
    
    def __init__(self, values: ArrayLike, copy: bool = False):
        ...
    
    def _validate(self) -> None:
        ...
    
    @classmethod
    def _from_sequence(cls, scalars: ArrayLike, dtype: Optional[Dtype] = None, copy: bool = False) -> Self:
        ...
    
    @classmethod
    def _from_sequence_of_strings(cls, strings: ArrayLike, dtype: Dtype, copy: bool = False) -> Self:
        ...
    
    @classmethod
    def _empty(cls, shape: tuple[int, ...], dtype: Dtype) -> Self:
        ...
    
    def __arrow_array__(self, type: Any = None) -> Any:
        ...

class StringArrayNumpySemantics(StringArray):
    _storage: str = 'python'
    _na_value: Union[libmissing.NA, np.nan] = np.nan
    
    def _validate(self) -> None:
        ...
    
    @classmethod
    def _from_sequence(cls, scalars: ArrayLike, dtype: Optional[Dtype] = None, copy: bool = False) -> Self:
        ...