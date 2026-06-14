from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from pandas._libs import missing as libmissing
from pandas._libs.arrays import NDArrayBacked

from pandas.core.dtypes.base import ExtensionDtype, StorageExtensionDtype

from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.floating import FloatingArray, FloatingDtype
from pandas.core.arrays.integer import IntegerArray, IntegerDtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray

if TYPE_CHECKING:
    import pyarrow

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        Dtype,
        DtypeObj,
        NumpySorter,
        NumpyValueArrayLike,
        Scalar,
        Self,
        npt,
        type_t,
    )

    from pandas import Series


class StringDtype(StorageExtensionDtype):
    _metadata: tuple[str, ...]
    storage: str
    _na_value: libmissing.NAType | float

    @property
    def name(self) -> str: ...
    @property
    def na_value(self) -> libmissing.NAType | float: ...

    def __init__(
        self,
        storage: str | None = ...,
        na_value: libmissing.NAType | float = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __reduce__(self) -> tuple[type[StringDtype], tuple[str, libmissing.NAType | float]]: ...

    @property
    def type(self) -> type[str]: ...

    @classmethod
    def construct_from_string(cls, string: str) -> StringDtype: ...
    def construct_array_type(
        self,
    ) -> type_t[BaseStringArray]: ...
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...
    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> StringArray | Any: ...


class BaseStringArray(ExtensionArray):
    def tolist(self) -> list[Any]: ...

    @classmethod
    def _from_scalars(cls, scalars: Any, dtype: DtypeObj) -> Self: ...
    def _formatter(self, boxed: bool = ...) -> Callable[[Any], str]: ...
    def _str_map(
        self,
        f: Callable[[str], Any],
        na_value: Any = ...,
        dtype: Dtype | None = ...,
        convert: bool = ...,
    ) -> ArrayLike: ...
    def _str_map_str_or_object(
        self,
        dtype: Dtype,
        na_value: Any,
        arr: np.ndarray,
        f: Callable[[str], Any],
        mask: npt.NDArray[np.bool_],
    ) -> ArrayLike: ...
    def _str_map_nan_semantics(
        self,
        f: Callable[[str], Any],
        na_value: Any = ...,
        dtype: Dtype | None = ...,
    ) -> ArrayLike: ...
    def view(self, dtype: str | None = ...) -> Self: ...


class StringArray(BaseStringArray, NumpyExtensionArray):
    _typ: str
    _storage: str
    _na_value: libmissing.NAType | float

    def __init__(self, values: Any, copy: bool = ...) -> None: ...
    def _validate(self) -> None: ...
    def _validate_scalar(self, value: Any) -> Any: ...

    @classmethod
    def _from_sequence(
        cls, scalars: Any, *, dtype: Dtype | None = ..., copy: bool = ...
    ) -> Self: ...
    @classmethod
    def _from_sequence_of_strings(
        cls, strings: Any, *, dtype: Dtype, copy: bool = ...
    ) -> Self: ...
    @classmethod
    def _empty(cls, shape: tuple[int, ...] | int, dtype: ExtensionDtype) -> Self: ...
    def __arrow_array__(self, type: Any = ...) -> pyarrow.Array: ...
    def _values_for_factorize(self) -> tuple[np.ndarray, Any]: ...
    def _maybe_convert_setitem_value(self, value: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def _putmask(self, mask: npt.NDArray[np.bool_], value: Any) -> None: ...
    def _where(self, mask: npt.NDArray[np.bool_], value: Any) -> Self: ...
    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]: ...
    def astype(self, dtype: Dtype, copy: bool = ...) -> ArrayLike: ...
    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = ...,
        keepdims: bool = ...,
        axis: AxisInt = ...,
        **kwargs: Any,
    ) -> Any: ...
    def _wrap_reduction_result(self, axis: AxisInt | None, result: Any) -> Any: ...
    def min(self, axis: AxisInt | None = ..., skipna: bool = ..., **kwargs: Any) -> Any: ...
    def max(self, axis: AxisInt | None = ..., skipna: bool = ..., **kwargs: Any) -> Any: ...
    def sum(
        self,
        *,
        axis: AxisInt | None = ...,
        skipna: bool = ...,
        min_count: int = ...,
        **kwargs: Any,
    ) -> Any: ...
    def value_counts(self, dropna: bool = ...) -> Series: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = ...,
        sorter: NumpySorter | None = ...,
    ) -> npt.NDArray[np.intp] | np.intp: ...
    def _cmp_method(self, other: Any, op: Any) -> Any: ...
    _arith_method: Any


class StringArrayNumpySemantics(StringArray):
    _storage: str
    _na_value: float

    def _validate(self) -> None: ...

    @classmethod
    def _from_sequence(
        cls, scalars: Any, *, dtype: Dtype | None = ..., copy: bool = ...
    ) -> Self: ...