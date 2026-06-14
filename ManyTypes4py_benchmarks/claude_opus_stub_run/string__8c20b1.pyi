from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from pandas._libs import lib, missing as libmissing
from pandas.core.dtypes.base import StorageExtensionDtype, ExtensionDtype
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.floating import FloatingArray, FloatingDtype
from pandas.core.arrays.integer import IntegerArray, IntegerDtype

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
    from functools import partial
    from pandas.core.arrays.string_arrow import (
        ArrowStringArray,
        ArrowStringArrayNumpySemantics,
    )


class StringDtype(StorageExtensionDtype):
    _metadata: tuple[str, ...]
    storage: str
    _na_value: lib.NoDefault | float  # actually libmissing.NA or np.nan

    @property
    def name(self) -> str: ...
    @property
    def na_value(self) -> libmissing.NAType | float: ...  # type: ignore[override]

    def __init__(
        self,
        storage: str | None = ...,
        na_value: object = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __reduce__(self) -> tuple[type[StringDtype], tuple[str, object]]: ...

    @property
    def type(self) -> type[str]: ...

    @classmethod
    def construct_from_string(cls, string: str) -> StringDtype: ...
    def construct_array_type(
        self,
    ) -> type_t[
        StringArray
        | StringArrayNumpySemantics
        | ArrowStringArray
        | ArrowStringArrayNumpySemantics
    ]: ...
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None: ...
    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> (
        StringArray
        | ArrowStringArray
        | ArrowStringArrayNumpySemantics
    ): ...


class BaseStringArray(ExtensionArray):
    def tolist(self) -> list[Any]: ...

    @classmethod
    def _from_scalars(cls, scalars: Sequence[Any], dtype: DtypeObj) -> Self: ...
    def _formatter(self, boxed: bool = ...) -> partial[str]: ...
    def _str_map(
        self,
        f: Any,
        na_value: object = ...,
        dtype: Dtype | None = ...,
        convert: bool = ...,
    ) -> ArrayLike | ExtensionArray: ...
    def _str_map_str_or_object(
        self,
        dtype: Dtype,
        na_value: object,
        arr: np.ndarray,
        f: Any,
        mask: npt.NDArray[np.bool_],
    ) -> ExtensionArray | np.ndarray: ...
    def _str_map_nan_semantics(
        self,
        f: Any,
        na_value: object = ...,
        dtype: Dtype | None = ...,
    ) -> ArrayLike | ExtensionArray: ...
    def view(self, dtype: Dtype | None = ...) -> Self: ...


class StringArray(BaseStringArray, NumpyExtensionArray):
    _typ: str
    _storage: str
    _na_value: libmissing.NAType | float

    def __init__(self, values: ArrayLike, copy: bool = ...) -> None: ...
    def _validate(self) -> None: ...
    def _validate_scalar(self, value: object) -> str | libmissing.NAType | float: ...

    @classmethod
    def _from_sequence(
        cls, scalars: ArrayLike, *, dtype: Dtype | None = ..., copy: bool = ...
    ) -> Self: ...
    @classmethod
    def _from_sequence_of_strings(
        cls, strings: ArrayLike, *, dtype: Dtype, copy: bool = ...
    ) -> Self: ...
    @classmethod
    def _empty(cls, shape: int | tuple[int, ...], dtype: ExtensionDtype) -> Self: ...
    def __arrow_array__(
        self, type: pyarrow.DataType | None = ...
    ) -> pyarrow.Array: ...
    def _values_for_factorize(self) -> tuple[np.ndarray, Any]: ...
    def _maybe_convert_setitem_value(self, value: object) -> object: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def _putmask(self, mask: npt.NDArray[np.bool_], value: Any) -> None: ...
    def _where(
        self, mask: npt.NDArray[np.bool_], value: Any
    ) -> StringArray: ...
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
    def min(
        self, axis: AxisInt | None = ..., skipna: bool = ..., **kwargs: Any
    ) -> Scalar: ...
    def max(
        self, axis: AxisInt | None = ..., skipna: bool = ..., **kwargs: Any
    ) -> Scalar: ...
    def sum(
        self,
        *,
        axis: AxisInt | None = ...,
        skipna: bool = ...,
        min_count: int = ...,
        **kwargs: Any,
    ) -> Scalar: ...
    def value_counts(self, dropna: bool = ...) -> Series: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = ...,
        sorter: NumpySorter | None = ...,
    ) -> npt.NDArray[np.intp] | np.intp: ...
    def _cmp_method(self, other: Any, op: Any) -> Any: ...
    _arith_method = _cmp_method


class StringArrayNumpySemantics(StringArray):
    _storage: str
    _na_value: float

    def _validate(self) -> None: ...

    @classmethod
    def _from_sequence(
        cls, scalars: ArrayLike, *, dtype: Dtype | None = ..., copy: bool = ...
    ) -> Self: ...