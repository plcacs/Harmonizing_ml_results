from __future__ import annotations
import operator
import re
from typing import TYPE_CHECKING, Union
import warnings
import numpy as np
from pandas._libs import lib, missing as libmissing
from pandas.compat import pa_version_under10p1, pa_version_under13p0, pa_version_under16p0
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_scalar, pandas_dtype
from pandas.core.dtypes.missing import isna
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.arrays.string_ import BaseStringArray, StringDtype
from pandas.core.strings.object_array import ObjectStringArrayMixin
if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc
if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pandas._typing import ArrayLike, Dtype, NpDtype, Self, npt
    from pandas.core.dtypes.dtypes import ExtensionDtype
    from pandas import Series
ArrowStringScalarOrNAT = Union[str, libmissing.NAType]

def _chk_pyarrow_available() -> None:
    if pa_version_under10p1:
        msg: str = 'pyarrow>=10.0.1 is required for PyArrow backed ArrowExtensionArray.'
        raise ImportError(msg)

def _is_string_view(typ) -> bool:
    return not pa_version_under16p0 and pa.types.is_string_view(typ)

class ArrowStringArray(ObjectStringArrayMixin, ArrowExtensionArray, BaseStringArray):
    """
    Extension array for string data in a ``pyarrow.ChunkedArray``.

    .. warning::

       ArrowStringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray
        The array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    :func:`array`
        The recommended function for creating a ArrowStringArray.
    Series.str
        The string methods are available on Series backed by
        a ArrowStringArray.

    Notes
    -----
    ArrowStringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(["This is", "some text", None, "data."], dtype="string[pyarrow]")
    <ArrowStringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string
    """
    _storage: str = 'pyarrow'
    _na_value: libmissing.NAType = libmissing.NA

    def __init__(self, values) -> None:
        _chk_pyarrow_available()
        if isinstance(values, (pa.Array, pa.ChunkedArray)) and (pa.types.is_string(values.type) or _is_string_view(values.type) or (pa.types.is_dictionary(values.type) and (pa.types.is_string(values.type.value_type) or pa.types.is_large_string(values.type.value_type) or _is_string_view(values.type.value_type)))):
            values = pc.cast(values, pa.large_string())
        super().__init__(values)
        self._dtype: StringDtype = StringDtype(storage=self._storage, na_value=self._na_value)
        if not pa.types.is_large_string(self._pa_array.type):
            raise ValueError('ArrowStringArray requires a PyArrow (chunked) array of large_string type')

    @classmethod
    def _box_pa_scalar(cls, value, pa_type=None):
        pa_scalar = super()._box_pa_scalar(value, pa_type)
        if pa.types.is_string(pa_scalar.type) and pa_type is None:
            pa_scalar = pc.cast(pa_scalar, pa.large_string())
        return pa_scalar

    @classmethod
    def _box_pa_array(cls, value, pa_type=None, copy=False):
        pa_array = super()._box_pa_array(value, pa_type)
        if pa.types.is_string(pa_array.type) and pa_type is None:
            pa_array = pc.cast(pa_array, pa.large_string())
        return pa_array

    def __len__(self) -> int:
        """
        Length of this array.

        Returns
        -------
        length : int
        """
        return len(self._pa_array)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        from pandas.core.arrays.masked import BaseMaskedArray
        _chk_pyarrow_available()
        if dtype and (not (isinstance(dtype, str) and dtype == 'string')):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, StringDtype) and dtype.storage == 'pyarrow'
        if isinstance(scalars, BaseMaskedArray):
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(result, copy=copy, convert_na_value=False)
            return cls(pa.array(result, mask=na_values, type=pa.large_string()))
        elif isinstance(scalars, (pa.Array, pa.ChunkedArray)):
            return cls(pc.cast(scalars, pa.large_string()))
        result = lib.ensure_string_array(scalars, copy=copy)
        return cls(pa.array(result, type=pa.large_string(), from_pandas=True))

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype, copy=False):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @property
    def dtype(self) -> StringDtype:
        """
        An instance of 'string[pyarrow]'.
        """
        return self._dtype

    def insert(self, loc, item) -> Union[ArrowStringArray, None]:
        if self.dtype.na_value is np.nan and item is np.nan:
            item = libmissing.NA
        if not isinstance(item, str) and item is not libmissing.NA:
            raise TypeError(f"Invalid value '{item}' for dtype 'str'. Value should be a string or missing value, got '{type(item).__name__}' instead.")
        return super().insert(loc, item)

    def _convert_bool_result(self, values, na=lib.no_default, method_name=None) -> np.ndarray:
        if na is not lib.no_default and (not isna(na)) and (not isinstance(na, bool)):
            warnings.warn(f"Allowing a non-bool 'na' in obj.str.{method_name} is deprecated and will raise in a future version.", FutureWarning, stacklevel=find_stack_level())
            na = bool(na)
        if self.dtype.na_value is np.nan:
            if na is lib.no_default or isna(na):
                values = values.fill_null(False)
            else:
                values = values.fill_null(na)
            return values.to_numpy()
        elif na is not lib.no_default and (not isna(na)):
            values = values.fill_null(na)
        return BooleanDtype().__from_arrow__(values)

    def _maybe_convert_setitem_value(self, value) -> np.ndarray:
        """Maybe convert value to be pyarrow compatible."""
        if is_scalar(value):
            if isna(value):
                value = None
            elif not isinstance(value, str):
                raise TypeError(f"Invalid value '{value}' for dtype 'str'. Value should be a string or missing value, got '{type(value).__name__}' instead.")
        else:
            value = np.array(value, dtype=object, copy=True)
            value[isna(value)] = None
            for v in value:
                if not (v is None or isinstance(v, str)):
                    raise TypeError("Invalid value for dtype 'str'. Value should be a string or missing value (or array of those).")
        return super()._maybe_convert_setitem_value(value)

    def isin(self, values) -> np.ndarray:
        value_set = [pa_scalar.as_py() for pa_scalar in [pa.scalar(value, from_pandas=True) for value in values] if pa_scalar.type in (pa.string(), pa.null(), pa.large_string())]
        if not len(value_set):
            return np.zeros(len(self), dtype=bool)
        result = pc.is_in(self._pa_array, value_set=pa.array(value_set, type=self._pa_array.type))
        return np.array(result, dtype=np.bool_)

    def astype(self, dtype, copy=True) -> Union[ArrowStringArray, np.ndarray]:
        dtype = pandas_dtype(dtype)
        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif isinstance(dtype, NumericDtype):
            data = self._pa_array.cast(pa.from_numpy_dtype(dtype.numpy_dtype))
            return dtype.__from_arrow__(data)
        elif isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.floating):
            return self.to_numpy(dtype=dtype, na_value=np.nan)
        return super().astype(dtype, copy=copy)

    # Other methods with type annotations omitted for brevity
