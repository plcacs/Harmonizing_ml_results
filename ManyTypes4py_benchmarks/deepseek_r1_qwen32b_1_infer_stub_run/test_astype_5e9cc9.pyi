from datetime import datetime, timedelta, tzinfo
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pytest
from pandas import (
    NA,
    Categorical,
    CategoricalDtype,
    DatetimeTZDtype,
    Index,
    Interval,
    NaT,
    Series,
    Timedelta,
    Timestamp,
)
from pandas._libs.tslibs import iNaT
from pandas._testing import tm

RANDS_CHARS: np.ndarray[Any, np.str_]

def rand_str(nchars: int) -> str:
    ...


class TestAstypeAPI:
    def test_astype_unitless_dt64_raises(self) -> None:
        ...

    def test_arg_for_errors_in_astype(self) -> None:
        ...

    @pytest.mark.parametrize('dtype_class', [dict, Series])
    def test_astype_dict_like(self, dtype_class: type) -> None:
        ...


class TestAstype:
    @pytest.mark.parametrize('tz', [None, str])
    def test_astype_object_to_dt64_non_nano(self, tz: Optional[str]) -> None:
        ...

    def test_astype_mixed_object_to_dt64tz(self) -> None:
        ...

    @pytest.mark.parametrize('dtype', np.typecodes['All'])
    def test_astype_empty_constructor_equality(self, dtype: str) -> None:
        ...

    @pytest.mark.parametrize('dtype', [str, np.str_])
    @pytest.mark.parametrize('data', [[str, str, str, str], [str, str, str, float, float]])
    def test_astype_str_map(self, dtype: Union[type, np.dtype], data: List[str], using_infer_string: bool) -> None:
        ...

    def test_astype_float_to_period(self) -> None:
        ...

    def test_astype_no_pandas_dtype(self) -> None:
        ...

    @pytest.mark.parametrize('dtype', [np.datetime64, np.timedelta64])
    def test_astype_generic_timestamp_no_frequency(self, dtype: np.dtype, request: pytest.FixtureRequest) -> None:
        ...

    def test_astype_dt64_to_str(self) -> None:
        ...

    def test_astype_dt64tz_to_str(self) -> None:
        ...

    def test_astype_datetime(self, unit: str) -> None:
        ...

    def test_astype_datetime64tz(self) -> None:
        ...

    def test_astype_str_cast_dt64(self) -> None:
        ...

    def test_astype_str_cast_td64(self) -> None:
        ...

    def test_dt64_series_astype_object(self) -> None:
        ...

    def test_td64_series_astype_object(self) -> None:
        ...

    @pytest.mark.parametrize('data, dtype', [([str, str, str], 'string[python]'), pytest.param([str, str, str], 'string[pyarrow]', marks=pytest.mark.skip), ([str, str, str], 'category'), ([Timestamp, Timestamp, Timestamp], None), ([Interval, Interval, Interval], None)])
    @pytest.mark.parametrize('errors', ['raise', 'ignore'])
    def test_astype_ignores_errors_for_extension_dtypes(self, data: List[Any], dtype: Optional[str], errors: str) -> None:
        ...

    def test_astype_from_float_to_str(self, any_float_dtype: np.dtype) -> None:
        ...

    @pytest.mark.parametrize('value, string_value', [(None, 'None'), (np.nan, 'nan'), (NA, '<NA>')])
    def test_astype_to_str_preserves_na(self, value: Any, string_value: str, using_infer_string: bool) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['float32', 'float64', 'int64', 'int32'])
    def test_astype(self, dtype: str) -> None:
        ...

    @pytest.mark.parametrize('value', [np.nan, np.inf])
    def test_astype_cast_nan_inf_int(self, any_int_numpy_dtype: np.dtype, value: float) -> None:
        ...

    def test_astype_cast_object_int_fail(self, any_int_numpy_dtype: np.dtype) -> None:
        ...

    def test_astype_float_to_uint_negatives_raise(self, float_numpy_dtype: np.dtype, any_unsigned_int_numpy_dtype: np.dtype) -> None:
        ...

    def test_astype_cast_object_int(self) -> None:
        ...

    def test_astype_unicode(self, using_infer_string: bool) -> None:
        ...

    def test_astype_bytes(self) -> None:
        ...

    def test_astype_nan_to_bool(self) -> None:
        ...

    def test_astype_ea_to_datetimetzdtype(self, any_numeric_ea_dtype: np.dtype) -> None:
        ...

    def test_astype_retain_attrs(self, any_numpy_dtype: np.dtype) -> None:
        ...


class TestAstypeString:
    @pytest.mark.parametrize('data, dtype', [([bool, NA], 'boolean'), ([str, NA], 'category'), ([str, str], 'datetime64[ns]'), ([str, str, NaT], 'datetime64[ns]'), ([str, NaT], 'datetime64[ns, US/Pacific]'), ([int, None], 'UInt16'), ([str, str], 'period[M]'), ([str, str, NaT], 'period[M]'), ([str, str, NaT], 'timedelta64[ns]')])
    def test_astype_string_to_extension_dtype_roundtrip(self, data: List[Any], dtype: str, request: pytest.FixtureRequest, nullable_string_dtype: str) -> None:
        ...


class TestAstypeCategorical:
    def test_astype_categorical_to_other(self) -> None:
        ...

    def test_astype_categorical_invalid_conversions(self) -> None:
        ...

    def test_astype_categoricaldtype(self) -> None:
        ...

    @pytest.mark.parametrize('name', [None, str])
    @pytest.mark.parametrize('dtype_ordered', [True, False])
    @pytest.mark.parametrize('series_ordered', [True, False])
    def test_astype_categorical_to_categorical(self, name: Optional[str], dtype_ordered: bool, series_ordered: bool) -> None:
        ...

    def test_astype_bool_missing_to_categorical(self) -> None:
        ...

    def test_astype_categories_raises(self) -> None:
        ...

    @pytest.mark.parametrize('items', [[str, str, str, str], [int, int, int, int]])
    def test_astype_from_categorical(self, items: List[Any]) -> None:
        ...

    def test_astype_from_categorical_with_keywords(self) -> None:
        ...

    def test_astype_timedelta64_with_np_nan(self) -> None:
        ...

    @pytest.mark.skip
    def test_astype_int_na_string(self) -> None:
        ...