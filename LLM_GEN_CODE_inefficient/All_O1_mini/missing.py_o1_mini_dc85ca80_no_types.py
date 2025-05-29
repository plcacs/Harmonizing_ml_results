import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from typing import Any, List, Tuple


class BaseMissingTests:

    def test_isna(self, data_missing):
        expected: np.ndarray = np.array([True, False])
        result: np.ndarray = pd.isna(data_missing)
        tm.assert_numpy_array_equal(result, expected)
        result = pd.Series(data_missing).isna()
        expected_series = pd.Series(expected)
        tm.assert_series_equal(result, expected_series)
        result = pd.Series(data_missing).drop([0, 1]).isna()
        expected_series = pd.Series([], dtype=bool)
        tm.assert_series_equal(result, expected_series)

    @pytest.mark.parametrize('na_func', ['isna', 'notna'])
    def test_isna_returns_copy(self, data_missing, na_func):
        result = pd.Series(data_missing)
        expected = result.copy()
        mask = getattr(result, na_func)()
        if isinstance(mask.dtype, pd.SparseDtype):
            mask = np.array(mask)
            mask.flags.writeable = True
        mask[:] = True
        tm.assert_series_equal(result, expected)

    def test_dropna_array(self, data_missing):
        result: ExtensionArray = data_missing.dropna()
        expected: ExtensionArray = data_missing[[1]]
        tm.assert_extension_array_equal(result, expected)

    def test_dropna_series(self, data_missing):
        ser: Series = pd.Series(data_missing)
        result: Series = ser.dropna()
        expected: Series = ser.iloc[[1]]
        tm.assert_series_equal(result, expected)

    def test_dropna_frame(self, data_missing):
        df: DataFrame = pd.DataFrame({'A': data_missing}, columns=pd.Index(
            ['A'], dtype=object))
        result: DataFrame = df.dropna()
        expected: DataFrame = df.iloc[[1]]
        tm.assert_frame_equal(result, expected)
        result = df.dropna(axis='columns')
        expected = pd.DataFrame(index=pd.RangeIndex(2), columns=pd.Index([],
            dtype=object))
        tm.assert_frame_equal(result, expected)
        df = pd.DataFrame({'A': data_missing, 'B': [1, np.nan]})
        result = df.dropna()
        expected = df.iloc[:0]
        tm.assert_frame_equal(result, expected)

    def test_fillna_scalar(self, data_missing):
        valid: Any = data_missing[1]
        result: ExtensionArray = data_missing.fillna(valid)
        expected: ExtensionArray = data_missing.fillna(valid)
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_with_none(self, data_missing):
        result: ExtensionArray = data_missing.fillna(None)
        expected: ExtensionArray = data_missing
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_limit_pad(self, data_missing):
        arr: ExtensionArray = data_missing.take([1, 0, 0, 0, 1])
        result: Series = pd.Series(arr).ffill(limit=2)
        expected: Series = pd.Series(data_missing.take([1, 1, 1, 0, 1]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('limit_area, input_ilocs, expected_ilocs', [(
        'outside', [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]), ('outside', [1, 0, 1,
        0, 1], [1, 0, 1, 0, 1]), ('outside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 
        1]), ('outside', [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]), ('inside', [1, 
        0, 0, 0, 1], [1, 1, 1, 1, 1]), ('inside', [1, 0, 1, 0, 1], [1, 1, 1,
        1, 1]), ('inside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]), ('inside', [0,
        1, 0, 1, 0], [0, 1, 1, 1, 0])])
    def test_ffill_limit_area(self, data_missing, limit_area, input_ilocs,
        expected_ilocs):
        arr: ExtensionArray = data_missing.take(input_ilocs)
        result: Series = pd.Series(arr).ffill(limit_area=limit_area)
        expected: Series = pd.Series(data_missing.take(expected_ilocs))
        tm.assert_series_equal(result, expected)

    def test_fillna_limit_backfill(self, data_missing):
        arr: ExtensionArray = data_missing.take([1, 0, 0, 0, 1])
        result: Series = pd.Series(arr).bfill(limit=2)
        expected: Series = pd.Series(data_missing.take([1, 0, 1, 1, 1]))
        tm.assert_series_equal(result, expected)

    def test_fillna_no_op_returns_copy(self, data):
        data = data[~data.isna()]
        valid: Any = data[0]
        result: ExtensionArray = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)
        result = data._pad_or_backfill(method='backfill')
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    def test_fillna_series(self, data_missing):
        fill_value: Any = data_missing[1]
        ser: Series = pd.Series(data_missing)
        result: Series = ser.fillna(fill_value)
        expected: Series = pd.Series(data_missing._from_sequence([
            fill_value, fill_value], dtype=data_missing.dtype))
        tm.assert_series_equal(result, expected)
        result = ser.fillna(expected)
        tm.assert_series_equal(result, expected)
        result = ser.fillna(ser)
        tm.assert_series_equal(result, ser)

    @pytest.mark.parametrize('fillna_method', ['ffill', 'bfill'])
    def test_fillna_series_method(self, data_missing, fillna_method):
        fill_value: Any = data_missing[1]
        if fillna_method == 'ffill':
            data_missing = data_missing[::-1]
        result: Series = getattr(pd.Series(data_missing), fillna_method)()
        expected: Series = pd.Series(data_missing._from_sequence([
            fill_value, fill_value], dtype=data_missing.dtype))
        tm.assert_series_equal(result, expected)

    def test_fillna_frame(self, data_missing):
        fill_value: Any = data_missing[1]
        result: DataFrame = pd.DataFrame({'A': data_missing, 'B': [1, 2]}
            ).fillna(fill_value)
        expected: DataFrame = pd.DataFrame({'A': data_missing.
            _from_sequence([fill_value, fill_value], dtype=data_missing.
            dtype), 'B': [1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_fillna_fill_other(self, data):
        result: DataFrame = pd.DataFrame({'A': data, 'B': [np.nan] * len(data)}
            ).fillna({'B': 0.0})
        expected: DataFrame = pd.DataFrame({'A': data, 'B': [0.0] * len(
            result)})
        tm.assert_frame_equal(result, expected)
