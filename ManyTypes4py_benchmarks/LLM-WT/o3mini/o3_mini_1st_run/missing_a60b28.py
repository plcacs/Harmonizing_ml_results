import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from typing import Any, List

class BaseMissingTests:

    def test_isna(self, data_missing: ExtensionArray) -> None:
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
    def test_isna_returns_copy(self, data_missing: ExtensionArray, na_func: str) -> None:
        result = pd.Series(data_missing)
        expected = result.copy()
        mask = getattr(result, na_func)()
        if isinstance(mask.dtype, pd.SparseDtype):
            mask = np.array(mask)
            mask.flags.writeable = True
        mask[:] = True
        tm.assert_series_equal(result, expected)

    def test_dropna_array(self, data_missing: ExtensionArray) -> None:
        result = data_missing.dropna()
        expected = data_missing[[1]]
        tm.assert_extension_array_equal(result, expected)

    def test_dropna_series(self, data_missing: ExtensionArray) -> None:
        ser = pd.Series(data_missing)
        result = ser.dropna()
        expected = ser.iloc[[1]]
        tm.assert_series_equal(result, expected)

    def test_dropna_frame(self, data_missing: ExtensionArray) -> None:
        df = pd.DataFrame({'A': data_missing}, columns=pd.Index(['A'], dtype=object))
        result = df.dropna()
        expected = df.iloc[[1]]
        tm.assert_frame_equal(result, expected)
        result = df.dropna(axis='columns')
        expected = pd.DataFrame(index=pd.RangeIndex(2), columns=pd.Index([]))
        tm.assert_frame_equal(result, expected)
        df = pd.DataFrame({'A': data_missing, 'B': [1, np.nan]})
        result = df.dropna()
        expected = df.iloc[:0]
        tm.assert_frame_equal(result, expected)

    def test_fillna_scalar(self, data_missing: ExtensionArray) -> None:
        valid: Any = data_missing[1]
        result = data_missing.fillna(valid)
        expected = data_missing.fillna(valid)
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_with_none(self, data_missing: ExtensionArray) -> None:
        result = data_missing.fillna(None)
        expected = data_missing
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_limit_pad(self, data_missing: ExtensionArray) -> None:
        arr = data_missing.take([1, 0, 0, 0, 1])
        result = pd.Series(arr).ffill(limit=2)
        expected = pd.Series(data_missing.take([1, 1, 1, 0, 1]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'limit_area, input_ilocs, expected_ilocs',
        [
            ('outside', [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),
            ('outside', [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]),
            ('outside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]),
            ('outside', [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]),
            ('inside', [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]),
            ('inside', [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]),
            ('inside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]),
            ('inside', [0, 1, 0, 1, 0], [0, 1, 1, 1, 0])
        ]
    )
    def test_ffill_limit_area(
        self,
        data_missing: ExtensionArray,
        limit_area: str,
        input_ilocs: List[int],
        expected_ilocs: List[int]
    ) -> None:
        arr = data_missing.take(input_ilocs)
        result = pd.Series(arr).ffill(limit_area=limit_area)
        expected = pd.Series(data_missing.take(expected_ilocs))
        tm.assert_series_equal(result, expected)

    def test_fillna_limit_backfill(self, data_missing: ExtensionArray) -> None:
        arr = data_missing.take([1, 0, 0, 0, 1])
        result = pd.Series(arr).bfill(limit=2)
        expected = pd.Series(data_missing.take([1, 0, 1, 1, 1]))
        tm.assert_series_equal(result, expected)

    def test_fillna_no_op_returns_copy(self, data: ExtensionArray) -> None:
        data = data[~data.isna()]
        valid: Any = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)
        result = data._pad_or_backfill(method='backfill')
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    def test_fillna_series(self, data_missing: ExtensionArray) -> None:
        fill_value: Any = data_missing[1]
        ser = pd.Series(data_missing)
        result = ser.fillna(fill_value)
        expected = pd.Series(
            data_missing._from_sequence([fill_value, fill_value], dtype=data_missing.dtype)
        )
        tm.assert_series_equal(result, expected)
        result = ser.fillna(expected)
        tm.assert_series_equal(result, expected)
        result = ser.fillna(ser)
        tm.assert_series_equal(result, ser)

    def test_fillna_series_method(self, data_missing: ExtensionArray, fillna_method: str) -> None:
        fill_value: Any = data_missing[1]
        if fillna_method == 'ffill':
            data_missing = data_missing[::-1]
        result = getattr(pd.Series(data_missing), fillna_method)()
        expected = pd.Series(
            data_missing._from_sequence([fill_value, fill_value], dtype=data_missing.dtype)
        )
        tm.assert_series_equal(result, expected)

    def test_fillna_frame(self, data_missing: ExtensionArray) -> None:
        fill_value: Any = data_missing[1]
        result = pd.DataFrame({'A': data_missing, 'B': [1, 2]}).fillna(fill_value)
        expected = pd.DataFrame({
            'A': data_missing._from_sequence([fill_value, fill_value], dtype=data_missing.dtype),
            'B': [1, 2]
        })
        tm.assert_frame_equal(result, expected)

    def test_fillna_fill_other(self, data: ExtensionArray) -> None:
        result = pd.DataFrame({'A': data, 'B': [np.nan] * len(data)}).fillna({'B': 0.0})
        expected = pd.DataFrame({'A': data, 'B': [0.0] * len(result)})
        tm.assert_frame_equal(result, expected)