import inspect
import operator
from typing import Any, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pytest

from pandas._typing import Dtype

from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype

import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort


class BaseMethodsTests:
    """Various Series and DataFrame methods."""

    def test_hash_pandas_object(self, data: pd.Series) -> None:
        # _hash_pandas_object should return a uint64 ndarray of the same length
        # as the data
        from pandas.core.util.hashing import _default_hash_key

        res = data._hash_pandas_object(
            encoding="utf-8", hash_key=_default_hash_key, categorize=False
        )
        assert res.dtype == np.uint64
        assert res.shape == data.shape

    def test_value_counts_default_dropna(self, data: pd.Series) -> None:
        # make sure we have consistent default dropna kwarg
        if not hasattr(data, "value_counts"):
            pytest.skip(f"value_counts is not implemented for {type(data)}")
        sig = inspect.signature(data.value_counts)
        kwarg = sig.parameters["dropna"]
        assert kwarg.default is True

    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data: pd.Series, dropna: bool) -> None:
        all_data = all_data[:10]
        if dropna:
            other = all_data[~all_data.isna()]
        else:
            other = all_data

        result = pd.Series(all_data).value_counts(dropna=dropna).sort_index()
        expected = pd.Series(other).value_counts(dropna=dropna).sort_index()

        tm.assert_series_equal(result, expected)

    def test_value_counts_with_normalize(self, data: pd.Series) -> None:
        # GH 33172
        data = data[:10].unique()
        values = np.array(data[~data.isna()])
        ser = pd.Series(data, dtype=data.dtype)

        result = ser.value_counts(normalize=True).sort_index()

        if not isinstance(data, pd.Categorical):
            expected = pd.Series(
                [1 / len(values)] * len(values), index=result.index, name="proportion"
            )
        else:
            expected = pd.Series(0.0, index=result.index, name="proportion")
            expected[result > 0] = 1 / len(values)

        if isinstance(data.dtype, pd.StringDtype) and data.dtype.na_value is np.nan:
            # TODO: avoid special-casing
            expected = expected.astype("float64")
        elif getattr(data.dtype, "storage", "") == "pyarrow" or isinstance(
            data.dtype, pd.ArrowDtype
        ):
            # TODO: avoid special-casing
            expected = expected.astype("double[pyarrow]")
        elif na_value_for_dtype(data.dtype) is pd.NA:
            # TODO(GH#44692): avoid special-casing
            expected = expected.astype("Float64")

        tm.assert_series_equal(result, expected)

    def test_count(self, data_missing: pd.Series) -> None:
        df = pd.DataFrame({"A": data_missing})
        result = df.count(axis="columns")
        expected = pd.Series([0, 1])
        tm.assert_series_equal(result, expected)

    def test_series_count(self, data_missing: pd.Series) -> None:
        # GH#26835
        ser = pd.Series(data_missing)
        result = ser.count()
        expected = 1
        assert result == expected

    def test_apply_simple_series(self, data: pd.Series) -> None:
        result = pd.Series(data).apply(id)
        assert isinstance(result, pd.Series)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data_missing: pd.Series, na_action: Optional[str]) -> None:
        result = data_missing.map(lambda x: x, na_action=na_action)
        expected = data_missing.to_numpy()
        tm.assert_numpy_array_equal(result, expected)

    def test_argsort(self, data_for_sorting: pd.Series) -> None:
        result = pd.Series(data_for_sorting).argsort()
        # argsort result gets passed to take, so should be np.intp
        expected = pd.Series(np.array([2, 0, 1], dtype=np.intp))
        tm.assert_series_equal(result, expected)

    def test_argsort_missing_array(self, data_missing_for_sorting: pd.Series) -> None:
        result = data_missing_for_sorting.argsort()
        # argsort result gets passed to take, so should be np.intp
        expected = np.array([2, 0, 1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

    def test_argsort_missing(self, data_missing_for_sorting: pd.Series) -> None:
        result = pd.Series(data_missing_for_sorting).argsort()
        expected = pd.Series(np.array([2, 0, 1], dtype=np.intp))
        tm.assert_series_equal(result, expected)

    def test_argmin_argmax(
        self,
        data_for_sorting: pd.Series,
        data_missing_for_sorting: pd.Series,
        na_value: Any,
    ) -> None:
        # GH 24382
        is_bool = data_for_sorting.dtype._is_boolean

        exp_argmax = 1
        exp_argmax_repeated = 3
        if is_bool:
            # See data_for_sorting docstring
            exp_argmax = 0
            exp_argmax_repeated = 1

        # data_for_sorting -> [B, C, A] with A < B < C
        assert data_for_sorting.argmax() == exp_argmax
        assert data_for_sorting.argmin() == 2

        # with repeated values -> first occurrence
        data = data_for_sorting.take([2, 0, 0, 1, 1, 2])
        assert data.argmax() == exp_argmax_repeated
        assert data.argmin() == 0

        # with missing values
        # data_missing_for_sorting -> [B, NA, A] with A < B and NA missing.
        assert data_missing_for_sorting.argmax() == 0
        assert data_missing_for_sorting.argmin() == 2

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_empty_array(
        self, method: str, data: pd.Series
    ) -> None:
        # GH 24382
        err_msg = "attempt to get"
        with pytest.raises(ValueError, match=err_msg):
            getattr(data[:0], method)()

    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_all_na(
        self, method: str, data: pd.Series, na_value: Any
    ) -> None:
        # all missing with skipna=True is the same as empty
        err_msg = "attempt to get"
        data_na = type(data)._from_sequence([na_value, na_value], dtype=data.dtype)
        with pytest.raises(ValueError, match=err_msg):
            getattr(data_na, method)()

    @pytest.mark.parametrize(
        "op_name, skipna, expected",
        [
            ("idxmax", True, 0),
            ("idxmin", True, 2),
            ("argmax", True, 0),
            ("argmin", True, 2),
            ("idxmax", False, -1),
            ("idxmin", False, -1),
            ("argmax", False, -1),
            ("argmin", False, -1),
        ],
    )
    def test_argreduce_series(
        self,
        data_missing_for_sorting: pd.Series,
        op_name: str,
        skipna: bool,
        expected: int,
    ) -> None:
        # data_missing_for_sorting -> [B, NA, A] with A < B and NA missing.
        ser = pd.Series(data_missing_for_sorting)
        if expected == -1:
            with pytest.raises(ValueError, match="Encountered an NA value"):
                getattr(ser, op_name)(skipna=skipna)
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            tm.assert_almost_equal(result, expected)

    def test_argmax_argmin_no_skipna_notimplemented(
        self, data_missing_for_sorting: pd.Series
    ) -> None:
        # GH#38733
        data = data_missing_for_sorting

        with pytest.raises(ValueError, match="Encountered an NA value"):
            data.argmin(skipna=False)

        with pytest.raises(ValueError, match="Encountered an NA value"):
            data.argmax(skipna=False)

    @pytest.mark.parametrize(
        "na_position, expected",
        [
            ("last", np.array([2, 0, 1], dtype=np.dtype("intp"))),
            ("first", np.array([1, 2, 0], dtype=np.dtype("intp"))),
        ],
    )
    def test_nargsort(
        self,
        data_missing_for_sorting: pd.Series,
        na_position: str,
        expected: np.ndarray,
    ) -> None:
        # GH 25439
        result = nargsort(data_missing_for_sorting, na_position=na_position)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(
        self,
        data_for_sorting: pd.Series,
        ascending: bool,
        sort_by_key: Any,
    ) -> None:
        ser = pd.Series(data_for_sorting)
        result = ser.sort_values(ascending=ascending, key=sort_by_key)
        expected = ser.iloc[[2, 0, 1]]
        if not ascending:
            # GH 35922. Expect stable sort
            if ser.nunique() == 2:
                expected = ser.iloc[[0, 1, 2]]
            else:
                expected = ser.iloc[[1, 0, 2]]

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_missing(
        self,
        data_missing_for_sorting: pd.Series,
        ascending: bool,
        sort_by_key: Any,
    ) -> None:
        ser = pd.Series(data_missing_for_sorting)
        result = ser.sort_values(ascending=ascending, key=sort_by_key)
        if ascending:
            expected = ser.iloc[[2, 0, 1]]
        else:
            expected = ser.iloc[[0, 2, 1]]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values_frame(self, data_for_sorting: pd.Series, ascending: bool) -> None:
        df = pd.DataFrame({"A": [1, 2, 1], "B": data_for_sorting})
        result = df.sort_values(["A", "B"])
        expected = pd.DataFrame(
            {"A": [1, 1, 2], "B": data_for_sorting.take([2, 0, 1])}, index=[2, 0, 1]
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("keep", ["first", "last", False])
    def test_duplicated(self, data: pd.Series, keep: Union[str, bool]) -> None:
        arr = data.take([0, 1, 0, 1])
        result = arr.duplicated(keep=keep)
        if keep == "first":
            expected = np.array([False, False, True, True])
        elif keep == "last":
            expected = np.array([True, True, False, False])
        else:
            expected = np.array([True, True, True, True])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("box", [pd.Series, lambda x: x])
    @pytest.mark.parametrize("method", [lambda x: x.unique(), pd.unique])
    def test_unique(self, data: pd.Series, box: Any, method: Any) -> None:
        duplicated = box(data._from_sequence([data[0], data[0]], dtype=data.dtype))

        result = method(duplicated)

        assert len(result) == 1
        assert isinstance(result, type(data))
        assert result[0] == duplicated[0]

    def test_factorize(self, data_for_grouping: pd.Series) -> None:
        codes, uniques = pd.factorize(data_for_grouping, use_na_sentinel=True)

        is_bool = data_for_grouping.dtype._is_boolean
        if is_bool:
            # only 2 unique values
            expected_codes = np.array([0, 0, -1, -1, 1, 1, 0, 0], dtype=np.intp)
            expected_uniques = data_for_grouping.take([0, 4])
        else:
            expected_codes = np.array([0, 0, -1, -1, 1, 1, 0, 2], dtype=np.intp)
            expected_uniques = data_for_grouping.take([0, 4, 7])

        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_extension_array_equal(uniques, expected_uniques)

    def test_factorize_equivalence(self, data_for_grouping: pd.Series) -> None:
        codes_1, uniques_1 = pd.factorize(data_for_grouping, use_na_sentinel=True)
        codes_2, uniques_2 = data_for_grouping.factorize(use_na_sentinel=True)

        tm.assert_numpy_array_equal(codes_1, codes_2)
        tm.assert_extension_array_equal(uniques_1, uniques_2)
        assert len(uniques_1) == len(pd.unique(uniques_1))
        assert uniques_1.dtype == data_for_grouping.dtype

    def test_factorize_empty(self, data: pd.Series) -> None:
        codes, uniques = pd.factorize(data[:0])
        expected_codes = np.array([], dtype=np.intp)
        expected_uniques = type(data)._from_sequence([], dtype=data[:0].dtype)

        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_extension_array_equal(uniques, expected_uniques)

    def test_fillna_limit_frame(self, data_missing: pd.Series) -> None:
        # GH#58001
        df = pd.DataFrame({"A": data_missing.take([0, 1, 0, 1])})
        expected = pd.DataFrame({"A": data_missing.take([1, 1, 0, 1])})
        result = df.fillna(value=data_missing[1], limit=1)
        tm.assert_frame_equal(result, expected)

    def test_fillna_limit_series(self, data_missing: pd.Series) -> None:
        # GH#58001
        ser = pd.Series(data_missing.take([0, 1, 0, 1]))
        expected = pd.Series(data_missing.take([1, 1, 0, 1]))
        result = ser.fillna(value=data_missing[1], limit=1)
        tm.assert_series_equal(result, expected)

    def test_fillna_copy_frame(self, data_missing: pd.Series) -> None:
        arr = data_missing.take([1, 1])
        df = pd.DataFrame({"A": arr})
        df_orig = df.copy()

        filled_val = df.iloc[0, 0]
        result = df.fillna(filled_val)

        result.iloc[0, 0] = filled_val

        tm.assert_frame_equal(df, df_orig)

    def test_fillna_copy_series(self, data_missing: pd.Series) -> None:
        arr = data_missing.take([1, 1])
        ser = pd.Series(arr, copy=False)
        ser_orig = ser.copy()

        filled_val = ser[0]
        result = ser.fillna(filled_val)
        result.iloc[0] = filled_val

        tm.assert_series_equal(ser, ser_orig)

    def test_fillna_length_mismatch(self, data_missing: pd.Series) -> None:
        msg = "Length of 'value' does not match."
        with pytest.raises(ValueError, match=msg):
            data_missing.fillna(data_missing.take([1]))

    # Subclasses can override if we expect e.g Sparse[bool], boolean, pyarrow[bool]
    _combine_le_expected_dtype: Dtype = NumpyEADtype("bool")

    def test_combine_le(self, data_repeated: Any) -> None:
        # GH 20825
        # Test that combine works when doing a <= (le) comparison
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            pd.array(
                [a <= b for (a, b) in zip(list(orig_data1), list(orig_data2))],
                dtype=self._combine_le_expected_dtype,
            )
        )
        tm.assert_series_equal(result, expected)

        val = s1