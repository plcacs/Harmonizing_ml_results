from datetime import (
    datetime,
    timedelta,
)

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest

from pandas._libs.algos import (
    Infinity,
    NegInfinity,
)

from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


class TestRank:
    s: Series = Series([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])
    df: DataFrame = DataFrame({"A": s, "B": s})
    results: Dict[str, np.ndarray] = {
        "average": np.array([1.5, 5.5, 7.0, 3.5, np.nan, 3.5, 1.5, 8.0, np.nan, 5.5]),
        "min": np.array([1, 5, 7, 3, np.nan, 3, 1, 8, np.nan, 5]),
        "max": np.array([2, 6, 7, 4, np.nan, 4, 2, 8, np.nan, 6]),
        "first": np.array([1, 5, 7, 3, np.nan, 4, 2, 8, np.nan, 6]),
        "dense": np.array([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3]),
    }

    def test_rank(self, float_frame: DataFrame) -> None:
        sp_stats = pytest.importorskip("scipy.stats")

        float_frame.loc[::2, "A"] = np.nan
        float_frame.loc[::3, "B"] = np.nan
        float_frame.loc[::4, "C"] = np.nan
        float_frame.loc[::5, "D"] = np.nan

        ranks0: DataFrame = float_frame.rank()
        ranks1: DataFrame = float_frame.rank(1)
        mask = np.isnan(float_frame.values)

        fvals = float_frame.fillna(np.inf).values

        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fvals)
        exp0[mask] = np.nan

        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fvals)
        exp1[mask] = np.nan

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # integers
        df: DataFrame = DataFrame(
            np.random.default_rng(2).integers(0, 5, size=40).reshape((10, 4))
        )

        result: DataFrame = df.rank()
        exp: DataFrame = df.astype(float).rank()
        tm.assert_frame_equal(result, exp)

        result = df.rank(1)
        exp = df.astype(float).rank(1)
        tm.assert_frame_equal(result, exp)

    def test_rank2(self) -> None:
        df: DataFrame = DataFrame([[1, 3, 2], [1, 2, 3]])
        expected: DataFrame = DataFrame([[1.0, 3.0, 2.0], [1, 2, 3]]) / 3.0
        result: DataFrame = df.rank(1, pct=True)
        tm.assert_frame_equal(result, expected)

        df = DataFrame([[1, 3, 2], [1, 2, 3]])
        expected = df.rank(0) / 2.0
        result = df.rank(0, pct=True)
        tm.assert_frame_equal(result, expected)

        df = DataFrame([["b", "c", "a"], ["a", "c", "b"]])
        expected = DataFrame([[2.0, 3.0, 1.0], [1, 3, 2]])
        result = df.rank(1, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[2.0, 1.5, 1.0], [1, 1.5, 2]])
        result = df.rank(0, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        df = DataFrame([["b", np.nan, "a"], ["a", "c", "b"]])
        expected = DataFrame([[2.0, np.nan, 1.0], [1.0, 3.0, 2.0]])
        result = df.rank(1, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[2.0, np.nan, 1.0], [1.0, 1.0, 2.0]])
        result = df.rank(0, numeric_only=False)
        tm.assert_frame_equal(result, expected)

        # f7u12, this does not work without extensive workaround
        data: List[List[Any]] = [
            [datetime(2001, 1, 5), np.nan, datetime(2001, 1, 2)],
            [datetime(2000, 1, 2), datetime(2000, 1, 3), datetime(2000, 1, 1)],
        ]
        df = DataFrame(data)

        # check the rank
        expected = DataFrame([[2.0, np.nan, 1.0], [2.0, 3.0, 1.0]])
        result = df.rank(1, numeric_only=False, ascending=True)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[1.0, np.nan, 2.0], [2.0, 1.0, 3.0]])
        result = df.rank(1, numeric_only=False, ascending=False)
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"a": [1e-20, -5, 1e-20 + 1e-40, 10, 1e60, 1e80, 1e-30]})
        exp = DataFrame({"a": [3.5, 1.0, 3.5, 5.0, 6.0, 7.0, 2.0]})
        tm.assert_frame_equal(df.rank(), exp)

    def test_rank_does_not_mutate(self) -> None:
        # GH#18521
        # Check rank does not mutate DataFrame
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), dtype="float64"
        )
        expected: DataFrame = df.copy()
        df.rank()
        result: DataFrame = df
        tm.assert_frame_equal(result, expected)

    def test_rank_mixed_frame(self, float_string_frame: DataFrame) -> None:
        float_string_frame["datetime"] = datetime.now()
        float_string_frame["timedelta"] = timedelta(days=1, seconds=1)

        float_string_frame.rank(numeric_only=False)
        with pytest.raises(TypeError, match="not supported between instances of"):
            float_string_frame.rank(axis=1)

    def test_rank_na_option(self, float_frame: DataFrame) -> None:
        sp_stats = pytest.importorskip("scipy.stats")

        float_frame.loc[::2, "A"] = np.nan
        float_frame.loc[::3, "B"] = np.nan
        float_frame.loc[::4, "C"] = np.nan
        float_frame.loc[::5, "D"] = np.nan

        # bottom
        ranks0: DataFrame = float_frame.rank(na_option="bottom")
        ranks1: DataFrame = float_frame.rank(1, na_option="bottom")

        fvals = float_frame.fillna(np.inf).values

        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fvals)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fvals)

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # top
        ranks0 = float_frame.rank(na_option="top")
        ranks1 = float_frame.rank(1, na_option="top")

        fval0 = float_frame.fillna((float_frame.min() - 1).to_dict()).values
        fval1 = float_frame.T
        fval1 = fval1.fillna((fval1.min() - 1).to_dict()).T
        fval1 = fval1.fillna(np.inf).values

        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, fval0)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, fval1)

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # descending

        # bottom
        ranks0 = float_frame.rank(na_option="top", ascending=False)
        ranks1 = float_frame.rank(1, na_option="top", ascending=False)

        fvals = float_frame.fillna(np.inf).values

        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, -fvals)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, -fvals)

        tm.assert_almost_equal(ranks0.values, exp0)
        tm.assert_almost_equal(ranks1.values, exp1)

        # descending

        # top
        ranks0 = float_frame.rank(na_option="bottom", ascending=False)
        ranks1 = float_frame.rank(1, na_option="bottom", ascending=False)

        fval0 = float_frame.fillna((float_frame.min() - 1).to_dict()).values
        fval1 = float_frame.T
        fval1 = fval1.fillna((fval1.min() - 1).to_dict()).T
        fval1 = fval1.fillna(np.inf).values

        exp0 = np.apply_along_axis(sp_stats.rankdata, 0, -fval0)
        exp1 = np.apply_along_axis(sp_stats.rankdata, 1, -fval1)

        tm.assert_numpy_array_equal(ranks0.values, exp0)
        tm.assert_numpy_array_equal(ranks1.values, exp1)

        # bad values throw error
        msg = "na_option must be one of 'keep', 'top', or 'bottom'"

        with pytest.raises(ValueError, match=msg):
            float_frame.rank(na_option="bad", ascending=False)

        # invalid type
        with pytest.raises(ValueError, match=msg):
            float_frame.rank(na_option=True, ascending=False)

    def test_rank_axis(self) -> None:
        # check if using axes' names gives the same result
        df: DataFrame = DataFrame([[2, 1], [4, 3]])
        tm.assert_frame_equal(df.rank(axis=0), df.rank(axis="index"))
        tm.assert_frame_equal(df.rank(axis=1), df.rank(axis="columns"))

    @pytest.mark.parametrize("ax", [0, 1])
    def test_rank_methods_frame(self, ax: int, rank_method: str) -> None:
        sp_stats = pytest.importorskip("scipy.stats")

        xs = np.random.default_rng(2).integers(0, 21, (100, 26))
        xs = (xs - 10.0) / 10.0
        cols = [chr(ord("z") - i) for i in range(xs.shape[1])]

        for vals in [xs, xs + 1e6, xs * 1e-6]:
            df: DataFrame = DataFrame(vals, columns=cols)

            result = df.rank(axis=ax, method=rank_method)
            sprank = np.apply_along_axis(
                sp_stats.rankdata,
                ax,
                vals,
                rank_method if rank_method != "first" else "ordinal",
            )
            sprank = sprank.astype(np.float64)
            expected = DataFrame(sprank, columns=cols).astype("float64")
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["O", "f8", "i8"])
    def test_rank_descending(self, rank_method: str, dtype: str) -> None:
        if "i" in dtype:
            df: DataFrame = self.df.dropna().astype(dtype)
        else:
            df = self.df.astype(dtype)

        res = df.rank(ascending=False)
        expected = (df.max() - df).rank()
        tm.assert_frame_equal(res, expected)

        expected = (df.max() - df).rank(method=rank_method)

        if dtype != "O":
            res2 = df.rank(method=rank_method, ascending=False, numeric_only=True)
            tm.assert_frame_equal(res2, expected)

        res3 = df.rank(method=rank_method, ascending=False, numeric_only=False)
        tm.assert_frame_equal(res3, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("dtype", [None, object])
    def test_rank_2d_tie_methods(self, rank_method: str, axis: int, dtype: Optional[Any]) -> None:
        df = self.df

        def _check2d(
            df_inner: DataFrame, expected: np.ndarray, method: str = "average", axis: int = 0
        ) -> None:
            exp_df = DataFrame({"A": expected, "B": expected})

            if axis == 1:
                df_inner = df_inner.T
                exp_df = exp_df.T

            result = df_inner.rank(method=rank_method, axis=axis)
            tm.assert_frame_equal(result, exp_df)

        frame: DataFrame = df if dtype is None else df.astype(dtype)
        _check2d(frame, self.results[rank_method], method=rank_method, axis=axis)

    @pytest.mark.parametrize(
        "rank_method,exp",
        [
            ("dense", [[1.0, 1.0, 1.0], [1.0, 0.5, 2.0 / 3], [1.0, 0.5, 1.0 / 3]]),
            (
                "min",
                [
                    [1.0 / 3, 1.0, 1.0],
                    [1.0 / 3, 1.0 / 3, 2.0 / 3],
                    [1.0 / 3, 1.0 / 3, 1.0 / 3],
                ],
            ),
            (
                "max",
                [[1.0, 1.0, 1.0], [1.0, 2.0 / 3, 2.0 / 3], [1.0, 2.0 / 3, 1.0 / 3]],
            ),
            (
                "average",
                [[2.0 / 3, 1.0, 1.0], [2.0 / 3, 0.5, 2.0 / 3], [2.0 / 3, 0.5, 1.0 / 3]],
            ),
            (
                "first",
                [
                    [1.0 / 3, 1.0, 1.0],
                    [2.0 / 3, 1.0 / 3, 2.0 / 3],
                    [3.0 / 3, 2.0 / 3, 1.0 / 3],
                ],
            ),
        ],
    )
    def test_rank_pct_true(self, rank_method: str, exp: List[List[float]]) -> None:
        # see gh-15630.

        df: DataFrame = DataFrame([[2012, 66, 3], [2012, 65, 2], [2012, 65, 1]])
        result = df.rank(method=rank_method, pct=True)

        expected = DataFrame(exp)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.single_cpu
    def test_pct_max_many_rows(self) -> None:
        # GH 18271
        df: DataFrame = DataFrame({"A": np.arange(2**24 + 1), "B": np.arange(2**24 + 1, 0, -1)})
        result = df.rank(pct=True).max()
        assert (result == 1).all()

    @pytest.mark.parametrize(
        "contents,dtype",
        [
            (
                [
                    -np.inf,
                    -50,
                    -1,
                    -1e-20,
                    -1e-25,
                    -1e-50,
                    0,
                    1e-40,
                    1e-20,
                    1e-10,
                    2,
                    40,
                    np.inf,
                ],
                "float64",
            ),
            (
                [
                    -np.inf,
                    -50,
                    -1,
                    -1e-20,
                    -1e-25,
                    -1e-45,
                    0,
                    1e-40,
                    1e-20,
                    1e-10,
                    2,
                    40,
                    np.inf,
                ],
                "float32",
            ),
            ([np.iinfo(np.uint8).min, 1, 2, 100, np.iinfo(np.uint8).max], "uint8"),
            (
                [
                    np.iinfo(np.int64).min,
                    -100,
                    0,
                    1,
                    9999,
                    100000,
                    1e10,
                    np.iinfo(np.int64).max,
                ],
                "int64",
            ),
            ([NegInfinity(), "1", "A", "BA", "Ba", "C", Infinity()], "object"),
            (
                [datetime(2001, 1, 1), datetime(2001, 1, 2), datetime(2001, 1, 5)],
                "datetime64",
            ),
        ],
    )
    def test_rank_inf_and_nan(
        self,
        contents: List[Any],
        dtype: str,
        frame_or_series: Callable[[np.ndarray], Union[Series, DataFrame]],
    ) -> None:
        dtype_na_map: Dict[str, Any] = {
            "float64": np.nan,
            "float32": np.nan,
            "object": None,
            "datetime64": np.datetime64("nat"),
        }
        # Insert nans at random positions if underlying dtype has missing
        # value. Then adjust the expected order by adding nans accordingly
        # This is for testing whether rank calculation is affected
        # when values are intertwined with nan values.
        values: np.ndarray = np.array(contents, dtype=dtype)
        exp_order: np.ndarray = np.array(range(len(values)), dtype="float64") + 1.0
        if dtype in dtype_na_map:
            na_value = dtype_na_map[dtype]
            nan_indices = np.random.default_rng(2).choice(range(len(values)), 5)
            values = np.insert(values, nan_indices, na_value)
            exp_order = np.insert(exp_order, nan_indices, np.nan)

        # Shuffle the testing array and expected results in the same way
        random_order = np.random.default_rng(2).permutation(len(values))
        obj = frame_or_series(values[random_order])
        expected_obj = frame_or_series(exp_order[random_order],)
        result = obj.rank()
        tm.assert_equal(result, expected_obj)

    def test_df_series_inf_nan_consistency(self) -> None:
        # GH#32593
        index: List[int] = [5, 4, 3, 2, 1, 6, 7, 8, 9, 10]
        col1: List[Union[int, float]] = [5, 4, 3, 5, 8, 5, 2, 1, 6, 6]
        col2: List[Union[float, int]] = [5, 4, np.nan, 5, 8, 5, np.inf, np.nan, 6, -np.inf]
        df: DataFrame = DataFrame(
            data={
                "col1": col1,
                "col2": col2,
            },
            index=index,
            dtype="f8",
        )
        df_result: DataFrame = df.rank()

        series_result: DataFrame = df.copy()
        series_result["col1"] = df["col1"].rank()
        series_result["col2"] = df["col2"].rank()

        tm.assert_frame_equal(df_result, series_result)

    def test_rank_both_inf(self) -> None:
        # GH#32593
        df: DataFrame = DataFrame({"a": [-np.inf, 0, np.inf]})
        expected: DataFrame = DataFrame({"a": [1.0, 2.0, 3.0]})
        result: DataFrame = df.rank()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "na_option,ascending,expected",
        [
            ("top", True, [3.0, 1.0, 2.0]),
            ("top", False, [2.0, 1.0, 3.0]),
            ("bottom", True, [2.0, 3.0, 1.0]),
            ("bottom", False, [1.0, 3.0, 2.0]),
        ],
    )
    def test_rank_inf_nans_na_option(
        self,
        frame_or_series: Callable[[np.ndarray], Union[Series, DataFrame]],
        rank_method: str,
        na_option: str,
        ascending: bool,
        expected: List[float],
    ) -> None:
        obj = frame_or_series([np.inf, np.nan, -np.inf])
        result = obj.rank(method=rank_method, na_option=na_option, ascending=ascending)
        expected_obj = frame_or_series(expected)
        tm.assert_equal(result, expected_obj)

    @pytest.mark.parametrize(
        "na_option,ascending,expected",
        [
            ("bottom", True, [1.0, 2.0, 4.0, 3.0]),
            ("bottom", False, [1.0, 2.0, 4.0, 3.0]),
            ("top", True, [2.0, 3.0, 1.0, 4.0]),
            ("top", False, [2.0, 3.0, 1.0, 4.0]),
        ],
    )
    def test_rank_object_first(
        self,
        frame_or_series: Callable[[np.ndarray], Union[Series, DataFrame]],
        na_option: str,
        ascending: bool,
        expected: List[float],
    ) -> None:
        obj = frame_or_series(["foo", "foo", None, "foo"])
        result = obj.rank(method="first", na_option=na_option, ascending=ascending)
        expected_obj = frame_or_series(expected)
        tm.assert_equal(result, expected_obj)

    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                {"a": [1, 2, "a"], "b": [4, 5, 6]},
                DataFrame({"b": [1.0, 2.0, 3.0]}, columns=Index(["b"], dtype=object)),
            ),
            ({"a": [1, 2, "a"]}, DataFrame(index=range(3), columns=[])),
        ],
    )
    def test_rank_mixed_axis_zero(self, data: Dict[str, List[Any]], expected: DataFrame) -> None:
        df: DataFrame = DataFrame(data, columns=Index(list(data.keys()), dtype=object))
        with pytest.raises(TypeError, match="'<' not supported between instances of"):
            df.rank()
        result = df.rank(numeric_only=True)
        tm.assert_frame_equal(result, expected)

    def test_rank_string_dtype(self, string_dtype_no_object: Any) -> None:
        # GH#55362
        obj: Series = Series(["foo", "foo", None, "foo"], dtype=string_dtype_no_object)
        result = obj.rank(method="first")
        exp_dtype: str = (
            "Float64" if string_dtype_no_object == "string[pyarrow]" else "float64"
        )
        if getattr(string_dtype_no_object, "storage", None) == "python":
            # TODO nullable string[python] should also return nullable Int64
            exp_dtype = "float64"
        expected = Series([1, 2, None, 3], dtype=exp_dtype)
        tm.assert_series_equal(result, expected)