from copy import deepcopy
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, concat
import pandas._testing as tm
import pytest


class TestIndexConcat:
    def test_concat_ignore_index(self, sort: bool) -> None:
        frame1: DataFrame = DataFrame(
            {"test1": ["a", "b", "c"], "test2": [1, 2, 3], "test3": [4.5, 3.2, 1.2]}
        )
        frame2: DataFrame = DataFrame({"test3": [5.2, 2.2, 4.3]})
        frame1.index = Index(["x", "y", "z"])
        frame2.index = Index(["x", "y", "q"])
        v1: DataFrame = concat([frame1, frame2], axis=1, ignore_index=True, sort=sort)
        nan = np.nan
        expected: DataFrame = DataFrame(
            [[nan, nan, nan, 4.3], ["a", 1, 4.5, 5.2], ["b", 2, 3.2, 2.2], ["c", 3, 1.2, nan]],
            index=Index(["q", "x", "y", "z"]),
        )
        if not sort:
            expected = expected.loc[["x", "y", "z", "q"]]
        tm.assert_frame_equal(v1, expected)

    @pytest.mark.parametrize(
        "name_in1,name_in2,name_in3,name_out",
        [
            ("idx", "idx", "idx", "idx"),
            ("idx", "idx", None, None),
            ("idx", None, None, None),
            ("idx1", "idx2", None, None),
            ("idx1", "idx1", "idx2", None),
            ("idx1", "idx2", "idx3", None),
            (None, None, None, None),
        ],
    )
    def test_concat_same_index_names(
        self,
        name_in1: Optional[str],
        name_in2: Optional[str],
        name_in3: Optional[str],
        name_out: Optional[str],
    ) -> None:
        indices: List[Index] = [
            Index(["a", "b", "c"], name=name_in1),
            Index(["b", "c", "d"], name=name_in2),
            Index(["c", "d", "e"], name=name_in3),
        ]
        frames: List[DataFrame] = [
            DataFrame({c: [0, 1, 2]}, index=i)
            for i, c in zip(indices, ["x", "y", "z"])
        ]
        result: DataFrame = concat(frames, axis=1)
        exp_ind: Index = Index(["a", "b", "c", "d", "e"], name=name_out)
        expected: DataFrame = DataFrame(
            {"x": [0, 1, 2, np.nan, np.nan], "y": [np.nan, 0, 1, 2, np.nan], "z": [np.nan, np.nan, 0, 1, 2]},
            index=exp_ind,
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_rename_index(self) -> None:
        a: DataFrame = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            columns=list("ABC"),
            index=Index(list("abc"), name="index_a"),
        )
        b: DataFrame = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            columns=list("ABC"),
            index=Index(list("abc"), name="index_b"),
        )
        result: DataFrame = concat([a, b], keys=["key0", "key1"], names=["lvl0", "lvl1"])
        exp: DataFrame = concat([a, b], keys=["key0", "key1"], names=["lvl0"])
        names: List[Optional[str]] = list(exp.index.names)
        names[1] = "lvl1"
        exp.index.set_names(names, inplace=True)
        tm.assert_frame_equal(result, exp)
        assert result.index.names == exp.index.names

    def test_concat_copy_index_series(self, axis: Union[int, str]) -> None:
        ser: Series = Series([1, 2])
        comb: Union[Series, DataFrame] = concat([ser, ser], axis=axis)
        if axis in [0, "index"]:
            assert comb.index is not ser.index
        else:
            assert comb.index is ser.index

    def test_concat_copy_index_frame(self, axis: Union[int, str]) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        comb: DataFrame = concat([df, df], axis=axis)
        if axis in [0, "index"]:
            assert not comb.index.is_(df.index)
            assert comb.columns.is_(df.columns)
        elif axis in [1, "columns"]:
            assert comb.index.is_(df.index)
            assert not comb.columns.is_(df.columns)

    def test_default_index(self) -> None:
        s1: Series = Series([1, 2, 3], name="x")
        s2: Series = Series([4, 5, 6], name="y")
        res: DataFrame = concat([s1, s2], axis=1, ignore_index=True)
        assert isinstance(res.columns, pd.RangeIndex)
        exp: DataFrame = DataFrame([[1, 4], [2, 5], [3, 6]])
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)
        s1 = Series([1, 2, 3])
        s2 = Series([4, 5, 6])
        res = concat([s1, s2], axis=1, ignore_index=False)
        assert isinstance(res.columns, pd.RangeIndex)
        exp = DataFrame([[1, 4], [2, 5], [3, 6]])
        exp.columns = pd.RangeIndex(2)
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)
        df1: DataFrame = DataFrame({"A": [1, 2], "B": [5, 6]})
        df2: DataFrame = DataFrame({"A": [3, 4], "B": [7, 8]})
        res = concat([df1, df2], axis=0, ignore_index=True)
        exp = DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]], columns=["A", "B"])
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)
        res = concat([df1, df2], axis=1, ignore_index=True)
        exp = DataFrame([[1, 5, 3, 7], [2, 6, 4, 8]])
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

    def test_dups_index(self) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).integers(0, 10, size=40).reshape(10, 4), columns=["A", "A", "C", "C"]
        )
        result: DataFrame = concat([df, df], axis=1)
        tm.assert_frame_equal(result.iloc[:, :4], df)
        tm.assert_frame_equal(result.iloc[:, 4:], df)
        result = concat([df, df], axis=0)
        tm.assert_frame_equal(result.iloc[:10], df)
        tm.assert_frame_equal(result.iloc[10:], df)
        df = concat(
            [
                DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=["A", "A", "B", "B"]),
                DataFrame(np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2), columns=["A", "C"]),
            ],
            axis=1,
        )
        result = concat([df, df], axis=1)
        tm.assert_frame_equal(result.iloc[:, :6], df)
        tm.assert_frame_equal(result.iloc[:, 6:], df)
        result = concat([df, df], axis=0)
        tm.assert_frame_equal(result.iloc[:10], df)
        tm.assert_frame_equal(result.iloc[10:], df)
        result = df.iloc[0:8, :]._append(df.iloc[8:])
        tm.assert_frame_equal(result, df)
        result = df.iloc[0:8, :]._append(df.iloc[8:9])._append(df.iloc[9:10])
        tm.assert_frame_equal(result, df)
        expected = concat([df, df], axis=0)
        result = df._append(df)
        tm.assert_frame_equal(result, expected)


class TestMultiIndexConcat:
    def test_concat_multiindex_with_keys(
        self, multiindex_dataframe_random_data: DataFrame
    ) -> None:
        frame: DataFrame = multiindex_dataframe_random_data
        index: Index = frame.index
        result: DataFrame = concat([frame, frame], keys=[0, 1], names=["iteration"])
        assert result.index.names == ("iteration",) + index.names
        tm.assert_frame_equal(result.loc[0], frame)
        tm.assert_frame_equal(result.loc[1], frame)
        assert result.index.nlevels == 3

    def test_concat_multiindex_with_none_in_index_names(self) -> None:
        index: MultiIndex = MultiIndex.from_product([[1], range(5)], names=["level1", None])
        df: DataFrame = DataFrame({"col": range(5)}, index=index, dtype=np.int32)
        result: DataFrame = concat([df, df], keys=[1, 2], names=["level2"])
        index_expected: MultiIndex = MultiIndex.from_product([[1, 2], [1], range(5)], names=["level2", "level1", None])
        expected: DataFrame = DataFrame({"col": list(range(5)) * 2}, index=index_expected, dtype=np.int32)
        tm.assert_frame_equal(result, expected)
        result = concat([df, df[:2]], keys=[1, 2], names=["level2"])
        level2: List[int] = [1] * 5 + [2] * 2
        level1: List[int] = [1] * 7
        no_name: List[int] = list(range(5)) + list(range(2))
        tuples: List[tuple] = list(zip(level2, level1, no_name))
        index_expected = MultiIndex.from_tuples(tuples, names=["level2", "level1", None])
        expected = DataFrame({"col": no_name}, index=index_expected, dtype=np.int32)
        tm.assert_frame_equal(result, expected)

    def test_concat_multiindex_rangeindex(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((9, 2)))
        df.index = MultiIndex(
            levels=[pd.RangeIndex(3), pd.RangeIndex(3)],
            codes=[np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3)],
        )
        res: DataFrame = concat([df.iloc[[2, 3, 4], :], df.iloc[[5], :]])
        exp: DataFrame = df.iloc[[2, 3, 4, 5], :]
        tm.assert_frame_equal(res, exp)

    def test_concat_multiindex_dfs_with_deepcopy(self) -> None:
        example_multiindex1: MultiIndex = MultiIndex.from_product([["a"], ["b"]])
        example_dataframe1: DataFrame = DataFrame([0], index=example_multiindex1)
        example_multiindex2: MultiIndex = MultiIndex.from_product([["a"], ["c"]])
        example_dataframe2: DataFrame = DataFrame([1], index=example_multiindex2)
        example_dict: dict = {"s1": example_dataframe1, "s2": example_dataframe2}
        expected_index: MultiIndex = MultiIndex(
            levels=[["s1", "s2"], ["a"], ["b", "c"]],
            codes=[[0, 1], [0, 0], [0, 1]],
            names=["testname", None, None],
        )
        expected: DataFrame = DataFrame([[0], [1]], index=expected_index)
        result_copy: DataFrame = concat(deepcopy(example_dict), names=["testname"])
        tm.assert_frame_equal(result_copy, expected)
        result_no_copy: DataFrame = concat(example_dict, names=["testname"])
        tm.assert_frame_equal(result_no_copy, expected)

    @pytest.mark.parametrize(
        "mi1_list", 
        [
            [["a"], range(2)],
            [["b"], np.arange(2.0, 4.0)],
            [["c"], ["A", "B"]],
            [["d"], pd.date_range(start="2017", end="2018", periods=2)]
        ]
    )
    @pytest.mark.parametrize(
        "mi2_list", 
        [
            [["a"], range(2)],
            [["b"], np.arange(2.0, 4.0)],
            [["c"], ["A", "B"]],
            [["d"], pd.date_range(start="2017", end="2018", periods=2)]
        ]
    )
    def test_concat_with_various_multiindex_dtypes(self, mi1_list: List[Any], mi2_list: List[Any]) -> None:
        mi1: MultiIndex = MultiIndex.from_product(mi1_list)
        mi2: MultiIndex = MultiIndex.from_product(mi2_list)
        df1: DataFrame = DataFrame(np.zeros((1, len(mi1))), columns=mi1)
        df2: DataFrame = DataFrame(np.zeros((1, len(mi2))), columns=mi2)
        if mi1_list[0] == mi2_list[0]:
            expected_mi: MultiIndex = MultiIndex(
                levels=[mi1_list[0], list(mi1_list[1])],
                codes=[[0, 0, 0, 0], [0, 1, 0, 1]],
            )
        else:
            expected_mi = MultiIndex(
                levels=[mi1_list[0] + mi2_list[0], list(mi1_list[1]) + list(mi2_list[1])],
                codes=[[0, 0, 1, 1], [0, 1, 2, 3]],
            )
        expected_df: DataFrame = DataFrame(np.zeros((1, len(expected_mi))), columns=expected_mi)
        with tm.assert_produces_warning(None):
            result_df: DataFrame = concat((df1, df2), axis=1)
        tm.assert_frame_equal(expected_df, result_df)

    def test_concat_multiindex_(self) -> None:
        df: DataFrame = DataFrame({"col": ["a", "b", "c"]}, index=["1", "2", "2"])
        df = concat([df], keys=["X"])
        iterables: List[List[str]] = [["X"], ["1", "2", "2"]]
        result_index = df.index
        expected_index: MultiIndex = MultiIndex.from_product(iterables)
        tm.assert_index_equal(result_index, expected_index)
        result_df: DataFrame = df
        expected_df: DataFrame = DataFrame({"col": ["a", "b", "c"]}, index=MultiIndex.from_product(iterables))
        tm.assert_frame_equal(result_df, expected_df)

    def test_concat_with_key_not_unique(self, performance_warning: Any) -> None:
        df1: DataFrame = DataFrame({"name": [1]})
        df2: DataFrame = DataFrame({"name": [2]})
        df3: DataFrame = DataFrame({"name": [3]})
        df_a: DataFrame = concat([df1, df2, df3], keys=["x", "y", "x"])
        with tm.assert_produces_warning(performance_warning, match="indexing past lexsort depth"):
            out_a: DataFrame = df_a.loc[("x", 0), :]
        df_b: DataFrame = DataFrame(
            {"name": [1, 2, 3]},
            index=MultiIndex(levels=[["x", "y"], range(1)], codes=[[0, 1, 0], [0, 0, 0]]),
        )
        with tm.assert_produces_warning(performance_warning, match="indexing past lexsort depth"):
            out_b: DataFrame = df_b.loc["x", 0]
        tm.assert_frame_equal(out_a, out_b)
        df1 = DataFrame({"name": ["a", "a", "b"]})
        df2 = DataFrame({"name": ["a", "b"]})
        df3 = DataFrame({"name": ["c", "d"]})
        df_a = concat([df1, df2, df3], keys=["x", "y", "x"])
        with tm.assert_produces_warning(performance_warning, match="indexing past lexsort depth"):
            out_a = df_a.loc[("x", 0), :]
        df_b = DataFrame(
            {"a": ["x", "x", "x", "y", "y", "x", "x"], "b": [0, 1, 2, 0, 1, 0, 1], "name": list("aababcd")}
        ).set_index(["a", "b"])
        df_b.index.names = [None, None]
        with tm.assert_produces_warning(performance_warning, match="indexing past lexsort depth"):
            out_b = df_b.loc[("x", 0), :]
        tm.assert_frame_equal(out_a, out_b)

    def test_concat_with_duplicated_levels(self) -> None:
        df1: DataFrame = DataFrame({"A": [1]}, index=["x"])
        df2: DataFrame = DataFrame({"A": [1]}, index=["y"])
        msg: str = "Level values not unique: \\['x', 'y', 'y'\\]"
        with pytest.raises(ValueError, match=msg):
            concat([df1, df2], keys=["x", "y"], levels=[["x", "y", "y"]])

    @pytest.mark.parametrize("levels", [[["x", "y"]], [["x", "y", "y"]]])
    def test_concat_with_levels_with_none_keys(self, levels: List[List[str]]) -> None:
        df1: DataFrame = DataFrame({"A": [1]}, index=["x"])
        df2: DataFrame = DataFrame({"A": [1]}, index=["y"])
        msg: str = "levels supported only when keys is not None"
        with pytest.raises(ValueError, match=msg):
            concat([df1, df2], levels=levels)

    def test_concat_range_index_result(self) -> None:
        df1: DataFrame = DataFrame({"a": [1, 2]})
        df2: DataFrame = DataFrame({"b": [1, 2]})
        result: DataFrame = concat([df1, df2], sort=True, axis=1)
        expected: DataFrame = DataFrame({"a": [1, 2], "b": [1, 2]})
        tm.assert_frame_equal(result, expected)
        expected_index: pd.RangeIndex = pd.RangeIndex(0, 2)
        tm.assert_index_equal(result.index, expected_index, exact=True)

    def test_concat_index_keep_dtype(self) -> None:
        df1: DataFrame = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype="object"))
        df2: DataFrame = DataFrame([[0, 1]], columns=Index([1, 2], dtype="object"))
        result: DataFrame = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        expected: DataFrame = DataFrame([[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype="object"))
        tm.assert_frame_equal(result, expected)

    def test_concat_index_keep_dtype_ea_numeric(self, any_numeric_ea_dtype: str) -> None:
        df1: DataFrame = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype))
        df2: DataFrame = DataFrame([[0, 1]], columns=Index([1, 2], dtype=any_numeric_ea_dtype))
        result: DataFrame = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        expected: DataFrame = DataFrame([[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["Int8", "Int16", "Int32"])
    def test_concat_index_find_common(self, dtype: str) -> None:
        df1: DataFrame = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype=dtype))
        df2: DataFrame = DataFrame([[0, 1]], columns=Index([1, 2], dtype="Int32"))
        result: DataFrame = concat([df1, df2], ignore_index=True, join="outer", sort=True)
        expected: DataFrame = DataFrame([[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype="Int32"))
        tm.assert_frame_equal(result, expected)

    def test_concat_axis_1_sort_false_rangeindex(self, using_infer_string: bool) -> None:
        s1: Series = Series(["a", "b", "c"])
        s2: Series = Series(["a", "b"])
        s3: Series = Series(["a", "b", "c", "d"])
        s4: Series = Series([], dtype=object if not using_infer_string else "str")
        result: DataFrame = concat([s1, s2, s3, s4], sort=False, join="outer", ignore_index=False, axis=1)
        expected: DataFrame = DataFrame(
            [["a"] * 3 + [np.nan], ["b"] * 3 + [np.nan], ["c", np.nan] * 2, [np.nan] * 2 + ["d"] + [np.nan]],
            dtype=object if not using_infer_string else "str",
        )
        tm.assert_frame_equal(result, expected, check_index_type=True, check_column_type=True)