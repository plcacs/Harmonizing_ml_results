"""test fancy indexing & misc"""

import array
from datetime import datetime
import re
import weakref
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pytest

from pandas.errors import IndexingError

from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
)

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    NaT,
    Series,
    date_range,
    offsets,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj

# ------------------------------------------------------------------------
# Indexing test cases


class TestFancy:
    """pure get/set item & fancy indexing"""

    def test_setitem_ndarray_1d(self) -> None:
        # GH5508

        # len of indexer vs length of the 1d ndarray
        df = DataFrame(index=Index(np.arange(1, 11), dtype=np.int64))
        df["foo"] = np.zeros(10, dtype=np.float64)
        df["bar"] = np.zeros(10, dtype=complex)

        # invalid
        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df.loc[df.index[2:5], "bar"] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])

        # valid
        df.loc[df.index[2:6], "bar"] = np.array([2.33j, 1.23 + 0.1j, 2.2, 1.0])

        result = df.loc[df.index[2:6], "bar"]
        expected = Series(
            [2.33j, 1.23 + 0.1j, 2.2, 1.0], index=[3, 4, 5, 6], name="bar"
        )
        tm.assert_series_equal(result, expected)

    def test_setitem_ndarray_1d_2(self) -> None:
        # GH5508

        # dtype getting changed?
        df = DataFrame(index=Index(np.arange(1, 11)))
        df["foo"] = np.zeros(10, dtype=np.float64)
        df["bar"] = np.zeros(10, dtype=complex)

        msg = "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df[2:5] = np.arange(1, 4) * 1j

    @pytest.mark.filterwarnings(
        "ignore:Series.__getitem__ treating keys as positions is deprecated:"
        "FutureWarning"
    )
    def test_getitem_ndarray_3d(
        self, index: Index, frame_or_series: Any, indexer_sli: Any
    ) -> None:
        # GH 25567
        obj = gen_obj(frame_or_series, index)
        idxr = indexer_sli(obj)
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))

        msgs = []
        if frame_or_series is Series and indexer_sli in [tm.setitem, tm.iloc]:
            msgs.append(r"Wrong number of dimensions. values.ndim > ndim \[3 > 1\]")
        if frame_or_series is Series or indexer_sli is tm.iloc:
            msgs.append(r"Buffer has wrong number of dimensions \(expected 1, got 3\)")
        if indexer_sli is tm.loc or (
            frame_or_series is Series and indexer_sli is tm.setitem
        ):
            msgs.append("Cannot index with multidimensional key")
        if frame_or_series is DataFrame and indexer_sli is tm.setitem:
            msgs.append("Index data must be 1-dimensional")
        if isinstance(index, pd.IntervalIndex) and indexer_sli is tm.iloc:
            msgs.append("Index data must be 1-dimensional")
        if isinstance(index, (pd.TimedeltaIndex, pd.DatetimeIndex, pd.PeriodIndex)):
            msgs.append("Data must be 1-dimensional")
        if len(index) == 0 or isinstance(index, pd.MultiIndex):
            msgs.append("positional indexers are out-of-bounds")
        if type(index) is Index and not isinstance(index._values, np.ndarray):
            # e.g. Int64
            msgs.append("values must be a 1D array")

            # string[pyarrow]
            msgs.append("only handle 1-dimensional arrays")

        msg = "|".join(msgs)

        potential_errors = (IndexError, ValueError, NotImplementedError)
        with pytest.raises(potential_errors, match=msg):
            idxr[nd3]

    @pytest.mark.filterwarnings(
        "ignore:Series.__setitem__ treating keys as positions is deprecated:"
        "FutureWarning"
    )
    def test_setitem_ndarray_3d(
        self, index: Index, frame_or_series: Any, indexer_sli: Any
    ) -> None:
        # GH 25567
        obj = gen_obj(frame_or_series, index)
        idxr = indexer_sli(obj)
        nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))

        if indexer_sli is tm.iloc:
            err = ValueError
            msg = f"Cannot set values with ndim > {obj.ndim}"
        else:
            err = ValueError
            msg = "|".join(
                [
                    r"Buffer has wrong number of dimensions \(expected 1, got 3\)",
                    "Cannot set values with ndim > 1",
                    "Index data must be 1-dimensional",
                    "Data must be 1-dimensional",
                    "Array conditional must be same shape as self",
                ]
            )

        with pytest.raises(err, match=msg):
            idxr[nd3] = 0

    def test_getitem_ndarray_0d(self) -> None:
        # GH#24924
        key = np.array(0)

        # dataframe __getitem__
        df = DataFrame([[1, 2], [3, 4]])
        result = df[key]
        expected = Series([1, 3], name=0)
        tm.assert_series_equal(result, expected)

        # series __getitem__
        ser = Series([1, 2])
        result = ser[key]
        assert result == 1

    def test_inf_upcast(self) -> None:
        # GH 16957
        # We should be able to use np.inf as a key
        # np.inf should cause an index to convert to float

        # Test with np.inf in rows
        df = DataFrame(columns=[0])
        df.loc[1] = 1
        df.loc[2] = 2
        df.loc[np.inf] = 3

        # make sure we can look up the value
        assert df.loc[np.inf, 0] == 3

        result = df.index
        expected = Index([1, 2, np.inf], dtype=np.float64)
        tm.assert_index_equal(result, expected)

    def test_setitem_dtype_upcast(self) -> None:
        # GH3216
        df = DataFrame([{"a": 1}, {"a": 3, "b": 2}])
        df["c"] = np.nan
        assert df["c"].dtype == np.float64

        with pytest.raises(TypeError, match="Invalid value"):
            df.loc[0, "c"] = "foo"

    @pytest.mark.parametrize("val", [3.14, "wxyz"])
    def test_setitem_dtype_upcast2(self, val: Union[float, str]) -> None:
        # GH10280
        df = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3),
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )

        left = df.copy()
        with pytest.raises(TypeError, match="Invalid value"):
            left.loc["a", "bar"] = val

    def test_setitem_dtype_upcast3(self) -> None:
        left = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3) / 10.0,
            index=list("ab"),
            columns=["foo", "bar", "baz"],
        )
        with pytest.raises(TypeError, match="Invalid value"):
            left.loc["a", "bar"] = "wxyz"

    def test_dups_fancy_indexing(self) -> None:
        # GH 3455

        df = DataFrame(np.eye(3), columns=["a", "a", "b"])
        result = df[["b", "a"]].columns
        expected = Index(["b", "a", "a"])
        tm.assert_index_equal(result, expected)

    def test_dups_fancy_indexing_across_dtypes(self) -> None:
        # across dtypes
        df = DataFrame([[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]], columns=list("aaaaaaa"))
        result = DataFrame([[1, 2, 1.0, 2.0, 3.0, "foo", "bar"]])
        result.columns = list("aaaaaaa")  # GH#3468

        # GH#3509 smoke tests for indexing with duplicate columns
        df.iloc[:, 4]
        result.iloc[:, 4]

        tm.assert_frame_equal(df, result)

    def test_dups_fancy_indexing_not_in_order(self) -> None:
        # GH 3561, dups not in selected order
        df = DataFrame(
            {"test": [5, 7, 9, 11], "test1": [4.0, 5, 6, 7], "other": list("abcd")},
            index=["A", "A", "B", "C"],
        )
        rows = ["C", "B"]
        expected = DataFrame(
            {"test": [11, 9], "test1": [7.0, 6], "other": ["d", "c"]}, index=rows
        )
        result = df.loc[rows]
        tm.assert_frame_equal(result, expected)

        result = df.loc[Index(rows)]
        tm.assert_frame_equal(result, expected)

        rows = ["C", "B", "E"]
        with pytest.raises(KeyError, match="not in index"):
            df.loc[rows]

        # see GH5553, make sure we use the right indexer
        rows = ["F", "G", "H", "C", "B", "E"]
        with pytest.raises(KeyError, match="not in index"):
            df.loc[rows]

    def test_dups_fancy_indexing_only_missing_label(self, using_infer_string: bool) -> None:
        # List containing only missing label
        dfnu = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), index=list("AABCD")
        )
        if using_infer_string:
            with pytest.raises(
                KeyError,
                match=re.escape(
                    "\"None of [Index(['E'], dtype='str')] are in the [index]\""
                ),
            ):
                dfnu.loc[["E"]]
        else:
            with pytest.raises(
                KeyError,
                match=re.escape(
                    "\"None of [Index(['E'], dtype='object')] are in the [index]\""
                ),
            ):
                dfnu.loc[["E"]]

    @pytest.mark.parametrize("vals", [[0, 1, 2], list("abc")])
    def test_dups_fancy_indexing_missing_label(self, vals: List[Any]) -> None:
        # GH 4619; duplicate indexer with missing label
        df = DataFrame({"A": vals})
        with pytest.raises(KeyError, match="not in index"):
            df.loc[[0, 8, 0]]

    def test_dups_fancy_indexing_non_unique(self) -> None:
        # non unique with non unique selector
        df = DataFrame({"test": [5, 7, 9, 11]}, index=["A", "A", "B", "C"])
        with pytest.raises(KeyError, match="not in index"):
            df.loc[["A", "A", "E"]]

    def test_dups_fancy_indexing2(self) -> None:
        # GH 5835
        # dups on index and missing values
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 5)),
            columns=["A", "B", "B", "B", "A"],
        )

        with pytest.raises(KeyError, match="not in index"):
            df.loc[:, ["A", "B", "C"]]

    def test_dups_fancy_indexing3(self) -> None:
        # GH 6504, multi-axis indexing
        df = DataFrame(
            np.random.default_rng(2).standard_normal((9, 2)),
            index=[1, 1, 1, 2, 2, 2, 3, 3, 3],
            columns=["a", "b"],
        )

        expected = df.iloc[0:6]
        result = df.loc[[1, 2]]
        tm.assert_frame_equal(result, expected)

        expected = df
        result = df.loc[:, ["a", "b"]]
        tm.assert_frame_equal(result, expected)

        expected = df.iloc[0:6, :]
        result = df.loc[[1, 2], ["a", "b"]]
        tm.assert_frame_equal(result, expected)

    def test_duplicate_int_indexing(self, indexer_sl: Any) -> None:
        # GH 17347
        ser = Series(range(3), index=[1, 1, 3])
        expected = Series(range(2), index=[1, 1])
        result = indexer_sl(ser)[[1]]
        tm.assert_series_equal(result, expected)

    def test_indexing_mixed_frame_bug(self) -> None:
        # GH3492
        df = DataFrame(
            {"a": {1: "aaa", 2: "bbb", 3: "ccc"}, "b": {1: 111, 2: 222, 3: 333}}
        )

        # this works, new column is created correctly
        df["test"] = df["a"].apply(lambda x: "_" if x == "aaa" else x)

        # this does not work, ie column test is not changed
        idx = df["test"] == "_"
        temp = df.loc[idx, "a"].apply(lambda x: "-----" if x == "aaa" else x)
        df.loc[idx, "test"] = temp
        assert df.iloc[0, 2] == "-----"

    def test_multitype_list_index_access(self) -> None:
        # GH 10610
        df = DataFrame(
            np.random.default_rng(2).random((10, 5)), columns=["a"] + [20, 21, 22, 23]
        )

        with pytest.raises(KeyError, match=re.escape("'[26, -8] not in index'")):
            df[[22, 26, -8]]
        assert df[21].shape[0] == df.shape[0]

    def test_set_index_nan(self) -> None:
        # GH 3586
        df = DataFrame(
            {
                "PRuid": {
                    17: "nonQC",
                    18: "nonQC",
                    19: "nonQC",
                    20: "10",
                    21: "11",
                    22: "12",
                    23: "13",
                    24: "24",
                    25: "35",
                    26: "46",
                    27: "47",
                    28: "48",
                    29: "59",
                    30: "10",
                },
                "QC": {
                    17: 0.0,
                    18: 0.0,
                    19: 0.0,
                    20: np.nan,
                    21: np.nan,
                    22: np.nan,
                    23: np.nan,
                    24: 1.0,
                    25: np.nan,
                    26: np.nan,
                    27: np.nan,
                    28: np.nan,
                    29: np.nan,
                    30: np.nan,
                },
                "data": {
                    17: 7.9544899999999998,
                    18: 8.0142609999999994,
                    19: 7.8591520000000008,
                    20: 0.86140349999999999,
                    21: 0.87853110000000001,
                    22: 0.8427041999999999,
                    23: 0.78587700000000005,
                    24: 0.73062459999999996,
                    25: 0.81668560000000001,
                    26: 0.81927080000000008,
                    27: 0.80705009999999999,
                    28: 0.81440240000000008,
                    29: 0.80140849999999997,
                    30: 0.81307740000000006,
                },
                "year": {
                    17: 2006,
                    18: 2007,
                    19: 2008,
                    20: 1985,
                    21: 1985,
                    22: 1985,
                    23: 1985,
                    24: 1985,
                    25: 1985,
                    26: 1985,
                    27: 1985,
                    28: 1985,
                    29: 1985,
                    30: 1986,
                },
            }
        ).reset_index()

        result = (
            df.set_index(["year", "PRuid", "QC"])
            .reset_index()
            .reindex(columns=df.columns)
        )
        tm.assert_frame_equal(result, df)

    def test_multi_assign(self) -> None:
        # GH 3626, an assignment of a sub-df