import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray


class TestSeriesReplace:
    def test_replace_explicit_none(self) -> None:
        # GH#36984 if the user explicitly passes value=None, give it to them
        ser = pd.Series([0, 0, ""], dtype=object)
        result = ser.replace("", None)
        expected = pd.Series([0, 0, None], dtype=object)
        tm.assert_series_equal(result, expected)

        # Cast column 2 to object to avoid implicit cast when setting entry to ""
        df = pd.DataFrame(np.zeros((3, 3))).astype({2: object})
        df.iloc[2, 2] = ""
        result = df.replace("", None)
        expected = pd.DataFrame(
            {
                0: np.zeros(3),
                1: np.zeros(3),
                2: np.array([0.0, 0.0, None], dtype=object),
            }
        )
        assert expected.iloc[2, 2] is None
        tm.assert_frame_equal(result, expected)

        # GH#19998 same thing with object dtype
        ser = pd.Series([10, 20, 30, "a", "a", "b", "a"])
        result = ser.replace("a", None)
        expected = pd.Series([10, 20, 30, None, None, "b", None])
        assert expected.iloc[-1] is None
        tm.assert_series_equal(result, expected)

    def test_replace_noop_doesnt_downcast(self) -> None:
        # GH#44498
        ser = pd.Series([None, None, pd.Timestamp("2021-12-16 17:31")], dtype=object)
        res = ser.replace({np.nan: None})  # should be a no-op
        tm.assert_series_equal(res, ser)
        assert res.dtype == object

        # same thing but different calling convention
        res = ser.replace(np.nan, None)
        tm.assert_series_equal(res, ser)
        assert res.dtype == object

    def test_replace(self) -> None:
        N = 50
        ser = pd.Series(np.random.default_rng(2).standard_normal(N))
        ser[0:4] = np.nan
        ser[6:10] = 0

        # replace list with a single value
        return_value = ser.replace([np.nan], -1, inplace=True)
        assert return_value is None

        exp = ser.fillna(-1)
        tm.assert_series_equal(ser, exp)

        rs = ser.replace(0.0, np.nan)
        ser[ser == 0.0] = np.nan
        tm.assert_series_equal(rs, ser)

        ser = pd.Series(
            np.fabs(np.random.default_rng(2).standard_normal(N)),
            pd.date_range("2020-01-01", periods=N),
            dtype=object,
        )
        ser[:5] = np.nan
        ser[6:10] = "foo"
        ser[20:30] = "bar"

        # replace list with a single value
        rs = ser.replace([np.nan, "foo", "bar"], -1)

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -1).all()
        assert (rs[20:30] == -1).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values
        rs = ser.replace({np.nan: -1, "foo": -2, "bar": -3})

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -2).all()
        assert (rs[20:30] == -3).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values with 2 lists
        rs2 = ser.replace([np.nan, "foo", "bar"], [-1, -2, -3])
        tm.assert_series_equal(rs, rs2)

        # replace inplace
        return_value = ser.replace([np.nan, "foo", "bar"], -1, inplace=True)
        assert return_value is None
        assert (ser[:5] == -1).all()
        assert (ser[6:10] == -1).all()
        assert (ser[20:30] == -1).all()

    def test_replace_nan_with_inf(self) -> None:
        ser = pd.Series([np.nan, 0, np.inf])
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))

        ser = pd.Series([np.nan, 0, "foo", "bar", np.inf, None, pd.NaT])
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))
        filled = ser.copy()
        filled[4] = 0
        tm.assert_series_equal(ser.replace(np.inf, 0), filled)

    def test_replace_listlike_value_listlike_target(self, datetime_series: pd.Series) -> None:
        ser = pd.Series(datetime_series.index)
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))

        # malformed
        msg = r"Replacement lists must match in length\. Expecting 3 got 2"
        with pytest.raises(ValueError, match=msg):
            ser.replace([1, 2, 3], [np.nan, 0])

        # ser is dt64 so can't hold 1 or 2, so this replace is a no-op
        result = ser.replace([1, 2], [np.nan, 0])
        tm.assert_series_equal(result, ser)

        ser = pd.Series([0, 1, 2, 3, 4])
        result = ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
        tm.assert_series_equal(result, pd.Series([4, 3, 2, 1, 0]))

    def test_replace_gh5319(self) -> None:
        # API change from 0.12?
        # GH 5319
        ser = pd.Series([0, np.nan, 2, 3, 4])
        msg = (
            "Series.replace must specify either 'value', "
            "a dict-like 'to_replace', or dict-like 'regex'"
        )
        with pytest.raises(ValueError, match=msg):
            ser.replace([np.nan])

        with pytest.raises(ValueError, match=msg):
            ser.replace(np.nan)

    def test_replace_datetime64(self) -> None:
        # GH 5797
        ser = pd.Series(pd.date_range("20130101", periods=5))
        expected = ser.copy()
        expected.loc[2] = pd.Timestamp("20120101")
        result = ser.replace({pd.Timestamp("20130103"): pd.Timestamp("20120101")})
        tm.assert_series_equal(result, expected)
        result = ser.replace(pd.Timestamp("20130103"), pd.Timestamp("20120101"))
        tm.assert_series_equal(result, expected)

    def test_replace_nat_with_tz(self) -> None:
        # GH 11792: Test with replacing NaT in a list with tz data
        ts = pd.Timestamp("2015/01/01", tz="UTC")
        s = pd.Series([pd.NaT, pd.Timestamp("2015/01/01", tz="UTC")])
        result = s.replace([np.nan, pd.NaT], pd.Timestamp.min)
        expected = pd.Series([pd.Timestamp.min, ts], dtype=object)
        tm.assert_series_equal(expected, result)

    def test_replace_timedelta_td64(self) -> None:
        tdi = pd.timedelta_range(0, periods=5)
        ser = pd.Series(tdi)

        # Using a single dict argument means we go through replace_list
        result = ser.replace({ser[1]: ser[3]})

        expected = pd.Series([ser[0], ser[3], ser[2], ser[3], ser[4]])
        tm.assert_series_equal(result, expected)

    def test_replace_with_single_list(self) -> None:
        ser = pd.Series([0, 1, 2, 3, 4])
        msg = (
            "Series.replace must specify either 'value', "
            "a dict-like 'to_replace', or dict-like 'regex'"
        )
        with pytest.raises(ValueError, match=msg):
            ser.replace([1, 2, 3])

        s = ser.copy()
        with pytest.raises(ValueError, match=msg):
            s.replace([1, 2, 3], inplace=True)

    def test_replace_mixed_types(self) -> None:
        ser = pd.Series(np.arange(5), dtype="int64")

        def check_replace(to_rep: List[int], val: List[Union[int, float, str, bool, pd.Timestamp]], expected: pd.Series) -> None:
            sc = ser.copy()
            result = ser.replace(to_rep, val)
            return_value = sc.replace(to_rep, val, inplace=True)
            assert return_value is None
            tm.assert_series_equal(expected, result)
            tm.assert_series_equal(expected, sc)

        # 3.0 can still be held in our int64 series, so we do not upcast GH#44940
        tr, v = [3], [3.0]
        check_replace(tr, v, ser)
        # Note this matches what we get with the scalars 3 and 3.0
        check_replace(tr[0], v[0], ser)

        # MUST upcast to float
        e = pd.Series([0, 1, 2, 3.5, 4])
        tr, v = [3], [3.5]
        check_replace(tr, v, e)

        # casts to object
        e = pd.Series([0, 1, 2, 3.5, "a"])
        tr, v = [3, 4], [3.5, "a"]
        check_replace(tr, v, e)

        # again casts to object
        e = pd.Series([0, 1, 2, 3.5, pd.Timestamp("20130101")])
        tr, v = [3, 4], [3.5, pd.Timestamp("20130101")]
        check_replace(tr, v, e)

        # casts to object
        e = pd.Series([0, 1, 2, 3.5, True], dtype="object")
        tr, v = [3, 4], [3.5, True]
        check_replace(tr, v, e)

        # test an object with dates + floats + integers + strings
        dr = pd.Series(pd.date_range("1/1/2001", "1/10/2001", freq="D"))
        result = dr.astype(object).replace([dr[0], dr[1], dr[2]], [1.0, 2, "a"])
        expected = pd.Series([1.0, 2, "a"] + dr[3:].tolist(), dtype=object)
        tm.assert_series_equal(result, expected)

    def test_replace_bool_with_string_no_op(self) -> None:
        s = pd.Series([True, False, True])
        result = s.replace("fun", "in-the-sun")
        tm.assert_series_equal(s, result)

    def test_replace_bool_with_string(self) -> None:
        # nonexistent elements
        s = pd.Series([True, False, True])
        result = s.replace(True, "2u")
        expected = pd.Series(["2u", False, "2u"])
        tm.assert_series_equal(expected, result)

    def test_replace_bool_with_bool(self) -> None:
        s = pd.Series([True, False, True])
        result = s.replace(True, False)
        expected = pd.Series([False] * len(s))
        tm.assert_series_equal(expected, result)

    def test_replace_with_dict_with_bool_keys(self) -> None:
        s = pd.Series([True, False, True])
        result = s.replace({"asdf": "asdb", True: "yes"})
        expected = pd.Series(["yes", False, "yes"])
        tm.assert_series_equal(result, expected)

    def test_replace_Int_with_na(self, any_int_ea_dtype: str) -> None:
        # GH 38267
        result = pd.Series([0, None], dtype=any_int_ea_dtype).replace(0, pd.NA)
        expected = pd.Series([pd.NA, pd.NA], dtype=any_int_ea_dtype)
        tm.assert_series_equal(result, expected)
        result = pd.Series([0, 1], dtype=any_int_ea_dtype).replace(0, pd.NA)
        result.replace(1, pd.NA, inplace=True)
        tm.assert_series_equal(result, expected)

    def test_replace2(self) -> None:
        N = 50
        ser = pd.Series(
            np.fabs(np.random.default_rng(2).standard_normal(N)),
            pd.date_range("2020-01-01", periods=N),
            dtype=object,
        )
        ser[:5] = np.nan
        ser[6:10] = "foo"
        ser[20:30] = "bar"

        # replace list with a single value
        rs = ser.replace([np.nan, "foo", "bar"], -1)

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -1).all()
        assert (rs[20:30] == -1).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values
        rs = ser.replace({np.nan: -1, "foo": -2, "bar": -3})

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -2).all()
        assert (rs[20:30] == -3).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values with 2 lists
        rs2 = ser.replace([np.nan, "foo", "bar"], [-1, -2, -3])
        tm.assert_series_equal(rs, rs2)

        # replace inplace
        return_value = ser.replace([np.nan, "foo", "bar"], -1, inplace=True)
        assert return_value is None
        assert (ser[:5] == -1).all()
        assert (ser[6:10] == -1).all()
        assert (ser[20:30] == -1).all()

    @pytest.mark.parametrize("inplace", [True, False])
    def test_replace_cascade(self, inplace: bool) -> None:
        # Test that replaced values are not replaced again
        # GH #50778
        ser = pd.Series([1, 2, 3])
        expected = pd.Series([2, 3, 4])

        res = ser.replace([1, 2, 3], [2, 3, 4], inplace=inplace)
        if inplace:
            tm.assert_series_equal(ser, expected)
        else:
            tm.assert_series_equal(res, expected)

    def test_replace_with_dictlike_and_string_dtype(self, nullable_string_dtype: str) -> None:
        # GH 32621, GH#44940
        ser = pd.Series(["one", "two", np.nan], dtype=nullable_string_dtype)
        expected = pd.Series(["1", "2", np.nan], dtype=nullable_string_dtype)
        result = ser.replace({"one": "1", "two": "2"})
        tm.assert_series_equal(expected, result)

    def test_replace_with_empty_dictlike(self) -> None:
        # GH 15289
        s = pd.Series(list("abcd"))
        tm.assert_series_equal(s, s.replace({}))

        empty_series = pd.Series([])
        tm.assert_series_equal(s, s.replace(empty_series))

    def test_replace_string_with_number(self) -> None:
        # GH 15743
        s = pd.Series([1, 2, 3])
        result = s.replace("2", np.nan)
        expected = pd.Series([1, 2, 3])
        tm.assert_series_equal(expected, result)

    def test_replace_replacer_equals_replacement(self) -> None:
        # GH 20656
        # make sure all replacers are matching against original values
        s = pd.Series(["a", "b"])
        expected = pd.Series(["b", "a"])
        result = s.replace({"a": "b", "b": "a"})
        tm.assert_series_equal(expected, result)

    def test_replace_unicode_with_number(self) -> None:
        # GH 15743
        s = pd.Series([1, 2, 3])
        result = s.replace("2", np.nan)
        expected = pd.Series([1, 2, 3])
        tm.assert_series_equal(expected, result)

    def test_replace_mixed_types_with_string(self) -> None:
        # Testing mixed
        s = pd.Series([1, 2, 3, "4", 4, 5])
        result = s.replace([2, "4"], np.nan)
        expected = pd.Series([1, np.nan, 3, np.nan, 4, 5], dtype=object)
        tm.assert_series_equal(expected, result)

    @pytest.mark.parametrize(
        "categorical, numeric",
        [
            (["A"], [1]),
            (["A", "B"], [1, 2]),
        ],
    )
    def test_replace_categorical(self,