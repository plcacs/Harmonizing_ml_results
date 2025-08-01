from __future__ import annotations
from typing import Any, Callable, Dict, Union, Type
import numpy as np
import pytest
from pandas import DataFrame, Index, RangeIndex, Series, date_range, period_range, timedelta_range
import pandas._testing as tm

FrameOrSeries = Union[DataFrame, Series]
IndexerFunc = Callable[[FrameOrSeries], FrameOrSeries]

def gen_obj(klass: Type[FrameOrSeries], index: Index) -> FrameOrSeries:
    if klass is Series:
        obj: FrameOrSeries = Series(np.arange(len(index)), index=index)
    else:
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), len(index))),
            index=index,
            columns=index,
        )
    return obj

class TestFloatIndexers:
    def check(
        self,
        result: FrameOrSeries,
        original: FrameOrSeries,
        indexer: Union[int, slice],
        getitem: bool,
    ) -> None:
        """
        comparator for results
        we need to take care if we are indexing on a
        Series or a frame
        """
        if isinstance(original, Series):
            expected = original.iloc[indexer]
        elif getitem:
            expected = original.iloc[:, indexer]
        else:
            expected = original.iloc[indexer]
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcde")),
            Index(list("abcde"), dtype="category"),
            date_range("2020-01-01", periods=5),
            timedelta_range("1 day", periods=5),
            period_range("2020-01-01", periods=5),
        ],
    )
    def test_scalar_non_numeric(
        self,
        index: Index,
        frame_or_series: Type[FrameOrSeries],
        indexer_sl: IndexerFunc,
    ) -> None:
        s: FrameOrSeries = gen_obj(frame_or_series, index)
        with pytest.raises(KeyError, match="^3.0$"):
            indexer_sl(s)[3.0]
        assert 3.0 not in s
        s2: FrameOrSeries = s.copy()
        indexer_sl(s2)[3.0] = 10
        if indexer_sl is tm.setitem:
            assert 3.0 in s2.axes[-1]
        elif indexer_sl is tm.loc:
            assert 3.0 in s2.axes[0]
        else:
            assert 3.0 not in s2.axes[0]
            assert 3.0 not in s2.axes[-1]

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcde")),
            Index(list("abcde"), dtype="category"),
            date_range("2020-01-01", periods=5),
            timedelta_range("1 day", periods=5),
            period_range("2020-01-01", periods=5),
        ],
    )
    def test_scalar_non_numeric_series_fallback(self, index: Index) -> None:
        s: Series = Series(np.arange(len(index)), index=index)
        with pytest.raises(KeyError, match="3"):
            s[3]
        with pytest.raises(KeyError, match="^3.0$"):
            s[3.0]

    def test_scalar_with_mixed(self, indexer_sl: IndexerFunc) -> None:
        s2: Series = Series([1, 2, 3], index=["a", "b", "c"])
        s3: Series = Series([1, 2, 3], index=["a", "b", 1.5])
        with pytest.raises(KeyError, match="^1.0$"):
            indexer_sl(s2)[1.0]
        with pytest.raises(KeyError, match="^1\\.0$"):
            indexer_sl(s2)[1.0]
        result = indexer_sl(s2)["b"]
        expected = 2
        assert result == expected
        with pytest.raises(KeyError, match="^1.0$"):
            indexer_sl(s3)[1.0]
        if indexer_sl is not tm.loc:
            with pytest.raises(KeyError, match="^1$"):
                s3[1]
        with pytest.raises(KeyError, match="^1\\.0$"):
            indexer_sl(s3)[1.0]
        result = indexer_sl(s3)[1.5]
        expected = 3
        assert result == expected

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_scalar_integer(
        self,
        index: Union[Index, RangeIndex],
        frame_or_series: Type[FrameOrSeries],
        indexer_sl: IndexerFunc,
    ) -> None:
        getitem: bool = indexer_sl is not tm.loc
        i = index
        obj: FrameOrSeries = gen_obj(frame_or_series, i)
        result = indexer_sl(obj)[3.0]
        self.check(result, obj, 3, getitem)
        if isinstance(obj, Series):

            def compare(x: Any, y: Any) -> None:
                assert x == y

            expected = 100
        else:
            compare = tm.assert_series_equal
            if getitem:
                expected = Series(100, index=range(len(obj)), name=3)
            else:
                expected = Series(100.0, index=range(len(obj)), name=3)
        s2: FrameOrSeries = obj.copy()
        indexer_sl(s2)[3.0] = 100
        result = indexer_sl(s2)[3.0]
        compare(result, expected)
        result = indexer_sl(s2)[3]
        compare(result, expected)

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_scalar_integer_contains_float(
        self, index: Union[Index, RangeIndex], frame_or_series: Type[FrameOrSeries]
    ) -> None:
        obj: FrameOrSeries = gen_obj(frame_or_series, index)
        assert 3.0 in obj

    def test_scalar_float(self, frame_or_series: Type[FrameOrSeries]) -> None:
        index: Index = Index(np.arange(5.0))
        s: FrameOrSeries = gen_obj(frame_or_series, index)
        indexer = index[3]
        for idxr in [tm.loc, tm.setitem]:
            getitem: bool = idxr is not tm.loc
            result = idxr(s)[indexer]
            self.check(result, s, 3, getitem)
            s2: FrameOrSeries = s.copy()
            result = idxr(s2)[indexer]
            self.check(result, s, 3, getitem)
            with pytest.raises(KeyError, match="^3\\.5$"):
                idxr(s)[3.5]
        assert 3.0 in s
        expected = s.iloc[3]
        s2 = s.copy()
        s2.iloc[3] = expected
        result = s2.iloc[3]
        self.check(result, s, 3, False)

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcde"), dtype=object),
            date_range("2020-01-01", periods=5),
            timedelta_range("1 day", periods=5),
            period_range("2020-01-01", periods=5),
        ],
    )
    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    def test_slice_non_numeric(
        self,
        index: Index,
        idx: slice,
        frame_or_series: Type[FrameOrSeries],
        indexer_sli: IndexerFunc,
    ) -> None:
        s: FrameOrSeries = gen_obj(frame_or_series, index)
        if indexer_sli is tm.iloc:
            msg = (
                f"cannot do positional indexing on {type(index).__name__} "
                f"with these indexers \\[(3|4)\\.0\\] of type float"
            )
        else:
            msg = (
                f"cannot do slice indexing on {type(index).__name__} "
                f"with these indexers \\[(3|4)(\\.0)?\\] of type (float|int)"
            )
        with pytest.raises(TypeError, match=msg):
            indexer_sli(s)[idx]
        if indexer_sli is tm.iloc:
            msg = "slice indices must be integers or None or have an __index__ method"
        with pytest.raises(TypeError, match=msg):
            indexer_sli(s)[idx] = 0

    def test_slice_integer(self) -> None:
        for index, oob in [
            (Index(np.arange(5, dtype=np.int64)), False),
            (RangeIndex(5), False),
            (Index(np.arange(5, dtype=np.int64) + 10), True),
        ]:
            s: Series = Series(range(5), index=index)
            for idx in [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)]:
                result = s.loc[idx]
                if oob:
                    indexer = slice(0, 0)
                else:
                    indexer = slice(3, 5)
                self.check(result, s, indexer, False)
            for idx in [slice(-6, 6), slice(-6.0, 6.0)]:
                result = s.loc[idx]
                if oob:
                    indexer = slice(0, 0)
                else:
                    indexer = slice(-6, 6)
                self.check(result, s, indexer, False)
            msg = (
                f"cannot do slice indexing on {type(index).__name__} "
                f"with these indexers \\[-6\\.0\\] of type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[slice(-6.0, 6.0)]
            for idx, res1 in [
                (slice(2.5, 4), slice(3, 5)),
                (slice(2, 3.5), slice(2, 4)),
                (slice(2.5, 3.5), slice(3, 4)),
            ]:
                result = s.loc[idx]
                if oob:
                    res = slice(0, 0)
                else:
                    res = res1
                self.check(result, s, res, False)
                msg = (
                    f"cannot do slice indexing on {type(index).__name__} "
                    f"with these indexers \\[(2|3)\\.5\\] of type float"
                )
                with pytest.raises(TypeError, match=msg):
                    s[idx]

    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    def test_integer_positional_indexing(self, idx: slice) -> None:
        """make sure that we are raising on positional indexing
        w.r.t. an integer index
        """
        s: Series = Series(range(2, 6), index=range(2, 6))
        result = s[2:4]
        expected = s.iloc[2:4]
        tm.assert_series_equal(result, expected)
        klass = RangeIndex
        msg = (
            f"cannot do (slice|positional) indexing on {klass.__name__} "
            f"with these indexers \\[(2|4)\\.0\\] of type float"
        )
        with pytest.raises(TypeError, match=msg):
            s[idx]
        with pytest.raises(TypeError, match=msg):
            s.iloc[idx]

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_slice_integer_frame_getitem(self, index: Union[Index, RangeIndex]) -> None:
        s: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), index=index
        )
        for idx in [slice(0.0, 1), slice(0, 1.0), slice(0.0, 1.0)]:
            result = s.loc[idx]
            indexer = slice(0, 2)
            self.check(result, s, indexer, True)
            msg = (
                f"cannot do slice indexing on {type(index).__name__} "
                f"with these indexers \\[(0|1)\\.0\\] of type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[idx]
        for idx in [slice(-10, 10), slice(-10.0, 10.0)]:
            result = s.loc[idx]
            self.check(result, s, slice(-10, 10), True)
        msg = (
            f"cannot do slice indexing on {type(index).__name__} "
            f"with these indexers \\[-10\\.0\\] of type float"
        )
        with pytest.raises(TypeError, match=msg):
            s[slice(-10.0, 10.0)]
        for idx, res in [
            (slice(0.5, 1), slice(1, 2)),
            (slice(0, 0.5), slice(0, 1)),
            (slice(0.5, 1.5), slice(1, 2)),
        ]:
            result = s.loc[idx]
            self.check(result, s, res, False)
            msg = (
                f"cannot do slice indexing on {type(index).__name__} "
                f"with these indexers \\[0\\.5\\] of type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[idx]

    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_float_slice_getitem_with_integer_index_raises(
        self, idx: slice, index: Union[Index, RangeIndex]
    ) -> None:
        s: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), index=index
        )
        sc: DataFrame = s.copy()
        sc.loc[idx] = 0
        result = sc.loc[idx].values.ravel()
        assert (result == 0).all()
        msg = (
            f"cannot do slice indexing on {type(index).__name__} "
            f"with these indexers \\[(3|4)\\.0\\] of type float"
        )
        with pytest.raises(TypeError, match=msg):
            s[idx] = 0
        with pytest.raises(TypeError, match=msg):
            s[idx]

    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    def test_slice_float(
        self, idx: slice, frame_or_series: Type[FrameOrSeries], indexer_sl: IndexerFunc
    ) -> None:
        index: Index = Index(np.arange(5.0)) + 0.1
        s: FrameOrSeries = gen_obj(frame_or_series, index)
        expected = s.iloc[3:4]
        result = indexer_sl(s)[idx]
        assert isinstance(result, type(s))
        tm.assert_equal(result, expected)
        s2: FrameOrSeries = s.copy()
        indexer_sl(s2)[idx] = 0
        result = indexer_sl(s2)[idx].values.ravel()
        assert (result == 0).all()

    def test_floating_index_doc_example(self) -> None:
        index: Index = Index([1.5, 2, 3, 4.5, 5])
        s: Series = Series(range(5), index=index)
        assert s[3] == 2
        assert s.loc[3] == 2
        assert s.iloc[3] == 3

    def test_floating_misc(self, indexer_sl: IndexerFunc) -> None:
        s: Series = Series(np.arange(5), index=np.arange(5) * 2.5, dtype=np.int64)
        result = indexer_sl(s)[1.0:3.0]
        expected = Series(1, index=[2.5])
        tm.assert_series_equal(result, expected)
        result = indexer_sl(s)[5.0]
        assert result == 2
        result = indexer_sl(s)[5]
        assert result == 2
        with pytest.raises(KeyError, match="^4$"):
            indexer_sl(s)[4]
        expected = Series([2, 0], index=Index([5.0, 0.0], dtype=np.float64))
        for fancy_idx in [[5.0, 0.0], np.array([5.0, 0.0])]:
            tm.assert_series_equal(indexer_sl(s)[fancy_idx], expected)
        expected = Series([2, 0], index=Index([5, 0], dtype="float64"))
        for fancy_idx in [[5, 0], np.array([5, 0])]:
            tm.assert_series_equal(indexer_sl(s)[fancy_idx], expected)
        result2 = indexer_sl(s)[2.0:5.0]
        result3 = indexer_sl(s)[2.0:5]
        result4 = indexer_sl(s)[2.1:5]
        tm.assert_series_equal(result2, result3)
        tm.assert_series_equal(result2, result4)
        result1 = indexer_sl(s)[[0.0, 5, 10]]
        result2 = s.iloc[[0, 2, 4]]
        tm.assert_series_equal(result1, result2)
        with pytest.raises(KeyError, match="not in index"):
            indexer_sl(s)[[1.6, 5, 10]]
        with pytest.raises(KeyError, match="not in index"):
            indexer_sl(s)[[0, 1, 2]]
        result = indexer_sl(s)[[2.5, 5]]
        tm.assert_series_equal(result, Series([1, 2], index=[2.5, 5.0]))
        result = indexer_sl(s)[[2.5]]
        tm.assert_series_equal(result, Series([1], index=[2.5]))

    def test_floatindex_slicing_bug(self, float_numpy_dtype: np.dtype) -> None:
        dtype: np.dtype = float_numpy_dtype
        ser: Dict[int, float] = {
            256: 2321.0,
            1: 78.0,
            2: 2716.0,
            3: 0.0,
            4: 369.0,
            5: 0.0,
            6: 269.0,
            7: 0.0,
            8: 0.0,
            9: 0.0,
            10: 3536.0,
            11: 0.0,
            12: 24.0,
            13: 0.0,
            14: 931.0,
            15: 0.0,
            16: 101.0,
            17: 78.0,
            18: 9643.0,
            19: 0.0,
            20: 0.0,
            21: 0.0,
            22: 63761.0,
            23: 0.0,
            24: 446.0,
            25: 0.0,
            26: 34773.0,
            27: 0.0,
            28: 729.0,
            29: 78.0,
            30: 0.0,
            31: 0.0,
            32: 3374.0,
            33: 0.0,
            34: 1391.0,
            35: 0.0,
            36: 361.0,
            37: 0.0,
            38: 61808.0,
            39: 0.0,
            40: 0.0,
            41: 0.0,
            42: 6677.0,
            43: 0.0,
            44: 802.0,
            45: 0.0,
            46: 2691.0,
            47: 0.0,
            48: 3582.0,
            49: 0.0,
            50: 734.0,
            51: 0.0,
            52: 627.0,
            53: 70.0,
            54: 2584.0,
            55: 0.0,
            56: 324.0,
            57: 0.0,
            58: 605.0,
            59: 0.0,
            60: 0.0,
            61: 0.0,
            62: 3989.0,
            63: 10.0,
            64: 42.0,
            65: 0.0,
            66: 904.0,
            67: 0.0,
            68: 88.0,
            69: 70.0,
            70: 8172.0,
            71: 0.0,
            72: 0.0,
            73: 0.0,
            74: 64902.0,
            75: 0.0,
            76: 347.0,
            77: 0.0,
            78: 36605.0,
            79: 0.0,
            80: 379.0,
            81: 70.0,
            82: 0.0,
            83: 0.0,
            84: 3001.0,
            85: 0.0,
            86: 1630.0,
            87: 7.0,
            88: 364.0,
            89: 0.0,
            90: 67404.0,
            91: 9.0,
            92: 0.0,
            93: 0.0,
            94: 7685.0,
            95: 0.0,
            96: 1017.0,
            97: 0.0,
            98: 2831.0,
            99: 0.0,
            100: 2963.0,
            101: 0.0,
            102: 854.0,
            103: 0.0,
            104: 0.0,
            105: 0.0,
            106: 0.0,
            107: 0.0,
            108: 0.0,
            109: 0.0,
            110: 0.0,
            111: 0.0,
            112: 0.0,
            113: 0.0,
            114: 0.0,
            115: 0.0,
            116: 0.0,
            117: 0.0,
            118: 0.0,
            119: 0.0,
            120: 0.0,
            121: 0.0,
            122: 0.0,
            123: 0.0,
            124: 0.0,
            125: 0.0,
            126: 67744.0,
            127: 22.0,
            128: 264.0,
            129: 0.0,
            260: 197.0,
            268: 0.0,
            265: 0.0,
            269: 0.0,
            261: 0.0,
            266: 1198.0,
            267: 0.0,
            262: 2629.0,
            258: 775.0,
            257: 0.0,
            263: 0.0,
            259: 0.0,
            264: 163.0,
            250: 10326.0,
            251: 0.0,
            252: 1228.0,
            253: 0.0,
            254: 2769.0,
            255: 0.0,
        }
        s: Series = Series(ser, dtype=dtype)
        result = s.value_counts()
        assert result.index.dtype == dtype
        str(result)