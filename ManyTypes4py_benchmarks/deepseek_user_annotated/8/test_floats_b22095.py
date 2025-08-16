import numpy as np
import pytest
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, cast

from pandas import (
    DataFrame,
    Index,
    RangeIndex,
    Series,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.indexes.base import Index as PandasIndex
from pandas.core.series import Series as PandasSeries
from pandas.core.frame import DataFrame as PandasDataFrame
from numpy.typing import NDArray

T = TypeVar('T', bound=Union[PandasSeries, PandasDataFrame])

def gen_obj(klass: Type[T], index: PandasIndex) -> T:
    if klass is Series:
        obj = Series(np.arange(len(index)), index=index)
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
        result: Union[PandasSeries, PandasDataFrame],
        original: Union[PandasSeries, PandasDataFrame],
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
        index: PandasIndex,
        frame_or_series: Type[T],
        indexer_sl: Any,
    ) -> None:
        # GH 4892
        # float_indexers should raise exceptions
        # on appropriate Index types & accessors

        s = gen_obj(frame_or_series, index)

        # getting
        with pytest.raises(KeyError, match="^3.0$"):
            indexer_sl(s)[3.0]

        # contains
        assert 3.0 not in s

        s2 = s.copy()
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
    def test_scalar_non_numeric_series_fallback(self, index: PandasIndex) -> None:
        # starting in 3.0, integer keys are always treated as labels, no longer
        #  fall back to positional.
        s = Series(np.arange(len(index)), index=index)

        with pytest.raises(KeyError, match="3"):
            s[3]
        with pytest.raises(KeyError, match="^3.0$"):
            s[3.0]

    def test_scalar_with_mixed(self, indexer_sl: Any) -> None:
        s2 = Series([1, 2, 3], index=["a", "b", "c"])
        s3 = Series([1, 2, 3], index=["a", "b", 1.5])

        # lookup in a pure string index with an invalid indexer

        with pytest.raises(KeyError, match="^1.0$"):
            indexer_sl(s2)[1.0]

        with pytest.raises(KeyError, match=r"^1\.0$"):
            indexer_sl(s2)[1.0]

        result = indexer_sl(s2)["b"]
        expected = 2
        assert result == expected

        # mixed index so we have label
        # indexing
        with pytest.raises(KeyError, match="^1.0$"):
            indexer_sl(s3)[1.0]

        if indexer_sl is not tm.loc:
            # as of 3.0, __getitem__ no longer falls back to positional
            with pytest.raises(KeyError, match="^1$"):
                s3[1]

        with pytest.raises(KeyError, match=r"^1\.0$"):
            indexer_sl(s3)[1.0]

        result = indexer_sl(s3)[1.5]
        expected = 3
        assert result == expected

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_scalar_integer(
        self,
        index: PandasIndex,
        frame_or_series: Type[T],
        indexer_sl: Any,
    ) -> None:
        getitem = indexer_sl is not tm.loc

        # test how scalar float indexers work on int indexes

        # integer index
        i = index
        obj = gen_obj(frame_or_series, i)

        # coerce to equal int

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

        s2 = obj.copy()
        indexer_sl(s2)[3.0] = 100

        result = indexer_sl(s2)[3.0]
        compare(result, expected)

        result = indexer_sl(s2)[3]
        compare(result, expected)

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_scalar_integer_contains_float(
        self,
        index: PandasIndex,
        frame_or_series: Type[T],
    ) -> None:
        # contains
        # integer index
        obj = gen_obj(frame_or_series, index)

        # coerce to equal int
        assert 3.0 in obj

    def test_scalar_float(self, frame_or_series: Type[T]) -> None:
        # scalar float indexers work on a float index
        index = Index(np.arange(5.0))
        s = gen_obj(frame_or_series, index)

        # assert all operations except for iloc are ok
        indexer = index[3]
        for idxr in [tm.loc, tm.setitem]:
            getitem = idxr is not tm.loc

            # getting
            result = idxr(s)[indexer]
            self.check(result, s, 3, getitem)

            # setting
            s2 = s.copy()

            result = idxr(s2)[indexer]
            self.check(result, s, 3, getitem)

            # random float is a KeyError
            with pytest.raises(KeyError, match=r"^3\.5$"):
                idxr(s)[3.5]

        # contains
        assert 3.0 in s

        # iloc succeeds with an integer
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
        index: PandasIndex,
        idx: slice,
        frame_or_series: Type[T],
        indexer_sli: Any,
    ) -> None:
        # GH 4892
        # float_indexers should raise exceptions
        # on appropriate Index types & accessors

        s = gen_obj(frame_or_series, index)

        # getitem
        if indexer_sli is tm.iloc:
            msg = (
                "cannot do positional indexing "
                rf"on {type(index).__name__} with these indexers \[(3|4)\.0\] of "
                "type float"
            )
        else:
            msg = (
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers "
                r"\[(3|4)(\.0)?\] "
                r"of type (float|int)"
            )
        with pytest.raises(TypeError, match=msg):
            indexer_sli(s)[idx]

        # setitem
        if indexer_sli is tm.iloc:
            # otherwise we keep the same message as above
            msg = "slice indices must be integers or None or have an __index__ method"
        with pytest.raises(TypeError, match=msg):
            indexer_sli(s)[idx] = 0

    def test_slice_integer(self) -> None:
        # same as above, but for Integer based indexes
        # these coerce to a like integer
        # oob indicates if we are out of bounds
        # of positional indexing
        for index, oob in [
            (Index(np.arange(5, dtype=np.int64)), False),
            (RangeIndex(5), False),
            (Index(np.arange(5, dtype=np.int64) + 10), True),
        ]:
            # s is an in-range index
            s = Series(range(5), index=index)

            # getitem
            for idx in [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)]:
                result = s.loc[idx]

                # these are all label indexing
                # except getitem which is positional
                # empty
                if oob:
                    indexer = slice(0, 0)
                else:
                    indexer = slice(3, 5)
                self.check(result, s, indexer, False)

            # getitem out-of-bounds
            for idx in [slice(-6, 6), slice(-6.0, 6.0)]:
                result = s.loc[idx]

                # these are all label indexing
                # except getitem which is positional
                # empty
                if oob:
                    indexer = slice(0, 0)
                else:
                    indexer = slice(-6, 6)
                self.check(result, s, indexer, False)

            # positional indexing
            msg = (
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers \[-6\.0\] of "
                "type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[slice(-6.0, 6.0)]

            # getitem odd floats
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

                # positional indexing
                msg = (
                    "cannot do slice indexing "
                    rf"on {type(index).__name__} with these indexers \[(2|3)\.5\] of "
                    "type float"
                )
                with pytest.raises(TypeError, match=msg):
                    s[idx]

    @pytest.mark.parametrize("idx", [slice(2, 4.0), slice(2.0, 4), slice(2.0, 4.0)])
    def test_integer_positional_indexing(self, idx: slice) -> None:
        """make sure that we are raising on positional indexing
        w.r.t. an integer index
        """
        s = Series(range(2, 6), index=range(2, 6))

        result = s[2:4]
        expected = s.iloc[2:4]
        tm.assert_series_equal(result, expected)

        klass = RangeIndex
        msg = (
            "cannot do (slice|positional) indexing "
            rf"on {klass.__name__} with these indexers \[(2|4)\.0\] of "
            "type float"
        )
        with pytest.raises(TypeError, match=msg):
            s[idx]
        with pytest.raises(TypeError, match=msg):
            s.iloc[idx]

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_slice_integer_frame_getitem(self, index: PandasIndex) -> None:
        # similar to above, but on the getitem dim (of a DataFrame)
        s = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=index)

        # getitem
        for idx in [slice(0.0, 1), slice(0, 1.0), slice(0.0, 1.0)]:
            result = s.loc[idx]
            indexer = slice(0, 2)
            self.check(result, s, indexer, False)

            # positional indexing
            msg = (
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers \[(0|1)\.0\] of "
                "type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[idx]

        # getitem out-of-bounds
        for idx in [slice(-10, 10), slice(-10.0, 10.0)]:
            result = s.loc[idx]
            self.check(result, s, slice(-10, 10), True)

        # positional indexing
        msg = (
            "cannot do slice indexing "
            rf"on {type(index).__name__} with these indexers \[-10\.0\] of "
            "type float"
        )
        with pytest.raises(TypeError, match=msg):
            s[slice(-10.0, 10.0)]

        # getitem odd floats
        for idx, res in [
            (slice(0.5, 1), slice(1, 2)),
            (slice(0, 0.5), slice(0, 1)),
            (slice(0.5, 1.5), slice(1, 2)),
        ]:
            result = s.loc[idx]
            self.check(result, s, res, False)

            # positional indexing
            msg = (
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers \[0\.5\] of "
                "type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[idx]

    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_float_slice_getitem_with_integer_index_raises(
        self,
        idx: slice,
        index: PandasIndex,
    ) -> None:
        # similar to above, but on the getitem dim (of a DataFrame)
        s = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=index)

        # setitem
        sc = s.copy()
        sc.loc[idx] = 0
        result = sc.loc[idx].values.ravel()
        assert (result == 0).all()

        # positional indexing
        msg = (
            "cannot do slice indexing "
            rf"on {type(index).__name__} with these indexers \[(3|4)\.0\] of "
            "type float"
        )
        with pytest.raises(TypeError, match=msg):
            s[idx] = 0

        with pytest.raises(TypeError, match=msg):
            s[idx]

    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    def test_slice_float(
        self,
        idx: slice,
        frame_or_series: Type[T],
        indexer_sl: Any,
    ) -> None:
        # same as above, but for floats
        index = Index(np.arange(5.0)) + 0.1
        s = gen_obj(frame_or_series, index)

        expected = s.iloc[3:4]

        # getitem
        result = indexer_sl(s)[idx]
        assert isinstance(result, type(s))
        tm.assert_equal(result, expected)

        # setitem
        s2 = s.copy()
        indexer_sl(s2)[idx] = 0
        result = indexer_sl(s2)[idx].values.