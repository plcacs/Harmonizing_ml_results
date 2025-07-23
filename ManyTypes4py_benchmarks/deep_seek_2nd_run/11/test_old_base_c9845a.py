from __future__ import annotations
from datetime import datetime
import weakref
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import is_integer_dtype, is_numeric_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
    CategoricalIndex,
    DatetimeIndex,
    DatetimeTZDtype,
    Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    StringDtype,
    TimedeltaIndex,
    isna,
    period_range,
)
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import BaseMaskedArray


class TestBase:
    @pytest.fixture(
        params=[
            RangeIndex(start=0, stop=20, step=2),
            Index(np.arange(5, dtype=np.float64)),
            Index(np.arange(5, dtype=np.float32)),
            Index(np.arange(5, dtype=np.uint64)),
            Index(range(0, 20, 2), dtype=np.int64),
            Index(range(0, 20, 2), dtype=np.int32),
            Index(range(0, 20, 2), dtype=np.int16),
            Index(range(0, 20, 2), dtype=np.int8),
            Index(list("abcde")),
            Index([0, "a", 1, "b", 2, "c"]),
            period_range("20130101", periods=5, freq="D"),
            TimedeltaIndex(
                [
                    "0 days 01:00:00",
                    "1 days 01:00:00",
                    "2 days 01:00:00",
                    "3 days 01:00:00",
                    "4 days 01:00:00",
                ],
                dtype="timedelta64[ns]",
                freq="D",
            ),
            DatetimeIndex(
                [
                    "2013-01-01",
                    "2013-01-02",
                    "2013-01-03",
                    "2013-01-04",
                    "2013-01-05",
                ],
                dtype="datetime64[ns]",
                freq="D",
            ),
            IntervalIndex.from_breaks(range(11), closed="right"),
        ]
    )
    def simple_index(self, request: pytest.FixtureRequest) -> Index:
        return request.param

    def test_pickle_compat_construction(self, simple_index: Index) -> None:
        if isinstance(simple_index, RangeIndex):
            pytest.skip("RangeIndex() is a valid constructor")
        msg = "|".join(
            [
                "Index\\(\\.\\.\\.\\) must be called with a collection of some kind, None was passed",
                "DatetimeIndex\\(\\) must be called with a collection of some kind, None was passed",
                "TimedeltaIndex\\(\\) must be called with a collection of some kind, None was passed",
                "__new__\\(\\) missing 1 required positional argument: 'data'",
                "__new__\\(\\) takes at least 2 arguments \\(1 given\\)",
                "'NoneType' object is not iterable",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            type(simple_index)()

    def test_shift(self, simple_index: Index) -> None:
        if isinstance(simple_index, (DatetimeIndex, TimedeltaIndex, PeriodIndex)):
            pytest.skip("Tested in test_ops/test_arithmetic")
        idx = simple_index
        msg = f"This method is only implemented for DatetimeIndex, PeriodIndex and TimedeltaIndex; Got type {type(idx).__name__}"
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1)
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1, 2)

    def test_constructor_name_unhashable(self, simple_index: Index) -> None:
        idx = simple_index
        with pytest.raises(TypeError, match="Index.name must be a hashable type"):
            type(idx)(idx, name=[])

    def test_create_index_existing_name(self, simple_index: Index) -> None:
        expected = simple_index.copy()
        if not isinstance(expected, MultiIndex):
            expected.name = "foo"
            result = Index(expected)
            tm.assert_index_equal(result, expected)
            result = Index(expected, name="bar")
            expected.name = "bar"
            tm.assert_index_equal(result, expected)
        else:
            expected.names = ["foo", "bar"]
            result = Index(expected)
            tm.assert_index_equal(
                result,
                Index(
                    Index(
                        [
                            ("foo", "one"),
                            ("foo", "two"),
                            ("bar", "one"),
                            ("baz", "two"),
                            ("qux", "one"),
                            ("qux", "two"),
                        ],
                        dtype="object",
                    ),
                    names=["foo", "bar"],
                ),
            )
            result = Index(expected, names=["A", "B"])
            tm.assert_index_equal(
                result,
                Index(
                    Index(
                        [
                            ("foo", "one"),
                            ("foo", "two"),
                            ("bar", "one"),
                            ("baz", "two"),
                            ("qux", "one"),
                            ("qux", "two"),
                        ],
                        dtype="object",
                    ),
                    names=["A", "B"],
                ),
            )

    def test_numeric_compat(self, simple_index: Index) -> None:
        idx = simple_index
        assert not isinstance(idx, MultiIndex)
        if type(idx) is Index:
            pytest.skip("Not applicable for Index")
        if is_numeric_dtype(simple_index.dtype) or isinstance(
            simple_index, TimedeltaIndex
        ):
            pytest.skip("Tested elsewhere.")
        typ = type(idx._data).__name__
        cls = type(idx).__name__
        lmsg = "|".join(
            [
                f"unsupported operand type\\(s\\) for \\*: '{typ}' and 'int'",
                f"cannot perform (__mul__|__truediv__|__floordiv__) with this index type: ({cls}|{typ})",
            ]
        )
        with pytest.raises(TypeError, match=lmsg):
            idx * 1
        rmsg = "|".join(
            [
                f"unsupported operand type\\(s\\) for \\*: 'int' and '{typ}'",
                f"cannot perform (__rmul__|__rtruediv__|__rfloordiv__) with this index type: ({cls}|{typ})",
            ]
        )
        with pytest.raises(TypeError, match=rmsg):
            1 * idx
        div_err = lmsg.replace("*", "/")
        with pytest.raises(TypeError, match=div_err):
            idx / 1
        div_err = rmsg.replace("*", "/")
        with pytest.raises(TypeError, match=div_err):
            1 / idx
        floordiv_err = lmsg.replace("*", "//")
        with pytest.raises(TypeError, match=floordiv_err):
            idx // 1
        floordiv_err = rmsg.replace("*", "//")
        with pytest.raises(TypeError, match=floordiv_err):
            1 // idx

    def test_logical_compat(self, simple_index: Index) -> None:
        if simple_index.dtype in (object, "string"):
            pytest.skip("Tested elsewhere.")
        idx = simple_index
        if idx.dtype.kind in "iufcbm":
            assert idx.all() == idx._values.all()
            assert idx.all() == idx.to_series().all()
            assert idx.any() == idx._values.any()
            assert idx.any() == idx.to_series().any()
        else:
            msg = "does not support operation '(any|all)'"
            with pytest.raises(TypeError, match=msg):
                idx.all()
            with pytest.raises(TypeError, match=msg):
                idx.any()

    def test_repr_roundtrip(self, simple_index: Index) -> None:
        if isinstance(simple_index, IntervalIndex):
            pytest.skip(f"Not a valid repr for {type(simple_index).__name__}")
        idx = simple_index
        tm.assert_index_equal(eval(repr(idx)), idx)

    def test_repr_max_seq_item_setting(self, simple_index: Index) -> None:
        if isinstance(simple_index, IntervalIndex):
            pytest.skip(f"Not a valid repr for {type(simple_index).__name__}")
        idx = simple_index
        idx = idx.repeat(50)
        with pd.option_context("display.max_seq_items", None):
            repr(idx)
            assert "..." not in str(idx)

    @pytest.mark.filterwarnings("ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning")
    def test_ensure_copied_data(self, index: Index) -> None:
        init_kwargs: Dict[str, Any] = {}
        if isinstance(index, PeriodIndex):
            init_kwargs["freq"] = index.freq
        elif isinstance(index, (RangeIndex, MultiIndex, CategoricalIndex)):
            pytest.skip(
                "RangeIndex cannot be initialized from data, MultiIndex and CategoricalIndex are tested separately"
            )
        elif index.dtype == object and index.inferred_type in ["boolean", "string"]:
            init_kwargs["dtype"] = index.dtype
        index_type = type(index)
        result = index_type(index.values, copy=True, **init_kwargs)
        if isinstance(index.dtype, DatetimeTZDtype):
            result = result.tz_localize("UTC").tz_convert(index.tz)
        if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
            index = index._with_freq(None)
        tm.assert_index_equal(index, result)
        if isinstance(index, PeriodIndex):
            result = index_type.from_ordinals(ordinals=index.asi8, **init_kwargs)
            tm.assert_numpy_array_equal(index.asi8, result.asi8, check_same="same")
        elif isinstance(index, IntervalIndex):
            pass
        elif type(index) is Index and (not isinstance(index.dtype, np.dtype)):
            result = index_type(index.values, copy=False, **init_kwargs)
            tm.assert_index_equal(result, index)
            if isinstance(index._values, BaseMaskedArray):
                assert np.shares_memory(
                    index._values._data, result._values._data
                )
                tm.assert_numpy_array_equal(
                    index._values._data, result._values._data, check_same="same"
                )
                assert np.shares_memory(
                    index._values._mask, result._values._mask
                )
                tm.assert_numpy_array_equal(
                    index._values._mask, result._values._mask, check_same="same"
                )
            elif isinstance(index.dtype, StringDtype) and index.dtype.storage == "python":
                assert np.shares_memory(
                    index._values._ndarray, result._values._ndarray
                )
                tm.assert_numpy_array_equal(
                    index._values._ndarray,
                    result._values._ndarray,
                    check_same="same",
                )
            elif isinstance(index.dtype, StringDtype) and index.dtype.storage == "pyarrow":
                assert tm.shares_memory(result._values, index._values)
            else:
                raise NotImplementedError(index.dtype)
        else:
            result = index_type(index.values, copy=False, **init_kwargs)
            tm.assert_numpy_array_equal(
                index.values, result.values, check_same="same"
            )

    def test_memory_usage(self, index: Index) -> None:
        index._engine.clear_mapping()
        result = index.memory_usage()
        if index.empty:
            assert result == 0
            return
        index.get_loc(index[0])
        result2 = index.memory_usage()
        result3 = index.memory_usage(deep=True)
        if not isinstance(index, (RangeIndex, IntervalIndex)) and (
            not (type(index) is Index and (not isinstance(index.dtype, np.dtype)))
        ):
            assert result2 > result
        if index.inferred_type == "object":
            assert result3 > result2

    def test_memory_usage_doesnt_trigger_engine(self, index: Index) -> None:
        index._cache.clear()
        assert "_engine" not in index._cache
        res_without_engine = index.memory_usage()
        assert "_engine" not in index._cache
        _ = index._engine
        assert "_engine" in index._cache
        res_with_engine = index.memory_usage()
        assert res_with_engine == res_without_engine
        if len(index) == 0:
            assert res_without_engine == 0
            assert res_with_engine == 0
        else:
            assert res_without_engine > 0
            assert res_with_engine > 0

    def test_argsort(self, index: Index) -> None:
        if isinstance(index, CategoricalIndex):
            pytest.skip(f"{type(self).__name__} separately tested")
        result = index.argsort()
        expected = np.array(index).argsort()
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_numpy_argsort(self, index: Index) -> None:
        result = np.argsort(index)
        expected = index.argsort()
        tm.assert_numpy_array_equal(result, expected)
        result = np.argsort(index, kind="mergesort")
        expected = index.argsort(kind="mergesort")
        tm.assert_numpy_array_equal(result, expected)
        if isinstance(index, (CategoricalIndex, RangeIndex)):
            msg = "the 'axis' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, axis=1)
            msg = "the 'order' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, order=("a", "b"))

    def test_repeat(self, simple_index: Index) -> None:
        rep = 2
        idx = simple_index.copy()
        new_index_cls = idx._constructor
        expected = new_index_cls(idx.values.repeat(rep), name=idx.name)
        tm.assert_index_equal(idx.repeat(rep), expected)
        idx = simple_index
        rep = np.arange(len(idx))
        expected = new_index_cls(idx.values.repeat(rep), name=idx.name)
        tm.assert_index_equal(idx.repeat(rep), expected)

    def test_numpy_repeat(self, simple_index: Index) -> None:
        rep = 2
        idx = simple_index
        expected = idx.repeat(rep)
        tm.assert_index_equal(np.repeat(idx, rep), expected)
        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.repeat(idx, rep, axis=0)

    def test_where(
        self, listlike_box: Any, simple_index: Index
    ) -> None:
        if isinstance(simple_index, (IntervalIndex, PeriodIndex)) or is_numeric_dtype(
            simple_index.dtype
        ):
            pytest.skip("Tested elsewhere.")
        klass = listlike_box
        idx = simple_index
        if isinstance(idx, (DatetimeIndex, TimedeltaIndex)):
            idx = idx._with_freq(None)
        cond = [True] * len(idx)
        result = idx.where(klass(cond))
        expected = idx
        tm.assert_index_equal(result, expected)
        cond = [False] + [True] * len(idx[1:])
        expected = Index([idx._na_value] + idx[1:].tolist(), dtype=idx.dtype)
        result = idx.where(klass(cond))
        tm.assert_index_equal(result, expected)

    def test_insert_base(self, index: Index) -> None:
        trimmed = index[1:4]
        if not len(index):
            pytest.skip("Not applicable for empty index")
        result = trimmed.insert(0, index[0])
        assert index[0:4].equals(result)

    def test_insert_out_of_bounds(
        self, index: Index, using_infer_string: bool
    ) -> None:
        if len(index) > 0:
            err = TypeError
        else:
            err = IndexError
        if len(index) == 0:
            msg = "index (0|0.5) is out of bounds for axis 0 with size 0"
        else:
            msg = "slice indices must be integers or None or have an __index__ method"
        if using_infer_string and (
            index.dtype == "string" or index.dtype == "category"
        ):
            msg = "loc must be an integer between"
        with pytest.raises(err, match=msg):
            index.insert(0.5, "foo")
        msg = "|".join(
            [
                "index -?\\d+ is out of bounds for axis 0 with size \\d+",
                "loc must be an integer between",
            ]
        )
        with pytest.raises(IndexError, match=msg):
            index.insert(len(index) + 1, 1)
        with pytest.raises(IndexError, match=msg):
            index.insert(-len(index) - 1, 1)

    def test_delete_base(self, index: Index) -> None:
        if not len(index):
            pytest.skip("Not applicable for empty index")
        if isinstance(index, RangeIndex):
            pytest.skip(f"{type(self).__name__} tested elsewhere")
        expected = index[1:]
        result = index.delete(0)
        assert result.equals(expected)
        assert result.name == expected.name
        expected = index[:-1]
        result = index.delete(-1)
        assert result.equals(expected)
        assert result.name == expected.name
        length = len(index)
        msg = f"index {length} is out of bounds for axis 0 with size {length}"
        with pytest.raises(IndexError, match=msg):
            index.delete(length)

    @pytest.mark.filterwarnings("ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning")
    def test_equals(self, index: Index) -> None:
        if isinstance(index, IntervalIndex