from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
import itertools
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pytest

from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.compat.numpy import np_version_gt2

import pandas as pd
from pandas import (
    Index,
    Series,
    TimedeltaIndex,
    DatetimeIndex,
    PeriodIndex,
)
from pandas._testing import tm
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)

###############################################################
# Index / Series common tests which may trigger dtype coercions
###############################################################


@pytest.fixture(autouse=True, scope="class")
def check_comprehensiveness(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    # Iterate over combination of dtype, method and klass
    # and ensure that each are contained within a collected test
    cls = request.cls
    combos = itertools.product(cls.klasses, cls.dtypes, [cls.method])

    def has_test(combo: Tuple[str, str, str]) -> bool:
        klass, dtype, method = combo
        cls_funcs = request.node.session.items
        return any(
            klass in x.name and dtype in x.name and method in x.name for x in cls_funcs
        )

    opts = request.config.option
    if opts.lf or opts.keyword:
        # If we are running with "last-failed" or -k foo, we expect to only
        #  run a subset of tests.
        yield

    else:
        for combo in combos:
            if not has_test(combo):
                raise AssertionError(
                    f"test method is not defined: {cls.__name__}, {combo}"
                )

        yield


class CoercionBase:
    klasses: List[str] = ["index", "series"]
    dtypes: List[str] = [
        "object",
        "int64",
        "float64",
        "complex128",
        "bool",
        "datetime64",
        "datetime64tz",
        "timedelta64",
        "period",
    ]

    @property
    def method(self) -> str:
        raise NotImplementedError(self)


class TestSetitemCoercion(CoercionBase):
    method: str = "setitem"

    # disable comprehensiveness tests, as most of these have been moved to
    #  tests.series.indexing.test_setitem in SetitemCastingEquivalents subclasses.
    klasses: List[str] = []

    def test_setitem_series_no_coercion_from_values_list(self) -> None:
        # GH35865 - int casted to str when internally calling np.array(ser.values)
        ser = pd.Series(["a", 1])
        ser[:] = list(ser.values)

        expected = pd.Series(["a", 1])

        tm.assert_series_equal(ser, expected)

    def _assert_setitem_index_conversion(
        self,
        original_series: Series,
        loc_key: Any,
        expected_index: Index,
        expected_dtype: Any,
    ) -> None:
        """test index's coercion triggered by assign key"""
        temp = original_series.copy()
        # GH#33469 pre-2.0 with int loc_key and temp.index.dtype == np.float64
        #  `temp[loc_key] = 5` treated loc_key as positional
        temp[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        # check dtype explicitly for sure
        assert temp.index.dtype == expected_dtype

        temp = original_series.copy()
        temp.loc[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        # check dtype explicitly for sure
        assert temp.index.dtype == expected_dtype

    @pytest.mark.parametrize(
        "val,exp_dtype", [("x", object), (5, IndexError), (1.1, object)]
    )
    def test_setitem_index_object(self, val: Any, exp_dtype: Any) -> None:
        obj = pd.Series([1, 2, 3, 4], index=pd.Index(list("abcd"), dtype=object))
        assert obj.index.dtype == object

        if exp_dtype is IndexError:
            with pytest.raises(exp_dtype):
                self._assert_setitem_index_conversion(obj, val, None, None)  # type: ignore
        else:
            exp_index = pd.Index(list("abcd") + [val], dtype=object)
            self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(5, np.int64), (1.1, np.float64), ("x", object)]
    )
    def test_setitem_index_int64(self, val: Any, exp_dtype: Any) -> None:
        obj = pd.Series([1, 2, 3, 4])
        assert obj.index.dtype == np.int64

        exp_index = pd.Index([0, 1, 2, 3, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(5, np.float64), (5.1, np.float64), ("x", object)]
    )
    def test_setitem_index_float64(self, val: Any, exp_dtype: Any, request: pytest.FixtureRequest) -> None:
        obj = pd.Series([1, 2, 3, 4], index=[1.1, 2.1, 3.1, 4.1])
        assert obj.index.dtype == np.float64

        exp_index = pd.Index([1.1, 2.1, 3.1, 4.1, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_series_period(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_complex128(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_bool(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_datetime64(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_datetime64tz(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_timedelta64(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_period(self) -> None:
        raise NotImplementedError


# ... (rest of the classes would follow the same pattern with type annotations)

