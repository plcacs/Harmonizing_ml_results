from __future__ import annotations

from functools import reduce
from itertools import product
import operator
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from pandas.compat import PY312
from pandas.errors import (
    NumExprClobberingError,
    PerformanceWarning,
    UndefinedVariableError,
)
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import (
    is_bool,
    is_float,
    is_list_like,
    is_scalar,
)

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.computation import (
    expr,
    pytables,
)
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import (
    BaseExprVisitor,
    PandasExprVisitor,
    PythonExprVisitor,
)
from pandas.core.computation.expressions import (
    NUMEXPR_INSTALLED,
    USE_NUMEXPR,
)
from pandas.core.computation.ops import (
    ARITH_OPS_SYMS,
    _binary_math_ops,
    _binary_ops_dict,
    _unary_math_ops,
)
from pandas.core.computation.scope import DEFAULT_GLOBALS


@pytest.fixture(
    params=(
        pytest.param(
            engine,
            marks=[
                pytest.mark.skipif(
                    engine == "numexpr" and not USE_NUMEXPR,
                    reason=f"numexpr enabled->{USE_NUMEXPR}, "
                    f"installed->{NUMEXPR_INSTALLED}",
                ),
                td.skip_if_no("numexpr"),
            ],
        )
        for engine in ENGINES
    )
)
def engine(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=expr.PARSERS)
def parser(request: pytest.FixtureRequest) -> str:
    return request.param


def _eval_single_bin(lhs: Any, cmp1: str, rhs: Any, engine: str) -> Any:
    c = _binary_ops_dict[cmp1]
    if ENGINES[engine].has_neg_frac:
        try:
            return c(lhs, rhs)
        except ValueError as e:
            if str(e).startswith(
                "negative number cannot be raised to a fractional power"
            ):
                return np.nan
            raise
    return c(lhs, rhs)


# TODO: using range(5) here is a kludge
@pytest.fixture(
    params=list(range(5)),
    ids=["DataFrame", "Series", "SeriesNaN", "DataFrameNaN", "float"],
)
def lhs(request: pytest.FixtureRequest) -> Union[DataFrame, Series, float]:
    nan_df1 = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    nan_df1[nan_df1 > 0.5] = np.nan

    opts = (
        DataFrame(np.random.default_rng(2).standard_normal((10, 5))),
        Series(np.random.default_rng(2).standard_normal(5)),
        Series([1, 2, np.nan, np.nan, 5]),
        nan_df1,
        np.random.default_rng(2).standard_normal(),
    )
    return opts[request.param]


rhs = lhs
midhs = lhs


@pytest.fixture
def idx_func_dict() -> Dict[str, Any]:
    return {
        "i": lambda n: Index(np.arange(n),
        "f": lambda n: Index(np.arange(n),