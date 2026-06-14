from __future__ import annotations
from typing import Callable, final
import numpy as np
import pandas as pd
from pandas import DataFrame, Index, Series

class BaseOpsUtil:
    series_scalar_exc: type[Exception] | None
    frame_scalar_exc: type[Exception] | None
    series_array_exc: type[Exception] | None
    divmod_exc: type[Exception] | None

    def _get_expected_exception(
        self, op_name: str, obj: object, other: object
    ) -> type[Exception] | None: ...

    def _cast_pointwise_result(
        self, op_name: str, obj: object, other: object, pointwise_result: pd.Series | pd.DataFrame
    ) -> pd.Series | pd.DataFrame: ...

    def get_op_from_name(self, op_name: str) -> Callable: ...

    @final
    def check_opname(self, ser: pd.Series | pd.DataFrame, op_name: str, other: object) -> None: ...

    @final
    def _combine(self, obj: pd.Series | pd.DataFrame, other: object, op: Callable) -> pd.Series | pd.DataFrame: ...

    @final
    def _check_op(
        self,
        ser: pd.Series | pd.DataFrame,
        op: Callable,
        other: object,
        op_name: str,
        exc: type[Exception] | None = ...,
    ) -> None: ...

    @final
    def _check_divmod_op(self, ser: object, op: Callable, other: object) -> None: ...

class BaseArithmeticOpsTests(BaseOpsUtil):
    """
    Various Series and DataFrame arithmetic ops methods.

    Subclasses supporting various ops should set the class variables
    to indicate that they support ops of that kind

    * series_scalar_exc = TypeError
    * frame_scalar_exc = TypeError
    * series_array_exc = TypeError
    * divmod_exc = TypeError
    """
    series_scalar_exc: type[Exception] | None
    frame_scalar_exc: type[Exception] | None
    series_array_exc: type[Exception] | None
    divmod_exc: type[Exception] | None

    def test_arith_series_with_scalar(self, data: object, all_arithmetic_operators: str) -> None: ...
    def test_arith_frame_with_scalar(self, data: object, all_arithmetic_operators: str) -> None: ...
    def test_arith_series_with_array(self, data: object, all_arithmetic_operators: str) -> None: ...
    def test_divmod(self, data: object) -> None: ...
    def test_divmod_series_array(self, data: object, data_for_twos: object) -> None: ...
    def test_add_series_with_extension_array(self, data: object) -> None: ...
    def test_direct_arith_with_ndframe_returns_not_implemented(
        self, data: object, box: type[pd.Series] | type[pd.DataFrame] | type[pd.Index], op_name: str
    ) -> None: ...

class BaseComparisonOpsTests(BaseOpsUtil):
    """Various Series and DataFrame comparison ops methods."""

    def _compare_other(
        self, ser: pd.Series, data: object, op: Callable, other: object
    ) -> None: ...
    def test_compare_scalar(self, data: object, comparison_op: Callable) -> None: ...
    def test_compare_array(self, data: object, comparison_op: Callable) -> None: ...

class BaseUnaryOpsTests(BaseOpsUtil):
    def test_invert(self, data: object) -> None: ...
    def test_unary_ufunc_dunder_equivalence(self, data: object, ufunc: Callable) -> None: ...