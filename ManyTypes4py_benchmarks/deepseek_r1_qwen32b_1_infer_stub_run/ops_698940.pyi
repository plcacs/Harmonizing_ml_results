from __future__ import annotations
from typing import Any, Callable, List, Optional, Type, Union
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype
from pandas.core import ops
from pytest import raises

class BaseOpsUtil:
    series_scalar_exc: Type[Exception]
    frame_scalar_exc: Type[Exception]
    series_array_exc: Type[Exception]
    divmod_exc: Type[Exception]

    def _get_expected_exception(self, op_name: str, obj: Any, other: Any) -> Type[Exception]:
        ...

    def _cast_pointwise_result(self, op_name: str, obj: Any, other: Any, pointwise_result: Any) -> Any:
        ...

    def get_op_from_name(self, op_name: str) -> Callable:
        ...

    @final
    def check_opname(self, ser: pd.Series, op_name: str, other: Any) -> None:
        ...

    @final
    def _combine(self, obj: Union[pd.Series, pd.DataFrame], other: Any, op: Callable) -> Any:
        ...

    @final
    def _check_op(self, ser: pd.Series, op: Callable, other: Any, op_name: str, exc: Optional[Type[Exception]] = NotImplementedError) -> None:
        ...

    @final
    def _check_divmod_op(self, ser: pd.Series, op: Callable, other: Any) -> None:
        ...

class BaseArithmeticOpsTests(BaseOpsUtil):
    series_scalar_exc: Type[Exception]
    frame_scalar_exc: Type[Exception]
    series_array_exc: Type[Exception]
    divmod_exc: Type[Exception]

    def test_arith_series_with_scalar(self, data: Any, all_arithmetic_operators: str) -> None:
        ...

    def test_arith_frame_with_scalar(self, data: Any, all_arithmetic_operators: str) -> None:
        ...

    def test_arith_series_with_array(self, data: Any, all_arithmetic_operators: str) -> None:
        ...

    def test_divmod(self, data: Any) -> None:
        ...

    def test_divmod_series_array(self, data: Any, data_for_twos: Any) -> None:
        ...

    def test_add_series_with_extension_array(self, data: Any) -> None:
        ...

    @pytest.mark.parametrize('box', [pd.Series, pd.DataFrame, pd.Index])
    @pytest.mark.parametrize('op_name', [x for x in tm.arithmetic_dunder_methods + tm.comparison_dunder_methods if not x.startswith('__r')])
    def test_direct_arith_with_ndframe_returns_not_implemented(self, data: Any, box: Callable, op_name: str) -> None:
        ...

class BaseComparisonOpsTests(BaseOpsUtil):
    def _compare_other(self, ser: pd.Series, data: Any, op: Callable, other: Any) -> None:
        ...

    def test_compare_scalar(self, data: Any, comparison_op: Callable) -> None:
        ...

    def test_compare_array(self, data: Any, comparison_op: Callable) -> None:
        ...

class BaseUnaryOpsTests(BaseOpsUtil):
    def test_invert(self, data: Any) -> None:
        ...

    @pytest.mark.parametrize('ufunc', [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data: Any, ufunc: Callable) -> None:
        ...