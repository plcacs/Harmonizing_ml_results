from __future__ import annotations
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    TypeVar,
    final,
)
from numpy import ndarray
from pandas import DataFrame, Series, Index
from pandas.core.dtypes.common import is_string_dtype
from pandas._testing import tm

T = TypeVar("T")

class BaseOpsUtil:
    series_scalar_exc: ClassVar[Type[TypeError]]
    frame_scalar_exc: ClassVar[Type[TypeError]]
    series_array_exc: ClassVar[Type[TypeError]]
    divmod_exc: ClassVar[Type[TypeError]]

    def _get_expected_exception(
        self, op_name: str, obj: Union[Series, DataFrame], other: Any
    ) -> Type[Exception]:
        ...

    def _cast_pointwise_result(
        self, op_name: str, obj: Union[Series, DataFrame], other: Any, pointwise_result: Any
    ) -> Any:
        ...

    def get_op_from_name(self, op_name: str) -> Callable:
        ...

    @final
    def check_opname(self, ser: T, op_name: str, other: Any) -> None:
        ...

    @final
    def _combine(
        self, obj: Union[Series, DataFrame], other: Any, op: Callable
    ) -> Union[Series, DataFrame]:
        ...

    @final
    def _check_op(
        self,
        ser: T,
        op: Callable,
        other: Any,
        op_name: str,
        exc: Optional[Type[Exception]] = NotImplementedError,
    ) -> None:
        ...

    @final
    def _check_divmod_op(self, ser: Series, op: Callable, other: Any) -> None:
        ...

class BaseArithmeticOpsTests(BaseOpsUtil):
    series_scalar_exc: ClassVar[Type[TypeError]]
    frame_scalar_exc: ClassVar[Type[TypeError]]
    series_array_exc: ClassVar[Type[TypeError]]
    divmod_exc: ClassVar[Type[TypeError]]

    def test_arith_series_with_scalar(
        self, data: Union[Series, ndarray], all_arithmetic_operators: str
    ) -> None:
        ...

    def test_arith_frame_with_scalar(
        self, data: Union[DataFrame, ndarray], all_arithmetic_operators: str
    ) -> None:
        ...

    def test_arith_series_with_array(
        self, data: Union[Series, ndarray], all_arithmetic_operators: str
    ) -> None:
        ...

    def test_divmod(self, data: Union[Series, ndarray]) -> None:
        ...

    def test_divmod_series_array(
        self, data: Union[Series, ndarray], data_for_twos: Union[Series, ndarray]
    ) -> None:
        ...

    def test_add_series_with_extension_array(self, data: Union[Series, ndarray]) -> None:
        ...

    @pytest.mark.parametrize("box", [Series, DataFrame, Index])
    @pytest.mark.parametrize(
        "op_name",
        [
            x
            for x in tm.arithmetic_dunder_methods + tm.comparison_dunder_methods
            if not x.startswith("__r")
        ],
    )
    def test_direct_arith_with_ndframe_returns_not_implemented(
        self, data: Union[Series, DataFrame, ndarray], box: Callable, op_name: str
    ) -> None:
        ...

class BaseComparisonOpsTests(BaseOpsUtil):
    def _compare_other(
        self,
        ser: Series,
        data: Union[Series, ndarray],
        op: Callable,
        other: Union[Any, Series],
    ) -> None:
        ...

    def test_compare_scalar(self, data: Union[Series, ndarray], comparison_op: Callable) -> None:
        ...

    def test_compare_array(
        self, data: Union[Series, ndarray], comparison_op: Callable
    ) -> None:
        ...

class BaseUnaryOpsTests(BaseOpsUtil):
    def test_invert(self, data: Union[Series, ndarray]) -> None:
        ...

    @pytest.mark.parametrize("ufunc", [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(
        self, data: Union[Series, ndarray], ufunc: Callable
    ) -> None:
        ...