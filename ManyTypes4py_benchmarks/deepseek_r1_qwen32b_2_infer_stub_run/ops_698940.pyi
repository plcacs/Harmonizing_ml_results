from __future__ import annotations
from typing import Any, Union, Type, Optional, List, Tuple, Final, Dict, Callable, TypeVar, overload
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.dtypes import ExtensionDtype

class BaseOpsUtil:
    series_scalar_exc: Type[BaseException]
    frame_scalar_exc: Type[BaseException]
    series_array_exc: Type[BaseException]
    divmod_exc: Type[BaseException]
    
    def _get_expected_exception(self, op_name: str, obj: Any, other: Any) -> Type[BaseException]: ...
    
    def _cast_pointwise_result(self, op_name: str, obj: Any, other: Any, pointwise_result: Any) -> Any: ...
    
    def get_op_from_name(self, op_name: str) -> Callable[..., Any]: ...
    
    @final
    def check_opname(self, ser: pd.Series, op_name: str, other: Any) -> None: ...
    
    @final
    def _combine(self, obj: Union[pd.Series, pd.DataFrame], other: Any, op: Callable[..., Any]) -> Union[pd.Series, pd.DataFrame]: ...
    
    @final
    def _check_op(self, ser: pd.Series, op: Callable[..., Any], other: Any, op_name: str, exc: Optional[Type[BaseException]] = NotImplementedError) -> None: ...
    
    @final
    def _check_divmod_op(self, ser: pd.Series, op: Callable[..., Any], other: Any) -> None: ...

class BaseArithmeticOpsTests(BaseOpsUtil):
    series_scalar_exc: Type[BaseException]
    frame_scalar_exc: Type[BaseException]
    series_array_exc: Type[BaseException]
    divmod_exc: Type[BaseException]
    
    def test_arith_series_with_scalar(self, data: ExtensionArray, all_arithmetic_operators: str) -> None: ...
    
    def test_arith_frame_with_scalar(self, data: ExtensionArray, all_arithmetic_operators: str) -> None: ...
    
    def test_arith_series_with_array(self, data: ExtensionArray, all_arithmetic_operators: str) -> None: ...
    
    def test_divmod(self, data: ExtensionArray) -> None: ...
    
    def test_divmod_series_array(self, data: ExtensionArray, data_for_twos: ExtensionArray) -> None: ...
    
    def test_add_series_with_extension_array(self, data: ExtensionArray) -> None: ...
    
    @pytest.mark.parametrize('box', [pd.Series, pd.DataFrame, pd.Index])
    @pytest.mark.parametrize('op_name', [x for x in tm.arithmetic_dunder_methods + tm.comparison_dunder_methods if not x.startswith('__r')])
    def test_direct_arith_with_ndframe_returns_not_implemented(self, data: ExtensionArray, box: type, op_name: str) -> None: ...

class BaseComparisonOpsTests(BaseOpsUtil):
    def _compare_other(self, ser: pd.Series, data: ExtensionArray, op: Callable[..., Any], other: Any) -> None: ...
    
    def test_compare_scalar(self, data: ExtensionArray, comparison_op: Callable[..., Any]) -> None: ...
    
    def test_compare_array(self, data: ExtensionArray, comparison_op: Callable[..., Any]) -> None: ...

class BaseUnaryOpsTests(BaseOpsUtil):
    def test_invert(self, data: ExtensionArray) -> None: ...
    
    @pytest.mark.parametrize('ufunc', [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data: ExtensionArray, ufunc: np.ufunc) -> None: ...