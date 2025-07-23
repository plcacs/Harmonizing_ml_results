from __future__ import annotations
from typing import Any, Callable, Dict, Final, Optional, Type, TypeVar, Union, final
import numpy as np
import pytest
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core import ops

T = TypeVar('T')
OpName = str
ExcType = Type[Exception]
OpFunc = Callable[[Any, Any], Any]

class BaseOpsUtil:
    series_scalar_exc: ExcType = TypeError
    frame_scalar_exc: ExcType = TypeError
    series_array_exc: ExcType = TypeError
    divmod_exc: ExcType = TypeError

    def _get_expected_exception(self, op_name: str, obj: Union[pd.Series, pd.DataFrame], other: Any) -> ExcType:
        if op_name in ['__divmod__', '__rdivmod__']:
            result = self.divmod_exc
        elif isinstance(obj, pd.Series) and isinstance(other, pd.Series):
            result = self.series_array_exc
        elif isinstance(obj, pd.Series):
            result = self.series_scalar_exc
        else:
            result = self.frame_scalar_exc
        return result

    def _cast_pointwise_result(self, op_name: str, obj: Union[pd.Series, pd.DataFrame], other: Any, pointwise_result: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        return pointwise_result

    def get_op_from_name(self, op_name: str) -> OpFunc:
        return tm.get_op_from_name(op_name)

    @final
    def check_opname(self, ser: Union[pd.Series, pd.DataFrame], op_name: str, other: Any) -> None:
        exc = self._get_expected_exception(op_name, ser, other)
        op = self.get_op_from_name(op_name)
        self._check_op(ser, op, other, op_name, exc)

    @final
    def _combine(self, obj: Union[pd.Series, pd.DataFrame], other: Any, op: OpFunc) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(obj, pd.DataFrame):
            if len(obj.columns) != 1:
                raise NotImplementedError
            expected = obj.iloc[:, 0].combine(other, op).to_frame()
        else:
            expected = obj.combine(other, op)
        return expected

    @final
    def _check_op(self, ser: Union[pd.Series, pd.DataFrame], op: OpFunc, other: Any, op_name: str, exc: Optional[ExcType] = NotImplementedError) -> None:
        if exc is None:
            result = op(ser, other)
            expected = self._combine(ser, other, op)
            expected = self._cast_pointwise_result(op_name, ser, other, expected)
            assert isinstance(result, type(ser))
            tm.assert_equal(result, expected)
        else:
            with pytest.raises(exc):
                op(ser, other)

    @final
    def _check_divmod_op(self, ser: Any, op: OpFunc, other: Any) -> None:
        if op is divmod:
            exc = self._get_expected_exception('__divmod__', ser, other)
        else:
            exc = self._get_expected_exception('__rdivmod__', ser, other)
        if exc is None:
            result_div, result_mod = op(ser, other)
            if op is divmod:
                expected_div, expected_mod = (ser // other, ser % other)
            else:
                expected_div, expected_mod = (other // ser, other % ser)
            tm.assert_series_equal(result_div, expected_div)
            tm.assert_series_equal(result_mod, expected_mod)
        else:
            with pytest.raises(exc):
                divmod(ser, other)

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
    series_scalar_exc: ExcType = TypeError
    frame_scalar_exc: ExcType = TypeError
    series_array_exc: ExcType = TypeError
    divmod_exc: ExcType = TypeError

    def test_arith_series_with_scalar(self, data: Any, all_arithmetic_operators: str) -> None:
        if all_arithmetic_operators == '__rmod__' and is_string_dtype(data.dtype):
            pytest.skip('Skip testing Python string formatting')
        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        self.check_opname(ser, op_name, ser.iloc[0])

    def test_arith_frame_with_scalar(self, data: Any, all_arithmetic_operators: str) -> None:
        if all_arithmetic_operators == '__rmod__' and is_string_dtype(data.dtype):
            pytest.skip('Skip testing Python string formatting')
        op_name = all_arithmetic_operators
        df = pd.DataFrame({'A': data})
        self.check_opname(df, op_name, data[0])

    def test_arith_series_with_array(self, data: Any, all_arithmetic_operators: str) -> None:
        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        self.check_opname(ser, op_name, pd.Series([ser.iloc[0]] * len(ser)))

    def test_divmod(self, data: Any) -> None:
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, 1)
        self._check_divmod_op(1, ops.rdivmod, ser)

    def test_divmod_series_array(self, data: Any, data_for_twos: Any) -> None:
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, data)
        other = data_for_twos
        self._check_divmod_op(other, ops.rdivmod, ser)
        other = pd.Series(other)
        self._check_divmod_op(other, ops.rdivmod, ser)

    def test_add_series_with_extension_array(self, data: Any) -> None:
        ser = pd.Series(data)
        exc = self._get_expected_exception('__add__', ser, data)
        if exc is not None:
            with pytest.raises(exc):
                ser + data
            return
        result = ser + data
        expected = pd.Series(data + data)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('box', [pd.Series, pd.DataFrame, pd.Index])
    @pytest.mark.parametrize('op_name', [x for x in tm.arithmetic_dunder_methods + tm.comparison_dunder_methods if not x.startswith('__r')])
    def test_direct_arith_with_ndframe_returns_not_implemented(self, data: Any, box: Any, op_name: str) -> None:
        other = box(data)
        if hasattr(data, op_name):
            result = getattr(data, op_name)(other)
            assert result is NotImplemented

class BaseComparisonOpsTests(BaseOpsUtil):
    """Various Series and DataFrame comparison ops methods."""

    def _compare_other(self, ser: pd.Series, data: Any, op: OpFunc, other: Any) -> None:
        if op.__name__ in ['eq', 'ne']:
            result = op(ser, other)
            expected = ser.combine(other, op)
            expected = self._cast_pointwise_result(op.__name__, ser, other, expected)
            tm.assert_series_equal(result, expected)
        else:
            exc = None
            try:
                result = op(ser, other)
            except Exception as err:
                exc = err
            if exc is None:
                expected = ser.combine(other, op)
                expected = self._cast_pointwise_result(op.__name__, ser, other, expected)
                tm.assert_series_equal(result, expected)
            else:
                with pytest.raises(type(exc)):
                    ser.combine(other, op)

    def test_compare_scalar(self, data: Any, comparison_op: OpFunc) -> None:
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, 0)

    def test_compare_array(self, data: Any, comparison_op: OpFunc) -> None:
        ser = pd.Series(data)
        other = pd.Series([data[0]] * len(data), dtype=data.dtype)
        self._compare_other(ser, data, comparison_op, other)

class BaseUnaryOpsTests(BaseOpsUtil):

    def test_invert(self, data: Any) -> None:
        ser = pd.Series(data, name='name')
        try:
            [~x for x in data[:10]]
        except TypeError:
            with pytest.raises(TypeError):
                ~ser
            with pytest.raises(TypeError):
                ~data
        else:
            result = ~ser
            expected = pd.Series(~data, name='name')
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('ufunc', [np.positive, np.negative, np.abs])
    def test_unary_ufunc_dunder_equivalence(self, data: Any, ufunc: Callable[[Any], Any]) -> None:
        attr = {np.positive: '__pos__', np.negative: '__neg__', np.abs: '__abs__'}[ufunc]
        exc = None
        try:
            result = getattr(data, attr)()
        except Exception as err:
            exc = err
            with pytest.raises((type(exc), TypeError)):
                ufunc(data)
        else:
            alt = ufunc(data)
            tm.assert_extension_array_equal(result, alt)
