import pandas as pd
import numpy as np
import operator

@pytest.fixture(autouse=True, params=[0, 1000000], ids=['numexpr', 'python'])
def switch_numexpr_min_elements(request, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(expr, '_MIN_ELEMENTS', request.param)
        yield

class TestSeriesFlexArithmetic:
    @pytest.mark.parametrize('opname', ['add', 'sub', 'mul', 'pow', 'truediv', 'floordiv'])
    def test_flex_method_equivalence(self, opname: str) -> None:
        # ...

    @pytest.mark.parametrize('target_add,input_value,expected_value', [('!', ['hello', 'world'], ['hello!', 'world!']), ('m', ['hello', 'world'], ['hellom', 'worldm'])])
    def test_string_addition(self, target_add: str, input_value: list, expected_value: list) -> None:
        # ...

    @pytest.mark.parametrize('axis', [0, None, 'index'])
    def test_ser_flex_cmp_return_dtypes(self, axis: int) -> None:
        # ...

    @pytest.mark.parametrize('opname', ['eq', 'ne', 'gt', 'lt', 'ge', 'le'])
    def test_ser_flex_cmp_return_dtypes_empty(self, opname: str) -> None:
        # ...

    @pytest.mark.parametrize('names', [(None, None, None), ('foo', 'bar', None), ('baz', 'baz', 'baz')])
    def test_ser_cmp_result_names(self, names: tuple) -> None:
        # ...

    @pytest.mark.parametrize('opname', ['eq', 'ne', 'gt', 'lt', 'ge', 'le'])
    def test_ser_flex_cmp_return_dtypes(self, opname: str) -> None:
        # ...

    def test_series_add_tz_mismatch_converts_to_utc(self) -> None:
        # ...

    @pytest.mark.parametrize('box', [list, tuple, np.array, Index, Series, pd.array])
    @pytest.mark.parametrize('flex', [True, False])
    def test_series_ops_name_retention(self, flex: bool, box: type, names: tuple, all_binary_operators: list) -> None:
        # ...

    def test_binop_maybe_preserve_name(self, datetime_series: Series) -> None:
        # ...

    def test_scalarop_preserve_name(self, datetime_series: Series) -> None:
        # ...
