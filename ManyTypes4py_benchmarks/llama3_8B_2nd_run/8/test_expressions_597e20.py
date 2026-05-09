import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import DataFrame
from pandas.core.computation import expressions as expr

@pytest.fixture
def _frame() -> DataFrame:
    return DataFrame(np.random.default_rng(2).standard_normal((10001, 4)), columns=list('ABCD'), dtype='float64')

@pytest.fixture
def _frame2() -> DataFrame:
    return DataFrame(np.random.default_rng(2).standard_normal((100, 4)), columns=list('ABCD'), dtype='float64')

@pytest.fixture
def _mixed(_frame: DataFrame) -> DataFrame:
    return DataFrame({'A': _frame['A'], 'B': _frame['B'].astype('float32'), 'C': _frame['C'].astype('int64'), 'D': _frame['D'].astype('int32')})

@pytest.fixture
def _mixed2(_frame2: DataFrame) -> DataFrame:
    return DataFrame({'A': _frame2['A'], 'B': _frame2['B'].astype('float32'), 'C': _frame2['C'].astype('int64'), 'D': _frame2['D'].astype('int32')})

@pytest.fixture
def _integer() -> DataFrame:
    return DataFrame(np.random.default_rng(2).integers(1, 100, size=(10001, 4)), columns=list('ABCD'), dtype='int64')

@pytest.fixture
def _integer_integers(_integer: DataFrame) -> DataFrame:
    return _integer * np.random.default_rng(2).integers(0, 2, size=np.shape(_integer))

@pytest.fixture
def _integer2() -> DataFrame:
    return DataFrame(np.random.default_rng(2).integers(1, 100, size=(101, 4)), columns=list('ABCD'), dtype='int64')

@pytest.fixture
def _array(_frame: DataFrame) -> np.ndarray:
    return _frame['A'].to_numpy()

@pytest.fixture
def _array2(_frame2: DataFrame) -> np.ndarray:
    return _frame2['A'].to_numpy()

@pytest.fixture
def _array_mixed(_mixed: DataFrame) -> np.ndarray:
    return _mixed['D'].to_numpy()

@pytest.fixture
def _array_mixed2(_mixed2: DataFrame) -> np.ndarray:
    return _mixed2['D'].to_numpy()

@pytest.mark.skipif(not expr.USE_NUMEXPR, reason='not using numexpr')
class TestExpressions:
    @staticmethod
    def call_op(df: DataFrame, other, flex: bool, opname: str) -> tuple:
        if flex:
            op = lambda x, y: getattr(x, opname)(y)
            op.__name__ = opname
        else:
            op = getattr(operator, opname)
        with option_context('compute.use_numexpr', False):
            expected = op(df, other)
        expr.get_test_result()
        result = op(df, other)
        return (result, expected)

    @pytest.mark.parametrize('fixture', ['_integer', '_integer2', '_integer_integers', '_frame', '_frame2', '_mixed', '_mixed2'])
    @pytest.mark.parametrize('flex', [True, False])
    @pytest.mark.parametrize('arith', ['add', 'sub', 'mul', 'mod', 'truediv', 'floordiv'])
    def test_run_arithmetic(self, request, fixture: str, flex: bool, arith: str, monkeypatch: pytest.MonkeyPatch):
        df = request.getfixturevalue(fixture)
        with monkeypatch.context() as m:
            m.setattr(expr, '_MIN_ELEMENTS', 0)
            result, expected = self.call_op(df, df, flex, arith)
            if arith == 'truediv':
                assert all((x.kind == 'f' for x in expected.dtypes.values))
            tm.assert_equal(expected, result)
            for i in range(len(df.columns)):
                result, expected = self.call_op(df.iloc[:, i], df.iloc[:, i], flex, arith)
                if arith == 'truediv':
                    assert expected.dtype.kind == 'f'
                tm.assert_equal(expected, result)

    # ... and so on
