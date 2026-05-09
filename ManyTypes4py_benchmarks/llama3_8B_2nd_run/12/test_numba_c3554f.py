import numpy as np
import pytest
from pandas.compat import is_platform_arm
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import DataFrame, Series, option_context, to_datetime
import pandas._testing as tm
from pandas.util.version import Version

@pytest.fixture(params=['single', 'table'])
def method(request) -> str:
    """method keyword in rolling/expanding/ewm constructor"""
    return request.param

@pytest.fixture(params=[['sum', {}], ['mean', {}], ['median', {}], ['max', {}], ['min', {}], ['var', {}], ['var', {'ddof': 0}], ['std', {}], ['std', {'ddof': 0}]])
def arithmetic_numba_supported_operators(request) -> tuple[str, dict]:
    return request.param

@pytest.fixture
def roll_frame() -> DataFrame:
    return DataFrame({'A': [1] * 20 + [2] * 12 + [3] * 8, 'B': np.arange(40)})

@td.skip_if_no('numba')
class TestEngine:
    ...

@td.skip_if_no('numba')
class TestEWM:
    ...

@td.skip_if_no('numba')
def test_use_global_config() -> None:
    ...

@td.skip_if_no('numba')
def test_invalid_kwargs_nopython() -> None:
    ...

@td.skip_if_no('numba')
@pytest.mark.slow
@pytest.mark.filterwarnings('ignore')
class TestTableMethod:
    ...

def test_npfunc_no_warnings() -> None:
    df = DataFrame({'col1': [1, 2, 3, 4, 5]})
    with tm.assert_produces_warning(False):
        df.col1.rolling(2).apply(np.prod, raw=True, engine='numba')
