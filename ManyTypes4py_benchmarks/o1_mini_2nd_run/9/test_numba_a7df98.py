import pytest
from pandas.compat import is_platform_arm
from pandas import DataFrame, Series, option_context
import pandas._testing as tm
from pandas.util.version import Version
from typing import Any, Dict, Tuple

pytestmark = [pytest.mark.single_cpu]
numba = pytest.importorskip('numba')
pytestmark.append(
    pytest.mark.skipif(
        Version(numba.__version__) == Version('0.61') and is_platform_arm(),
        reason=f'Segfaults on ARM platforms with numba {numba.__version__}'
    )
)

@pytest.mark.filterwarnings('ignore')
class TestEngine:

    def test_cython_vs_numba_frame(
        self,
        sort: bool,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        numba_supported_reductions: Tuple[str, Dict[str, Any]]
    ) -> None:
        func, kwargs = numba_supported_reductions
        df: DataFrame = DataFrame({'a': [3, 2, 3, 2], 'b': range(4), 'c': range(1, 5)})
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        gb = df.groupby('a', sort=sort)
        result = getattr(gb, func)(engine='numba', engine_kwargs=engine_kwargs, **kwargs)
        expected = getattr(gb, func)(**kwargs)
        tm.assert_frame_equal(result, expected)

    def test_cython_vs_numba_getitem(
        self,
        sort: bool,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        numba_supported_reductions: Tuple[str, Dict[str, Any]]
    ) -> None:
        func, kwargs = numba_supported_reductions
        df: DataFrame = DataFrame({'a': [3, 2, 3, 2], 'b': range(4), 'c': range(1, 5)})
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        gb = df.groupby('a', sort=sort)['c']
        result = getattr(gb, func)(engine='numba', engine_kwargs=engine_kwargs, **kwargs)
        expected = getattr(gb, func)(**kwargs)
        tm.assert_series_equal(result, expected)

    def test_cython_vs_numba_series(
        self,
        sort: bool,
        nogil: bool,
        parallel: bool,
        nopython: bool,
        numba_supported_reductions: Tuple[str, Dict[str, Any]]
    ) -> None:
        func, kwargs = numba_supported_reductions
        ser: Series = Series(range(3), index=[1, 2, 1], name='foo')
        engine_kwargs: Dict[str, bool] = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
        gb = ser.groupby(level=0, sort=sort)
        result = getattr(gb, func)(engine='numba', engine_kwargs=engine_kwargs, **kwargs)
        expected = getattr(gb, func)(**kwargs)
        tm.assert_series_equal(result, expected)

    def test_as_index_false_unsupported(
        self, 
        numba_supported_reductions: Tuple[str, Dict[str, Any]]
    ) -> None:
        func, kwargs = numba_supported_reductions
        df: DataFrame = DataFrame({'a': [3, 2, 3, 2], 'b': range(4), 'c': range(1, 5)})
        gb = df.groupby('a', as_index=False)
        with pytest.raises(NotImplementedError, match='as_index=False'):
            getattr(gb, func)(engine='numba', **kwargs)

    def test_no_engine_doesnt_raise(self) -> None:
        df: DataFrame = DataFrame({'a': [3, 2, 3, 2], 'b': range(4), 'c': range(1, 5)})
        gb = df.groupby('a')
        with option_context('compute.use_numba', True):
            res: DataFrame = gb.agg({'b': 'first'})
        expected: DataFrame = gb.agg({'b': 'first'})
        tm.assert_frame_equal(res, expected)
