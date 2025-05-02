from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import Series, isna
import pandas._testing as tm
from pandas.core import nanops
from numpy.typing import NDArray

use_bn: bool = nanops._USE_BOTTLENECK

T = TypeVar('T')

@pytest.fixture
def disable_bottleneck(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(nanops, '_USE_BOTTLENECK', False)
        yield

@pytest.fixture
def arr_shape() -> Tuple[int, int]:
    return (11, 7)

@pytest.fixture
def arr_float(arr_shape: Tuple[int, int]) -> NDArray[np.float64]:
    return np.random.default_rng(2).standard_normal(arr_shape)

@pytest.fixture
def arr_complex(arr_float: NDArray[np.float64]) -> NDArray[np.complex128]:
    return arr_float + arr_float * 1j

@pytest.fixture
def arr_int(arr_shape: Tuple[int, int]) -> NDArray[np.int64]:
    return np.random.default_rng(2).integers(-10, 10, arr_shape)

@pytest.fixture
def arr_bool(arr_shape: Tuple[int, int]) -> NDArray[np.bool_]:
    return np.random.default_rng(2).integers(0, 2, arr_shape) == 0

@pytest.fixture
def arr_str(arr_float: NDArray[np.float64]) -> NDArray[np.bytes_]:
    return np.abs(arr_float).astype('S')

@pytest.fixture
def arr_utf(arr_float: NDArray[np.float64]) -> NDArray[np.str_]:
    return np.abs(arr_float).astype('U')

@pytest.fixture
def arr_date(arr_shape: Tuple[int, int]) -> NDArray[np.datetime64]:
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype('M8[ns]')

@pytest.fixture
def arr_tdelta(arr_shape: Tuple[int, int]) -> NDArray[np.timedelta64]:
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype('m8[ns]')

@pytest.fixture
def arr_nan(arr_shape: Tuple[int, int]) -> NDArray[np.float64]:
    return np.tile(np.nan, arr_shape)

@pytest.fixture
def arr_float_nan(arr_float: NDArray[np.float64], arr_nan: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.vstack([arr_float, arr_nan])

@pytest.fixture
def arr_nan_float1(arr_nan: NDArray[np.float64], arr_float: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.vstack([arr_nan, arr_float])

@pytest.fixture
def arr_nan_nan(arr_nan: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.vstack([arr_nan, arr_nan])

@pytest.fixture
def arr_inf(arr_float: NDArray[np.float64]) -> NDArray[np.float64]:
    return arr_float * np.inf

@pytest.fixture
def arr_float_inf(arr_float: NDArray[np.float64], arr_inf: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.vstack([arr_float, arr_inf])

@pytest.fixture
def arr_nan_inf(arr_nan: NDArray[np.float64], arr_inf: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.vstack([arr_nan, arr_inf])

@pytest.fixture
def arr_float_nan_inf(arr_float: NDArray[np.float64], arr_nan: NDArray[np.float64], arr_inf: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.vstack([arr_float, arr_nan, arr_inf])

@pytest.fixture
def arr_nan_nan_inf(arr_nan: NDArray[np.float64], arr_inf: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.vstack([arr_nan, arr_nan, arr_inf])

@pytest.fixture
def arr_obj(arr_float: NDArray[np.float64], arr_int: NDArray[np.int64], arr_bool: NDArray[np.bool_], arr_complex: NDArray[np.complex128], arr_str: NDArray[np.bytes_], arr_utf: NDArray[np.str_], arr_date: NDArray[np.datetime64], arr_tdelta: NDArray[np.timedelta64]) -> NDArray[np.object_]:
    return np.vstack([arr_float.astype('O'), arr_int.astype('O'), arr_bool.astype('O'), arr_complex.astype('O'), arr_str.astype('O'), arr_utf.astype('O'), arr_date.astype('O'), arr_tdelta.astype('O')])

@pytest.fixture
def arr_nan_nanj(arr_nan: NDArray[np.float64]) -> NDArray[np.complex128]:
    with np.errstate(invalid='ignore'):
        return arr_nan + arr_nan * 1j

@pytest.fixture
def arr_complex_nan(arr_complex: NDArray[np.complex128], arr_nan_nanj: NDArray[np.complex128]) -> NDArray[np.complex128]:
    with np.errstate(invalid='ignore'):
        return np.vstack([arr_complex, arr_nan_nanj])

@pytest.fixture
def arr_nan_infj(arr_inf: NDArray[np.float64]) -> NDArray[np.complex128]:
    with np.errstate(invalid='ignore'):
        return arr_inf * 1j

@pytest.fixture
def arr_complex_nan_infj(arr_complex: NDArray[np.complex128], arr_nan_infj: NDArray[np.complex128]) -> NDArray[np.complex128]:
    with np.errstate(invalid='ignore'):
        return np.vstack([arr_complex, arr_nan_infj])

@pytest.fixture
def arr_float_1d(arr_float: NDArray[np.float64]) -> NDArray[np.float64]:
    return arr_float[:, 0]

@pytest.fixture
def arr_nan_1d(arr_nan: NDArray[np.float64]) -> NDArray[np.float64]:
    return arr_nan[:, 0]

@pytest.fixture
def arr_float_nan_1d(arr_float_nan: NDArray[np.float64]) -> NDArray[np.float64]:
    return arr_float_nan[:, 0]

@pytest.fixture
def arr_float1_nan_1d(arr_float1_nan: NDArray[np.float64]) -> NDArray[np.float64]:
    return arr_float1_nan[:, 0]

@pytest.fixture
def arr_nan_float1_1d(arr_nan_float1: NDArray[np.float64]) -> NDArray[np.float64]:
    return arr_nan_float1[:, 0]

class TestnanopsDataFrame:
    def setup_method(self) -> None:
        nanops._USE_BOTTLENECK = False
        arr_shape = (11, 7)
        self.arr_float: NDArray[np.float64] = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_float1: NDArray[np.float64] = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_complex: NDArray[np.complex128] = self.arr_float + self.arr_float1 * 1j
        self.arr_int: NDArray[np.int64] = np.random.default_rng(2).integers(-10, 10, arr_shape)
        self.arr_bool: NDArray[np.bool_] = np.random.default_rng(2).integers(0, 2, arr_shape) == 0
        self.arr_str: NDArray[np.bytes_] = np.abs(self.arr_float).astype('S')
        self.arr_utf: NDArray[np.str_] = np.abs(self.arr_float).astype('U')
        self.arr_date: NDArray[np.datetime64] = np.random.default_rng(2).integers(0, 20000, arr_shape).astype('M8[ns]')
        self.arr_tdelta: NDArray[np.timedelta64] = np.random.default_rng(2).integers(0, 20000, arr_shape).astype('m8[ns]')
        self.arr_nan: NDArray[np.float64] = np.tile(np.nan, arr_shape)
        self.arr_float_nan: NDArray[np.float64] = np.vstack([self.arr_float, self.arr_nan])
        self.arr_float1_nan: NDArray[np.float64] = np.vstack([self.arr_float1, self.arr_nan])
        self.arr_nan_float1: NDArray[np.float64] = np.vstack([self.arr_nan, self.arr_float1])
        self.arr_nan_nan: NDArray[np.float64] = np.vstack([self.arr_nan, self.arr_nan])
        self.arr_inf: NDArray[np.float64] = self.arr_float * np.inf
        self.arr_float_inf: NDArray[np.float64] = np.vstack([self.arr_float, self.arr_inf])
        self.arr_nan_inf: NDArray[np.float64] = np.vstack([self.arr_nan, self.arr_inf])
        self.arr_float_nan_inf: NDArray[np.float64] = np.vstack([self.arr_float, self.arr_nan, self.arr_inf])
        self.arr_nan_nan_inf: NDArray[np.float64] = np.vstack([self.arr_nan, self.arr_nan, self.arr_inf])
        self.arr_obj: NDArray[np.object_] = np.vstack([self.arr_float.astype('O'), self.arr_int.astype('O'), self.arr_bool.astype('O'), self.arr_complex.astype('O'), self.arr_str.astype('O'), self.arr_utf.astype('O'), self.arr_date.astype('O'), self.arr_tdelta.astype('O')])
        with np.errstate(invalid='ignore'):
            self.arr_nan_nanj: NDArray[np.complex128] = self.arr_nan + self.arr_nan * 1j
            self.arr_complex_nan: NDArray[np.complex128] = np.vstack([self.arr_complex, self.arr_nan_nanj])
            self.arr_nan_infj: NDArray[np.complex128] = self.arr_inf * 1j
            self.arr_complex_nan_infj: NDArray[np.complex128] = np.vstack([self.arr_complex, self.arr_nan_infj])
        self.arr_float_2d: NDArray[np.float64] = self.arr_float
        self.arr_float1_2d: NDArray[np.float64] = self.arr_float1
        self.arr_nan_2d: NDArray[np.float64] = self.arr_nan
        self.arr_float_nan_2d: NDArray[np.float64] = self.arr_float_nan
        self.arr_float1_nan_2d: NDArray[np.float64] = self.arr_float1_nan
        self.arr_nan_float1_2d: NDArray[np.float64] = self.arr_nan_float1
        self.arr_float_1d: NDArray[np.float64] = self.arr_float[:, 0]
        self.arr_float1_1d: NDArray[np.float64] = self.arr_float1[:, 0]
        self.arr_nan_1d: NDArray[np.float64] = self.arr_nan[:, 0]
        self.arr_float_nan_1d: NDArray[np.float64] = self.arr_float_nan[:, 0]
        self.arr_float1_nan_1d: NDArray[np.float64] = self.arr_float1_nan[:, 0]
        self.arr_nan_float1_1d: NDArray[np.float64] = self.arr_nan_float1[:, 0]

    def teardown_method(self) -> None:
        nanops._USE_BOTTLENECK = use_bn

    def check_results(self, targ: Any, res: Any, axis: Optional[int], check_dtype: bool = True) -> None:
        res = getattr(res, 'asm8', res)
        if axis != 0 and hasattr(targ, 'shape') and targ.ndim and (targ.shape != res.shape):
            res = np.split(res, [targ.shape[0]], axis=0)[0]
        try:
            tm.assert_almost_equal(targ, res, check_dtype=check_dtype)
        except AssertionError:
            if hasattr(targ, 'dtype') and targ.dtype == 'm8[ns]':
                raise
            if not hasattr(res, 'dtype') or res.dtype.kind not in ['c', 'O']:
                raise
            if res.dtype.kind == 'O':
                if targ.dtype.kind != 'O':
                    res = res.astype(targ.dtype)
                else:
                    cast_dtype = 'c16' if hasattr(np, 'complex128') else 'f8'
                    res = res.astype(cast_dtype)
                    targ = targ.astype(cast_dtype)
            elif targ.dtype.kind == 'O':
                raise
            tm.assert_almost_equal(np.real(targ), np.real(res), check_dtype=check_dtype)
            tm.assert_almost_equal(np.imag(targ), np.imag(res), check_dtype=check_dtype)

    def check_fun_data(self, testfunc: Callable[..., Any], targfunc: Callable[..., Any], testar: str, testarval: NDArray[Any], targarval: NDArray[Any], skipna: bool, check_dtype: bool = True, empty_targfunc: Optional[Callable[..., Any]] = None, **kwargs: Any) -> None:
        for axis in list(range(targarval.ndim)) + [None]:
            targartempval = targarval if skipna else testarval
            if skipna and empty_targfunc and isna(targartempval).all():
                targ = empty_targfunc(targartempval, axis=axis, **kwargs)
            else:
                targ = targfunc(targartempval, axis=axis, **kwargs)
            if targartempval.dtype == object and (targfunc is np.any or targfunc is np.all):
                if isinstance(targ, np.ndarray):
                    targ = targ.astype(bool)
                else:
                    targ = bool(targ)
            if testfunc.__name__ in ['nanargmax', 'nanargmin'] and (testar.startswith('arr_nan') or (testar.endswith('nan') and (not skipna or axis == 1))):
                with pytest.raises(ValueError, match='Encountered .* NA value'):
                    testfunc(testarval, axis=axis, skipna=skipna, **kwargs)
                return
            res = testfunc(testarval, axis=axis, skipna=skipna, **kwargs)
            if isinstance(targ, np.complex128) and isinstance(res, float) and np.isnan(targ) and np.isnan(res):
                targ = res
            self.check_results(targ, res, axis, check_dtype=check_dtype)
            if skipna:
                res = testfunc(testarval, axis=axis, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
            if axis is None:
                res = testfunc(testarval, skipna=skipna, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
            if skipna and axis is None:
                res = testfunc(testarval, **kwargs)
                self.check_results(targ, res, axis, check_dtype=check_dtype)
        if testarval.ndim <= 1:
            return
        testarval2 = np.take(testarval, 0, axis=-1)
        targarval2 = np.take(targarval, 0, axis=-1)
        self.check_fun_data(testfunc, targfunc, testar, testarval2, targarval2, skipna=skipna, check_dtype=check_dtype, empty_targfunc=empty_targfunc, **kwargs)

    def check_fun(self, testfunc: Callable[..., Any], targfunc: Callable[..., Any], testar: str, skipna: bool, empty_targfunc: Optional[Callable[..., Any]] = None, **kwargs: Any) -> None:
        targar = testar
        if testar.endswith('_nan') and hasattr(self, testar[:-4]):
            targar = testar[:-4]
        testarval = getattr(self,