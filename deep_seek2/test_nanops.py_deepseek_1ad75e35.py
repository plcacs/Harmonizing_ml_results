from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_integer_dtype

import pandas as pd
from pandas import Series, isna
import pandas._testing as tm
from pandas.core import nanops

use_bn: bool = nanops._USE_BOTTLENECK


@pytest.fixture
def disable_bottleneck(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setattr(nanops, "_USE_BOTTLENECK", False)
        yield


@pytest.fixture
def arr_shape() -> Tuple[int, int]:
    return 11, 7


@pytest.fixture
def arr_float(arr_shape: Tuple[int, int]) -> np.ndarray:
    return np.random.default_rng(2).standard_normal(arr_shape)


@pytest.fixture
def arr_complex(arr_float: np.ndarray) -> np.ndarray:
    return arr_float + arr_float * 1j


@pytest.fixture
def arr_int(arr_shape: Tuple[int, int]) -> np.ndarray:
    return np.random.default_rng(2).integers(-10, 10, arr_shape)


@pytest.fixture
def arr_bool(arr_shape: Tuple[int, int]) -> np.ndarray:
    return np.random.default_rng(2).integers(0, 2, arr_shape) == 0


@pytest.fixture
def arr_str(arr_float: np.ndarray) -> np.ndarray:
    return np.abs(arr_float).astype("S")


@pytest.fixture
def arr_utf(arr_float: np.ndarray) -> np.ndarray:
    return np.abs(arr_float).astype("U")


@pytest.fixture
def arr_date(arr_shape: Tuple[int, int]) -> np.ndarray:
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype("M8[ns]")


@pytest.fixture
def arr_tdelta(arr_shape: Tuple[int, int]) -> np.ndarray:
    return np.random.default_rng(2).integers(0, 20000, arr_shape).astype("m8[ns]")


@pytest.fixture
def arr_nan(arr_shape: Tuple[int, int]) -> np.ndarray:
    return np.tile(np.nan, arr_shape)


@pytest.fixture
def arr_float_nan(arr_float: np.ndarray, arr_nan: np.ndarray) -> np.ndarray:
    return np.vstack([arr_float, arr_nan])


@pytest.fixture
def arr_nan_float1(arr_nan: np.ndarray, arr_float: np.ndarray) -> np.ndarray:
    return np.vstack([arr_nan, arr_float])


@pytest.fixture
def arr_nan_nan(arr_nan: np.ndarray) -> np.ndarray:
    return np.vstack([arr_nan, arr_nan])


@pytest.fixture
def arr_inf(arr_float: np.ndarray) -> np.ndarray:
    return arr_float * np.inf


@pytest.fixture
def arr_float_inf(arr_float: np.ndarray, arr_inf: np.ndarray) -> np.ndarray:
    return np.vstack([arr_float, arr_inf])


@pytest.fixture
def arr_nan_inf(arr_nan: np.ndarray, arr_inf: np.ndarray) -> np.ndarray:
    return np.vstack([arr_nan, arr_inf])


@pytest.fixture
def arr_float_nan_inf(arr_float: np.ndarray, arr_nan: np.ndarray, arr_inf: np.ndarray) -> np.ndarray:
    return np.vstack([arr_float, arr_nan, arr_inf])


@pytest.fixture
def arr_nan_nan_inf(arr_nan: np.ndarray, arr_inf: np.ndarray) -> np.ndarray:
    return np.vstack([arr_nan, arr_nan, arr_inf])


@pytest.fixture
def arr_obj(
    arr_float: np.ndarray,
    arr_int: np.ndarray,
    arr_bool: np.ndarray,
    arr_complex: np.ndarray,
    arr_str: np.ndarray,
    arr_utf: np.ndarray,
    arr_date: np.ndarray,
    arr_tdelta: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        [
            arr_float.astype("O"),
            arr_int.astype("O"),
            arr_bool.astype("O"),
            arr_complex.astype("O"),
            arr_str.astype("O"),
            arr_utf.astype("O"),
            arr_date.astype("O"),
            arr_tdelta.astype("O"),
        ]
    )


@pytest.fixture
def arr_nan_nanj(arr_nan: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return arr_nan + arr_nan * 1j


@pytest.fixture
def arr_complex_nan(arr_complex: np.ndarray, arr_nan_nanj: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.vstack([arr_complex, arr_nan_nanj])


@pytest.fixture
def arr_nan_infj(arr_inf: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return arr_inf * 1j


@pytest.fixture
def arr_complex_nan_infj(arr_complex: np.ndarray, arr_nan_infj: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.vstack([arr_complex, arr_nan_infj])


@pytest.fixture
def arr_float_1d(arr_float: np.ndarray) -> np.ndarray:
    return arr_float[:, 0]


@pytest.fixture
def arr_nan_1d(arr_nan: np.ndarray) -> np.ndarray:
    return arr_nan[:, 0]


@pytest.fixture
def arr_float_nan_1d(arr_float_nan: np.ndarray) -> np.ndarray:
    return arr_float_nan[:, 0]


@pytest.fixture
def arr_float1_nan_1d(arr_float1_nan: np.ndarray) -> np.ndarray:
    return arr_float1_nan[:, 0]


@pytest.fixture
def arr_nan_float1_1d(arr_nan_float1: np.ndarray) -> np.ndarray:
    return arr_nan_float1[:, 0]


class TestnanopsDataFrame:
    def setup_method(self) -> None:
        nanops._USE_BOTTLENECK = False

        arr_shape: Tuple[int, int] = (11, 7)

        self.arr_float: np.ndarray = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_float1: np.ndarray = np.random.default_rng(2).standard_normal(arr_shape)
        self.arr_complex: np.ndarray = self.arr_float + self.arr_float1 * 1j
        self.arr_int: np.ndarray = np.random.default_rng(2).integers(-10, 10, arr_shape)
        self.arr_bool: np.ndarray = np.random.default_rng(2).integers(0, 2, arr_shape) == 0
        self.arr_str: np.ndarray = np.abs(self.arr_float).astype("S")
        self.arr_utf: np.ndarray = np.abs(self.arr_float).astype("U")
        self.arr_date: np.ndarray = (
            np.random.default_rng(2).integers(0, 20000, arr_shape).astype("M8[ns]")
        )
        self.arr_tdelta: np.ndarray = (
            np.random.default_rng(2).integers(0, 20000, arr_shape).astype("m8[ns]")
        )

        self.arr_nan: np.ndarray = np.tile(np.nan, arr_shape)
        self.arr_float_nan: np.ndarray = np.vstack([self.arr_float, self.arr_nan])
        self.arr_float1_nan: np.ndarray = np.vstack([self.arr_float1, self.arr_nan])
        self.arr_nan_float1: np.ndarray = np.vstack([self.arr_nan, self.arr_float1])
        self.arr_nan_nan: np.ndarray = np.vstack([self.arr_nan, self.arr_nan])

        self.arr_inf: np.ndarray = self.arr_float * np.inf
        self.arr_float_inf: np.ndarray = np.vstack([self.arr_float, self.arr_inf])

        self.arr_nan_inf: np.ndarray = np.vstack([self.arr_nan, self.arr_inf])
        self.arr_float_nan_inf: np.ndarray = np.vstack([self.arr_float, self.arr_nan, self.arr_inf])
        self.arr_nan_nan_inf: np.ndarray = np.vstack([self.arr_nan, self.arr_nan, self.arr_inf])
        self.arr_obj: np.ndarray = np.vstack(
            [
                self.arr_float.astype("O"),
                self.arr_int.astype("O"),
                self.arr_bool.astype("O"),
                self.arr_complex.astype("O"),
                self.arr_str.astype("O"),
                self.arr_utf.astype("O"),
                self.arr_date.astype("O"),
                self.arr_tdelta.astype("O"),
            ]
        )

        with np.errstate(invalid="ignore"):
            self.arr_nan_nanj: np.ndarray = self.arr_nan + self.arr_nan * 1j
            self.arr_complex_nan: np.ndarray = np.vstack([self.arr_complex, self.arr_nan_nanj])

            self.arr_nan_infj: np.ndarray = self.arr_inf * 1j
            self.arr_complex_nan_infj: np.ndarray = np.vstack([self.arr_complex, self.arr_nan_infj])

        self.arr_float_2d: np.ndarray = self.arr_float
        self.arr_float1_2d: np.ndarray = self.arr_float1

        self.arr_nan_2d: np.ndarray = self.arr_nan
        self.arr_float_nan_2d: np.ndarray = self.arr_float_nan
        self.arr_float1_nan_2d: np.ndarray = self.arr_float1_nan
        self.arr_nan_float1_2d: np.ndarray = self.arr_nan_float1

        self.arr_float_1d: np.ndarray = self.arr_float[:, 0]
        self.arr_float1_1d: np.ndarray = self.arr_float1[:, 0]

        self.arr_nan_1d: np.ndarray = self.arr_nan[:, 0]
        self.arr_float_nan_1d: np.ndarray = self.arr_float_nan[:, 0]
        self.arr_float1_nan_1d: np.ndarray = self.arr_float1_nan[:, 0]
        self.arr_nan_float1_1d: np.ndarray = self.arr_nan_float1[:, 0]

    def teardown_method(self) -> None:
        nanops._USE_BOTTLENECK = use_bn

    def check_results(
        self,
        targ: Union[np.ndarray, float, int],
        res: Union[np.ndarray, float, int],
        axis: Optional[int],
        check_dtype: bool = True,
    ) -> None:
        res = getattr(res, "asm8", res)

        if (
            axis != 0
            and hasattr(targ, "shape")
            and targ.ndim
            and targ.shape != res.shape
        ):
            res = np.split(res, [targ.shape[0]], axis=0)[0]

        try:
            tm.assert_almost_equal(targ, res, check_dtype=check_dtype)
        except AssertionError:
            # handle timedelta dtypes
            if hasattr(targ, "dtype") and targ.dtype == "m8[ns]":
                raise

            # There are sometimes rounding errors with
            # complex and object dtypes.
            # If it isn't one of those, re-raise the error.
            if not hasattr(res, "dtype") or res.dtype.kind not in ["c", "O"]:
                raise
            # convert object dtypes to something that can be split into
            # real and imaginary parts
            if res.dtype.kind == "O":
                if targ.dtype.kind != "O":
                    res = res.astype(targ.dtype)
                else:
                    cast_dtype = "c16" if hasattr(np, "complex128") else "f8"
                    res = res.astype(cast_dtype)
                    targ = targ.astype(cast_dtype)
            # there should never be a case where numpy returns an object
            # but nanops doesn't, so make that an exception
            elif targ.dtype.kind == "O":
                raise
            tm.assert_almost_equal(np.real(targ), np.real(res), check_dtype=check_dtype)
            tm.assert_almost_equal(np.imag(targ), np.imag(res), check_dtype=check_dtype)

    def check_fun_data(
        self,
        testfunc: Callable,
        targfunc: Callable,
        testar: str,
        testarval: np.ndarray,
        targarval: np.ndarray,
        skipna: bool,
        check_dtype: bool = True,
        empty_targfunc: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        for axis in list(range(targarval.ndim)) + [None]:
            targartempval = targarval if skipna else testarval
            if skipna and empty_targfunc and isna(targartempval).all():
                targ = empty_targfunc(targartempval, axis=axis, **kwargs)
            else:
                targ = targfunc(targartempval, axis=axis, **kwargs)

            if targartempval.dtype == object and (
                targfunc is np.any or targfunc is np.all
            ):
                # GH#12863 the numpy functions will retain e.g. floatiness
                if isinstance(targ, np.ndarray):
                    targ = targ.astype(bool)
                else:
                    targ = bool(targ)

            if testfunc.__name__ in ["nanargmax", "nanargmin"] and (
                testar.startswith("arr_nan")
                or (testar.endswith("nan") and (not skipna or axis == 1))
            ):
                with pytest.raises(ValueError, match="Encountered .* NA value"):
                    testfunc(testarval, axis=axis, skipna=skipna, **kwargs)
                return
            res = testfunc(testarval, axis=axis, skipna=skipna, **kwargs)

            if (
                isinstance(targ, np.complex128)
                and isinstance(res, float)
                and np.isnan(targ)
                and np.isnan(res)
            ):
                # GH#18463
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

        # Recurse on lower-dimension
        testarval2 = np.take(testarval, 0, axis=-1)
        targarval2 = np.take(targarval, 0, axis=-1)
        self.check_fun_data(
            testfunc,
            targfunc,
            testar,
            testarval2,
            targarval2,
            skipna=skipna,
            check_dtype=check_dtype,
            empty_targfunc=empty_targfunc,
            **kwargs,
        )

    def check_fun(
        self,
        testfunc: Callable,
        targfunc: Callable,
        testar: str,
        skipna: bool,
        empty_targfunc: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        targar = testar
        if testar.endswith("_nan") and hasattr(self, testar[:-4]):
            targar = testar[:-4]

        testarval = getattr(self, testar)
        targarval = getattr(self, targar)
        self.check_fun_data(
            testfunc,
            targfunc,
            testar,
            testarval,
            targarval,
            skipna=skipna,
            empty_targfunc=empty_targfunc,
            **kwargs,
        )

    def check_funs(
        self,
        testfunc: Callable,
        targfunc: Callable,
        skipna: bool,
        allow_complex: bool = True,
        allow_all_nan: bool = True,
        allow_date: bool = True,
        allow_tdelta: bool = True,
        allow_obj: bool = True,
        **kwargs: Any,
    ) -> None:
        self.check_fun(testfunc, targfunc, "arr_float", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_float_nan", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_int", skipna, **kwargs)
        self.check_fun(testfunc, targfunc, "arr_bool", skipna, **kwargs)
        objs = [
            self.arr_float.astype("O"),
            self.arr_int.astype("O"),
            self.arr_bool.astype("O"),
        ]

        if allow_all_nan:
            self.check_fun(testfunc, targfunc, "arr_nan", skipna, **kwargs)

        if allow_complex:
            self.check_fun(testfunc, targfunc, "arr_complex", skipna, **kwargs)
            self.check_fun(testfunc, targfunc, "arr_complex_nan", skipna, **kwargs)
            if allow_all_nan:
                self.check_fun(testfunc, targfunc, "arr_nan_nanj", skipna, **kwargs)
            objs += [self.arr_complex.astype("O")]

        if allow_date:
            targfunc(self.arr_date)
            self.check_fun(testfunc, targfunc, "arr_date", skipna, **kwargs)
            objs += [self.arr_date.astype("O")]

        if allow_tdelta:
            try:
                targfunc(self.arr_tdelta)
            except TypeError:
                pass
            else:
                self.check_fun(testfunc, targfunc, "arr_tdelta", skipna, **kwargs)
                objs += [self.arr_tdelta.astype("O")]

        if allow_obj:
            self.arr_obj = np.vstack(objs)
            # some nanops handle object dtypes better than their numpy
            # counterparts, so the numpy functions need to be given something
            # else
            if allow_obj == "convert":
                targfunc = partial(
                    self._badobj_wrap, func=targfunc, allow_complex=allow_complex
                )
            self.check_fun(testfunc, targfunc, "arr_obj", skipna, **kwargs)

    def _badobj_wrap(
        self, value: np.ndarray, func: Callable, allow_complex: bool = True, **kwargs: Any
    ) -> Any:
        if value.dtype.kind == "O":
            if allow_complex:
                value = value.astype("c16")
            else:
                value = value.astype("f8")
        return func(value, **kwargs)

    @pytest.mark.parametrize(
        "nan_op,np_op", [(nanops.nanany, np.any), (nanops.nanall, np.all)]
    )
    def test_nan_funcs(self, nan_op: Callable, np_op: Callable, skipna: bool) -> None:
        self.check_funs(nan_op, np_op, skipna, allow_all_nan=False, allow_date=False)

    def test_nansum(self, skipna: bool) -> None:
        self.check_funs(
            nanops.nansum,
            np.sum,
            skipna,
            allow_date=False,
            check_dtype=False,
            empty_targfunc=np.nansum,
        )

    def test_nanmean(self, skipna: bool) -> None:
        self.check_funs(
            nanops.nanmean, np.mean, skipna, allow_obj=False, allow_date=False
        )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nanmedian(self, skipna: bool) -> None:
        self.check_funs(
            nanops.nanmedian,
            np.median,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
        )

    @pytest.mark.parametrize("ddof", range(3))
    def test_nanvar(self, ddof: int, skipna: bool) -> None:
        self.check_funs(
            nanops.nanvar,
            np.var,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
            ddof=ddof,
        )

    @pytest.mark.parametrize("ddof", range(3))
    def test_nanstd(self, ddof: int, skipna: bool) -> None:
        self.check_funs(
            nanops.nanstd,
            np.std,
            skipna,
            allow_complex=False,
            allow_date=False,
            allow_obj="convert",
            ddof=ddof,
        )

    @pytest.mark.parametrize("ddof", range(3))
    def test_nansem(self, ddof: int, skipna: bool) -> None:
        sp_stats = pytest.importorskip("scipy.stats")

        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nansem,
                sp_stats.sem,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
                allow_obj="convert",
                ddof=ddof,
            )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize(
        "nan_op,np_op", [(nanops.nanmin, np.min), (nanops.nanmax, np.max)]
    )
    def test_nanops_with_warnings(self, nan_op: Callable, np_op: Callable, skipna: bool) -> None:
        self.check_funs(nan_op, np_op, skipna, allow_obj=False)

    def _argminmax_wrap(
        self, value: np.ndarray, axis: Optional[int] = None, func: Optional[Callable] = None
    ) -> Any:
        res = func(value, axis)
        nans = np.min(value, axis)
        nullnan = isna(nans)
        if res.ndim:
            res[nullnan] = -1
        elif (hasattr(nullnan, "all") and nullnan.all()) or (
            not hasattr(nullnan, "all") and nullnan
        ):
            res = -1
        return res

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nanargmax(self, skipna: bool) -> None:
        func = partial(self._argminmax_wrap, func=np.argmax)
        self.check_funs(nanops.nanargmax, func, skipna, allow_obj=False)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nanargmin(self, skipna: bool) -> None:
        func = partial(self._argminmax_wrap, func=np.argmin)
        self.check_funs(nanops.nanargmin, func, skipna, allow_obj=False)

    def _skew_kurt_wrap(
        self, values: np.ndarray, axis: Optional[int] = None, func: Optional[Callable] = None
    ) -> Any:
        if not isinstance(values.dtype.type, np.floating):
            values = values.astype("f8")
        result = func(values, axis=axis, bias=False)
        # fix for handling cases where all elements in an axis are the same
        if isinstance(result, np.ndarray):
            result[np.max(values, axis=axis) == np.min(values, axis=axis)] = 0
            return result
        elif np.max(values) == np.min(values):
            return 0.0
        return result

    def test_nanskew(self, skipna: bool) -> None:
        sp_stats = pytest.importorskip("scipy.stats")

        func = partial(self._skew_kurt_wrap, func=sp_stats.skew)
        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nanskew,
                func,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
            )

    def test_nankurt(self, skipna: bool) -> None:
        sp_stats = pytest.importorskip("scipy.stats")

        func1 = partial(sp_stats.kurtosis, fisher=True)
        func = partial(self._skew_kurt_wrap, func=func1)
        with np.errstate(invalid="ignore"):
            self.check_funs(
                nanops.nankurt,
                func,
                skipna,
                allow_complex=False,
                allow_date=False,
                allow_tdelta=False,
            )

    def test_nanprod(self, skipna: bool) -> None:
        self.check_funs(
            nanops.nanprod,
            np.prod,
            skipna,
            allow_date=False,
            allow_tdelta=False,
            empty_targfunc=np.nanprod,
        )

    def check_nancorr_nancov_2d(
        self, checkfun: Callable, targ0: float, targ1: float, **kwargs: Any
    ) -> None:
        res00 = checkfun(self.arr_float_2d, self.arr_float1_2d, **kwargs)
        res01 = checkfun(
            self.arr_float_2d,
            self.arr_float1_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ0, res00)
        tm.assert_almost_equal(targ0, res01)

        res10 = checkfun(self.arr_float_nan_2d, self.arr_float1_nan_2d, **kwargs)
        res11 = checkfun(
            self.arr_float_nan_2d,
            self.arr_float1_nan_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ1, res10)
        tm.assert_almost_equal(targ1, res11)

        targ2 = np.nan
        res20 = checkfun(self.arr_nan_2d, self.arr_float1_2d, **kwargs)
        res21 = checkfun(self.arr_float_2d, self.arr_nan_2d, **kwargs)
        res22 = checkfun(self.arr_nan_2d, self.arr_nan_2d, **kwargs)
        res23 = checkfun(self.arr_float_nan_2d, self.arr_nan_float1_2d, **kwargs)
        res24 = checkfun(
            self.arr_float_nan_2d,
            self.arr_nan_float1_2d,
            min_periods=len(self.arr_float_2d) - 1,
            **kwargs,
        )
        res25 = checkfun(
            self.arr_float_2d,
            self.arr_float1_2d,
            min_periods=len(self.arr_float_2d) + 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)

    def check_nancorr_nancov_1d(
        self, checkfun: Callable, targ0: float, targ1: float, **kwargs: Any
    ) -> None:
        res00 = checkfun(self.arr_float_1d, self.arr_float1_1d, **kwargs)
        res01 = checkfun(
            self.arr_float_1d,
            self.arr_float1_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ0, res00)
        tm.assert_almost_equal(targ0, res01)

        res10 = checkfun(self.arr_float_nan_1d, self.arr_float1_nan_1d, **kwargs)
        res11 = checkfun(
            self.arr_float_nan_1d,
            self.arr_float1_nan_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ1, res10)
        tm.assert_almost_equal(targ1, res11)

        targ2 = np.nan
        res20 = checkfun(self.arr_nan_1d, self.arr_float1_1d, **kwargs)
        res21 = checkfun(self.arr_float_1d, self.arr_nan_1d, **kwargs)
        res22 = checkfun(self.arr_nan_1d, self.arr_nan_1d, **kwargs)
        res23 = checkfun(self.arr_float_nan_1d, self.arr_nan_float1_1d, **kwargs)
        res24 = checkfun(
            self.arr_float_nan_1d,
            self.arr_nan_float1_1d,
            min_periods=len(self.arr_float_1d) - 1,
            **kwargs,
        )
        res25 = checkfun(
            self.arr_float_1d,
            self.arr_float1_1d,
            min_periods=len(self.arr_float_1d) + 1,
            **kwargs,
        )
        tm.assert_almost_equal(targ2, res20)
        tm.assert_almost_equal(targ2, res21)
        tm.assert_almost_equal(targ2, res22)
        tm.assert_almost_equal(targ2, res23)
        tm.assert_almost_equal(targ2, res24)
        tm.assert_almost_equal(targ2, res25)

    def test_nancorr(self) -> None:
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1)
        targ0 = np.corrcoef(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="pearson")

    def test_nancorr_pearson(self) -> None:
        targ0 = np.corrcoef(self.arr_float_2d, self.arr_float1_2d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0, 1]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method="pearson")
        targ0 = np.corrcoef(self.arr_float_1d, self.arr_float1_1d)[0, 1]
        targ1 = np.corrcoef(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0, 1]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="pearson")

    def test_nancorr_kendall(self) -> None:
        sp_stats = pytest.importorskip("scipy.stats")

        targ0 = sp_stats.kendalltau(self.arr_float_2d, self.arr_float1_2d)[0]
        targ1 = sp_stats.kendalltau(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        self.check_nancorr_nancov_2d(nanops.nancorr, targ0, targ1, method="kendall")
        targ0 = sp_stats.kendalltau(self.arr_float_1d, self.arr_float1_1d)[0]
        targ1 = sp_stats.kendalltau(self.arr_float_1d.flat, self.arr_float1_1d.flat)[0]
        self.check_nancorr_nancov_1d(nanops.nancorr, targ0, targ1, method="kendall")

    def test_nancorr_spearman(self) -> None:
        sp_stats = pytest.importorskip("scipy.stats")

        targ0 = sp_stats.spearmanr(self.arr_float_2d, self.arr_float1_2d)[0]
        targ1 = sp_stats.spearmanr(self.arr_float_2d.flat, self.arr_float1_2d.flat)[0]
        self.check_nancorr_nancov_2d(nanops.nancorr