from collections.abc import Sequence
from typing import Any, Callable, ClassVar, Literal, Type, TypeVar, Union
from collections import deque
import numpy as np
import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray
from pandas.core.arrays.sparse import SparseDtype
import pytest

_T = TypeVar("_T")

@pytest.fixture
def ufunc(request: pytest.FixtureRequest) -> Callable[..., Any]:
    ...

@pytest.fixture
def sparse(request: pytest.FixtureRequest) -> bool:
    ...

@pytest.fixture
def arrays_for_binary_ufunc() -> tuple[np.ndarray, np.ndarray]:
    ...

@pytest.mark.parametrize("ufunc", [np.positive, np.floor, np.exp])
def test_unary_ufunc(
    ufunc: Callable[..., Any], sparse: bool
) -> None:
    ...

@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_array(
    flip: bool,
    sparse: bool,
    ufunc: Callable[..., Any],
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None:
    ...

@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_index(
    flip: bool,
    sparse: bool,
    ufunc: Callable[..., Any],
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None:
    ...

@pytest.mark.parametrize("shuffle", [True, False], ids=["unaligned", "aligned"])
@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_series(
    flip: bool,
    shuffle: bool,
    sparse: bool,
    ufunc: Callable[..., Any],
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None:
    ...

@pytest.mark.parametrize("flip", [True, False])
def test_binary_ufunc_scalar(
    ufunc: Callable[..., Any],
    sparse: bool,
    flip: bool,
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None:
    ...

@pytest.mark.parametrize("ufunc", [np.divmod])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_multiple_output_binary_ufuncs(
    ufunc: Callable[..., Any],
    sparse: bool,
    shuffle: bool,
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None:
    ...

def test_multiple_output_ufunc(
    sparse: bool,
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None:
    ...

def test_binary_ufunc_drops_series_name(
    ufunc: Callable[..., Any],
    sparse: bool,
    arrays_for_binary_ufunc: tuple[np.ndarray, np.ndarray],
) -> None:
    ...

def test_object_series_ok() -> None:
    ...

@pytest.fixture
def values_for_np_reduce(request: pytest.FixtureRequest) -> Any:
    ...

class TestNumpyReductions:
    def test_multiply(
        self,
        values_for_np_reduce: Any,
        box_with_array: Any,
        request: pytest.FixtureRequest,
    ) -> None:
        ...

    def test_add(
        self,
        values_for_np_reduce: Any,
        box_with_array: Any,
    ) -> None:
        ...

    def test_max(
        self,
        values_for_np_reduce: Any,
        box_with_array: Any,
    ) -> None:
        ...

    def test_min(
        self,
        values_for_np_reduce: Any,
        box_with_array: Any,
    ) -> None:
        ...

@pytest.mark.parametrize("type_", [list, deque, tuple])
def test_binary_ufunc_other_types(type_: Type[Sequence[int]]) -> None:
    ...

def test_object_dtype_ok() -> None:
    ...

def test_outer() -> None:
    ...

def test_np_matmul() -> None:
    ...

@pytest.mark.parametrize("box", [pd.Index, pd.Series])
def test_np_matmul_1D(box: Type[Union[pd.Index, pd.Series]]) -> None:
    ...

def test_array_ufuncs_for_many_arguments() -> None:
    ...

@pytest.mark.xfail(reason="see https://github.com/pandas-dev/pandas/pull/51082")
def test_np_fix() -> None:
    ...