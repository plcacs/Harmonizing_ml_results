from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
from pandas import IntervalDtype, IntervalIndex
import pytest

class ConstructorTests:
    @pytest.mark.parametrize('breaks_and_expected_subtype', [
        (List[Union[List[int], np.ndarray, Index]], np.dtype),
        (np.ndarray, np.dtype),
        (Index, np.dtype),
        (Index, np.dtype),
        (Index, np.dtype),
        (List, np.dtype),
        (List, np.dtype),
        (List, np.dtype)
    ])
    @pytest.mark.parametrize('name', [Optional[str]])
    def test_constructor(self, constructor: Callable, breaks_and_expected_subtype: Tuple[Any, np.dtype], closed: str, name: Optional[str]) -> None: ...

    @pytest.mark.parametrize('breaks, subtype', [
        (Index, np.dtype),
        (Index, np.dtype),
        (Index, np.dtype),
        (Index, np.dtype),
        (List, np.dtype),
        (List, np.dtype)
    ])
    def test_constructor_dtype(self, constructor: Callable, breaks: Union[List, Index], subtype: np.dtype) -> None: ...

    @pytest.mark.parametrize('breaks', [Union[List, np.ndarray, Index]])
    def test_constructor_pass_closed(self, constructor: Callable, breaks: Union[List, np.ndarray, Index]) -> None: ...

    @pytest.mark.parametrize('breaks', [List[float]])
    def test_constructor_nan(self, constructor: Callable, breaks: List[float], closed: str) -> None: ...

    @pytest.mark.parametrize('breaks', [Union[List, np.ndarray]])
    def test_constructor_empty(self, constructor: Callable, breaks: Union[List, np.ndarray], closed: str) -> None: ...

    @pytest.mark.parametrize('breaks', [Union[Tuple, List, np.ndarray]])
    def test_constructor_string(self, constructor: Callable, breaks: Union[Tuple, List, np.ndarray]) -> None: ...

    @pytest.mark.parametrize('cat_constructor', [Callable])
    def test_constructor_categorical_valid(self, constructor: Callable, cat_constructor: Callable) -> None: ...

    def test_generic_errors(self, constructor: Callable) -> None: ...

class TestFromArrays(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Callable: ...

    def get_kwargs_from_breaks(self, breaks: Union[List, np.ndarray, Index], closed: str = 'right') -> dict: ...

    def test_constructor_errors(self) -> None: ...

    @pytest.mark.parametrize('left_subtype, right_subtype', [(np.dtype, np.dtype)])
    def test_mixed_float_int(self, left_subtype: np.dtype, right_subtype: np.dtype) -> None: ...

    @pytest.mark.parametrize('interval_cls', [Union[IntervalArray, IntervalIndex]])
    def test_from_arrays_mismatched_datetimelike_resos(self, interval_cls: Union[IntervalArray, IntervalIndex]) -> None: ...

class TestFromBreaks(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Callable: ...

    def get_kwargs_from_breaks(self, breaks: Union[List, np.ndarray, Index], closed: str = 'right') -> dict: ...

    def test_constructor_errors(self) -> None: ...

    def test_length_one(self) -> None: ...

    def test_left_right_dont_share_data(self) -> None: ...

class TestFromTuples(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Callable: ...

    def get_kwargs_from_breaks(self, breaks: Union[List, np.ndarray, Index], closed: str = 'right') -> dict: ...

    def test_constructor_errors(self) -> None: ...

    def test_na_tuples(self) -> None: ...

class TestClassConstructors(ConstructorTests):
    @pytest.fixture
    def constructor(self) -> Callable: ...

    def get_kwargs_from_breaks(self, breaks: Union[List, np.ndarray, Index], closed: str = 'right') -> dict: ...

    def test_generic_errors(self, constructor: Callable) -> None: ...

    def test_constructor_string(self) -> None: ...

    @pytest.mark.parametrize('klass', [Union[IntervalIndex, Callable]])
    def test_constructor_errors(self, klass: Union[IntervalIndex, Callable]) -> None: ...

    @pytest.mark.parametrize('data, closed', [(Union[List, IntervalIndex], str)])
    def test_override_inferred_closed(self, constructor: Callable, data: Union[List, IntervalIndex], closed: str) -> None: ...

    @pytest.mark.parametrize('values_constructor', [Callable])
    def test_index_object_dtype(self, values_constructor: Callable) -> None: ...

    def test_index_mixed_closed(self) -> None: ...

@pytest.mark.parametrize('timezone', [str])
def test_interval_index_subtype(timezone: str, inclusive_endpoints_fixture: str) -> None: ...

def test_dtype_closed_mismatch() -> None: ...

@pytest.mark.parametrize('dtype', [Union[str, pytest.param]])
def test_ea_dtype(dtype: Union[str, pytest.param]) -> None: ...