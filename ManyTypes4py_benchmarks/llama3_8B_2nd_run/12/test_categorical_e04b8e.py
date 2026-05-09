from pandas._config import using_string_dtype
import numpy as np
import pytest
from pandas import Categorical
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tests.extension import base

def make_data() -> np.ndarray:
    while True:
        values = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)
        if values[0] != values[1]:
            break
    return values

@pytest.fixture
def dtype() -> CategoricalDtype:
    return CategoricalDtype()

@pytest.fixture
def data() -> Categorical:
    """Length-100 array for this type.

    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    return Categorical(make_data())

@pytest.fixture
def data_missing() -> Categorical:
    """Length 2 array with [NA, Valid]"""
    return Categorical([np.nan, 'A'])

@pytest.fixture
def data_for_sorting() -> Categorical:
    return Categorical(['A', 'B', 'C'], categories=['C', 'A', 'B'], ordered=True)

@pytest.fixture
def data_missing_for_sorting() -> Categorical:
    return Categorical(['A', None, 'B'], categories=['B', 'A'], ordered=True)

@pytest.fixture
def data_for_grouping() -> Categorical:
    return Categorical(['a', 'a', None, None, 'b', 'b', 'a', 'c'])

class TestCategorical(base.ExtensionTests):
    def test_contains(self, data: Categorical, data_missing: Categorical) -> None:
        # ...

    def test_empty(self, dtype: CategoricalDtype) -> None:
        # ...

    # ...

class Test2DCompat(base.NDArrayBacked2DTests):
    def test_repr_2d(self, data: Categorical) -> None:
        # ...
