from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    Tuple,
    Union,
)
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.indexes.datetimes import DatetimeIndex

P = ParamSpec("P")

@pytest.fixture
def dti() -> DatetimeIndex:
    ...

@pytest.fixture
def _test_series(dti: DatetimeIndex) -> Series:
    ...

@pytest.fixture
def test_frame(dti: DatetimeIndex, _test_series: Series) -> DataFrame:
    ...

def test_str(_test_series: Series) -> None:
    ...

def test_api(_test_series: Series) -> None:
    ...

def test_groupby_resample_api() -> DataFrame:
    ...

def test_groupby_resample_on_api() -> DataFrame:
    ...

def test_resample_group_keys() -> DataFrame:
    ...

def test_pipe(test_frame: DataFrame, _test_series: Series) -> None:
    ...

def test_getitem(test_frame: DataFrame) -> None:
    ...

@pytest.mark.parametrize('key', [['D'], ['A', 'D']])
def test_select_bad_cols(key: List[str], test_frame: DataFrame) -> Callable[[P], None]:
    ...

def test_attribute_access(test_frame: DataFrame) -> None:
    ...

def test_api_compat_before_use(attr: str, test_frame: DataFrame) -> None:
    ...

@pytest.mark.parametrize('key', [['D'], ['A', 'D']])
def tests_raises_on_nuisance(key: List[str], using_infer_string: bool) -> None:
    ...

def test_downsample_but_actually_upsampling() -> Series:
    ...

def test_combined_up_downsampling_of_irregular() -> Series:
    ...

def test_transform_series(_test_series: Series) -> Series:
    ...

@pytest.mark.parametrize('on', [None, 'date'])
def test_transform_frame(on: Optional[str]) -> DataFrame:
    ...

@pytest.mark.parametrize('func', [lambda x: x.resample('20min', group_keys=False), lambda x: x.groupby(pd.Grouper(freq='20min'), group_keys=False)], ids=['resample', 'groupby'])
def test_apply_without_aggregation(func: Callable[[Series], Any], _test_series: Series) -> Series:
    ...

def test_apply_without_aggregation2(_test_series: Series) -> Series:
    ...

def test_agg_consistency() -> None:
    ...

def test_agg_consistency_int_str_column_mix() -> None:
    ...