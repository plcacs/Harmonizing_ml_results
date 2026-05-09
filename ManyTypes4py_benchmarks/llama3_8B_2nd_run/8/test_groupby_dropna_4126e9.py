import numpy as np
import pytest
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args

@pytest.mark.parametrize('dropna, tuples, outputs', [(True, [['A', 'B'], ['B', 'A']], {'c': [13.0, 123.23], 'd': [12.0, 123.0], 'e': [1.0, 1.0]}), (False, [['A', 'B'], ['A', np.nan], ['B', 'A']], {'c': [13.0, 12.3, 123.23], 'd': [12.0, 233.0, 123.0], 'e': [1.0, 12.0, 1.0]})], 
                             type: List[Tuple[str, str]], 
                             type: Dict[str, float])
def test_groupby_dropna_multi_index_dataframe_nan_in_one_group(dropna: bool, tuples: List[Tuple[str, str]], outputs: Dict[str, float], nulls_fixture: str):
    # ... code ...

@pytest.mark.parametrize('dropna, idx, outputs', [(True, ['A', 'B'], {'b': [123.23, 13.0], 'c': [123.0, 13.0], 'd': [1.0, 13.0]}), (False, ['A', 'B', np.nan], {'b': [123.23, 13.0, 12.3], 'c': [123.0, 13.0, 233.0], 'd': [1.0, 13.0, 12.0]})], 
                             type: List[str], 
                             type: Dict[str, float])
def test_groupby_dropna_normal_index_dataframe(dropna: bool, idx: List[str], outputs: Dict[str, float]):
    # ... code ...

@pytest.mark.parametrize('dropna, datetime1, datetime2', [(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-01')), (pd.Timedelta('-2 days'), pd.Timedelta('-1 days')), (pd.Period('2020-01-01'), pd.Period('2020-02-01'))], 
                             type: pd.Timestamp, 
                             type: pd.Timestamp)
def test_groupby_dropna_datetime_like_data(dropna: bool, datetime1: pd.Timestamp, datetime2: pd.Timestamp, unique_nulls_fixture: str, unique_nulls_fixture2: str):
    # ... code ...

@pytest.mark.parametrize('sequence_index', range(3 ** 4), type=int)
@pytest.mark.parametrize('dtype', [None, 'UInt8', 'Int8', 'UInt16', 'Int16', 'UInt32', 'Int32', 'UInt64', 'Int64', 'Float32', 'Float64', 'category', 'string', pytest.param('string[pyarrow]', marks=pytest.mark.skipif(pa_version_under10p1, reason='pyarrow is not installed')), 'datetime64[ns]', 'period[D]', 'Sparse[float]'], type=Union[None, str])
@pytest.mark.parametrize('test_series', [True, False], type=bool)
def test_no_sort_keep_na(sequence_index: int, dtype: Union[None, str], test_series: bool, as_index: bool):
    # ... code ...

@pytest.mark.parametrize('index_kind', ['range', 'single', 'multi'], type=str)
def test_categorical_reducers(reduction_func: str, observed: bool, sort: bool, as_index: bool, index_kind: str):
    # ... code ...

@pytest.mark.parametrize('transformation_func', ['cumcount', 'ngroup'], type=str)
def test_categorical_transformers(transformation_func: str, observed: bool, sort: bool, as_index: bool):
    # ... code ...

@pytest.mark.parametrize('method', ['head', 'tail'], type=str)
def test_categorical_head_tail(method: str, observed: bool, sort: bool, as_index: bool):
    # ... code ...

def test_categorical_agg():
    # ... code ...

def test_categorical_transform():
    # ... code ...
