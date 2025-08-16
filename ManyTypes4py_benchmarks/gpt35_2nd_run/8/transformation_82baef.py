from typing import Any, Callable, Dict, List, Union, Optional
import numpy as np
import pandas as pd
from numpy import nan
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions import empirical_distribution as ed
from toolz import curry, merge, compose, mapcat
from fklearn.common_docstrings import learner_return_docstring, learner_pred_fn_docstring
from fklearn.training.utils import log_learner_time
from fklearn.types import LearnerReturnType, LearnerLogType
from fklearn.preprocessing.schema import column_duplicatable

@curry
@log_learner_time(learner_name='selector')
def selector(df: pd.DataFrame, training_columns: List[str], predict_columns: Optional[List[str]] = None) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@column_duplicatable('columns_to_cap')
@curry
@log_learner_time(learner_name='capper')
def capper(df: pd.DataFrame, columns_to_cap: List[str], precomputed_caps: Optional[Dict[str, Any]] = None) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@column_duplicatable('columns_to_floor')
@curry
@log_learner_time(learner_name='floorer')
def floorer(df: pd.DataFrame, columns_to_floor: List[str], precomputed_floors: Optional[Dict[str, Any]] = None) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@curry
@log_learner_time(learner_name='ecdfer')
def ecdfer(df: pd.DataFrame, ascending: bool = True, prediction_column: str = 'prediction', ecdf_column: str = 'prediction_ecdf', max_range: int = 1000) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@curry
@log_learner_time(learner_name='discrete_ecdfer')
def discrete_ecdfer(df: pd.DataFrame, ascending: bool = True, prediction_column: str = 'prediction', ecdf_column: str = 'prediction_ecdf', max_range: int = 1000, round_method: Callable) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@curry
def prediction_ranger(df: pd.DataFrame, prediction_min: float, prediction_max: float, prediction_column: str = 'prediction') -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

def apply_replacements(df: pd.DataFrame, columns: List[str], vec: Dict[str, Dict[Any, Any]], replace_unseen: Any) -> pd.DataFrame:
    ...

@column_duplicatable('value_maps')
@curry
@log_learner_time(learner_name='value_mapper')
def value_mapper(df: pd.DataFrame, value_maps: Dict[str, Dict[Any, Any]], ignore_unseen: bool = True, replace_unseen_to: Any = np.nan) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@column_duplicatable('columns_to_truncate')
@curry
@log_learner_time(learner_name='truncate_categorical')
def truncate_categorical(df: pd.DataFrame, columns_to_truncate: List[str], percentile: float, replacement: Union[int, str, float, nan] = -9999, replace_unseen: Union[int, str, float, nan] = -9999, store_mapping: bool = False) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@column_duplicatable('columns_to_rank')
@curry
@log_learner_time(learner_name='rank_categorical')
def rank_categorical(df: pd.DataFrame, columns_to_rank: List[str], replace_unseen: Union[int, str, float, nan] = nan, store_mapping: bool = False) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='count_categorizer')
def count_categorizer(df: pd.DataFrame, columns_to_categorize: List[str], replace_unseen: int = -1, store_mapping: bool = False) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='label_categorizer')
def label_categorizer(df: pd.DataFrame, columns_to_categorize: List[str], replace_unseen: Union[int, str, float, nan] = nan, store_mapping: bool = False) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@column_duplicatable('columns_to_bin')
@curry
@log_learner_time(learner_name='quantile_biner')
def quantile_biner(df: pd.DataFrame, columns_to_bin: List[str], q: int = 4, right: bool = False) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='onehot_categorizer')
def onehot_categorizer(df: pd.DataFrame, columns_to_categorize: List[str], hardcode_nans: bool = False, drop_first_column: bool = False, store_mapping: bool = False) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='target_categorizer')
def target_categorizer(df: pd.DataFrame, columns_to_categorize: List[str], target_column: str, smoothing: float = 1.0, ignore_unseen: bool = True, store_mapping: bool = False) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@column_duplicatable('columns_to_scale')
@curry
@log_learner_time(learner_name='standard_scaler')
def standard_scaler(df: pd.DataFrame, columns_to_scale: List[str]) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@column_duplicatable('columns_to_transform')
@curry
@log_learner_time(learner_name='custom_transformer')
def custom_transformer(df: pd.DataFrame, columns_to_transform: List[str], transformation_function: Callable, is_vectorized: bool = False) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@curry
@log_learner_time(learner_name='null_injector')
def null_injector(df: pd.DataFrame, proportion: float, columns_to_inject: Optional[List[str]] = None, groups: Optional[List[List[str]]] = None, seed: int = 1) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...

@curry
@log_learner_time(learner_name='missing_warner')
def missing_warner(df: pd.DataFrame, cols_list: List[str], new_column_name: str = 'has_unexpected_missing', detailed_warning: bool = False, detailed_column_name: Optional[str] = None) -> Tuple[Callable, pd.DataFrame, Dict[str, Any]]:
    ...
