from typing import Any, Callable, Dict, List, Union, Optional, Tuple, TypeVar, overload
import numpy as np
import pandas as pd
from numpy import nan, ndarray
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions import empirical_distribution as ed
from toolz import curry, merge, compose, mapcat
from fklearn.common_docstrings import learner_return_docstring, learner_pred_fn_docstring
from fklearn.training.utils import log_learner_time
from fklearn.types import LearnerReturnType, LearnerLogType
from fklearn.preprocessing.schema import column_duplicatable
from pandas import DataFrame, Series
from statsmodels.distributions.empirical_distribution import ECDF

@curry
@log_learner_time(learner_name='selector')
def selector(df: DataFrame, training_columns: List[str], predict_columns: Optional[List[str]] = ...) -> LearnerReturnType: ...

@column_duplicatable('columns_to_cap')
@curry
@log_learner_time(learner_name='capper')
def capper(df: DataFrame, columns_to_cap: List[str], precomputed_caps: Optional[Dict[str, float]] = ...) -> LearnerReturnType: ...

@column_duplicatable('columns_to_floor')
@curry
@log_learner_time(learner_name='floorer')
def floorer(df: DataFrame, columns_to_floor: List[str], precomputed_floors: Optional[Dict[str, float]] = ...) -> LearnerReturnType: ...

@curry
@log_learner_time(learner_name='ecdfer')
def ecdfer(df: DataFrame, ascending: bool = ..., prediction_column: str = ..., ecdf_column: str = ..., max_range: int = ...) -> LearnerReturnType: ...

@curry
@log_learner_time(learner_name='discrete_ecdfer')
def discrete_ecdfer(df: DataFrame, ascending: bool = ..., prediction_column: str = ..., ecdf_column: str = ..., max_range: int = ..., round_method: Callable[[float], int] = ...) -> LearnerReturnType: ...

@curry
def prediction_ranger(df: DataFrame, prediction_min: float, prediction_max: float, prediction_column: str = ...) -> LearnerReturnType: ...

def apply_replacements(df: DataFrame, columns: List[str], vec: Dict[str, Dict[Any, Any]], replace_unseen: Any) -> DataFrame: ...

@column_duplicatable('value_maps')
@curry
@log_learner_time(learner_name='value_mapper')
def value_mapper(df: DataFrame, value_maps: Dict[str, Dict[Any, Any]], ignore_unseen: bool = ..., replace_unseen_to: Any = ...) -> LearnerReturnType: ...

@column_duplicatable('columns_to_truncate')
@curry
@log_learner_time(learner_name='truncate_categorical')
def truncate_categorical(df: DataFrame, columns_to_truncate: List[str], percentile: float, replacement: Union[int, str, float] = ..., replace_unseen: Union[int, str, float] = ..., store_mapping: bool = ...) -> LearnerReturnType: ...

@column_duplicatable('columns_to_rank')
@curry
@log_learner_time(learner_name='rank_categorical')
def rank_categorical(df: DataFrame, columns_to_rank: List[str], replace_unseen: Union[int, str, float] = ..., store_mapping: bool = ...) -> LearnerReturnType: ...

@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='count_categorizer')
def count_categorizer(df: DataFrame, columns_to_categorize: List[str], replace_unseen: int = ..., store_mapping: bool = ...) -> LearnerReturnType: ...

@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='label_categorizer')
def label_categorizer(df: DataFrame, columns_to_categorize: List[str], replace_unseen: Union[int, str, float] = ..., store_mapping: bool = ...) -> LearnerReturnType: ...

@column_duplicatable('columns_to_bin')
@curry
@log_learner_time(learner_name='quantile_biner')
def quantile_biner(df: DataFrame, columns_to_bin: List[str], q: Union[int, List[float]] = ..., right: bool = ...) -> LearnerReturnType: ...

@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='onehot_categorizer')
def onehot_categorizer(df: DataFrame, columns_to_categorize: List[str], hardcode_nans: bool = ..., drop_first_column: bool = ..., store_mapping: bool = ...) -> LearnerReturnType: ...

@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='target_categorizer')
def target_categorizer(df: DataFrame, columns_to_categorize: List[str], target_column: str, smoothing: float = ..., ignore_unseen: bool = ..., store_mapping: bool = ...) -> LearnerReturnType: ...

@column_duplicatable('columns_to_scale')
@curry
@log_learner_time(learner_name='standard_scaler')
def standard_scaler(df: DataFrame, columns_to_scale: List[str]) -> LearnerReturnType: ...

@column_duplicatable('columns_to_transform')
@curry
@log_learner_time(learner_name='custom_transformer')
def custom_transformer(df: DataFrame, columns_to_transform: List[str], transformation_function: Callable[[DataFrame], DataFrame], is_vectorized: bool = ...) -> LearnerReturnType: ...

@curry
@log_learner_time(learner_name='null_injector')
def null_injector(df: DataFrame, proportion: float, columns_to_inject: Optional[List[str]] = ..., groups: Optional[List[List[str]]] = ..., seed: int = ...) -> LearnerReturnType: ...

@curry
@log_learner_time(learner_name='missing_warner')
def missing_warner(df: DataFrame, cols_list: List[str], new_column_name: str = ..., detailed_warning: bool = ..., detailed_column_name: Optional[str] = ...) -> LearnerReturnType: ...