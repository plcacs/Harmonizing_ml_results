from typing import Any, Callable, Dict, List, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions import empirical_distribution as ed
from toolz import curry, merge, compose, mapcat
from fklearn.common_docstrings import learner_return_docstring, learner_pred_fn_docstring
from fklearn.training.utils import log_learner_time
from fklearn.types import LearnerReturnType, LearnerLogType
from fklearn.preprocessing.schema import column_duplicatable

@curry
@log_learner_time(learner_name='selector')
def selector(df: pd.DataFrame, training_columns: List[str], predict_columns: Union[List[str], None] = None) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    ...

@curry
@log_learner_time(learner_name='capper')
def capper(df: pd.DataFrame, columns_to_cap: List[str], precomputed_caps: Union[Dict[str, float], None] = None) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    ...

@curry
@log_learner_time(learner_name='floorer')
def floorer(df: pd.DataFrame, columns_to_floor: List[str], precomputed_floors: Union[Dict[str, float], None] = None) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    ...

@curry
@log_learner_time(learner_name='ecdfer')
def ecdfer(df: pd.DataFrame, ascending: bool, prediction_column: str, ecdf_column: str, max_range: int) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    ...

@curry
@log_learner_time(learner_name='discrete_ecdfer')
def discrete_ecdfer(df: pd.DataFrame, ascending: bool, prediction_column: str, ecdf_column: str, max_range: int, round_method: Callable[[float], float]) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    ...

@curry
@log_learner_time(learner_name='prediction_ranger')
def prediction_ranger(df: pd.DataFrame, prediction_min: float, prediction_max: float, prediction_column: str