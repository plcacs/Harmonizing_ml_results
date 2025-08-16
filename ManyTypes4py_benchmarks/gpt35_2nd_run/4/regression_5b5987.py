from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
from toolz import merge, curry, assoc
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn import __version__ as sk_version
from fklearn.common_docstrings import learner_pred_fn_docstring, learner_return_docstring
from fklearn.types import LearnerReturnType
from fklearn.training.utils import log_learner_time, expand_features_encoded

@curry
@log_learner_time(learner_name='linear_regression_learner')
def linear_regression_learner(df: pd.DataFrame, features: List[str], target: str, params: Dict[str, Any] = None, prediction_column: str = 'prediction', weight_column: str = None, encode_extra_cols: bool = True) -> LearnerReturnType:
    ...

@curry
@log_learner_time(learner_name='xgb_regression_learner')
def xgb_regression_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Dict[str, Any] = None, prediction_column: str = 'prediction', weight_column: str = None, encode_extra_cols: bool = True) -> LearnerReturnType:
    ...

@curry
@log_learner_time(learner_name='catboost_regressor_learner')
def catboost_regressor_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Dict[str, Any] = None, prediction_column: str = 'prediction', weight_column: str = None) -> LearnerReturnType:
    ...

@curry
@log_learner_time(learner_name='gp_regression_learner')
def gp_regression_learner(df: pd.DataFrame, features: List[str], target: str, kernel: kernels.Kernel = None, alpha: float = 0.1, extra_variance: Union[float, str] = 'fit', return_std: bool = False, extra_params: Dict[str, Any] = None, prediction_column: str = 'prediction', encode_extra_cols: bool = True) -> LearnerReturnType:
    ...

@curry
@log_learner_time(learner_name='lgbm_regression_learner')
def lgbm_regression_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Dict[str, Any] = None, prediction_column: str = 'prediction', weight_column: str = None, encode_extra_cols: bool = True) -> LearnerReturnType:
    ...

@curry
@log_learner_time(learner_name='custom_supervised_model_learner')
def custom_supervised_model_learner(df: pd.DataFrame, features: List[str], target: str, model: Any, supervised_type: str, log: Dict[str, Dict], prediction_column: str = 'prediction') -> LearnerReturnType:
    ...

@curry
@log_learner_time(learner_name='elasticnet_regression_learner')
def elasticnet_regression_learner(df: pd.DataFrame, features: List[str], target: str, params: Dict[str, Any] = None, prediction_column: str = 'prediction', weight_column: str = None, encode_extra_cols: bool = True) -> LearnerReturnType:
    ...
