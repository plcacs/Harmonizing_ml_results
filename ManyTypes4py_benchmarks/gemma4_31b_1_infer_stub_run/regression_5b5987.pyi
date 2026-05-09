from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

# Define a type for the prediction function returned by the learners
# Most learners return a function that takes a DataFrame and returns a DataFrame
PredictFn = Callable[[pd.DataFrame, bool], pd.DataFrame]
# Some learners (like linear_regression_learner) return a function that only takes one argument
SimplePredictFn = Callable[[pd.DataFrame], pd.DataFrame]

# The return type of the learners is a tuple: (predict_function, predictions_df, log_dict)
LearnerReturnType = Tuple[Union[PredictFn, SimplePredictFn], pd.DataFrame, Dict[str, Any]]

def linear_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

def xgb_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

def catboost_regressor_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
) -> LearnerReturnType: ...

def gp_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    kernel: Optional[kernels.Kernel] = None,
    alpha: float = 0.1,
    extra_variance: Union[str, float] = 'fit',
    return_std: bool = False,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

def lgbm_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

def custom_supervised_model_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model: Any,
    supervised_type: str,
    log: Dict[str, Any],
    prediction_column: str = 'prediction',
) -> LearnerReturnType: ...

def elasticnet_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...