```python
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from toolz import curry
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from fklearn.types import LearnerReturnType

@curry
def linear_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...
) -> LearnerReturnType: ...

@curry
def xgb_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...
) -> LearnerReturnType: ...

@curry
def catboost_regressor_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...
) -> LearnerReturnType: ...

@curry
def gp_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    kernel: Optional[Any] = ...,
    alpha: float = ...,
    extra_variance: Union[str, float] = ...,
    return_std: bool = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    encode_extra_cols: bool = ...
) -> LearnerReturnType: ...

@curry
def lgbm_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...
) -> LearnerReturnType: ...

@curry
def custom_supervised_model_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model: Any,
    supervised_type: str,
    log: Dict[str, Dict[str, Any]],
    prediction_column: str = ...
) -> LearnerReturnType: ...

@curry
def elasticnet_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...
) -> LearnerReturnType: ...
```