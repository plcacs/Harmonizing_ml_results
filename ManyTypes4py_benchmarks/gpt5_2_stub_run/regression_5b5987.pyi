from typing import Any, Dict, List, Optional, Callable, Union
import pandas
from fklearn.types import LearnerReturnType

np: Any = ...
pd: Any = ...
merge: Any = ...
curry: Any = ...
assoc: Any = ...
LinearRegression: Any = ...
ElasticNet: Any = ...
GaussianProcessRegressor: Any = ...
kernels: Any = ...
sk_version: str = ...
learner_pred_fn_docstring: Any = ...
learner_return_docstring: Any = ...
log_learner_time: Any = ...
expand_features_encoded: Any = ...

def linear_regression_learner(
    df: pandas.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> LearnerReturnType: ...

def xgb_regression_learner(
    df: pandas.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> LearnerReturnType: ...

def catboost_regressor_learner(
    df: pandas.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
) -> LearnerReturnType: ...

def gp_regression_learner(
    df: pandas.DataFrame,
    features: List[str],
    target: str,
    kernel: Any = ...,
    alpha: float = ...,
    extra_variance: Optional[Union[float, str]] = ...,
    return_std: bool = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    encode_extra_cols: bool = ...,
) -> LearnerReturnType: ...

def lgbm_regression_learner(
    df: pandas.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> LearnerReturnType: ...

def custom_supervised_model_learner(
    df: pandas.DataFrame,
    features: List[str],
    target: str,
    model: Any,
    supervised_type: str,
    log: Dict[str, Any],
    prediction_column: str = ...,
) -> LearnerReturnType: ...

def elasticnet_regression_learner(
    df: pandas.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> LearnerReturnType: ...