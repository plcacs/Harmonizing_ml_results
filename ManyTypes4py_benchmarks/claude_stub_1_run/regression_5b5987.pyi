```pyi
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

def linear_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

def xgb_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

def catboost_regressor_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

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
    encode_extra_cols: bool = ...,
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

def lgbm_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

def custom_supervised_model_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model: Any,
    supervised_type: str,
    log: Dict[str, Dict[str, Any]],
    prediction_column: str = ...,
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

def elasticnet_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...
```