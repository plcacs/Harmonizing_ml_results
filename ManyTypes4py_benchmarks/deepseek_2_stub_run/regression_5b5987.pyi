```python
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import Kernel
from fklearn.types import LearnerReturnType

def linear_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...
) -> LearnerReturnType: ...

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

def gp_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    kernel: Optional[Kernel] = ...,
    alpha: float = ...,
    extra_variance: Union[str, float] = ...,
    return_std: bool = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    encode_extra_cols: bool = ...
) -> LearnerReturnType: ...

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

def custom_supervised_model_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model: Any,
    supervised_type: str,
    log: Dict[str, Dict[str, Any]],
    prediction_column: str = ...
) -> LearnerReturnType: ...

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