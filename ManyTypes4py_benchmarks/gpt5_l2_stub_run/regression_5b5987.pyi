from typing import Any, Dict, List, Optional, Union
import pandas as pd
from sklearn.gaussian_process import kernels as gp_kernels
from fklearn.types import LearnerReturnType

def linear_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = None,
    prediction_column: str = "prediction",
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
    prediction_column: str = "prediction",
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
    prediction_column: str = "prediction",
    weight_column: Optional[str] = None,
) -> LearnerReturnType: ...

def gp_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    kernel: Optional[gp_kernels.Kernel] = None,
    alpha: float = 0.1,
    extra_variance: Union[float, str, None] = "fit",
    return_std: bool = False,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = "prediction",
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

def lgbm_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = "prediction",
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

def custom_supervised_model_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model: Any,
    supervised_type: str,
    log: Dict[str, Dict[str, Any]],
    prediction_column: str = "prediction",
) -> LearnerReturnType: ...

def elasticnet_regression_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = None,
    prediction_column: str = "prediction",
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...