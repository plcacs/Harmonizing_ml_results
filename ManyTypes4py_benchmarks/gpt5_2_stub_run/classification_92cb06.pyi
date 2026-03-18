from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd

sk_version: str = ...

def logistic_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> Tuple[Callable[..., pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

def xgb_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> Tuple[Callable[..., pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

def _get_catboost_shap_values(
    df: pd.DataFrame,
    cbr: Any,
    features: List[str],
    target: str,
    weights: Any,
    cat_features: Any,
) -> Any: ...

def catboost_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> Tuple[Callable[..., pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

def nlp_logistic_classification_learner(
    df: pd.DataFrame,
    text_feature_cols: List[str],
    target: str,
    vectorizer_params: Optional[Dict[str, Any]] = ...,
    logistic_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
) -> Tuple[Callable[..., pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

def lgbm_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
    valid_sets: Any = ...,
    valid_names: Any = ...,
    feval: Any = ...,
    init_model: Any = ...,
    feature_name: Any = ...,
    categorical_feature: Any = ...,
    keep_training_booster: bool = ...,
    callbacks: Any = ...,
    dataset_init_score: Any = ...,
) -> Tuple[Callable[..., pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...