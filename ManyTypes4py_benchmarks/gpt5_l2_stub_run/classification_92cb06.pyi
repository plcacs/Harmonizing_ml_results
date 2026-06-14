from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from pathlib import Path
import numpy as np
import pandas as pd

def logistic_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

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
) -> Tuple[Callable[[pd.DataFrame, bool], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

def _get_catboost_shap_values(
    df: pd.DataFrame,
    cbr: Any,
    features: List[str],
    target: str,
    weights: Optional[Union[List[Any], np.ndarray]],
    cat_features: Optional[List[str]],
) -> np.ndarray: ...

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
) -> Tuple[Callable[[pd.DataFrame, bool], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

def nlp_logistic_classification_learner(
    df: pd.DataFrame,
    text_feature_cols: List[str],
    target: str,
    vectorizer_params: Optional[Dict[str, Any]] = ...,
    logistic_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...

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
    valid_sets: Optional[List[pd.DataFrame]] = ...,
    valid_names: Optional[List[str]] = ...,
    feval: Optional[Union[Callable[..., Any], List[Callable[..., Any]]]] = ...,
    init_model: Optional[Union[str, Path, Any]] = ...,
    feature_name: Union[List[str], Literal["auto"]] = ...,
    categorical_feature: Union[List[Union[str, int]], Literal["auto"]] = ...,
    keep_training_booster: bool = ...,
    callbacks: Optional[List[Callable[..., Any]]] = ...,
    dataset_init_score: Optional[Union[List[float], List[List[float]], np.ndarray, pd.Series, pd.DataFrame]] = ...,
) -> Tuple[Callable[[pd.DataFrame, bool], pd.DataFrame], pd.DataFrame, Dict[str, Any]]: ...