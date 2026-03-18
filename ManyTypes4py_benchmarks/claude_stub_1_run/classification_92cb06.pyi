```pyi
from typing import List, Any, Optional, Callable, Tuple, Union, Literal
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path

def logistic_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[dict] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, dict]: ...

def xgb_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[dict] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> Tuple[Callable[[pd.DataFrame, bool], pd.DataFrame], pd.DataFrame, dict]: ...

def _get_catboost_shap_values(
    df: pd.DataFrame,
    cbr: Any,
    features: List[str],
    target: str,
    weights: Optional[List[Any]],
    cat_features: Optional[List[str]],
) -> Any: ...

def catboost_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[dict] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> Tuple[Callable[[pd.DataFrame, bool], pd.DataFrame], pd.DataFrame, dict]: ...

def nlp_logistic_classification_learner(
    df: pd.DataFrame,
    text_feature_cols: List[str],
    target: str,
    vectorizer_params: Optional[dict] = None,
    logistic_params: Optional[dict] = None,
    prediction_column: str = 'prediction',
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, dict]: ...

def lgbm_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[dict] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
    valid_sets: Optional[List[pd.DataFrame]] = None,
    valid_names: Optional[List[str]] = None,
    feval: Optional[Union[Callable[..., Any], List[Callable[..., Any]]]] = None,
    init_model: Optional[Union[str, Path, Any]] = None,
    feature_name: Union[List[str], Literal['auto']] = 'auto',
    categorical_feature: Union[List[Union[str, int]], Literal['auto']] = 'auto',
    keep_training_booster: bool = False,
    callbacks: Optional[List[Callable[..., Any]]] = None,
    dataset_init_score: Optional[Union[List[Any], List[List[Any]], npt.NDArray[Any], pd.Series[Any], pd.DataFrame]] = None,
) -> Tuple[Callable[[pd.DataFrame, bool], pd.DataFrame], pd.DataFrame, dict]: ...
```