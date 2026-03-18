from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal, TYPE_CHECKING
import pandas as pd
import numpy.typing as npt
from pathlib import Path
from fklearn.types import LearnerReturnType

if TYPE_CHECKING:
    from lightgbm import Booster

def logistic_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> LearnerReturnType: ...

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
) -> LearnerReturnType: ...

def _get_catboost_shap_values(
    df: pd.DataFrame,
    cbr: Any,
    features: List[str],
    target: str,
    weights: List[Any],
    cat_features: Optional[List[str]],
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
) -> LearnerReturnType: ...

def nlp_logistic_classification_learner(
    df: pd.DataFrame,
    text_feature_cols: List[str],
    target: str,
    vectorizer_params: Optional[Dict[str, Any]] = ...,
    logistic_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
) -> LearnerReturnType: ...

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
    valid_sets: Optional[Any] = ...,
    valid_names: Optional[List[str]] = ...,
    feval: Optional[Union[Callable[..., Any], List[Callable[..., Any]]]] = ...,
    init_model: Optional[Union[str, Path, "Booster"]] = ...,
    feature_name: Union[List[str], Literal["auto"]] = ...,
    categorical_feature: Union[List[Union[str, int]], Literal["auto"]] = ...,
    keep_training_booster: bool = ...,
    callbacks: Optional[List[Callable[..., Any]]] = ...,
    dataset_init_score: Optional[Union[List[Any], List[List[Any]], npt.NDArray[Any], pd.Series, pd.DataFrame]] = ...,
) -> LearnerReturnType: ...