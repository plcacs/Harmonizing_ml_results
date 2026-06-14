from typing import List, Optional, Callable, Union, TYPE_CHECKING
import pandas as pd
from pathlib import Path
from fklearn.types import LearnerReturnType, LearnerLogType, LogType

if TYPE_CHECKING:
    from lightgbm import Booster


def logistic_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[dict] = ...,
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
    extra_params: Optional[dict] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> LearnerReturnType: ...

def _get_catboost_shap_values(
    df: pd.DataFrame,
    cbr: object,
    features: List[str],
    target: str,
    weights: Optional[list],
    cat_features: Optional[List[str]],
) -> object: ...

def catboost_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[dict] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
) -> LearnerReturnType: ...

def nlp_logistic_classification_learner(
    df: pd.DataFrame,
    text_feature_cols: List[str],
    target: str,
    vectorizer_params: Optional[dict] = ...,
    logistic_params: Optional[dict] = ...,
    prediction_column: str = ...,
) -> LearnerReturnType: ...

def lgbm_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = ...,
    num_estimators: int = ...,
    extra_params: Optional[dict] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...,
    valid_sets: Optional[List[pd.DataFrame]] = ...,
    valid_names: Optional[List[str]] = ...,
    feval: Optional[Union[Callable, List[Callable]]] = ...,
    init_model: Optional[Union[str, Path, "Booster"]] = ...,
    feature_name: Union[List[str], str] = ...,
    categorical_feature: Union[List[str], List[int], str] = ...,
    keep_training_booster: bool = ...,
    callbacks: Optional[List[Callable]] = ...,
    dataset_init_score: Optional[object] = ...,
) -> LearnerReturnType: ...