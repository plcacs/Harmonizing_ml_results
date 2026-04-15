from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
    Literal,
)
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from toolz import curry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fklearn.types import LearnerReturnType, LearnerLogType, LogType

if TYPE_CHECKING:
    from lightgbm import Booster
    import xgboost as xgb
    from catboost import CatBoostClassifier, Pool
    import lightgbm as lgbm
    import shap

@curry
def logistic_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = None,
    prediction_column: str = "prediction",
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

@curry
def xgb_classification_learner(
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

@curry
def _get_catboost_shap_values(
    df: pd.DataFrame,
    cbr: Any,
    features: List[str],
    target: str,
    weights: Optional[List[float]],
    cat_features: Optional[List[str]],
) -> np.ndarray: ...

@curry
def catboost_classification_learner(
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

@curry
def nlp_logistic_classification_learner(
    df: pd.DataFrame,
    text_feature_cols: List[str],
    target: str,
    vectorizer_params: Optional[Dict[str, Any]] = None,
    logistic_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = "prediction",
) -> LearnerReturnType: ...

@curry
def lgbm_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = "prediction",
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
    valid_sets: Optional[List[pd.DataFrame]] = None,
    valid_names: Optional[List[str]] = None,
    feval: Optional[Union[Callable, List[Callable]]] = None,
    init_model: Optional[Union[str, Path, "Booster"]] = None,
    feature_name: Union[List[str], Literal["auto"]] = "auto",
    categorical_feature: Union[List[Union[str, int]], Literal["auto"]] = "auto",
    keep_training_booster: bool = False,
    callbacks: Optional[List[Callable]] = None,
    dataset_init_score: Optional[
        Union[
            List[float],
            List[List[float]],
            np.ndarray,
            pd.Series,
            pd.DataFrame,
        ]
    ] = None,
) -> LearnerReturnType: ...