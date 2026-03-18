```python
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fklearn.types import LearnerReturnType, LearnerLogType, LogType

if TYPE_CHECKING:
    from lightgbm import Booster
    import xgboost as xgb
    import catboost as catboost
    import lightgbm as lgbm
    import shap

def logistic_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...,
    weight_column: Optional[str] = ...,
    encode_extra_cols: bool = ...
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
    encode_extra_cols: bool = ...
) -> LearnerReturnType: ...

def _get_catboost_shap_values(
    df: pd.DataFrame,
    cbr: Any,
    features: List[str],
    target: str,
    weights: List[Any],
    cat_features: List[str]
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
    encode_extra_cols: bool = ...
) -> LearnerReturnType: ...

def nlp_logistic_classification_learner(
    df: pd.DataFrame,
    text_feature_cols: List[str],
    target: str,
    vectorizer_params: Optional[Dict[str, Any]] = ...,
    logistic_params: Optional[Dict[str, Any]] = ...,
    prediction_column: str = ...
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
    valid_sets: Optional[List[pd.DataFrame]] = ...,
    valid_names: Optional[List[str]] = ...,
    feval: Optional[Union[Callable[..., Any], List[Callable[..., Any]]]] = ...,
    init_model: Optional[Union[str, Path, "Booster"]] = ...,
    feature_name: Union[List[str], str] = ...,
    categorical_feature: Union[List[Union[str, int]], str] = ...,
    keep_training_booster: bool = ...,
    callbacks: Optional[List[Callable[..., Any]]] = ...,
    dataset_init_score: Optional[Union[List[Any], List[List[Any]], np.ndarray, pd.Series, pd.DataFrame]] = ...
) -> LearnerReturnType: ...
```