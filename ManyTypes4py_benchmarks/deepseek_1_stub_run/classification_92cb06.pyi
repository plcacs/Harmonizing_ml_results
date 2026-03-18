```python
from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
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
    import catboost
    import lightgbm as lgbm
    import shap

def logistic_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[dict] = None,
    prediction_column: str = "prediction",
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

def xgb_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[dict] = None,
    prediction_column: str = "prediction",
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

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
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[dict] = None,
    prediction_column: str = "prediction",
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

def nlp_logistic_classification_learner(
    df: pd.DataFrame,
    text_feature_cols: List[str],
    target: str,
    vectorizer_params: Optional[dict] = None,
    logistic_params: Optional[dict] = None,
    prediction_column: str = "prediction",
) -> LearnerReturnType: ...

def lgbm_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[dict] = None,
    prediction_column: str = "prediction",
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
    valid_sets: Optional[Any] = None,
    valid_names: Optional[List[str]] = None,
    feval: Optional[Any] = None,
    init_model: Optional[Any] = None,
    feature_name: Any = "auto",
    categorical_feature: Any = "auto",
    keep_training_booster: bool = False,
    callbacks: Optional[List[Callable]] = None,
    dataset_init_score: Optional[Any] = None,
) -> LearnerReturnType: ...
```