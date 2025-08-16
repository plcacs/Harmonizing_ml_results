from typing import List, Any, Optional, Callable, Tuple, Union, Literal
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from toolz import curry, merge, assoc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import __version__ as sk_version
from fklearn.types import LearnerReturnType, LearnerLogType, LogType
from fklearn.common_docstrings import learner_return_docstring, learner_pred_fn_docstring
from fklearn.training.utils import log_learner_time, expand_features_encoded
if TYPE_CHECKING:
    from lightgbm import Booster

@curry
@log_learner_time(learner_name='logistic_classification_learner')
def logistic_classification_learner(df: pd.DataFrame, features: List[str], target: str, params: Optional[dict] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> Tuple[Callable, pd.DataFrame, dict]:
    ...

@curry
@log_learner_time(learner_name='xgb_classification_learner')
def xgb_classification_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[dict] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> Tuple[Callable, pd.DataFrame, dict]:
    ...

@curry
def _get_catboost_shap_values(df: pd.DataFrame, cbr: Any, features: List[str], target: str, weights: List, cat_features: List[str]) -> Any:
    ...

@curry
@log_learner_time(learner_name='catboost_classification_learner')
def catboost_classification_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[dict] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> Tuple[Callable, pd.DataFrame, dict]:
    ...

@curry
@log_learner_time(learner_name='nlp_logistic_classification_learner')
def nlp_logistic_classification_learner(df: pd.DataFrame, text_feature_cols: List[str], target: str, vectorizer_params: Optional[dict] = None, logistic_params: Optional[dict] = None, prediction_column: str = 'prediction') -> Tuple[Callable, pd.DataFrame, dict]:
    ...

@curry
@log_learner_time(learner_name='lgbm_classification_learner')
def lgbm_classification_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[dict] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True, valid_sets: Optional[List[pd.DataFrame]] = None, valid_names: Optional[List[str]] = None, feval: Optional[Callable] = None, init_model: Optional[str] = None, feature_name: Union[List[str], Literal['auto']] = 'auto', categorical_feature: Union[List[str, int], Literal['auto']] = 'auto', keep_training_booster: bool = False, callbacks: Optional[List[Callable]] = None, dataset_init_score: Optional[Union[List, List[List], np.ndarray, pd.Series, pd.DataFrame]] = None) -> Tuple[Callable, pd.DataFrame, dict]:
    ...
