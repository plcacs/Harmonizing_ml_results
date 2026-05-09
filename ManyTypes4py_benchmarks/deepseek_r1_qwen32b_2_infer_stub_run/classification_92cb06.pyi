from typing import List, Optional, Union, Callable, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import Booster
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from fklearn.types import LearnerReturnType, LearnerLogType

def logistic_classification_learner(df: pd.DataFrame, features: List[str], target: str, params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> LearnerReturnType:
    ...

def xgb_classification_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> LearnerReturnType:
    ...

def _get_catboost_shap_values(df: pd.DataFrame, cbr: Any, features: List[str], target: str, weights: Optional[List[float]] = None, cat_features: Optional[List[str]] = None) -> np.ndarray:
    ...

def catboost_classification_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> LearnerReturnType:
    ...

def nlp_logistic_classification_learner(df: pd.DataFrame, text_feature_cols: List[str], target: str, vectorizer_params: Optional[Dict[str, Any]] = None, logistic_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction') -> LearnerReturnType:
    ...

def lgbm_classification_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True, valid_sets: Optional[List[pd.DataFrame]] = None, valid_names: Optional[List[str]] = None, feval: Optional[Callable] = None, init_model: Optional[Union[str, Path, Booster]] = None, feature_name: Union[List[str], str] = 'auto', categorical_feature: Union[List[Union[str, int]], str] = 'auto', keep_training_booster: bool = False, callbacks: Optional[List[Callable]] = None, dataset_init_score: Optional[Union[List[float], np.ndarray, pd.Series, pd.DataFrame]] = None) -> LearnerReturnType:
    ...