from typing import List, Dict, Optional, Callable, Tuple, Any
import pandas as pd
import numpy as np
from lightgbm import Booster
from catboost import CatBoostClassifier

def logistic_classification_learner(df: pd.DataFrame, features: List[str], target: str, params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    ...

def xgb_classification_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    ...

def catboost_classification_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    ...

def nlp_logistic_classification_learner(df: pd.DataFrame, text_feature_cols: List[str], target: str, vectorizer_params: Optional[Dict[str, Any]] = None, logistic_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction') -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    ...

def lgbm_classification_learner(df: pd.DataFrame, features: List[str], target: str, learning_rate: float = 0.1, num_estimators: int = 100, extra_params: Optional[Dict[str, Any]] = None, prediction_column: str = 'prediction', weight_column: Optional[str] = None, encode_extra_cols: bool = True, valid_sets: Optional[List[pd.DataFrame]] = None, valid_names: Optional[List[str]] = None, feval: Optional[Callable] = None, init_model: Optional[Any] = None, feature_name: str = 'auto', categorical_feature: str = 'auto', keep_training_booster: bool = False, callbacks: Optional[List[Callable]] = None, dataset_init_score: Optional[Any] = None) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    ...