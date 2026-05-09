from typing import List, Any, Optional, Callable, Tuple, Union, Literal, Dict
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from fklearn.types import LearnerReturnType

# The functions are decorated with @curry, which typically means they return 
# either a partially applied function or the final result. 
# For the purpose of the stub, we define the full signature.

def logistic_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

def xgb_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

def _get_catboost_shap_values(
    df: pd.DataFrame,
    cbr: Any,
    features: List[str],
    target: str,
    weights: Union[List[Any], npt.NDArray],
    cat_features: Optional[List[str]],
) -> npt.NDArray: ...

def catboost_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
) -> LearnerReturnType: ...

def nlp_logistic_classification_learner(
    df: pd.DataFrame,
    text_feature_cols: List[str],
    target: str,
    vectorizer_params: Optional[Dict[str, Any]] = None,
    logistic_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
) -> LearnerReturnType: ...

def lgbm_classification_learner(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    learning_rate: float = 0.1,
    num_estimators: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    prediction_column: str = 'prediction',
    weight_column: Optional[str] = None,
    encode_extra_cols: bool = True,
    valid_sets: Optional[List[pd.DataFrame]] = None,
    valid_names: Optional[List[str]] = None,
    feval: Optional[Union[Callable, List[Callable]]] = None,
    init_model: Optional[Union[str, Path, Any]] = None,
    feature_name: Union[List[str], Literal['auto']] = 'auto',
    categorical_feature: Union[List[Union[str, int]], Literal['auto']] = 'auto',
    keep_training_booster: bool = False,
    callbacks: Optional[List[Callable]] = None,
    dataset_init_score: Optional[Union[List[Any], List[List[Any]], npt.NDArray, pd.Series, pd.DataFrame]] = None,
) -> LearnerReturnType: ...