from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from fklearn.types import LearnerFnType, LearnerReturnType, PredictFnType

TREATMENT_FEATURE: str = ...

def _append_treatment_feature(features: List[str], treatment_feature: str) -> List[str]: ...
def _get_learner_features(learner: Callable[..., LearnerFnType]) -> List[str]: ...
def _get_unique_treatments(df: pd.DataFrame, treatment_col: str, control_name: str) -> List[str]: ...
def _filter_by_treatment(df: pd.DataFrame, treatment_col: str, treatment_name: str, control_name: str) -> pd.DataFrame: ...
def _create_treatment_flag(df: pd.DataFrame, treatment_col: str, treatment_name: str, control_name: str) -> pd.DataFrame: ...
def _fit_by_treatment(
    df: pd.DataFrame,
    learner: LearnerFnType,
    treatment_col: str,
    control_name: str,
    treatments: List[str],
) -> Tuple[Dict[str, PredictFnType], Dict[str, Any]]: ...
def _predict_by_treatment_flag(
    df: pd.DataFrame,
    learner_fcn: PredictFnType,
    is_treatment: bool,
    prediction_column: str,
) -> np.ndarray: ...
def _simulate_treatment_effect(
    df: pd.DataFrame,
    treatments: List[str],
    control_name: str,
    learners: Dict[str, PredictFnType],
    prediction_column: str,
) -> pd.DataFrame: ...
def causal_s_classification_learner(
    df: pd.DataFrame,
    treatment_col: str,
    control_name: str,
    prediction_column: str,
    learner: Callable[..., LearnerFnType],
    learner_transformers: Optional[List[LearnerFnType]] = ...,
) -> Tuple[PredictFnType, pd.DataFrame, Dict[str, Any]]: ...
def _simulate_t_learner_treatment_effect(
    df: pd.DataFrame,
    learners: Dict[str, PredictFnType],
    treatments: List[str],
    control_name: str,
    prediction_column: str,
) -> pd.DataFrame: ...
def _get_model_fcn(
    df: pd.DataFrame,
    treatment_col: str,
    treatment_name: str,
    learner: LearnerFnType,
) -> LearnerReturnType: ...
def _get_learners(
    df: pd.DataFrame,
    control_learner: LearnerFnType,
    treatment_learner: LearnerFnType,
    unique_treatments: List[str],
    control_name: str,
    treatment_col: str,
) -> Tuple[Dict[str, PredictFnType], Dict[str, Any]]: ...
def causal_t_classification_learner(
    df: pd.DataFrame,
    treatment_col: str,
    control_name: str,
    prediction_column: str,
    learner: Callable[..., LearnerFnType],
    treatment_learner: Optional[Callable[..., LearnerFnType]] = ...,
    learner_transformers: Optional[List[LearnerFnType]] = ...,
) -> Tuple[PredictFnType, pd.DataFrame, Dict[str, Any]]: ...