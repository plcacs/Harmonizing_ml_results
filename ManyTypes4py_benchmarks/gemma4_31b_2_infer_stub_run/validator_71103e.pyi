import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Tuple, Union, Optional, overload
from fklearn.types import EvalFnType, LearnerFnType, LogType, SplitterFnType, ValidatorReturnType, PerturbFnType

# Type aliases based on the provided implementation and fklearn.types
# LearnerFnType: (pd.DataFrame) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LogType]
# EvalFnType: (pd.DataFrame) -> Dict[str, Any]
# SplitterFnType: (pd.DataFrame) -> Tuple[List[Tuple[np.ndarray, List[np.ndarray]]], List[Any]]

def validator_iteration(
    data: pd.DataFrame,
    train_index: np.ndarray,
    test_indexes: List[np.ndarray],
    fold_num: int,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]: ...

@overload
def validator(
    train_data: pd.DataFrame,
    split_fn: SplitterFnType,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    perturb_fn_train: PerturbFnType = ...,
    perturb_fn_test: PerturbFnType = ...,
    predict_oof: bool = ...,
    return_eval_logs_on_train: bool = ...,
    return_all_train_logs: bool = ...,
    verbose: bool = ...,
    drop_empty_folds: bool = ...,
) -> Dict[str, Any]: ...

@overload
def validator(
    *,
    train_data: pd.DataFrame,
    split_fn: SplitterFnType,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    perturb_fn_train: PerturbFnType = ...,
    perturb_fn_test: PerturbFnType = ...,
    predict_oof: bool = ...,
    return_eval_logs_on_train: bool = ...,
    return_all_train_logs: bool = ...,
    verbose: bool = ...,
    drop_empty_folds: bool = ...,
) -> Dict[str, Any]: ...

@curry
def validator(
    train_data: pd.DataFrame,
    split_fn: SplitterFnType,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    perturb_fn_train: PerturbFnType = ...,
    perturb_fn_test: PerturbFnType = ...,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    return_all_train_logs: bool = False,
    verbose: bool = False,
    drop_empty_folds: bool = False,
) -> Dict[str, Any]: ...

def parallel_validator_iteration(
    train_data: pd.DataFrame,
    fold: Tuple[int, Tuple[np.ndarray, List[np.ndarray]]],
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    predict_oof: bool,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]: ...

@overload
def parallel_validator(
    train_data: pd.DataFrame,
    split_fn: SplitterFnType,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    n_jobs: int = ...,
    predict_oof: bool = ...,
    return_eval_logs_on_train: bool = ...,
    verbose: bool = ...,
) -> Dict[str, Any]: ...

@overload
def parallel_validator(
    *,
    train_data: pd.DataFrame,
    split_fn: SplitterFnType,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    n_jobs: int = ...,
    predict_oof: bool = ...,
    return_eval_logs_on_train: bool = ...,
    verbose: bool = ...,
) -> Dict[str, Any]: ...

@curry
def parallel_validator(
    train_data: pd.DataFrame,
    split_fn: SplitterFnType,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    n_jobs: int = 1,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]: ...