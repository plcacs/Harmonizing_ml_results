from typing import Dict, List, Tuple, Any
from numpy import ndarray
import pandas as pd
from fklearn.types import EvalFnType, LearnerFnType, LogType, SplitterFnType, ValidatorReturnType, PerturbFnType

def validator_iteration(
    data: pd.DataFrame,
    train_index: ndarray,
    test_indexes: List[ndarray],
    fold_num: int,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    predict_oof: bool = ...,
    return_eval_logs_on_train: bool = ...,
    verbose: bool = ...
) -> Dict[str, Any]: ...

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
    drop_empty_folds: bool = ...
) -> ValidatorReturnType: ...

def parallel_validator_iteration(
    train_data: pd.DataFrame,
    fold: Tuple[int, Tuple[ndarray, List[ndarray]]],
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    predict_oof: bool,
    return_eval_logs_on_train: bool = ...,
    verbose: bool = ...
) -> Dict[str, Any]: ...

def parallel_validator(
    train_data: pd.DataFrame,
    split_fn: SplitterFnType,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    n_jobs: int = ...,
    predict_oof: bool = ...,
    return_eval_logs_on_train: bool = ...,
    verbose: bool = ...
) -> ValidatorReturnType: ...