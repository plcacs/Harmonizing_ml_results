from typing import Any, Dict, List, Tuple
import pandas as pd
from fklearn.types import EvalFnType, LearnerFnType, SplitterFnType, PerturbFnType

def validator_iteration(
    data: pd.DataFrame,
    train_index: Any,
    test_indexes: List[Any],
    fold_num: int,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    predict_oof: bool = ...,
    return_eval_logs_on_train: bool = ...,
    verbose: bool = ...,
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
    drop_empty_folds: bool = ...,
) -> Dict[str, Any]: ...

def parallel_validator_iteration(
    train_data: pd.DataFrame,
    fold: Any,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    predict_oof: bool,
    return_eval_logs_on_train: bool = ...,
    verbose: bool = ...,
) -> Dict[str, Any]: ...

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