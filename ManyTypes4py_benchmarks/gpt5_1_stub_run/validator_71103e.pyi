from typing import Any, Dict

gc: Any
inspect: Any
warnings: Any
pd: Any
Parallel: Any
delayed: Any
compose: Any
assoc: Any
curry: Any
dissoc: Any
first: Any
map: Any
partial: Any
pipe: Any
identity: Any
tqdm: Any
EvalFnType: Any
LearnerFnType: Any
LogType: Any
SplitterFnType: Any
ValidatorReturnType: Any
PerturbFnType: Any

def validator_iteration(
    data: Any,
    train_index: Any,
    test_indexes: Any,
    fold_num: int,
    train_fn: Any,
    eval_fn: Any,
    predict_oof: bool = ...,
    return_eval_logs_on_train: bool = ...,
    verbose: bool = ...,
) -> Dict[str, Any]: ...

def validator(
    train_data: Any,
    split_fn: Any,
    train_fn: Any,
    eval_fn: Any,
    perturb_fn_train: Any = ...,
    perturb_fn_test: Any = ...,
    predict_oof: bool = ...,
    return_eval_logs_on_train: bool = ...,
    return_all_train_logs: bool = ...,
    verbose: bool = ...,
    drop_empty_folds: bool = ...,
) -> Dict[str, Any]: ...

def parallel_validator_iteration(
    train_data: Any,
    fold: Any,
    train_fn: Any,
    eval_fn: Any,
    predict_oof: bool,
    return_eval_logs_on_train: bool = ...,
    verbose: bool = ...,
) -> Dict[str, Any]: ...

def parallel_validator(
    train_data: Any,
    split_fn: Any,
    train_fn: Any,
    eval_fn: Any,
    n_jobs: int = ...,
    predict_oof: bool = ...,
    return_eval_logs_on_train: bool = ...,
    verbose: bool = ...,
) -> Dict[str, Any]: ...