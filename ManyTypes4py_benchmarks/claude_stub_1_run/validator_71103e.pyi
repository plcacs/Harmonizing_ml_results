```pyi
from typing import Any, Callable, Dict, List, Tuple, Optional
import pandas as pd

def validator_iteration(
    data: pd.DataFrame,
    train_index: Any,
    test_indexes: List[Any],
    fold_num: int,
    train_fn: Callable[[pd.DataFrame], Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]],
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]: ...

def validator(
    train_data: pd.DataFrame,
    split_fn: Callable[[pd.DataFrame], Tuple[List[Tuple[Any, List[Any]]], List[Any]]],
    train_fn: Callable[[pd.DataFrame], Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]],
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    perturb_fn_train: Callable[[pd.DataFrame], pd.DataFrame] = ...,
    perturb_fn_test: Callable[[pd.DataFrame], pd.DataFrame] = ...,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    return_all_train_logs: bool = False,
    verbose: bool = False,
    drop_empty_folds: bool = False,
) -> Dict[str, Any]: ...

def parallel_validator_iteration(
    train_data: pd.DataFrame,
    fold: Tuple[int, Tuple[Any, List[Any]]],
    train_fn: Callable[[pd.DataFrame], Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]],
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    predict_oof: bool,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]: ...

def parallel_validator(
    train_data: pd.DataFrame,
    split_fn: Callable[[pd.DataFrame], Tuple[List[Tuple[Any, List[Any]]], List[Any]]],
    train_fn: Callable[[pd.DataFrame], Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, Any]]],
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    n_jobs: int = 1,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]: ...
```