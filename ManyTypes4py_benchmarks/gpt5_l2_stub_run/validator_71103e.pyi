from typing import Callable, Dict, List, Tuple
import pandas as pd
from numpy import ndarray
from toolz.functoolz import identity

def validator_iteration(
    data: pd.DataFrame,
    train_index: ndarray,
    test_indexes: List[ndarray],
    fold_num: int,
    train_fn: Callable[[pd.DataFrame], Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, object]]],
    eval_fn: Callable[[pd.DataFrame], Dict[str, object]],
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False,
) -> Dict[str, object]: ...

def validator(
    train_data: pd.DataFrame,
    split_fn: Callable[[pd.DataFrame], Tuple[List[Tuple[ndarray, List[ndarray]]], List[Dict[str, object]]]],
    train_fn: Callable[[pd.DataFrame], Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, object]]],
    eval_fn: Callable[[pd.DataFrame], Dict[str, object]],
    perturb_fn_train: Callable[[pd.DataFrame], pd.DataFrame] = identity,
    perturb_fn_test: Callable[[pd.DataFrame], pd.DataFrame] = identity,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    return_all_train_logs: bool = False,
    verbose: bool = False,
    drop_empty_folds: bool = False,
) -> Dict[str, object]: ...

def parallel_validator_iteration(
    train_data: pd.DataFrame,
    fold: Tuple[int, Tuple[ndarray, List[ndarray]]],
    train_fn: Callable[[pd.DataFrame], Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, object]]],
    eval_fn: Callable[[pd.DataFrame], Dict[str, object]],
    predict_oof: bool,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False,
) -> Dict[str, object]: ...

def parallel_validator(
    train_data: pd.DataFrame,
    split_fn: Callable[[pd.DataFrame], Tuple[List[Tuple[ndarray, List[ndarray]]], List[Dict[str, object]]]],
    train_fn: Callable[[pd.DataFrame], Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict[str, object]]],
    eval_fn: Callable[[pd.DataFrame], Dict[str, object]],
    n_jobs: int = 1,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False,
) -> Dict[str, object]: ...