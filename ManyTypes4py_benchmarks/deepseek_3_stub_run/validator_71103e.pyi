import pandas as pd
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, 
    Iterable, Iterator, Sequence
)
from typing_extensions import TypeAlias
from numpy import ndarray
from toolz.functoolz import identity
from fklearn.types import (
    EvalFnType, LearnerFnType, LogType, 
    SplitterFnType, ValidatorReturnType, PerturbFnType
)

# Type aliases for clarity
IndexList: TypeAlias = List[ndarray]
FoldType: TypeAlias = Tuple[ndarray, IndexList]
FoldList: TypeAlias = List[FoldType]
LogTuple: TypeAlias = Tuple[Any, Dict[str, Any]]
LogList: TypeAlias = List[Dict[str, Any]]
ValidatorResult: TypeAlias = Dict[str, Any]

def validator_iteration(
    data: pd.DataFrame,
    train_index: ndarray,
    test_indexes: IndexList,
    fold_num: int,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False
) -> Dict[str, Any]: ...

def validator(
    train_data: pd.DataFrame,
    split_fn: SplitterFnType,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    perturb_fn_train: PerturbFnType = identity,
    perturb_fn_test: PerturbFnType = identity,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    return_all_train_logs: bool = False,
    verbose: bool = False,
    drop_empty_folds: bool = False
) -> ValidatorReturnType: ...

def parallel_validator_iteration(
    train_data: pd.DataFrame,
    fold: Tuple[int, FoldType],
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    predict_oof: bool,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False
) -> Dict[str, Any]: ...

def parallel_validator(
    train_data: pd.DataFrame,
    split_fn: SplitterFnType,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    n_jobs: int = 1,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False
) -> ValidatorReturnType: ...