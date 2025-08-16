import gc
import inspect
import warnings
from typing import Dict, Tuple, List
import pandas as pd
from joblib import Parallel, delayed
from toolz import compose
from toolz.curried import assoc, curry, dissoc, first, map, partial, pipe
from toolz.functoolz import identity
from tqdm import tqdm
from fklearn.types import EvalFnType, LearnerFnType, LogType
from fklearn.types import SplitterFnType, ValidatorReturnType, PerturbFnType

def validator_iteration(data: pd.DataFrame, train_index: np.ndarray, test_indexes: List[np.ndarray], fold_num: int, train_fn: LearnerFnType, eval_fn: EvalFnType, predict_oof: bool = False, return_eval_logs_on_train: bool = False, verbose: bool = False) -> LogType:
    ...

@curry
def validator(train_data: pd.DataFrame, split_fn: SplitterFnType, train_fn: LearnerFnType, eval_fn: EvalFnType, perturb_fn_train: PerturbFnType = identity, perturb_fn_test: PerturbFnType = identity, predict_oof: bool = False, return_eval_logs_on_train: bool = False, return_all_train_logs: bool = False, verbose: bool = False, drop_empty_folds: bool = False) -> List[LogType]:
    ...

def parallel_validator_iteration(train_data: pd.DataFrame, fold: Tuple[int, Tuple[np.ndarray, List[np.ndarray]]], train_fn: LearnerFnType, eval_fn: EvalFnType, predict_oof: bool, return_eval_logs_on_train: bool = False, verbose: bool = False) -> LogType:
    ...

@curry
def parallel_validator(train_data: pd.DataFrame, split_fn: SplitterFnType, train_fn: LearnerFnType, eval_fn: EvalFnType, n_jobs: int = 1, predict_oof: bool = False, return_eval_logs_on_train: bool = False, verbose: bool = False) -> List[LogType]:
    ...
