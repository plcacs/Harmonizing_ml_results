import gc
import inspect
import warnings
from typing import Dict, Tuple, List, Optional, Union, Any
import pandas as pd
from joblib import Parallel
from toolz import compose
from toolz.curried import assoc, curry, dissoc, first, map, partial, pipe
from toolz.functoolz import identity
from tqdm import tqdm
from fklearn.types import EvalFnType, LearnerFnType, LogType, SplitterFnType, ValidatorReturnType, PerturbFnType

@curry
def validator_iteration(data: pd.DataFrame, train_index: pd.Index, test_indexes: List[pd.Index], fold_num: int, train_fn: LearnerFnType, eval_fn: EvalFnType, predict_oof: bool = False, return_eval_logs_on_train: bool = False, verbose: bool = False) -> Dict[str, Union[int, Dict, List]]:
    ...

@curry
def validator(train_data: pd.DataFrame, split_fn: SplitterFnType, train_fn: LearnerFnType, eval_fn: EvalFnType, perturb_fn_train: PerturbFnType = identity, perturb_fn_test: PerturbFnType = identity, predict_oof: bool = False, return_eval_logs_on_train: bool = False, return_all_train_logs: bool = False, verbose: bool = False, drop_empty_folds: bool = False) -> ValidatorReturnType:
    ...

def parallel_validator_iteration(train_data: pd.DataFrame, fold: Tuple[int, Tuple[pd.Index, List[pd.Index]]], train_fn: LearnerFnType, eval_fn: EvalFnType, predict_oof: bool, return_eval_logs_on_train: bool = False, verbose: bool = False) -> Dict[str, Union[int, Dict, List]]:
    ...

@curry
def parallel_validator(train_data: pd.DataFrame, split_fn: SplitterFnType, train_fn: LearnerFnType, eval_fn: EvalFnType, n_jobs: int = 1, predict_oof: bool = False, return_eval_logs_on_train: bool = False, verbose: bool = False) -> Dict[str, Union[List[Dict], List]]:
    ...