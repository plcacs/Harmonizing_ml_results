from typing import Dict, Tuple, List

def validator_iteration(data: pd.DataFrame, train_index: np.array, test_indexes: List[np.array], fold_num: int, train_fn: LearnerFnType, eval_fn: EvalFnType, predict_oof: bool = False, return_eval_logs_on_train: bool = False, verbose: bool = False) -> Dict:
    ...

@curry
def validator(train_data: pd.DataFrame, split_fn: SplitterFnType, train_fn: LearnerFnType, eval_fn: EvalFnType, perturb_fn_train: PerturbFnType = identity, perturb_fn_test: PerturbFnType = identity, predict_oof: bool = False, return_eval_logs_on_train: bool = False, return_all_train_logs: bool = False, verbose: bool = False, drop_empty_folds: bool = False) -> List[Dict]:
    ...

def parallel_validator_iteration(train_data: pd.DataFrame, fold: Tuple[int, Tuple[np.array, List[np.array]]], train_fn: LearnerFnType, eval_fn: EvalFnType, predict_oof: bool, return_eval_logs_on_train: bool = False, verbose: bool = False) -> Dict:
    ...

@curry
def parallel_validator(train_data: pd.DataFrame, split_fn: SplitterFnType, train_fn: LearnerFnType, eval_fn: EvalFnType, n_jobs: int = 1, predict_oof: bool = False, return_eval_logs_on_train: bool = False, verbose: bool = False) -> List[Dict]:
    ...
