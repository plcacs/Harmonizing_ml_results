#!/usr/bin/env python3
from typing import Callable, Any, List, Tuple, Dict, Iterable
import gc
import inspect
import warnings
import pandas as pd
from numpy import ndarray
from joblib import Parallel, delayed
from toolz import compose
from toolz.curried import assoc, curry, dissoc, first, map, partial, pipe
from toolz.functoolz import identity
from tqdm import tqdm
from fklearn.types import EvalFnType, LearnerFnType, LogType
from fklearn.types import SplitterFnType, ValidatorReturnType, PerturbFnType

# Function parallel_validator_iteration will be used in parallel computation.
def parallel_validator_iteration(
    data: pd.DataFrame,
    fold: Tuple[int, Tuple[ndarray, List[ndarray]]],
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    predict_oof: bool,
    return_eval_logs_on_train: bool,
    verbose: bool
) -> Dict[str, Any]:
    fold_num, (train_index, test_indexes) = fold
    return func_htexezpt(
        data,
        train_index,
        test_indexes,
        fold_num,
        train_fn,
        eval_fn,
        predict_oof,
        return_eval_logs_on_train,
        verbose
    )

def func_htexezpt(
    data: pd.DataFrame,
    train_index: ndarray,
    test_indexes: List[ndarray],
    fold_num: int,
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform an iteration of train test split, training and evaluation.
    """
    train_data: pd.DataFrame = data.iloc[train_index]
    empty_set_warn: str = (
        'Splitter on validator_iteration in generating an empty training dataset. train_data.shape is %s'
        % str(train_data.shape)
    )
    if train_data.shape[0] == 0:
        warnings.warn(empty_set_warn)
    predict_fn, train_out, train_log = train_fn(train_data)
    if return_eval_logs_on_train:
        train_log = assoc(train_log, 'eval_results', eval_fn(train_out))
    eval_results: List[Any] = []
    oof_predictions: List[Any] = []
    if verbose:
        print(f'Running validation for {fold_num} fold.')
    for test_index in (tqdm(test_indexes) if verbose else test_indexes):
        test_predictions = predict_fn(data.iloc[test_index])
        eval_results.append(eval_fn(test_predictions))
        if predict_oof:
            oof_predictions.append(test_predictions)
    logs: Dict[str, Any] = {'fold_num': fold_num, 'train_log': train_log, 'eval_results': eval_results}
    if predict_oof:
        return assoc(logs, 'oof_predictions', oof_predictions)
    return logs

@curry
def func_asaet836(
    train_data: pd.DataFrame,
    split_fn: Callable[[pd.DataFrame], Tuple[List[Tuple[ndarray, List[ndarray]]], List[Any]]],
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    perturb_fn_train: PerturbFnType = identity,
    perturb_fn_test: PerturbFnType = identity,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    return_all_train_logs: bool = False,
    verbose: bool = False,
    drop_empty_folds: bool = False
) -> Dict[str, Any]:
    """
    Splits the training data into folds and performs a train-evaluation sequence on each fold.
    """
    folds, logs = split_fn(train_data)
    train_fn = compose(train_fn, perturb_fn_train)
    eval_fn = compose(eval_fn, perturb_fn_test)

    def func_z2ihwgg9(
        fold: Tuple[int, Tuple[ndarray, List[ndarray]]]
    ) -> Dict[str, Any]:
        fold_num, (train_index, test_indexes) = fold
        test_contains_null_folds: bool = max(list(map(lambda x: len(x) == 0, test_indexes)))
        train_fold_is_null: bool = (len(train_index) == 0)
        if (train_fold_is_null or test_contains_null_folds) and drop_empty_folds:
            return {'empty_fold': True}
        else:
            iter_results: Dict[str, Any] = func_htexezpt(
                train_data,
                train_index,
                test_indexes,
                fold_num,
                train_fn,
                eval_fn,
                predict_oof,
                return_eval_logs_on_train,
                verbose
            )
            return assoc(iter_results, 'empty_fold', False)

    # Use func_z2ihwgg9 as the folding function.
    fold_iter = func_z2ihwgg9
    zipped_logs: Iterable[Tuple[Any, Any]] = pipe(folds, enumerate, map(fold_iter), partial(zip, logs))
    
    def func_dp31k6d9(
        log_tuple: Iterable[Tuple[Any, Dict[str, Any]]]
    ) -> Tuple[Iterable[Tuple[Any, Dict[str, Any]]], List[Any]]:
        split_log_error: List[Any] = []
        new_validator_logs: List[Dict[str, Any]] = []
        new_split_log: List[Any] = []
        for split_log, validator_log in log_tuple:
            if not validator_log.get('empty_fold', False):
                new_validator_logs.append(dissoc(validator_log, 'empty_fold'))
                new_split_log.append(split_log)
            else:
                split_log_error.append(split_log)
        return list(zip(new_split_log, new_validator_logs)), split_log_error

    if drop_empty_folds:
        zipped_logs, zipped_error_logs = func_dp31k6d9(zipped_logs)
    else:
        zipped_error_logs = []

    def _join_split_log(
        log_tuple: Tuple[Any, Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        split_log, validator_log = log_tuple
        train_log_out: Dict[str, Any] = {'train_log': validator_log['train_log']}
        return train_log_out, assoc(dissoc(validator_log, 'train_log'), 'split_log', split_log)

    def func_qjhy4rx3(perturbator: Callable[..., Any]) -> List[str]:
        args: Any = inspect.getfullargspec(perturbator).kwonlydefaults
        return args['cols'] if args and 'cols' in args else []

    train_logs_and_validator = list(map(_join_split_log, zipped_logs))
    train_logs_, validator_logs = zip(*train_logs_and_validator)  # type: ignore
    if return_all_train_logs:
        train_logs: Dict[str, Any] = {'train_log': [log['train_log'] for log in train_logs_]}
    else:
        train_logs = first(train_logs_)
    perturbator_log: Dict[str, Any] = {'perturbated_train': [], 'perturbated_test': []}
    if perturb_fn_train != identity:
        perturbator_log['perturbated_train'] = func_qjhy4rx3(perturb_fn_train)
    if perturb_fn_test != identity:
        perturbator_log['perturbated_test'] = func_qjhy4rx3(perturb_fn_test)
    train_logs = assoc(train_logs, 'perturbator_log', perturbator_log)
    if drop_empty_folds:
        train_logs = assoc(train_logs, 'fold_error_logs', zipped_error_logs)
    return assoc(train_logs, 'validator_log', list(validator_logs))

def func_gvwkjrt4(
    train_data: pd.DataFrame,
    fold: Tuple[int, Tuple[ndarray, List[ndarray]]],
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    predict_oof: bool,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    fold_num, (train_index, test_indexes) = fold
    return func_htexezpt(
        train_data,
        train_index,
        test_indexes,
        fold_num,
        train_fn,
        eval_fn,
        predict_oof,
        return_eval_logs_on_train,
        verbose
    )

@curry
def func_oyag0n8g(
    train_data: pd.DataFrame,
    split_fn: Callable[[pd.DataFrame], Tuple[List[Tuple[ndarray, List[ndarray]]], List[Any]]],
    train_fn: LearnerFnType,
    eval_fn: EvalFnType,
    n_jobs: int = 1,
    predict_oof: bool = False,
    return_eval_logs_on_train: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Splits the training data and performs train-evaluation sequences on each fold in parallel.
    """
    folds, logs = split_fn(train_data)
    result: List[Dict[str, Any]] = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(parallel_validator_iteration)(
            train_data, x, train_fn, eval_fn, predict_oof, return_eval_logs_on_train, verbose
        ) for x in enumerate(folds)
    )
    gc.collect()
    train_log: Dict[str, Any] = {'train_log': [fold_result['train_log'] for fold_result in result]}

    @curry
    def func_46gyfmx6(d: Dict[str, Any], key: str) -> Dict[str, Any]:
        return dissoc(d, key)

    validator_logs: List[Dict[str, Any]] = list(
        map(
            lambda log_tuple: assoc(log_tuple[1], 'split_log', log_tuple[0]),
            zip(logs, result)
        )
    )
    validator_logs = list(map(func_46gyfmx6(key='train_log'), validator_logs))
    return assoc(train_log, 'validator_log', validator_logs)
