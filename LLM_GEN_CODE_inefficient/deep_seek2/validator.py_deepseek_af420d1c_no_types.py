import gc
import inspect
import warnings
from typing import Dict, Tuple, List, Callable, Any, Optional, Union, Sequence
import pandas as pd
from joblib import Parallel, delayed
from toolz import compose
from toolz.curried import assoc, curry, dissoc, first, map, partial, pipe
from toolz.functoolz import identity
from tqdm import tqdm
from fklearn.types import EvalFnType, LearnerFnType, LogType
from fklearn.types import SplitterFnType, ValidatorReturnType, PerturbFnType

def validator_iteration(data, train_index, test_indexes, fold_num, train_fn, eval_fn, predict_oof=False, return_eval_logs_on_train=False, verbose=False):
    """
    Perform an iteration of train test split, training and evaluation.
    """
    train_data = data.iloc[train_index]
    empty_set_warn = 'Splitter on validator_iteration in generating an empty training dataset. train_data.shape is %s' % str(train_data.shape)
    warnings.warn(empty_set_warn) if train_data.shape[0] == 0 else None
    predict_fn, train_out, train_log = train_fn(train_data)
    if return_eval_logs_on_train:
        train_log = assoc(train_log, 'eval_results', eval_fn(train_out))
    eval_results = []
    oof_predictions = []
    if verbose:
        print(f'Running validation for {fold_num} fold.')
    for test_index in tqdm(test_indexes) if verbose else test_indexes:
        test_predictions = predict_fn(data.iloc[test_index])
        eval_results.append(eval_fn(test_predictions))
        if predict_oof:
            oof_predictions.append(test_predictions)
    logs = {'fold_num': fold_num, 'train_log': train_log, 'eval_results': eval_results}
    return assoc(logs, 'oof_predictions', oof_predictions) if predict_oof else logs

@curry
def validator(train_data, split_fn, train_fn, eval_fn, perturb_fn_train=identity, perturb_fn_test=identity, predict_oof=False, return_eval_logs_on_train=False, return_all_train_logs=False, verbose=False, drop_empty_folds=False):
    """
    Splits the training data into folds given by the split function and performs a train-evaluation sequence on each
    fold by calling ``validator_iteration`` given the evaluation function.
    """
    folds, logs = split_fn(train_data)
    train_fn = compose(train_fn, perturb_fn_train)
    eval_fn = compose(eval_fn, perturb_fn_test)

    def fold_iter(fold):
        fold_num, (train_index, test_indexes) = fold
        test_contains_null_folds = max(map(lambda x: len(x) == 0, test_indexes))
        train_fold_is_null = len(train_index) == 0
        if (train_fold_is_null or test_contains_null_folds) and drop_empty_folds:
            return {'empty_fold': True}
        else:
            iter_results = validator_iteration(train_data, train_index, test_indexes, fold_num, train_fn, eval_fn, predict_oof, return_eval_logs_on_train, verbose)
        return assoc(iter_results, 'empty_fold', False)
    zipped_logs = pipe(folds, enumerate, map(fold_iter), partial(zip, logs))

    def clean_logs(log_tuple):
        split_log_error = list()
        new_validator_logs = list()
        new_split_log = list()
        for split_log, validator_log in log_tuple:
            if not validator_log['empty_fold']:
                new_validator_logs.append(dissoc(validator_log, 'empty_fold'))
                new_split_log.append(split_log)
            else:
                split_log_error.append(split_log)
        return (zip(new_split_log, new_validator_logs), split_log_error)
    if drop_empty_folds:
        zipped_logs, zipped_error_logs = clean_logs(zipped_logs)

    def _join_split_log(log_tuple):
        train_log = {}
        split_log, validator_log = log_tuple
        train_log['train_log'] = validator_log['train_log']
        return (train_log, assoc(dissoc(validator_log, 'train_log'), 'split_log', split_log))

    def get_perturbed_columns(perturbator):
        args = inspect.getfullargspec(perturbator).kwonlydefaults
        return args['cols'] if args else []
    train_logs_, validator_logs = zip(*map(_join_split_log, zipped_logs))
    if return_all_train_logs:
        train_logs = {'train_log': [log['train_log'] for log in train_logs_]}
    else:
        train_logs = first(train_logs_)
    perturbator_log = {'perturbated_train': [], 'perturbated_test': []}
    if perturb_fn_train != identity:
        perturbator_log['perturbated_train'] = get_perturbed_columns(perturb_fn_train)
    if perturb_fn_test != identity:
        perturbator_log['perturbated_test'] = get_perturbed_columns(perturb_fn_test)
    train_logs = assoc(train_logs, 'perturbator_log', perturbator_log)
    if drop_empty_folds:
        train_logs = assoc(train_logs, 'fold_error_logs', zipped_error_logs)
    return assoc(train_logs, 'validator_log', list(validator_logs))

def parallel_validator_iteration(train_data, fold, train_fn, eval_fn, predict_oof, return_eval_logs_on_train=False, verbose=False):
    fold_num, (train_index, test_indexes) = fold
    return validator_iteration(train_data, train_index, test_indexes, fold_num, train_fn, eval_fn, predict_oof, return_eval_logs_on_train, verbose)

@curry
def parallel_validator(train_data, split_fn, train_fn, eval_fn, n_jobs=1, predict_oof=False, return_eval_logs_on_train=False, verbose=False):
    """
    Splits the training data into folds given by the split function and
    performs a train-evaluation sequence on each fold. Tries to run each
    fold in parallel using up to n_jobs processes.
    """
    folds, logs = split_fn(train_data)
    result = Parallel(n_jobs=n_jobs, backend='threading')((delayed(parallel_validator_iteration)(train_data, x, train_fn, eval_fn, predict_oof, return_eval_logs_on_train, verbose) for x in enumerate(folds)))
    gc.collect()
    train_log = {'train_log': [fold_result['train_log'] for fold_result in result]}

    @curry
    def kwdissoc(d, key):
        return dissoc(d, key)
    validator_logs = pipe(result, partial(zip, logs), map(lambda log_tuple: assoc(log_tuple[1], 'split_log', log_tuple[0])), map(kwdissoc(key='train_log')), list)
    return assoc(train_log, 'validator_log', validator_logs)