import operator
from datetime import datetime, timedelta
from itertools import chain, repeat, starmap
from typing import Callable, Iterable, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.utils import check_random_state
from toolz.curried import curry, partial, pipe, assoc, accumulate, map, filter
from fklearn.common_docstrings import splitter_return_docstring
from fklearn.types import DateType, LogType, SplitterReturnType

def _log_time_fold(time_fold: Tuple[pd.Series, pd.Series]) -> dict:
    train_time, test_time = time_fold
    return {'train_start': train_time.min(), 'train_end': train_time.max(), 'train_size': train_time.shape[0], 'test_start': test_time.min(), 'test_end': test_time.max(), 'test_size': test_time.shape[0]}

def _get_lc_folds(date_range: pd.DatetimeIndex, date_fold_filter_fn: Callable[[pd.Timestamp], pd.DataFrame], test_time: pd.Series, time_column: str, min_samples: int) -> List[Tuple[pd.Series, pd.Series]]:
    return pipe(date_range, map(date_fold_filter_fn), map(lambda df: df[time_column]), filter(lambda s: len(s.index) > min_samples), lambda train: zip(train, repeat(test_time)), list)

def _get_sc_folds(date_range: pd.DatetimeIndex, date_fold_filter_fn: Callable[[pd.Period], pd.DataFrame], time_column: str, min_samples: int) -> List[pd.Series]:
    return pipe(date_range, map(date_fold_filter_fn), map(lambda df: df[time_column]), filter(lambda s: len(s.index) > min_samples), list)

def _get_sc_test_fold_idx_and_logs(test_data: pd.DataFrame, train_time: pd.Series, time_column: str, first_test_moment: pd.Timestamp, last_test_moment: pd.Timestamp, min_samples: int, freq: str) -> Tuple[List[dict], List[List[np.ndarray]]]:
    periods_range = pd.period_range(start=first_test_moment, end=last_test_moment, freq=freq)

    def date_filter_fn(period: pd.Period) -> pd.DataFrame:
        return test_data[test_data[time_column].dt.to_period(freq) == period]
    folds = _get_sc_folds(periods_range, date_filter_fn, time_column, min_samples)
    logs = list(map(_log_time_fold, zip(repeat(train_time), folds)))
    test_indexes = list(map(lambda test: [test.index], folds))
    return (logs, test_indexes)

def _lc_fold_to_indexes(folds: List[Tuple[pd.Series, pd.Series]]) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    return list(starmap(lambda train, test: (train.index, [test.index]), folds))

@curry
def k_fold_splitter(train_data: pd.DataFrame, n_splits: int, random_state: Union[int, None] = None, stratify_column: Union[str, None] = None) -> SplitterReturnType:
    if stratify_column is not None:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(train_data, train_data[stratify_column])
    else:
        folds = KFold(n_splits, shuffle=True, random_state=random_state).split(train_data)
    result = list(map(lambda f: (f[0], [f[1]]), folds))
    logs = [{'train_size': len(fold[0]), 'test_size': train_data.shape[0] - len(fold[0])} for fold in result]
    return (result, logs)
k_fold_splitter.__doc__ += splitter_return_docstring

@curry
def out_of_time_and_space_splitter(train_data: pd.DataFrame, n_splits: int, in_time_limit: Union[str, datetime], time_column: str, space_column: str, holdout_gap: timedelta = timedelta(days=0)) -> SplitterReturnType:
    train_data = train_data.reset_index()
    space_folds = GroupKFold(n_splits).split(train_data, groups=train_data[space_column])
    if isinstance(in_time_limit, str):
        in_time_limit = datetime.strptime(in_time_limit, '%Y-%m-%d')
    folds = pipe(space_folds, partial(starmap, lambda f_train, f_test: [train_data.iloc[f_train][time_column], train_data.iloc[f_test][time_column]]), partial(starmap, lambda train, test: (train[train <= in_time_limit], test[test > in_time_limit + holdout_gap])), list)
    logs = list(map(_log_time_fold, folds))
    folds_indexes = _lc_fold_to_indexes(folds)
    return (folds_indexes, logs)
out_of_time_and_space_splitter.__doc__ += splitter_return_docstring

@curry
def time_and_space_learning_curve_splitter(train_data: pd.DataFrame, training_time_limit: str, space_column: str, time_column: str, freq: str = 'M', space_hold_percentage: float = 0.5, holdout_gap: timedelta = timedelta(days=0), random_state: Union[int, None] = None, min_samples: int = 1000) -> SplitterReturnType:
    train_data = train_data.reset_index()
    first_moment = train_data[time_column].min()
    date_range = pd.date_range(start=first_moment, end=training_time_limit, freq=freq)
    rng = check_random_state(random_state)
    out_of_space_mask = pipe(train_data, lambda df: df[df[time_column] > date_range[-1]], lambda df: df[space_column].unique(), lambda array: rng.choice(array, int(len(array) * space_hold_percentage), replace=False), lambda held_space: train_data[space_column].isin(held_space))
    training_time_limit_dt = datetime.strptime(training_time_limit, '%Y-%m-%d') + holdout_gap
    test_time = train_data[(train_data[time_column] > training_time_limit_dt) & out_of_space_mask][time_column]

    def date_filter_fn(date: pd.Timestamp) -> pd.DataFrame:
        return train_data[(train_data[time_column] <= date) & ~out_of_space_mask]
    folds = _get_lc_folds(date_range, date_filter_fn, test_time, time_column, min_samples)
    logs = list(map(_log_time_fold, folds))
    folds_indexes = _lc_fold_to_indexes(folds)
    return (folds_indexes, logs)
time_and_space_learning_curve_splitter.__doc__ += splitter_return_docstring

@curry
def time_learning_curve_splitter(train_data: pd.DataFrame, training_time_limit: str, time_column: str, freq: str = 'M', holdout_gap: timedelta = timedelta(days=0), min_samples: int = 1000) -> SplitterReturnType:
    train_data = train_data.reset_index()
    first_moment = train_data[time_column].min()
    date_range = pd.date_range(start=first_moment, end=training_time_limit, freq=freq)
    effective_training_time_end = date_range[-1]
    test_time = train_data[train_data[time_column] > effective_training_time_end + holdout_gap][time_column]

    def date_filter_fn(date: pd.Timestamp) -> pd.DataFrame:
        return train_data[train_data[time_column] <= date]
    folds = _get_lc_folds(date_range, date_filter_fn, test_time, time_column, min_samples)
    logs = list(map(_log_time_fold, folds))
    folds_indexes = _lc_fold_to_indexes(folds)
    return (folds_indexes, logs)
time_learning_curve_splitter.__doc__ += splitter_return_docstring

@curry
def reverse_time_learning_curve_splitter(train_data: pd.DataFrame, time_column: str, training_time_limit: str, lower_time_limit: Union[str, None] = None, freq: str = 'MS', holdout_gap: timedelta = timedelta(days=0), min_samples: int = 1000) -> SplitterReturnType:
    train_data = train_data.reset_index()
    first_moment = lower_time_limit if lower_time_limit else train_data[time_column].min()
    date_range = pd.date_range(start=first_moment, end=training_time_limit, freq=freq)
    effective_training_time_end = date_range[-1]
    train_range = train_data[train_data[time_column] <= effective_training_time_end]
    test_time = train_data[train_data[time_column] > effective_training_time_end + holdout_gap][time_column]

    def date_filter_fn(date: pd.Timestamp) -> pd.DataFrame:
        return train_range.loc[train_data[time_column] >= date]
    folds = _get_lc_folds(date_range[::-1], date_filter_fn, test_time, time_column, min_samples)
    logs = list(map(_log_time_fold, folds))
    folds_indexes = _lc_fold_to_indexes(folds)
    return (folds_indexes, logs)
reverse_time_learning_curve_splitter.__doc__ += splitter_return_docstring

@curry
def spatial_learning_curve_splitter(train_data: pd.DataFrame, space_column: str, time_column: str, training_limit: Union[datetime, str], holdout_gap: timedelta = timedelta(days=0), train_percentages: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0), random_state: Union[int, None] = None) -> SplitterReturnType:
    if np.min(np.array(train_percentages)) < 0 or np.max(np.array(train_percentages)) > 1:
        raise ValueError('Train percentages must be between 0 and 1')
    if isinstance(training_limit, str):
        training_limit = datetime.strptime(training_limit, '%Y-%m-%d')
    if training_limit < train_data[time_column].min() or training_limit > train_data[time_column].max():
        raise ValueError('Temporal training limit should be within datasets temporal bounds (min and max times)')
    if timedelta(days=0) > holdout_gap:
        raise ValueError('Holdout gap cannot be negative')
    if holdout_gap >= train_data[time_column].max() - training_limit:
        raise ValueError('After taking the gap into account, there should be enough time for the holdout set')
    train_data = train_data.reset_index()
    spatial_ids = train_data[space_column].sample(frac=1, random_state=random_state).unique()
    cumulative_ids = pipe(spatial_ids, lambda ids: (np.array(train_percentages) * len(ids)).astype(int), lambda idx: np.split(spatial_ids, idx)[:-1], lambda to_list: map(lambda x: x.tolist(), to_list), lambda drop_empty: filter(None, drop_empty), accumulate(operator.add))
    validation_set = train_data[train_data[time_column] > training_limit + holdout_gap]
    train_data = train_data[train_data[time_column] <= training_limit]
    folds = [(train_data[train_data[space_column].isin(ids)][time_column], validation_set[time_column]) for ids in cumulative_ids]
    folds_indices = _lc_fold_to_indexes(folds)
    logs = [assoc(learner, 'percentage', p) for learner, p in zip(map(_log_time_fold, folds), train_percentages)]
    return (folds_indices, logs)
spatial_learning_curve_splitter.__doc__ += splitter_return_docstring

@curry
def stability_curve_time_splitter(train_data: pd.DataFrame, training_time_limit: str, time_column: str, freq: str = 'M', min_samples: int = 1000) -> SplitterReturnType:
    train_data = train_data.reset_index()
    train_time = train_data[train_data[time_column] <= training_time_limit][time_column]
    test_data = train_data[train_data[time_column] > training_time_limit]
    first_test_moment = test_data[time_column].min()
    last_test_moment = test_data[time_column].max()
    logs, test_indexes = _get_sc_test_fold_idx_and_logs(test_data, train_time, time_column, first_test_moment, last_test_moment, min_samples, freq)
    logs = [{k: [dic[k] for dic in logs] for k in logs[0]}]
    flattened_test_indices = list(chain.from_iterable(test_indexes))
    return ([(train_time.index, flattened_test_indices)], logs)
stability_curve_time_splitter.__doc__ += splitter_return_docstring

@curry
def stability_curve_time_in_space_splitter(train_data: pd.DataFrame, training_time_limit: str, space_column: str, time_column: str, freq: str = 'M', space_hold_percentage: float = 0.5, random_state: Union[int, None] = None, min_samples: int = 1000) -> SplitterReturnType:
    train_data = train_data.reset_index()
    rng = check_random_state(random_state)
    train_time = train_data[train_data[time_column] <= training_time_limit][time_column]
    test_data = pipe(train_data, lambda trand_df: trand_df.iloc[train_time.index][space_column].unique(), lambda space: rng.choice(space, int(len(space) * space_hold_percentage), replace=False), lambda held_space: train_data[(train_data[time_column] > training_time_limit) & train_data[space_column].isin(held_space)])
    first_test_moment = test_data[time_column].min()
    last_test_moment = test_data[time_column].max()
    logs, test_indexes = _get_sc_test_fold_idx_and_logs(test_data, train_time, time_column, first_test_moment, last_test_moment, min_samples, freq)
    logs = [{k: [dic[k] for dic in logs] for k in logs[0]}]
    flattened_test_indices = list(chain.from_iterable(test_indexes))
    return ([(train_time.index, flattened_test_indices)], logs)
stability_curve_time_in_space_splitter.__doc__ += splitter_return_docstring

@curry
def stability_curve_time_space_splitter(train_data: pd.DataFrame, training_time_limit: str, space_column: str, time_column: str, freq: str = 'M', space_hold_percentage: float = 0.5, random_state: Union[int, None] = None, min_samples: int = 1000) -> SplitterReturnType:
    train_data = train_data.reset_index()
    rng = check_random_state(random_state)
    train_time = train_data[train_data[time_column] <= training_time_limit][time_column]
    train_index = train_time.index.values
    train_space = train_data.iloc[train_index][space_column].unique()
    held_space = rng.choice(train_space, int(len(train_space) * space_hold_percentage), replace=False)
    test_data = train_data[(train_data[time_column] > training_time_limit) & ~train_data[space_column].isin(held_space)]
    train_index = train_data[(train_data[time_column] <= training_time_limit) & train_data[space_column].isin(held_space)].index.values
    first_test_moment = test_data[time_column].min()
    last_test_moment = test_data[time_column].max()
    logs, test_indexes = _get_sc_test_fold_idx_and_logs(test_data, train_time, time_column, first_test_moment, last_test_moment, min_samples, freq)
    logs = [{k: [dic[k] for dic in logs] for k in logs[0]}]
    flattened_test_indices = list(chain.from_iterable(test_indexes))
    return ([(train_index, flattened_test_indices)], logs)
stability_curve_time_space_splitter.__doc__ += splitter_return_docstring

@curry
def forward_stability_curve_time_splitter(train_data: pd.DataFrame, training_time_start: Union[datetime, str], training_time_end: Union[datetime, str], time_column: str, holdout_gap: timedelta = timedelta(days=0), holdout_size: timedelta = timedelta(days=90), step: timedelta = timedelta(days=90), move_training_start_with_steps: bool = True) -> SplitterReturnType:
    if isinstance(training_time_start, str):
        training_time_start = datetime.strptime(training_time_start, '%Y-%m-%d')
    if isinstance(training_time_end, str):
        training_time_end = datetime.strptime(training_time_end, '%Y-%m-%d')
    train_data = train_data.reset_index()
    max_date = train_data[time_column].max()
    if not train_data[time_column].min() <= training_time_start < training_time_end <= max_date:
        raise ValueError('Temporal training limits should be within datasets temporal bounds (min and max times)')
    if timedelta(days=0) > holdout_gap:
        raise ValueError('Holdout gap cannot be negative')
    if timedelta(days=0) > holdout_size:
        raise ValueError('Holdout size cannot be negative')
    n_folds = int(np.ceil((max_date - holdout_size - holdout_gap - training_time_end) / step))
    if n_folds <= 0:
        raise ValueError('After taking the gap and holdout into account, there should be enough time for the holdout set')
    train_ranges = [(training_time_start + i * step * move_training_start_with_steps, training_time_end + i * step) for i in range(n_folds)]
    test_ranges = [(training_time_end + holdout_gap + i * step, training_time_end + holdout_gap + holdout_size + i * step) for i in range(n_folds)]
    train_idx = [train_data[(train_data[time_column] >= start) & (train_data[time_column] < end)].index for start, end in train_ranges]
    test_idx = [[train_data[(train_data[time_column] >= start) & (train_data[time_column] < end)].index] for start, end in test_ranges]
    logs = [_log_time_fold((train_data.iloc[i][time_column], train_data.iloc[j[0]][time_column])) for i, j in zip(train_idx, test_idx)]
    return (list(zip(train_idx, test_idx)), logs)
forward_stability_curve_time_splitter.__doc__ += splitter_return_docstring
