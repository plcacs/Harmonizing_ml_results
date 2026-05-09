import operator
from datetime import datetime, timedelta
from itertools import chain, repeat, starmap
from typing import Callable, Iterable, List, Tuple, Union, Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.utils import check_random_state
from toolz.curried import curry, partial, pipe, assoc, accumulate, map, filter
from fklearn.common_docstrings import splitter_return_docstring
from fklearn.types import DateType, LogType, SplitterReturnType

def _log_time_fold(time_fold: Tuple[pd.Series, pd.Series]) -> Dict[str, Union[datetime, int]]:
    train_time, test_time = time_fold
    return {'train_start': train_time.min(), 'train_end': train_time.max(), 'train_size': train_time.shape[0], 'test_start': test_time.min(), 'test_end': test_time.max(), 'test_size': test_time.shape[0]}

def _get_lc_folds(date_range: pd.DatetimeIndex, date_fold_filter_fn: Callable[[Any], pd.DataFrame], test_time: pd.Series, time_column: str, min_samples: int) -> List[Tuple[pd.Series, pd.Series]]:
    return pipe(date_range, map(date_fold_filter_fn), map(lambda df: df[time_column]), filter(lambda s: len(s.index) > min_samples), lambda train: zip(train, repeat(test_time)), list)

def _get_sc_folds(date_range: pd.DatetimeIndex, date_fold_filter_fn: Callable[[Any], pd.DataFrame], time_column: str, min_samples: int) -> List[pd.Series]:
    return pipe(date_range, map(date_fold_filter_fn), map(lambda df: df[time_column]), filter(lambda s: len(s.index) > min_samples), list)

def _get_sc_test_fold_idx_and_logs(test_data: pd.DataFrame, train_time: pd.Series, time_column: str, first_test_moment: datetime, last_test_moment: datetime, min_samples: int, freq: str) -> Tuple[List[Dict[str, Any]], List[List[pd.Index]]]:
    periods_range = pd.period_range(start=first_test_moment, end=last_test_moment, freq=freq)

    def date_filter_fn(period: Any) -> pd.DataFrame:
        return test_data[test_data[time_column].dt.to_period(freq) == period]
    folds = _get_sc_folds(periods_range, date_filter_fn, time_column, min_samples)
    logs = list(map(_log_time_fold, zip(repeat(train_time), folds)))
    test_indexes = list(map(lambda test: [test.index], folds))
    return (logs, test_indexes)

def _lc_fold_to_indexes(folds: Iterable[Tuple[pd.Series, pd.Series]]) -> List[Tuple[pd.Index, List[pd.Index]]]:
    return list(starmap(lambda train, test: (train.index, [test.index]), folds))

@curry
def k_fold_splitter(train_data: pd.DataFrame, n_splits: int, random_state: Optional[int] = None, stratify_column: Optional[str] = None) -> Tuple[List[Tuple[pd.Index, List[pd.Index]]], List[Dict[str, int]]]:
    """
    Makes K random train/test split folds for cross validation.
    The folds are made so that every sample is used at least once for
    evaluating and K-1 times for training.

    If stratified is set to True, the split preserves the distribution of stratify_column

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split into K-Folds for cross validation.

    n_splits : int
        The number of folds K for the K-Fold cross validation strategy.

    random_state : int
        Seed to be used by the random number generator.

    stratify_column : string
        Column name in train_data to be used for stratified split.
    """
    if stratify_column is not None:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(train_data, train_data[stratify_column])
    else:
        folds = KFold(n_splits, shuffle=True, random_state=random_state).split(train_data)
    result = list(map(lambda f: (f[0], [f[1]]), folds))
    logs = [{'train_size': len(fold[0]), 'test_size': train_data.shape[0] - len(fold[0])} for fold in result]
    return (result, logs)
k_fold_splitter.__doc__ += splitter_return_docstring

@curry
def out_of_time_and_space_splitter(train_data: pd.DataFrame, n_splits: int, in_time_limit: Union[str, datetime], time_column: str, space_column: str, holdout_gap: timedelta = timedelta(days=0)) -> Tuple[List[Tuple[pd.Index, List[pd.Index]]], List[Dict[str, Any]]]:
    """
    Makes K grouped train/test split folds for cross validation.
    The folds are made so that every ID is used at least once for
    evaluating and K-1 times for training. Also, for each fold, evaluation
    will always be out-of-ID and out-of-time.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split into K out-of-time and ID
        folds for cross validation.

    n_splits : int
        The number of folds K for the K-Fold cross validation strategy.

    in_time_limit : str or datetime.datetime
        A String representing the end time of the training data.
        It should be in the same format as the Date column in `train_data`.

    time_column : str
        The name of the Date column of `train_data`.

    space_column : str
        The name of the ID column of `train_data`.

    holdout_gap: datetime.timedelta
        Timedelta of the gap between the end of the training period and the start of the validation period.
    """
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
def time_and_space_learning_curve_splitter(train_data: pd.DataFrame, training_time_limit: str, space_column: str, time_column: str, freq: str = 'M', space_hold_percentage: float = 0.5, holdout_gap: timedelta = timedelta(days=0), random_state: Optional[int] = None, min_samples: int = 1000) -> Tuple[List[Tuple[pd.Index, List[pd.Index]]], List[Dict[str, Any]]]:
    """
    Splits the data into temporal buckets given by the specified frequency.
    Uses a fixed out-of-ID and time hold out set for every fold.
    Training size increases per fold, with more recent data being added in each fold.
    Useful for learning curve validation, that is, for seeing how hold out performance
    increases as the training size increases with more recent data.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split for learning curve estimation.

    training_time_limit : str
        The Date String for the end of the testing period. Should be of the same
        format as `time_column`.

    space_column : str
        The name of the ID column of `train_data`.

    time_column : str
        The name of the Date column of `train_data`.

    freq : str
        The temporal frequency.
        See: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    space_hold_percentage : float
        The proportion of hold out IDs.

    holdout_gap: datetime.timedelta
        Timedelta of the gap between the end of the training period and the start of the validation period.

    random_state : int
        A seed for the random number generator for ID sampling across train and
        hold out sets.

    min_samples : int
        The minimum number of samples required in the split to keep the split.
    """
    train_data = train_data.reset_index()
    first_moment = train_data[time_column].min()
    date_range = pd.date_range(start=first_moment, end=training_time_limit, freq=freq)
    rng = check_random_state(random_state)
    out_of_space_mask = pipe(train_data, lambda df: df[df[time_column] > date_range[-1]], lambda df: df[space_column].unique(), lambda array: rng.choice(array, int(len(array) * space_hold_percentage), replace=False), lambda held_space: train_data[space_column].isin(held_space))
    training_time_limit_dt = datetime.strptime(training_time_limit, '%Y-%m-%d') + holdout_gap
    test_time = train_data[(train_data[time_column] > training_time_limit_dt) & out_of_space_mask][time_column]

    def date_filter_fn(date: Any) -> pd.DataFrame:
        return train_data[(train_data[time_column] <= date) & ~out_of_space_mask]
    folds = _get_lc_folds(date_range, date_filter_fn, test_time, time_column, min_samples)
    logs = list(map(_log_time_fold, folds))
    folds_indexes = _lc_fold_to_indexes(folds)
    return (folds_indexes, logs)
time_and_space_learning_curve_splitter.__doc__ += splitter_return_docstring

@curry
def time_learning_curve_splitter(train_data: pd.DataFrame, training_time_limit: str, time_column: str, freq: str = 'M', holdout_gap: timedelta = timedelta(days=0), min_samples: int = 1000) -> Tuple[List[Tuple[pd.Index, List[pd.Index]]], List[Dict[str, Any]]]:
    """
    Splits the data into temporal buckets given by the specified frequency.

    Uses a fixed out-of-ID and time hold out set for every fold.
    Training size increases per fold, with more recent data being added in each fold.
    Useful for learning curve validation, that is, for seeing how hold out performance
    increases as the training size increases with more recent data.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split for learning curve estimation.

    training_time_limit : str
        The Date String for the end of the testing period. Should be of the same
        format as `time_column`.

    time_column : str
        The name of the Date column of `train_data`.

    freq : str
        The temporal frequency.
        See: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    holdout_gap: datetime.timedelta
        Timedelta of the gap between the end of the training period and the start of the validation period.

    min_samples : int
        The minimum number of samples required in the split to keep the split.
    """
    train_data = train_data.reset_index()
    first_moment = train_data[time_column].min()
    date_range = pd.date_range(start=first_moment, end=training_time_limit, freq=freq)
    effective_training_time_end = date_range[-1]
    test_time = train_data[train_data[time_column] > effective_training_time_end + holdout_gap][time_column]

    def date_filter_fn(date: Any) -> pd.DataFrame:
        return train_data[train_data[time_column] <= date]
    folds = _get_lc_folds(date_range, date_filter_fn, test_time, time_column, min_samples)
    logs = list(map(_log_time_fold, folds))
    folds_indexes = _lc_fold_to_indexes(folds)
    return (folds_indexes, logs)
time_learning_curve_splitter.__doc__ += splitter_return_docstring

@curry
def reverse_time_learning_curve_splitter(train_data: pd.DataFrame, time_column: str, training_time_limit: str, lower_time_limit: Optional[str] = None, freq: str = 'MS', holdout_gap: timedelta = timedelta(days=0), min_samples: int = 1000) -> Tuple[List[Tuple[pd.Index, List[pd.Index]]], List[Dict[str, Any]]]:
    """
    Splits the data into temporal buckets given by the specified frequency.
    Uses a fixed out-of-ID and time hold out set for every fold.
    Training size increases per fold, with less recent data being added in each fold.
    Useful for inverse learning curve validation, that is, for seeing how hold out
    performance increases as the training size increases with less recent data.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split into inverse learning curve estimation.

    time_column : str
        The name of the Date column of `train_data`.

    training_time_limit : str
        The Date String for the end of the training period. Should be of the same
        format as `time_column`.

    lower_time_limit : str
        A Date String for the begining of the training period. This allows limiting
        the learning curve from bellow, avoiding heavy computation with very old data.

    freq : str
        The temporal frequency.
        See: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    holdout_gap: datetime.timedelta
        Timedelta of the gap between the end of the training period and the start of the validation period.

    min_samples : int
        The minimum number of samples required in the split to keep the split.
    """
    train_data = train_data.reset_index()
    first_moment = lower_time_limit if lower_time_limit else train_data[time_column].min()
    date_range = pd.date_range(start=first_moment, end=training_time_limit, freq=freq)
    effective_training_time_end = date_range[-1]
    train_range = train_data[train_data[time_column] <= effective_training_time_end]
    test_time = train_data[train_data[time_column] > effective_training_time_end + holdout_gap][time_column]

    def date_filter_fn(date: Any) -> pd.DataFrame:
        return train_range.loc[train_data[time_column] >= date]
    folds = _get_lc_folds(date_range[::-1], date_filter_fn, test_time, time_column, min_samples)
    logs = list(map(_log_time_fold, folds))
    folds_indexes = _lc_fold_to_indexes(folds)
    return (folds_indexes, logs)
reverse_time_learning_curve_splitter.__doc__ += splitter_return_docstring

@curry
def spatial_learning_curve_splitter(train_data: pd.DataFrame, space_column: str, time_column: str, training_limit: Union[datetime, str], holdout_gap: timedelta = timedelta(days=0), train_percentages: Union[List[float], Tuple[float, ...]] = (0.25, 0.5, 0.75, 1.0), random_state: Optional[int] = None) -> Tuple[List[Tuple[pd.Index, List[pd.Index]]], List[Dict[str, Any]]]: