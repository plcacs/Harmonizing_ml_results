import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, List, Tuple, Union, Optional, Dict
from fklearn.types import SplitterReturnType

def _log_time_fold(time_fold: Tuple[pd.Series, pd.Series]) -> Dict[str, Union[datetime, int]]: ...

def _get_lc_folds(
    date_range: pd.DatetimeIndex,
    date_fold_filter_fn: Callable[[Any], pd.DataFrame],
    test_time: pd.Series,
    time_column: str,
    min_samples: int,
) -> List[Tuple[pd.Series, pd.Series]]: ...

def _get_sc_folds(
    date_range: pd.PeriodIndex,
    date_fold_filter_fn: Callable[[Any], pd.DataFrame],
    time_column: str,
    min_samples: int,
) -> List[pd.Series]: ...

def _get_sc_test_fold_idx_and_logs(
    test_data: pd.DataFrame,
    train_time: pd.Series,
    time_column: str,
    first_test_moment: datetime,
    last_test_moment: datetime,
    min_samples: int,
    freq: str,
) -> Tuple[List[Dict[str, Any]], List[List[pd.Index]]]: ...

def _lc_fold_to_indexes(folds: Iterable[Tuple[pd.Series, pd.Series]]) -> List[Tuple[pd.Index, List[pd.Index]]]: ...

@curry
def k_fold_splitter(
    train_data: pd.DataFrame,
    n_splits: int,
    random_state: Optional[int] = None,
    stratify_column: Optional[str] = None,
) -> SplitterReturnType: ...

@curry
def out_of_time_and_space_splitter(
    train_data: pd.DataFrame,
    n_splits: int,
    in_time_limit: Union[str, datetime],
    time_column: str,
    space_column: str,
    holdout_gap: timedelta = timedelta(days=0),
) -> SplitterReturnType: ...

@curry
def time_and_space_learning_curve_splitter(
    train_data: pd.DataFrame,
    training_time_limit: str,
    space_column: str,
    time_column: str,
    freq: str = 'M',
    space_hold_percentage: float = 0.5,
    holdout_gap: timedelta = timedelta(days=0),
    random_state: Optional[int] = None,
    min_samples: int = 1000,
) -> SplitterReturnType: ...

@curry
def time_learning_curve_splitter(
    train_data: pd.DataFrame,
    training_time_limit: str,
    time_column: str,
    freq: str = 'M',
    holdout_gap: timedelta = timedelta(days=0),
    min_samples: int = 1000,
) -> SplitterReturnType: ...

@curry
def reverse_time_learning_curve_splitter(
    train_data: pd.DataFrame,
    time_column: str,
    training_time_limit: str,
    lower_time_limit: Optional[str] = None,
    freq: str = 'MS',
    holdout_gap: timedelta = timedelta(days=0),
    min_samples: int = 1000,
) -> SplitterReturnType: ...

@curry
def spatial_learning_curve_splitter(
    train_data: pd.DataFrame,
    space_column: str,
    time_column: str,
    training_limit: Union[datetime, str],
    holdout_gap: timedelta = timedelta(days=0),
    train_percentages: Union[List[float], Tuple[float, ...]] = (0.25, 0.5, 0.75, 1.0),
    random_state: Optional[int] = None,
) -> SplitterReturnType: ...

@curry
def stability_curve_time_splitter(
    train_data: pd.DataFrame,
    training_time_limit: str,
    time_column: str,
    freq: str = 'M',
    min_samples: int = 1000,
) -> SplitterReturnType: ...

@curry
def stability_curve_time_in_space_splitter(
    train_data: pd.DataFrame,
    training_time_limit: str,
    space_column: str,
    time_column: str,
    freq: str = 'M',
    space_hold_percentage: float = 0.5,
    random_state: Optional[int] = None,
    min_samples: int = 1000,
) -> SplitterReturnType: ...

@curry
def stability_curve_time_space_splitter(
    train_data: pd.DataFrame,
    training_time_limit: str,
    space_column: str,
    time_column: str,
    freq: str = 'M',
    space_hold_percentage: float = 0.5,
    random_state: Optional[int] = None,
    min_samples: int = 1000,
) -> SplitterReturnType: ...

@curry
def forward_stability_curve_time_splitter(
    train_data: pd.DataFrame,
    training_time_start: Union[datetime, str],
    training_time_end: Union[datetime, str],
    time_column: str,
    holdout_gap: timedelta = timedelta(days=0),
    holdout_size: timedelta = timedelta(days=90),
    step: timedelta = timedelta(days=90),
    move_training_start_with_steps: bool = True,
) -> SplitterReturnType: ...

# Since @curry is used, these functions are technically wrapped. 
# In a .pyi file, we represent the final intended signature.
# The 'curry' decorator from toolz is a runtime implementation detail.
# We assume 'curry' is imported or available in the environment.
from toolz.curried import curry