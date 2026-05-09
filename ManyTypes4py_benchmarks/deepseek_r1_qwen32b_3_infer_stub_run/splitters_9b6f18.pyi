from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    Literal,
)
from pandas import DataFrame, Series, Index
from numpy import ndarray
from toolz.curried import Curry

def _log_time_fold(time_fold: Tuple[Series[datetime], Series[datetime]]) -> Dict[str, Union[datetime, int]]:
    ...

def _get_lc_folds(
    date_range: Iterable[datetime],
    date_fold_filter_fn: Callable[[datetime], DataFrame],
    test_time: Series[datetime],
    time_column: str,
    min_samples: int,
) -> List[Tuple[Series[datetime], Series[datetime]]]:
    ...

def _get_sc_folds(
    date_range: Iterable[datetime],
    date_filter_fn: Callable[[datetime], DataFrame],
    time_column: str,
    min_samples: int,
) -> List[Series[datetime]]:
    ...

def _get_sc_test_fold_idx_and_logs(
    test_data: DataFrame,
    train_time: Series[datetime],
    time_column: str,
    first_test_moment: datetime,
    last_test_moment: datetime,
    min_samples: int,
    freq: str,
) -> Tuple[List[Dict[str, Any]], List[List[Index]]]:
    ...

def _lc_fold_to_indexes(folds: List[Tuple[Series[datetime], Series[datetime]]]) -> List[Tuple[Index, List[Index]]]:
    ...

@Curry
def k_fold_splitter(
    train_data: DataFrame,
    n_splits: int,
    random_state: Optional[int] = None,
    stratify_column: Optional[str] = None,
) -> Tuple[List[Tuple[Index, List[Index]]], List[Dict[str, int]]]:
    ...

@Curry
def out_of_time_and_space_splitter(
    train_data: DataFrame,
    n_splits: int,
    in_time_limit: Union[str, datetime],
    time_column: str,
    space_column: str,
    holdout_gap: timedelta = timedelta(days=0),
) -> Tuple[List[Tuple[Index, List[Index]]], List[Dict[str, Union[datetime, int]]]]:
    ...

@Curry
def time_and_space_learning_curve_splitter(
    train_data: DataFrame,
    training_time_limit: str,
    space_column: str,
    time_column: str,
    freq: str = 'M',
    space_hold_percentage: float = 0.5,
    holdout_gap: timedelta = timedelta(days=0),
    random_state: Optional[int] = None,
    min_samples: int = 1000,
) -> Tuple[List[Tuple[Index, List[Index]]], List[Dict[str, Union[datetime, int]]]]:
    ...

@Curry
def time_learning_curve_splitter(
    train_data: DataFrame,
    training_time_limit: str,
    time_column: str,
    freq: str = 'M',
    holdout_gap: timedelta = timedelta(days=0),
    min_samples: int = 1000,
) -> Tuple[List[Tuple[Index, List[Index]]], List[Dict[str, Union[datetime, int]]]]:
    ...

@Curry
def reverse_time_learning_curve_splitter(
    train_data: DataFrame,
    time_column: str,
    training_time_limit: str,
    lower_time_limit: Optional[str] = None,
    freq: str = 'MS',
    holdout_gap: timedelta = timedelta(days=0),
    min_samples: int = 1000,
) -> Tuple[List[Tuple[Index, List[Index]]], List[Dict[str, Union[datetime, int]]]]:
    ...

@Curry
def spatial_learning_curve_splitter(
    train_data: DataFrame,
    space_column: str,
    time_column: str,
    training_limit: Union[datetime, str],
    holdout_gap: timedelta = timedelta(days=0),
    train_percentages: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    random_state: Optional[int] = None,
) -> Tuple[List[Tuple[Index, List[Index]]], List[Dict[str, Union[datetime, int, float]]]]:
    ...

@Curry
def stability_curve_time_splitter(
    train_data: DataFrame,
    training_time_limit: str,
    time_column: str,
    freq: str = 'M',
    min_samples: int = 1000,
) -> Tuple[List[Tuple[Index, List[Index]]], List[Dict[str, Union[datetime, int]]]]:
    ...

@Curry
def stability_curve_time_in_space_splitter(
    train_data: DataFrame,
    training_time_limit: str,
    space_column: str,
    time_column: str,
    freq: str = 'M',
    space_hold_percentage: float = 0.5,
    random_state: Optional[int] = None,
    min_samples: int = 1000,
) -> Tuple[List[Tuple[Index, List[Index]]], List[Dict[str, Union[datetime, int]]]]:
    ...

@Curry
def stability_curve_time_space_splitter(
    train_data: DataFrame,
    training_time_limit: str,
    space_column: str,
    time_column: str,
    freq: str = 'M',
    space_hold_percentage: float = 0.5,
    random_state: Optional[int] = None,
    min_samples: int = 1000,
) -> Tuple[List[Tuple[Index, List[Index]]], List[Dict[str, Union[datetime, int]]]]:
    ...

@Curry
def forward_stability_curve_time_splitter(
    train_data: DataFrame,
    training_time_start: Union[datetime, str],
    training_time_end: Union[datetime, str],
    time_column: str,
    holdout_gap: timedelta = timedelta(days=0),
    holdout_size: timedelta = timedelta(days=90),
    step: timedelta = timedelta(days=90),
    move_training_start_with_steps: bool = True,
) -> Tuple[List[Tuple[List[Index], List[List[Index]]]], List[Dict[str, Union[datetime, int]]]]:
    ...