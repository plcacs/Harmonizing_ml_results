from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TypeVar,
    overload
)
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions.empirical_distribution import ECDF
from toolz import curry
from fklearn.types import LearnerReturnType, LearnerLogType

T = TypeVar('T')
U = TypeVar('U')

@curry
def selector(
    df: DataFrame,
    training_columns: List[str],
    predict_columns: Optional[List[str]] = None
) -> LearnerReturnType: ...

@curry
def capper(
    df: DataFrame,
    columns_to_cap: List[str],
    precomputed_caps: Optional[Dict[str, float]] = None
) -> LearnerReturnType: ...

@curry
def floorer(
    df: DataFrame,
    columns_to_floor: List[str],
    precomputed_floors: Optional[Dict[str, float]] = None
) -> LearnerReturnType: ...

@curry
def ecdfer(
    df: DataFrame,
    ascending: bool = True,
    prediction_column: str = "prediction",
    ecdf_column: str = "prediction_ecdf",
    max_range: int = 1000
) -> LearnerReturnType: ...

@curry
def discrete_ecdfer(
    df: DataFrame,
    ascending: bool = True,
    prediction_column: str = "prediction",
    ecdf_column: str = "prediction_ecdf",
    max_range: int = 1000,
    round_method: Callable[[float], int] = int
) -> LearnerReturnType: ...

@curry
def prediction_ranger(
    df: DataFrame,
    prediction_min: float,
    prediction_max: float,
    prediction_column: str = "prediction"
) -> LearnerReturnType: ...

def apply_replacements(
    df: DataFrame,
    columns: List[str],
    vec: Dict[str, Dict[Any, Any]],
    replace_unseen: Any
) -> DataFrame: ...

@curry
def value_mapper(
    df: DataFrame,
    value_maps: Dict[str, Dict[Any, Any]],
    ignore_unseen: bool = True,
    replace_unseen_to: Any = np.nan
) -> LearnerReturnType: ...

@curry
def truncate_categorical(
    df: DataFrame,
    columns_to_truncate: List[str],
    percentile: float,
    replacement: Union[int, str, float] = -9999,
    replace_unseen: Union[int, str, float] = -9999,
    store_mapping: bool = False
) -> LearnerReturnType: ...

@curry
def rank_categorical(
    df: DataFrame,
    columns_to_rank: List[str],
    replace_unseen: Union[int, str, float] = np.nan,
    store_mapping: bool = False
) -> LearnerReturnType: ...

@curry
def count_categorizer(
    df: DataFrame,
    columns_to_categorize: List[str],
    replace_unseen: int = -1,
    store_mapping: bool = False
) -> LearnerReturnType: ...

@curry
def label_categorizer(
    df: DataFrame,
    columns_to_categorize: List[str],
    replace_unseen: Union[int, str, float] = np.nan,
    store_mapping: bool = False
) -> LearnerReturnType: ...

@curry
def quantile_biner(
    df: DataFrame,
    columns_to_bin: List[str],
    q: Union[int, List[float]] = 4,
    right: bool = False
) -> LearnerReturnType: ...

@curry
def onehot_categorizer(
    df: DataFrame,
    columns_to_categorize: List[str],
    hardcode_nans: bool = False,
    drop_first_column: bool = False,
    store_mapping: bool = False
) -> LearnerReturnType: ...

@curry
def target_categorizer(
    df: DataFrame,
    columns_to_categorize: List[str],
    target_column: str,
    smoothing: float = 1.0,
    ignore_unseen: bool = True,
    store_mapping: bool = False
) -> LearnerReturnType: ...

@curry
def standard_scaler(
    df: DataFrame,
    columns_to_scale: List[str]
) -> LearnerReturnType: ...

@curry
def custom_transformer(
    df: DataFrame,
    columns_to_transform: List[str],
    transformation_function: Callable[[DataFrame], DataFrame],
    is_vectorized: bool = False
) -> LearnerReturnType: ...

@curry
def null_injector(
    df: DataFrame,
    proportion: float,
    columns_to_inject: Optional[List[str]] = None,
    groups: Optional[List[List[str]]] = None,
    seed: int = 1
) -> LearnerReturnType: ...

@curry
def missing_warner(
    df: DataFrame,
    cols_list: List[str],
    new_column_name: str = "has_unexpected_missing",
    detailed_warning: bool = False,
    detailed_column_name: Optional[str] = None
) -> LearnerReturnType: ...