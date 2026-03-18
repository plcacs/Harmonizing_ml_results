```python
from typing import Any, Callable, Dict, List, Optional, Union, overload
import numpy as np
import pandas as pd
from numpy import nan
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions.empirical_distribution import ECDF
from fklearn.types import LearnerReturnType, LearnerLogType

def selector(
    df: pd.DataFrame,
    training_columns: List[str],
    predict_columns: Optional[List[str]] = None
) -> LearnerReturnType: ...

def capper(
    df: pd.DataFrame,
    columns_to_cap: List[str],
    precomputed_caps: Optional[Dict[str, Any]] = None
) -> LearnerReturnType: ...

def floorer(
    df: pd.DataFrame,
    columns_to_floor: List[str],
    precomputed_floors: Optional[Dict[str, Any]] = None
) -> LearnerReturnType: ...

def ecdfer(
    df: pd.DataFrame,
    ascending: bool = True,
    prediction_column: str = "prediction",
    ecdf_column: str = "prediction_ecdf",
    max_range: int = 1000
) -> LearnerReturnType: ...

def discrete_ecdfer(
    df: pd.DataFrame,
    ascending: bool = True,
    prediction_column: str = "prediction",
    ecdf_column: str = "prediction_ecdf",
    max_range: int = 1000,
    round_method: Callable = int
) -> LearnerReturnType: ...

def prediction_ranger(
    df: pd.DataFrame,
    prediction_min: float,
    prediction_max: float,
    prediction_column: str = "prediction"
) -> LearnerReturnType: ...

def apply_replacements(
    df: pd.DataFrame,
    columns: List[str],
    vec: Dict[str, Dict[Any, Any]],
    replace_unseen: Any
) -> pd.DataFrame: ...

def value_mapper(
    df: pd.DataFrame,
    value_maps: Dict[str, Dict[Any, Any]],
    ignore_unseen: bool = True,
    replace_unseen_to: Any = nan
) -> LearnerReturnType: ...

def truncate_categorical(
    df: pd.DataFrame,
    columns_to_truncate: List[str],
    percentile: float,
    replacement: Any = -9999,
    replace_unseen: Any = -9999,
    store_mapping: bool = False
) -> LearnerReturnType: ...

def rank_categorical(
    df: pd.DataFrame,
    columns_to_rank: List[str],
    replace_unseen: Any = nan,
    store_mapping: bool = False
) -> LearnerReturnType: ...

def count_categorizer(
    df: pd.DataFrame,
    columns_to_categorize: List[str],
    replace_unseen: int = -1,
    store_mapping: bool = False
) -> LearnerReturnType: ...

def label_categorizer(
    df: pd.DataFrame,
    columns_to_categorize: List[str],
    replace_unseen: Any = nan,
    store_mapping: bool = False
) -> LearnerReturnType: ...

def quantile_biner(
    df: pd.DataFrame,
    columns_to_bin: List[str],
    q: Union[int, List[float]] = 4,
    right: bool = False
) -> LearnerReturnType: ...

def onehot_categorizer(
    df: pd.DataFrame,
    columns_to_categorize: List[str],
    hardcode_nans: bool = False,
    drop_first_column: bool = False,
    store_mapping: bool = False
) -> LearnerReturnType: ...

def target_categorizer(
    df: pd.DataFrame,
    columns_to_categorize: List[str],
    target_column: str,
    smoothing: float = 1.0,
    ignore_unseen: bool = True,
    store_mapping: bool = False
) -> LearnerReturnType: ...

def standard_scaler(
    df: pd.DataFrame,
    columns_to_scale: List[str]
) -> LearnerReturnType: ...

def custom_transformer(
    df: pd.DataFrame,
    columns_to_transform: List[str],
    transformation_function: Callable[[pd.DataFrame], pd.DataFrame],
    is_vectorized: bool = False
) -> LearnerReturnType: ...

def null_injector(
    df: pd.DataFrame,
    proportion: float,
    columns_to_inject: Optional[List[str]] = None,
    groups: Optional[List[List[str]]] = None,
    seed: int = 1
) -> LearnerReturnType: ...

def missing_warner(
    df: pd.DataFrame,
    cols_list: List[str],
    new_column_name: str = "has_unexpected_missing",
    detailed_warning: bool = False,
    detailed_column_name: Optional[str] = None
) -> LearnerReturnType: ...
```