from typing import Any, Callable, Dict, List, Union, Optional, Tuple, overload
import numpy as np
import pandas as pd
from fklearn.types import LearnerReturnType, LearnerLogType

# The functions are decorated with @curry, which means they can be called 
# with a subset of arguments and return a function. 
# However, for .pyi stubs, we typically represent the full signature.

def selector(
    df: pd.DataFrame, 
    training_columns: List[str], 
    predict_columns: Optional[List[str]] = None
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def capper(
    df: pd.DataFrame, 
    columns_to_cap: List[str], 
    precomputed_caps: Optional[Dict[str, float]] = None
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def floorer(
    df: pd.DataFrame, 
    columns_to_floor: List[str], 
    precomputed_floors: Optional[Dict[str, float]] = None
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def ecdfer(
    df: pd.DataFrame, 
    ascending: bool = True, 
    prediction_column: str = 'prediction', 
    ecdf_column: str = 'prediction_ecdf', 
    max_range: int = 1000
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def discrete_ecdfer(
    df: pd.DataFrame, 
    ascending: bool = True, 
    prediction_column: str = 'prediction', 
    ecdf_column: str = 'prediction_ecdf', 
    max_range: int = 1000, 
    round_method: Callable[[Any], Any] = int
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def prediction_ranger(
    df: pd.DataFrame, 
    prediction_min: float, 
    prediction_max: float, 
    prediction_column: str = 'prediction'
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

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
    replace_unseen_to: Any = np.nan
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def truncate_categorical(
    df: pd.DataFrame, 
    columns_to_truncate: List[str], 
    percentile: float, 
    replacement: Union[int, str, float, Any] = -9999, 
    replace_unseen: Union[int, str, float, Any] = -9999, 
    store_mapping: bool = False
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def rank_categorical(
    df: pd.DataFrame, 
    columns_to_rank: List[str], 
    replace_unseen: Any = np.nan, 
    store_mapping: bool = False
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def count_categorizer(
    df: pd.DataFrame, 
    columns_to_categorize: List[str], 
    replace_unseen: int = -1, 
    store_mapping: bool = False
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def label_categorizer(
    df: pd.DataFrame, 
    columns_to_categorize: List[str], 
    replace_unseen: Any = np.nan, 
    store_mapping: bool = False
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def quantile_biner(
    df: pd.DataFrame, 
    columns_to_bin: List[str], 
    q: Union[int, List[float]] = 4, 
    right: bool = False
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def onehot_categorizer(
    df: pd.DataFrame, 
    columns_to_categorize: List[str], 
    hardcode_nans: bool = False, 
    drop_first_column: bool = False, 
    store_mapping: bool = False
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def target_categorizer(
    df: pd.DataFrame, 
    columns_to_categorize: List[str], 
    target_column: str, 
    smoothing: float = 1.0, 
    ignore_unseen: bool = True, 
    store_mapping: bool = False
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def standard_scaler(
    df: pd.DataFrame, 
    columns_to_scale: List[str]
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def custom_transformer(
    df: pd.DataFrame, 
    columns_to_transform: List[str], 
    transformation_function: Callable[[Any], Any], 
    is_vectorized: bool = False
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def null_injector(
    df: pd.DataFrame, 
    proportion: float, 
    columns_to_inject: Optional[List[str]] = None, 
    groups: Optional[List[List[str]]] = None, 
    seed: int = 1
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...

def missing_warner(
    df: pd.DataFrame, 
    cols_list: List[str], 
    new_column_name: str = 'has_unexpected_missing', 
    detailed_warning: bool = False, 
    detailed_column_name: Optional[str] = None
) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, LearnerLogType]: ...