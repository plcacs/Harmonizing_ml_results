from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import numpy as np
import pandas as pd
from fklearn.types import EvalFnType, EvalReturnType, PredictFnType, UncurriedEvalFnType

def generic_sklearn_evaluator(
    name_prefix: str,
    sklearn_metric: Callable[..., float],
) -> Callable[..., Dict[str, float]]: ...

@__builtins__  # type: ignore[misc]
def auc_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def roc_auc_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def pr_auc_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def precision_evaluator(
    test_data: pd.DataFrame,
    threshold: float = ...,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def recall_evaluator(
    test_data: pd.DataFrame,
    threshold: float = ...,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def fbeta_score_evaluator(
    test_data: pd.DataFrame,
    threshold: float = ...,
    beta: float = ...,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def logloss_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def brier_score_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def expected_calibration_error_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
    n_bins: int = ...,
    bin_choice: str = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def r2_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def mse_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def mean_prediction_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def correlation_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def linear_coefficient_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def spearman_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def ndcg_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    k: Optional[int] = ...,
    exponential_gain: bool = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def combined_evaluators(
    test_data: pd.DataFrame,
    evaluators: List[Callable[[pd.DataFrame], Dict[str, Any]]],
) -> Dict[str, Any]: ...

@__builtins__  # type: ignore[misc]
def split_evaluator(
    test_data: pd.DataFrame,
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    split_col: str,
    split_values: Optional[Any] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, Any]: ...

@__builtins__  # type: ignore[misc]
def temporal_split_evaluator(
    test_data: pd.DataFrame,
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    time_col: str,
    time_format: str = ...,
    split_values: Optional[Any] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, Any]: ...

@__builtins__  # type: ignore[misc]
def permutation_evaluator(
    test_data: pd.DataFrame,
    predict_fn: PredictFnType,
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    baseline: bool = ...,
    features: Optional[List[str]] = ...,
    shuffle_all_at_once: bool = ...,
    random_state: Optional[int] = ...,
) -> Dict[str, Any]: ...

@__builtins__  # type: ignore[misc]
def hash_evaluator(
    test_data: pd.DataFrame,
    hash_columns: Optional[List[str]] = ...,
    eval_name: Optional[str] = ...,
    consider_index: bool = ...,
) -> Dict[str, int]: ...

@__builtins__  # type: ignore[misc]
def exponential_coefficient_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

@__builtins__  # type: ignore[misc]
def logistic_coefficient_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...
```

Wait, I need to fix the stub. The `@curry` decorator from `toolz` makes functions callable in a curried fashion. Let me produce a proper stub:

```python
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd
from fklearn.types import PredictFnType

def generic_sklearn_evaluator(
    name_prefix: str,
    sklearn_metric: Callable[..., float],
) -> Callable[..., Dict[str, float]]: ...

def auc_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def roc_auc_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def pr_auc_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def precision_evaluator(
    test_data: pd.DataFrame,
    threshold: float = ...,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def recall_evaluator(
    test_data: pd.DataFrame,
    threshold: float = ...,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def fbeta_score_evaluator(
    test_data: pd.DataFrame,
    threshold: float = ...,
    beta: float = ...,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def logloss_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def brier_score_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def expected_calibration_error_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
    n_bins: int = ...,
    bin_choice: str = ...,
) -> Dict[str, float]: ...

def r2_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def mse_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    weight_column: Optional[str] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def mean_prediction_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def correlation_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def linear_coefficient_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def spearman_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def ndcg_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    k: Optional[int] = ...,
    exponential_gain: bool = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def combined_evaluators(
    test_data: pd.DataFrame,
    evaluators: List[Callable[[pd.DataFrame], Dict[str, Any]]],
) -> Dict[str, Any]: ...

def split_evaluator(
    test_data: pd.DataFrame,
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    split_col: str,
    split_values: Optional[Any] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, Any]: ...

def temporal_split_evaluator(
    test_data: pd.DataFrame,
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    time_col: str,
    time_format: str = ...,
    split_values: Optional[Any] = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, Any]: ...

def permutation_evaluator(
    test_data: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    baseline: bool = ...,
    features: Optional[List[str]] = ...,
    shuffle_all_at_once: bool = ...,
    random_state: Optional[int] = ...,
) -> Dict[str, Any]: ...

def hash_evaluator(
    test_data: pd.DataFrame,
    hash_columns: Optional[List[str]] = ...,
    eval_name: Optional[str] = ...,
    consider_index: bool = ...,
) -> Dict[str, int]: ...

def exponential_coefficient_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...

def logistic_coefficient_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = ...,
    target_column: str = ...,
    eval_name: Optional[str] = ...,
) -> Dict[str, float]: ...