from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd
from fklearn.types import EvalFnType, EvalReturnType, PredictFnType, UncurriedEvalFnType

def generic_sklearn_evaluator(
    name_prefix: str, sklearn_metric: Callable
) -> Callable[..., Dict[str, Any]]: ...

@EvalFnType
def auc_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def roc_auc_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def pr_auc_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def precision_evaluator(
    test_data: pd.DataFrame,
    threshold: float = 0.5,
    prediction_column: str = "prediction",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def recall_evaluator(
    test_data: pd.DataFrame,
    threshold: float = 0.5,
    prediction_column: str = "prediction",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def fbeta_score_evaluator(
    test_data: pd.DataFrame,
    threshold: float = 0.5,
    beta: float = 1.0,
    prediction_column: str = "prediction",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def logloss_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def brier_score_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def expected_calibration_error_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    eval_name: Optional[str] = None,
    n_bins: int = 100,
    bin_choice: str = "count",
) -> Dict[str, float]: ...

@EvalFnType
def r2_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def mse_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def mean_prediction_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def correlation_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def linear_coefficient_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def spearman_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def ndcg_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    k: Optional[int] = None,
    exponential_gain: bool = True,
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def combined_evaluators(
    test_data: pd.DataFrame,
    evaluators: List[Callable[[pd.DataFrame], Dict[str, Any]]],
) -> Dict[str, Any]: ...

@EvalFnType
def split_evaluator(
    test_data: pd.DataFrame,
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    split_col: str,
    split_values: Optional[Any] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, Any]: ...

@EvalFnType
def temporal_split_evaluator(
    test_data: pd.DataFrame,
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    time_col: str,
    time_format: str = "%Y-%m",
    split_values: Optional[Any] = None,
    eval_name: Optional[str] = None,
) -> Dict[str, Any]: ...

@EvalFnType
def permutation_evaluator(
    test_data: pd.DataFrame,
    predict_fn: PredictFnType,
    eval_fn: Callable[[pd.DataFrame], Dict[str, Any]],
    baseline: bool = True,
    features: Optional[List[str]] = None,
    shuffle_all_at_once: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]: ...

@EvalFnType
def hash_evaluator(
    test_data: pd.DataFrame,
    hash_columns: Optional[List[str]] = None,
    eval_name: Optional[str] = None,
    consider_index: bool = False,
) -> Dict[str, Any]: ...

@EvalFnType
def exponential_coefficient_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...

@EvalFnType
def logistic_coefficient_evaluator(
    test_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: str = "target",
    eval_name: Optional[str] = None,
) -> Dict[str, float]: ...