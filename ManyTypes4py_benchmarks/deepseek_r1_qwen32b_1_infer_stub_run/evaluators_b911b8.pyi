import warnings
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union
from pandas import DataFrame
from numpy import float64
from scipy.optimize import OptimizeResult
from sklearn.linear_model import LogisticRegression
from fklearn.types import EvalFnType, EvalReturnType, UncurriedEvalFnType

def generic_sklearn_evaluator(name_prefix: str, sklearn_metric: Callable) -> EvalFnType:
    ...

@curry
def auc_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def roc_auc_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def pr_auc_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def precision_evaluator(test_data: DataFrame, threshold: float = 0.5, prediction_column: str = 'prediction', target_column: str = 'target', weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def recall_evaluator(test_data: DataFrame, threshold: float = 0.5, prediction_column: str = 'prediction', target_column: str = 'target', weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def fbeta_score_evaluator(test_data: DataFrame, threshold: float = 0.5, beta: float = 1.0, prediction_column: str = 'prediction', target_column: str = 'target', weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def logloss_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def brier_score_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def expected_calibration_error_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', eval_name: Optional[str] = None, n_bins: int = 100, bin_choice: Literal['count', 'prob'] = 'count') -> EvalReturnType:
    ...

@curry
def r2_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def mse_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def mean_prediction_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def correlation_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def linear_coefficient_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def spearman_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def ndcg_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', k: Optional[int] = None, exponential_gain: bool = True, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def combined_evaluators(test_data: DataFrame, evaluators: Iterable[Callable]) -> EvalReturnType:
    ...

@curry
def split_evaluator(test_data: DataFrame, eval_fn: EvalFnType, split_col: str, split_values: Optional[List[Any]] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def temporal_split_evaluator(test_data: DataFrame, eval_fn: EvalFnType, time_col: str, time_format: str = '%Y-%m', split_values: Optional[List[str]] = None, eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def permutation_evaluator(test_data: DataFrame, predict_fn: Callable[[DataFrame], DataFrame], eval_fn: EvalFnType, baseline: bool = True, features: Optional[List[str]] = None, shuffle_all_at_once: bool = False, random_state: Optional[int] = None) -> EvalReturnType:
    ...

@curry
def hash_evaluator(test_data: DataFrame, hash_columns: Optional[List[str]] = None, eval_name: Optional[str] = None, consider_index: bool = False) -> EvalReturnType:
    ...

@curry
def exponential_coefficient_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', eval_name: Optional[str] = None) -> EvalReturnType:
    ...

@curry
def logistic_coefficient_evaluator(test_data: DataFrame, prediction_column: str = 'prediction', target_column: str = 'target', eval_name: Optional[str] = None) -> EvalReturnType:
    ...