from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol, Union
from pandas import DataFrame
from fklearn.types import EvalFnType, EvalReturnType, PredictFnType, UncurriedEvalFnType

class _EvalCallable(Protocol):
    def __call__(
        self,
        test_data: DataFrame,
        prediction_column: str = ...,
        target_column: str = ...,
        weight_column: Optional[str] = ...,
        eval_name: Optional[str] = ...,
        **kwargs: Any
    ) -> Dict[str, float]: ...

def generic_sklearn_evaluator(name_prefix: str, sklearn_metric: Callable[..., float]) -> _EvalCallable: ...
def auc_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def roc_auc_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def pr_auc_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def precision_evaluator(
    test_data: DataFrame,
    threshold: float = 0.5,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def recall_evaluator(
    test_data: DataFrame,
    threshold: float = 0.5,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def fbeta_score_evaluator(
    test_data: DataFrame,
    threshold: float = 0.5,
    beta: float = 1.0,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def logloss_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def brier_score_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def expected_calibration_error_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    eval_name: Optional[str] = None,
    n_bins: int = 100,
    bin_choice: str = 'count'
) -> Dict[str, float]: ...
def r2_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def mse_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    weight_column: Optional[str] = None,
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def mean_prediction_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def correlation_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def linear_coefficient_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def spearman_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def ndcg_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    k: Optional[int] = None,
    exponential_gain: bool = True,
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def combined_evaluators(
    test_data: DataFrame,
    evaluators: Iterable[Callable[[DataFrame], Dict[str, Any]]]
) -> Dict[str, Any]: ...
def split_evaluator(
    test_data: DataFrame,
    eval_fn: Callable[[DataFrame], Dict[str, float]],
    split_col: str,
    split_values: Optional[Iterable[object]] = None,
    eval_name: Optional[str] = None
) -> Dict[str, Dict[str, float]]: ...
def temporal_split_evaluator(
    test_data: DataFrame,
    eval_fn: Callable[[DataFrame], Dict[str, float]],
    time_col: str,
    time_format: str = '%Y-%m',
    split_values: Optional[Iterable[str]] = None,
    eval_name: Optional[str] = None
) -> Dict[str, Dict[str, float]]: ...
def permutation_evaluator(
    test_data: DataFrame,
    predict_fn: Callable[[DataFrame], DataFrame],
    eval_fn: Callable[[DataFrame], Dict[str, float]],
    baseline: bool = True,
    features: Optional[Iterable[str]] = None,
    shuffle_all_at_once: bool = False,
    random_state: Optional[int] = None
) -> Dict[str, Union[Mapping[str, float], Mapping[str, Mapping[str, float]]]]: ...
def hash_evaluator(
    test_data: DataFrame,
    hash_columns: Optional[Iterable[str]] = None,
    eval_name: Optional[str] = None,
    consider_index: bool = False
) -> Dict[str, int]: ...
def exponential_coefficient_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...
def logistic_coefficient_evaluator(
    test_data: DataFrame,
    prediction_column: str = 'prediction',
    target_column: str = 'target',
    eval_name: Optional[str] = None
) -> Dict[str, float]: ...