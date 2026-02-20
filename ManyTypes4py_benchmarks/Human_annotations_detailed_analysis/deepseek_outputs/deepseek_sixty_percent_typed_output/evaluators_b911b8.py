import warnings
from typing import Any, Callable, Iterable, List, Optional, Union, Dict

import numpy as np
import pandas as pd
import toolz as fp
from pandas.util import hash_pandas_object
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             fbeta_score, log_loss, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from toolz import curry, last, first
from scipy import optimize
from sklearn.linear_model import LogisticRegression

from fklearn.types import (EvalFnType, EvalReturnType, PredictFnType,
                           UncurriedEvalFnType)


def generic_sklearn_evaluator(name_prefix: str, sklearn_metric: Callable[..., float]) -> UncurriedEvalFnType:
    """
    Returns an evaluator build from a metric from sklearn.metrics

    Parameters
    ----------
    name_prefix: str
        The default name of the evaluator will be name_prefix + target_column.

    sklearn_metric: Callable
        Metric function from sklearn.metrics. It should take as parameters y_true, y_score, kwargs.

    Returns
    ----------
    eval_fn: Callable
       An evaluator function that uses the provided metric
    """

    def p(test_data: pd.DataFrame,
          prediction_column: str = "prediction",
          target_column: str = "target",
          weight_column: Optional[str] = None,
          eval_name: Optional[str] = None,
          **kwargs: Any) -> EvalReturnType:
        try:
            score = sklearn_metric(test_data[target_column],
                                   test_data[prediction_column],
                                   sample_weight=None if weight_column is None else test_data[weight_column],
                                   **kwargs)
        except ValueError:
            # this might happen if there's only one class in the fold
            score = np.nan

        if eval_name is None:
            eval_name = name_prefix + target_column

        return {eval_name: score}

    return p


@curry
def auc_evaluator(test_data: pd.DataFrame,
                  prediction_column: str = "prediction",
                  target_column: str = "target",
                  weight_column: Optional[str] = None,
                  eval_name: Optional[str] = None) -> EvalReturnType:
    """
    Computes the ROC AUC score, given true label and prediction scores.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction scores.

    target_column : String
        The name of the column in `test_data` with the binary target.

    weight_column : String (default=None)
        The name of the column in `test_data` with the sample weights.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the ROC AUC Score
    """

    warnings.warn("The method `auc_evaluator` will be renamed to `roc_auc_evaluator` in the next major release 2.0.0."
                  " Please use `roc_auc_evaluator` instead of `auc_evaluator` for Area Under the Curve of the"
                  " Receiver Operating Characteristics curve.")

    return roc_auc_evaluator(test_data, prediction_column, target_column, weight_column, eval_name)


@curry
def roc_auc_evaluator(test_data: pd.DataFrame,
                      prediction_column: str = "prediction",
                      target_column: str = "target",
                      weight_column: Optional[str] = None,
                      eval_name: Optional[str] = None) -> EvalReturnType:
    """
    Computes the ROC AUC score, given true label and prediction scores.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction scores.

    target_column : String
        The name of the column in `test_data` with the binary target.

    weight_column : String (default=None)
        The name of the column in `test_data` with the sample weights.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the ROC AUC Score
    """

    eval_fn = generic_sklearn_evaluator("roc_auc_evaluator__", roc_auc_score)
    eval_data = test_data.assign(**{target_column: lambda df: df[target_column].astype(int)})

    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)


@curry
def pr_auc_evaluator(test_data: pd.DataFrame,
                     prediction_column: str = "prediction",
                     target_column: str = "target",
                     weight_column: Optional[str] = None,
                     eval_name: Optional[str] = None) -> EvalReturnType:
    """
    Computes the PR AUC score, given true label and prediction scores.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction scores.

    target_column : String
        The name of the column in `test_data` with the binary target.

    weight_column : String (default=None)
        The name of the column in `test_data` with the sample weights.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    A log-like dictionary with the PR AUC Score
    """

    eval_fn = generic_sklearn_evaluator("pr_auc_evaluator__", average_precision_score)
    eval_data = test_data.assign(**{target_column: lambda df: df[target_column].astype(int)})

    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)


@curry
def precision_evaluator(test_data: pd.DataFrame,
                        threshold: float = 0.5,
                        prediction_column: str = "prediction",
                        target_column: str = "target",
                        weight_column: Optional[str] = None,
                        eval_name: Optional[str] = None) -> EvalReturnType:
    """
    Computes the precision score, given true label and prediction scores.

    Parameters
    ----------
    test_data : pandas.DataFrame
        A Pandas' DataFrame with target and prediction scores.

    threshold : float
        A threshold for the prediction column above which samples
         will be classified as 1

    prediction_column : str
        The name of the column in `test_data` with the prediction scores.

    target_column : str
        The name of the column in `test_data` with the binary target.

    weight_column : String (default=None)
        The name of the column in `test_data` with the sample weights.

    eval_name : str, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the Precision Score
    """
    eval_fn = generic_sklearn_evaluator("precision_evaluator__", precision_score)
    eval_data = test_data.assign(**{prediction_column: (test_data[prediction_column] > threshold).astype(int)})

    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)


@curry
def recall_evaluator(test_data: pd.DataFrame,
                     threshold: float = 0.5,
                     prediction_column: str = "prediction",
                     target_column: str = "target",
                     weight_column: Optional[str] = None,
                     eval_name: Optional[str] = None) -> EvalReturnType:
    """
    Computes the recall score, given true label and prediction scores.

    Parameters
    ----------

    test_data : pandas.DataFrame
        A Pandas' DataFrame with target and prediction scores.

    threshold : float
        A threshold for the prediction column above which samples
         will be classified as 1

    prediction_column : str
        The name of the column in `test_data` with the prediction scores.

    target_column : str
        The name of the column in `test_data` with the binary target.

    weight_column : String (default=None)
        The name of the column in `test_data` with the sample weights.

    eval_name : str, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the Precision Score
    """

    eval_data = test_data.assign(**{prediction_column: (test_data[prediction_column] > threshold).astype(int)})
    eval_fn = generic_sklearn_evaluator("recall_evaluator__", recall_score)

    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)


@curry
def fbeta_score_evaluator(test_data: pd.DataFrame,
                          threshold: float = 0.5,
                          beta: float = 1.0,
                          prediction_column: str = "prediction",
                          target_column: str = "target",
                          weight_column: Optional[str] = None,
                          eval_name: Optional[str] = None) -> EvalReturnType:
    """
    Computes the F-beta score, given true label and prediction scores.

    Parameters
    ----------

    test_data : pandas.DataFrame
        A Pandas' DataFrame with target and prediction scores.

    threshold : float
        A threshold for the prediction column above which samples
         will be classified as 1

    beta : float
        The beta parameter determines the weight of precision in the combined score.
        beta < 1 lends more weight to precision, while beta > 1 favors recall
        (beta -> 0 considers only precision, beta -> inf only recall).

    prediction_column : str
        The name of the column in `test_data` with the prediction scores.

    target_column : str
        The name of the column in `test_data` with the binary target.

    weight_column : String (default=None)
        The name of the column in `test_data` with the sample weights.

    eval_name : str, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the Precision Score
    """

    eval_data = test_data.assign(**{prediction_column: (test_data[prediction_column] > threshold).astype(int)})
    eval_fn = generic_sklearn_evaluator("fbeta_evaluator__", fbeta_score)

    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name, beta=beta)


@curry
def logloss_evaluator(test_data: pd.DataFrame,
                      prediction_column: str = "prediction",
                      target_column: str = "target",
                      weight_column: Optional[str] = None,
                      eval_name: Optional[str] = None) -> EvalReturnType:
    """
    Computes the logloss score, given true label and prediction scores.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction scores.

    target_column : String
        The name of the column in `test_data` with the binary target.

    weight_column : String (default=None)
        The name of the column in `test_data` with the sample weights.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the logloss score.
    """

    eval_fn = generic_sklearn_evaluator("logloss_evaluator__", log_loss)
    eval_data = test_data.assign(**{target_column: lambda df: df[target_column].astype(int)})

    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)


@curry
def brier_score_evaluator(test_data: pd.DataFrame,
                          prediction_column: str = "prediction",
                          target_column: str = "target",
                          weight_column: Optional[str] = None,
                          eval_name: Optional[str] = None) -> EvalReturnType:
    """
    Computes the Brier score, given true label and prediction scores.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction scores.

    target_column : String
        The name of the column in `test_data` with the binary target.

    weight_column : String (default=None)
        The name of the column in `test_data` with the sample weights.

    eval_name : String, optional (default=None)
        The name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the Brier score.
    """

    eval_fn = generic_sklearn_evaluator("brier_score_evaluator__", brier_score_loss)
    eval_data = test_data.assign(**{target_column: lambda df: df[target_column].astype(int)})

    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)


@curry
def expected_calibration_error_evaluator(test_data: pd.DataFrame,
                                         prediction_column: str = "prediction",
                                         target_column: str = "target",
                                         eval_name: Optional[str] = None,
                                         n_bins: int = 100,
                                         bin_choice: str = "count") -> EvalReturnType:
    """
    Computes the expected calibration error (ECE), given true label and prediction scores.
    See "On Calibration of Modern Neural Networks"(https://arxiv.org/abs/1706.04599) for more information.

    The ECE is the distance between the actuals observed frequency and the predicted probabilities,
    for a given choice of bins.

    Perfect calibration results in a score of 0.

    For example, if for the bin [0, 0.1] we have the three data points:
      1. prediction: 0.1, actual: 0
      2. prediction: 0.05, actual: 1
      3. prediction: 0.0, actual 0

    Then the predicted average is (0.1 + 0.05 + 0.00)/3 = 0.05, and the empirical frequency is (0 + 1 + 0)/3 = 1/3.
    Therefore, the distance for this bin is::

        |1/3 - 0.05| ~= 0.28.

    Graphical intuition::

        Actuals (empirical frequency between 0 and 1)
        |     *
        |   *
        | *
         ______ Predictions (probabilties between 0 and 1)

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction scores.

    target_column : String
        The name of the column in `test_data` with the binary target.

    eval_name : String, optional (default=None)
        The name of the evaluator as it will appear in the logs.

    n_bins: Int (default=100)
        The number of bins.
        This is a trade-off between the number of points in each bin and the probability range they span.
        You want a small enough range that still contains a significant number of points for the distance to work.

    bin_choice: String (default="count")
        Two possibilities:
        "count" for equally populated bins (e.g. uses `pandas.qcut` for the bins)
        "prob" for equally spaced probabilities (e.g. uses `pandas.cut` for the bins),
        with distance weighed by the number of samples in each bin.

    Returns
    -------
    log: dict
       A log-like dictionary with the expected calibration error.
    """

    if eval_name is None:
        eval_name = "expected_calibration_error_evaluator__" + target_column

    if bin_choice == "count":
        bins = pd.qcut(test_data[prediction_column], q=n_bins)
    elif bin_choice == "prob":
        bins = pd.cut(test_data[prediction_column], bins=n_bins)
    else:
        raise AttributeError("Invalid bin_choice")

    metric_df = pd.DataFrame({"bins": bins,
                              "predictions": test_data[prediction_column],
                              "actuals": test_data[target_column]})

    agg_df = metric_df.groupby("bins").agg({"bins": "count", "predictions": "mean", "actuals": "mean"})

    sample_weight = None
    if bin_choice == "prob":
        sample_weight = agg_df["bins"].values

    distance = mean_absolute_error(agg_df["actuals"].values, agg_df["predictions"].values, sample_weight=sample_weight)

    return {eval_name: distance}


@curry
def r2_evaluator(test_data: pd.DataFrame,
                 prediction_column: str = "prediction",
                 target_column: str = "target",
                 weight_column: Optional[str] = None,
                 eval_name: Optional[str] = None) -> EvalReturnType:
    """
    Computes the R2 score, given true label and predictions.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction.

    target_column : String
        The name of the column in `test_data` with the continuous target.

    weight_column : String (default=None)
        The name of the column in `test_data` with the sample weights.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the R2 Score
    """

    eval_fn = generic_sklearn_evaluator("r2_evaluator__", r2_score)

    return eval_fn(test_data, prediction_column, target_column, weight_column, eval_name)


@curry
def mse_evaluator(test_data: pd.DataFrame,
                  prediction_column: