#!/usr/bin/env python3
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional
import numpy as np
import pandas as pd
import toolz as fp
from pandas.util import hash_pandas_object
from sklearn.metrics import (average_precision_score, brier_score_loss, fbeta_score, log_loss,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from toolz import curry, last, first
from scipy import optimize
from sklearn.linear_model import LogisticRegression
# from fklearn.types import EvalFnType, EvalReturnType, PredictFnType, UncurriedEvalFnType

def generic_sklearn_evaluator(name_prefix: str, sklearn_metric: Callable[..., float]
                              ) -> Callable[[pd.DataFrame, str, str, Optional[str], Optional[str]], Dict[str, float]]:
    """
    Returns an evaluator built from a metric from sklearn.metrics.
    """
    def p(test_data: pd.DataFrame, prediction_column: str = 'prediction', target_column: str = 'target',
          weight_column: Optional[str] = None, eval_name: Optional[str] = None, **kwargs: Any) -> Dict[str, float]:
        try:
            score = sklearn_metric(
                test_data[target_column],
                test_data[prediction_column],
                sample_weight=None if weight_column is None else test_data[weight_column],
                **kwargs)
        except ValueError:
            score = np.nan
        if eval_name is None:
            eval_name = name_prefix + target_column
        return {eval_name: score}
    return p

@curry
def auc_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                  target_column: str = 'target', weight_column: Optional[str] = None,
                  eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the ROC AUC score, given true label and prediction scores.
    """
    warnings.warn('The method `auc_evaluator` will be renamed to `roc_auc_evaluator` in the next major release 2.0.0. '
                  'Please use `roc_auc_evaluator` instead of `auc_evaluator` for Area Under the Curve of the '
                  'Receiver Operating Characteristics curve.')
    return roc_auc_evaluator(test_data, prediction_column, target_column, weight_column, eval_name)

@curry
def roc_auc_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                      target_column: str = 'target', weight_column: Optional[str] = None,
                      eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the ROC AUC score, given true label and prediction scores.
    """
    eval_fn = generic_sklearn_evaluator('roc_auc_evaluator__', roc_auc_score)
    eval_data = test_data.assign(**{target_column: lambda df: df[target_column].astype(int)})
    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)

@curry
def pr_auc_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                     target_column: str = 'target', weight_column: Optional[str] = None,
                     eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the PR AUC score, given true label and prediction scores.
    """
    eval_fn = generic_sklearn_evaluator('pr_auc_evaluator__', average_precision_score)
    eval_data = test_data.assign(**{target_column: lambda df: df[target_column].astype(int)})
    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)

@curry
def precision_evaluator(test_data: pd.DataFrame, threshold: float = 0.5,
                        prediction_column: str = 'prediction', target_column: str = 'target',
                        weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the precision score, given true label and prediction scores.
    """
    eval_fn = generic_sklearn_evaluator('precision_evaluator__', precision_score)
    eval_data = test_data.assign(**{prediction_column: (test_data[prediction_column] > threshold).astype(int)})
    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)

@curry
def recall_evaluator(test_data: pd.DataFrame, threshold: float = 0.5,
                     prediction_column: str = 'prediction', target_column: str = 'target',
                     weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the recall score, given true label and prediction scores.
    """
    eval_data = test_data.assign(**{prediction_column: (test_data[prediction_column] > threshold).astype(int)})
    eval_fn = generic_sklearn_evaluator('recall_evaluator__', recall_score)
    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)

@curry
def fbeta_score_evaluator(test_data: pd.DataFrame, threshold: float = 0.5, beta: float = 1.0,
                          prediction_column: str = 'prediction', target_column: str = 'target',
                          weight_column: Optional[str] = None, eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the F-beta score, given true label and prediction scores.
    """
    eval_data = test_data.assign(**{prediction_column: (test_data[prediction_column] > threshold).astype(int)})
    eval_fn = generic_sklearn_evaluator('fbeta_evaluator__', fbeta_score)
    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name, beta=beta)

@curry
def logloss_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                      target_column: str = 'target', weight_column: Optional[str] = None,
                      eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the logloss score, given true label and prediction scores.
    """
    eval_fn = generic_sklearn_evaluator('logloss_evaluator__', log_loss)
    eval_data = test_data.assign(**{target_column: lambda df: df[target_column].astype(int)})
    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)

@curry
def brier_score_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                          target_column: str = 'target', weight_column: Optional[str] = None,
                          eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the Brier score, given true label and prediction scores.
    """
    eval_fn = generic_sklearn_evaluator('brier_score_evaluator__', brier_score_loss)
    eval_data = test_data.assign(**{target_column: lambda df: df[target_column].astype(int)})
    return eval_fn(eval_data, prediction_column, target_column, weight_column, eval_name)

@curry
def expected_calibration_error_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                                         target_column: str = 'target', eval_name: Optional[str] = None,
                                         n_bins: int = 100, bin_choice: str = 'count') -> Dict[str, float]:
    """
    Computes the expected calibration error (ECE), given true label and prediction scores.
    """
    if eval_name is None:
        eval_name = 'expected_calibration_error_evaluator__' + target_column
    if bin_choice == 'count':
        bins = pd.qcut(test_data[prediction_column], q=n_bins)
    elif bin_choice == 'prob':
        bins = pd.cut(test_data[prediction_column], bins=n_bins)
    else:
        raise AttributeError('Invalid bin_choice')
    metric_df = pd.DataFrame({'bins': bins,
                              'predictions': test_data[prediction_column],
                              'actuals': test_data[target_column]})
    agg_df = metric_df.groupby('bins').agg({'bins': 'count', 'predictions': 'mean', 'actuals': 'mean'})
    sample_weight: Optional[np.ndarray] = None
    if bin_choice == 'prob':
        sample_weight = agg_df['bins'].values
    distance = mean_absolute_error(agg_df['actuals'].values, agg_df['predictions'].values, sample_weight=sample_weight)
    return {eval_name: distance}

@curry
def r2_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                 target_column: str = 'target', weight_column: Optional[str] = None,
                 eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the R2 score, given true label and predictions.
    """
    eval_fn = generic_sklearn_evaluator('r2_evaluator__', r2_score)
    return eval_fn(test_data, prediction_column, target_column, weight_column, eval_name)

@curry
def mse_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                  target_column: str = 'target', weight_column: Optional[str] = None,
                  eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the Mean Squared Error, given true label and predictions.
    """
    eval_fn = generic_sklearn_evaluator('mse_evaluator__', mean_squared_error)
    return eval_fn(test_data, prediction_column, target_column, weight_column, eval_name)

@curry
def mean_prediction_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                              eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes mean for the specified column.
    """
    if eval_name is None:
        eval_name = 'mean_evaluator__' + prediction_column
    return {eval_name: test_data[prediction_column].mean()}

@curry
def correlation_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                          target_column: str = 'target', eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the Pearson correlation between prediction and target.
    """
    if eval_name is None:
        eval_name = 'correlation_evaluator__' + target_column
    score = test_data[[prediction_column, target_column]].corr(method='pearson').iloc[0, 1]
    return {eval_name: score}

@curry
def linear_coefficient_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                                 target_column: str = 'target', eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the linear coefficient from regressing the outcome on the prediction.
    """
    if eval_name is None:
        eval_name = 'linear_coefficient_evaluator__' + target_column
    cov_mat = test_data[[prediction_column, target_column]].cov()
    score = cov_mat.iloc[0, 1] / cov_mat.iloc[0, 0]
    return {eval_name: score}

@curry
def spearman_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                       target_column: str = 'target', eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the Spearman correlation between prediction and target.
    """
    if eval_name is None:
        eval_name = 'spearman_evaluator__' + target_column
    score = test_data[[prediction_column, target_column]].corr(method='spearman').iloc[0, 1]
    return {eval_name: score}

@curry
def ndcg_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                   target_column: str = 'target', k: Optional[int] = None, exponential_gain: bool = True,
                   eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the Normalized Discount Cumulative Gain (NDCG).
    """
    if k is not None and not 0 < k <= len(test_data[prediction_column]):
        raise ValueError('k must be between [1, len(test_data[prediction_column])].')
    if eval_name is None:
        eval_name = f'ndcg_evaluator__{target_column}'
    rel = np.argsort(test_data[prediction_column])[::-1][:k]
    cum_gain = test_data[target_column][rel]
    ideal_cum_gain = np.sort(test_data[target_column])[::-1][:k]
    if exponential_gain:
        cum_gain = 2 ** cum_gain - 1
        ideal_cum_gain = 2 ** ideal_cum_gain - 1
    discount = np.log2(np.arange(len(cum_gain)) + 2.0)
    dcg = np.sum(cum_gain / discount)
    idcg = np.sum(ideal_cum_gain / discount)
    ndcg_score = dcg / idcg
    return {eval_name: ndcg_score}

@curry
def combined_evaluators(test_data: pd.DataFrame,
                        evaluators: List[Callable[[pd.DataFrame], Dict[str, float]]]
                        ) -> Dict[str, float]:
    """
    Combine partially applied evaluation functions.
    """
    return fp.merge((e(test_data) for e in evaluators))

@curry
def split_evaluator(test_data: pd.DataFrame, eval_fn: Callable[[pd.DataFrame], Dict[str, float]],
                    split_col: str, split_values: Optional[Iterable[Any]] = None,
                    eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Splits the dataset by a column and evaluates model performance in each split.
    """
    if split_values is None:
        split_values = test_data[split_col].unique()
    if eval_name is None:
        eval_name = 'split_evaluator__' + split_col
    return {eval_name + '_' + str(value): eval_fn(test_data.loc[lambda df: df[split_col] == value])
            for value in split_values}

@curry
def temporal_split_evaluator(test_data: pd.DataFrame, eval_fn: Callable[[pd.DataFrame], Dict[str, float]],
                             time_col: str, time_format: str = '%Y-%m',
                             split_values: Optional[Iterable[Any]] = None,
                             eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Splits the dataset into temporal categories by a time column and evaluates model performance.
    """
    formatted_time_col = test_data[time_col].dt.strftime(time_format)
    unique_values = formatted_time_col.unique()
    if eval_name is None:
        eval_name = 'split_evaluator__' + time_col
    if split_values is None:
        split_values = unique_values
    elif not all((sv in unique_values for sv in split_values)):
        raise ValueError('All split values must be present in the column (after date formatting it)')
    return {eval_name + '_' + str(value): eval_fn(test_data.loc[lambda df: formatted_time_col == value])
            for value in split_values}

@curry
def permutation_evaluator(test_data: pd.DataFrame,
                          predict_fn: Callable[[pd.DataFrame], pd.DataFrame],
                          eval_fn: Callable[[pd.DataFrame], Dict[str, float]],
                          baseline: bool = True,
                          features: Optional[List[str]] = None,
                          shuffle_all_at_once: bool = False,
                          random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Permutation importance evaluator.
    """
    if features is None:
        features = list(test_data.columns)

    def col_shuffler(f: str) -> np.ndarray:
        return test_data[f].sample(frac=1.0, random_state=random_state).values

    def permutation_eval(features_to_shuffle: List[str]) -> Dict[str, float]:
        shuffled_cols = {f: col_shuffler(f) for f in features_to_shuffle}
        return eval_fn(predict_fn(test_data.assign(**shuffled_cols)))
    if shuffle_all_at_once:
        permutation_results = {'-'.join(features): permutation_eval(features)}
    else:
        permutation_results = {f: permutation_eval([f]) for f in features}
    feature_importance = {'permutation_importance': permutation_results}
    if baseline:
        baseline_results = {'permutation_importance_baseline': eval_fn(predict_fn(test_data))}
    else:
        baseline_results = {}
    return fp.merge(feature_importance, baseline_results)

@curry
def hash_evaluator(test_data: pd.DataFrame, hash_columns: Optional[List[str]] = None,
                   eval_name: Optional[str] = None, consider_index: bool = False) -> Dict[str, Any]:
    """
    Computes the hash of a pandas dataframe filtered by specified columns.
    """
    if hash_columns is None:
        hash_columns = list(test_data.columns)

    def calculate_dataframe_hash(df: pd.DataFrame, eval_name: str) -> Dict[str, Any]:
        return {eval_name: hash_pandas_object(df).sum()}
    if eval_name is None:
        eval_name = 'hash_evaluator__' + '_'.join(sorted(hash_columns))
    eval_data = test_data[hash_columns]
    if not consider_index:
        return calculate_dataframe_hash(eval_data.set_index(np.zeros(len(eval_data), dtype='int')), eval_name)
    return calculate_dataframe_hash(eval_data, eval_name)

@curry
def exponential_coefficient_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                                      target_column: str = 'target', eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the exponential coefficient between prediction and target.
    """
    if eval_name is None:
        eval_name = 'exponential_coefficient_evaluator__' + target_column
    popt = optimize.curve_fit(lambda t, a0, a1: a0 * np.exp(a1 * t),
                              test_data[prediction_column],
                              test_data[target_column])[0]
    score = popt[-1]
    return {eval_name: score}

@curry
def logistic_coefficient_evaluator(test_data: pd.DataFrame, prediction_column: str = 'prediction',
                                   target_column: str = 'target', eval_name: Optional[str] = None) -> Dict[str, float]:
    """
    Computes the logistic coefficient between prediction and target.
    """
    if eval_name is None:
        eval_name = 'logistic_coefficient_evaluator__' + target_column
    score = LogisticRegression(penalty=None, multi_class='ovr').fit(
        test_data[[prediction_column]], test_data[target_column]).coef_[0][0]
    return {eval_name: score}