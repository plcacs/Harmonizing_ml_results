from typing import Any, Callable, Dict, List, NamedTuple, Optional
import numpy as np
import sklearn.metrics as skmetrics
from snorkel.utils import filter_labels, to_int_label_array


class Metric(NamedTuple):
    """Specification for a metric and the subset of [golds, preds, probs] it expects."""
    func: Callable[..., float]
    inputs: List[str] = ['golds', 'preds']


def metric_score(
    golds: Optional[np.ndarray] = None,
    preds: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
    metric: str = 'accuracy',
    filter_dict: Optional[Dict[str, List[Any]]] = None,
    **kwargs: Any
) -> float:
    """Evaluate a standard metric on a set of predictions/probabilities.

    Parameters
    ----------
    golds
        An array of gold (int) labels
    preds
        An array of (int) predictions
    probs
        An [n_datapoints, n_classes] array of probabilistic (float) predictions
    metric
        The name of the metric to calculate
    filter_dict
        A mapping from label set name to the labels that should be filtered out for
        that label set

    Returns
    -------
    float
        The value of the requested metric

    Raises
    ------
    ValueError
        The requested metric is not currently supported
    ValueError
        The user attempted to calculate roc_auc score for a non-binary problem
    """
    if metric not in METRICS:
        msg = f'The metric you provided ({metric}) is not currently implemented.'
        raise ValueError(msg)
    golds_array = to_int_label_array(golds) if golds is not None else None
    preds_array = to_int_label_array(preds) if preds is not None else None
    label_dict: Dict[str, Optional[np.ndarray]] = {'golds': golds_array, 'preds': preds_array, 'probs': probs}
    if filter_dict:
        if set(filter_dict.keys()).difference(set(label_dict.keys())):
            raise ValueError("filter_dict must only include keys in ['golds', 'preds', 'probs']")
        label_dict = filter_labels(label_dict, filter_dict)
    metric_obj: Metric = METRICS[metric]
    for label_name in metric_obj.inputs:
        if label_dict[label_name] is None:
            raise ValueError(f'Metric {metric} requires access to {label_name}.')
    label_sets: List[np.ndarray] = [label_dict[label_name] for label_name in metric_obj.inputs]
    return metric_obj.func(*label_sets, **kwargs)


def _coverage_score(preds: np.ndarray) -> float:
    return np.sum(preds != -1) / len(preds)


def _roc_auc_score(golds: np.ndarray, probs: np.ndarray) -> float:
    if not probs.shape[1] == 2:
        raise ValueError('Metric roc_auc is currently only defined for binary problems.')
    return skmetrics.roc_auc_score(golds, probs[:, 1])


def _f1_score(golds: np.ndarray, preds: np.ndarray) -> float:
    if golds.max() <= 1:
        return skmetrics.f1_score(golds, preds)
    else:
        raise ValueError('f1 not supported for multiclass. Try f1_micro or f1_macro instead.')


def _f1_micro_score(golds: np.ndarray, preds: np.ndarray) -> float:
    return skmetrics.f1_score(golds, preds, average='micro')


def _f1_macro_score(golds: np.ndarray, preds: np.ndarray) -> float:
    return skmetrics.f1_score(golds, preds, average='macro')


METRICS: Dict[str, Metric] = {
    'accuracy': Metric(skmetrics.accuracy_score),
    'coverage': Metric(_coverage_score, ['preds']),
    'precision': Metric(skmetrics.precision_score),
    'recall': Metric(skmetrics.recall_score),
    'f1': Metric(_f1_score, ['golds', 'preds']),
    'f1_micro': Metric(_f1_micro_score, ['golds', 'preds']),
    'f1_macro': Metric(_f1_macro_score, ['golds', 'preds']),
    'fbeta': Metric(skmetrics.fbeta_score),
    'matthews_corrcoef': Metric(skmetrics.matthews_corrcoef),
    'roc_auc': Metric(_roc_auc_score, ['golds', 'probs'])
}
