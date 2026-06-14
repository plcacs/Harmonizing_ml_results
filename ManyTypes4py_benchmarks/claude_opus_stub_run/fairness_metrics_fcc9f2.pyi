from typing import Optional, Dict, Callable, Union
import torch
from scipy.stats import wasserstein_distance as _wasserstein_distance
from torch.distributions.kl import kl_divergence as _kl_divergence
from allennlp.training.metrics.metric import Metric


class Independence(Metric):
    _num_classes: int
    _num_protected_variable_labels: int
    _predicted_label_counts: torch.Tensor
    _total_predictions: torch.Tensor
    _predicted_label_counts_by_protected_variable_label: torch.Tensor
    _dist_metric: Union[type(_kl_divergence), type(_wasserstein_distance)]

    def __init__(
        self,
        num_classes: int,
        num_protected_variable_labels: int,
        dist_metric: str = "kl_divergence",
    ) -> None: ...
    def __call__(
        self,
        predicted_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None: ...
    def get_metric(
        self, reset: bool = False
    ) -> Dict[int, torch.FloatTensor]: ...
    def reset(self) -> None: ...


class Separation(Metric):
    _num_classes: int
    _num_protected_variable_labels: int
    _predicted_label_counts_by_gold_label: torch.Tensor
    _total_predictions: torch.Tensor
    _predicted_label_counts_by_gold_label_and_protected_variable_label: torch.Tensor
    _dist_metric: Union[type(_kl_divergence), type(_wasserstein_distance)]

    def __init__(
        self,
        num_classes: int,
        num_protected_variable_labels: int,
        dist_metric: str = "kl_divergence",
    ) -> None: ...
    def __call__(
        self,
        predicted_labels: torch.Tensor,
        gold_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None: ...
    def get_metric(
        self, reset: bool = False
    ) -> Dict[int, Dict[int, torch.FloatTensor]]: ...
    def reset(self) -> None: ...


class Sufficiency(Metric):
    _num_classes: int
    _num_protected_variable_labels: int
    _gold_label_counts_by_predicted_label: torch.Tensor
    _total_predictions: torch.Tensor
    _gold_label_counts_by_predicted_label_and_protected_variable_label: torch.Tensor
    _dist_metric: Union[type(_kl_divergence), type(_wasserstein_distance)]

    def __init__(
        self,
        num_classes: int,
        num_protected_variable_labels: int,
        dist_metric: str = "kl_divergence",
    ) -> None: ...
    def __call__(
        self,
        predicted_labels: torch.Tensor,
        gold_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None: ...
    def get_metric(
        self, reset: bool = False
    ) -> Dict[int, Dict[int, torch.FloatTensor]]: ...
    def reset(self) -> None: ...