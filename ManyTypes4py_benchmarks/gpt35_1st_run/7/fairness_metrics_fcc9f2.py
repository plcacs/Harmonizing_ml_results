from typing import Optional, Dict
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

@Metric.register('independence')
class Independence(Metric):
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
    def __call__(self, predicted_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> None:
    def get_metric(self, reset: bool = False) -> Dict[int, torch.FloatTensor]:
    def reset(self) -> None:

@Metric.register('separation')
class Separation(Metric):
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
    def __call__(self, predicted_labels: torch.Tensor, gold_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> None:
    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.FloatTensor]]:
    def reset(self) -> None:

@Metric.register('sufficiency')
class Sufficiency(Metric):
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
    def __call__(self, predicted_labels: torch.Tensor, gold_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> None:
    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.FloatTensor]]:
    def reset(self) -> None:
