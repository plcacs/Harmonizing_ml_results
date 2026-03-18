```pyi
from typing import Optional, Dict, Any
from scipy.stats import wasserstein_distance
import torch
import torch.distributed as dist
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

class Independence(Metric):
    _num_classes: int
    _num_protected_variable_labels: int
    _predicted_label_counts: torch.Tensor
    _total_predictions: torch.Tensor
    _predicted_label_counts_by_protected_variable_label: torch.Tensor
    _dist_metric: Any
    
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = ...) -> None: ...
    def __call__(self, predicted_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.Tensor] = ...) -> None: ...
    def get_metric(self, reset: bool = ...) -> Dict[int, torch.Tensor]: ...
    def reset(self) -> None: ...

class Separation(Metric):
    _num_classes: int
    _num_protected_variable_labels: int
    _predicted_label_counts_by_gold_label: torch.Tensor
    _total_predictions: torch.Tensor
    _predicted_label_counts_by_gold_label_and_protected_variable_label: torch.Tensor
    _dist_metric: Any
    
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = ...) -> None: ...
    def __call__(self, predicted_labels: torch.Tensor, gold_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.Tensor] = ...) -> None: ...
    def get_metric(self, reset: bool = ...) -> Dict[int, Dict[int, torch.Tensor]]: ...
    def reset(self) -> None: ...

class Sufficiency(Metric):
    _num_classes: int
    _num_protected_variable_labels: int
    _gold_label_counts_by_predicted_label: torch.Tensor
    _total_predictions: torch.Tensor
    _gold_label_counts_by_predicted_label_and_protected_variable_label: torch.Tensor
    _dist_metric: Any
    
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = ...) -> None: ...
    def __call__(self, predicted_labels: torch.Tensor, gold_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.Tensor] = ...) -> None: ...
    def get_metric(self, reset: bool = ...) -> Dict[int, Dict[int, torch.Tensor]]: ...
    def reset(self) -> None: ...
```