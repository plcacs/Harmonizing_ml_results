```python
from typing import Any, Callable, Dict, Optional, Union
import torch
from allennlp.training.metrics.metric import Metric

@Metric.register('independence')
class Independence(Metric):
    _num_classes: int
    _num_protected_variable_labels: int
    _predicted_label_counts: torch.Tensor
    _total_predictions: torch.Tensor
    _predicted_label_counts_by_protected_variable_label: torch.Tensor
    _dist_metric: Union[Callable[..., Any], Callable[..., torch.Tensor]]
    
    def __init__(
        self,
        num_classes: int,
        num_protected_variable_labels: int,
        dist_metric: str = 'kl_divergence'
    ) -> None: ...
    
    def __call__(
        self,
        predicted_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None
    ) -> None: ...
    
    def get_metric(self, reset: bool = False) -> Dict[int, torch.Tensor]: ...
    
    def reset(self) -> None: ...

@Metric.register('separation')
class Separation(Metric):
    _num_classes: int
    _num_protected_variable_labels: int
    _predicted_label_counts_by_gold_label: torch.Tensor
    _total_predictions: torch.Tensor
    _predicted_label_counts_by_gold_label_and_protected_variable_label: torch.Tensor
    _dist_metric: Union[Callable[..., Any], Callable[..., torch.Tensor]]
    
    def __init__(
        self,
        num_classes: int,
        num_protected_variable_labels: int,
        dist_metric: str = 'kl_divergence'
    ) -> None: ...
    
    def __call__(
        self,
        predicted_labels: torch.Tensor,
        gold_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None
    ) -> None: ...
    
    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.Tensor]]: ...
    
    def reset(self) -> None: ...

@Metric.register('sufficiency')
class Sufficiency(Metric):
    _num_classes: int
    _num_protected_variable_labels: int
    _gold_label_counts_by_predicted_label: torch.Tensor
    _total_predictions: torch.Tensor
    _gold_label_counts_by_predicted_label_and_protected_variable_label: torch.Tensor
    _dist_metric: Union[Callable[..., Any], Callable[..., torch.Tensor]]
    
    def __init__(
        self,
        num_classes: int,
        num_protected_variable_labels: int,
        dist_metric: str = 'kl_divergence'
    ) -> None: ...
    
    def __call__(
        self,
        predicted_labels: torch.Tensor,
        gold_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None
    ) -> None: ...
    
    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.Tensor]]: ...
    
    def reset(self) -> None: ...
```