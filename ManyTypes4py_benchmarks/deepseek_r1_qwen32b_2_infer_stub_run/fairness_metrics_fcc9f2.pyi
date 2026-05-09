from typing import Optional, Dict, List
from torch import Tensor

class Independence:
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
        ...
    
    def __call__(self, predicted_labels: Tensor, protected_variable_labels: Tensor, mask: Optional[Tensor] = None) -> None:
        ...
    
    def get_metric(self, reset: bool = False) -> Dict[int, Tensor]:
        ...
    
    def reset(self) -> None:
        ...

class Separation:
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
        ...
    
    def __call__(self, predicted_labels: Tensor, gold_labels: Tensor, protected_variable_labels: Tensor, mask: Optional[Tensor] = None) -> None:
        ...
    
    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, Tensor]]:
        ...
    
    def reset(self) -> None:
        ...

class Sufficiency:
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
        ...
    
    def __call__(self, predicted_labels: Tensor, gold_labels: Tensor, protected_variable_labels: Tensor, mask: Optional[Tensor] = None) -> None:
        ...
    
    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, Tensor]]:
        ...
    
    def reset(self) -> None:
        ...