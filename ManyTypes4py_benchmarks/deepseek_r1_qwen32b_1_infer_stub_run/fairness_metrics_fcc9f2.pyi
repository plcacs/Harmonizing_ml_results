"""
Fairness metrics are based on:
1. Barocas, S.; Hardt, M.; and Narayanan, A. 2019. Fairness and machine learning.
2. Zhang, B. H.; Lemoine, B.; and Mitchell, M. 2018. Mitigating unwanted biases with adversarial learning.
3. Hardt, M.; Price, E.; Srebro, N.; et al. 2016. Equality of opportunity in supervised learning.
4. Beutel, A.; Chen, J.; Zhao, Z.; and Chi, E. H. 2017. Data decisions and theoretical implications when adversarially learning fair representations.
"""

from typing import Dict, Optional, List, Tuple, Literal
from torch import Tensor
import torch.distributed as dist

class Independence:
    """
    Measures the statistical independence of the protected variable from predictions.
    """
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: Literal['kl_divergence', 'wasserstein']) -> None:
        ...

    def __call__(self, predicted_labels: Tensor, protected_variable_labels: Tensor, mask: Optional[Tensor] = None) -> None:
        ...

    def get_metric(self, reset: bool = False) -> Dict[int, Tensor]:
        ...

    def reset(self) -> None:
        ...

class Separation:
    """
    Allows correlation between predictions and protected variable to the extent justified by gold labels.
    """
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: Literal['kl_divergence', 'wasserstein']) -> None:
        ...

    def __call__(self, predicted_labels: Tensor, gold_labels: Tensor, protected_variable_labels: Tensor, mask: Optional[Tensor] = None) -> None:
        ...

    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, Tensor]]:
        ...

    def reset(self) -> None:
        ...

class Sufficiency:
    """
    Ensures predictions are sufficient for determining gold labels without relying on protected variables.
    """
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: Literal['kl_divergence', 'wasserstein']) -> None:
        ...

    def __call__(self, predicted_labels: Tensor, gold_labels: Tensor, protected_variable_labels: Tensor, mask: Optional[Tensor] = None) -> None:
        ...

    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, Tensor]]:
        ...

    def reset(self) -> None:
        ...