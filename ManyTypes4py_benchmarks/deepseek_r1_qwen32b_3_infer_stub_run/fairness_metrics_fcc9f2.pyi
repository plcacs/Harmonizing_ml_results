"""
Fairness metrics are based on:

1. Barocas, S.; Hardt, M.; and Narayanan, A. 2019. [Fairness and machine learning](https://fairmlbook.org).

2. Zhang, B. H.; Lemoine, B.; and Mitchell, M. 2018. [Mitigating unwanted biases with adversarial learning]
(https://api.semanticscholar.org/CorpusID:9424845).
In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, 335-340.

3. Hardt, M.; Price, E.; Srebro, N.; et al. 2016. [Equality of opportunity in supervised learning]
(https://api.semanticscholar.org/CorpusID:7567061). In Advances in Neural Information Processing Systems,
3315–3323.

4. Beutel, A.; Chen, J.; Zhao, Z.; and Chi, E. H. 2017. [Data decisions and theoretical implications when
adversarially learning fair representations](https://api.semanticscholar.org/CorpusID:24990444).
arXiv preprint arXiv:1707.00075.

It is provably [impossible](https://fairmlbook.org/pdf/classification.pdf) (pg. 18) to satisfy any two of
Independence, Separation, and Sufficiency simultaneously, except in degenerate cases.
"""

from typing import Dict, Optional
from torch import Tensor
from allennlp.training.metrics.metric import Metric

@Metric.register('independence')
class Independence(Metric):
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: Optional[str] = 'kl_divergence') -> None:
        ...

    def __call__(self, predicted_labels: Tensor, protected_variable_labels: Tensor, mask: Optional[Tensor] = None) -> None:
        ...

    def get_metric(self, reset: bool = False) -> Dict[int, Tensor]:
        ...

    def reset(self) -> None:
        ...

@Metric.register('separation')
class Separation(Metric):
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: Optional[str] = 'kl_divergence') -> None:
        ...

    def __call__(self, predicted_labels: Tensor, gold_labels: Tensor, protected_variable_labels: Tensor, mask: Optional[Tensor] = None) -> None:
        ...

    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, Tensor]]:
        ...

    def reset(self) -> None:
        ...

@Metric.register('sufficiency')
class Sufficiency(Metric):
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: Optional[str] = 'kl_divergence') -> None:
        ...

    def __call__(self, predicted_labels: Tensor, gold_labels: Tensor, protected_variable_labels: Tensor, mask: Optional[Tensor] = None) -> None:
        ...

    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, Tensor]]:
        ...

    def reset(self) -> None:
        ...