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
from typing import Optional, Dict, Union, Callable
from scipy.stats import wasserstein_distance
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from allennlp.training.metrics.metric import Metric

@Metric.register('independence')
class Independence(Metric):
    """
    [Independence](https://fairmlbook.org) (pg. 9) measures the statistical independence
    of the protected variable from predictions. It has been explored through many equivalent
    terms or variants, such as demographic parity, statistical parity, group fairness, and
    disparate impact.

    # Parameters

    num_classes : `int`
        Number of classes.
    num_protected_variable_labels : `int`
        Number of protected variable labels.
    dist_metric : `str`
        Distance metric (kl_divergence, wasserstein) for calculating the distance between the distribution
        over predicted labels and the distribution over predicted labels given a sensitive attribute.


    !!! Note
        Assumes integer labels, with each item to be classified having a single correct class.
    """
    _num_classes: int
    _num_protected_variable_labels: int
    _predicted_label_counts: torch.Tensor
    _total_predictions: torch.Tensor
    _predicted_label_counts_by_protected_variable_label: torch.Tensor
    _dist_metric: Union[Callable[[Categorical, Categorical], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]]
    
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None: ...
    def __call__(self, predicted_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> None: ...
    def get_metric(self, reset: bool = False) -> Dict[int, torch.Tensor]: ...
    def reset(self) -> None: ...

@Metric.register('separation')
class Separation(Metric):
    """
    [Separation](https://fairmlbook.org) (pg. 12) allows correlation between the
    predictions and the protected variable to the extent that it is justified by
    the gold labels.

    # Parameters

    num_classes : `int`
        Number of classes.
    num_protected_variable_labels : `int`
        Number of protected variable labels.
    dist_metric : `str`
        Distance metric (kl_divergence, wasserstein) for calculating the distance between
        the distribution over predicted labels given a gold label and a sensitive attribute from the
        distribution over predicted labels given only the gold label. If both distributions do not
        have equal support, you should use wasserstein distance.

    !!! Note
        Assumes integer labels, with each item to be classified having a single correct class.
    """
    _num_classes: int
    _num_protected_variable_labels: int
    _predicted_label_counts_by_gold_label: torch.Tensor
    _total_predictions: torch.Tensor
    _predicted_label_counts_by_gold_label_and_protected_variable_label: torch.Tensor
    _dist_metric: Union[Callable[[Categorical, Categorical], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]]
    
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None: ...
    def __call__(self, predicted_labels: torch.Tensor, gold_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> None: ...
    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.Tensor]]: ...
    def reset(self) -> None: ...

@Metric.register('sufficiency')
class Sufficiency(Metric):
    """
    [Sufficiency](https://fairmlbook.org) (pg. 14) is satisfied by the predictions
    when the protected variable and gold labels are clear from context.

    # Parameters

    num_classes : `int`
        Number of classes.
    num_protected_variable_labels : `int`
        Number of protected variable labels.
    dist_metric : `str`
        Distance metric (kl_divergence, wasserstein) for calculating the distance between
        the distribution over gold labels given a predicted label and a sensitive attribute from the
        distribution over gold labels given only the predicted label. If both distributions do not
        have equal support, you should use wasserstein distance.

    !!! Note
        Assumes integer labels, with each item to be classified having
        a single correct class.
    """
    _num_classes: int
    _num_protected_variable_labels: int
    _gold_label_counts_by_predicted_label: torch.Tensor
    _total_predictions: torch.Tensor
    _gold_label_counts_by_predicted_label_and_protected_variable_label: torch.Tensor
    _dist_metric: Union[Callable[[Categorical, Categorical], torch.Tensor], Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]]
    
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None: ...
    def __call__(self, predicted_labels: torch.Tensor, gold_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> None: ...
    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.Tensor]]: ...
    def reset(self) -> None: ...