from typing import Optional, Dict
from scipy.stats import wasserstein_distance
import torch
import torch.distributed as dist
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

@Metric.register('independence')
class Independence(Metric):
    """
    [Independence](https://fairmlbook.org) (pg. 9) measures the statistical independence
    of the protected variable from predictions. It has been explored through many equivalent
    terms or variants, such as demographic parity, statistical parity, group fairness, and
    disparate impact.

    # Parameters

    num_classes : int
        Number of classes.
    num_protected_variable_labels : int
        Number of protected variable labels.
    dist_metric : str
        Distance metric (kl_divergence, wasserstein) for calculating the distance between the distribution
        over predicted labels and the distribution over predicted labels given a sensitive attribute.

    !!! Note
        Assumes integer labels, with each item to be classified having a single correct class.
    """

    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
        # ... rest of the code ...

    def get_metric(self, reset: bool = False) -> Dict[int, torch.FloatTensor]:
        # ... rest of the code ...

@Metric.register('separation')
class Separation(Metric):
    """
    [Separation](https://fairmlbook.org) (pg. 12) allows correlation between the
    predictions and the protected variable to the extent that it is justified by
    the gold labels.

    # Parameters

    num_classes : int
        Number of classes.
    num_protected_variable_labels : int
        Number of protected variable labels.
    dist_metric : str
        Distance metric (kl_divergence, wasserstein) for calculating the distance between
        the distribution over predicted labels given a gold label and a sensitive attribute from the
        distribution over predicted labels given only the gold label. If both distributions do not
        have equal support, you should use wasserstein distance.

    !!! Note
        Assumes integer labels, with each item to be classified having a single correct class.
    """

    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
        # ... rest of the code ...

    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.FloatTensor]]:
        # ... rest of the code ...

@Metric.register('sufficiency')
class Sufficiency(Metric):
    """
    [Sufficiency](https://fairmlbook.org) (pg. 14) is satisfied by the predictions
    when the protected variable and gold labels are clear from context.

    # Parameters

    num_classes : int
        Number of classes.
    num_protected_variable_labels : int
        Number of protected variable labels.
    dist_metric : str
        Distance metric (kl_divergence, wasserstein) for calculating the distance between
        the distribution over gold labels given a predicted label and a sensitive attribute from the
        distribution over gold labels given only the predicted label. If both distributions do not
        have equal support, you should use wasserstein distance.

    !!! Note
        Assumes integer labels, with each item to be classified having
        a single correct class.
    """

    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
        # ... rest of the code ...

    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.FloatTensor]]:
        # ... rest of the code ...
