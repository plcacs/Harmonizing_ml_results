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

from typing import Optional, Dict, Union, Callable, Any

from scipy.stats import wasserstein_distance

import torch
import torch.distributed as dist
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("independence")
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

    def __init__(
        self,
        num_classes: int,
        num_protected_variable_labels: int,
        dist_metric: str = "kl_divergence",
    ) -> None:
        self._num_classes = num_classes
        self._num_protected_variable_labels = num_protected_variable_labels
        self._predicted_label_counts: torch.Tensor = torch.zeros(num_classes)
        self._total_predictions: torch.Tensor = torch.tensor(0)
        self._predicted_label_counts_by_protected_variable_label: torch.Tensor = torch.zeros(
            (num_protected_variable_labels, num_classes)
        self._dist_metric: Union[Callable[..., torch.Tensor], Callable[..., float]]
        if dist_metric == "kl_divergence":
            self._dist_metric = kl_divergence
        elif dist_metric == "wasserstein":
            self._dist_metric = wasserstein_distance
        else:
            raise ConfigurationError(
                "supported distance metrics in initialization are 'kl_divergence' and 'wasserstein'"
            )

    def __call__(
        self,
        predicted_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        """
        # Parameters

        predicted_labels : `torch.Tensor`, required.
            A tensor of predicted integer class labels of shape (batch_size, ...). Represented as C.
        protected_variable_labels : `torch.Tensor`, required.
            A tensor of integer protected variable labels of shape (batch_size, ...). It must be the same
            shape as the `predicted_labels` tensor. Represented as A.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_labels`.

        !!! Note
            All tensors are expected to be on the same device.
        """
        predicted_labels, protected_variable_labels, mask = self.detach_tensors(
            predicted_labels, protected_variable_labels, mask
        )

        # Some sanity checks.
        if predicted_labels.size() != protected_variable_labels.size():
            raise ConfigurationError(
                "protected_variable_labels must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(protected_variable_labels.size())
            )
        if mask is not None and predicted_labels.size() != mask.size():
            raise ConfigurationError(
                "mask must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(mask.size())
            )
        if (predicted_labels >= self._num_classes).any():
            raise ConfigurationError(
                "predicted_labels contains an id >= {}, "
                "the number of classes.".format(self._num_classes)
            )
        if (protected_variable_labels >= self._num_protected_variable_labels).any():
            raise ConfigurationError(
                "protected_variable_labels contains an id >= {}, "
                "the number of protected variable labels.".format(
                    self._num_protected_variable_labels
                )
            )

        device = predicted_labels.device
        self._predicted_label_counts = self._predicted_label_counts.to(device)
        self._predicted_label_counts_by_protected_variable_label = (
            self._predicted_label_counts_by_protected_variable_label.to(device)
        )
        self._total_predictions = self._total_predictions.to(device)

        if mask is not None:
            predicted_labels = predicted_labels[mask]
            protected_variable_labels = protected_variable_labels[mask]
        else:
            predicted_labels = predicted_labels.flatten()
            protected_variable_labels = protected_variable_labels.flatten()

        _predicted_label_counts = predicted_labels.float().histc(
            bins=self._num_classes, min=0, max=self._num_classes - 1
        )
        _total_predictions = torch.tensor(predicted_labels.nelement()).to(device)

        _predicted_label_counts_by_protected_variable_label = torch.zeros(
            (self._num_protected_variable_labels, self._num_classes)
        ).to(device)
        for a in range(self._num_protected_variable_labels):
            _predicted_label_counts_by_protected_variable_label[a] = (
                predicted_labels[protected_variable_labels == a]
                .float()
                .histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
            )

        if is_distributed():
            _predicted_label_counts = _predicted_label_counts.to(device)
            dist.all_reduce(_predicted_label_counts, op=dist.ReduceOp.SUM)

            _total_predictions = _total_predictions.to(device)
            dist.all_reduce(_total_predictions, op=dist.ReduceOp.SUM)

            _predicted_label_counts_by_protected_variable_label = (
                _predicted_label_counts_by_protected_variable_label.to(device)
            )
            dist.all_reduce(
                _predicted_label_counts_by_protected_variable_label, op=dist.ReduceOp.SUM
            )

        self._predicted_label_counts += _predicted_label_counts
        self._total_predictions += _total_predictions
        self._predicted_label_counts_by_protected_variable_label += (
            _predicted_label_counts_by_protected_variable_label
        )

    def get_metric(self, reset: bool = False) -> Dict[int, torch.FloatTensor]:
        """
        # Returns

        distances : `Dict[int, torch.FloatTensor]`
            A dictionary mapping each protected variable label a to the KL divergence or Wasserstein distance
             of P(C | A = a) from P(C). A distance of nearly 0 implies fairness on the basis of Independence.
        """
        distances: Dict[int, torch.FloatTensor] = {}
        if self._total_predictions == 0:
            distances = {
                a: torch.tensor(float("nan")) for a in range(self._num_protected_variable_labels)
            }
            return distances

        C_dist = Categorical(self._predicted_label_counts / self._total_predictions)
        if self._dist_metric == wasserstein_distance:
            C_dist_probs: torch.Tensor = C_dist.probs
            C_dist = C_dist_probs
        for a in range(self._num_protected_variable_labels):
            C_given_a_dist = Categorical(
                self._predicted_label_counts_by_protected_variable_label[a]
                / self._total_predictions
            )
            if self._dist_metric == kl_divergence:
                distances[a] = self._dist_metric(C_given_a_dist, C_dist)
            elif self._dist_metric == wasserstein_distance:
                C_given_a_dist_probs: torch.Tensor = C_given_a_dist.probs
                C_given_a_dist = C_given_a_dist_probs
                # scipy expects empirical values as well as probabilities
                label_values = torch.tensor(range(self._num_classes))
                distances[a] = self._dist_metric(label_values, label_values, C_given_a_dist, C_dist)
        if reset:
            self.reset()
        return distances

    def reset(self) -> None:
        self._predicted_label_counts = torch.zeros(self._num_classes)
        self._total_predictions = torch.tensor(0)
        self._predicted_label_counts_by_protected_variable_label = torch.zeros(
            (self._num_protected_variable_labels, self._num_classes)
        )


@Metric.register("separation")
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

    def __init__(
        self,
        num_classes: int,
        num_protected_variable_labels: int,
        dist_metric: str = "kl_divergence",
    ) -> None:
        self._num_classes = num_classes
        self._num_protected_variable_labels = num_protected_variable_labels
        self._predicted_label_counts_by_gold_label: torch.Tensor = torch.zeros((num_classes, num_classes))
        self._total_predictions: torch.Tensor = torch.tensor(0)
        self._predicted_label_counts_by_gold_label_and_protected_variable_label: torch.Tensor = torch.zeros(
            (num_classes, num_protected_variable_labels, num_classes))
        self._dist_metric: Union[Callable[..., torch.Tensor], Callable[..., float]]
        if dist_metric == "kl_divergence":
            self._dist_metric = kl_divergence
        elif dist_metric == "wasserstein":
            self._dist_metric = wasserstein_distance
        else:
            raise ConfigurationError(
                "supported distance metrics in initialization are 'kl_divergence' and 'wasserstein'"
            )

    def __call__(
        self,
        predicted_labels: torch.Tensor,
        gold_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        """
        # Parameters

        predicted_labels : `torch.Tensor`, required.
            A tensor of predicted integer class labels of shape (batch_size, ...). Represented as C.
        gold_labels : `torch.Tensor`, required.
            A tensor of ground-truth integer class labels of shape (batch_size, ...). It must be the same
            shape as the `predicted_labels` tensor. Represented as Y.
        protected_variable_labels : `torch.Tensor`, required.
            A tensor of integer protected variable labels of shape (batch_size, ...). It must be the same
            shape as the `predicted_labels` tensor. Represented as A.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_labels`.

        !!! Note
            All tensors are expected to be on the same device.
        """
        predicted_labels, gold_labels, protected_variable_labels, mask = self.detach_tensors(
            predicted_labels, gold_labels, protected_variable_labels, mask
        )

        # Some sanity checks.
        if predicted_labels.size() != protected_variable_labels.size():
            raise ConfigurationError(
                "protected_variable_labels must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(protected_variable_labels.size())
            )
        if predicted_labels.size() != gold_labels.size():
            raise ConfigurationError(
                "gold_labels must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(gold_labels.size())
            )
        if mask is not None and predicted_labels.size() != mask.size():
            raise ConfigurationError(
                "mask must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(mask.size())
            )
        if (predicted_labels >= self._num_classes).any():
            raise ConfigurationError(
                "predicted_labels contains an id >= {}, "
                "the number of classes.".format(self._num_classes)
            )
        if (gold_labels >= self._num_classes).any():
            raise ConfigurationError(
                "gold_labels contains an id >= {}, "
                "the number of classes.".format(self._num_classes)
            )
        if (protected_variable_labels >= self._num_protected_variable_labels).any():
            raise ConfigurationError(
                "protected_variable_labels contains an id >= {}, "
                "the number of protected variable labels.".format(
                    self._num_protected_variable_labels
                )
            )

        device = predicted_labels.device
        self._predicted_label_counts_by_gold_label = self._predicted_label_counts_by_gold_label.to(
            device
        )
        self._predicted_label_counts_by_gold_label_and_protected_variable_label = (
            self._predicted_label_counts_by_gold_label_and_protected_variable_label.to(device)
        self._total_predictions = self._total_predictions.to(device)

        if mask is not None:
            predicted_labels = predicted_labels[mask]
            gold_labels = gold_labels[mask]
            protected_variable_labels = protected_variable_labels[mask]
        else:
            predicted_labels = predicted_labels.flatten()
            gold_labels = gold_labels.flatten()
            protected_variable_labels = protected_variable_labels.flatten()

        _total_predictions = torch.tensor(predicted_labels.nelement()).to(device)
        _predicted_label_counts_by_gold_label = torch.zeros(
            (self._num_classes, self._num_classes)
        ).to(device)
        _predicted_label_counts_by_gold_label_and_protected_variable_label = torch.zeros(
            (self._num_classes, self._num_protected_variable_labels, self._num_classes)
        ).to(device)
        for y in range(self._num_classes):
            _predicted_label_counts_by_gold_label[y] = (
                predicted_labels[gold_labels == y]
                .float()
                .histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
            )
            for a in range(self._num_protected_variable_labels):
                _predicted_label_counts_by_gold_label_and_protected_variable_label[y][a] = (
                    predicted_labels[(gold_labels == y) & (protected_variable_labels == a)]
                    .float()
                    .histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
                )

        if is_distributed():
            _total_predictions = _total_predictions.to(device)
            dist.all_reduce(_total_predictions, op=dist.ReduceOp.SUM)

            _predicted_label_counts_by_gold_label = _predicted_label_counts_by_gold_label.to(device)
            dist.all_reduce(_predicted_label_counts_by_gold_label[y], op=dist.ReduceOp.SUM)

            _predicted_label_counts_by_gold_label_and_protected_variable_label = (
                _predicted_label_counts_by_gold_label_and_protected_variable_label.to(device)
            )
            dist.all_reduce(
                _predicted_label_counts_by_gold_label_and_protected_variable_label,
                op=dist.ReduceOp.SUM,
            )

        self._total_predictions += _total_predictions
        self._predicted_label_counts_by_gold_label += _predicted_label_counts_by_gold_label
        self._predicted_label_counts_by_gold_label_and_protected_variable_label += (
            _predicted_label_counts_by_gold_label_and_protected_variable_label
        )

    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.FloatTensor]]:
        """
        # Returns

        distances : `Dict[int, Dict[int, torch.FloatTensor]]`
            A dictionary mapping each class label y to a dictionary mapping each protected
            variable label a to the KL divergence or Wasserstein distance of P(C | A = a, Y = y) from
            P(C | Y = y). A distance of nearly 0 implies fairness on the basis of Separation.

        !!! Note
            If a class label is not present in Y conditioned on a protected variable label,
            the expected behavior is that the