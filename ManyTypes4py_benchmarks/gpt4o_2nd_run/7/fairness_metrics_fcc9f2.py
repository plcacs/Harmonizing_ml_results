from typing import Optional, Dict, Union
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
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
        self._num_classes = num_classes
        self._num_protected_variable_labels = num_protected_variable_labels
        self._predicted_label_counts = torch.zeros(num_classes)
        self._total_predictions = torch.tensor(0)
        self._predicted_label_counts_by_protected_variable_label = torch.zeros((num_protected_variable_labels, num_classes))
        if dist_metric == 'kl_divergence':
            self._dist_metric = kl_divergence
        elif dist_metric == 'wasserstein':
            self._dist_metric = wasserstein_distance
        else:
            raise ConfigurationError("supported distance metrics in initialization are 'kl_divergence' and 'wasserstein'")

    def __call__(self, predicted_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> None:
        predicted_labels, protected_variable_labels, mask = self.detach_tensors(predicted_labels, protected_variable_labels, mask)
        if predicted_labels.size() != protected_variable_labels.size():
            raise ConfigurationError('protected_variable_labels must be of same size as predicted_labels but found tensor of shape: {}'.format(protected_variable_labels.size()))
        if mask is not None and predicted_labels.size() != mask.size():
            raise ConfigurationError('mask must be of same size as predicted_labels but found tensor of shape: {}'.format(mask.size()))
        if (predicted_labels >= self._num_classes).any():
            raise ConfigurationError('predicted_labels contains an id >= {}, the number of classes.'.format(self._num_classes))
        if (protected_variable_labels >= self._num_protected_variable_labels).any():
            raise ConfigurationError('protected_variable_labels contains an id >= {}, the number of protected variable labels.'.format(self._num_protected_variable_labels))
        device = predicted_labels.device
        self._predicted_label_counts = self._predicted_label_counts.to(device)
        self._predicted_label_counts_by_protected_variable_label = self._predicted_label_counts_by_protected_variable_label.to(device)
        self._total_predictions = self._total_predictions.to(device)
        if mask is not None:
            predicted_labels = predicted_labels[mask]
            protected_variable_labels = protected_variable_labels[mask]
        else:
            predicted_labels = predicted_labels.flatten()
            protected_variable_labels = protected_variable_labels.flatten()
        _predicted_label_counts = predicted_labels.float().histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
        _total_predictions = torch.tensor(predicted_labels.nelement()).to(device)
        _predicted_label_counts_by_protected_variable_label = torch.zeros((self._num_protected_variable_labels, self._num_classes)).to(device)
        for a in range(self._num_protected_variable_labels):
            _predicted_label_counts_by_protected_variable_label[a] = predicted_labels[protected_variable_labels == a].float().histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
        if is_distributed():
            _predicted_label_counts = _predicted_label_counts.to(device)
            dist.all_reduce(_predicted_label_counts, op=dist.ReduceOp.SUM)
            _total_predictions = _total_predictions.to(device)
            dist.all_reduce(_total_predictions, op=dist.ReduceOp.SUM)
            _predicted_label_counts_by_protected_variable_label = _predicted_label_counts_by_protected_variable_label.to(device)
            dist.all_reduce(_predicted_label_counts_by_protected_variable_label, op=dist.ReduceOp.SUM)
        self._predicted_label_counts += _predicted_label_counts
        self._total_predictions += _total_predictions
        self._predicted_label_counts_by_protected_variable_label += _predicted_label_counts_by_protected_variable_label

    def get_metric(self, reset: bool = False) -> Dict[int, torch.FloatTensor]:
        distances = {}
        if self._total_predictions == 0:
            distances = {a: torch.tensor(float('nan')) for a in range(self._num_protected_variable_labels)}
            return distances
        C_dist = Categorical(self._predicted_label_counts / self._total_predictions)
        if self._dist_metric == wasserstein_distance:
            C_dist = C_dist.probs
        for a in range(self._num_protected_variable_labels):
            C_given_a_dist = Categorical(self._predicted_label_counts_by_protected_variable_label[a] / self._total_predictions)
            if self._dist_metric == kl_divergence:
                distances[a] = self._dist_metric(C_given_a_dist, C_dist)
            elif self._dist_metric == wasserstein_distance:
                C_given_a_dist = C_given_a_dist.probs
                label_values = torch.tensor(range(self._num_classes))
                distances[a] = self._dist_metric(label_values, label_values, C_given_a_dist, C_dist)
        if reset:
            self.reset()
        return distances

    def reset(self) -> None:
        self._predicted_label_counts = torch.zeros(self._num_classes)
        self._total_predictions = torch.tensor(0)
        self._predicted_label_counts_by_protected_variable_label = torch.zeros((self._num_protected_variable_labels, self._num_classes))

@Metric.register('separation')
class Separation(Metric):
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
        self._num_classes = num_classes
        self._num_protected_variable_labels = num_protected_variable_labels
        self._predicted_label_counts_by_gold_label = torch.zeros((num_classes, num_classes))
        self._total_predictions = torch.tensor(0)
        self._predicted_label_counts_by_gold_label_and_protected_variable_label = torch.zeros((num_classes, num_protected_variable_labels, num_classes))
        if dist_metric == 'kl_divergence':
            self._dist_metric = kl_divergence
        elif dist_metric == 'wasserstein':
            self._dist_metric = wasserstein_distance
        else:
            raise ConfigurationError("supported distance metrics in initialization are 'kl_divergence' and 'wasserstein'")

    def __call__(self, predicted_labels: torch.Tensor, gold_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> None:
        predicted_labels, gold_labels, protected_variable_labels, mask = self.detach_tensors(predicted_labels, gold_labels, protected_variable_labels, mask)
        if predicted_labels.size() != protected_variable_labels.size():
            raise ConfigurationError('protected_variable_labels must be of same size as predicted_labels but found tensor of shape: {}'.format(protected_variable_labels.size()))
        if predicted_labels.size() != gold_labels.size():
            raise ConfigurationError('gold_labels must be of same size as predicted_labels but found tensor of shape: {}'.format(gold_labels.size()))
        if mask is not None and predicted_labels.size() != mask.size():
            raise ConfigurationError('mask must be of same size as predicted_labels but found tensor of shape: {}'.format(mask.size()))
        if (predicted_labels >= self._num_classes).any():
            raise ConfigurationError('predicted_labels contains an id >= {}, the number of classes.'.format(self._num_classes))
        if (gold_labels >= self._num_classes).any():
            raise ConfigurationError('gold_labels contains an id >= {}, the number of classes.'.format(self._num_classes))
        if (protected_variable_labels >= self._num_protected_variable_labels).any():
            raise ConfigurationError('protected_variable_labels contains an id >= {}, the number of protected variable labels.'.format(self._num_protected_variable_labels))
        device = predicted_labels.device
        self._predicted_label_counts_by_gold_label = self._predicted_label_counts_by_gold_label.to(device)
        self._predicted_label_counts_by_gold_label_and_protected_variable_label = self._predicted_label_counts_by_gold_label_and_protected_variable_label.to(device)
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
        _predicted_label_counts_by_gold_label = torch.zeros((self._num_classes, self._num_classes)).to(device)
        _predicted_label_counts_by_gold_label_and_protected_variable_label = torch.zeros((self._num_classes, self._num_protected_variable_labels, self._num_classes)).to(device)
        for y in range(self._num_classes):
            _predicted_label_counts_by_gold_label[y] = predicted_labels[gold_labels == y].float().histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
            for a in range(self._num_protected_variable_labels):
                _predicted_label_counts_by_gold_label_and_protected_variable_label[y][a] = predicted_labels[(gold_labels == y) & (protected_variable_labels == a)].float().histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
        if is_distributed():
            _total_predictions = _total_predictions.to(device)
            dist.all_reduce(_total_predictions, op=dist.ReduceOp.SUM)
            _predicted_label_counts_by_gold_label = _predicted_label_counts_by_gold_label.to(device)
            dist.all_reduce(_predicted_label_counts_by_gold_label[y], op=dist.ReduceOp.SUM)
            _predicted_label_counts_by_gold_label_and_protected_variable_label = _predicted_label_counts_by_gold_label_and_protected_variable_label.to(device)
            dist.all_reduce(_predicted_label_counts_by_gold_label_and_protected_variable_label, op=dist.ReduceOp.SUM)
        self._total_predictions += _total_predictions
        self._predicted_label_counts_by_gold_label += _predicted_label_counts_by_gold_label
        self._predicted_label_counts_by_gold_label_and_protected_variable_label += _predicted_label_counts_by_gold_label_and_protected_variable_label

    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.FloatTensor]]:
        distances = {}
        if self._total_predictions == 0:
            distances = {y: {a: torch.tensor(float('nan')) for a in range(self._num_protected_variable_labels)} for y in range(self._num_classes)}
            return distances
        for y in range(self._num_classes):
            probs = self._predicted_label_counts_by_gold_label[y] / self._total_predictions
            C_given_y_dist = Categorical(probs)
            if self._dist_metric == wasserstein_distance:
                C_given_y_dist = C_given_y_dist.probs
            distances[y] = {}
            for a in range(self._num_protected_variable_labels):
                probs = self._predicted_label_counts_by_gold_label_and_protected_variable_label[y][a] / self._total_predictions
                if self._dist_metric == kl_divergence:
                    if probs.sum() == 0:
                        distances[y][a] = torch.tensor(float('nan'))
                        continue
                    C_given_a_and_y_dist = Categorical(probs)
                    distances[y][a] = self._dist_metric(C_given_a_and_y_dist, C_given_y_dist)
                elif self._dist_metric == wasserstein_distance:
                    C_given_a_and_y_dist = Categorical(probs).probs
                    label_values = torch.tensor(range(self._num_classes))
                    distances[y][a] = self._dist_metric(label_values, label_values, C_given_a_and_y_dist, C_given_y_dist)
        if reset:
            self.reset()
        return distances

    def reset(self) -> None:
        self._predicted_label_counts_by_gold_label = torch.zeros((self._num_classes, self._num_classes))
        self._total_predictions = torch.tensor(0)
        self._predicted_label_counts_by_gold_label_and_protected_variable_label = torch.zeros((self._num_classes, self._num_protected_variable_labels, self._num_classes))

@Metric.register('sufficiency')
class Sufficiency(Metric):
    def __init__(self, num_classes: int, num_protected_variable_labels: int, dist_metric: str = 'kl_divergence') -> None:
        self._num_classes = num_classes
        self._num_protected_variable_labels = num_protected_variable_labels
        self._gold_label_counts_by_predicted_label = torch.zeros((num_classes, num_classes))
        self._total_predictions = torch.tensor(0)
        self._gold_label_counts_by_predicted_label_and_protected_variable_label = torch.zeros((num_classes, num_protected_variable_labels, num_classes))
        if dist_metric == 'kl_divergence':
            self._dist_metric = kl_divergence
        elif dist_metric == 'wasserstein':
            self._dist_metric = wasserstein_distance
        else:
            raise ConfigurationError("supported distance metrics in initialization are 'kl_divergence' and 'wasserstein'")

    def __call__(self, predicted_labels: torch.Tensor, gold_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> None:
        predicted_labels, gold_labels, protected_variable_labels, mask = self.detach_tensors(predicted_labels, gold_labels, protected_variable_labels, mask)
        if predicted_labels.size() != protected_variable_labels.size():
            raise ConfigurationError('protected_variable_labels must be of same size as predicted_labels but found tensor of shape: {}'.format(protected_variable_labels.size()))
        if predicted_labels.size() != gold_labels.size():
            raise ConfigurationError('gold_labels must be of same size as predicted_labels but found tensor of shape: {}'.format(gold_labels.size()))
        if mask is not None and predicted_labels.size() != mask.size():
            raise ConfigurationError('mask must be of same size as predicted_labels but found tensor of shape: {}'.format(mask.size()))
        if (predicted_labels >= self._num_classes).any():
            raise ConfigurationError('predicted_labels contains an id >= {}, the number of classes.'.format(self._num_classes))
        if (gold_labels >= self._num_classes).any():
            raise ConfigurationError('gold_labels contains an id >= {}, the number of classes.'.format(self._num_classes))
        if (protected_variable_labels >= self._num_protected_variable_labels).any():
            raise ConfigurationError('protected_variable_labels contains an id >= {}, the number of protected variable labels.'.format(self._num_protected_variable_labels))
        device = predicted_labels.device
        self._gold_label_counts_by_predicted_label = self._gold_label_counts_by_predicted_label.to(device)
        self._gold_label_counts_by_predicted_label_and_protected_variable_label = self._gold_label_counts_by_predicted_label_and_protected_variable_label.to(device)
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
        _gold_label_counts_by_predicted_label = torch.zeros((self._num_classes, self._num_classes)).to(device)
        _gold_label_counts_by_predicted_label_and_protected_variable_label = torch.zeros((self._num_classes, self._num_protected_variable_labels, self._num_classes)).to(device)
        for c in range(self._num_classes):
            _gold_label_counts_by_predicted_label[c] = gold_labels[predicted_labels == c].float().histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
            for a in range(self._num_protected_variable_labels):
                _gold_label_counts_by_predicted_label_and_protected_variable_label[c][a] = gold_labels[(predicted_labels == c) & (protected_variable_labels == a)].float().histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
        if is_distributed():
            _total_predictions = _total_predictions.to(device)
            dist.all_reduce(_total_predictions, op=dist.ReduceOp.SUM)
            _gold_label_counts_by_predicted_label = _gold_label_counts_by_predicted_label.to(device)
            dist.all_reduce(_gold_label_counts_by_predicted_label[c], op=dist.ReduceOp.SUM)
            _gold_label_counts_by_predicted_label_and_protected_variable_label = _gold_label_counts_by_predicted_label_and_protected_variable_label.to(device)
            dist.all_reduce(_gold_label_counts_by_predicted_label_and_protected_variable_label, op=dist.ReduceOp.SUM)
        self._total_predictions += _total_predictions
        self._gold_label_counts_by_predicted_label += _gold_label_counts_by_predicted_label
        self._gold_label_counts_by_predicted_label_and_protected_variable_label += _gold_label_counts_by_predicted_label_and_protected_variable_label

    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.FloatTensor]]:
        distances = {}
        if self._total_predictions == 0:
            distances = {c: {a: torch.tensor(float('nan')) for a in range(self._num_protected_variable_labels)} for c in range(self._num_classes)}
            return distances
        for c in range(self._num_classes):
            probs = self._gold_label_counts_by_predicted_label[c] / self._total_predictions
            if self._dist_metric == kl_divergence:
                if probs.sum() == 0:
                    distances[c] = {a: torch.tensor(float('nan')) for a in range(self._num_protected_variable_labels)}
                    continue
            Y_given_c_dist = Categorical(probs)
            distances[c] = {}
            if self._dist_metric == wasserstein_distance:
                Y_given_c_dist = Y_given_c_dist.probs
            for a in range(self._num_protected_variable_labels):
                probs = self._gold_label_counts_by_predicted_label_and_protected_variable_label[c][a] / self._total_predictions
                if self._dist_metric == kl_divergence:
                    if probs.sum() == 0:
                        distances[c][a] = torch.tensor(float('nan'))
                        continue
                    Y_given_a_and_c_dist = Categorical(probs)
                    distances[c][a] = self._dist_metric(Y_given_a_and_c_dist, Y_given_c_dist)
                elif self._dist_metric == wasserstein_distance:
                    Y_given_a_and_c_dist = Categorical(probs).probs
                    label_values = torch.tensor(range(self._num_classes))
                    distances[c][a] = self._dist_metric(label_values, label_values, Y_given_a_and_c_dist, Y_given_c_dist)
        if reset:
            self.reset()
        return distances

    def reset(self) -> None:
        self._gold_label_counts_by_predicted_label = torch.zeros((self._num_classes, self._num_classes))
        self._total_predictions = torch.tensor(0)
        self._gold_label_counts_by_predicted_label_and_protected_variable_label = torch.zeros((self._num_classes, self._num_protected_variable_labels, self._num_classes))
