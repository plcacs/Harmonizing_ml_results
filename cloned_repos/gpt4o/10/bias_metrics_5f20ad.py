from typing import Optional, Dict, Union, List
import torch
import torch.distributed as dist
from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics.metric import Metric

class WordEmbeddingAssociationTest:
    def __call__(self, target_embeddings1: torch.Tensor, target_embeddings2: torch.Tensor, attribute_embeddings1: torch.Tensor, attribute_embeddings2: torch.Tensor) -> torch.FloatTensor:
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise ConfigurationError('target_embeddings1 and target_embeddings2 must have at least two dimensions.')
        if attribute_embeddings1.ndim < 2 or attribute_embeddings2.ndim < 2:
            raise ConfigurationError('attribute_embeddings1 and attribute_embeddings2 must have at least two dimensions.')
        if target_embeddings1.size() != target_embeddings2.size():
            raise ConfigurationError('target_embeddings1 and target_embeddings2 must be of the same size.')
        if attribute_embeddings1.size(dim=-1) != attribute_embeddings2.size(dim=-1) or attribute_embeddings1.size(dim=-1) != target_embeddings1.size(dim=-1):
            raise ConfigurationError('All embeddings must have the same dimensionality.')
        target_embeddings1 = target_embeddings1.flatten(end_dim=-2)
        target_embeddings2 = target_embeddings2.flatten(end_dim=-2)
        attribute_embeddings1 = attribute_embeddings1.flatten(end_dim=-2)
        attribute_embeddings2 = attribute_embeddings2.flatten(end_dim=-2)
        target_embeddings1 = torch.nn.functional.normalize(target_embeddings1, p=2, dim=-1)
        target_embeddings2 = torch.nn.functional.normalize(target_embeddings2, p=2, dim=-1)
        attribute_embeddings1 = torch.nn.functional.normalize(attribute_embeddings1, p=2, dim=-1)
        attribute_embeddings2 = torch.nn.functional.normalize(attribute_embeddings2, p=2, dim=-1)
        X_sim_A = torch.mm(target_embeddings1, attribute_embeddings1.t())
        X_sim_B = torch.mm(target_embeddings1, attribute_embeddings2.t())
        Y_sim_A = torch.mm(target_embeddings2, attribute_embeddings1.t())
        Y_sim_B = torch.mm(target_embeddings2, attribute_embeddings2.t())
        X_union_Y_sim_A = torch.cat([X_sim_A, Y_sim_A])
        X_union_Y_sim_B = torch.cat([X_sim_B, Y_sim_B])
        s_X_A_B = torch.mean(X_sim_A, dim=-1) - torch.mean(X_sim_B, dim=-1)
        s_Y_A_B = torch.mean(Y_sim_A, dim=-1) - torch.mean(Y_sim_B, dim=-1)
        s_X_Y_A_B = torch.mean(s_X_A_B) - torch.mean(s_Y_A_B)
        S_X_union_Y_A_B = torch.mean(X_union_Y_sim_A, dim=-1) - torch.mean(X_union_Y_sim_B, dim=-1)
        return s_X_Y_A_B / torch.std(S_X_union_Y_A_B, unbiased=False)

class EmbeddingCoherenceTest:
    def __call__(self, target_embeddings1: torch.Tensor, target_embeddings2: torch.Tensor, attribute_embeddings: torch.Tensor) -> torch.FloatTensor:
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise ConfigurationError('target_embeddings1 and target_embeddings2 must have at least two dimensions.')
        if attribute_embeddings.ndim < 2:
            raise ConfigurationError('attribute_embeddings must have at least two dimensions.')
        if target_embeddings1.size() != target_embeddings2.size():
            raise ConfigurationError('target_embeddings1 and target_embeddings2 must be of the same size.')
        if attribute_embeddings.size(dim=-1) != target_embeddings1.size(dim=-1):
            raise ConfigurationError('All embeddings must have the same dimensionality.')
        mean_target_embedding1 = target_embeddings1.flatten(end_dim=-2).mean(dim=0)
        mean_target_embedding2 = target_embeddings2.flatten(end_dim=-2).mean(dim=0)
        attribute_embeddings = attribute_embeddings.flatten(end_dim=-2)
        mean_target_embedding1 = torch.nn.functional.normalize(mean_target_embedding1, p=2, dim=-1)
        mean_target_embedding2 = torch.nn.functional.normalize(mean_target_embedding2, p=2, dim=-1)
        attribute_embeddings = torch.nn.functional.normalize(attribute_embeddings, p=2, dim=-1)
        AB_sim_m = torch.matmul(attribute_embeddings, mean_target_embedding1)
        AB_sim_f = torch.matmul(attribute_embeddings, mean_target_embedding2)
        return self.spearman_correlation(AB_sim_m, AB_sim_f)

    def _get_ranks(self, x: torch.Tensor) -> torch.Tensor:
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp)
        ranks[tmp] = torch.arange(x.size(0), device=ranks.device)
        return ranks

    def spearman_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
        x_rank = self._get_ranks(x)
        y_rank = self._get_ranks(y)
        n = x.size(0)
        upper = 6 * torch.sum((x_rank - y_rank).pow(2))
        down = n * (n ** 2 - 1.0)
        return 1.0 - upper / down

@Metric.register('nli')
class NaturalLanguageInference(Metric):
    def __init__(self, neutral_label: int = 2, taus: List[float] = [0.5, 0.7]) -> None:
        self.neutral_label = neutral_label
        self.taus = taus
        self._nli_probs_sum = 0.0
        self._num_neutral_predictions = 0.0
        self._num_neutral_above_taus = {tau: 0.0 for tau in taus}
        self._total_predictions = 0

    def __call__(self, nli_probabilities: torch.Tensor) -> None:
        nli_probabilities = nli_probabilities.detach()
        if nli_probabilities.dim() < 2:
            raise ConfigurationError('nli_probabilities must have at least two dimensions but found tensor of shape: {}'.format(nli_probabilities.size()))
        if nli_probabilities.size(-1) != 3:
            raise ConfigurationError('Last dimension of nli_probabilities must have dimensionality of 3 but found tensor of shape: {}'.format(nli_probabilities.size()))
        _nli_neutral_probs = nli_probabilities[..., self.neutral_label]
        self._nli_probs_sum += dist_reduce_sum(_nli_neutral_probs.sum().item())
        self._num_neutral_predictions += dist_reduce_sum((nli_probabilities.argmax(dim=-1) == self.neutral_label).float().sum().item())
        for tau in self.taus:
            self._num_neutral_above_taus[tau] += dist_reduce_sum((_nli_neutral_probs > tau).float().sum().item())
        self._total_predictions += dist_reduce_sum(_nli_neutral_probs.numel())

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self._total_predictions == 0:
            nli_scores = {'net_neutral': 0.0, 'fraction_neutral': 0.0, **{'threshold_{}'.format(tau): 0.0 for tau in self.taus}}
        else:
            nli_scores = {'net_neutral': self._nli_probs_sum / self._total_predictions, 'fraction_neutral': self._num_neutral_predictions / self._total_predictions, **{'threshold_{}'.format(tau): self._num_neutral_above_taus[tau] / self._total_predictions for tau in self.taus}}
        if reset:
            self.reset()
        return nli_scores

    def reset(self) -> None:
        self._nli_probs_sum = 0.0
        self._num_neutral_predictions = 0.0
        self._num_neutral_above_taus = {tau: 0.0 for tau in self.taus}
        self._total_predictions = 0

@Metric.register('association_without_ground_truth')
class AssociationWithoutGroundTruth(Metric):
    def __init__(self, num_classes: int, num_protected_variable_labels: int, association_metric: str = 'npmixy', gap_type: str = 'ova') -> None:
        self._num_classes = num_classes
        self._num_protected_variable_labels = num_protected_variable_labels
        self._joint_counts_by_protected_variable_label = torch.zeros((num_protected_variable_labels, num_classes))
        self._protected_variable_label_counts = torch.zeros(num_protected_variable_labels)
        self._y_counts = torch.zeros(num_classes)
        self._total_predictions = torch.tensor(0)
        self.IMPLEMENTED_ASSOCIATION_METRICS = set(['npmixy', 'npmiy', 'pmisq', 'pmi'])
        if association_metric in self.IMPLEMENTED_ASSOCIATION_METRICS:
            self.association_metric = association_metric
        else:
            raise NotImplementedError(f'Association metric {association_metric} has not been implemented!')
        if gap_type == 'ova':
            self.gap_func = self._ova_gap
        elif gap_type == 'pairwise':
            self.gap_func = self._pairwise_gaps
        else:
            raise NotImplementedError(f'Gap type {gap_type} has not been implemented!')

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
        self._joint_counts_by_protected_variable_label = self._joint_counts_by_protected_variable_label.to(device)
        self._protected_variable_label_counts = self._protected_variable_label_counts.to(device)
        self._y_counts = self._y_counts.to(device)
        self._total_predictions = self._total_predictions.to(device)
        if mask is not None:
            predicted_labels = predicted_labels[mask]
            protected_variable_labels = protected_variable_labels[mask]
        else:
            predicted_labels = predicted_labels.flatten()
            protected_variable_labels = protected_variable_labels.flatten()
        _total_predictions = torch.tensor(predicted_labels.nelement()).to(device)
        _y_counts = torch.zeros(self._num_classes).to(device)
        _y_counts = torch.zeros_like(_y_counts, dtype=predicted_labels.dtype).scatter_add_(0, predicted_labels, torch.ones_like(predicted_labels))
        _joint_counts_by_protected_variable_label = torch.zeros((self._num_protected_variable_labels, self._num_classes)).to(device)
        _protected_variable_label_counts = torch.zeros(self._num_protected_variable_labels).to(device)
        for x in range(self._num_protected_variable_labels):
            x_mask = (protected_variable_labels == x).long()
            _joint_counts_by_protected_variable_label[x] = torch.zeros(self._num_classes).to(device)
            _joint_counts_by_protected_variable_label[x] = torch.zeros_like(_joint_counts_by_protected_variable_label[x], dtype=x_mask.dtype).scatter_add_(0, predicted_labels, x_mask)
            _protected_variable_label_counts[x] = torch.tensor(x_mask.sum()).to(device)
        if is_distributed():
            _total_predictions = _total_predictions.to(device)
            dist.all_reduce(_total_predictions, op=dist.ReduceOp.SUM)
            _y_counts = _y_counts.to(device)
            dist.all_reduce(_y_counts, op=dist.ReduceOp.SUM)
            _joint_counts_by_protected_variable_label = _joint_counts_by_protected_variable_label.to(device)
            dist.all_reduce(_joint_counts_by_protected_variable_label, op=dist.ReduceOp.SUM)
            _protected_variable_label_counts = _protected_variable_label_counts.to(device)
            dist.all_reduce(_protected_variable_label_counts, op=dist.ReduceOp.SUM)
        self._total_predictions += _total_predictions
        self._y_counts += _y_counts
        self._joint_counts_by_protected_variable_label += _joint_counts_by_protected_variable_label
        self._protected_variable_label_counts += _protected_variable_label_counts

    def get_metric(self, reset: bool = False) -> Dict[int, Union[torch.FloatTensor, Dict[int, torch.FloatTensor]]]:
        gaps = {}
        for x in range(self._num_protected_variable_labels):
            gaps[x] = self.gap_func(x)
        if reset:
            self.reset()
        return gaps

    def reset(self) -> None:
        self._joint_counts_by_protected_variable_label = torch.zeros((self._num_protected_variable_labels, self._num_classes))
        self._protected_variable_label_counts = torch.zeros(self._num_protected_variable_labels)
        self._y_counts = torch.zeros(self._num_classes)
        self._total_predictions = torch.tensor(0)

    def _ova_gap(self, x: int) -> torch.FloatTensor:
        device = self._y_counts.device
        pmi_terms = self._all_pmi_terms()
        pmi_not_x = torch.sum(pmi_terms[torch.arange(self._num_protected_variable_labels, device=device) != x], dim=0)
        pmi_not_x /= self._num_protected_variable_labels - 1
        gap = pmi_terms[x] - pmi_not_x
        return torch.where(~gap.isinf(), gap, torch.tensor(float('nan')).to(device))

    def _pairwise_gaps(self, x: int) -> Dict[int, torch.FloatTensor]:
        device = self._y_counts.device
        pmi_terms = self._all_pmi_terms()
        pairwise_gaps = {}
        for not_x in range(self._num_protected_variable_labels):
            gap = pmi_terms[x] - pmi_terms[not_x]
            pairwise_gaps[not_x] = torch.where(~gap.isinf(), gap, torch.tensor(float('nan')).to(device))
        return pairwise_gaps

    def _all_pmi_terms(self) -> torch.FloatTensor:
        if self._total_predictions == 0:
            return torch.full((self._num_protected_variable_labels, self._num_classes), float('nan'))
        device = self._y_counts.device
        prob_y = torch.zeros(self._num_classes).to(device)
        torch.div(self._y_counts, self._total_predictions, out=prob_y)
        joint = torch.zeros((self._num_protected_variable_labels, self._num_classes)).to(device)
        torch.div(self._joint_counts_by_protected_variable_label, self._total_predictions, out=joint)
        if self.association_metric == 'pmisq':
            torch.square_(joint)
        pmi_terms = torch.log(torch.div(joint, (self._protected_variable_label_counts / self._total_predictions).unsqueeze(-1) * prob_y))
        if self.association_metric == 'npmixy':
            pmi_terms.div_(torch.log(joint))
        elif self.association_metric == 'npmiy':
            pmi_terms.div_(torch.log(prob_y))
        return pmi_terms
