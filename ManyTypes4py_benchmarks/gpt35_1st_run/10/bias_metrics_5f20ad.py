from typing import Optional, Dict, Union, List
import torch
import torch.distributed as dist
from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics.metric import Metric

class WordEmbeddingAssociationTest:
    def __call__(self, target_embeddings1: torch.Tensor, target_embeddings2: torch.Tensor, attribute_embeddings1: torch.Tensor, attribute_embeddings2: torch.Tensor) -> torch.FloatTensor:
        ...

class EmbeddingCoherenceTest:
    def __call__(self, target_embeddings1: torch.Tensor, target_embeddings2: torch.Tensor, attribute_embeddings: torch.Tensor) -> torch.FloatTensor:
        ...

    def _get_ranks(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def spearman_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ...

@Metric.register('nli')
class NaturalLanguageInference(Metric):
    def __init__(self, neutral_label: int = 2, taus: List[float] = [0.5, 0.7]):
        ...

    def __call__(self, nli_probabilities: torch.Tensor):
        ...

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        ...

    def reset(self):
        ...

@Metric.register('association_without_ground_truth')
class AssociationWithoutGroundTruth(Metric):
    def __init__(self, num_classes: int, num_protected_variable_labels: int, association_metric: str = 'npmixy', gap_type: str = 'ova'):
        ...

    def __call__(self, predicted_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        ...

    def get_metric(self, reset: bool = False) -> Dict[int, Union[torch.FloatTensor, Dict[int, torch.FloatTensor]]]:
        ...

    def reset(self):
        ...

    def _ova_gap(self, x: int) -> torch.Tensor:
        ...

    def _pairwise_gaps(self, x: int) -> Dict[int, torch.Tensor]:
        ...

    def _all_pmi_terms(self) -> torch.Tensor:
        ...
