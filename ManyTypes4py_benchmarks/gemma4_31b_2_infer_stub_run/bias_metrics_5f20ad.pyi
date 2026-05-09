from typing import Optional, Dict, Union, List, Callable
import torch
from allennlp.training.metrics.metric import Metric

class WordEmbeddingAssociationTest:
    """
    Word Embedding Association Test (WEAT) score measures the unlikelihood there is no
    difference between two sets of target words in terms of their relative similarity
    to two sets of attribute words by computing the probability that a random
    permutation of attribute words would produce the observed (or greater) difference
    in sample means. Analog of Implicit Association Test from psychology for word embeddings.
    """
    def __call__(self, target_embeddings1: torch.Tensor, target_embeddings2: torch.Tensor, attribute_embeddings1: torch.Tensor, attribute_embeddings2: torch.Tensor) -> torch.Tensor: ...

class EmbeddingCoherenceTest:
    """
    Embedding Coherence Test (ECT) score measures if groups of words
    have stereotypical associations by computing the Spearman Coefficient
    of lists of attribute embeddings sorted based on their similarity to
    target embeddings.
    """
    def __call__(self, target_embeddings1: torch.Tensor, target_embeddings2: torch.Tensor, attribute_embeddings: torch.Tensor) -> torch.Tensor: ...
    def _get_ranks(self, x: torch.Tensor) -> torch.Tensor: ...
    def spearman_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

class NaturalLanguageInference(Metric):
    """
    Natural language inference scores measure the effect biased associations have on decisions
    made downstream, given neutrally-constructed pairs of sentences differing only in the subject.
    """
    neutral_label: int
    taus: List[float]
    _nli_probs_sum: float
    _num_neutral_predictions: float
    _num_neutral_above_taus: Dict[float, float]
    _total_predictions: int

    def __init__(self, neutral_label: int = 2, taus: List[float] = ...) -> None: ...
    def __call__(self, nli_probabilities: torch.Tensor) -> None: ...
    def get_metric(self, reset: bool = False) -> Dict[str, float]: ...
    def reset(self) -> None: ...

class AssociationWithoutGroundTruth(Metric):
    """
    Association without ground truth, from: Aka, O.; Burke, K.; Bäuerle, A.;
    Greer, C.; and Mitchell, M. 2021. Measuring model biases in the absence of ground
    truth. arXiv preprint arXiv:2103.03417.
    """
    _num_classes: int
    _num_protected_variable_labels: int
    _joint_counts_by_protected_variable_label: torch.Tensor
    _protected_variable_label_counts: torch.Tensor
    _y_counts: torch.Tensor
    _total_predictions: torch.Tensor
    IMPLEMENTED_ASSOCIATION_METRICS: set[str]
    association_metric: str
    gap_func: Callable[[int], Union[torch.Tensor, Dict[int, torch.Tensor]]]

    def __init__(self, num_classes: int, num_protected_variable_labels: int, association_metric: str = 'npmixy', gap_type: str = 'ova') -> None: ...
    def __call__(self, predicted_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None: ...
    def get_metric(self, reset: bool = False) -> Dict[int, Union[torch.Tensor, Dict[int, torch.Tensor]]]: ...
    def reset(self) -> None: ...
    def _ova_gap(self, x: int) -> torch.Tensor: ...
    def _pairwise_gaps(self, x: int) -> Dict[int, torch.Tensor]: ...
    def _all_pmi_terms(self) -> torch.Tensor: ...
    def detach_tensors(self, predicted_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: ...