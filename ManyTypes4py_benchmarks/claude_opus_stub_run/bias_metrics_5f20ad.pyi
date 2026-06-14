from typing import Dict, List, Optional, Union

import torch

from allennlp.training.metrics.metric import Metric


class WordEmbeddingAssociationTest:
    def __call__(
        self,
        target_embeddings1: torch.Tensor,
        target_embeddings2: torch.Tensor,
        attribute_embeddings1: torch.Tensor,
        attribute_embeddings2: torch.Tensor,
    ) -> torch.FloatTensor: ...


class EmbeddingCoherenceTest:
    def __call__(
        self,
        target_embeddings1: torch.Tensor,
        target_embeddings2: torch.Tensor,
        attribute_embeddings: torch.Tensor,
    ) -> torch.FloatTensor: ...
    def _get_ranks(self, x: torch.Tensor) -> torch.Tensor: ...
    def spearman_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor: ...


class NaturalLanguageInference(Metric):
    neutral_label: int
    taus: List[float]
    _nli_probs_sum: float
    _num_neutral_predictions: float
    _num_neutral_above_taus: Dict[float, float]
    _total_predictions: int

    def __init__(
        self,
        neutral_label: int = 2,
        taus: List[float] = ...,
    ) -> None: ...
    def __call__(self, nli_probabilities: torch.Tensor) -> None: ...  # type: ignore[override]
    def get_metric(self, reset: bool = False) -> Dict[str, float]: ...
    def reset(self) -> None: ...


class AssociationWithoutGroundTruth(Metric):
    _num_classes: int
    _num_protected_variable_labels: int
    _joint_counts_by_protected_variable_label: torch.Tensor
    _protected_variable_label_counts: torch.Tensor
    _y_counts: torch.Tensor
    _total_predictions: torch.Tensor
    IMPLEMENTED_ASSOCIATION_METRICS: set
    association_metric: str
    gap_func: object

    def __init__(
        self,
        num_classes: int,
        num_protected_variable_labels: int,
        association_metric: str = "npmixy",
        gap_type: str = "ova",
    ) -> None: ...
    def __call__(  # type: ignore[override]
        self,
        predicted_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None: ...
    def get_metric(
        self, reset: bool = False
    ) -> Dict[int, Union[torch.FloatTensor, Dict[int, torch.FloatTensor]]]: ...
    def reset(self) -> None: ...
    def _ova_gap(self, x: int) -> torch.FloatTensor: ...
    def _pairwise_gaps(self, x: int) -> Dict[int, torch.FloatTensor]: ...
    def _all_pmi_terms(self) -> torch.Tensor: ...