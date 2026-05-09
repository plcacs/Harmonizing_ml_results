class WordEmbeddingAssociationTest:
    def __call__(self, target_embeddings1: torch.Tensor, target_embeddings2: torch.Tensor, attribute_embeddings1: torch.Tensor, attribute_embeddings2: torch.Tensor) -> torch.FloatTensor:
        ...

class EmbeddingCoherenceTest:
    def __call__(self, target_embeddings1: torch.Tensor, target_embeddings2: torch.Tensor, attribute_embeddings: torch.Tensor) -> torch.FloatTensor:
        ...

class NaturalLanguageInference:
    def __init__(self, neutral_label: int = 2, taus: List[float] = [0.5, 0.7]):
        ...

    def __call__(self, nli_probabilities: torch.Tensor) -> None:
        ...

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        ...

class AssociationWithoutGroundTruth:
    def __init__(self, num_classes: int, num_protected_variable_labels: int, association_metric: str = 'npmixy', gap_type: str = 'ova'):
        ...

    def __call__(self, predicted_labels: torch.Tensor, protected_variable_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> None:
        ...

    def get_metric(self, reset: bool = False) -> Dict[int, Union[torch.FloatTensor, Dict[int, torch.FloatTensor]]]:
        ...
