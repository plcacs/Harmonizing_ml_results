"""
A Model wrapper to mitigate biases in
contextual embeddings during finetuning
on a downstream task and test time.

Based on: Dev, S., Li, T., Phillips, J.M., & Srikumar, V. (2020).
[On Measuring and Mitigating Biased Inferences of Word Embeddings]
(https://api.semanticscholar.org/CorpusID:201670701).
ArXiv, abs/1908.09369.
"""
from typing import Any, Dict, List, Optional, Union, Tuple

from allennlp.fairness.bias_mitigator_wrappers import BiasMitigatorWrapper
from allennlp.common.lazy import Lazy
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import find_embedding_layer

@Model.register('bias_mitigator_applicator')
class BiasMitigatorApplicator(Model):
    """
    Wrapper class to apply bias mitigation to any pretrained Model.

    # Parameters

    vocab : `Vocabulary`
        Vocabulary of base model.
    base_model : `Model`
        Base model for which to mitigate biases.
    bias_mitigator : `Lazy[BiasMitigatorWrapper]`
        Bias mitigator to apply to base model.
    """

    def __init__(
        self, 
        vocab: Vocabulary, 
        base_model: Model, 
        bias_mitigator: Lazy[BiasMitigatorWrapper], 
        **kwargs: Any
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.base_model = base_model
        embedding_layer = find_embedding_layer(self.base_model)
        self.bias_mitigator = bias_mitigator.construct(embedding_layer=embedding_layer)
        embedding_layer.register_forward_hook(self.bias_mitigator)
        self.vocab = self.base_model.vocab
        self._regularizer = self.base_model._regularizer

    def train(self, mode: bool = True) -> "BiasMitigatorApplicator":
        super().train(mode)
        self.base_model.train(mode)
        self.bias_mitigator.train(mode)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.base_model.forward(*args, **kwargs)

    def forward_on_instance(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.base_model.forward_on_instance(*args, **kwargs)

    def forward_on_instances(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.base_model.forward_on_instances(*args, **kwargs)

    def get_regularization_penalty(self, *args: Any, **kwargs: Any) -> Union[float, torch.Tensor]:
        return self.base_model.get_regularization_penalty(*args, **kwargs)

    def get_parameters_for_histogram_logging(self, *args: Any, **kwargs: Any) -> List[Tuple[str, torch.Tensor]]:
        return self.base_model.get_parameters_for_histogram_logging(*args, **kwargs)

    def get_parameters_for_histogram_tensorboard_logging(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        return self.base_model.get_parameters_for_histogram_tensorboard_logging(*args, **kwargs)

    def make_output_human_readable(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.base_model.make_output_human_readable(*args, **kwargs)

    def get_metrics(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.base_model.get_metrics(*args, **kwargs)

    def _get_prediction_device(self, *args: Any, **kwargs: Any) -> torch.device:
        return self.base_model._get_prediction_device(*args, **kwargs)

    def _maybe_warn_for_unseparable_batches(self, *args: Any, **kwargs: Any) -> None:
        return self.base_model._maybe_warn_for_unseparable_batches(*args, **kwargs)

    def extend_embedder_vocab(self, *args: Any, **kwargs: Any) -> None:
        return self.base_model.extend_embedder_vocab(*args, **kwargs)
