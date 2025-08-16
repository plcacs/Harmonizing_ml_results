from typing import Dict, Optional
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.fairness.bias_direction_wrappers import BiasDirectionWrapper
from allennlp.modules.feedforward import FeedForward
from allennlp.models import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import find_embedding_layer
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.callbacks.backward import OnBackwardException
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer

@Model.register('adversarial_bias_mitigator')
class AdversarialBiasMitigator(Model):
    def __init__(self, vocab: Vocabulary, predictor: Model, adversary: Model, bias_direction: BiasDirectionWrapper, predictor_output_key: str, **kwargs):
        ...

    def train(self, mode: bool = True):
        ...

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        ...

    def forward_on_instance(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        ...

    def forward_on_instances(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        ...

    def get_regularization_penalty(self, *args, **kwargs) -> torch.Tensor:
        ...

    def get_parameters_for_histogram_logging(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        ...

    def get_parameters_for_histogram_tensorboard_logging(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        ...

    def make_output_human_readable(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        ...

    def get_metrics(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        ...

    def _get_prediction_device(self, *args, **kwargs) -> torch.device:
        ...

    def _maybe_warn_for_unseparable_batches(self, *args, **kwargs) -> None:
        ...

    def extend_embedder_vocab(self, *args, **kwargs) -> None:
        ...

@Model.register('feedforward_regression_adversary')
class FeedForwardRegressionAdversary(Model):
    def __init__(self, vocab: Vocabulary, feedforward: FeedForward, initializer: Optional[InitializerApplicator] = InitializerApplicator(), **kwargs):
        ...

    def forward(self, input: torch.FloatTensor, label: torch.FloatTensor) -> Dict[str, torch.Tensor]:
        ...

@TrainerCallback.register('adversarial_bias_mitigator_backward')
class AdversarialBiasMitigatorBackwardCallback(TrainerCallback):
    def __init__(self, serialization_dir: str, adversary_loss_weight: float = 1.0):
        ...

    def on_backward(self, trainer, batch_outputs, backward_called, **kwargs) -> bool:
        ...
