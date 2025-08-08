from typing import Dict, Optional, Any, Tuple, Union, List
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
from torch import Tensor


@Model.register('adversarial_bias_mitigator')
class AdversarialBiasMitigator(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        predictor: Model,
        adversary: Model,
        bias_direction: BiasDirectionWrapper,
        predictor_output_key: str,
        **kwargs: Any
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.predictor = predictor
        self.adversary = adversary
        embedding_layer = find_embedding_layer(self.predictor)
        self.bias_direction = bias_direction
        self.predetermined_bias_direction = self.bias_direction(embedding_layer)
        self._adversary_label_hook = _AdversaryLabelHook(self.predetermined_bias_direction)
        embedding_layer.register_forward_hook(self._adversary_label_hook)
        self.vocab = self.predictor.vocab
        self._regularizer = self.predictor._regularizer
        self.predictor_output_key = predictor_output_key

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self.predictor.train(mode)
        self.adversary.train(mode)
        self.bias_direction.train(mode)

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        predictor_output_dict = self.predictor.forward(*args, **kwargs)
        adversary_output_dict = self.adversary.forward(
            predictor_output_dict[self.predictor_output_key],
            self._adversary_label_hook.adversary_label
        )
        adversary_output_dict = {('adversary_' + k): v for k, v in adversary_output_dict.items()}
        output_dict = {**predictor_output_dict, **adversary_output_dict}
        return output_dict

    def forward_on_instance(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.predictor.forward_on_instance(*args, **kwargs)

    def forward_on_instances(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.predictor.forward_on_instances(*args, **kwargs)

    def get_regularization_penalty(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.predictor.get_regularization_penalty(*args, **kwargs)

    def get_parameters_for_histogram_logging(self, *args: Any, **kwargs: Any) -> List[str]:
        return self.predictor.get_parameters_for_histogram_logging(*args, **kwargs)

    def get_parameters_for_histogram_tensorboard_logging(self, *args: Any, **kwargs: Any) -> Dict[str, List[str]]:
        return self.predictor.get_parameters_for_histogram_tensorboard_logging(*args, **kwargs)

    def make_output_human_readable(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.predictor.make_output_human_readable(*args, **kwargs)

    def get_metrics(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        return self.predictor.get_metrics(*args, **kwargs)

    def _get_prediction_device(self, *args: Any, **kwargs: Any) -> torch.device:
        return self.predictor._get_prediction_device(*args, **kwargs)

    def _maybe_warn_for_unseparable_batches(self, *args: Any, **kwargs: Any) -> None:
        return self.predictor._maybe_warn_for_unseparable_batches(*args, **kwargs)

    def extend_embedder_vocab(self, *args: Any, **kwargs: Any) -> None:
        return self.predictor.extend_embedder_vocab(*args, **kwargs)


@Model.register('feedforward_regression_adversary')
class FeedForwardRegressionAdversary(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        feedforward: FeedForward,
        initializer: Optional[InitializerApplicator] = InitializerApplicator(),
        **kwargs: Any
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._feedforward = feedforward
        self._loss = torch.nn.MSELoss()
        initializer(self)

    def forward(self, input: Tensor, label: Tensor) -> Dict[str, Tensor]:
        pred = self._feedforward(input)
        return {'loss': self._loss(pred, label)}


@TrainerCallback.register('adversarial_bias_mitigator_backward')
class AdversarialBiasMitigatorBackwardCallback(TrainerCallback):
    def __init__(self, serialization_dir: str, adversary_loss_weight: float = 1.0) -> None:
        super().__init__(serialization_dir)
        self.adversary_loss_weight = adversary_loss_weight

    def on_backward(
        self,
        trainer: GradientDescentTrainer,
        batch_outputs: Dict[str, Tensor],
        backward_called: bool,
        **kwargs: Any
    ) -> bool:
        if backward_called:
            raise OnBackwardException()
        if not hasattr(trainer.model, 'predictor') or not hasattr(trainer.model, 'adversary'):
            raise ConfigurationError(
                'Model is expected to have `predictor` and `adversary` data members.'
            )
        trainer.optimizer.zero_grad()
        batch_outputs['adversary_loss'].backward(retain_graph=True)
        adversary_loss_grad = {
            name: param.grad.clone()
            for name, param in trainer.model.predictor.named_parameters()
            if param.grad is not None
        }
        trainer.model.predictor.zero_grad()
        batch_outputs['loss'].backward()
        with torch.no_grad():
            for name, param in trainer.model.predictor.named_parameters():
                if param.grad is not None:
                    unit_adversary_loss_grad = adversary_loss_grad[name] / torch.linalg.norm(adversary_loss_grad[name])
                    param.grad -= (param.grad * unit_adversary_loss_grad * unit_adversary_loss_grad).sum()
                    param.grad -= (self.adversary_loss_weight * adversary_loss_grad[name])
        batch_outputs['adversary_loss'] = batch_outputs['adversary_loss'].detach()
        return True


class _AdversaryLabelHook:
    def __init__(self, predetermined_bias_direction: Tensor) -> None:
        self.predetermined_bias_direction = predetermined_bias_direction
        self.adversary_label: Tensor

    def __call__(self, module: torch.nn.Module, module_in: Tuple[Tensor], module_out: Tensor) -> None:
        with torch.no_grad():
            module_out = module_out.mean(dim=1)
            self.adversary_label = torch.matmul(
                module_out,
                self.predetermined_bias_direction.to(module_out.device)
            ).unsqueeze(-1)
