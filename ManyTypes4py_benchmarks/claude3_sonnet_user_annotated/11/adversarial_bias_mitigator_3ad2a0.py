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


@Model.register("adversarial_bias_mitigator")
class AdversarialBiasMitigator(Model):
    """
    Wrapper class to adversarially mitigate biases in any pretrained Model.

    # Parameters

    vocab : `Vocabulary`
        Vocabulary of predictor.
    predictor : `Model`
        Model for which to mitigate biases.
    adversary : `Model`
        Model that attempts to recover protected variable values from predictor's predictions.
    bias_direction : `BiasDirectionWrapper`
        Bias direction used by adversarial bias mitigator.
    predictor_output_key : `str`
        Key corresponding to output in `output_dict` of predictor that should be passed as input
        to adversary.

    !!! Note
        adversary must use same vocab as predictor, if it requires a vocab.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        predictor: Model,
        adversary: Model,
        bias_direction: BiasDirectionWrapper,
        predictor_output_key: str,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.predictor: Model = predictor
        self.adversary: Model = adversary

        # want to keep adversary label hook during evaluation
        embedding_layer = find_embedding_layer(self.predictor)
        self.bias_direction: BiasDirectionWrapper = bias_direction
        self.predetermined_bias_direction: torch.Tensor = self.bias_direction(embedding_layer)
        self._adversary_label_hook: _AdversaryLabelHook = _AdversaryLabelHook(self.predetermined_bias_direction)
        embedding_layer.register_forward_hook(self._adversary_label_hook)

        self.vocab: Vocabulary = self.predictor.vocab
        self._regularizer = self.predictor._regularizer

        self.predictor_output_key: str = predictor_output_key

    def train(self, mode: bool = True) -> 'AdversarialBiasMitigator':
        super().train(mode)
        self.predictor.train(mode)
        self.adversary.train(mode)
        # appropriately change requires_grad
        # in bias direction when train() and
        # eval() are called
        self.bias_direction.train(mode)
        return self

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        predictor_output_dict: Dict[str, torch.Tensor] = self.predictor.forward(*args, **kwargs)
        adversary_output_dict: Dict[str, torch.Tensor] = self.adversary.forward(
            predictor_output_dict[self.predictor_output_key],
            self._adversary_label_hook.adversary_label,
        )
        # prepend "adversary_" to every key in adversary_output_dict
        # to distinguish from predictor_output_dict keys
        adversary_output_dict = {("adversary_" + k): v for k, v in adversary_output_dict.items()}
        output_dict: Dict[str, torch.Tensor] = {**predictor_output_dict, **adversary_output_dict}
        return output_dict

    # Delegate Model function calls to predictor
    # Currently doing this manually because difficult to
    # dynamically forward __getattribute__ due to
    # behind-the-scenes usage of dunder attributes by torch.nn.Module
    # and predictor inheriting from Model
    # Assumes Model is relatively stable

    def forward_on_instance(self, *args, **kwargs):
        return self.predictor.forward_on_instance(*args, **kwargs)

    def forward_on_instances(self, *args, **kwargs):
        return self.predictor.forward_on_instances(*args, **kwargs)

    def get_regularization_penalty(self, *args, **kwargs):
        return self.predictor.get_regularization_penalty(*args, **kwargs)

    def get_parameters_for_histogram_logging(self, *args, **kwargs):
        return self.predictor.get_parameters_for_histogram_logging(*args, **kwargs)

    def get_parameters_for_histogram_tensorboard_logging(self, *args, **kwargs):
        return self.predictor.get_parameters_for_histogram_tensorboard_logging(*args, **kwargs)

    def make_output_human_readable(self, *args, **kwargs):
        return self.predictor.make_output_human_readable(*args, **kwargs)

    def get_metrics(self, *args, **kwargs):
        return self.predictor.get_metrics(*args, **kwargs)

    def _get_prediction_device(self, *args, **kwargs):
        return self.predictor._get_prediction_device(*args, **kwargs)

    def _maybe_warn_for_unseparable_batches(self, *args, **kwargs):
        return self.predictor._maybe_warn_for_unseparable_batches(*args, **kwargs)

    def extend_embedder_vocab(self, *args, **kwargs):
        return self.predictor.extend_embedder_vocab(*args, **kwargs)


@Model.register("feedforward_regression_adversary")
class FeedForwardRegressionAdversary(Model):
    """
    This `Model` implements a simple feedforward regression adversary.

    Registered as a `Model` with name "feedforward_regression_adversary".

    # Parameters

    vocab : `Vocabulary`
    feedforward : `FeedForward`
        A feedforward layer.
    initializer : `Optional[InitializerApplicator]`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        feedforward: FeedForward,
        initializer: Optional[InitializerApplicator] = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._feedforward: FeedForward = feedforward
        self._loss: torch.nn.MSELoss = torch.nn.MSELoss()
        initializer(self)  # type: ignore

    def forward(  # type: ignore
        self, input: torch.FloatTensor, label: torch.FloatTensor
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        input : `torch.FloatTensor`
            A tensor of size (batch_size, ...).
        label : `torch.FloatTensor`
            A tensor of the same size as input.

        # Returns

        An output dictionary consisting of:
            - `loss` : `torch.FloatTensor`
                A scalar loss to be optimised.
        """

        pred: torch.FloatTensor = self._feedforward(input)
        return {"loss": self._loss(pred, label)}


@TrainerCallback.register("adversarial_bias_mitigator_backward")
class AdversarialBiasMitigatorBackwardCallback(TrainerCallback):
    """
    Performs backpropagation for adversarial bias mitigation.
    While the adversary's gradients are computed normally,
    the predictor's gradients are computed such that updates to the
    predictor's parameters will not aid the adversary and will
    make it more difficult for the adversary to recover protected variables.

    !!! Note
        Intended to be used with `AdversarialBiasMitigator`.
        trainer.model is expected to have `predictor` and `adversary` data members.

    # Parameters

    adversary_loss_weight : `float`, optional (default = `1.0`)
        Quantifies how difficult predictor makes it for adversary to recover protected variables.
    """

    def __init__(self, serialization_dir: str, adversary_loss_weight: float = 1.0) -> None:
        super().__init__(serialization_dir)
        self.adversary_loss_weight: float = adversary_loss_weight

    def on_backward(
        self,
        trainer: GradientDescentTrainer,
        batch_outputs: Dict[str, torch.Tensor],
        backward_called: bool,
        **kwargs,
    ) -> bool:
        if backward_called:
            raise OnBackwardException()

        if not hasattr(trainer.model, "predictor") or not hasattr(trainer.model, "adversary"):
            raise ConfigurationError(
                "Model is expected to have `predictor` and `adversary` data members."
            )

        trainer.optimizer.zero_grad()
        # `retain_graph=True` prevents computation graph from being erased
        batch_outputs["adversary_loss"].backward(retain_graph=True)
        # trainer.model is expected to have `predictor` and `adversary` data members
        adversary_loss_grad: Dict[str, torch.Tensor] = {
            name: param.grad.clone()
            for name, param in trainer.model.predictor.named_parameters()
            if param.grad is not None
        }

        trainer.model.predictor.zero_grad()
        batch_outputs["loss"].backward()

        with torch.no_grad():
            for name, param in trainer.model.predictor.named_parameters():
                if param.grad is not None:
                    unit_adversary_loss_grad: torch.Tensor = adversary_loss_grad[name] / torch.linalg.norm(
                        adversary_loss_grad[name]
                    )
                    # prevent predictor from accidentally aiding adversary
                    # by removing projection of predictor loss grad onto adversary loss grad
                    param.grad -= (
                        (param.grad * unit_adversary_loss_grad) * unit_adversary_loss_grad
                    ).sum()
                    # make it difficult for adversary to recover protected variables
                    param.grad -= self.adversary_loss_weight * adversary_loss_grad[name]

        # remove adversary_loss from computation graph
        batch_outputs["adversary_loss"] = batch_outputs["adversary_loss"].detach()
        return True


class _AdversaryLabelHook:
    def __init__(self, predetermined_bias_direction: torch.Tensor) -> None:
        self.predetermined_bias_direction: torch.Tensor = predetermined_bias_direction
        self.adversary_label: torch.Tensor = None  # type: ignore

    def __call__(self, module: torch.nn.Module, module_in: torch.Tensor, module_out: torch.Tensor) -> None:
        """
        Called as forward hook.
        """
        with torch.no_grad():
            # mean pooling over static word embeddings to get sentence embedding
            module_out = module_out.mean(dim=1)
            self.adversary_label = torch.matmul(
                module_out, self.predetermined_bias_direction.to(module_out.device)
            ).unsqueeze(-1)
