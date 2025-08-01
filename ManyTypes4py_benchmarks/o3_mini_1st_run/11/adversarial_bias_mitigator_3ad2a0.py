#!/usr/bin/env python3
"""
A Model wrapper to adversarially mitigate biases in
predictions produced by a pretrained model for a downstream task.

The documentation and explanations are heavily based on:
Zhang, B.H., Lemoine, B., & Mitchell, M. (2018).
[Mitigating Unwanted Biases with Adversarial Learning]
(https://api.semanticscholar.org/CorpusID:9424845).
Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society.
and [Mitigating Unwanted Biases in Word Embeddings
with Adversarial Learning](https://colab.research.google.com/notebooks/
ml_fairness/adversarial_debiasing.ipynb) colab notebook.

Adversarial networks mitigate some biases based on the idea that
predicting an outcome Y given an input X should ideally be independent
of some protected variable Z. Informally, "knowing Y would not help
you predict Z any better than chance" (Zaldivar et al., 2018). This
can be achieved using two networks in a series, where the first attempts to predict
Y using X as input, and the second attempts to use the predicted value of Y to recover Z.
Please refer to Figure 1 of [Mitigating Unwanted Biases with Adversarial Learning]
(https://api.semanticscholar.org/CorpusID:9424845). Ideally, we would
like the first network to predict Y without permitting the second network to predict
Z any better than chance.

For common NLP tasks, it's usually clear what X and Y are,
but Z is not always available. We can construct our own Z by:

1. computing a bias direction (e.g. for binary gender)

2. computing the inner product of static sentence embeddings and the bias direction

Training adversarial networks is extremely difficult. It is important to:

1. lower the step size of both the predictor and adversary to train both
models slowly to avoid parameters diverging,

2. initialize the parameters of the adversary to be small to avoid the predictor
overfitting against a sub-optimal adversary,

3. increase the adversary’s learning rate to prevent divergence if the
predictor is too good at hiding the protected variable from the adversary.
"""
from typing import Dict, Optional, Any, Tuple
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
    """
    Wrapper class to adversarially mitigate biases in any pretrained Model.

    # Parameters

    vocab : Vocabulary
        Vocabulary of predictor.
    predictor : Model
        Model for which to mitigate biases.
    adversary : Model
        Model that attempts to recover protected variable values from predictor's predictions.
    bias_direction : BiasDirectionWrapper
        Bias direction used by adversarial bias mitigator.
    predictor_output_key : str
        Key corresponding to output in output_dict of predictor that should be passed as input
        to adversary.

    !!! Note
        adversary must use same vocab as predictor, if it requires a vocab.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 predictor: Model,
                 adversary: Model,
                 bias_direction: BiasDirectionWrapper,
                 predictor_output_key: str,
                 **kwargs: Any) -> None:
        super().__init__(vocab, **kwargs)
        self.predictor: Model = predictor
        self.adversary: Model = adversary
        embedding_layer = find_embedding_layer(self.predictor)
        self.bias_direction: BiasDirectionWrapper = bias_direction
        self.predetermined_bias_direction: torch.Tensor = self.bias_direction(embedding_layer)
        self._adversary_label_hook: _AdversaryLabelHook = _AdversaryLabelHook(self.predetermined_bias_direction)
        embedding_layer.register_forward_hook(self._adversary_label_hook)
        self.vocab = self.predictor.vocab
        self._regularizer = self.predictor._regularizer
        self.predictor_output_key: str = predictor_output_key

    def train(self, mode: bool = True) -> 'AdversarialBiasMitigator':
        super().train(mode)
        self.predictor.train(mode)
        self.adversary.train(mode)
        self.bias_direction.train(mode)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        predictor_output_dict: Dict[str, Any] = self.predictor.forward(*args, **kwargs)
        adversary_output_dict: Dict[str, Any] = self.adversary.forward(
            predictor_output_dict[self.predictor_output_key],
            self._adversary_label_hook.adversary_label  # type: ignore
        )
        adversary_output_dict = {'adversary_' + k: v for k, v in adversary_output_dict.items()}
        output_dict: Dict[str, Any] = {**predictor_output_dict, **adversary_output_dict}
        return output_dict

    def forward_on_instance(self, *args: Any, **kwargs: Any) -> Any:
        return self.predictor.forward_on_instance(*args, **kwargs)

    def forward_on_instances(self, *args: Any, **kwargs: Any) -> Any:
        return self.predictor.forward_on_instances(*args, **kwargs)

    def get_regularization_penalty(self, *args: Any, **kwargs: Any) -> Any:
        return self.predictor.get_regularization_penalty(*args, **kwargs)

    def get_parameters_for_histogram_logging(self, *args: Any, **kwargs: Any) -> Any:
        return self.predictor.get_parameters_for_histogram_logging(*args, **kwargs)

    def get_parameters_for_histogram_tensorboard_logging(self, *args: Any, **kwargs: Any) -> Any:
        return self.predictor.get_parameters_for_histogram_tensorboard_logging(*args, **kwargs)

    def make_output_human_readable(self, *args: Any, **kwargs: Any) -> Any:
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
    """
    This Model implements a simple feedforward regression adversary.

    Registered as a Model with name "feedforward_regression_adversary".

    # Parameters

    vocab : Vocabulary
    feedforward : FeedForward
        A feedforward layer.
    initializer : Optional[InitializerApplicator], optional (default=InitializerApplicator())
        If provided, will be used to initialize the model parameters.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs: Any) -> None:
        super().__init__(vocab, **kwargs)
        self._feedforward: FeedForward = feedforward
        self._loss = torch.nn.MSELoss()
        initializer(self)

    def forward(self, input: torch.FloatTensor, label: torch.FloatTensor) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        input : torch.FloatTensor
            A tensor of size (batch_size, ...).
        label : torch.FloatTensor
            A tensor of the same size as input.

        # Returns

        An output dictionary consisting of:
            - `loss` : torch.FloatTensor
                A scalar loss to be optimised.
        """
        pred: torch.Tensor = self._feedforward(input)
        return {'loss': self._loss(pred, label)}

@TrainerCallback.register('adversarial_bias_mitigator_backward')
class AdversarialBiasMitigatorBackwardCallback(TrainerCallback):
    """
    Performs backpropagation for adversarial bias mitigation.
    While the adversary's gradients are computed normally,
    the predictor's gradients are computed such that updates to the
    predictor's parameters will not aid the adversary and will
    make it more difficult for the adversary to recover protected variables.

    !!! Note
        Intended to be used with AdversarialBiasMitigator.
        trainer.model is expected to have predictor and adversary data members.

    # Parameters

    adversary_loss_weight : float, optional (default = 1.0)
        Quantifies how difficult predictor makes it for adversary to recover protected variables.
    """

    def __init__(self, serialization_dir: str, adversary_loss_weight: float = 1.0) -> None:
        super().__init__(serialization_dir)
        self.adversary_loss_weight: float = adversary_loss_weight

    def on_backward(self,
                    trainer: GradientDescentTrainer,
                    batch_outputs: Dict[str, Any],
                    backward_called: bool,
                    **kwargs: Any) -> bool:
        if backward_called:
            raise OnBackwardException()
        if not (hasattr(trainer.model, 'predictor') and hasattr(trainer.model, 'adversary')):
            raise ConfigurationError('Model is expected to have `predictor` and `adversary` data members.')
        trainer.optimizer.zero_grad()
        batch_outputs['adversary_loss'].backward(retain_graph=True)
        adversary_loss_grad: Dict[str, torch.Tensor] = {
            name: param.grad.clone() for name, param in trainer.model.predictor.named_parameters() if param.grad is not None
        }
        trainer.model.predictor.zero_grad()
        batch_outputs['loss'].backward()
        with torch.no_grad():
            for name, param in trainer.model.predictor.named_parameters():
                if param.grad is not None:
                    unit_adversary_loss_grad: torch.Tensor = adversary_loss_grad[name] / torch.linalg.norm(adversary_loss_grad[name])
                    param.grad -= (param.grad * unit_adversary_loss_grad * unit_adversary_loss_grad).sum()
                    param.grad -= self.adversary_loss_weight * adversary_loss_grad[name]
        batch_outputs['adversary_loss'] = batch_outputs['adversary_loss'].detach()
        return True

class _AdversaryLabelHook:
    def __init__(self, predetermined_bias_direction: torch.Tensor) -> None:
        self.predetermined_bias_direction: torch.Tensor = predetermined_bias_direction
        # adversary_label will be created during forward hook execution.
        self.adversary_label: Optional[torch.Tensor] = None

    def __call__(self,
                 module: torch.nn.Module,
                 module_in: Tuple[Any, ...],
                 module_out: torch.Tensor) -> None:
        """
        Called as forward hook.
        """
        with torch.no_grad():
            module_out = module_out.mean(dim=1)
            self.adversary_label = torch.matmul(
                module_out,
                self.predetermined_bias_direction.to(module_out.device)
            ).unsqueeze(-1)