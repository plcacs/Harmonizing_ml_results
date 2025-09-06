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
    def __init__(self, vocab: Vocabulary, predictor: Model, adversary: Model, bias_direction: BiasDirectionWrapper, predictor_output_key: str, **kwargs) -> None:
    
    def func_lyw3hhv0(self, mode: bool = True) -> None:
    
    def func_mf8dq6ik(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
    
    def func_rjpjwy37(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
    
    def func_s2o7gif1(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
    
    def func_g97clzrp(self, *args, **kwargs) -> torch.Tensor:
    
    def func_56frtwkv(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
    
    def func_7jeygut6(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
    
    def func_ej4j1tup(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
    
    def func_7bwykmq5(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
    
    def func_gfzlett3(self, *args, **kwargs) -> torch.device:
    
    def func_hk8okx1s(self, *args, **kwargs) -> None:
    
    def func_2jbc11qg(self, *args, **kwargs) -> None:

@Model.register('feedforward_regression_adversary')
class FeedForwardRegressionAdversary(Model):
    def __init__(self, vocab: Vocabulary, feedforward: FeedForward, initializer: Optional[InitializerApplicator] = InitializerApplicator(), **kwargs) -> None:
    
    def func_mf8dq6ik(self, input: torch.FloatTensor, label: torch.FloatTensor) -> Dict[str, torch.Tensor]:

@TrainerCallback.register('adversarial_bias_mitigator_backward')
class AdversarialBiasMitigatorBackwardCallback(TrainerCallback):
    def __init__(self, serialization_dir: str, adversary_loss_weight: float = 1.0) -> None:
    
    def func_xlnm1r5f(self, trainer, batch_outputs, backward_called, **kwargs) -> bool:
