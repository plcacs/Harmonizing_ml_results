from typing import Any, Dict, List, Tuple, Union, Optional
from allennlp.common import Params, Registrable, Lazy
from allennlp.common.checks import ConfigurationError
import torch
import transformers
import logging

ParameterGroupsType = List[Tuple[List[str], Dict[str, Any]]]

def make_parameter_groups(model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                          groups: Optional[ParameterGroupsType] = None) -> List[Dict[str, Any]]:
    ...

class Optimizer(torch.optim.Optimizer, Registrable):
    ...

    @staticmethod
    def default(model_parameters: List[Tuple[str, torch.nn.Parameter]]) -> 'Optimizer':
        ...

@Optimizer.register('multi')
class MultiOptimizer(Optimizer):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 optimizers: Dict[str, Dict[str, Any]], 
                 parameter_groups: Optional[ParameterGroupsType] = None):
        ...

    def step(self) -> None:
        ...

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        ...

    def load_state_dict(self, training_state: Dict[str, Dict[str, Any]]) -> None:
        ...

    def zero_grad(self, set_to_none: bool = False) -> None:
        ...

@Optimizer.register('adam')
class AdamOptimizer(Optimizer, torch.optim.Adam):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-08, 
                 weight_decay: float = 0.0, 
                 amsgrad: bool = False):
        ...

@Optimizer.register('sparse_adam')
class SparseAdamOptimizer(Optimizer, torch.optim.SparseAdam):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-08):
        ...

@Optimizer.register('adamax')
class AdamaxOptimizer(Optimizer, torch.optim.Adamax):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 lr: float = 0.002, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-08, 
                 weight_decay: float = 0.0):
        ...

@Optimizer.register('adamw')
class AdamWOptimizer(Optimizer, torch.optim.AdamW):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-08, 
                 weight_decay: float = 0.01, 
                 amsgrad: bool = False):
        ...

@Optimizer.register('huggingface_adamw')
class HuggingfaceAdamWOptimizer(Optimizer, transformers.AdamW):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 lr: float = 1e-05, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-08, 
                 weight_decay: float = 0.0, 
                 correct_bias: bool = True):
        ...

@Optimizer.register('huggingface_adafactor')
class HuggingfaceAdafactor(Optimizer, transformers.Adafactor):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 lr: Optional[float] = None, 
                 eps: Tuple[float, float] = (1e-30, 0.001), 
                 clip_threshold: float = 1.0, 
                 decay_rate: float = -0.8, 
                 beta1: Optional[float] = None, 
                 weight_decay: float = 0.0, 
                 scale_parameter: bool = True, 
                 relative_step: bool = True, 
                 warmup_init: bool = False):
        ...

@Optimizer.register('adagrad')
class AdagradOptimizer(Optimizer, torch.optim.Adagrad):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 lr: float = 0.01, 
                 lr_decay: float = 0.0, 
                 weight_decay: float = 0.0, 
                 initial_accumulator_value: float = 0.0, 
                 eps: float = 1e-10):
        ...

@Optimizer.register('adadelta')
class AdadeltaOptimizer(Optimizer, torch.optim.Adadelta):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 lr: float = 1.0, 
                 rho: float = 0.9, 
                 eps: float = 1e-06, 
                 weight_decay: float = 0.0):
        ...

@Optimizer.register('sgd')
class SgdOptimizer(Optimizer, torch.optim.SGD):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 lr: float, 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 momentum: float = 0.0, 
                 dampening: float = 0, 
                 weight_decay: float = 0.0, 
                 nesterov: bool = False):
        ...

@Optimizer.register('rmsprop')
class RmsPropOptimizer(Optimizer, torch.optim.RMSprop):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 lr: float = 0.01, 
                 alpha: float = 0.99, 
                 eps: float = 1e-08, 
                 weight_decay: float = 0.0, 
                 momentum: float = 0.0, 
                 centered: bool = False):
        ...

@Optimizer.register('averaged_sgd')
class AveragedSgdOptimizer(Optimizer, torch.optim.ASGD):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 lr: float = 0.01, 
                 lambd: float = 0.0001, 
                 alpha: float = 0.75, 
                 t0: float = 1000000.0, 
                 weight_decay: float = 0.0):
        ...

@Optimizer.register('dense_sparse_adam')
class DenseSparseAdam(Optimizer, torch.optim.Optimizer):
    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 parameter_groups: Optional[ParameterGroupsType] = None, 
                 lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-08):
        ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        ...
