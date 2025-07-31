#!/usr/bin/env python3
"""
AllenNLP just uses
[PyTorch optimizers](https://pytorch.org/docs/master/optim.html),
with a thin wrapper to allow registering them and instantiating them `from_params`.

The available optimizers are

* [adadelta](https://pytorch.org/docs/master/optim.html#torch.optim.Adadelta)
* [adagrad](https://pytorch.org/docs/master/optim.html#torch.optim.Adagrad)
* [adam](https://pytorch.org/docs/master/optim.html#torch.optim.Adam)
* [adamw](https://pytorch.org/docs/master/optim.html#torch.optim.AdamW)
* [huggingface_adamw](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adamw-pytorch)
* [huggingface_adafactor](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adafactor-pytorch)
* [sparse_adam](https://pytorch.org/docs/master/optim.html#torch.optim.SparseAdam)
* [sgd](https://pytorch.org/docs/master/optim.html#torch.optim.SGD)
* [rmsprop](https://pytorch.org/docs/master/optim.html#torch.optim.RMSprop)
* [adamax](https://pytorch.org/docs/master/optim.html#torch.optim.Adamax)
* [averaged_sgd](https://pytorch.org/docs/master/optim.html#torch.optim.ASGD)
"""
import copy
import logging
import re
import math
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
import transformers
from allennlp.common import Params, Registrable, Lazy
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)
ParameterGroupsType = List[Tuple[List[str], Dict[str, Any]]]
ParameterGroupsReturnType = Union[List[torch.nn.Parameter], List[Dict[str, Any]]]

def make_parameter_groups(model_parameters: List[Tuple[str, torch.nn.Parameter]],
                          groups: Optional[ParameterGroupsType] = None
                         ) -> ParameterGroupsReturnType:
    """
    Takes a list of model parameters with associated names ... See docstring above.
    """
    if groups:
        parameter_groups: List[Dict[str, Any]] = [{'params': []} for _ in range(len(groups) + 1)]
        for k in range(len(groups)):
            parameter_groups[k].update(groups[k][1])
        regex_use_counts: Dict[str, int] = {}
        parameter_group_names: List[set] = [set() for _ in range(len(groups) + 1)]
        for name, param in model_parameters:
            group_index: Optional[int] = None
            for k, group_regexes in enumerate(groups):
                for regex in group_regexes[0]:
                    if regex not in regex_use_counts:
                        regex_use_counts[regex] = 0
                    if re.search(regex, name):
                        if group_index is not None and group_index != k:
                            raise ValueError('{} was specified in two separate parameter groups'.format(name))
                        group_index = k
                        regex_use_counts[regex] += 1
            if group_index is not None:
                parameter_groups[group_index]['params'].append(param)
                parameter_group_names[group_index].add(name)
            else:
                parameter_groups[-1]['params'].append(param)
                parameter_group_names[-1].add(name)
        no_grad_group_indices: List[int] = []
        for k, (names, group) in enumerate(zip(parameter_group_names, parameter_groups)):
            if group.get('requires_grad') is False:
                no_grad_group_indices.append(k)
                logger.info('Disabling gradient for the following parameters: %s', names)
                for param in group['params']:
                    param.requires_grad_(False)
                unused_options = {key: val for key, val in group.items() if key not in ('params', 'requires_grad')}
                if unused_options:
                    logger.warning('Ignoring unused options %s for %s', unused_options, names)
        parameter_group_names = [names for k, names in enumerate(parameter_group_names) if k not in no_grad_group_indices]
        parameter_groups = [group for k, group in enumerate(parameter_groups) if k not in no_grad_group_indices]
        logger.info('Done constructing parameter groups.')
        for k in range(len(parameter_groups)):
            group_options = {key: val for key, val in parameter_groups[k].items() if key != 'params'}
            logger.info('Group %s: %s, %s', k, list(parameter_group_names[k]), group_options)
        for regex, count in regex_use_counts.items():
            if count == 0:
                logger.warning('When constructing parameter groups, %s does not match any parameter name', regex)
    else:
        parameter_groups = [param for name, param in model_parameters]
    num_parameters = 0
    for parameter_group in parameter_groups:
        if isinstance(parameter_group, dict):
            num_parameters += sum((parameter.numel() for parameter in parameter_group['params']))
        else:
            num_parameters += parameter_group.numel()
    logger.info('Number of trainable parameters: %s', num_parameters)
    return parameter_groups

class Optimizer(torch.optim.Optimizer, Registrable):
    """
    This class just allows us to implement `Registrable` for PyTorch Optimizers.
    """
    default_implementation: str = 'adam'

    @staticmethod
    def default(model_parameters: List[Tuple[str, torch.nn.Parameter]]) -> 'Optimizer':
        return Optimizer.from_params(model_parameters=model_parameters, params=Params({}))

@Optimizer.register('multi')
class MultiOptimizer(Optimizer):
    """
    A `MultiOptimizer` creates a dictionary of `Optimizer`s keyed on some 'name'.
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 optimizers: Dict[str, Lazy],
                 parameter_groups: List[Tuple[List[str], Dict[str, Any]]]
                ) -> None:
        if 'default' not in optimizers:
            raise ConfigurationError("No optimizer was provided for the 'default' group. Please provide an Optimizer under the name 'default'")
        optimizer_name_to_parameter_groups: Dict[str, List[Tuple[List[str], Dict[str, Any]]]] = {optimizer_name: [] for optimizer_name in optimizers.keys()}
        for parameter_group in parameter_groups:
            regexes, pg_overrides = parameter_group
            optimizer_name = pg_overrides.get('optimizer_name', 'default')
            optimizer_name_to_parameter_groups[optimizer_name].append(parameter_group)
        optimizer_name_to_model_parameters: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {optimizer_name: [] for optimizer_name in optimizers.keys()}
        for model_parameter_tuple in model_parameters:
            parameter_name, parameter_tensor = model_parameter_tuple
            for regexes, pg_overrides in parameter_groups:
                if any((re.search(regex, parameter_name) for regex in regexes)):
                    optimizer_name = pg_overrides.get('optimizer_name', 'default')
                    optimizer_name_to_model_parameters[optimizer_name].append(model_parameter_tuple)
                    break
            else:
                optimizer_name_to_model_parameters['default'].append(model_parameter_tuple)
        for optimizer_name, optimizer_parameters in optimizer_name_to_model_parameters.items():
            if optimizer_name != 'default' and len(optimizer_parameters) == 0:
                raise ConfigurationError(f"Optimizer '{optimizer_name}' did not receive any parameters. If you are using `parameter_groups`, please make sure that the regexes you have provided match the desired model parameters, or that the `name` value of this optimizer matches that of the parameter group you are trying to assign to it. Alternatively, you can remove this optimizer from the provided `optimizers` if it is not relevant to a particular parameter group.")
        if len(optimizer_name_to_model_parameters['default']) == 0:
            del optimizers['default']
            del optimizer_name_to_model_parameters['default']
            del optimizer_name_to_parameter_groups['default']
        self.optimizers: Dict[str, Optimizer] = {optimizer_name:
            lazy_optimizer.construct(model_parameters=optimizer_name_to_model_parameters[optimizer_name],
                                     parameter_groups=optimizer_name_to_parameter_groups[optimizer_name])
                                     for optimizer_name, lazy_optimizer in optimizers.items()}
        parameter_groups = copy.deepcopy(parameter_groups)
        for parameter_group in parameter_groups:
            regexes, pg_overrides = parameter_group
            optimizer_name = pg_overrides.get('optimizer_name', 'default')
            optimizer = self.optimizers[optimizer_name]
            for key, value in optimizer.defaults.items():
                if key not in pg_overrides:
                    pg_overrides[key] = value
        made_parameter_groups = make_parameter_groups(model_parameters, parameter_groups)
        if 'default' in self.optimizers:
            for key, value in self.optimizers['default'].defaults.items():
                made_parameter_groups[-1][key] = value
        super().__init__(made_parameter_groups, {})

    def step(self) -> None:
        """
        Takes an optimization step for each optimizer.
        """
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        """
        Creates an optimizer state dictionary.
        """
        optimizer_state_dict: Dict[str, Any] = {f'{optimizer_key}_optimizer': optimizer.state_dict() for optimizer_key, optimizer in self.optimizers.items()}
        return optimizer_state_dict

    def load_state_dict(self, training_state: Dict[str, Any]) -> None:
        """
        Loads each optimizer's state_dict.
        """
        for optimizer_key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(training_state[f'{optimizer_key}_optimizer'])

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Sets parameter gradients to zero or None.
        """
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none)

@Optimizer.register('adam')
class AdamOptimizer(Optimizer, torch.optim.Adam):
    """
    Registered as an `Optimizer` with name "adam".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False
                ) -> None:
        super().__init__(params=make_parameter_groups(model_parameters, parameter_groups),
                         lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

@Optimizer.register('sparse_adam')
class SparseAdamOptimizer(Optimizer, torch.optim.SparseAdam):
    """
    Registered as an `Optimizer` with name "sparse_adam".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08
                ) -> None:
        super().__init__(params=make_parameter_groups(model_parameters, parameter_groups),
                         lr=lr, betas=betas, eps=eps)

@Optimizer.register('adamax')
class AdamaxOptimizer(Optimizer, torch.optim.Adamax):
    """
    Registered as an `Optimizer` with name "adamax".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 lr: float = 0.002,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08,
                 weight_decay: float = 0.0
                ) -> None:
        super().__init__(params=make_parameter_groups(model_parameters, parameter_groups),
                         lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

@Optimizer.register('adamw')
class AdamWOptimizer(Optimizer, torch.optim.AdamW):
    """
    Registered as an `Optimizer` with name "adamw".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08,
                 weight_decay: float = 0.01,
                 amsgrad: bool = False
                ) -> None:
        super().__init__(params=make_parameter_groups(model_parameters, parameter_groups),
                         lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

@Optimizer.register('huggingface_adamw')
class HuggingfaceAdamWOptimizer(Optimizer, transformers.AdamW):
    """
    Registered as an `Optimizer` with name "huggingface_adamw".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 lr: float = 1e-05,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08,
                 weight_decay: float = 0.0,
                 correct_bias: bool = True
                ) -> None:
        super().__init__(params=make_parameter_groups(model_parameters, parameter_groups),
                         lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)

@Optimizer.register('huggingface_adafactor')
class HuggingfaceAdafactor(Optimizer, transformers.Adafactor):
    """
    Registered as an `Optimizer` with name "huggingface_adafactor".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 lr: Optional[float] = None,
                 eps: Union[Tuple[float, float], float] = (1e-30, 0.001),
                 clip_threshold: float = 1.0,
                 decay_rate: float = -0.8,
                 beta1: Optional[float] = None,
                 weight_decay: float = 0.0,
                 scale_parameter: bool = True,
                 relative_step: bool = True,
                 warmup_init: bool = False
                ) -> None:
        super().__init__(params=make_parameter_groups(model_parameters, parameter_groups),
                         lr=lr, eps=eps, clip_threshold=clip_threshold, decay_rate=decay_rate,
                         beta1=beta1, weight_decay=weight_decay, scale_parameter=scale_parameter,
                         relative_step=relative_step, warmup_init=warmup_init)

@Optimizer.register('adagrad')
class AdagradOptimizer(Optimizer, torch.optim.Adagrad):
    """
    Registered as an `Optimizer` with name "adagrad".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 lr: float = 0.01,
                 lr_decay: float = 0.0,
                 weight_decay: float = 0.0,
                 initial_accumulator_value: float = 0.0,
                 eps: float = 1e-10
                ) -> None:
        super().__init__(params=make_parameter_groups(model_parameters, parameter_groups),
                         lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                         initial_accumulator_value=initial_accumulator_value, eps=eps)

@Optimizer.register('adadelta')
class AdadeltaOptimizer(Optimizer, torch.optim.Adadelta):
    """
    Registered as an `Optimizer` with name "adadelta".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 lr: float = 1.0,
                 rho: float = 0.9,
                 eps: float = 1e-06,
                 weight_decay: float = 0.0
                ) -> None:
        super().__init__(params=make_parameter_groups(model_parameters, parameter_groups),
                         lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)

@Optimizer.register('sgd')
class SgdOptimizer(Optimizer, torch.optim.SGD):
    """
    Registered as an `Optimizer` with name "sgd".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 lr: float,
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 momentum: float = 0.0,
                 dampening: float = 0,
                 weight_decay: float = 0.0,
                 nesterov: bool = False
                ) -> None:
        super().__init__(params=make_parameter_groups(model_parameters, parameter_groups),
                         lr=lr, momentum=momentum, dampening=dampening,
                         weight_decay=weight_decay, nesterov=nesterov)

@Optimizer.register('rmsprop')
class RmsPropOptimizer(Optimizer, torch.optim.RMSprop):
    """
    Registered as an `Optimizer` with name "rmsprop".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 lr: float = 0.01,
                 alpha: float = 0.99,
                 eps: float = 1e-08,
                 weight_decay: float = 0.0,
                 momentum: float = 0.0,
                 centered: bool = False
                ) -> None:
        super().__init__(params=make_parameter_groups(model_parameters, parameter_groups),
                         lr=lr, alpha=alpha, eps=eps,
                         weight_decay=weight_decay, momentum=momentum, centered=centered)

@Optimizer.register('averaged_sgd')
class AveragedSgdOptimizer(Optimizer, torch.optim.ASGD):
    """
    Registered as an `Optimizer` with name "averaged_sgd".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 lr: float = 0.01,
                 lambd: float = 0.0001,
                 alpha: float = 0.75,
                 t0: float = 1000000.0,
                 weight_decay: float = 0.0
                ) -> None:
        super().__init__(params=make_parameter_groups(model_parameters, parameter_groups),
                         lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)

@Optimizer.register('dense_sparse_adam')
class DenseSparseAdam(Optimizer, torch.optim.Optimizer):
    """
    NOTE: This class has been copied verbatim from the separate Dense and Sparse versions of Adam in PyTorch.
    Registered as an `Optimizer` with name "dense_sparse_adam".
    """
    def __init__(self,
                 model_parameters: List[Tuple[str, torch.nn.Parameter]],
                 parameter_groups: Optional[ParameterGroupsType] = None,
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-08
                ) -> None:
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults: Dict[str, Any] = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(make_parameter_groups(model_parameters, parameter_groups), defaults)

    def step(self, closure: Optional[Any] = None) -> Optional[Any]:
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values: torch.Tensor) -> torch.Tensor:
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)
                    old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                    exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
                    exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                    exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))
                    numer = exp_avg_update_values.add_(old_exp_avg_values)
                    exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                    denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
                    del exp_avg_update_values, exp_avg_sq_update_values
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                    p.data.add_(make_sparse(-step_size * numer.div_(denom)))
                else:
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
