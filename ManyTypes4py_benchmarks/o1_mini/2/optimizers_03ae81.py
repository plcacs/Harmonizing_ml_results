import copy
import logging
import re
import math
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
import torch
import transformers
from allennlp.common import Params, Registrable, Lazy
from allennlp.common.checks import ConfigurationError

logger: logging.Logger = logging.getLogger(__name__)
ParameterGroupsType: List[Tuple[List[str], Dict[str, Any]]] = List[Tuple[List[str], Dict[str, Any]]]

def make_parameter_groups(
    model_parameters: List[Tuple[str, torch.nn.Parameter]],
    groups: Optional[ParameterGroupsType] = None
) -> Union[List[Dict[str, Any]], List[torch.nn.Parameter]]:
    """
    Takes a list of model parameters with associated names (typically coming from something like
    `model.named_parameters()`), along with a grouping (as specified below), and prepares them to be passed
    to the `__init__` function of a `torch.Optimizer`.  This means separating the parameters into
    groups with the given regexes, and prepping whatever keyword arguments are given for those
    regexes in `groups`.
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
                            raise ValueError(f'{name} was specified in two separate parameter groups')
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
                unused_options: Dict[str, Any] = {key: val for key, val in group.items() if key not in ('params', 'requires_grad')}
                if unused_options:
                    logger.warning('Ignoring unused options %s for %s', unused_options, names)
        parameter_group_names = [
            names for k, names in enumerate(parameter_group_names) if k not in no_grad_group_indices
        ]
        parameter_groups = [
            group for k, group in enumerate(parameter_groups) if k not in no_grad_group_indices
        ]
        logger.info('Done constructing parameter groups.')
        for k in range(len(parameter_groups)):
            group_options: Dict[str, Any] = {key: val for key, val in parameter_groups[k].items() if key != 'params'}
            logger.info('Group %s: %s, %s', k, list(parameter_group_names[k]), group_options)
        for regex, count in regex_use_counts.items():
            if count == 0:
                logger.warning('When constructing parameter groups, %s does not match any parameter name', regex)
    else:
        parameter_groups: List[torch.nn.Parameter] = [param for name, param in model_parameters]
    num_parameters: int = 0
    for parameter_group in parameter_groups:
        if isinstance(parameter_group, dict):
            num_parameters += sum(parameter.numel() for parameter in parameter_group['params'])
        else:
            num_parameters += parameter_group.numel()
    logger.info('Number of trainable parameters: %s', num_parameters)
    return parameter_groups

class Optimizer(torch.optim.Optimizer, Registrable):
    """
    This class just allows us to implement `Registrable` for Pytorch Optimizers.  We do something a
    little bit different with `Optimizers`, because they are implemented as classes in PyTorch, and
    we want to use those classes.  To make things easy, we just inherit from those classes, using
    multiple inheritance to also inherit from `Optimizer`.  The only reason we do this is to make
    type inference on parameters possible, so we can construct these objects using our configuration
    framework. If you are writing your own script, you can safely ignore these classes and just use
    the `torch.optim` classes directly.
    
    If you are implementing one of these classes, the `model_parameters` and `parameter_groups`
    arguments to `__init__` are important, and should always be present.  The trainer will pass
    the trainable parameters in the model to the optimizer using the name `model_parameters`, so if
    you use a different name, your code will crash.  Nothing will technically crash if you use a
    name other than `parameter_groups` for your second argument, it will just be annoyingly
    inconsistent.
    
    Most subclasses of `Optimizer` take both a `model_parameters` and a `parameter_groups`
    constructor argument. The `model_parameters` argument does not get an entry in a typical
    AllenNLP configuration file, but the `parameter_groups` argument does (if you want a non-default
    value).  See the documentation for the `make_parameter_groups` function for more information on
    how the `parameter_groups` argument should be specified.
    """
    default_implementation: str = 'adam'

    @staticmethod
    def default(model_parameters: List[Tuple[str, torch.nn.Parameter]]) -> 'Optimizer':
        return Optimizer.from_params(model_parameters=model_parameters, params=Params({}))

    def step(self, closure: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        raise NotImplementedError

class MultiOptimizer(Optimizer):
    """
    A `MultiOptimizer` creates a dictionary of `Optimizer`s keyed on some 'name'.
    Each Optimizer contains its own set of parameters which are obtained using
    regex matches for certain model parameters.

    This optimizer works by taking in a parameter `optimizers` which contains a list of `Optimizers`
    with their keyword arguments, and a parameter `parameter_groups`, which contains regexes and their
    corresponding optimizer and optional non-default optimizer options for this group.
    The regexes in the parameter groups are assigned to their optimizer based on the 'name' argument
    where the 'name' value should be the same for the optimizer and parameter group.
    You should specify a default optimizer with 'name': 'default' which will be used for all
    parameters which didn't obtain a regex match or when your parameter group doesn't contain a 'name'
    parameter.

    # Parameters

    optimizers: `Dict[str, Lazy[Optimizer]]`
        A dictionary of optimizers to use. Each key corresponds to the 'name' of the optimizer.
    
    parameter_groups:  `List[Tuple[List[str], Dict[str, Any]]]`, optional (default = `None`)
        See the docstring of `make_parameter_groups` for what this parameter should look like. It
        should follow the same format as there, except an additional 'optimizer_name' argument should be
        provided to match this group to its own optimizer. Optimizer options can also be set for this
        group which will override the default options.
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        optimizers: Dict[str, Lazy[Optimizer]],
        parameter_groups: ParameterGroupsType
    ) -> None:
        if 'default' not in optimizers:
            raise ConfigurationError(
                "No optimizer was provided for the 'default' group. Please provide an Optimizer under the name 'default'"
            )
        optimizer_name_to_parameter_groups: Dict[str, List[Tuple[List[str], Dict[str, Any]]]] = {
            optimizer_name: [] for optimizer_name in optimizers.keys()
        }
        for parameter_group in parameter_groups:
            regexes, pg_overrides = parameter_group
            optimizer_name: str = pg_overrides.get('optimizer_name', 'default')
            optimizer_name_to_parameter_groups[optimizer_name].append(parameter_group)
        optimizer_name_to_model_parameters: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {
            optimizer_name: [] for optimizer_name in optimizers.keys()
        }
        for model_parameter_tuple in model_parameters:
            parameter_name, parameter_tensor = model_parameter_tuple
            for regexes, pg_overrides in parameter_groups:
                if any(re.search(regex, parameter_name) for regex in regexes):
                    optimizer_name = pg_overrides.get('optimizer_name', 'default')
                    optimizer_name_to_model_parameters[optimizer_name].append(model_parameter_tuple)
                    break
            else:
                optimizer_name_to_model_parameters['default'].append(model_parameter_tuple)
        for optimizer_name, optimizer_parameters in optimizer_name_to_model_parameters.items():
            if optimizer_name != 'default' and len(optimizer_parameters) == 0:
                raise ConfigurationError(
                    f"Optimizer '{optimizer_name}' did not receive any parameters. If you are using `parameter_groups`, please make sure that the regexes you have provided match the desired model parameters, or that the `name` value of this optimizer  matches that of the parameter group you are trying to assign to it. Alternatively, you can remove this optimizer from the provided `optimizers` if it is not relevant to a particular parameter group."
                )
        if len(optimizer_name_to_model_parameters['default']) == 0:
            del optimizers['default']
            del optimizer_name_to_model_parameters['default']
            del optimizer_name_to_parameter_groups['default']
        self.optimizers: Dict[str, torch.optim.Optimizer] = {
            optimizer_name: lazy_optimizer.construct(
                model_parameters=optimizer_name_to_model_parameters[optimizer_name],
                parameter_groups=optimizer_name_to_parameter_groups[optimizer_name]
            )
            for optimizer_name, lazy_optimizer in optimizers.items()
        }
        parameter_groups = copy.deepcopy(parameter_groups)
        for parameter_group in parameter_groups:
            regexes, pg_overrides = parameter_group
            optimizer_name = pg_overrides.get('optimizer_name', 'default')
            optimizer = self.optimizers[optimizer_name]
            for key, value in optimizer.defaults.items():
                if key not in pg_overrides:
                    pg_overrides[key] = value
        made_parameter_groups: Union[List[Dict[str, Any]], List[torch.nn.Parameter]] = make_parameter_groups(
            model_parameters, parameter_groups
        )
        if 'default' in self.optimizers:
            for key, value in self.optimizers['default'].defaults.items():
                if isinstance(made_parameter_groups, list) and len(made_parameter_groups) > 0:
                    made_parameter_groups[-1][key] = value
        super().__init__(made_parameter_groups, {})  # type: ignore

    def step(self, closure: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        """
        Takes an optimization step for each optimizer.
        """
        for optimizer in self.optimizers.values():
            optimizer.step()
        return None

    def state_dict(self) -> Dict[str, Any]:
        """
        Creates an object `optimizer_state_dict`, which is a dictionary mapping an optimizer key to its
        `state_dict`. This dictionary is used as the value for 'optimizer' in the 'training_states' dictionary in
        the `gradient_descent` `Trainer`, e.g.
        