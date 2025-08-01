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
from typing import Any, Dict, List, Tuple, Union, Optional, Iterator, Set, Callable, TypeVar, cast
import torch
import transformers
from allennlp.common import Params, Registrable, Lazy
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)

ParameterGroupsType = List[Tuple[List[str], Dict[str, Any]]]
ModelParametersType = Iterator[Tuple[str, torch.nn.Parameter]]

def make_parameter_groups(
    model_parameters: ModelParametersType, 
    groups: Optional[ParameterGroupsType] = None
) -> Union[List[Dict[str, Any]], List[torch.nn.Parameter]]:
    """
    Takes a list of model parameters with associated names (typically coming from something like
    `model.named_parameters()`), along with a grouping (as specified below), and prepares them to be passed
    to the `__init__` function of a `torch.Optimizer`.  This means separating the parameters into
    groups with the given regexes, and prepping whatever keyword arguments are given for those
    regexes in `groups`.

    `groups` contains something like:

    