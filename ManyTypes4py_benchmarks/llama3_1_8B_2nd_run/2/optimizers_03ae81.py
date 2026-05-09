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

def make_parameter_groups(model_parameters: List[Tuple[str, torch.nn.Parameter]], groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None) -> List[Dict[str, Any]]:
    """
    Takes a list of model parameters with associated names (typically coming from something like
    `model.named_parameters()`), along with a grouping (as specified below), and prepares them to be passed
    to the `__init__` function of a `torch.Optimizer`.  This means separating the parameters into
    groups with the given regexes, and prepping whatever keyword arguments are given for those
    regexes in `groups`.

    `groups` contains something like:

    