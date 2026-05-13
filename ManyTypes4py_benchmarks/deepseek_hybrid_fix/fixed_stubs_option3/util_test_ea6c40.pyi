import json
import random
from typing import NamedTuple, Any, Union, Callable, Dict, List, Optional, Tuple, overload
import numpy
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import torch
import pytest
from flaky import flaky
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, run_distributed_test
from allennlp.common.util import sanitize
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, TokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.nn import util
from allennlp.nn.parallel import ShardedModuleMixin
from allennlp.models import load_archive

class TestNnUtil(AllenNlpTestCase): ...

def global_distributed_func(global_rank: int, world_size: int, gpu_id: int, function: Callable[..., Any], func_kwargs: Dict[str, Any], desired_values: torch.Tensor) -> None: ...

class DistributedFixtureModel(torch.nn.Module): ...
class DistributedFixtureSubmodule(torch.nn.Module): ...
class ShardedDistributedFixtureSubmodule(DistributedFixtureSubmodule, ShardedModuleMixin): ...

def _dist_load_ok(global_rank: int, world_size: int, gpu_id: int) -> None: ...
def _dist_load_with_errors(global_rank: int, world_size: int, gpu_id: int) -> None: ...
@pytest.mark.parametrize('test_func', [_dist_load_ok, _dist_load_with_errors])
def test_load_state_dict_distributed(test_func: Callable[[int, int, int], None]) -> None: ...