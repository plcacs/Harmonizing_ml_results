from typing import Any, Dict, List, Set, Union, Mapping
import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import Backbone
from allennlp.models.model import Model
from allennlp.models.heads import Head
from allennlp.nn import InitializerApplicator

def get_forward_arguments(module: torch.nn.Module) -> Set[str]:

@Model.register('multitask')
class MultiTaskModel(Model):
    def __init__(self, vocab: Vocabulary, backbone: Backbone, heads: Dict[str, Head], *, loss_weights: Dict[str, float] = None, arg_name_mapping: Dict[str, Dict[str, str]] = None, allowed_arguments: Dict[str, Set[str]] = None, initializer: InitializerApplicator = InitializerApplicator(), **kwargs: Any):

    def forward(self, **kwargs: Any) -> Dict[str, Any]:

    def _get_arguments(self, available_args: Dict[str, Any], component: str) -> Dict[str, Any]:

    def get_metrics(self, reset: bool = False) -> Dict[str, Any]:

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
