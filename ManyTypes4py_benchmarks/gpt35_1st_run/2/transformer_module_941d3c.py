import logging
import os
from os import PathLike
from typing import TYPE_CHECKING, Optional, Dict, Union, List, Any, TypeVar, Type
import re
import warnings
import torch
import torch.distributed as dist
from allennlp.common.util import is_distributed, is_global_primary
from allennlp.nn.parallel import ShardedModuleMixin
from allennlp.nn.module import Module
from allennlp.nn.util import StateDictType, read_state_dict, _check_incompatible_keys
if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
logger: logging.Logger = logging.getLogger(__name__)
_T = TypeVar('_T', bound='TransformerModule')

class TransformerModule(Module):
    _pretrained_mapping: Dict[str, str] = {}
    _pretrained_relevant_module: Optional[Union[str, List[str]]] = None
    _pretrained_ignore: Optional[List[str]] = None
    _pretrained_allow_missing: Optional[List[str]] = None

    @classmethod
    def _get_mapping(cls, mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        ...

    def _get_mapped_state_dict(self, state_dict: StateDictType, mapping: Optional[Dict[str, str]] = None) -> StateDictType:
        ...

    @classmethod
    def _get_relevant_submodule_state(cls, state_dict: StateDictType, relevant_module: Optional[Union[str, List[str]]] = None) -> StateDictType:
        ...

    @classmethod
    def _get_pretrained_state_dict(cls, model_name: str, weights_path: Optional[Union[str, PathLike]] = None, relevant_module: Optional[str] = None, ignore: Optional[List[str]] = None) -> StateDictType:
        ...

    @classmethod
    def _from_config(cls, config: 'PretrainedConfig', **kwargs: Any) -> '_T':
        ...

    @classmethod
    def from_pretrained_module(cls, model_name: str, *, load_weights: bool = True, weights_path: Optional[Union[str, PathLike]] = None, auto_config_kwargs: Optional[Dict[str, Any]] = None, mapping: Optional[Dict[str, str]] = None, relevant_module: Optional[str] = None, ignore: Optional[List[str]] = None, allow_missing: Optional[List[str]] = None, strict: bool = True, **kwargs: Any) -> '_T':
        ...

def _get_mapped_state_dict(module: TransformerModule, state_dict: StateDictType, mapping: Optional[Dict[str, str]] = None) -> StateDictType:
    ...
