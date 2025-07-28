import logging
import os
from os import PathLike
from typing import TYPE_CHECKING, Optional, Dict, Union, List, Any, TypeVar, Type, Iterator, Tuple
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

_T = TypeVar('_T', bound='TransformerModule')


class TransformerModule(Module):
    """
    Base class to help with generalized loading of pretrained weights.

    Subclasses should override `_from_config()` if you want to instantiate them with
    `from_pretrained_module()`.
    """
    _pretrained_mapping: Dict[str, str] = {}
    "\n    An optional mapping for each class that determines any differences in the module\n    names between the class modules and the HuggingFace model's modules.\n    Keys correspond to HuggingFace submodule names, values correspond to submodules names of this module.\n    "
    _pretrained_relevant_module: Optional[Union[str, List[str]]] = None
    '\n    An optional string or list of strings which contains the expected name of the module\n    in the HuggingFace pretrained model. It can be a list to account for different names in different\n    models. The search is carried out in the order of the list.\n    '
    _pretrained_ignore: Optional[List[str]] = None
    '\n    An optional list of regular expressions that define which weights to ignore from a pretrained state_dict.\n    '
    _pretrained_allow_missing: Optional[List[str]] = None
    '\n    An optional list of regular expressions that specifies which weights are allowed to be missing\n    from the pretrained state dictionary.\n    '

    @classmethod
    def _get_mapping(cls, mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Returns the mapping to be used, based on the optional `mapping` overrides
        and the default module-level mapping.
        """
        combined_mapping: Dict[str, str] = {}
        combined_mapping.update(cls._pretrained_mapping)
        if mapping is not None:
            combined_mapping.update(mapping)
        return combined_mapping

    def _get_mapped_state_dict(self, state_dict: Dict[str, Any], mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Recursively map keys in a HuggingFace `state_dict` to the corresponding keys
        for this module and all submodules.
        """
        return _get_mapped_state_dict(self, state_dict, mapping=mapping)

    @classmethod
    def _get_relevant_submodule_state(cls, state_dict: Dict[str, Any], relevant_module: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Returns the relevant part of the `state_dict`.
        """
        relevant_modules: Optional[List[str]] = None
        if relevant_module:
            relevant_modules = [relevant_module] if isinstance(relevant_module, str) else relevant_module
        elif isinstance(cls._pretrained_relevant_module, str):
            relevant_modules = [cls._pretrained_relevant_module]
        elif isinstance(cls._pretrained_relevant_module, list):
            relevant_modules = cls._pretrained_relevant_module
        if relevant_modules:
            found: bool = False
            for module_name in relevant_modules:
                relevant_keys = {key for key in state_dict.keys() if key.startswith(module_name + '.')}
                if relevant_keys:
                    state_dict = {key.replace(module_name + '.', '', 1): value for key, value in state_dict.items() if key in relevant_keys}
                    found = True
                    break
            if not found:
                warnings.warn(f'{relevant_modules} was not found at top level of state_dict!', UserWarning)
        return state_dict

    @classmethod
    def _get_pretrained_state_dict(
        cls,
        model_name: str,
        weights_path: Optional[Union[str, PathLike]] = None,
        relevant_module: Optional[Union[str, List[str]]] = None,
        ignore: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get a HuggingFace pretrained `state_dict` corresponding to this module.
        """
        if weights_path is None:
            from transformers.file_utils import WEIGHTS_NAME
            if os.path.isdir(model_name):
                local_weights_path: str = os.path.join(model_name, WEIGHTS_NAME)
                if os.path.isfile(local_weights_path):
                    logger.info('Found weights at local path %s', local_weights_path)
                    weights_path = local_weights_path
            if weights_path is None:
                from allennlp.common.file_utils import cached_path
                weights_path = cached_path(f'hf://{model_name}/{WEIGHTS_NAME}')
        logger.info('Reading state dict from %s', weights_path)
        state_dict: Dict[str, Any] = read_state_dict(
            weights_path, ignore=ignore if ignore is not None else cls._pretrained_ignore, strict=False
        )
        state_dict = cls._get_relevant_submodule_state(state_dict, relevant_module=relevant_module)
        return state_dict

    @classmethod
    def _from_config(cls: Type[_T], config: "PretrainedConfig", **kwargs: Any) -> _T:
        """
        Instantiate this module from a HuggingFace config. Subclasses should override
        this method if you want to be able to instantiate them with `from_pretrained_module()`.
        """
        raise NotImplementedError

    @classmethod
    def from_pretrained_module(
        cls: Type[_T],
        model_name: str,
        *,
        load_weights: bool = True,
        weights_path: Optional[Union[str, PathLike]] = None,
        auto_config_kwargs: Optional[Dict[str, Any]] = None,
        mapping: Optional[Dict[str, str]] = None,
        relevant_module: Optional[Union[str, List[str]]] = None,
        ignore: Optional[List[str]] = None,
        allow_missing: Optional[List[str]] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> _T:
        """
        Initialize this module from a corresponding model on HuggingFace.
        """
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, **(auto_config_kwargs or {}))
        model: _T = cls._from_config(config, **kwargs)
        if load_weights:
            state_dict: Optional[Dict[str, Any]] = None
            if is_global_primary():
                pretrained_state_dict = cls._get_pretrained_state_dict(
                    model_name, weights_path=weights_path, relevant_module=relevant_module, ignore=ignore
                )
                state_dict = model._get_mapped_state_dict(pretrained_state_dict, mapping=mapping)
            logger.info('Loading state_dict into module')
            if not is_distributed():
                assert state_dict is not None
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            else:
                dist.barrier()
                missing_keys, unexpected_keys = model.load_state_dict_distributed(state_dict, strict=False)
            if allow_missing is None:
                allow_missing = cls._pretrained_allow_missing
            if allow_missing:
                missing_keys = [k for k in missing_keys if not any((re.match(p, k) for p in allow_missing))]
            _check_incompatible_keys(model, missing_keys, unexpected_keys, strict)
        return model


def _get_mapped_state_dict(
    module: Module, state_dict: Dict[str, Any], mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    if isinstance(module, TransformerModule):
        combined_mapping: Dict[str, str] = module._get_mapping(mapping)
    else:
        combined_mapping = {}
    for hf_key, cls_key in sorted(combined_mapping.items(), key=lambda x: x[0].count('.'), reverse=True):
        relevant_keys = {key for key in state_dict.keys() if key == hf_key or key.startswith(hf_key + '.')}
        for key in relevant_keys:
            new_key = key.replace(hf_key, cls_key, 1)
            if new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(key)
    for name, submodule in module.named_children():
        if isinstance(submodule, ShardedModuleMixin):
            submodule = submodule.get_original_module()
        relevant_keys = {key for key in state_dict.keys() if key.startswith(name + '.')}
        module_state_dict: Dict[str, Any] = {key.replace(name + '.', '', 1): state_dict.pop(key) for key in relevant_keys}
        module_state_dict = _get_mapped_state_dict(submodule, module_state_dict)
        for key, value in module_state_dict.items():
            state_dict[name + '.' + key] = value
    return state_dict

logger = logging.getLogger(__name__)