#!/usr/bin/env python3
import os
from typing import Union, Tuple, OrderedDict, Dict, NamedTuple, List, Optional, Any, Sequence, TYPE_CHECKING
import torch
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
from allennlp.common import Registrable
from allennlp.common.util import int_to_device
from allennlp.nn.parallel.sharded_module_mixin import ShardedModuleMixin

if TYPE_CHECKING:
    from allennlp.models import Model

StateDictType = Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]


class LoadStateDictReturnType(NamedTuple):
    # Add fields as necessary.
    pass


class DdpWrappedModel:
    """
    The type of the wrapped model returned from [`DdpAccelerator.wrap_model`](#wrap_model).
    """

    def __init__(self, model: Any, local_rank: Optional[int] = None, world_size: Optional[int] = None) -> None:
        self.model: Any = model
        self.local_rank: int = local_rank if local_rank is not None else dist.get_rank()
        self.world_size: int = world_size if world_size is not None else dist.get_world_size()
        self.is_primary: bool = self.local_rank == 0

    @property
    def is_sharded(self) -> bool:
        return isinstance(self.model, ShardedModuleMixin)

    @staticmethod
    def consolidate_sharded_state(sharded_state_files: Any) -> Any:
        raise NotImplementedError

    def load_state_dict(self, state_dict: StateDictType, strict: bool = True) -> Any:
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args: Any, **kwargs: Any) -> StateDictType:
        return self.model.state_dict(*args, **kwargs)

    def clip_grad_norm_(self, max_norm: float) -> float:
        return clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], max_norm)

    def init_grad_scaler(self) -> amp.GradScaler:
        return amp.GradScaler()


class DdpAccelerator(Registrable):
    """
    A `DdpAccelerator` is a generalization of PyTorch's `DistributedDataParallel` class.

    This is primarly used within the :class:`allennlp.training.trainer.GradientDescentTrainer` to allow
    for different DDP implementations, such as FairScale's [`FullyShardedDataParallel`]
    (https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html).

    In a typical AllenNLP configuration file, `local_rank`, `world_size`, and `cuda_device`
    should not be specified.

    !!! Warning
        This API is experimental and may change in the future.
    """
    default_implementation: str = 'torch'

    def __init__(self, local_rank: Optional[int] = None, world_size: Optional[int] = None, cuda_device: int = -1) -> None:
        self.local_rank: int = local_rank if local_rank is not None else dist.get_rank()
        self.world_size: int = world_size if world_size is not None else dist.get_world_size()
        self.is_primary: bool = self.local_rank == 0
        self.cuda_device: torch.device = int_to_device(cuda_device)

    def wrap_model(self, model: torch.nn.Module) -> Tuple[torch.nn.Module, DdpWrappedModel]:
        """
        Wrap the AllenNLP `Model`, returning the original model (possibly on a different device)
        and the [wrapper model](#ddpwrappedmodel).
        """
        raise NotImplementedError

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap an individual module. By default this just returns the module,
        but some subclass implementations such as
        :class:`allennlp.nn.parallel.fairscale_fsdp_accelerator.FairScaleFsdpAccelerator` do more.
        """
        return module


@DdpAccelerator.register('torch')
class TorchDdpAccelerator(DdpAccelerator):
    """
    The default implementation of `DdpAccelerator`, which is just a thin wrapper
    around PyTorch's `DistributedDataParallel`.
    """

    def __init__(self, *, find_unused_parameters: bool = False, local_rank: Optional[int] = None, world_size: Optional[int] = None, cuda_device: int = -1) -> None:
        super().__init__(local_rank=local_rank, world_size=world_size, cuda_device=cuda_device)
        self._ddp_kwargs: Dict[str, Any] = {'find_unused_parameters': find_unused_parameters}

    def wrap_model(self, model: torch.nn.Module) -> Tuple[torch.nn.Module, DdpWrappedModel]:
        if self.cuda_device != torch.device('cpu'):
            model = model.cuda(self.cuda_device)  # type: ignore[union-attr]
        wrapped_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=None if self.cuda_device == torch.device('cpu') else [self.cuda_device],
            **self._ddp_kwargs
        )
        wrapped_model._register_state_dict_hook(TorchDdpAccelerator._remove_torch_ddp_prefix)
        wrapped_model._register_load_state_dict_pre_hook(TorchDdpAccelerator._add_torch_ddp_prefix)
        return model, DdpWrappedModel(wrapped_model, local_rank=self.local_rank, world_size=self.world_size)

    @staticmethod
    def _add_torch_ddp_prefix(state_dict: Dict[str, torch.Tensor], prefix: str, *args: Any) -> None:
        for key in list(state_dict.keys()):
            if key.startswith(prefix + 'module.'):
                continue
            new_key: str = prefix + 'module.' + key
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    @staticmethod
    def _remove_torch_ddp_prefix(module: Any, state_dict: Dict[str, torch.Tensor], prefix: str, *args: Any) -> None:
        for key in list(state_dict.keys()):
            if not key.startswith(prefix + 'module.'):
                continue
            new_key: str = key.replace(prefix + 'module.', '', 1)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]