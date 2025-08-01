import os
from typing import Tuple, Union, Optional, TYPE_CHECKING, List, Any, Dict, Sequence
from fairscale.nn import FullyShardedDataParallel as FS_FSDP
from fairscale.nn.wrap import enable_wrap, wrap
from fairscale.nn.misc import FlattenParamsWrapper
from fairscale.optim.grad_scaler import GradScaler
import torch
from torch.cuda import amp
from allennlp.nn.parallel.sharded_module_mixin import ShardedModuleMixin
from allennlp.nn.parallel.ddp_accelerator import (
    DdpAccelerator,
    DdpWrappedModel,
    StateDictType,
    LoadStateDictReturnType,
)
if TYPE_CHECKING:
    from allennlp.models import Model
    from torch.nn import Module

class _FSDP(FS_FSDP, ShardedModuleMixin):
    """
    Same as FairScale's [`FullyShardedDataParallel`]
    (https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html) but also implements
    the mixin methods from :class:`allennlp.nn.parallel.sharded_module_mixin.ShardedModuleMixin`.
    """

    def get_original_module(self) -> "torch.nn.Module":
        module: Any = self.module
        if isinstance(module, FlattenParamsWrapper):
            module = module.module
        return module

class FairScaleFsdpWrappedModel(DdpWrappedModel):
    """
    The wrapped model type returned from [`FairScaleFsdpWrappedModel.wrap_model`](#wrap_model).
    """

    @staticmethod
    def consolidate_sharded_state(sharded_state_files: Sequence[str]) -> Dict[str, Any]:
        shard_weights: List[Any] = []
        shard_metadata: List[Any] = []
        for path in sharded_state_files:
            shard_state: Dict[str, Any] = torch.load(path, map_location='cpu')
            shard_weights.append(shard_state['weights'])
            shard_metadata.append(shard_state['metadata'])
        return _FSDP.consolidate_shard_weights(shard_weights, shard_metadata)

    def load_state_dict(self, state_dict: StateDictType, strict: bool = True) -> LoadStateDictReturnType:
        return self.model.load_local_state_dict(state_dict['weights'], strict=strict)

    def state_dict(self, *args: Any, **kwargs: Any) -> StateDictType:
        weights: Any = self.model.local_state_dict(*args, **kwargs)
        metadata: Any = self.model.local_metadata_dict()
        return {'weights': weights, 'metadata': metadata}

    def clip_grad_norm_(self, max_norm: float) -> Any:
        return self.model.clip_grad_norm_(max_norm)

    def init_grad_scaler(self) -> GradScaler:
        return GradScaler()

@DdpAccelerator.register('fairscale_fsdp')
class FairScaleFsdpAccelerator(DdpAccelerator):
    """
    A :class:`allennlp.nn.parallel.ddp_accelerator.DdpAccelerator` for FairScale's [`FullyShardedDataParallel`]
    (https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html).

    To save memory while initializing a model, you should call [`.wrap_module()`](#wrap_module) on submodules
    as they're created.

    See the :class:`allennlp.modules.transformer.t5.T5` class for an example of how to use this.
    """
    def __init__(
        self,
        *,
        mixed_precision: bool = False,
        reshard_after_forward: bool = True,
        flatten_parameters: bool = True,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        cuda_device: Union[int, torch.device] = -1
    ) -> None:
        super().__init__(local_rank=local_rank, world_size=world_size, cuda_device=cuda_device)
        self._fsdp_kwargs: Dict[str, Any] = {
            'compute_device': self.cuda_device,
            'mixed_precision': mixed_precision,
            'reshard_after_forward': reshard_after_forward,
            'flatten_parameters': flatten_parameters
        }
        if mixed_precision:
            self._fsdp_kwargs['move_params_to_cpu'] = True
            self._fsdp_kwargs['clear_autocast_cache'] = True

    def wrap_model(self, model: "torch.nn.Module") -> Tuple["torch.nn.Module", FairScaleFsdpWrappedModel]:
        wrapped_model: _FSDP = _FSDP(model, **self._fsdp_kwargs)
        if not self._fsdp_kwargs['mixed_precision'] and self.cuda_device != torch.device('cpu'):
            wrapped_model = wrapped_model.cuda()  # type: ignore
        for module in wrapped_model.modules():
            if isinstance(module, _FSDP):
                module._reset_lazy_init()
        return (model, FairScaleFsdpWrappedModel(wrapped_model, local_rank=self.local_rank, world_size=self.world_size))

    def wrap_module(self, module: "torch.nn.Module") -> "torch.nn.Module":
        with enable_wrap(wrapper_cls=_FSDP, **self._fsdp_kwargs):
            wrapped_module: "torch.nn.Module" = wrap(module)
        return wrapped_module