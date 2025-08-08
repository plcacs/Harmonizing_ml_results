from typing import Union, Tuple, OrderedDict, Dict, NamedTuple, List, Optional, Any, Sequence, TYPE_CHECKING
import torch
from allennlp.common import Registrable
from allennlp.nn.parallel.sharded_module_mixin import ShardedModuleMixin

StateDictType = Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]

class LoadStateDictReturnType(NamedTuple):
    pass

class DdpWrappedModel:
    def __init__(self, model: torch.nn.Module, local_rank: Optional[int] = None, world_size: Optional[int] = None):
        self.model = model
        self.local_rank = local_rank if local_rank is not None else dist.get_rank()
        self.world_size = world_size if world_size is not None else dist.get_world_size()
        self.is_primary = self.local_rank == 0

    @property
    def is_sharded(self) -> bool:
        return isinstance(self.model, ShardedModuleMixin)

    @staticmethod
    def consolidate_sharded_state(sharded_state_files: List[str]) -> None:
        raise NotImplementedError

    def load_state_dict(self, state_dict: StateDictType, strict: bool = True) -> None:
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs) -> StateDictType:
        return self.model.state_dict(*args, **kwargs)

    def clip_grad_norm_(self, max_norm: float) -> None:
        return clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], max_norm)

    def init_grad_scaler(self) -> amp.GradScaler:
        return amp.GradScaler()

class DdpAccelerator(Registrable):
    default_implementation: str = 'torch'

    def __init__(self, local_rank: Optional[int] = None, world_size: Optional[int] = None, cuda_device: int = -1):
        self.local_rank = local_rank if local_rank is not None else dist.get_rank()
        self.world_size = world_size if world_size is not None else dist.get_world_size()
        self.is_primary = local_rank == 0
        self.cuda_device = int_to_device(cuda_device)

    def wrap_model(self, model: torch.nn.Module) -> Tuple[torch.nn.Module, DdpWrappedModel]:
        raise NotImplementedError

    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return module

@DdpAccelerator.register('torch')
class TorchDdpAccelerator(DdpAccelerator):
    def __init__(self, find_unused_parameters: bool = False, local_rank: Optional[int] = None, world_size: Optional[int] = None, cuda_device: int = -1):
        super().__init__(local_rank=local_rank, world_size=world_size, cuda_device=cuda_device)
        self._ddp_kwargs = {'find_unused_parameters': find_unused_parameters}

    def wrap_model(self, model: torch.nn.Module) -> Tuple[torch.nn.Module, DdpWrappedModel]:
        if self.cuda_device != torch.device('cpu'):
            model = model.cuda(self.cuda_device)
        wrapped_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None if self.cuda_device == torch.device('cpu') else [self.cuda_device], **self._ddp_kwargs)
        wrapped_model._register_state_dict_hook(TorchDdpAccelerator._remove_torch_ddp_prefix)
        wrapped_model._register_load_state_dict_pre_hook(TorchDdpAccelerator._add_torch_ddp_prefix)
        return (model, DdpWrappedModel(wrapped_model, local_rank=self.local_rank, world_size=self.world_size))

    @staticmethod
    def _add_torch_ddp_prefix(state_dict: StateDictType, prefix: str, *args) -> None:
        for key in list(state_dict.keys()):
            if key.startswith(prefix + 'module.'):
                continue
            new_key = prefix + 'module.' + key
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    @staticmethod
    def _remove_torch_ddp_prefix(module: torch.nn.Module, state_dict: StateDictType, prefix: str, *args) -> None:
        for key in list(state_dict.keys()):
            if not key.startswith(prefix + 'module.'):
                continue
            new_key = key.replace(prefix + 'module.', '', 1)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
