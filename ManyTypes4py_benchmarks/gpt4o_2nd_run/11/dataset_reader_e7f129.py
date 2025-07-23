from dataclasses import dataclass
import itertools
from os import PathLike
from typing import Iterable, Iterator, Optional, Union, TypeVar, Dict, List
import logging
import warnings
import torch.distributed as dist
from allennlp.data.instance import Instance
from allennlp.common import util
from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)

@dataclass
class WorkerInfo:
    """
    Contains information about the worker context when a `DatasetReader`
    is being used within a multi-process `DataLoader`.

    From a `DatasetReader` this can accessed with the [`get_worker_info()`](#get_worker_info) method.
    """
    num_workers: int
    id: int

@dataclass
class DistributedInfo:
    """
    Contains information about the global process rank and total world size when the reader is being
    used within distributed training.

    From a `DatasetReader` this can be accessed with the [`get_distributed_info()`](#get_distributed_info) method.
    """
    world_size: int
    global_rank: int

_T = TypeVar('_T')
PathOrStr = Union[PathLike, str]
DatasetReaderInput = Union[PathOrStr, List[PathOrStr], Dict[str, PathOrStr]]

class DatasetReader(Registrable):
    def __init__(self, max_instances: Optional[int] = None, manual_distributed_sharding: bool = False, manual_multiprocess_sharding: bool = False, serialization_dir: Optional[str] = None) -> None:
        if max_instances is not None and max_instances < 0:
            raise ValueError('If specified, max_instances should be a positive int')
        self.max_instances = max_instances
        self.manual_distributed_sharding = manual_distributed_sharding
        self.manual_multiprocess_sharding = manual_multiprocess_sharding
        self.serialization_dir = serialization_dir
        self._worker_info: Optional[WorkerInfo] = None
        self._distributed_info: Optional[DistributedInfo] = None
        if util.is_distributed():
            self._distributed_info = DistributedInfo(dist.get_world_size(), dist.get_rank())

    def read(self, file_path: DatasetReaderInput) -> Iterator[Instance]:
        for instance in self._multi_worker_islice(self._read(file_path)):
            if self._worker_info is None:
                self.apply_token_indexers(instance)
            yield instance

    def _read(self, file_path: DatasetReaderInput) -> Iterable[Instance]:
        raise NotImplementedError

    def text_to_instance(self, *inputs) -> Instance:
        raise NotImplementedError

    def apply_token_indexers(self, instance: Instance) -> None:
        pass

    def get_worker_info(self) -> Optional[WorkerInfo]:
        return self._worker_info

    def get_distributed_info(self) -> Optional[DistributedInfo]:
        return self._distributed_info

    def _set_worker_info(self, info: WorkerInfo) -> None:
        self._worker_info = info

    def _set_distributed_info(self, info: DistributedInfo) -> None:
        self._distributed_info = info

    def shard_iterable(self, iterable: Iterable[_T]) -> Iterator[_T]:
        if not self.manual_distributed_sharding or not self.manual_multiprocess_sharding:
            raise ValueError('self.shard_iterable() was called but self.manual_distributed_sharding and self.manual_multiprocess_sharding was not set to True. Did you forget to call super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True) in your constructor?')
        sharded_slice = iter(iterable)
        if util.is_distributed():
            sharded_slice = itertools.islice(sharded_slice, dist.get_rank(), None, dist.get_world_size())
        if self._worker_info is not None:
            sharded_slice = itertools.islice(sharded_slice, self._worker_info.id, None, self._worker_info.num_workers)
        if self.max_instances is not None:
            sharded_slice = itertools.islice(sharded_slice, self.max_instances)
        return sharded_slice

    def _multi_worker_islice(self, iterable: Iterable[_T]) -> Iterator[_T]:
        sharded_slice = iter(iterable)
        max_instances = self.max_instances
        if self._distributed_info is not None:
            if max_instances is not None:
                if self._distributed_info.global_rank < max_instances % self._distributed_info.world_size:
                    max_instances = max_instances // self._distributed_info.world_size + 1
                else:
                    max_instances = max_instances // self._distributed_info.world_size
            if not self.manual_distributed_sharding:
                sharded_slice = itertools.islice(sharded_slice, self._distributed_info.global_rank, None, self._distributed_info.world_size)
        if self._worker_info is not None:
            if max_instances is not None:
                if self._worker_info.id < max_instances % self._worker_info.num_workers:
                    max_instances = max_instances // self._worker_info.num_workers + 1
                else:
                    max_instances = max_instances // self._worker_info.num_workers
            if not self.manual_multiprocess_sharding:
                warnings.warn("Using multi-process data loading without setting DatasetReader.manual_multiprocess_sharding to True.\nDid you forget to set this?\nIf you're not handling the multi-process sharding logic within your _read() method, there is probably no benefit to using more than one worker.", UserWarning)
                sharded_slice = itertools.islice(sharded_slice, self._worker_info.id, None, self._worker_info.num_workers)
        if max_instances is not None:
            sharded_slice = itertools.islice(sharded_slice, max_instances)
        return sharded_slice
