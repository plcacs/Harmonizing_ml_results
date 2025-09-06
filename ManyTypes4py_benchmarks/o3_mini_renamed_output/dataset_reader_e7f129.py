from dataclasses import dataclass
import itertools
from os import PathLike
from typing import Iterable, Iterator, Optional, Union, TypeVar, Dict, List, Any
import logging
import warnings
import torch.distributed as dist
from allennlp.data.instance import Instance
from allennlp.common import util
from allennlp.common.registrable import Registrable

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
PathOrStr = Union[PathLike, str]
DatasetReaderInput = Union[PathOrStr, List[PathOrStr], Dict[str, PathOrStr]]


@dataclass
class WorkerInfo:
    num_workers: int
    id: int


@dataclass
class DistributedInfo:
    world_size: int
    global_rank: int


class DatasetReader(Registrable):
    def __init__(
        self,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
        manual_multiprocess_sharding: bool = False,
        serialization_dir: Optional[str] = None,
    ) -> None:
        if max_instances is not None and max_instances < 0:
            raise ValueError("If specified, max_instances should be a positive int")
        self.max_instances: Optional[int] = max_instances
        self.manual_distributed_sharding: bool = manual_distributed_sharding
        self.manual_multiprocess_sharding: bool = manual_multiprocess_sharding
        self.serialization_dir: Optional[str] = serialization_dir
        self._worker_info: Optional[WorkerInfo] = None
        self._distributed_info: Optional[DistributedInfo] = None
        if util.is_distributed():
            self._distributed_info = DistributedInfo(dist.get_world_size(), dist.get_rank())

    def _read(self, file_path: DatasetReaderInput) -> Iterable[Instance]:
        # This method should be overridden by subclasses. Here we delegate to func_88q1eh0d.
        return self.func_88q1eh0d(file_path)

    def _multi_worker_islice(self, iterable: Iterable[_T]) -> Iterator[_T]:
        return self.func_r0bslp0n(iterable)

    def func_ki1rvvjj(self, file_path: DatasetReaderInput) -> Iterator[Instance]:
        """
        Returns an iterator of instances that can be read from the file path.
        """
        for instance in self._multi_worker_islice(self._read(file_path)):
            if self._worker_info is None:
                self.apply_token_indexers(instance)
            yield instance

    def func_88q1eh0d(self, file_path: DatasetReaderInput) -> Iterable[Instance]:
        """
        Reads the instances from the given `file_path` and returns them as an
        `Iterable`.
        """
        raise NotImplementedError

    def func_31y3zem4(self, *inputs: Any) -> Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        `Instance`.
        """
        raise NotImplementedError

    def func_wmhs5jyj(self, instance: Instance) -> None:
        """
        If `Instance`s created by this reader contain `TextField`s without `token_indexers`,
        this method can be overriden to set the `token_indexers` of those fields.
        """
        pass

    def func_0wwdtp2q(self) -> Optional[WorkerInfo]:
        """
        Provides a [`WorkerInfo`](#WorkerInfo) object when the reader is being used within a
        worker of a multi-process `DataLoader`.
        """
        return self._worker_info

    def func_gkj52vc1(self) -> Optional[DistributedInfo]:
        """
        Provides a [`DistributedInfo`](#DistributedInfo) object when the reader is being
        used within distributed training.
        """
        return self._distributed_info

    def func_sapctiq6(self, info: WorkerInfo) -> None:
        """
        Should only be used internally.
        """
        self._worker_info = info

    def func_mo98ndsp(self, info: DistributedInfo) -> None:
        """
        Should only be used internally.
        """
        self._distributed_info = info

    def func_jz9hey95(self, iterable: Iterable[_T]) -> Iterator[_T]:
        """
        Helper method that determines which items in an iterable object to skip based
        on the current node rank (for distributed training) and worker ID (for multi-process data loading).
        """
        if not self.manual_distributed_sharding or not self.manual_multiprocess_sharding:
            raise ValueError(
                "self.shard_iterable() was called but self.manual_distributed_sharding and self.manual_multiprocess_sharding was not set to True. Did you forget to call super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True) in your constructor?"
            )
        sharded_slice: Iterator[_T] = iter(iterable)
        if util.is_distributed():
            sharded_slice = itertools.islice(sharded_slice, dist.get_rank(), None, dist.get_world_size())
        if self._worker_info is not None:
            sharded_slice = itertools.islice(sharded_slice, self._worker_info.id, None, self._worker_info.num_workers)
        if self.max_instances is not None:
            sharded_slice = itertools.islice(sharded_slice, self.max_instances)
        return sharded_slice

    def func_r0bslp0n(self, iterable: Iterable[_T]) -> Iterator[_T]:
        """
        This is just like `shard_iterable` but is for internal use only.
        """
        sharded_slice: Iterator[_T] = iter(iterable)
        max_instances: Optional[int] = self.max_instances
        if self._distributed_info is not None:
            if max_instances is not None:
                if self._distributed_info.global_rank < max_instances % self._distributed_info.world_size:
                    max_instances = (max_instances // self._distributed_info.world_size + 1)
                else:
                    max_instances = (max_instances // self._distributed_info.world_size)
            if not self.manual_distributed_sharding:
                sharded_slice = itertools.islice(
                    sharded_slice, self._distributed_info.global_rank, None, self._distributed_info.world_size
                )
        if self._worker_info is not None:
            if max_instances is not None:
                if self._worker_info.id < max_instances % self._worker_info.num_workers:
                    max_instances = (max_instances // self._worker_info.num_workers + 1)
                else:
                    max_instances = (max_instances // self._worker_info.num_workers)
            if not self.manual_multiprocess_sharding:
                warnings.warn(
                    (
                        "Using multi-process data loading without setting DatasetReader.manual_multiprocess_sharding to True.\n"
                        "Did you forget to set this?\n"
                        "If you're not handling the multi-process sharding logic within your _read() method, there is probably no benefit to using more than one worker."
                    ),
                    UserWarning,
                )
                sharded_slice = itertools.islice(
                    sharded_slice, self._worker_info.id, None, self._worker_info.num_workers
                )
        if max_instances is not None:
            sharded_slice = itertools.islice(sharded_slice, max_instances)
        return sharded_slice

    def apply_token_indexers(self, instance: Instance) -> None:
        """
        A hook for applying token indexers to an instance.
        """
        self.func_wmhs5jyj(instance)