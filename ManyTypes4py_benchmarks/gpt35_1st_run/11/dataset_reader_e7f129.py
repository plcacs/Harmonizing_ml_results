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
logger: logging.Logger = logging.getLogger(__name__)

@dataclass
class WorkerInfo:
    num_workers: int
    id: int

@dataclass
class DistributedInfo:
    world_size: int
    global_rank: int

_T = TypeVar('_T')
PathOrStr = Union[PathLike, str]
DatasetReaderInput = Union[PathOrStr, List[PathOrStr], Dict[str, PathOrStr]]

class DatasetReader(Registrable):
    max_instances: Optional[int]
    manual_distributed_sharding: bool
    manual_multiprocess_sharding: bool
    serialization_dir: Optional[str]
    _worker_info: Optional[WorkerInfo]
    _distributed_info: Optional[DistributedInfo]

    def __init__(self, max_instances: Optional[int] = None, manual_distributed_sharding: bool = False, manual_multiprocess_sharding: bool = False, serialization_dir: Optional[str] = None) -> None:
        ...

    def read(self, file_path: str) -> Iterable[Instance]:
        ...

    def _read(self, file_path: str) -> Iterable[Instance]:
        ...

    def text_to_instance(self, *inputs) -> Instance:
        ...

    def apply_token_indexers(self, instance: Instance) -> None:
        ...

    def get_worker_info(self) -> Optional[WorkerInfo]:
        ...

    def get_distributed_info(self) -> Optional[DistributedInfo]:
        ...

    def _set_worker_info(self, info: WorkerInfo) -> None:
        ...

    def _set_distributed_info(self, info: DistributedInfo) -> None:
        ...

    def shard_iterable(self, iterable: Iterable[_T]) -> Iterator[_T]:
        ...

    def _multi_worker_islice(self, iterable: Iterable[_T]) -> Iterator[_T]:
        ...
