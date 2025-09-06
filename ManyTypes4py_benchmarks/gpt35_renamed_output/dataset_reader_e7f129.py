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

    def func_ki1rvvjj(self, file_path: str) -> Iterable[Instance]:
        ...

    def func_88q1eh0d(self, file_path: str) -> Iterable[Instance]:
        ...

    def func_31y3zem4(self, *inputs: Any) -> Instance:
        ...

    def func_wmhs5jyj(self, instance: Instance) -> None:
        ...

    def func_0wwdtp2q(self) -> Optional[WorkerInfo]:
        ...

    def func_gkj52vc1(self) -> Optional[DistributedInfo]:
        ...

    def func_sapctiq6(self, info: WorkerInfo) -> None:
        ...

    def func_mo98ndsp(self, info: DistributedInfo) -> None:
        ...

    def func_jz9hey95(self, iterable: Iterable[_T]) -> Iterable[_T]:
        ...

    def func_r0bslp0n(self, iterable: Iterable[_T]) -> Iterable[_T]:
        ...
