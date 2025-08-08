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
    def __init__(
        self,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
        manual_multiprocess_sharding: bool = False,
        serialization_dir: Optional[str] = None
    ) -> None:
        if max_instances is not None and max_instances < 0:
            raise ValueError(
                'If specified, max_instances should be a positive int')
        self.max_instances = max_instances
        self.manual_distributed_sharding = manual_distributed_sharding
        self.manual_multiprocess_sharding = manual_multiprocess_sharding
        self.serialization_dir = serialization_dir
        self._worker_info: Optional[WorkerInfo] = None
        self._distributed_info: Optional[DistributedInfo] = None
        if util.is_distributed():
            self._distributed_info = DistributedInfo(dist.get_world_size(),
                dist.get_rank())

    def func_ki1rvvjj(self, file_path: PathOrStr) -> Iterator[Instance]:
        """
        Returns an iterator of instances that can be read from the file path.
        """
        for instance in self._multi_worker_islice(self._read(file_path)):
            if self._worker_info is None:
                self.apply_token_indexers(instance)
            yield instance

    def func_88q1eh0d(self, file_path: PathOrStr) -> Iterable[Instance]:
        """
        Reads the instances from the given `file_path` and returns them as an
        `Iterable`.

        You are strongly encouraged to use a generator so that users can
        read a dataset in a lazy way, if they so choose.
        """
        raise NotImplementedError

    def func_31y3zem4(self, *inputs: Any) -> Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        `Instance`.  The primary intended use for this is with a
        :class:`~allennlp.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`_read` and what happens at
        model serving time, or any other time you want to make a prediction from new data.  We need
        to process the data in the same way it was done at training time.  Allowing the
        `DatasetReader` to process new text lets us accomplish this, as we can just call
        `DatasetReader.text_to_instance` when serving predictions.

        The input type here is rather vaguely specified, unfortunately.  The `Predictor` will
        have to make some assumptions about the kind of `DatasetReader` that it's using, in order
        to pass it the right information.
        """
        raise NotImplementedError

    def func_wmhs5jyj(self, instance: Instance) -> None:
        """
        If `Instance`s created by this reader contain `TextField`s without `token_indexers`,
        this method can be overriden to set the `token_indexers` of those fields.

        E.g. if you have you have `"source"` `TextField`, you could implement this method like this:

        