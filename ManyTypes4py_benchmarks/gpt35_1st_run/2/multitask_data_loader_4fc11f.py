from typing import Any, Dict, Iterable, Iterator, Union, Optional
import itertools
import math
import torch
from allennlp.common import util
from allennlp.data.batch import Batch
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data.data_loaders.multitask_scheduler import MultiTaskScheduler
from allennlp.data.data_loaders.multitask_epoch_sampler import MultiTaskEpochSampler
from allennlp.data.dataset_readers.multitask import MultiTaskDatasetReader
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
import allennlp.nn.util as nn_util

def maybe_shuffle_instances(loader: DataLoader, shuffle: bool) -> Iterable[Instance]:
    if shuffle:
        return util.shuffle_iterable(loader.iter_instances())
    else:
        return loader.iter_instances()

@DataLoader.register('multitask')
class MultiTaskDataLoader(DataLoader):
    def __init__(self, reader: MultiTaskDatasetReader, data_path: Dict[str, str], scheduler: MultiTaskScheduler, *, sampler: Optional[MultiTaskEpochSampler] = None, instances_per_epoch: Optional[int] = None, num_workers: Optional[Dict[str, int]] = None, max_instances_in_memory: Optional[Dict[str, int]] = None, start_method: Optional[Dict[str, str]] = None, instance_queue_size: Optional[Dict[str, int]] = None, instance_chunk_size: Optional[Dict[str, int]] = None, shuffle: bool = True, cuda_device: Optional[Union[int, str, torch.device]] = None) -> None:
    
    def __len__(self) -> int:
    
    def __iter__(self) -> Iterator[TensorDict]:
    
    def iter_instances(self) -> Iterable[Instance]:
    
    def index_with(self, vocab: Vocabulary) -> None:
    
    def set_target_device(self, device: Union[int, str, torch.device]) -> None:
    
    def _get_instances_for_epoch(self) -> Dict[str, Iterable[Instance]]:
    
    def _make_data_loader(self, key: str) -> MultiProcessDataLoader:
