import logging
import math
from typing import List, Iterable, Tuple, Sequence, Optional, Iterator
import random
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.samplers.batch_sampler import BatchSampler

logger = logging.getLogger(__name__)

def add_noise_to_value(value: float, noise_param: float) -> float:
    noise_value = value * noise_param
    noise = random.uniform(-noise_value, noise_value)
    return value + noise

@BatchSampler.register('bucket')
class BucketBatchSampler(BatchSampler):
    def __init__(self, batch_size: int, sorting_keys: Optional[List[str]] = None, padding_noise: float = 0.1, drop_last: bool = False, shuffle: bool = True) -> None:
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        if not shuffle:
            self.padding_noise = 0.0

    def _argsort_by_padding(self, instances: Iterable[Instance]) -> Tuple[List[Instance], List[int]]:
        if not self.sorting_keys:
            logger.info('No sorting keys given; trying to guess a good one')
            self._guess_sorting_keys(instances)
            logger.info(f'Using {self.sorting_keys} as the sorting keys')
        instances_with_lengths = []
        for instance in instances:
            lengths = []
            noisy_lengths = []
            for field_name in self.sorting_keys:
                if field_name not in instance.fields:
                    raise ConfigurationError(f'Sorting key "{field_name}" is not a field in instance. Available fields/keys are {list(instance.fields.keys())}.')
                lengths.append(len(instance.fields[field_name]))
                noisy_lengths.append(add_noise_to_value(lengths[-1], self.padding_noise))
            instances_with_lengths.append((noisy_lengths, lengths, instance))
        with_indices = [(x, i) for i, x in enumerate(instances_with_lengths)]
        with_indices.sort(key=lambda x: x[0][0])
        return ([instance_with_index[-1] for instance_with_index in with_indices], [instance_with_index[0][1] for instance_with_index in with_indices])

    def get_batch_indices(self, instances: Iterable[Instance]) -> Iterator[List[Instance]]:
        indices, _ = self._argsort_by_padding(instances)
        batches = []
        for group in lazy_groups_of(indices, self.batch_size):
            batch_indices = list(group)
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            batches.append(batch_indices)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def _guess_sorting_keys(self, instances: Iterable[Instance], num_instances: int = 10) -> None:
        max_length = 0.0
        longest_field = None
        for i, instance in enumerate(instances):
            for field_name, field in instance.fields.items():
                length = len(field)
                if length > max_length:
                    max_length = length
                    longest_field = field_name
            if i > num_instances:
                break
        if not longest_field:
            raise AssertionError('Found no field that needed padding; we are surprised you got this error, please open an issue on github')
        self.sorting_keys = [longest_field]

    def get_num_batches(self, instances: Iterable[Instance]) -> int:
        batch_count_float = len(instances) / self.batch_size
        if self.drop_last:
            return math.floor(batch_count_float)
        else:
            return math.ceil(batch_count_float)

    def get_batch_size(self) -> int:
        return self.batch_size
