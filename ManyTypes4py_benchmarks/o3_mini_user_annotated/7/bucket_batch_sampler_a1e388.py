import logging
import math
from typing import List, Iterable, Tuple, Sequence, Optional, Any, Dict
import random

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.samplers.batch_sampler import BatchSampler

logger = logging.getLogger(__name__)


def add_noise_to_value(value: int, noise_param: float) -> float:
    noise_value: float = value * noise_param
    noise: float = random.uniform(-noise_value, noise_value)
    return value + noise


@BatchSampler.register("bucket")
class BucketBatchSampler(BatchSampler):
    """
    A sampler which by default, argsorts batches with respect to the maximum input lengths `per
    batch`. You can provide a list of field names (or pass none, in which case they
    will be inferred) which the dataset will be sorted by before doing this batching, causing inputs
    with similar length to be batched together, making computation more efficient (as less time is
    wasted on padded elements of the batch).

    Parameters
    ----------
    batch_size : int
        The size of each batch of instances yielded when calling the data_loader.
    sorting_keys : Optional[List[str]]
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.
        If this is not given we try to auto-detect the right keys by iterating through a few instances
        upfront, reading all of the padding keys and seeing which one has the longest length.
    padding_noise : float, optional (default=0.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic. This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    drop_last : bool, optional (default=False)
        If True, the sampler will drop the last batch if its size would be less than batch_size.
    shuffle : bool, optional (default=True)
        If False, the sampler won't shuffle the batches. padding_noise will be ignored and set to 0.0.
    """

    def __init__(
        self,
        batch_size: int,
        sorting_keys: Optional[List[str]] = None,
        padding_noise: float = 0.1,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        self.sorting_keys: Optional[List[str]] = sorting_keys
        self.padding_noise: float = padding_noise
        self.batch_size: int = batch_size
        self.drop_last: bool = drop_last
        self.shuffle: bool = shuffle
        if not shuffle:
            self.padding_noise = 0.0

    def _argsort_by_padding(
        self, instances: Iterable[Instance]
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Argsorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided). `sorting_keys`
        is a list of field names.
        """
        if not self.sorting_keys:
            logger.info("No sorting keys given; trying to guess a good one")
            self._guess_sorting_keys(instances)
            logger.info(f"Using {self.sorting_keys} as the sorting keys")
        instances_with_lengths: List[Tuple[List[float], List[int], Instance]] = []
        for instance in instances:
            lengths: List[int] = []
            noisy_lengths: List[float] = []
            for field_name in self.sorting_keys:  # type: ignore
                if field_name not in instance.fields:
                    raise ConfigurationError(
                        f'Sorting key "{field_name}" is not a field in instance. '
                        f"Available fields/keys are {list(instance.fields.keys())}."
                    )
                # Assuming that the field supports len().
                field_length: int = len(instance.fields[field_name])
                lengths.append(field_length)
                noisy_lengths.append(add_noise_to_value(field_length, self.padding_noise))
            instances_with_lengths.append((noisy_lengths, lengths, instance))
        with_indices: List[Tuple[Tuple[List[float], List[int], Instance], int]] = [
            (x, i) for i, x in enumerate(instances_with_lengths)
        ]
        with_indices.sort(key=lambda x: x[0][0])
        sorted_indices: List[int] = [instance_with_index[-1] for instance_with_index in with_indices]
        original_lengths: List[List[int]] = [instance_with_index[0][1] for instance_with_index in with_indices]
        return sorted_indices, original_lengths

    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        indices, _ = self._argsort_by_padding(instances)
        batches: List[List[int]] = []
        for group in lazy_groups_of(indices, self.batch_size):
            batch_indices: List[int] = list(group)
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            batches.append(batch_indices)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def _guess_sorting_keys(self, instances: Iterable[Instance], num_instances: int = 10) -> None:
        """
        Infer the keys used for sorting the dataset for bucketing by examining a subset of instances.
        """
        max_length: float = 0.0
        longest_field: Optional[str] = None
        for i, instance in enumerate(instances):
            # instance.fields is assumed to be a dict of field names to fields.
            for field_name, field in instance.fields.items():
                field_length: int = len(field)
                if field_length > max_length:
                    max_length = field_length
                    longest_field = field_name
            if i > num_instances:
                break

        if not longest_field:
            raise AssertionError(
                "Found no field that needed padding; we are surprised you got this error, please "
                "open an issue on github"
            )
        self.sorting_keys = [longest_field]

    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        batch_count_float: float = len(instances) / self.batch_size
        if self.drop_last:
            return math.floor(batch_count_float)
        else:
            return math.ceil(batch_count_float)

    def get_batch_size(self) -> Optional[int]:
        return self.batch_size