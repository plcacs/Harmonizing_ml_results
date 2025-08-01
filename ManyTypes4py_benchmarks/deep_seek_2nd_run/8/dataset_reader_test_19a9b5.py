from itertools import islice
from typing import Optional, List, Set, Iterator, Any, Dict, Type, cast
import pytest
import torch.distributed as dist
from allennlp.common import util as common_util
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader, WorkerInfo
from allennlp.data.fields import LabelField
TOTAL_INSTANCES: int = 100

class MockDatasetReader(DatasetReader):

    def _read(self, file_path: str) -> Iterator[Instance]:
        for i in range(TOTAL_INSTANCES):
            yield self.text_to_instance(i)

    def text_to_instance(self, index: int) -> Instance:
        return Instance({'index': LabelField(index, skip_indexing=True)})

class MockMmpsDatasetReader(DatasetReader):
    """
    Implements manual multi-process sharding (MMPS).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(manual_multiprocess_sharding=True, **kwargs)

    def _read(self, file_path: str) -> Iterator[Instance]:
        start_index: int = 0
        step_size: int = 1
        worker_info: Optional[WorkerInfo] = self.get_worker_info()
        if worker_info is not None:
            start_index += worker_info.id
            step_size *= worker_info.num_workers
        for i in islice(range(TOTAL_INSTANCES), start_index, None, step_size):
            yield self.text_to_instance(i)

    def text_to_instance(self, index: int) -> Instance:
        return Instance({'index': LabelField(index, skip_indexing=True)})

class MockMdsDatasetReader(DatasetReader):
    """
    Implements manual distributed sharding (MDS).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(manual_distributed_sharding=True, **kwargs)

    def _read(self, file_path: str) -> Iterator[Instance]:
        start_index: int = 0
        step_size: int = 1
        if common_util.is_distributed():
            start_index += cast(int, dist.get_rank())
            step_size *= cast(int, dist.get_world_size())
        for i in islice(range(TOTAL_INSTANCES), start_index, None, step_size):
            yield self.text_to_instance(i)

    def text_to_instance(self, index: int) -> Instance:
        return Instance({'index': LabelField(index, skip_indexing=True)})

class MockMmpdsDatasetReader(DatasetReader):
    """
    Implements manual multi-process and distributed sharding (MMPDS).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs)

    def _read(self, file_path: str) -> Iterator[Instance]:
        for i in self.shard_iterable(range(TOTAL_INSTANCES)):
            yield self.text_to_instance(i)

    def text_to_instance(self, index: int) -> Instance:
        return Instance({'index': LabelField(index, skip_indexing=True)})

@pytest.mark.parametrize('world_size, num_workers, max_instances', [(4, 2, None), (4, 2, 67), (4, None, None), (4, None, None), (None, 2, None), (None, 2, 67), (None, None, None), (None, None, 67)])
@pytest.mark.parametrize('reader_class', [MockDatasetReader, MockMmpsDatasetReader, MockMdsDatasetReader, MockMmpdsDatasetReader])
def test_instance_slicing(
    monkeypatch: pytest.MonkeyPatch,
    reader_class: Type[DatasetReader],
    world_size: Optional[int],
    num_workers: Optional[int],
    max_instances: Optional[int]
) -> None:
    """
    Ensure that the intances read by each worker are always unique and the total
    adds up to `max_instances`.
    """
    results: List[Set[int]] = []
    minimum_expected_result_size: int = max_instances or TOTAL_INSTANCES
    maximum_expected_result_size: int = max_instances or TOTAL_INSTANCES
    if world_size is not None and num_workers is not None:
        minimum_expected_result_size //= world_size
        minimum_expected_result_size //= num_workers
        maximum_expected_result_size = minimum_expected_result_size + 1
        for global_rank in range(world_size):
            monkeypatch.setattr(common_util, 'is_distributed', lambda: True)
            monkeypatch.setattr(dist, 'get_rank', lambda: global_rank)
            monkeypatch.setattr(dist, 'get_world_size', lambda: world_size)
            for worker_id in range(num_workers):
                reader: DatasetReader = reader_class(max_instances=max_instances)
                reader._set_worker_info(WorkerInfo(num_workers, worker_id))
                result: Set[int] = set((x['index'].label for x in reader.read('the-path-doesnt-matter')))
                results.append(result)
    elif world_size is not None:
        minimum_expected_result_size //= world_size
        maximum_expected_result_size = minimum_expected_result_size + 1
        for global_rank in range(world_size):
            monkeypatch.setattr(common_util, 'is_distributed', lambda: True)
            monkeypatch.setattr(dist, 'get_rank', lambda: global_rank)
            monkeypatch.setattr(dist, 'get_world_size', lambda: world_size)
            reader = reader_class(max_instances=max_instances)
            result = set((x['index'].label for x in reader.read('the-path-doesnt-matter')))
            results.append(result)
    elif num_workers is not None:
        minimum_expected_result_size //= num_workers
        maximum_expected_result_size = minimum_expected_result_size + 1
        for worker_id in range(num_workers):
            reader = reader_class(max_instances=max_instances)
            reader._set_worker_info(WorkerInfo(num_workers, worker_id))
            result = set((x['index'].label for x in reader.read('the-path-doesnt-matter')))
            results.append(result)
    else:
        reader = reader_class(max_instances=max_instances)
        result = set((x['index'].label for x in reader.read('the-path-doesnt-matter')))
        results.append(result)
    union: Set[int] = set()
    total: int = 0
    for result in results:
        union |= result
        total += len(result)
        assert minimum_expected_result_size <= len(result)
        assert len(result) <= maximum_expected_result_size
    assert len(union) == total == (max_instances or TOTAL_INSTANCES)
