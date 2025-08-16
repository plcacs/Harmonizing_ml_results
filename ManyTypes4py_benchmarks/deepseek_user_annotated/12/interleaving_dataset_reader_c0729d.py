from typing import Dict, Mapping, Iterable, Union, Optional, Any, Iterator
import json


from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import (
    DatasetReader,
    PathOrStr,
    WorkerInfo,
    DistributedInfo,
)
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance

_VALID_SCHEMES = {"round_robin", "all_at_once"}


@DatasetReader.register("interleaving")
class InterleavingDatasetReader(DatasetReader):
    def __init__(
        self,
        readers: Dict[str, DatasetReader],
        dataset_field_name: str = "dataset",
        scheme: str = "round_robin",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._readers = readers
        self._dataset_field_name = dataset_field_name

        if scheme not in _VALID_SCHEMES:
            raise ConfigurationError(f"invalid scheme: {scheme}")
        self._scheme = scheme

    def _set_worker_info(self, info: Optional[WorkerInfo]) -> None:
        super()._set_worker_info(info)
        for reader in self._readers.values():
            reader._set_worker_info(info)

    def _set_distributed_info(self, info: Optional[DistributedInfo]) -> None:
        super()._set_distributed_info(info)
        for reader in self._readers.values():
            reader._set_distributed_info(info)

    def _read_round_robin(self, datasets: Mapping[str, Iterable[Instance]]) -> Iterator[Instance]:
        remaining = set(datasets)
        dataset_iterators = {key: iter(dataset) for key, dataset in datasets.items()}

        while remaining:
            for key, dataset in dataset_iterators.items():
                if key in remaining:
                    try:
                        instance = next(dataset)
                        instance.fields[self._dataset_field_name] = MetadataField(key)
                        yield instance
                    except StopIteration:
                        remaining.remove(key)

    def _read_all_at_once(self, datasets: Mapping[str, Iterable[Instance]]) -> Iterator[Instance]:
        for key, dataset in datasets.items():
            for instance in dataset:
                instance.fields[self._dataset_field_name] = MetadataField(key)
                yield instance

    def _read(self, file_path: Union[str, Dict[str, PathOrStr]]) -> Iterator[Instance]:
        if isinstance(file_path, str):
            try:
                file_paths = json.loads(file_path)
            except json.JSONDecodeError:
                raise ConfigurationError(
                    "the file_path for the InterleavingDatasetReader "
                    "needs to be a JSON-serialized dictionary {reader_name -> file_path}"
                )
        else:
            file_paths = file_path

        if file_paths.keys() != self._readers.keys():
            raise ConfigurationError("mismatched keys")

        datasets = {key: reader.read(file_paths[key]) for key, reader in self._readers.items()}

        if self._scheme == "round_robin":
            yield from self._read_round_robin(datasets)
        elif self._scheme == "all_at_once":
            yield from self._read_all_at_once(datasets)
        else:
            raise RuntimeError("impossible to get here")

    def text_to_instance(self, dataset_key: str, *args: Any, **kwargs: Any) -> Instance:
        return self._readers[dataset_key].text_to_instance(*args, **kwargs)

    def apply_token_indexers(self, instance: Instance) -> None:
        dataset = instance.fields[self._dataset_field_name].metadata
        self._readers[dataset].apply_token_indexers(instance)
