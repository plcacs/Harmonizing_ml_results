    def __init__(self, readers: Dict[str, DatasetReader], dataset_field_name: str = 'dataset', scheme: str = 'round_robin', **kwargs: Any) -> None:
    def _set_worker_info(self, info: WorkerInfo) -> None:
    def _set_distributed_info(self, info: DistributedInfo) -> None:
    def _read_round_robin(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
    def _read_all_at_once(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
    def _read(self, file_path: Union[str, Mapping[str, str]]) -> Iterable[Instance]:
    def text_to_instance(self, dataset_key: str, *args: Any, **kwargs: Any) -> Instance:
    def apply_token_indexers(self, instance: Instance) -> None:
