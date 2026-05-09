@DataLoader.register('multiprocess')
class MultiProcessDataLoader(DataLoader):
    def __init__(self, reader: DatasetReader, data_path: DatasetReaderInput, *, 
                 batch_size: Optional[int] = None, drop_last: bool = False, shuffle: bool = False, 
                 batch_sampler: Optional[BatchSampler] = None, batches_per_epoch: Optional[int] = None, 
                 num_workers: int = 0, max_instances_in_memory: Optional[int] = None, 
                 start_method: str = 'fork', cuda_device: Optional[Union[int, str, torch.device]] = None, 
                 quiet: bool = False, collate_fn: DataCollator = DefaultDataCollator()):
        ...

    def index_with(self, vocab: Vocabulary) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[TensorDict]:
        ...

    def iter_instances(self) -> Iterator[Instance]:
        ...

    def set_target_device(self, device: Union[int, str, torch.device]) -> None:
        ...

    def _iter_batches(self) -> Iterator[TensorDict]:
        ...

    def _start_instance_workers(self, queue: mp.JoinableQueue, ctx: mp.context.BaseContext) -> Tuple[List[BaseProcess], List[Connection]]:
        ...

    def _start_batch_workers(self, queue: mp.JoinableQueue, ctx: mp.context.BaseContext) -> Tuple[List[BaseProcess], List[Connection]]:
        ...

    def _join_workers(self, workers: List[BaseProcess], queue: mp.JoinableQueue, txs: List[Connection]) -> None:
        ...

    def _safe_queue_put(self, worker_id: int, item: Any, queue: mp.JoinableQueue, rx: Connection) -> bool:
        ...

    def _instance_worker(self, worker_id: int, queue: mp.JoinableQueue, lock: mp.RLock, rx: Connection) -> None:
        ...

    def _batch_worker(self, worker_id: int, queue: mp.JoinableQueue, lock: mp.RLock, rx: Connection) -> None:
        ...

    def _gather_instances(self, queue: mp.JoinableQueue) -> Iterator[Instance]:
        ...

    def _index_instance(self, instance: Instance) -> Instance:
        ...

    def _instances_to_batches(self, instance_iterator: Iterable[Instance], move_to_device: bool) -> Iterator[TensorDict]:
        ...

    def _maybe_tqdm(self, iterator: Iterable[Any], **tqdm_kwargs: Any) -> Iterable[Any]:
        ...

class WorkerError(Exception):
    def __init__(self, original_err_repr: str, traceback: List[str]):
        ...
