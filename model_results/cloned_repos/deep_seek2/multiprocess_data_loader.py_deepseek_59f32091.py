from collections import deque
import logging
from multiprocessing.process import BaseProcess
from multiprocessing.connection import Connection
import random
import traceback
import select
from queue import Full
from typing import List, Iterator, Optional, Iterable, Union, TypeVar, Tuple, Any, Dict, Callable

import torch
import torch.multiprocessing as mp

from allennlp.common.util import lazy_groups_of, shuffle_iterable
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.data.data_loaders.data_collator import DataCollator, DefaultDataCollator
from allennlp.data.dataset_readers import DatasetReader, WorkerInfo, DatasetReaderInput
from allennlp.data.fields import TextField
from allennlp.data.samplers import BatchSampler
from allennlp.data.vocabulary import Vocabulary
import allennlp.nn.util as nn_util

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

@DataLoader.register("multiprocess")
class MultiProcessDataLoader(DataLoader):
    def __init__(
        self,
        reader: DatasetReader,
        data_path: DatasetReaderInput,
        *,
        batch_size: Optional[int] = None,
        drop_last: bool = False,
        shuffle: bool = False,
        batch_sampler: Optional[BatchSampler] = None,
        batches_per_epoch: Optional[int] = None,
        num_workers: int = 0,
        max_instances_in_memory: Optional[int] = None,
        start_method: str = "fork",
        cuda_device: Optional[Union[int, str, torch.device]] = None,
        quiet: bool = False,
        collate_fn: DataCollator = DefaultDataCollator(),
    ) -> None:
        if num_workers < 0:
            raise ValueError("num_workers cannot be a negative number")

        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if batch_sampler is not None:
            if batch_size is not None:
                raise ValueError("batch_sampler option is mutually exclusive with batch_size")
            if drop_last:
                raise ValueError("batch_sampler option is mutually exclusive with drop_last")
            if shuffle:
                raise ValueError("batch_sampler option is mutually exclusive with shuffle")
        elif batch_size is None:
            raise ValueError("batch_size is required when batch_sampler is not supplied")

        if batches_per_epoch is not None and batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be at least 1")

        if max_instances_in_memory is not None:
            if batch_size is not None and max_instances_in_memory < batch_size:
                raise ValueError("max_instances_in_memory must be at least batch_size")
            elif max_instances_in_memory < 1:
                raise ValueError("max_instances_in_memory must be at least 1")

        self.reader = reader
        self.data_path = data_path
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_sampler = batch_sampler
        self.batches_per_epoch = batches_per_epoch
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.max_instances_in_memory = max_instances_in_memory
        self.start_method = start_method
        self.quiet = quiet
        self.cuda_device: Optional[torch.device] = None
        if cuda_device is not None:
            if not isinstance(cuda_device, torch.device):
                self.cuda_device = torch.device(cuda_device)
            else:
                self.cuda_device = cuda_device

        self._worker_cuda_safe = self.start_method in {"spawn", "forkserver"}

        effective_batch_size = (
            self.batch_size if self.batch_sampler is None else self.batch_sampler.get_batch_size()
        )
        self._max_instance_queue_size = (
            None
            if max_instances_in_memory is None
            else 2 * self.num_workers * max_instances_in_memory
        )
        self._max_batch_queue_size = (
            None
            if max_instances_in_memory is None
            else 2 * self.num_workers * max_instances_in_memory // (effective_batch_size or 1)
        )

        self._instances: Optional[List[Instance]] = None
        self._batch_generator: Optional[Iterator[TensorDict]] = None
        self._vocab: Optional[Vocabulary] = None

        if self.max_instances_in_memory is None:
            deque(self.iter_instances(), maxlen=0)

    def index_with(self, vocab: Vocabulary) -> None:
        self._vocab = vocab
        if self._instances:
            for instance in self._instances:
                instance.index_fields(vocab)

    def __len__(self) -> int:
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch
        elif self.max_instances_in_memory is None:
            if not self._instances:
                deque(self.iter_instances(), maxlen=0)

            if self.batch_sampler is not None:
                return self.batch_sampler.get_num_batches(self._instances)  # type: ignore

            num_instances = len(self._instances)  # type: ignore
            batch_size: int = self.batch_size  # type: ignore
            if self.drop_last or num_instances % batch_size == 0:
                return num_instances // batch_size
            else:
                return 1 + num_instances // batch_size
        else:
            raise TypeError

    def __iter__(self) -> Iterator[TensorDict]:
        if self._vocab is None:
            raise ValueError(
                "This DataLoader has not been indexed with a Vocabulary yet. "
                "Did you forget to call DataLoader.index_with(vocab)?"
            )

        if self.batches_per_epoch is None:
            yield from self._iter_batches()
        else:
            if self._batch_generator is not None:
                batch_generator = self._batch_generator
                self._batch_generator = None
            else:
                batch_generator = self._iter_batches()
            for i in range(self.batches_per_epoch):
                try:
                    yield next(batch_generator)
                except StopIteration:
                    batch_generator = self._iter_batches()
                    yield next(batch_generator)
            self._batch_generator = batch_generator

    def iter_instances(self) -> Iterator[Instance]:
        if self._instances:
            yield from self._instances
        else:
            if self.max_instances_in_memory is None:
                self._instances = []

            if self.num_workers <= 0:
                for instance in self._maybe_tqdm(
                    self.reader.read(self.data_path), desc="loading instances"
                ):
                    self.reader.apply_token_indexers(instance)
                    if self.max_instances_in_memory is None:
                        self._instances.append(instance)  # type: ignore
                    if self._vocab is not None:
                        instance.index_fields(self._vocab)
                    yield instance
            else:
                ctx = mp.get_context(self.start_method)
                queue: mp.JoinableQueue = (
                    ctx.JoinableQueue()
                    if self._max_instance_queue_size is None
                    else ctx.JoinableQueue(maxsize=self._max_instance_queue_size)
                )
                workers, txs = self._start_instance_workers(queue, ctx)

                try:
                    for instance in self._maybe_tqdm(
                        self._gather_instances(queue), desc="loading instances"
                    ):
                        if self.max_instances_in_memory is None:
                            self._instances.append(instance)  # type: ignore
                        yield instance
                finally:
                    if hasattr(queue, "close"):
                        queue.close()  # type: ignore[attr-defined]
                    self._join_workers(workers, queue, txs)

    def set_target_device(self, device: torch.device) -> None:
        self.cuda_device = device

    def _iter_batches(self) -> Iterator[TensorDict]:
        if self._instances is not None or self.num_workers <= 0:
            for batch in self._instances_to_batches(self.iter_instances(), move_to_device=True):
                yield batch
        else:
            ctx = mp.get_context(self.start_method)

            queue: mp.JoinableQueue = (
                ctx.JoinableQueue()
                if self._max_batch_queue_size is None
                else ctx.JoinableQueue(maxsize=self._max_batch_queue_size)
            )
            workers, txs = self._start_batch_workers(queue, ctx)

            try:
                done_count: int = 0
                while done_count < self.num_workers:
                    for batch, worker_error in iter(queue.get, (None, None)):
                        if worker_error is not None:
                            e, tb = worker_error
                            raise WorkerError(e, tb)

                        if not self._worker_cuda_safe and self.cuda_device is not None:
                            batch = nn_util.move_to_device(batch, self.cuda_device)
                        yield batch
                        queue.task_done()
                    done_count += 1
            finally:
                if hasattr(queue, "close"):
                    queue.close()  # type: ignore[attr-defined]
                self._join_workers(workers, queue, txs)

    def _start_instance_workers(
        self, queue: mp.JoinableQueue, ctx
    ) -> Tuple[List[BaseProcess], List[Connection]]:
        Tqdm.set_lock(mp.RLock())
        workers: List[BaseProcess] = []
        txs: List[Connection] = []
        for worker_id in range(self.num_workers):
            rx, tx = ctx.Pipe(duplex=False)
            worker: BaseProcess = ctx.Process(
                target=self._instance_worker,
                args=(worker_id, queue, Tqdm.get_lock(), rx),
                daemon=True,
            )
            worker.start()
            workers.append(worker)
            txs.append(tx)
        return workers, txs

    def _start_batch_workers(
        self, queue: mp.JoinableQueue, ctx
    ) -> Tuple[List[BaseProcess], List[Connection]]:
        Tqdm.set_lock(mp.RLock())
        workers: List[BaseProcess] = []
        txs: List[Connection] = []
        for worker_id in range(self.num_workers):
            rx, tx = ctx.Pipe(duplex=False)
            worker: BaseProcess = ctx.Process(
                target=self._batch_worker, args=(worker_id, queue, Tqdm.get_lock(), rx), daemon=True
            )
            worker.start()
            workers.append(worker)
            txs.append(tx)
        return workers, txs

    def _join_workers(self, workers: List[BaseProcess], queue, txs: List[Connection]) -> None:
        for _ in range(len(workers)):
            try:
                queue.task_done()
            except ValueError:
                break
        for tx in txs:
            tx.send("stop")

        for i, worker in enumerate(workers):
            worker.join(1)
            if worker.is_alive():
                logger.warning("terminating worker %s", i)
                worker.terminate()

    def _safe_queue_put(
        self, worker_id: int, item: Any, queue: mp.JoinableQueue, rx: Connection
    ) -> bool:
        while True:
            if rx.poll():
                logger.warning(
                    "worker %d received stop message from parent, exiting now", worker_id
                )
                queue.cancel_join_thread()
                return False
            fds, _, _ = select.select([rx.fileno()], [], [], 0)
            if fds:
                logger.warning("worker %d parent process has died, exiting now", worker_id)
                queue.cancel_join_thread()
                return False
            try:
                queue.put(item, True, 0.1)
                return True
            except Full:
                continue

    def _instance_worker(
        self, worker_id: int, queue: mp.JoinableQueue, lock, rx: Connection
    ) -> None:
        Tqdm.set_lock(lock)
        try:
            self.reader._set_worker_info(WorkerInfo(self.num_workers, worker_id))
            instances = self.reader.read(self.data_path)
            checked_for_token_indexers: bool = False
            for instance in instances:
                if not checked_for_token_indexers:
                    for field_name, field in instance.fields.items():
                        if isinstance(field, TextField) and field._token_indexers is not None:
                            raise ValueError(
                                f"Found a TextField ({field_name}) with token_indexers already "
                                "applied, but you're using num_workers > 0 in your data loader. "
                                "Make sure your dataset reader's text_to_instance() method doesn't "
                                "add any token_indexers to the TextFields it creates. Instead, the token_indexers "
                                "should be added to the instances in the apply_token_indexers() method of your "
                                "dataset reader (which you'll have to implement if you haven't done "
                                "so already)."
                            )
                    checked_for_token_indexers = True
                if self._safe_queue_put(worker_id, (instance, None), queue, rx):
                    continue
                else:
                    return
        except Exception as e:
            if not self._safe_queue_put(
                worker_id, (None, (repr(e), traceback.format_exc())), queue, rx
            ):
                return

        queue.put((None, None))
        queue.join()

    def _batch_worker(self, worker_id: int, queue: mp.JoinableQueue, lock, rx: Connection) -> None:
        Tqdm.set_lock(lock)
        try:
            self.reader._set_worker_info(WorkerInfo(self.num_workers, worker_id))
            instances = self.reader.read(self.data_path)
            for batch in self._instances_to_batches(
                instances, move_to_device=self._worker_cuda_safe
            ):
                if self._safe_queue_put(worker_id, (batch, None), queue, rx):
                    continue
                else:
                    return
        except Exception as e:
            if not self._safe_queue_put(
                worker_id, (None, (repr(e), traceback.format_exc())), queue, rx
            ):
                return

        queue.put((None, None))
        queue.join()

    def _gather_instances(self, queue: mp.JoinableQueue) -> Iterable[Instance]:
        done_count: int = 0
        while done_count < self.num_workers:
            for instance, worker_error in iter(queue.get, (None, None)):
                if worker_error is not None:
                    e, tb = worker_error
                    raise WorkerError(e, tb)

                self.reader.apply_token_indexers(instance)
                if self._vocab is not None:
                    instance.index_fields(self._vocab)
                yield instance
                queue.task_done()
            done_count += 1

    def _index_instance(self, instance: Instance) -> Instance:
        self.reader.apply_token_indexers(instance)
        assert self._vocab is not None
        instance.index_fields(self._vocab)
        return instance

    def _instances_to_batches(
        self, instance_iterator: Iterable[Instance], move_to_device: bool
    ) -> Iterator[TensorDict]:
        instance_iterator = (self._index_instance(instance) for instance in instance_iterator)

        if move_to_device and self.cuda_device is not None:
            tensorize: Callable[[List[Instance]], TensorDict] = lambda batch: nn_util.move_to_device(
                self.collate_fn(batch), self.cuda_device
            )
        else:
            tensorize = self.collate_fn

        if self.batch_sampler is not None:
            instance_chunks: Iterable[List[Instance]]

            if self.max_instances_in_memory is not None:
                instance_chunks = lazy_groups_of(instance_iterator, self.max_instances_in_memory)
            else:
                instance_chunks = [list(instance_iterator)]

            for instances in instance_chunks:
                batches = (
                    [instances[i] for i in batch_indices]
                    for batch_indices in self.batch_sampler.get_batch_indices(instances)
                )
                for batch in batches:
                    yield tensorize(batch)
        else:
            assert self.batch_size is not None

            if self.shuffle:
                if self.max_instances_in_memory is not None:
                    instance_iterator = shuffle_iterable(
                        instance_iterator,
                        self.max_instances_in_memory,
                    )
                else:
                    instance_iterator = list(instance_iterator)
                    random.shuffle(instance_iterator)

            for batch in lazy_groups_of(instance_iterator, self.batch_size):
                if self.drop_last and len(batch) < self.batch_size:
                    break
                yield tensorize(batch)

    def _maybe_tqdm(self, iterator: Iterable[_T], **tqdm_kwargs) -> Iterable[_T]:
        if self.quiet:
            return iterator
        return Tqdm.tqdm(iterator, **tqdm_kwargs)


class WorkerError(Exception):
    def __init__(self, original_err_repr: str, traceback: List[str]) -> None:
        super().__init__(
            f"worker raised {original_err_repr}\n\n"
            "  Traceback from worker:\n  " + "".join(traceback)
            .replace("Traceback (most recent call last):\n", "")
            .replace("\n", "\n  ")
        )
