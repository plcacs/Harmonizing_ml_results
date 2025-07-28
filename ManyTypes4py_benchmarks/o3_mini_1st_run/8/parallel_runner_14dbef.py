from __future__ import annotations
from concurrent.futures import Executor, ProcessPoolExecutor
from multiprocessing.managers import BaseProxy, SyncManager
from multiprocessing.reduction import ForkingPickler
from pickle import PicklingError
from typing import TYPE_CHECKING, Any, Optional, Dict, Iterable

from kedro.io import CatalogProtocol, DatasetNotFoundError, MemoryDataset, SharedMemoryDataset
from kedro.runner.runner import AbstractRunner

if TYPE_CHECKING:
    from collections.abc import Iterable as IterableABC
    from pluggy import PluginManager
    from kedro.pipeline import Pipeline
    from kedro.pipeline.node import Node


class ParallelRunnerManager(SyncManager):
    MemoryDataset = MemoryDataset  # type: Any

ParallelRunnerManager.register('MemoryDataset', MemoryDataset)


class ParallelRunner(AbstractRunner):
    def __init__(self, 
                 max_workers: Optional[int] = None, 
                 is_async: bool = False, 
                 extra_dataset_patterns: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        default_dataset_pattern: Dict[str, Dict[str, str]] = {'{default}': {'type': 'SharedMemoryDataset'}}
        self._extra_dataset_patterns: Dict[str, Dict[str, str]] = extra_dataset_patterns or default_dataset_pattern
        super().__init__(is_async=is_async, extra_dataset_patterns=self._extra_dataset_patterns)
        self._manager: ParallelRunnerManager = ParallelRunnerManager()
        self._manager.start()
        self._max_workers: int = self._validate_max_workers(max_workers)

    def __del__(self) -> None:
        self._manager.shutdown()

    @classmethod
    def _validate_nodes(cls, nodes: Iterable[Node]) -> None:
        unserialisable: list[Node] = []
        for node in nodes:
            try:
                ForkingPickler.dumps(node)
            except (AttributeError, PicklingError):
                unserialisable.append(node)
        if unserialisable:
            raise AttributeError(
                f'The following nodes cannot be serialised: {sorted(unserialisable)}\n'
                'In order to utilize multiprocessing you need to make sure all nodes are serialisable, '
                'i.e. nodes should not include lambda functions, nested functions, closures, etc.\n'
                'If you are using custom decorators ensure they are correctly decorated using functools.wraps().'
            )

    @classmethod
    def _validate_catalog(cls, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        datasets: Dict[str, Any] = catalog._datasets
        unserialisable: list[str] = []
        for name, dataset in datasets.items():
            if getattr(dataset, '_SINGLE_PROCESS', False):
                unserialisable.append(name)
                continue
            try:
                ForkingPickler.dumps(dataset)
            except (AttributeError, PicklingError):
                unserialisable.append(name)
        if unserialisable:
            raise AttributeError(
                f'The following datasets cannot be used with multiprocessing: {sorted(unserialisable)}\n'
                'In order to utilize multiprocessing you need to make sure all datasets are serialisable, '
                'i.e. datasets should not make use of lambda functions, nested functions, closures etc.\n'
                'If you are using custom decorators ensure they are correctly decorated using functools.wraps().'
            )
        memory_datasets: list[str] = []
        for name, dataset in datasets.items():
            if name in pipeline.all_outputs() and isinstance(dataset, MemoryDataset) and (not isinstance(dataset, BaseProxy)):
                memory_datasets.append(name)
        if memory_datasets:
            raise AttributeError(
                f'The following datasets are memory datasets: {sorted(memory_datasets)}\n'
                'ParallelRunner does not support output to externally created MemoryDatasets'
            )

    def _set_manager_datasets(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        for dataset in pipeline.datasets():
            try:
                catalog.exists(dataset)
            except DatasetNotFoundError:
                pass
        for name, ds in catalog._datasets.items():
            if isinstance(ds, SharedMemoryDataset):
                ds.set_manager(self._manager)

    def _get_required_workers_count(self, pipeline: Pipeline) -> int:
        required_processes: int = len(pipeline.nodes) - len(pipeline.grouped_nodes) + 1
        return min(required_processes, self._max_workers)

    def _get_executor(self, max_workers: int) -> Executor:
        return ProcessPoolExecutor(max_workers=max_workers)

    def _run(self, 
             pipeline: Pipeline, 
             catalog: CatalogProtocol, 
             hook_manager: Optional[PluginManager] = None, 
             session_id: Optional[str] = None) -> None:
        if not self._is_async:
            self._logger.info(
                'Using synchronous mode for loading and saving data. Use the --async flag for potential performance gains. '
                'https://docs.kedro.org/en/stable/nodes_and_pipelines/run_a_pipeline.html#load-and-save-asynchronously'
            )
        super()._run(pipeline=pipeline, catalog=catalog, session_id=session_id)