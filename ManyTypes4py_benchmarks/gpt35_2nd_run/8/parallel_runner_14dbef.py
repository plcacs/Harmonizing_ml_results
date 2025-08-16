from __future__ import annotations
from concurrent.futures import Executor, ProcessPoolExecutor
from multiprocessing.managers import BaseProxy, SyncManager
from multiprocessing.reduction import ForkingPickler
from pickle import PicklingError
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from kedro.io import CatalogProtocol, DatasetNotFoundError, MemoryDataset, SharedMemoryDataset
from kedro.runner.runner import AbstractRunner

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pluggy import PluginManager
    from kedro.pipeline import Pipeline
    from kedro.pipeline.node import Node

class ParallelRunnerManager(SyncManager):
    def register(self, name: str, type: Any) -> None: ...

class ParallelRunner(AbstractRunner):
    def __init__(self, max_workers: Optional[int] = None, is_async: bool = False, extra_dataset_patterns: Optional[Dict[str, Dict[str, str]]] = None) -> None: ...

    def __del__(self) -> None: ...

    @classmethod
    def _validate_nodes(cls, nodes: List[Node]) -> None: ...

    @classmethod
    def _validate_catalog(cls, catalog: CatalogProtocol, pipeline: Pipeline) -> None: ...

    def _set_manager_datasets(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None: ...

    def _get_required_workers_count(self, pipeline: Pipeline) -> int: ...

    def _get_executor(self, max_workers: int) -> Executor: ...

    def _run(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: Optional[PluginManager] = None, session_id: Optional[str] = None) -> None: ...
