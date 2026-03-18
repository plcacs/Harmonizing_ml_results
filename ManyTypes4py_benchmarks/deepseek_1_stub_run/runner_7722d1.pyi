```python
from __future__ import annotations
from abc import ABC, abstractmethod
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from typing import TYPE_CHECKING, Any
from pluggy import PluginManager
from kedro.io import CatalogProtocol, MemoryDataset, SharedMemoryDataset
from kedro.pipeline import Pipeline
from kedro.runner.task import Task

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from pluggy import PluginManager
    from kedro.pipeline.node import Node

_MAX_WINDOWS_WORKERS: int = ...

class AbstractRunner(ABC):
    _is_async: bool
    _extra_dataset_patterns: Any | None
    
    def __init__(self, is_async: bool = ..., extra_dataset_patterns: Any | None = ...) -> None: ...
    
    @property
    def _logger(self) -> logging.Logger: ...
    
    def run(
        self,
        pipeline: Pipeline,
        catalog: CatalogProtocol,
        hook_manager: PluginManager | None = ...,
        session_id: Any | None = ...
    ) -> dict[str, Any]: ...
    
    def run_only_missing(
        self,
        pipeline: Pipeline,
        catalog: CatalogProtocol,
        hook_manager: PluginManager
    ) -> dict[str, Any]: ...
    
    @abstractmethod
    def _get_executor(self, max_workers: int) -> Executor: ...
    
    @abstractmethod
    def _run(
        self,
        pipeline: Pipeline,
        catalog: CatalogProtocol,
        hook_manager: PluginManager | None = ...,
        session_id: Any | None = ...
    ) -> None: ...
    
    @staticmethod
    def _raise_runtime_error(
        todo_nodes: set[Node],
        done_nodes: set[Node],
        ready: set[Node],
        done: set[Future[Any]] | None
    ) -> None: ...
    
    def _suggest_resume_scenario(
        self,
        pipeline: Pipeline,
        done_nodes: Collection[Node],
        catalog: CatalogProtocol
    ) -> None: ...
    
    @staticmethod
    def _release_datasets(
        node: Node,
        catalog: CatalogProtocol,
        load_counts: dict[str, int],
        pipeline: Pipeline
    ) -> None: ...
    
    def _validate_catalog(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None: ...
    
    def _validate_nodes(self, node: Iterable[Node]) -> None: ...
    
    def _set_manager_datasets(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None: ...
    
    def _get_required_workers_count(self, pipeline: Pipeline) -> int: ...
    
    @classmethod
    def _validate_max_workers(cls, max_workers: int | None) -> int: ...

def _find_nodes_to_resume_from(
    pipeline: Pipeline,
    unfinished_nodes: Collection[Node],
    catalog: CatalogProtocol
) -> set[str]: ...

def _find_all_nodes_for_resumed_pipeline(
    pipeline: Pipeline,
    unfinished_nodes: Iterable[Node],
    catalog: CatalogProtocol
) -> set[Node]: ...

def _nodes_with_external_inputs(nodes_of_interest: Collection[Node]) -> set[Node]: ...

def _enumerate_non_persistent_inputs(node: Node, catalog: CatalogProtocol) -> set[str]: ...

def _enumerate_nodes_with_outputs(pipeline: Pipeline, outputs: Iterable[str]) -> list[Node]: ...

def _find_initial_node_group(pipeline: Pipeline, nodes: Collection[Node]) -> list[Node]: ...

def run_node(
    node: Node,
    catalog: CatalogProtocol,
    hook_manager: PluginManager,
    is_async: bool = ...,
    session_id: Any | None = ...
) -> Node: ...
```