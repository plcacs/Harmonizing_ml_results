"""Stub file for runner_7722d1 module."""

from __future__ import annotations
from collections import Counter, deque
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from typing import (
    Any,
    Callable,
    Counter,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from pluggy import PluginManager
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.io import CatalogProtocol

class AbstractRunner(ABC):
    """Stub for AbstractRunner class."""
    def __init__(self, is_async: bool = ..., extra_dataset_patterns: Optional[List[str]] = ...) -> None:
        ...
    
    @property
    def _logger(self) -> Any:
        ...
    
    def run(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: Optional[PluginManager] = ..., session_id: Optional[str] = ...) -> Dict[str, Any]:
        ...
    
    def run_only_missing(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: PluginManager) -> Dict[str, Any]:
        ...
    
    @abstractmethod
    def _get_executor(self, max_workers: int) -> Executor:
        ...
    
    @abstractmethod
    def _run(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: Optional[PluginManager] = ..., session_id: Optional[str] = ...) -> None:
        ...
    
    def _validate_catalog(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        ...
    
    def _validate_nodes(self, nodes: List[Node]) -> None:
        ...
    
    def _set_manager_datasets(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        ...
    
    def _get_required_workers_count(self, pipeline: Pipeline) -> int:
        ...
    
    @staticmethod
    def _raise_runtime_error(todo_nodes: Set[Node], done_nodes: Set[Node], ready: Set[Node], done: Set[Future]) -> None:
        ...
    
    @staticmethod
    def _suggest_resume_scenario(pipeline: Pipeline, done_nodes: Set[Node], catalog: CatalogProtocol) -> None:
        ...
    
    @staticmethod
    def _release_datasets(node: Node, catalog: CatalogProtocol, load_counts: Counter[str], pipeline: Pipeline) -> None:
        ...
    
    @classmethod
    def _validate_max_workers(cls, max_workers: Optional[int]) -> int:
        ...

def _find_nodes_to_resume_from(pipeline: Pipeline, unfinished_nodes: Iterable[Node], catalog: CatalogProtocol) -> Set[str]:
    ...

def _find_all_nodes_for_resumed_pipeline(pipeline: Pipeline, unfinished_nodes: Iterable[Node], catalog: CatalogProtocol) -> Set[Node]:
    ...

def _nodes_with_external_inputs(nodes_of_interest: Iterable[Node]) -> Set[Node]:
    ...

def _enumerate_non_persistent_inputs(node: Node, catalog: CatalogProtocol) -> Set[str]:
    ...

def _enumerate_nodes_with_outputs(pipeline: Pipeline, outputs: Iterable[str]) -> List[Node]:
    ...

def _find_initial_node_group(pipeline: Pipeline, nodes: Iterable[Node]) -> List[Node]:
    ...

def run_node(node: Node, catalog: CatalogProtocol, hook_manager: PluginManager, is_async: bool = ..., session_id: Optional[str] = ...) -> Node:
    ...