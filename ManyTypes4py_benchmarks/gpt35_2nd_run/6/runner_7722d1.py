from __future__ import annotations
import logging
import os
import sys
from abc import ABC, abstractmethod
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from itertools import chain
from typing import TYPE_CHECKING, Any, Set

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from pluggy import PluginManager
    from kedro.pipeline.node import Node

class AbstractRunner(ABC):
    def __init__(self, is_async: bool = False, extra_dataset_patterns: Set[str] = None) -> None:
        self._is_async = is_async
        self._extra_dataset_patterns = extra_dataset_patterns

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(self.__module__)

    def run(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: PluginManager = None, session_id: Any = None) -> Any:
        ...

    def run_only_missing(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: PluginManager) -> Any:
        ...

    @abstractmethod
    def _get_executor(self, max_workers: int) -> Executor:
        ...

    @abstractmethod
    def _run(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: PluginManager = None, session_id: Any = None) -> None:
        ...

    def _raise_runtime_error(self, todo_nodes: Set[Node], done_nodes: Set[Node], ready: Set[Node], done: Set[Future]) -> None:
        ...

    def _suggest_resume_scenario(self, pipeline: Pipeline, done_nodes: Set[Node], catalog: CatalogProtocol) -> None:
        ...

    def _release_datasets(self, node: Node, catalog: CatalogProtocol, load_counts: Counter, pipeline: Pipeline) -> None:
        ...

    def _validate_catalog(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        ...

    def _validate_nodes(self, node: Node) -> None:
        ...

    def _set_manager_datasets(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        ...

    def _get_required_workers_count(self, pipeline: Pipeline) -> int:
        ...

    @classmethod
    def _validate_max_workers(cls, max_workers: int) -> int:
        ...

def _find_nodes_to_resume_from(pipeline: Pipeline, unfinished_nodes: Set[Node], catalog: CatalogProtocol) -> Set[str]:
    ...

def _find_all_nodes_for_resumed_pipeline(pipeline: Pipeline, unfinished_nodes: Set[Node], catalog: CatalogProtocol) -> Set[Node]:
    ...

def _nodes_with_external_inputs(nodes_of_interest: Set[Node]) -> Set[Node]:
    ...

def _enumerate_non_persistent_inputs(node: Node, catalog: CatalogProtocol) -> Set[str]:
    ...

def _enumerate_nodes_with_outputs(pipeline: Pipeline, outputs: Set[str]) -> Set[Node]:
    ...

def _find_initial_node_group(pipeline: Pipeline, nodes: Set[Node]) -> List[Node]:
    ...

def run_node(node: Node, catalog: CatalogProtocol, hook_manager: PluginManager, is_async: bool = False, session_id: Any = None) -> Node:
    ...
