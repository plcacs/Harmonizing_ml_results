from __future__ import annotations
import logging
import os
import sys
from abc import ABC, abstractmethod
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from typing import TYPE_CHECKING, Any
from pluggy import PluginManager
from kedro import KedroDeprecationWarning
from kedro.framework.hooks.manager import _NullPluginManager
from kedro.io import CatalogProtocol, MemoryDataset, SharedMemoryDataset
from kedro.pipeline import Pipeline
from kedro.runner.task import Task

_MAX_WINDOWS_WORKERS: int = 61

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from pluggy import PluginManager
    from kedro.pipeline.node import Node

class AbstractRunner(ABC):
    def __init__(self, is_async: bool = False, extra_dataset_patterns: Any = None) -> None:
        self._is_async: bool = is_async
        self._extra_dataset_patterns: Any = extra_dataset_patterns

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

    @staticmethod
    def _raise_runtime_error(todo_nodes: Collection, done_nodes: Collection, ready: Collection, done: Collection) -> None:
        ...

    def _suggest_resume_scenario(self, pipeline: Pipeline, done_nodes: Collection, catalog: CatalogProtocol) -> None:
        ...

    @staticmethod
    def _release_datasets(node: Node, catalog: CatalogProtocol, load_counts: Counter, pipeline: Pipeline) -> None:
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

def _find_nodes_to_resume_from(pipeline: Pipeline, unfinished_nodes: Collection, catalog: CatalogProtocol) -> set:
    ...

def _find_all_nodes_for_resumed_pipeline(pipeline: Pipeline, unfinished_nodes: Collection, catalog: CatalogProtocol) -> set:
    ...

def _nodes_with_external_inputs(nodes_of_interest: Collection) -> set:
    ...

def _enumerate_non_persistent_inputs(node: Node, catalog: CatalogProtocol) -> set:
    ...

def _enumerate_nodes_with_outputs(pipeline: Pipeline, outputs: Collection) -> list:
    ...

def _find_initial_node_group(pipeline: Pipeline, nodes: Collection) -> list:
    ...

def run_node(node: Node, catalog: CatalogProtocol, hook_manager: PluginManager, is_async: bool = False, session_id: Any = None) -> Node:
    ...
