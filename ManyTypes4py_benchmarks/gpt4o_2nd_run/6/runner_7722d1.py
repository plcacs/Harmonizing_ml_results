from __future__ import annotations
import inspect
import logging
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from itertools import chain
from typing import TYPE_CHECKING, Any, Optional, Set, Dict
from pluggy import PluginManager
from kedro import KedroDeprecationWarning
from kedro.framework.hooks.manager import _NullPluginManager
from kedro.io import CatalogProtocol, MemoryDataset, SharedMemoryDataset
from kedro.pipeline import Pipeline
from kedro.runner.task import Task

_MAX_WINDOWS_WORKERS = 61

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from kedro.pipeline.node import Node

class AbstractRunner(ABC):
    def __init__(self, is_async: bool = False, extra_dataset_patterns: Optional[Collection[str]] = None):
        self._is_async = is_async
        self._extra_dataset_patterns = extra_dataset_patterns

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(self.__module__)

    def run(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: Optional[PluginManager] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        warmed_up_ds = []
        for ds in pipeline.datasets():
            if ds in catalog:
                warmed_up_ds.append(ds)
                _ = catalog._get_dataset(ds)
        unsatisfied = pipeline.inputs() - set(warmed_up_ds)
        if unsatisfied:
            raise ValueError(f'Pipeline input(s) {unsatisfied} not found in the {catalog.__class__.__name__}')
        catalog = catalog.shallow_copy(extra_dataset_patterns=self._extra_dataset_patterns)
        hook_or_null_manager = hook_manager or _NullPluginManager()
        registered_ds = [ds for ds in pipeline.datasets() if ds in catalog]
        if self._is_async:
            self._logger.info('Asynchronous mode is enabled for loading and saving data')
        self._run(pipeline, catalog, hook_or_null_manager, session_id)
        self._logger.info('Pipeline execution completed successfully.')
        memory_datasets = {ds_name for ds_name, ds in catalog._datasets.items() if isinstance(ds, MemoryDataset) or isinstance(ds, SharedMemoryDataset)}
        free_outputs = pipeline.outputs() - (set(registered_ds) - memory_datasets)
        run_output = {ds_name: catalog.load(ds_name) for ds_name in free_outputs}
        if self._extra_dataset_patterns:
            catalog.config_resolver.remove_runtime_patterns(self._extra_dataset_patterns)
        return run_output

    def run_only_missing(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: PluginManager) -> Dict[str, Any]:
        free_outputs = pipeline.outputs() - set(catalog.list())
        missing = {ds for ds in catalog.list() if not catalog.exists(ds)}
        to_build = free_outputs | missing
        to_rerun = pipeline.only_nodes_with_outputs(*to_build) + pipeline.from_inputs(*to_build)
        unregistered_ds = pipeline.datasets() - set(catalog.list())
        output_to_unregistered = pipeline.only_nodes_with_outputs(*unregistered_ds)
        input_from_unregistered = to_rerun.inputs() & unregistered_ds
        to_rerun += output_to_unregistered.to_outputs(*input_from_unregistered)
        return self.run(to_rerun, catalog, hook_manager)

    @abstractmethod
    def _get_executor(self, max_workers: int) -> Executor:
        pass

    @abstractmethod
    def _run(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: Optional[PluginManager] = None, session_id: Optional[str] = None) -> None:
        nodes = pipeline.nodes
        self._validate_catalog(catalog, pipeline)
        self._validate_nodes(nodes)
        self._set_manager_datasets(catalog, pipeline)
        load_counts = Counter(chain.from_iterable((n.inputs for n in pipeline.nodes)))
        node_dependencies = pipeline.node_dependencies
        todo_nodes = set(node_dependencies.keys())
        done_nodes = set()
        futures: Set[Future] = set()
        done = None
        max_workers = self._get_required_workers_count(pipeline)
        with self._get_executor(max_workers) as pool:
            while True:
                ready = {n for n in todo_nodes if node_dependencies[n] <= done_nodes}
                todo_nodes -= ready
                for node in ready:
                    task = Task(node=node, catalog=catalog, hook_manager=hook_manager, is_async=self._is_async, session_id=session_id)
                    if isinstance(pool, ProcessPoolExecutor):
                        task.parallel = True
                    futures.add(pool.submit(task))
                if not futures:
                    if todo_nodes:
                        self._raise_runtime_error(todo_nodes, done_nodes, ready, done)
                    break
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        node = future.result()
                    except Exception:
                        self._suggest_resume_scenario(pipeline, done_nodes, catalog)
                        raise
                    done_nodes.add(node)
                    self._logger.info('Completed node: %s', node.name)
                    self._logger.info('Completed %d out of %d tasks', len(done_nodes), len(nodes))
                    self._release_datasets(node, catalog, load_counts, pipeline)

    @staticmethod
    def _raise_runtime_error(todo_nodes: Set[Node], done_nodes: Set[Node], ready: Set[Node], done: Optional[Set[Future]]) -> None:
        debug_data = {'todo_nodes': todo_nodes, 'done_nodes': done_nodes, 'ready_nodes': ready, 'done_futures': done}
        debug_data_str = '\n'.join((f'{k} = {v}' for k, v in debug_data.items()))
        raise RuntimeError(f'Unable to schedule new tasks although some nodes have not been run:\n{debug_data_str}')

    def _suggest_resume_scenario(self, pipeline: Pipeline, done_nodes: Set[Node], catalog: CatalogProtocol) -> None:
        remaining_nodes = set(pipeline.nodes) - set(done_nodes)
        postfix = ''
        if done_nodes:
            start_node_names = _find_nodes_to_resume_from(pipeline=pipeline, unfinished_nodes=remaining_nodes, catalog=catalog)
            start_nodes_str = ','.join(sorted(start_node_names))
            postfix += f'  --from-nodes "{start_nodes_str}"'
        if not postfix:
            self._logger.warning('No nodes ran. Repeat the previous command to attempt a new run.')
        else:
            self._logger.warning(f'There are {len(remaining_nodes)} nodes that have not run.\nYou can resume the pipeline run from the nearest nodes with persisted inputs by adding the following argument to your previous command:\n{postfix}')

    @staticmethod
    def _release_datasets(node: Node, catalog: CatalogProtocol, load_counts: Counter, pipeline: Pipeline) -> None:
        for dataset in node.inputs:
            load_counts[dataset] -= 1
            if load_counts[dataset] < 1 and dataset not in pipeline.inputs():
                catalog.release(dataset)
        for dataset in node.outputs:
            if load_counts[dataset] < 1 and dataset not in pipeline.outputs():
                catalog.release(dataset)

    def _validate_catalog(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        pass

    def _validate_nodes(self, node: Collection[Node]) -> None:
        pass

    def _set_manager_datasets(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        pass

    def _get_required_workers_count(self, pipeline: Pipeline) -> int:
        return 1

    @classmethod
    def _validate_max_workers(cls, max_workers: Optional[int]) -> int:
        if max_workers is None:
            max_workers = os.cpu_count() or 1
            if sys.platform == 'win32':
                max_workers = min(_MAX_WINDOWS_WORKERS, max_workers)
        elif max_workers <= 0:
            raise ValueError('max_workers should be positive')
        return max_workers

def _find_nodes_to_resume_from(pipeline: Pipeline, unfinished_nodes: Collection[Node], catalog: CatalogProtocol) -> Set[str]:
    nodes_to_be_run = _find_all_nodes_for_resumed_pipeline(pipeline, unfinished_nodes, catalog)
    persistent_ancestors = _find_initial_node_group(pipeline, nodes_to_be_run)
    return {n.name for n in persistent_ancestors}

def _find_all_nodes_for_resumed_pipeline(pipeline: Pipeline, unfinished_nodes: Iterable[Node], catalog: CatalogProtocol) -> Set[Node]:
    nodes_to_run = set(unfinished_nodes)
    initial_nodes = _nodes_with_external_inputs(unfinished_nodes)
    queue, visited = (deque(initial_nodes), set(initial_nodes))
    while queue:
        current_node = queue.popleft()
        nodes_to_run.add(current_node)
        non_persistent_inputs = _enumerate_non_persistent_inputs(current_node, catalog)
        for node in _enumerate_nodes_with_outputs(pipeline, non_persistent_inputs):
            if node in visited:
                continue
            visited.add(node)
            queue.append(node)
    nodes_to_run = set(pipeline.from_nodes(*(n.name for n in nodes_to_run)).nodes)
    return nodes_to_run

def _nodes_with_external_inputs(nodes_of_interest: Iterable[Node]) -> Set[Node]:
    p_nodes_of_interest = Pipeline(nodes_of_interest)
    p_nodes_with_external_inputs = p_nodes_of_interest.only_nodes_with_inputs(*p_nodes_of_interest.inputs())
    return set(p_nodes_with_external_inputs.nodes)

def _enumerate_non_persistent_inputs(node: Node, catalog: CatalogProtocol) -> Set[str]:
    catalog_datasets = catalog._datasets
    non_persistent_inputs = set()
    for node_input in node.inputs:
        if node_input.startswith('params:'):
            continue
        if node_input not in catalog_datasets or catalog_datasets[node_input]._EPHEMERAL:
            non_persistent_inputs.add(node_input)
    return non_persistent_inputs

def _enumerate_nodes_with_outputs(pipeline: Pipeline, outputs: Collection[str]) -> Collection[Node]:
    parent_pipeline = pipeline.only_nodes_with_outputs(*outputs)
    return parent_pipeline.nodes

def _find_initial_node_group(pipeline: Pipeline, nodes: Collection[Node]) -> Collection[Node]:
    node_names = {n.name for n in nodes}
    if len(node_names) == 0:
        return []
    sub_pipeline = pipeline.only_nodes(*node_names)
    initial_nodes = sub_pipeline.grouped_nodes[0]
    return initial_nodes

def run_node(node: Node, catalog: CatalogProtocol, hook_manager: PluginManager, is_async: bool = False, session_id: Optional[str] = None) -> Node:
    warnings.warn('`run_node()` has been deprecated and will be removed in Kedro 0.20.0', KedroDeprecationWarning)
    if is_async and inspect.isgeneratorfunction(node.func):
        raise ValueError(f"Async data loading and saving does not work with nodes wrapping generator functions. Please make sure you don't use `yield` anywhere in node {node!s}.")
    task = Task(node=node, catalog=catalog, hook_manager=hook_manager, is_async=is_async, session_id=session_id)
    node = task.execute()
    return node
