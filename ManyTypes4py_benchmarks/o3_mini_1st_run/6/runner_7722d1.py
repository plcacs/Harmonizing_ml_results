#!/usr/bin/env python
"""``AbstractRunner`` is the base class for all ``Pipeline`` runner
implementations.
"""
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
from typing import Any, Deque, Collection, Iterable, Optional, Set, List, Dict
from pluggy import PluginManager
from kedro import KedroDeprecationWarning
from kedro.framework.hooks.manager import _NullPluginManager
from kedro.io import CatalogProtocol, MemoryDataset, SharedMemoryDataset
from kedro.pipeline import Pipeline
from kedro.runner.task import Task

if TYPE_CHECKING:
    from kedro.pipeline.node import Node

_MAX_WINDOWS_WORKERS = 61

class AbstractRunner(ABC):
    """``AbstractRunner`` is the base class for all ``Pipeline`` runner
    implementations.
    """

    def __init__(self, is_async: bool = False, extra_dataset_patterns: Optional[Any] = None) -> None:
        """Instantiates the runner class.

        Args:
            is_async: If True, the node inputs and outputs are loaded and saved
                asynchronously with threads. Defaults to False.
            extra_dataset_patterns: Extra dataset factory patterns to be added to the catalog
                during the run. This is used to set the default datasets on the Runner instances.
        """
        self._is_async: bool = is_async
        self._extra_dataset_patterns: Optional[Any] = extra_dataset_patterns

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(self.__module__)

    def run(
        self, 
        pipeline: Pipeline, 
        catalog: CatalogProtocol, 
        hook_manager: Optional[PluginManager] = None, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the ``Pipeline`` using the datasets provided by ``catalog``
        and save results back to the same objects.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: An implemented instance of ``CatalogProtocol`` from which to fetch data.
            hook_manager: The ``PluginManager`` to activate hooks.
            session_id: The id of the session.

        Raises:
            ValueError: Raised when ``Pipeline`` inputs cannot be satisfied.

        Returns:
            Any node outputs that cannot be processed by the catalog.
            These are returned in a dictionary, where the keys are defined
            by the node outputs.
        """
        warmed_up_ds: List[str] = []
        for ds in pipeline.datasets():
            if ds in catalog:
                warmed_up_ds.append(ds)
                _ = catalog._get_dataset(ds)
        unsatisfied: Set[str] = pipeline.inputs() - set(warmed_up_ds)
        if unsatisfied:
            raise ValueError(f'Pipeline input(s) {unsatisfied} not found in the {catalog.__class__.__name__}')
        catalog = catalog.shallow_copy(extra_dataset_patterns=self._extra_dataset_patterns)
        hook_or_null_manager: PluginManager = hook_manager or _NullPluginManager()
        registered_ds: List[str] = [ds for ds in pipeline.datasets() if ds in catalog]
        if self._is_async:
            self._logger.info('Asynchronous mode is enabled for loading and saving data')
        self._run(pipeline, catalog, hook_or_null_manager, session_id)
        self._logger.info('Pipeline execution completed successfully.')
        memory_datasets: Set[str] = {
            ds_name
            for ds_name, ds in catalog._datasets.items()
            if isinstance(ds, MemoryDataset) or isinstance(ds, SharedMemoryDataset)
        }
        free_outputs: Set[str] = pipeline.outputs() - (set(registered_ds) - memory_datasets)
        run_output: Dict[str, Any] = {ds_name: catalog.load(ds_name) for ds_name in free_outputs}
        if self._extra_dataset_patterns:
            catalog.config_resolver.remove_runtime_patterns(self._extra_dataset_patterns)
        return run_output

    def run_only_missing(
        self, 
        pipeline: Pipeline, 
        catalog: CatalogProtocol, 
        hook_manager: PluginManager
    ) -> Dict[str, Any]:
        """Run only the missing outputs from the ``Pipeline`` using the
        datasets provided by ``catalog``, and save results back to the
        same objects.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: An implemented instance of ``CatalogProtocol`` from which to fetch data.
            hook_manager: The ``PluginManager`` to activate hooks.

        Raises:
            ValueError: Raised when ``Pipeline`` inputs cannot be
                satisfied.

        Returns:
            Any node outputs that cannot be processed by the
            catalog. These are returned in a dictionary, where
            the keys are defined by the node outputs.
        """
        free_outputs: Set[str] = pipeline.outputs() - set(catalog.list())
        missing: Set[str] = {ds for ds in catalog.list() if not catalog.exists(ds)}
        to_build: Set[str] = free_outputs | missing
        to_rerun: Pipeline = pipeline.only_nodes_with_outputs(*to_build) + pipeline.from_inputs(*to_build)
        unregistered_ds: Set[str] = pipeline.datasets() - set(catalog.list())
        output_to_unregistered: Pipeline = pipeline.only_nodes_with_outputs(*unregistered_ds)
        input_from_unregistered: Set[str] = to_rerun.inputs() & unregistered_ds
        to_rerun += output_to_unregistered.to_outputs(*input_from_unregistered)
        return self.run(to_rerun, catalog, hook_manager)

    @abstractmethod
    def _get_executor(self, max_workers: int) -> Executor:
        """Abstract method to provide the correct executor (e.g., ThreadPoolExecutor or ProcessPoolExecutor)."""
        pass

    @abstractmethod
    def _run(
        self, 
        pipeline: Pipeline, 
        catalog: CatalogProtocol, 
        hook_manager: Optional[PluginManager] = None, 
        session_id: Optional[str] = None
    ) -> None:
        """The abstract interface for running pipelines, assuming that the
         inputs have already been checked and normalized by run().
         This contains the common pipeline execution logic using an executor.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: An implemented instance of ``CatalogProtocol`` from which to fetch data.
            hook_manager: The ``PluginManager`` to activate hooks.
            session_id: The id of the session.
        """
        nodes: List[Node] = pipeline.nodes
        self._validate_catalog(catalog, pipeline)
        self._validate_nodes(nodes)
        self._set_manager_datasets(catalog, pipeline)
        load_counts: Counter[str] = Counter(chain.from_iterable((n.inputs for n in pipeline.nodes)))
        node_dependencies: Dict[Node, Set[Node]] = pipeline.node_dependencies
        todo_nodes: Set[Node] = set(node_dependencies.keys())
        done_nodes: Set[Node] = set()
        futures: Set[Future[Any]] = set()
        done: Set[Future[Any]] = set()
        max_workers: int = self._get_required_workers_count(pipeline)
        with self._get_executor(max_workers) as pool:
            while True:
                ready: Set[Node] = {n for n in todo_nodes if node_dependencies[n] <= done_nodes}
                todo_nodes -= ready
                for node in ready:
                    task: Task = Task(node=node, catalog=catalog, hook_manager=hook_manager, is_async=self._is_async, session_id=session_id)
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
                        node_result: Node = future.result()
                    except Exception:
                        self._suggest_resume_scenario(pipeline, done_nodes, catalog)
                        raise
                    done_nodes.add(node_result)
                    self._logger.info('Completed node: %s', node_result.name)
                    self._logger.info('Completed %d out of %d tasks', len(done_nodes), len(nodes))
                    self._release_datasets(node_result, catalog, load_counts, pipeline)

    @staticmethod
    def _raise_runtime_error(
        todo_nodes: Set[Node], 
        done_nodes: Set[Node], 
        ready: Set[Node], 
        done: Set[Future[Any]]
    ) -> None:
        debug_data: Dict[str, Any] = {
            'todo_nodes': todo_nodes, 
            'done_nodes': done_nodes, 
            'ready_nodes': ready, 
            'done_futures': done
        }
        debug_data_str: str = '\n'.join((f'{k} = {v}' for k, v in debug_data.items()))
        raise RuntimeError(f'Unable to schedule new tasks although some nodes have not been run:\n{debug_data_str}')

    def _suggest_resume_scenario(
        self, 
        pipeline: Pipeline, 
        done_nodes: Set[Node], 
        catalog: CatalogProtocol
    ) -> None:
        """
        Suggest a command to the user to resume a run after it fails.
        The run should be started from the point closest to the failure
        for which persisted input exists.

        Args:
            pipeline: the ``Pipeline`` of the run.
            done_nodes: the ``Node``s that executed successfully.
            catalog: an implemented instance of ``CatalogProtocol`` of the run.
        """
        remaining_nodes: Set[Node] = set(pipeline.nodes) - set(done_nodes)
        postfix: str = ''
        if done_nodes:
            start_node_names: Set[str] = _find_nodes_to_resume_from(pipeline=pipeline, unfinished_nodes=remaining_nodes, catalog=catalog)
            start_nodes_str: str = ','.join(sorted(start_node_names))
            postfix += f'  --from-nodes "{start_nodes_str}"'
        if not postfix:
            self._logger.warning('No nodes ran. Repeat the previous command to attempt a new run.')
        else:
            self._logger.warning(
                f'There are {len(remaining_nodes)} nodes that have not run.\n'
                f'You can resume the pipeline run from the nearest nodes with persisted inputs by adding the following argument to your previous command:\n{postfix}'
            )

    @staticmethod
    def _release_datasets(
        node: Node, 
        catalog: CatalogProtocol, 
        load_counts: Counter[str], 
        pipeline: Pipeline
    ) -> None:
        """Decrement dataset load counts and release any datasets we've finished with"""
        for dataset in node.inputs:
            load_counts[dataset] -= 1
            if load_counts[dataset] < 1 and dataset not in pipeline.inputs():
                catalog.release(dataset)
        for dataset in node.outputs:
            if load_counts[dataset] < 1 and dataset not in pipeline.outputs():
                catalog.release(dataset)

    def _validate_catalog(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        pass

    def _validate_nodes(self, nodes: Iterable[Node]) -> None:
        pass

    def _set_manager_datasets(self, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        pass

    def _get_required_workers_count(self, pipeline: Pipeline) -> int:
        return 1

    @classmethod
    def _validate_max_workers(cls, max_workers: Optional[int]) -> int:
        """
        Validates and returns the number of workers. Sets to os.cpu_count() or 1 if max_workers is None,
        and limits max_workers to 61 on Windows.

        Args:
            max_workers: Desired number of workers. If None, defaults to os.cpu_count() or 1.

        Returns:
            A valid number of workers to use.

        Raises:
            ValueError: If max_workers is set and is not positive.
        """
        if max_workers is None:
            max_workers = os.cpu_count() or 1
            if sys.platform == 'win32':
                max_workers = min(_MAX_WINDOWS_WORKERS, max_workers)
        elif max_workers <= 0:
            raise ValueError('max_workers should be positive')
        return max_workers

def _find_nodes_to_resume_from(
    pipeline: Pipeline, 
    unfinished_nodes: Collection[Node], 
    catalog: CatalogProtocol
) -> Set[str]:
    """Given a collection of unfinished nodes in a pipeline using
    a certain catalog, find the node names to pass to pipeline.from_nodes()
    to cover all unfinished nodes, including any additional nodes
    that should be re-run if their outputs are not persisted.

    Args:
        pipeline: the ``Pipeline`` to find starting nodes for.
        unfinished_nodes: collection of ``Node``s that have not finished yet.
        catalog: an implemented instance of ``CatalogProtocol`` of the run.

    Returns:
        Set of node names to pass to pipeline.from_nodes() to continue
        the run.
    """
    nodes_to_be_run: Set[Node] = _find_all_nodes_for_resumed_pipeline(pipeline, unfinished_nodes, catalog)
    persistent_ancestors: List[Node] = _find_initial_node_group(pipeline, nodes_to_be_run)
    return {n.name for n in persistent_ancestors}

def _find_all_nodes_for_resumed_pipeline(
    pipeline: Pipeline, 
    unfinished_nodes: Collection[Node], 
    catalog: CatalogProtocol
) -> Set[Node]:
    """Breadth-first search approach to finding the complete set of
    ``Node``s which need to run to cover all unfinished nodes,
    including any additional nodes that should be re-run if their outputs
    are not persisted.

    Args:
        pipeline: the ``Pipeline`` to analyze.
        unfinished_nodes: the iterable of ``Node``s which have not finished yet.
        catalog: an implemented instance of ``CatalogProtocol`` of the run.

    Returns:
        A set containing all input unfinished ``Node``s and all remaining
        ``Node``s that need to run in case their outputs are not persisted.
    """
    nodes_to_run: Set[Node] = set(unfinished_nodes)
    initial_nodes: Set[Node] = _nodes_with_external_inputs(unfinished_nodes)
    queue: Deque[Node] = deque(initial_nodes)
    visited: Set[Node] = set(initial_nodes)
    while queue:
        current_node: Node = queue.popleft()
        nodes_to_run.add(current_node)
        non_persistent_inputs: Set[str] = _enumerate_non_persistent_inputs(current_node, catalog)
        for node in _enumerate_nodes_with_outputs(pipeline, non_persistent_inputs):
            if node in visited:
                continue
            visited.add(node)
            queue.append(node)
    nodes_to_run = set(pipeline.from_nodes(*(n.name for n in nodes_to_run)).nodes)
    return nodes_to_run

def _nodes_with_external_inputs(nodes_of_interest: Collection[Node]) -> Set[Node]:
    """For given ``Node``s, find their subset which depends on
    external inputs of the ``Pipeline`` they constitute. External inputs
    are pipeline inputs not produced by other ``Node``s in the ``Pipeline``.

    Args:
        nodes_of_interest: the ``Node``s to analyze.

    Returns:
        A set of ``Node``s that depend on external inputs
        of nodes of interest.
    """
    p_nodes_of_interest: Pipeline = Pipeline(list(nodes_of_interest))
    p_nodes_with_external_inputs: Pipeline = p_nodes_of_interest.only_nodes_with_inputs(*p_nodes_of_interest.inputs())
    return set(p_nodes_with_external_inputs.nodes)

def _enumerate_non_persistent_inputs(node: Node, catalog: CatalogProtocol) -> Set[str]:
    """Enumerate non-persistent input datasets of a ``Node``.

    Args:
        node: the ``Node`` to check the inputs of.
        catalog: an implemented instance of ``CatalogProtocol`` of the run.

    Returns:
        Set of names of non-persistent inputs of given ``Node``.
    """
    catalog_datasets: Dict[str, Any] = catalog._datasets
    non_persistent_inputs: Set[str] = set()
    for node_input in node.inputs:
        if node_input.startswith('params:'):
            continue
        if node_input not in catalog_datasets or catalog_datasets[node_input]._EPHEMERAL:
            non_persistent_inputs.add(node_input)
    return non_persistent_inputs

def _enumerate_nodes_with_outputs(pipeline: Pipeline, outputs: Collection[str]) -> List[Node]:
    """For given outputs, returns a list containing nodes that
    generate them in the given ``Pipeline``.

    Args:
        pipeline: the ``Pipeline`` to search for nodes in.
        outputs: the dataset names to find source nodes for.

    Returns:
        A list of all ``Node``s that are producing ``outputs``.
    """
    parent_pipeline: Pipeline = pipeline.only_nodes_with_outputs(*outputs)
    return parent_pipeline.nodes

def _find_initial_node_group(pipeline: Pipeline, nodes: Collection[Node]) -> List[Node]:
    """Given a collection of ``Node``s in a ``Pipeline``,
    find the initial group of ``Node``s to be run (in topological order).

    This can be used to define a sub-pipeline with the smallest possible
    set of nodes to pass to --from-nodes.

    Args:
        pipeline: the ``Pipeline`` to search for initial ``Node``s in.
        nodes: the ``Node``s to find initial group for.

    Returns:
        A list of initial ``Node``s to run given inputs (in topological order).
    """
    node_names: Set[str] = {n.name for n in nodes}
    if not node_names:
        return []
    sub_pipeline: Pipeline = pipeline.only_nodes(*node_names)
    initial_nodes: List[Node] = sub_pipeline.grouped_nodes[0]
    return initial_nodes

def run_node(
    node: Node, 
    catalog: CatalogProtocol, 
    hook_manager: PluginManager, 
    is_async: bool = False, 
    session_id: Optional[str] = None
) -> Node:
    """Run a single `Node` with inputs from and outputs to the `catalog`.

    Args:
        node: The ``Node`` to run.
        catalog: An implemented instance of ``CatalogProtocol`` containing the node's inputs and outputs.
        hook_manager: The ``PluginManager`` to activate hooks.
        is_async: If True, the node inputs and outputs are loaded and saved
            asynchronously with threads. Defaults to False.
        session_id: The session id of the pipeline run.

    Raises:
        ValueError: Raised if is_async is set to True for nodes wrapping
            generator functions.

    Returns:
        The node after execution.
    """
    warnings.warn(
        '`run_node()` has been deprecated and will be removed in Kedro 0.20.0', 
        KedroDeprecationWarning
    )
    if is_async and inspect.isgeneratorfunction(node.func):
        raise ValueError(
            f"Async data loading and saving does not work with nodes wrapping generator functions. "
            f"Please make sure you don't use `yield` anywhere in node {node!r}."
        )
    task: Task = Task(node=node, catalog=catalog, hook_manager=hook_manager, is_async=is_async, session_id=session_id)
    node_result: Node = task.execute()
    return node_result
