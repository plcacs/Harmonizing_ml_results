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
from typing import TYPE_CHECKING, Any, Collection, Iterable, PluginManager, Type

class AbstractRunner(ABC):
    """``AbstractRunner`` is the base class for all ``Pipeline`` runner
    implementations.
    """

    def __init__(self, is_async: bool = False, extra_dataset_patterns: Any = None) -> None:
        """Instantiates the runner class.

        Args:
            is_async: If True, the node inputs and outputs are loaded and saved
                asynchronously with threads. Defaults to False.
            extra_dataset_patterns: Extra dataset factory patterns to be added to the catalog
                during the run. This is used to set the default datasets on the Runner instances.

        """
        self._is_async: bool = is_async
        self._extra_dataset_patterns: Any = extra_dataset_patterns

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(self.__module__)

    def run(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: PluginManager = None, session_id: str = None) -> Any:
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
        ...

    def run_only_missing(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: PluginManager) -> Any:
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
        ...

    @abstractmethod
    def _get_executor(self, max_workers: int) -> Executor:
        """Abstract method to provide the correct executor (e.g., ThreadPoolExecutor or ProcessPoolExecutor)."""
        pass

    @abstractmethod
    def _run(self, pipeline: Pipeline, catalog: CatalogProtocol, hook_manager: PluginManager = None, session_id: str = None) -> None:
        """The abstract interface for running pipelines, assuming that the
         inputs have already been checked and normalized by run().
         This contains the Common pipeline execution logic using an executor.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: An implemented instance of ``CatalogProtocol`` from which to fetch data.
            hook_manager: The ``PluginManager`` to activate hooks.
            session_id: The id of the session.
        """
        ...

    @staticmethod
    def _raise_runtime_error(todo_nodes: Collection[Node], done_nodes: Collection[Node], ready: Collection[Node], done: Future) -> None:
        """Raises a runtime error when unable to schedule new tasks."""
        ...

    def _suggest_resume_scenario(self, pipeline: Pipeline, done_nodes: Collection[Node], catalog: CatalogProtocol) -> None:
        """Suggest a command to the user to resume a run after it fails."""
        ...

    def _release_datasets(self, node: Node, catalog: CatalogProtocol, load_counts: Counter, pipeline: Pipeline) -> None:
        """Decrement dataset load counts and release any datasets we've finished with"""
        ...

    @classmethod
    def _validate_max_workers(cls, max_workers: int) -> int:
        """Validates and returns the number of workers. Sets to os.cpu_count() or 1 if max_workers is None,
        and limits max_workers to 61 on Windows.

        Args:
            max_workers: Desired number of workers. If None, defaults to os.cpu_count() or 1.

        Returns:
            A valid number of workers to use.

        Raises:
            ValueError: If max_workers is set and is not positive.
        """
        ...

    @classmethod
    def _validate_catalog(cls, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        """Validates the catalog."""
        ...

    @classmethod
    def _validate_nodes(cls, node: Node) -> None:
        """Validates the node."""
        ...

    @classmethod
    def _set_manager_datasets(cls, catalog: CatalogProtocol, pipeline: Pipeline) -> None:
        """Sets the manager datasets."""
        ...

    @classmethod
    def _get_required_workers_count(cls, pipeline: Pipeline) -> int:
        """Gets the required workers count."""
        ...

def _find_nodes_to_resume_from(pipeline: Pipeline, unfinished_nodes: Collection[Node], catalog: CatalogProtocol) -> Collection[str]:
    """Given a collection of unfinished nodes in a pipeline using
    a certain catalog, find the node names to pass to pipeline.from_nodes()
    to cover all unfinished nodes, including any additional nodes
    that should be re-run if their outputs are not persisted.

    Args:
        pipeline: the ``Pipeline`` to find starting nodes for.
        unfinished_nodes: collection of ``Node``s that have not finished yet
        catalog: an implemented instance of ``CatalogProtocol`` of the run.

    Returns:
        Set of node names to pass to pipeline.from_nodes() to continue
        the run.

    """
    ...

def _find_all_nodes_for_resumed_pipeline(pipeline: Pipeline, unfinished_nodes: Collection[Node], catalog: CatalogProtocol) -> Collection[Node]:
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
    ...

def _nodes_with_external_inputs(nodes_of_interest: Collection[Node]) -> Collection[Node]:
    """For given ``Node``s , find their subset which depends on
    external inputs of the ``Pipeline`` they constitute. External inputs
    are pipeline inputs not produced by other ``Node``s in the ``Pipeline``.

    Args:
        nodes_of_interest: the ``Node``s to analyze.

    Returns:
        A set of ``Node``s that depend on external inputs
        of nodes of interest.

    """
    ...

def _enumerate_non_persistent_inputs(node: Node, catalog: CatalogProtocol) -> Collection[str]:
    """Enumerate non-persistent input datasets of a ``Node``.

    Args:
        node: the ``Node`` to check the inputs of.
        catalog: an implemented instance of ``CatalogProtocol`` of the run.

    Returns:
        Set of names of non-persistent inputs of given ``Node``.

    """
    ...

def _enumerate_nodes_with_outputs(pipeline: Pipeline, outputs: Collection[str]) -> Collection[Node]:
    """For given outputs, returns a list containing nodes that
    generate them in the given ``Pipeline``.

    Args:
        pipeline: the ``Pipeline`` to search for nodes in.
        outputs: the dataset names to find source nodes for.

    Returns:
        A list of all ``Node``s that are producing ``outputs``.

    """
    ...

def _find_initial_node_group(pipeline: Pipeline, nodes: Collection[Node]) -> Collection[Node]:
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
    ...

def run_node(node: Node, catalog: CatalogProtocol, hook_manager: PluginManager, is_async: bool = False, session_id: str = None) -> Node:
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
        The node argument.

    """
    ...
