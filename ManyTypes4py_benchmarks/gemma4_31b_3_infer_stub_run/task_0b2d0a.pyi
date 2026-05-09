from __future__ import annotations
from typing import Any, Optional, Union, Dict, Iterable, Iterator
from pluggy import PluginManager
from kedro.io import CatalogProtocol
from kedro.pipeline.node import Node

class TaskError(Exception):
    """``TaskError`` raised by ``Task``
    in case of failure of provided task arguments

    """
    ...

class Task:
    node: Node
    catalog: CatalogProtocol
    hook_manager: Optional[PluginManager]
    is_async: bool
    session_id: Optional[Any]
    parallel: bool

    def __init__(
        self,
        node: Node,
        catalog: CatalogProtocol,
        is_async: bool,
        hook_manager: Optional[PluginManager] = None,
        session_id: Optional[Any] = None,
        parallel: bool = False,
    ) -> None: ...

    def execute(self) -> Node: ...

    def __call__(self) -> Node: ...

    @staticmethod
    def _bootstrap_subprocess(package_name: str, logging_config: Optional[Dict[str, Any]] = None) -> None: ...

    @staticmethod
    def _run_node_synchronization(
        package_name: Optional[str] = None,
        logging_config: Optional[Dict[str, Any]] = None,
    ) -> PluginManager: ...

    def _run_node_sequential(
        self,
        node: Node,
        catalog: CatalogProtocol,
        hook_manager: PluginManager,
        session_id: Optional[Any] = None,
    ) -> Node: ...

    def _run_node_async(
        self,
        node: Node,
        catalog: CatalogProtocol,
        hook_manager: PluginManager,
        session_id: Optional[Any] = None,
    ) -> Node: ...

    @staticmethod
    def _synchronous_dataset_load(
        dataset_name: str,
        node: Node,
        catalog: CatalogProtocol,
        hook_manager: PluginManager,
    ) -> Any: ...

    @staticmethod
    def _collect_inputs_from_hook(
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        is_async: bool,
        hook_manager: PluginManager,
        session_id: Optional[Any] = None,
    ) -> Dict[str, Any]: ...

    @staticmethod
    def _call_node_run(
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        is_async: bool,
        hook_manager: PluginManager,
        session_id: Optional[Any] = None,
    ) -> Dict[str, Any]: ...