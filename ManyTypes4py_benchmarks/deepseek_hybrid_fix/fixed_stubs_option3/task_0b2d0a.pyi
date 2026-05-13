from __future__ import annotations
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from kedro.framework.hooks.manager import _create_hook_manager, _register_hooks, _register_hooks_entry_points
from kedro.framework.project import settings

if TYPE_CHECKING:
    from pluggy import PluginManager
    from kedro.io import CatalogProtocol
    from kedro.pipeline.node import Node

class TaskError(Exception):
    pass

class Task:
    def __init__(
        self,
        node: Node,
        catalog: CatalogProtocol,
        is_async: bool,
        hook_manager: Optional[PluginManager] = None,
        session_id: Optional[str] = None,
        parallel: bool = False,
    ) -> None: ...

    def execute(self) -> Node: ...

    def __call__(self) -> Node: ...

    @staticmethod
    def _bootstrap_subprocess(
        package_name: str,
        logging_config: Optional[Dict[str, Any]] = None,
    ) -> None: ...

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
        session_id: Optional[str] = None,
    ) -> Node: ...

    def _run_node_async(
        self,
        node: Node,
        catalog: CatalogProtocol,
        hook_manager: PluginManager,
        session_id: Optional[str] = None,
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
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    @staticmethod
    def _call_node_run(
        node: Node,
        catalog: CatalogProtocol,
        inputs: Dict[str, Any],
        is_async: bool,
        hook_manager: PluginManager,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]: ...