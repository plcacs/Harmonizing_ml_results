from typing import Any, Optional, Dict
from pluggy import PluginManager
from kedro.io import CatalogProtocol
from kedro.pipeline.node import Node

class TaskError(Exception):
    ...

class Task:
    node: Node
    catalog: CatalogProtocol
    hook_manager: Optional[PluginManager]
    is_async: bool
    session_id: Optional[str]
    parallel: bool

    def __init__(
        self,
        node: Node,
        catalog: CatalogProtocol,
        is_async: bool,
        hook_manager: Optional[PluginManager] = ...,
        session_id: Optional[str] = ...,
        parallel: bool = ...,
    ) -> None: ...
    def execute(self) -> Node: ...
    def __call__(self) -> Node: ...
    @staticmethod
    def _bootstrap_subprocess(package_name: Optional[str], logging_config: Optional[dict[str, Any]] = ...) -> None: ...
    @staticmethod
    def _run_node_synchronization(package_name: Optional[str] = ..., logging_config: Optional[dict[str, Any]] = ...) -> PluginManager: ...
    def _run_node_sequential(self, node: Node, catalog: CatalogProtocol, hook_manager: PluginManager, session_id: Optional[str] = ...) -> Node: ...
    def _run_node_async(self, node: Node, catalog: CatalogProtocol, hook_manager: PluginManager, session_id: Optional[str] = ...) -> Node: ...
    @staticmethod
    def _synchronous_dataset_load(dataset_name: str, node: Node, catalog: CatalogProtocol, hook_manager: PluginManager) -> Any: ...
    @staticmethod
    def _collect_inputs_from_hook(
        node: Node,
        catalog: CatalogProtocol,
        inputs: dict[str, Any],
        is_async: bool,
        hook_manager: PluginManager,
        session_id: Optional[str] = ...,
    ) -> dict[str, Any]: ...
    @staticmethod
    def _call_node_run(
        node: Node,
        catalog: CatalogProtocol,
        inputs: dict[str, Any],
        is_async: bool,
        hook_manager: PluginManager,
        session_id: Optional[str] = ...,
    ) -> dict[str, Any]: ...