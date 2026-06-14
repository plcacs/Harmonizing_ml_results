from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pluggy import PluginManager
    from kedro.io import CatalogProtocol
    from kedro.pipeline.node import Node

class TaskError(Exception): ...

class Task:
    node: Node
    catalog: CatalogProtocol
    hook_manager: PluginManager | None
    is_async: bool
    session_id: str | None
    parallel: bool

    def __init__(
        self,
        node: Node,
        catalog: CatalogProtocol,
        is_async: bool,
        hook_manager: PluginManager | None = None,
        session_id: str | None = None,
        parallel: bool = False,
    ) -> None: ...

    def execute(self) -> Node: ...

    def __call__(self) -> Node: ...

    @staticmethod
    def _bootstrap_subprocess(
        package_name: str, logging_config: dict[str, Any] | None = None
    ) -> None: ...

    @staticmethod
    def _run_node_synchronization(
        package_name: str | None = None,
        logging_config: dict[str, Any] | None = None,
    ) -> PluginManager: ...

    def _run_node_sequential(
        self,
        node: Node,
        catalog: CatalogProtocol,
        hook_manager: PluginManager,
        session_id: str | None = None,
    ) -> Node: ...

    def _run_node_async(
        self,
        node: Node,
        catalog: CatalogProtocol,
        hook_manager: PluginManager,
        session_id: str | None = None,
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
        inputs: dict[str, Any],
        is_async: bool,
        hook_manager: PluginManager,
        session_id: str | None = None,
    ) -> dict[str, Any]: ...

    @staticmethod
    def _call_node_run(
        node: Node,
        catalog: CatalogProtocol,
        inputs: dict[str, Any],
        is_async: bool,
        hook_manager: PluginManager,
        session_id: str | None = None,
    ) -> dict[str, Any]: ...