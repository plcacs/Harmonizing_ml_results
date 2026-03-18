```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar
from collections.abc import Iterable, Iterator
from concurrent.futures import Future

if TYPE_CHECKING:
    from pluggy import PluginManager
    from kedro.io import CatalogProtocol
    from kedro.pipeline.node import Node

class TaskError(Exception):
    """``TaskError`` raised by ``Task``
    in case of failure of provided task arguments

    """
    pass

class Task:
    node: Any = ...
    catalog: Any = ...
    hook_manager: Any = ...
    is_async: bool = ...
    session_id: str | None = ...
    parallel: bool = ...

    def __init__(
        self,
        node: Any,
        catalog: Any,
        is_async: bool,
        hook_manager: Any = ...,
        session_id: str | None = ...,
        parallel: bool = ...
    ) -> None: ...

    def execute(self) -> Any: ...

    def __call__(self) -> Any: ...

    @staticmethod
    def _bootstrap_subprocess(
        package_name: str,
        logging_config: dict[str, Any] | None = ...
    ) -> None: ...

    @staticmethod
    def _run_node_synchronization(
        package_name: str | None = ...,
        logging_config: dict[str, Any] | None = ...
    ) -> PluginManager: ...

    def _run_node_sequential(
        self,
        node: Node,
        catalog: CatalogProtocol,
        hook_manager: PluginManager,
        session_id: str | None = ...
    ) -> Node: ...

    def _run_node_async(
        self,
        node: Node,
        catalog: CatalogProtocol,
        hook_manager: PluginManager,
        session_id: str | None = ...
    ) -> Node: ...

    @staticmethod
    def _synchronous_dataset_load(
        dataset_name: str,
        node: Node,
        catalog: CatalogProtocol,
        hook_manager: PluginManager
    ) -> Any: ...

    @staticmethod
    def _collect_inputs_from_hook(
        node: Node,
        catalog: CatalogProtocol,
        inputs: dict[str, Any],
        is_async: bool,
        hook_manager: PluginManager,
        session_id: str | None = ...
    ) -> dict[str, Any]: ...

    @staticmethod
    def _call_node_run(
        node: Node,
        catalog: CatalogProtocol,
        inputs: dict[str, Any],
        is_async: bool,
        hook_manager: PluginManager,
        session_id: str | None = ...
    ) -> dict[str, Any]: ...
```