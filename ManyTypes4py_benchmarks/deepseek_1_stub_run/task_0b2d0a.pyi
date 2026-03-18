```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any
from collections.abc import Iterator
from concurrent.futures import Future

if TYPE_CHECKING:
    from pluggy import PluginManager
    from kedro.io import CatalogProtocol
    from kedro.pipeline.node import Node

class TaskError(Exception):
    pass

class Task:
    node: Any
    catalog: Any
    hook_manager: Any
    is_async: bool
    session_id: Any
    parallel: bool
    
    def __init__(
        self,
        node: Any,
        catalog: Any,
        is_async: bool,
        hook_manager: Any = ...,
        session_id: Any = ...,
        parallel: bool = ...
    ) -> None: ...
    
    def execute(self) -> Any: ...
    
    def __call__(self) -> Any: ...
    
    @staticmethod
    def _bootstrap_subprocess(package_name: Any, logging_config: Any = ...) -> None: ...
    
    @staticmethod
    def _run_node_synchronization(package_name: Any = ..., logging_config: Any = ...) -> Any: ...
    
    def _run_node_sequential(
        self,
        node: Any,
        catalog: Any,
        hook_manager: Any,
        session_id: Any = ...
    ) -> Any: ...
    
    def _run_node_async(
        self,
        node: Any,
        catalog: Any,
        hook_manager: Any,
        session_id: Any = ...
    ) -> Any: ...
    
    @staticmethod
    def _synchronous_dataset_load(
        dataset_name: Any,
        node: Any,
        catalog: Any,
        hook_manager: Any
    ) -> Any: ...
    
    @staticmethod
    def _collect_inputs_from_hook(
        node: Any,
        catalog: Any,
        inputs: Any,
        is_async: bool,
        hook_manager: Any,
        session_id: Any = ...
    ) -> dict[Any, Any]: ...
    
    @staticmethod
    def _call_node_run(
        node: Any,
        catalog: Any,
        inputs: Any,
        is_async: bool,
        hook_manager: Any,
        session_id: Any = ...
    ) -> dict[Any, Any]: ...
```