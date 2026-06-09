from typing import Any

# === Internal dependency: kedro.framework.hooks ===
# re-export: from .manager import _create_hook_manager

# === Internal dependency: kedro.io ===
# re-export: from .core import AbstractDataset
# re-export: from .core import DatasetError
# re-export: from .data_catalog import DataCatalog
# re-export: from .lambda_dataset import LambdaDataset
# re-export: from .memory_dataset import MemoryDataset

# === Internal dependency: kedro.pipeline.modular_pipeline ===
def pipeline(pipe: Iterable[Node | Pipeline] | Pipeline, *, inputs: str | set[str] | dict[str, str] | None = ..., outputs: str | set[str] | dict[str, str] | None = ..., parameters: str | set[str] | dict[str, str] | None = ..., tags: str | Iterable[str] | None = ..., namespace: str | None = ...) -> Pipeline: ...

# === Internal dependency: kedro.runner ===
# re-export: from .parallel_runner import ParallelRunner

# === Internal dependency: kedro.runner.parallel_runner ===
class ParallelRunnerManager(SyncManager): ...
def _run_node_synchronization(node: Node, catalog: CatalogProtocol, is_async: bool = ..., session_id: str | None = ..., package_name: str | None = ..., logging_config: dict[str, Any] | None = ...) -> Node: ...
_MAX_WINDOWS_WORKERS: int

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Internal dependency: tests.runner.conftest ===
def source() -> Any: ...
def identity(arg) -> Any: ...
def sink(arg) -> Any: ...
def exception_fn(*args) -> Any: ...
def return_none(arg) -> Any: ...
def return_not_serialisable(arg) -> Any: ...