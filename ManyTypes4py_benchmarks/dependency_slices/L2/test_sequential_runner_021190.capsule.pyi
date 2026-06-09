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
# re-export: from .sequential_runner import SequentialRunner

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Internal dependency: tests.runner.conftest ===
def source() -> Any: ...
def identity(arg) -> Any: ...
def sink(arg) -> Any: ...
def exception_fn(*args) -> Any: ...