# === Internal dependency: kedro.framework.hooks ===
from .manager import _create_hook_manager

# === Internal dependency: kedro.io ===
from .core import AbstractDataset
from .core import DatasetError
from .data_catalog import DataCatalog
from .lambda_dataset import LambdaDataset
from .memory_dataset import MemoryDataset

# === Internal dependency: kedro.pipeline.modular_pipeline ===
def pipeline(pipe, *, inputs=..., outputs=..., parameters=..., tags=..., namespace=...): ...

# === Internal dependency: kedro.runner ===
from .parallel_runner import ParallelRunner

# === Internal dependency: kedro.runner.parallel_runner ===
class ParallelRunnerManager(SyncManager): ...
def _run_node_synchronization(node, catalog, is_async=..., session_id=..., package_name=..., logging_config=...): ...
_MAX_WINDOWS_WORKERS = 61

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Internal dependency: tests.runner.conftest ===
def source(): ...
def identity(arg): ...
def sink(arg): ...
def exception_fn(*args): ...
def return_none(arg): ...
def return_not_serialisable(arg): ...