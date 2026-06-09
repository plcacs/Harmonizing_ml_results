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
from .sequential_runner import SequentialRunner

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Internal dependency: tests.runner.conftest ===
def source(): ...
def identity(arg): ...
def sink(arg): ...
def exception_fn(*args): ...