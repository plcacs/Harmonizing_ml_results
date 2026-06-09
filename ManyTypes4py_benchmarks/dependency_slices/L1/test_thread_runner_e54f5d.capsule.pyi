# === Internal dependency: kedro.framework.hooks ===
from .manager import _create_hook_manager

# === Internal dependency: kedro.io ===
from .core import AbstractDataset
from .core import DatasetError
from .data_catalog import DataCatalog
from .memory_dataset import MemoryDataset

# === Internal dependency: kedro.pipeline.modular_pipeline ===
def pipeline(pipe, *, inputs=..., outputs=..., parameters=..., tags=..., namespace=...): ...

# === Internal dependency: kedro.runner ===
from .thread_runner import ThreadRunner

# === Third-party dependency: pytest ===
# Used symbols: mark, raises, warns

# === Internal dependency: tests.runner.conftest ===
def source(): ...
def identity(arg): ...
def sink(arg): ...
def exception_fn(*args): ...
def return_none(arg): ...