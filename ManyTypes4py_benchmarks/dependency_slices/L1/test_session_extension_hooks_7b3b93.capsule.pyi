from typing import Any

# === Third-party dependency: dynaconf.validator ===
class Validator:
    def __init__(self, *names: str, must_exist: bool | None = ..., required: bool | None = ..., condition: Callable[[Any], bool] | None = ..., when: Validator | None = ..., env: str | Sequence[str] | None = ..., messages: dict[str, str] | None = ..., cast: Callable[[Any], Any] | None = ..., default: Any | Callable[[Any, Validator], Any] | None = ..., description: str | None = ..., apply_default_on_none: bool | None = ..., **operations: Any) -> None: ...

# === Internal dependency: kedro.framework.context.context ===
def _convert_paths_to_absolute_posix(project_path, conf_dictionary): ...

# === Internal dependency: kedro.framework.hooks ===
from .manager import _create_hook_manager
from .markers import hook_impl

# === Internal dependency: kedro.framework.hooks.manager ===
def _register_hooks(hook_manager, hooks): ...
def _register_hooks_entry_points(hook_manager, disabled_plugins): ...

# === Internal dependency: kedro.framework.project ===
class _ProjectSettings(LazySettings):
    def __init__(self, *args, **kwargs): ...
class _ProjectPipelines(MutableMapping):
    def __init__(self): ...
settings = _ProjectSettings(...)
pipelines = _ProjectPipelines(...)

# === Internal dependency: kedro.framework.session ===
from .session import KedroSession

# === Internal dependency: kedro.io ===
from .data_catalog import DataCatalog
from .memory_dataset import MemoryDataset

# === Internal dependency: kedro.runner ===
from .parallel_runner import ParallelRunner

# === Internal dependency: kedro.runner.runner ===
def _run_node_async(node, catalog, hook_manager, session_id=...): ...

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, testing

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Internal dependency: tests.framework.session.conftest ===
def _assert_hook_call_record_has_expected_parameters(call_record, expected_parameters): ...
def _assert_pipeline_equal(p, q): ...
def assert_exceptions_equal(e1, e2): ...
def _mock_imported_settings_paths(mocker, mock_settings): ...