from typing import Any

# === Third-party dependency: dynaconf.validator ===
class Validator:
    def __init__(self, *names: str, must_exist: bool | None = ..., required: bool | None = ..., condition: Callable[[Any], bool] | None = ..., when: Validator | None = ..., env: str | Sequence[str] | None = ..., messages: dict[str, str] | None = ..., cast: Callable[[Any], Any] | None = ..., default: Any | Callable[[Any, Validator], Any] | None = ..., description: str | None = ..., apply_default_on_none: bool | None = ..., **operations: Any) -> None: ...

# === Internal dependency: kedro.framework.context.context ===
def _convert_paths_to_absolute_posix(project_path: Path, conf_dictionary: dict[str, Any]) -> dict[str, Any]: ...

# === Internal dependency: kedro.framework.hooks ===
# re-export: from .manager import _create_hook_manager
# re-export: from .markers import hook_impl

# === Internal dependency: kedro.framework.hooks.manager ===
def _register_hooks(hook_manager: PluginManager, hooks: Iterable[Any]) -> None: ...
def _register_hooks_entry_points(hook_manager: PluginManager, disabled_plugins: Iterable[str]) -> None: ...

# === Internal dependency: kedro.framework.project ===
class _ProjectSettings(LazySettings):
    ...
class _ProjectPipelines(MutableMapping): ...
settings: _ProjectSettings
pipelines: _ProjectPipelines

# === Internal dependency: kedro.framework.session ===
# re-export: from .session import KedroSession

# === Internal dependency: kedro.io ===
# re-export: from .data_catalog import DataCatalog
# re-export: from .memory_dataset import MemoryDataset

# === Internal dependency: kedro.runner ===
# re-export: from .parallel_runner import ParallelRunner

# === Internal dependency: kedro.runner.runner ===
def _run_node_async(node: Node, catalog: CatalogProtocol, hook_manager: PluginManager, session_id: str | None = ...) -> Node: ...

# === Third-party dependency: pandas ===
# Used symbols: DataFrame, testing

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Internal dependency: tests.framework.session.conftest ===
def _assert_hook_call_record_has_expected_parameters(call_record: logging.LogRecord, expected_parameters: list[str]) -> Any: ...
def _assert_pipeline_equal(p: Pipeline, q: Pipeline) -> Any: ...
def assert_exceptions_equal(e1: Exception, e2: Exception) -> Any: ...
def _mock_imported_settings_paths(mocker, mock_settings) -> Any: ...