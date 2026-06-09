from typing import Any

# === Third-party dependency: click ===
# Used symbols: get_current_context

# === Internal dependency: kedro ===
__version__: str

# === Internal dependency: kedro.config ===
# re-export: from .abstract_config import AbstractConfigLoader

# === Internal dependency: kedro.framework.context ===
# re-export: from .context import KedroContext

# === Internal dependency: kedro.framework.hooks ===
# re-export: from .manager import _create_hook_manager

# === Internal dependency: kedro.framework.hooks.manager ===
def _register_hooks(hook_manager: PluginManager, hooks: Iterable[Any]) -> None: ...
def _register_hooks_entry_points(hook_manager: PluginManager, disabled_plugins: Iterable[str]) -> None: ...

# === Internal dependency: kedro.framework.project ===
def validate_settings() -> None: ...
settings: _ProjectSettings
pipelines: _ProjectPipelines

# === Internal dependency: kedro.framework.session.store ===
class BaseSessionStore(UserDict): ...

# === Internal dependency: kedro.io.core ===
def generate_timestamp() -> str: ...

# === Internal dependency: kedro.runner ===
# re-export: from .runner import AbstractRunner
# re-export: from .sequential_runner import SequentialRunner
# re-export: from .thread_runner import ThreadRunner

# === Internal dependency: kedro.utils ===
def _find_kedro_project(current_dir: Path) -> Any: ...