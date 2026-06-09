# === Third-party dependency: click ===
# Used symbols: get_current_context

# === Internal dependency: kedro ===
__version__ = '0.19.8'

# === Internal dependency: kedro.config ===
from .abstract_config import AbstractConfigLoader

# === Internal dependency: kedro.framework.context ===
from .context import KedroContext

# === Internal dependency: kedro.framework.hooks ===
from .manager import _create_hook_manager

# === Internal dependency: kedro.framework.hooks.manager ===
def _register_hooks(hook_manager, hooks): ...
def _register_hooks_entry_points(hook_manager, disabled_plugins): ...

# === Internal dependency: kedro.framework.project ===
class _ProjectSettings(LazySettings):
    def __init__(self, *args, **kwargs): ...
class _ProjectPipelines(MutableMapping):
    def __init__(self): ...
def validate_settings(): ...
settings = _ProjectSettings(...)
pipelines = _ProjectPipelines(...)

# === Internal dependency: kedro.framework.session.store ===
class BaseSessionStore(UserDict): ...

# === Internal dependency: kedro.io.core ===
def generate_timestamp(): ...

# === Internal dependency: kedro.runner ===
from .runner import AbstractRunner
from .sequential_runner import SequentialRunner
from .thread_runner import ThreadRunner

# === Internal dependency: kedro.utils ===
def _find_kedro_project(current_dir): ...