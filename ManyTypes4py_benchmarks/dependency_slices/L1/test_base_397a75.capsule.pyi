from typing import Any

# === Third-party dependency: click ===
# Used symbols: UsageError, option

# === Internal dependency: faust.cli ===
from .base import AppCommand
from .base import Command
from .base import call_command

# === Internal dependency: faust.cli.base ===
class argument: ...
class option: ...
def compat_option(*args, state_key, callback=..., expose_value=..., **kwargs): ...
def find_app(app, *, symbol_by_name=..., imp=..., attr_name=...): ...
class _Group(click.Group): ...
def _prepare_cli(ctx, app, quiet, debug, workdir, datadir, json, no_color, loop): ...
DEFAULT_LOGLEVEL = 'WARN'

# === Internal dependency: faust.types._env ===
def _getenv(name, *default, prefices=...): ...
CONSOLE_PORT = int(...)

# === Unresolved dependency: mode ===
# Used unresolved symbols: Worker

# === Third-party dependency: mode.utils.mocks ===
class Mock(unittest.mock.Mock):
    ...
class AsyncMock(unittest.mock.Mock):
    def __init__(self, *args: Any, name: str = ..., **kwargs: Any) -> None: ...
call: _Call
patch: Any

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises