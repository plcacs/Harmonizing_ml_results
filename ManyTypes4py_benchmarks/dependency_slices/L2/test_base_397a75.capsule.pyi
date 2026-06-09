from typing import Any

# === Third-party dependency: click ===
# Used symbols: UsageError, option

# === Internal dependency: faust.cli ===
# re-export: from .base import AppCommand
# re-export: from .base import Command
# re-export: from .base import call_command

# === Internal dependency: faust.cli.base ===
class argument: ...
class option: ...
DEFAULT_LOGLEVEL: str
def compat_option(*args: Any, state_key: str, callback: Callable[[click.Context, click.Parameter, Any], Any] = ..., expose_value: bool = ..., **kwargs: Any) -> Callable[[Any], click.Parameter]: ...
def find_app(app: str, *, symbol_by_name: Callable = ..., imp: Callable = ..., attr_name: str = ...) -> AppT: ...
class _Group(click.Group): ...
def _prepare_cli(ctx: click.Context, app: Union[AppT, str], quiet: bool, debug: bool, workdir: str, datadir: str, json: bool, no_color: bool, loop: str) -> None: ...

# === Internal dependency: faust.types._env ===
CONSOLE_PORT: int

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