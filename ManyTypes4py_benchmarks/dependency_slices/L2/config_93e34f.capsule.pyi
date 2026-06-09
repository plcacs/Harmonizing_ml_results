from typing import Any

# === Internal dependency: chalice ===
# re-export: from chalice.app import __version__ as chalice_version
__version__ = __version__

# === Internal dependency: chalice.app ===
class Chalice(DecoratorAPI): ...
__version__: Any

# === Internal dependency: chalice.constants ===
DEFAULT_STAGE_NAME: str
DEFAULT_HANDLER_NAME: str