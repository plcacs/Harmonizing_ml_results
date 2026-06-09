# === Internal dependency: chalice ===
from chalice.app import __version__ as chalice_version
__version__ = chalice_version

# === Internal dependency: chalice.app ===
class Chalice(DecoratorAPI): ...

# === Internal dependency: chalice.constants ===
DEFAULT_STAGE_NAME = 'dev'
DEFAULT_HANDLER_NAME = 'api_handler'