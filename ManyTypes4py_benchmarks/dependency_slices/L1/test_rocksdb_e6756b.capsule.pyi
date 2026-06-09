from typing import Any

# === Internal dependency: faust.exceptions ===
class ImproperlyConfigured(FaustError): ...

# === Internal dependency: faust.stores.rocksdb ===
class RocksDBOptions:
    def __init__(self, max_open_files=..., write_buffer_size=..., max_write_buffer_number=..., target_file_size_base=..., block_cache_size=..., block_cache_compressed_size=..., bloom_filter_size=..., **kwargs): ...
    def open(self, path, *, read_only=...): ...
    def as_options(self): ...
class Store(base.SerializedStore):
    def __init__(self, url, app, table, *, key_index_size=..., options=..., **kwargs): ...
_max_open_files = platforms.max_open_files(...)
DEFAULT_MAX_OPEN_FILES = _max_open_files
DEFAULT_WRITE_BUFFER_SIZE = 67108864
DEFAULT_MAX_WRITE_BUFFER_NUMBER = 3
DEFAULT_TARGET_FILE_SIZE_BASE = 67108864
DEFAULT_BLOCK_CACHE_SIZE = 2 * 1024 ** 3
DEFAULT_BLOCK_CACHE_COMPRESSED_SIZE = 500 * 1024 ** 2
DEFAULT_BLOOM_FILTER_SIZE = 3

# === Internal dependency: faust.types ===
from .tuples import TP

# === Internal dependency: faust.utils.platforms ===
def max_open_files(): ...

# === Third-party dependency: mode.utils.mocks ===
class Mock(unittest.mock.Mock):
    ...
class AsyncMock(unittest.mock.Mock):
    def __init__(self, *args: Any, name: str = ..., **kwargs: Any) -> None: ...
call: _Call
patch: Any

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises, yield_fixture

# === Third-party dependency: yarl ===
# Used symbols: URL