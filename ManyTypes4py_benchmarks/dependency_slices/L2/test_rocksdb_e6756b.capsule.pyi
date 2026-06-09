from typing import Any

# === Internal dependency: faust.exceptions ===
class ImproperlyConfigured(FaustError): ...

# === Internal dependency: faust.stores.rocksdb ===
class RocksDBOptions:
    def __init__(self, max_open_files: int = ..., write_buffer_size: int = ..., max_write_buffer_number: int = ..., target_file_size_base: int = ..., block_cache_size: int = ..., block_cache_compressed_size: int = ..., bloom_filter_size: int = ..., **kwargs: Any) -> None: ...
    def open(self, path: Path, *, read_only: bool = ...) -> DB: ...
    def as_options(self) -> Options: ...
class Store(base.SerializedStore):
    def __init__(self, url: Union[str, URL], app: AppT, table: CollectionT, *, key_index_size: int = ..., options: Mapping[str, Any] = ..., **kwargs: Any) -> None: ...
_max_open_files: max_open_files
DEFAULT_MAX_OPEN_FILES = _max_open_files
DEFAULT_WRITE_BUFFER_SIZE: int
DEFAULT_MAX_WRITE_BUFFER_NUMBER: int
DEFAULT_TARGET_FILE_SIZE_BASE: int
DEFAULT_BLOCK_CACHE_SIZE: Any
DEFAULT_BLOCK_CACHE_COMPRESSED_SIZE: Any
DEFAULT_BLOOM_FILTER_SIZE: int

# === Internal dependency: faust.types ===
# re-export: from .tuples import TP

# === Internal dependency: faust.utils.platforms ===
def max_open_files() -> Optional[int]: ...

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