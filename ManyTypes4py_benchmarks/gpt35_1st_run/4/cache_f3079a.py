from typing import Dict, Any, Union, Optional

LOG: logging.Logger
_CACHED_FILE_MINIMUM_SURVIVAL: int
_CACHED_FILE_MAXIMUM_SURVIVAL: int
_CACHED_SIZE_TRIGGER: int
_PICKLE_VERSION: int
_VERSION_TAG: str
_DEFAULT_CACHE_PATH: Path
_CACHE_CLEAR_THRESHOLD: int

def _get_default_cache_path() -> Path:
    ...

def _get_cache_clear_lock_path(cache_path: Optional[Path] = None) -> Path:
    ...

class _NodeCacheItem:
    def __init__(self, node: Any, lines: Any, change_time: Optional[float] = None) -> None:
        ...

def load_module(hashed_grammar: str, file_io: Any, cache_path: Optional[Path] = None) -> Any:
    ...

def _load_from_file_system(hashed_grammar: str, path: str, p_time: float, cache_path: Optional[Path] = None) -> Any:
    ...

def _set_cache_item(hashed_grammar: str, path: str, module_cache_item: _NodeCacheItem) -> None:
    ...

def try_to_save_module(hashed_grammar: str, file_io: Any, module: Any, lines: Any, pickling: bool = True, cache_path: Optional[Path] = None) -> None:
    ...

def _save_to_file_system(hashed_grammar: str, path: str, item: _NodeCacheItem, cache_path: Optional[Path] = None) -> None:
    ...

def clear_cache(cache_path: Optional[Path] = None) -> None:
    ...

def clear_inactive_cache(cache_path: Optional[Path] = None, inactivity_threshold: int = _CACHED_FILE_MAXIMUM_SURVIVAL) -> bool:
    ...

def _touch(path: str) -> bool:
    ...

def _remove_cache_and_update_lock(cache_path: Optional[Path] = None) -> bool:
    ...

def _get_hashed_path(hashed_grammar: str, path: str, cache_path: Optional[Path] = None) -> str:
    ...

def _get_cache_directory_path(cache_path: Optional[Path] = None) -> Path:
    ...
