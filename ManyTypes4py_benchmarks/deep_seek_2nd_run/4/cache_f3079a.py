import time
import os
import sys
import hashlib
import gc
import shutil
import platform
import logging
import warnings
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, cast
LOG: logging.Logger = logging.getLogger(__name__)
_CACHED_FILE_MINIMUM_SURVIVAL: int = 60 * 10
'\nCached files should survive at least a few minutes.\n'
_CACHED_FILE_MAXIMUM_SURVIVAL: int = 60 * 60 * 24 * 30
'\nMaximum time for a cached file to survive if it is not\naccessed within.\n'
_CACHED_SIZE_TRIGGER: int = 600
"\nThis setting limits the amount of cached files. It's basically a way to start\ngarbage collection.\n\nThe reasoning for this limit being as big as it is, is the following:\n\nNumpy, Pandas, Matplotlib and Tensorflow together use about 500 files. This\nmakes Jedi use ~500mb of memory. Since we might want a bit more than those few\nlibraries, we just increase it a bit.\n"
_PICKLE_VERSION: int = 33
'\nVersion number (integer) for file system cache.\n\nIncrement this number when there are any incompatible changes in\nthe parser tree classes.  For example, the following changes\nare regarded as incompatible.\n\n- A class name is changed.\n- A class is moved to another module.\n- A __slot__ of a class is changed.\n'
_VERSION_TAG: str = '%s-%s%s-%s' % (platform.python_implementation(), sys.version_info[0], sys.version_info[1], _PICKLE_VERSION)
"\nShort name for distinguish Python implementations and versions.\n\nIt's a bit similar to `sys.implementation.cache_tag`.\nSee: http://docs.python.org/3/library/sys.html#sys.implementation\n"

def _get_default_cache_path() -> Path:
    if platform.system().lower() == 'windows':
        dir_ = Path(os.getenv('LOCALAPPDATA') or '~', 'Parso', 'Parso')
    elif platform.system().lower() == 'darwin':
        dir_ = Path('~', 'Library', 'Caches', 'Parso')
    else:
        dir_ = Path(os.getenv('XDG_CACHE_HOME') or '~/.cache', 'parso')
    return dir_.expanduser()
_default_cache_path: Path = _get_default_cache_path()
'\nThe path where the cache is stored.\n\nOn Linux, this defaults to ``~/.cache/parso/``, on OS X to\n``~/Library/Caches/Parso/`` and on Windows to ``%LOCALAPPDATA%\\Parso\\Parso\\``.\nOn Linux, if environment variable ``$XDG_CACHE_HOME`` is set,\n``$XDG_CACHE_HOME/parso`` is used instead of the default one.\n'
_CACHE_CLEAR_THRESHOLD: int = 60 * 60 * 24

def _get_cache_clear_lock_path(cache_path: Optional[Path] = None) -> Path:
    """
    The path where the cache lock is stored.

    Cache lock will prevent continous cache clearing and only allow garbage
    collection once a day (can be configured in _CACHE_CLEAR_THRESHOLD).
    """
    cache_path = cache_path or _default_cache_path
    return cache_path.joinpath('PARSO-CACHE-LOCK')
parser_cache: Dict[str, Dict[str, _NodeCacheItem]] = {}

class _NodeCacheItem:

    def __init__(self, node: Any, lines: Any, change_time: Optional[float] = None) -> None:
        self.node: Any = node
        self.lines: Any = lines
        if change_time is None:
            change_time = time.time()
        self.change_time: float = change_time
        self.last_used: float = change_time

def load_module(hashed_grammar: str, file_io: Any, cache_path: Optional[Path] = None) -> Optional[Any]:
    """
    Returns a module or None, if it fails.
    """
    p_time = file_io.get_last_modified()
    if p_time is None:
        return None
    try:
        module_cache_item = parser_cache[hashed_grammar][file_io.path]
        if p_time <= module_cache_item.change_time:
            module_cache_item.last_used = time.time()
            return module_cache_item.node
    except KeyError:
        return _load_from_file_system(hashed_grammar, file_io.path, p_time, cache_path=cache_path)

def _load_from_file_system(hashed_grammar: str, path: str, p_time: float, cache_path: Optional[Path] = None) -> Optional[Any]:
    cache_path = _get_hashed_path(hashed_grammar, path, cache_path=cache_path)
    try:
        if p_time > os.path.getmtime(cache_path):
            return None
        with open(cache_path, 'rb') as f:
            gc.disable()
            try:
                module_cache_item = pickle.load(f)
            finally:
                gc.enable()
    except FileNotFoundError:
        return None
    else:
        _set_cache_item(hashed_grammar, path, module_cache_item)
        LOG.debug('pickle loaded: %s', path)
        return module_cache_item.node

def _set_cache_item(hashed_grammar: str, path: str, module_cache_item: _NodeCacheItem) -> None:
    if sum((len(v) for v in parser_cache.values())) >= _CACHED_SIZE_TRIGGER:
        cutoff_time = time.time() - _CACHED_FILE_MINIMUM_SURVIVAL
        for key, path_to_item_map in parser_cache.items():
            parser_cache[key] = {path: node_item for path, node_item in path_to_item_map.items() if node_item.last_used > cutoff_time}
    parser_cache.setdefault(hashed_grammar, {})[path] = module_cache_item

def try_to_save_module(hashed_grammar: str, file_io: Any, module: Any, lines: Any, pickling: bool = True, cache_path: Optional[Path] = None) -> None:
    path = file_io.path
    try:
        p_time = None if path is None else file_io.get_last_modified()
    except OSError:
        p_time = None
        pickling = False
    item = _NodeCacheItem(module, lines, p_time)
    _set_cache_item(hashed_grammar, path, item)
    if pickling and path is not None:
        try:
            _save_to_file_system(hashed_grammar, path, item, cache_path=cache_path)
        except PermissionError:
            warnings.warn('Tried to save a file to %s, but got permission denied.' % path, Warning)
        else:
            _remove_cache_and_update_lock(cache_path=cache_path)

def _save_to_file_system(hashed_grammar: str, path: str, item: _NodeCacheItem, cache_path: Optional[Path] = None) -> None:
    with open(_get_hashed_path(hashed_grammar, path, cache_path=cache_path), 'wb') as f:
        pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)

def clear_cache(cache_path: Optional[Path] = None) -> None:
    if cache_path is None:
        cache_path = _default_cache_path
    shutil.rmtree(cache_path)
    parser_cache.clear()

def clear_inactive_cache(cache_path: Optional[Path] = None, inactivity_threshold: int = _CACHED_FILE_MAXIMUM_SURVIVAL) -> bool:
    if cache_path is None:
        cache_path = _default_cache_path
    if not cache_path.exists():
        return False
    for dirname in os.listdir(cache_path):
        version_path = cache_path.joinpath(dirname)
        if not version_path.is_dir():
            continue
        for file in os.scandir(version_path):
            if file.stat().st_atime + _CACHED_FILE_MAXIMUM_SURVIVAL <= time.time():
                try:
                    os.remove(file.path)
                except OSError:
                    continue
    else:
        return True

def _touch(path: Union[str, Path]) -> bool:
    try:
        os.utime(path, None)
    except FileNotFoundError:
        try:
            file = open(path, 'a')
            file.close()
        except (OSError, IOError):
            return False
    return True

def _remove_cache_and_update_lock(cache_path: Optional[Path] = None) -> bool:
    lock_path = _get_cache_clear_lock_path(cache_path=cache_path)
    try:
        clear_lock_time = os.path.getmtime(lock_path)
    except FileNotFoundError:
        clear_lock_time = None
    if clear_lock_time is None or clear_lock_time + _CACHE_CLEAR_THRESHOLD <= time.time():
        if not _touch(lock_path):
            return False
        clear_inactive_cache(cache_path=cache_path)
    return True

def _get_hashed_path(hashed_grammar: str, path: str, cache_path: Optional[Path] = None) -> str:
    directory = _get_cache_directory_path(cache_path=cache_path)
    file_hash = hashlib.sha256(str(path).encode('utf-8')).hexdigest()
    return os.path.join(directory, '%s-%s.pkl' % (hashed_grammar, file_hash))

def _get_cache_directory_path(cache_path: Optional[Path] = None) -> str:
    if cache_path is None:
        cache_path = _default_cache_path
    directory = cache_path.joinpath(_VERSION_TAG)
    if not directory.exists():
        os.makedirs(directory)
    return str(directory)
