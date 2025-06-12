import abc
import json
import os
import struct
import sys
import tempfile
import warnings
import weakref
from collections.abc import Iterable, Set, Mapping, MutableSet
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha384
from os import PathLike, getenv
from pathlib import Path, PurePath
from queue import Queue
from threading import Thread
from typing import (
    TYPE_CHECKING, Any, Literal, Optional, Union, cast, Dict, Iterator, Tuple,
    List, ByteString, TypeVar, Generic, final, overload, Callable, Sequence,
    AbstractSet, FrozenSet, MutableMapping, Type, NoReturn, BinaryIO, IO,
    TextIO, AnyStr, ContextManager, Collection, Generator, Iterable as IterableT
)
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
from hypothesis.configuration import storage_directory
from hypothesis.errors import HypothesisException, HypothesisWarning
from hypothesis.internal.conjecture.choice import ChoiceT
from hypothesis.utils.conventions import UniqueIdentifier, not_set

__all__ = [
    'DirectoryBasedExampleDatabase', 'ExampleDatabase', 'GitHubArtifactDatabase',
    'InMemoryExampleDatabase', 'MultiplexedDatabase', 'ReadOnlyDatabase'
]

if TYPE_CHECKING:
    from typing import TypeAlias
    StrPathT: TypeAlias = Union[str, PathLike[str]]
else:
    StrPathT = Union[str, PathLike[str]]

T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')

def _usable_dir(path: Union[str, PathLike[str]]) -> bool:
    path = Path(path)
    try:
        while not path.exists():
            path = path.parent
        return path.is_dir() and os.access(path, os.R_OK | os.W_OK | os.X_OK)
    except PermissionError:
        return False

def _db_for_path(path: Union[UniqueIdentifier, StrPathT, None] = not_set) -> 'ExampleDatabase':
    if path is not_set:
        if os.getenv('HYPOTHESIS_DATABASE_FILE') is not None:
            raise HypothesisException(
                'The $HYPOTHESIS_DATABASE_FILE environment variable no longer has any effect. '
                'Configure your database location via a settings profile instead.\n'
                'https://hypothesis.readthedocs.io/en/latest/settings.html#settings-profiles'
            )
        path = storage_directory('examples', intent_to_write=False)
        if not _usable_dir(path):
            warnings.warn(
                f'The database setting is not configured, and the default location is unusable - '
                f'falling back to an in-memory database for this session.  path={path!r}',
                HypothesisWarning, stacklevel=3
            )
            return InMemoryExampleDatabase()
    if path in (None, ':memory:'):
        return InMemoryExampleDatabase()
    path = cast(StrPathT, path)
    return DirectoryBasedExampleDatabase(path)

class _EDMeta(abc.ABCMeta):
    def __call__(self, *args: Any, **kwargs: Any) -> 'ExampleDatabase':
        if self is ExampleDatabase:
            return _db_for_path(*args, **kwargs)
        return super().__call__(*args, **kwargs)

if 'sphinx' in sys.modules:
    try:
        from sphinx.ext.autodoc import _METACLASS_CALL_BLACKLIST
        _METACLASS_CALL_BLACKLIST.append('hypothesis.database._EDMeta.__call__')
    except Exception:
        pass

class ExampleDatabase(metaclass=_EDMeta):
    @abc.abstractmethod
    def save(self, key: bytes, value: bytes) -> None: ...
    
    @abc.abstractmethod
    def fetch(self, key: bytes) -> Iterator[bytes]: ...
    
    @abc.abstractmethod
    def delete(self, key: bytes, value: bytes) -> None: ...
    
    def move(self, src: bytes, dest: bytes, value: bytes) -> None: ...

class InMemoryExampleDatabase(ExampleDatabase):
    def __init__(self) -> None:
        self.data: Dict[bytes, Set[bytes]] = {}
    
    def __repr__(self) -> str:
        return f'InMemoryExampleDatabase({self.data!r})'
    
    def fetch(self, key: bytes) -> Iterator[bytes]:
        yield from self.data.get(key, set())
    
    def save(self, key: bytes, value: bytes) -> None:
        self.data.setdefault(key, set()).add(bytes(value))
    
    def delete(self, key: bytes, value: bytes) -> None:
        self.data.get(key, set()).discard(bytes(value))

def _hash(key: bytes) -> str:
    return sha384(key).hexdigest()[:16]

class DirectoryBasedExampleDatabase(ExampleDatabase):
    def __init__(self, path: StrPathT) -> None:
        self.path: Path = Path(path)
        self.keypaths: Dict[bytes, Path] = {}
    
    def __repr__(self) -> str:
        return f'DirectoryBasedExampleDatabase({self.path!r})'
    
    def _key_path(self, key: bytes) -> Path:
        try:
            return self.keypaths[key]
        except KeyError:
            pass
        self.keypaths[key] = self.path / _hash(key)
        return self.keypaths[key]
    
    def _value_path(self, key: bytes, value: bytes) -> Path:
        return self._key_path(key) / _hash(value)
    
    def fetch(self, key: bytes) -> Iterator[bytes]:
        kp = self._key_path(key)
        if not kp.is_dir():
            return
        for path in os.listdir(kp):
            try:
                yield (kp / path).read_bytes()
            except OSError:
                pass
    
    def save(self, key: bytes, value: bytes) -> None: ...
    
    def move(self, src: bytes, dest: bytes, value: bytes) -> None: ...
    
    def delete(self, key: bytes, value: bytes) -> None: ...

class ReadOnlyDatabase(ExampleDatabase):
    def __init__(self, db: ExampleDatabase) -> None:
        self._wrapped: ExampleDatabase = db
    
    def __repr__(self) -> str:
        return f'ReadOnlyDatabase({self._wrapped!r})'
    
    def fetch(self, key: bytes) -> Iterator[bytes]:
        yield from self._wrapped.fetch(key)
    
    def save(self, key: bytes, value: bytes) -> None: ...
    
    def delete(self, key: bytes, value: bytes) -> None: ...

class MultiplexedDatabase(ExampleDatabase):
    def __init__(self, *dbs: ExampleDatabase) -> None:
        self._wrapped: Tuple[ExampleDatabase, ...] = dbs
    
    def __repr__(self) -> str:
        return f'MultiplexedDatabase({", ".join(map(repr, self._wrapped))})'
    
    def fetch(self, key: bytes) -> Iterator[bytes]:
        seen: Set[bytes] = set()
        for db in self._wrapped:
            for value in db.fetch(key):
                if value not in seen:
                    yield value
                    seen.add(value)
    
    def save(self, key: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.save(key, value)
    
    def delete(self, key: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.delete(key, value)
    
    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.move(src, dest, value)

class GitHubArtifactDatabase(ExampleDatabase):
    def __init__(
        self,
        owner: str,
        repo: str,
        artifact_name: str = 'hypothesis-example-db',
        cache_timeout: timedelta = timedelta(days=1),
        path: Optional[StrPathT] = None
    ) -> None: ...
    
    def __repr__(self) -> str: ...
    
    def fetch(self, key: bytes) -> Iterator[bytes]: ...
    
    def save(self, key: bytes, value: bytes) -> NoReturn: ...
    
    def move(self, src: bytes, dest: bytes, value: bytes) -> NoReturn: ...
    
    def delete(self, key: bytes, value: bytes) -> NoReturn: ...

class BackgroundWriteDatabase(ExampleDatabase):
    def __init__(self, db: ExampleDatabase) -> None: ...
    
    def __repr__(self) -> str: ...
    
    def fetch(self, key: bytes) -> Iterator[bytes]: ...
    
    def save(self, key: bytes, value: bytes) -> None: ...
    
    def delete(self, key: bytes, value: bytes) -> None: ...
    
    def move(self, src: bytes, dest: bytes, value: bytes) -> None: ...

def _pack_uleb128(value: int) -> bytes: ...

def _unpack_uleb128(buffer: bytes) -> Tuple[int, int]: ...

def choices_to_bytes(ir: Sequence[Union[bool, float, int, bytes, str]]) -> bytes: ...

def _choices_from_bytes(buffer: bytes) -> Tuple[Union[bool, float, int, bytes, str], ...]: ...

def choices_from_bytes(buffer: bytes) -> Optional[Tuple[Union[bool, float, int, bytes, str], ...]]: ...
