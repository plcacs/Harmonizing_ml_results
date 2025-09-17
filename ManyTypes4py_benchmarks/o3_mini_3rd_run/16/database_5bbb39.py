import abc
import json
import os
import struct
import sys
import tempfile
import warnings
import weakref
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha384
from os import PathLike, getenv
from pathlib import Path, PurePath
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Any, Iterable as IterableType, Literal, Optional, Union, cast

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from hypothesis.configuration import storage_directory
from hypothesis.errors import HypothesisException, HypothesisWarning
from hypothesis.internal.conjecture.choice import ChoiceT
from hypothesis.utils.conventions import UniqueIdentifier, not_set

__all__ = [
    'DirectoryBasedExampleDatabase',
    'ExampleDatabase',
    'GitHubArtifactDatabase',
    'InMemoryExampleDatabase',
    'MultiplexedDatabase',
    'ReadOnlyDatabase',
]

if TYPE_CHECKING:
    from typing import TypeAlias

StrPathT = Union[str, PathLike[str]]


def _usable_dir(path: StrPathT) -> bool:
    """
    Returns True if the desired path can be used as database path because
    either the directory exists and can be used, or its root directory can
    be used and we can make the directory as needed.
    """
    path_obj = Path(path)
    try:
        while not path_obj.exists():
            path_obj = path_obj.parent
        return path_obj.is_dir() and os.access(path_obj, os.R_OK | os.W_OK | os.X_OK)
    except PermissionError:
        return False


def _db_for_path(path: Any = not_set) -> "ExampleDatabase":
    if path is not_set:
        if os.getenv('HYPOTHESIS_DATABASE_FILE') is not None:
            raise HypothesisException(
                'The $HYPOTHESIS_DATABASE_FILE environment variable no longer has any effect.  '
                'Configure your database location via a settings profile instead.\n'
                'https://hypothesis.readthedocs.io/en/latest/settings.html#settings-profiles'
            )
        path = storage_directory('examples', intent_to_write=False)
        if not _usable_dir(path):
            warnings.warn(
                f'The database setting is not configured, and the default location is unusable - '
                f'falling back to an in-memory database for this session.  path={path!r}',
                HypothesisWarning,
                stacklevel=3,
            )
            return InMemoryExampleDatabase()
    if path in (None, ':memory:'):
        return InMemoryExampleDatabase()
    path_cast = cast(StrPathT, path)
    return DirectoryBasedExampleDatabase(path_cast)


class _EDMeta(abc.ABCMeta):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
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
    """An abstract base class for storing examples in Hypothesis' internal format.

    An ExampleDatabase maps each ``bytes`` key to many distinct ``bytes``
    values, like a ``Mapping[bytes, AbstractSet[bytes]]``.
    """

    @abc.abstractmethod
    def save(self, key: bytes, value: bytes) -> None:
        """Save ``value`` under ``key``.

        If this value is already present for this key, silently do nothing.
        """
        raise NotImplementedError(f'{type(self).__name__}.save')

    @abc.abstractmethod
    def fetch(self, key: bytes) -> IterableType[bytes]:
        """Return an iterable over all values matching this key."""
        raise NotImplementedError(f'{type(self).__name__}.fetch')

    @abc.abstractmethod
    def delete(self, key: bytes, value: bytes) -> None:
        """Remove this value from this key.

        If this value is not present, silently do nothing.
        """
        raise NotImplementedError(f'{type(self).__name__}.delete')

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        """Move ``value`` from key ``src`` to key ``dest``. Equivalent to
        ``delete(src, value)`` followed by ``save(src, value)``, but may
        have a more efficient implementation.

        Note that ``value`` will be inserted at ``dest`` regardless of whether
        it is currently present at ``src``.
        """
        if src == dest:
            self.save(src, value)
            return
        self.delete(src, value)
        self.save(dest, value)


class InMemoryExampleDatabase(ExampleDatabase):
    """A non-persistent example database, implemented in terms of a dict of sets.

    This can be useful if you call a test function several times in a single
    session, or for testing other database implementations, but because it
    does not persist between runs we do not recommend it for general use.
    """

    def __init__(self) -> None:
        self.data: dict[bytes, set[bytes]] = {}

    def __repr__(self) -> str:
        return f'InMemoryExampleDatabase({self.data!r})'

    def fetch(self, key: bytes) -> IterableType[bytes]:
        yield from self.data.get(key, set())

    def save(self, key: bytes, value: bytes) -> None:
        self.data.setdefault(key, set()).add(bytes(value))

    def delete(self, key: bytes, value: bytes) -> None:
        self.data.get(key, set()).discard(bytes(value))


def _hash(key: bytes) -> str:
    return sha384(key).hexdigest()[:16]


class DirectoryBasedExampleDatabase(ExampleDatabase):
    """Use a directory to store Hypothesis examples as files.

    Each test corresponds to a directory, and each example to a file within that
    directory.
    """

    def __init__(self, path: StrPathT) -> None:
        self.path: Path = Path(path)
        self.keypaths: dict[bytes, Path] = {}

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

    def fetch(self, key: bytes) -> IterableType[bytes]:
        kp = self._key_path(key)
        if not kp.is_dir():
            return
        for path in os.listdir(kp):
            try:
                yield (kp / path).read_bytes()
            except OSError:
                pass

    def save(self, key: bytes, value: bytes) -> None:
        try:
            self._key_path(key).mkdir(exist_ok=True, parents=True)
            path = self._value_path(key, value)
            if not path.exists():
                fd, tmpname = tempfile.mkstemp()
                tmppath = Path(tmpname)
                os.write(fd, value)
                os.close(fd)
                try:
                    tmppath.rename(path)
                except OSError:
                    tmppath.unlink()
                assert not tmppath.exists()
        except OSError:
            pass

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        if src == dest:
            self.save(src, value)
            return
        try:
            os.renames(self._value_path(src, value), self._value_path(dest, value))
        except OSError:
            self.delete(src, value)
            self.save(dest, value)

    def delete(self, key: bytes, value: bytes) -> None:
        try:
            self._value_path(key, value).unlink()
        except OSError:
            pass


class ReadOnlyDatabase(ExampleDatabase):
    """A wrapper to make the given database read-only.

    The implementation passes through ``fetch``, and turns ``save``, ``delete``, and
    ``move`` into silent no-ops.
    """

    def __init__(self, db: ExampleDatabase) -> None:
        assert isinstance(db, ExampleDatabase)
        self._wrapped: ExampleDatabase = db

    def __repr__(self) -> str:
        return f'ReadOnlyDatabase({self._wrapped!r})'

    def fetch(self, key: bytes) -> IterableType[bytes]:
        yield from self._wrapped.fetch(key)

    def save(self, key: bytes, value: bytes) -> None:
        pass

    def delete(self, key: bytes, value: bytes) -> None:
        pass


class MultiplexedDatabase(ExampleDatabase):
    """A wrapper around multiple databases.

    Each operation will be run against all of the wrapped databases.
    """

    def __init__(self, *dbs: ExampleDatabase) -> None:
        assert all((isinstance(db, ExampleDatabase) for db in dbs))
        self._wrapped: tuple[ExampleDatabase, ...] = dbs

    def __repr__(self) -> str:
        return 'MultiplexedDatabase({})'.format(', '.join(map(repr, self._wrapped)))

    def fetch(self, key: bytes) -> IterableType[bytes]:
        seen: set[bytes] = set()
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
    """
    A file-based database loaded from a GitHub Actions artifact.
    """

    def __init__(
        self,
        owner: str,
        repo: str,
        artifact_name: str = 'hypothesis-example-db',
        cache_timeout: timedelta = timedelta(days=1),
        path: Optional[StrPathT] = None,
    ) -> None:
        self.owner: str = owner
        self.repo: str = repo
        self.artifact_name: str = artifact_name
        self.cache_timeout: timedelta = cache_timeout
        self.token: Optional[str] = getenv('GITHUB_TOKEN')
        if path is None:
            self.path: Path = Path(storage_directory(f'github-artifacts/{self.artifact_name}/'))
        else:
            self.path = Path(path)
        self._initialized: bool = False
        self._disabled: bool = False
        self._artifact: Optional[Path] = None
        self._access_cache: Optional[dict[PurePath, set[PurePath]]] = None
        self._read_only_message: str = (
            'This database is read-only. Please wrap this class with ReadOnlyDatabasei.e. '
            'ReadOnlyDatabase(GitHubArtifactDatabase(...)).'
        )

    def __repr__(self) -> str:
        return f'GitHubArtifactDatabase(owner={self.owner!r}, repo={self.repo!r}, artifact_name={self.artifact_name!r})'

    def _prepare_for_io(self) -> None:
        assert self._artifact is not None, 'Artifact not loaded.'
        if self._initialized:
            return
        try:
            with ZipFile(self._artifact) as f:
                if f.testzip():
                    raise BadZipFile
            self._access_cache = {}
            with ZipFile(self._artifact) as zf:
                namelist = zf.namelist()
                for filename in namelist:
                    fileinfo = zf.getinfo(filename)
                    p_filename = PurePath(filename)
                    if fileinfo.is_dir():
                        self._access_cache[p_filename] = set()
                    else:
                        keypath = p_filename.parent
                        if keypath not in self._access_cache:
                            self._access_cache[keypath] = set()
                        self._access_cache[keypath].add(p_filename)
        except BadZipFile:
            warnings.warn(
                'The downloaded artifact from GitHub is invalid. This could be because the artifact was corrupted, or because the artifact was not created by Hypothesis. ',
                HypothesisWarning,
                stacklevel=3,
            )
            self._disabled = True
        self._initialized = True

    def _initialize_db(self) -> None:
        storage_directory(self.path.name)
        self.path.mkdir(exist_ok=True, parents=True)
        cached_artifacts = sorted(self.path.glob('*.zip'), key=lambda a: datetime.fromisoformat(a.stem.replace('_', ':')))
        for artifact in cached_artifacts[:-1]:
            artifact.unlink()
        try:
            found_artifact = cached_artifacts[-1]
        except IndexError:
            found_artifact = None
        if found_artifact is not None and datetime.now(timezone.utc) - datetime.fromisoformat(found_artifact.stem.replace('_', ':')) < self.cache_timeout:
            self._artifact = found_artifact
        else:
            new_artifact = self._fetch_artifact()
            if new_artifact:
                if found_artifact is not None:
                    found_artifact.unlink()
                self._artifact = new_artifact
            elif found_artifact is not None:
                warnings.warn(
                    f'Using an expired artifact as a fallback for the database: {found_artifact}',
                    HypothesisWarning,
                    stacklevel=2,
                )
                self._artifact = found_artifact
            else:
                warnings.warn("Couldn't acquire a new or existing artifact. Disabling database.", HypothesisWarning, stacklevel=2)
                self._disabled = True
                return
        self._prepare_for_io()

    def _get_bytes(self, url: str) -> Optional[bytes]:
        request = Request(
            url,
            headers={
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28 ',
                'Authorization': f'Bearer {self.token}',
            },
        )
        warning_message: Optional[str] = None
        response_bytes: Optional[bytes] = None
        try:
            with urlopen(request) as response:
                response_bytes = response.read()
        except HTTPError as e:
            if e.code == 401:
                warning_message = 'Authorization failed when trying to download artifact from GitHub. Check that you have a valid GITHUB_TOKEN set in your environment.'
            else:
                warning_message = 'Could not get the latest artifact from GitHub. This could be because because the repository or artifact does not exist. '
        except URLError:
            warning_message = 'Could not connect to GitHub to get the latest artifact. '
        except TimeoutError:
            warning_message = 'Could not connect to GitHub to get the latest artifact (connection timed out).'
        if warning_message is not None:
            warnings.warn(warning_message, HypothesisWarning, stacklevel=4)
            return None
        return response_bytes

    def _fetch_artifact(self) -> Optional[Path]:
        url = f'https://api.github.com/repos/{self.owner}/{self.repo}/actions/artifacts'
        response_bytes = self._get_bytes(url)
        if response_bytes is None:
            return None
        artifacts = json.loads(response_bytes)['artifacts']
        artifacts = [a for a in artifacts if a['name'] == self.artifact_name]
        if not artifacts:
            return None
        artifact = max(artifacts, key=lambda a: a['created_at'])
        url = artifact['archive_download_url']
        artifact_bytes = self._get_bytes(url)
        if artifact_bytes is None:
            return None
        timestamp = datetime.now(timezone.utc).isoformat().replace(':', '_')
        artifact_path = self.path / f'{timestamp}.zip'
        try:
            artifact_path.write_bytes(artifact_bytes)
        except OSError:
            warnings.warn('Could not save the latest artifact from GitHub. ', HypothesisWarning, stacklevel=3)
            return None
        return artifact_path

    @staticmethod
    @lru_cache(maxsize=None)
    def _key_path(key: bytes) -> PurePath:
        return PurePath(_hash(key) + '/')

    def fetch(self, key: bytes) -> IterableType[bytes]:
        if self._disabled:
            return
        if not self._initialized:
            self._initialize_db()
            if self._disabled:
                return
        assert self._artifact is not None
        assert self._access_cache is not None
        kp: PurePath = GitHubArtifactDatabase._key_path(key)
        with ZipFile(self._artifact) as zf:
            filenames = self._access_cache.get(kp, ())
            for filename in filenames:
                with zf.open(filename.as_posix()) as f:
                    yield f.read()

    def save(self, key: bytes, value: bytes) -> None:
        raise RuntimeError(self._read_only_message)

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        raise RuntimeError(self._read_only_message)

    def delete(self, key: bytes, value: bytes) -> None:
        raise RuntimeError(self._read_only_message)


class BackgroundWriteDatabase(ExampleDatabase):
    """A wrapper which defers writes on the given database to a background thread.
    """

    def __init__(self, db: ExampleDatabase) -> None:
        self._db: ExampleDatabase = db
        self._queue: Queue[tuple[str, tuple[Any, ...]]] = Queue()
        self._thread: Thread = Thread(target=self._worker, daemon=True)
        self._thread.start()
        weakref.finalize(self, self._join, 0.1)

    def __repr__(self) -> str:
        return f'BackgroundWriteDatabase({self._db!r})'

    def _worker(self) -> None:
        while True:
            method, args = self._queue.get()
            getattr(self._db, method)(*args)
            self._queue.task_done()

    def _join(self, timeout: Optional[float] = None) -> None:
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                self._queue.all_tasks_done.wait(timeout)

    def fetch(self, key: bytes) -> IterableType[bytes]:
        self._join()
        return self._db.fetch(key)

    def save(self, key: bytes, value: bytes) -> None:
        self._queue.put(('save', (key, value)))

    def delete(self, key: bytes, value: bytes) -> None:
        self._queue.put(('delete', (key, value)))

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        self._queue.put(('move', (src, dest, value)))


def _pack_uleb128(value: int) -> bytes:
    """
    Serialize an integer into variable-length bytes using LEB128.
    """
    parts = bytearray()
    assert value >= 0
    while True:
        byte = value & ((1 << 7) - 1)
        value >>= 7
        if value:
            byte |= 1 << 7
        parts.append(byte)
        if not value:
            break
    return bytes(parts)


def _unpack_uleb128(buffer: bytes) -> tuple[int, int]:
    """
    Inverts _pack_uleb128, returns (bytes_read, value).
    """
    value = 0
    for i, byte in enumerate(buffer):
        n = byte & ((1 << 7) - 1)
        value |= n << (i * 7)
        if not (byte >> 7):
            break
    return (i + 1, value)


def choices_to_bytes(ir: IterableType[Union[bool, float, int, bytes, str]]) -> bytes:
    """Serialize a list of IR elements to a bytestring. Inverts choices_from_bytes."""
    parts: list[bytes] = []
    for elem in ir:
        if isinstance(elem, bool):
            parts.append(b'\x01' if elem else b'\x00')
            continue
        if isinstance(elem, float):
            tag = 1 << 5
            elem_bytes = struct.pack('!d', elem)
        elif isinstance(elem, int):
            tag = 2 << 5
            # Use at least 1 byte for integer representation
            num_bytes = max(1, (elem.bit_length() + 7) // 8)
            elem_bytes = elem.to_bytes(num_bytes, 'big', signed=True)
        elif isinstance(elem, bytes):
            tag = 3 << 5
            elem_bytes = elem
        else:
            assert isinstance(elem, str)
            tag = 4 << 5
            elem_bytes = elem.encode(errors='surrogatepass')
        size = len(elem_bytes)
        if size < 31:
            parts.append((tag | size).to_bytes(1, 'big'))
        else:
            parts.append((tag | 31).to_bytes(1, 'big'))
            parts.append(_pack_uleb128(size))
        parts.append(elem_bytes)
    return b''.join(parts)


def _choices_from_bytes(buffer: bytes) -> list[Union[bool, float, int, bytes, str]]:
    parts: list[Union[bool, float, int, bytes, str]] = []
    idx = 0
    while idx < len(buffer):
        tag = buffer[idx] >> 5
        size = buffer[idx] & 31
        idx += 1
        if tag == 0:
            parts.append(bool(size))
            continue
        if size == 31:
            offset, size = _unpack_uleb128(buffer[idx:])
            idx += offset
        chunk = buffer[idx:idx + size]
        idx += size
        if tag == 1:
            assert size == 8, 'expected float64'
            parts.extend(struct.unpack('!d', chunk))
        elif tag == 2:
            parts.append(int.from_bytes(chunk, 'big', signed=True))
        elif tag == 3:
            parts.append(chunk)
        else:
            assert tag == 4
            parts.append(chunk.decode(errors='surrogatepass'))
    return parts


def choices_from_bytes(buffer: bytes) -> Optional[tuple[Union[bool, float, int, bytes, str], ...]]:
    """
    Deserialize a bytestring to a tuple of choices. Inverts choices_to_bytes.
    Returns None if the given bytestring is not a valid serialization.
    """
    try:
        return tuple(_choices_from_bytes(buffer))
    except Exception:
        return None
