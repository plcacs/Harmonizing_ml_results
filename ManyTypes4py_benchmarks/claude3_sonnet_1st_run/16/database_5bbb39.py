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
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Literal, Optional, Set, Tuple, Type, TypeVar, Union, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
from hypothesis.configuration import storage_directory
from hypothesis.errors import HypothesisException, HypothesisWarning
from hypothesis.internal.conjecture.choice import ChoiceT
from hypothesis.utils.conventions import UniqueIdentifier, not_set

__all__ = ['DirectoryBasedExampleDatabase', 'ExampleDatabase', 'GitHubArtifactDatabase', 'InMemoryExampleDatabase', 'MultiplexedDatabase', 'ReadOnlyDatabase']

if TYPE_CHECKING:
    from typing import TypeAlias

StrPathT = Union[str, PathLike[str]]
T = TypeVar('T')

def _usable_dir(path: StrPathT) -> bool:
    """
    Returns True if the desired path can be used as database path because
    either the directory exists and can be used, or its root directory can
    be used and we can make the directory as needed.
    """
    path = Path(path)
    try:
        while not path.exists():
            path = path.parent
        return path.is_dir() and os.access(path, os.R_OK | os.W_OK | os.X_OK)
    except PermissionError:
        return False

def _db_for_path(path: Any = not_set) -> 'ExampleDatabase':
    if path is not_set:
        if os.getenv('HYPOTHESIS_DATABASE_FILE') is not None:
            raise HypothesisException('The $HYPOTHESIS_DATABASE_FILE environment variable no longer has any effect.  Configure your database location via a settings profile instead.\nhttps://hypothesis.readthedocs.io/en/latest/settings.html#settings-profiles')
        path = storage_directory('examples', intent_to_write=False)
        if not _usable_dir(path):
            warnings.warn(f'The database setting is not configured, and the default location is unusable - falling back to an in-memory database for this session.  path={path!r}', HypothesisWarning, stacklevel=3)
            return InMemoryExampleDatabase()
    if path in (None, ':memory:'):
        return InMemoryExampleDatabase()
    path = cast(StrPathT, path)
    return DirectoryBasedExampleDatabase(path)

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
    def fetch(self, key: bytes) -> Iterator[bytes]:
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
        self.data: Dict[bytes, Set[bytes]] = {}

    def __repr__(self) -> str:
        return f'InMemoryExampleDatabase({self.data!r})'

    def fetch(self, key: bytes) -> Iterator[bytes]:
        yield from self.data.get(key, ())

    def save(self, key: bytes, value: bytes) -> None:
        self.data.setdefault(key, set()).add(bytes(value))

    def delete(self, key: bytes, value: bytes) -> None:
        self.data.get(key, set()).discard(bytes(value))

def _hash(key: bytes) -> str:
    return sha384(key).hexdigest()[:16]

class DirectoryBasedExampleDatabase(ExampleDatabase):
    """Use a directory to store Hypothesis examples as files.

    Each test corresponds to a directory, and each example to a file within that
    directory.  While the contents are fairly opaque, a
    ``DirectoryBasedExampleDatabase`` can be shared by checking the directory
    into version control, for example with the following ``.gitignore``::

        # Ignore files cached by Hypothesis...
        .hypothesis/*
        # except for the examples directory
        !.hypothesis/examples/

    Note however that this only makes sense if you also pin to an exact version of
    Hypothesis, and we would usually recommend implementing a shared database with
    a network datastore - see :class:`~hypothesis.database.ExampleDatabase`, and
    the :class:`~hypothesis.database.MultiplexedDatabase` helper.
    """

    def __init__(self, path: StrPathT) -> None:
        self.path = Path(path)
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

    Note that this disables Hypothesis' automatic discarding of stale examples.
    It is designed to allow local machines to access a shared database (e.g. from CI
    servers), without propagating changes back from a local or in-development branch.
    """

    def __init__(self, db: ExampleDatabase) -> None:
        assert isinstance(db, ExampleDatabase)
        self._wrapped = db

    def __repr__(self) -> str:
        return f'ReadOnlyDatabase({self._wrapped!r})'

    def fetch(self, key: bytes) -> Iterator[bytes]:
        yield from self._wrapped.fetch(key)

    def save(self, key: bytes, value: bytes) -> None:
        pass

    def delete(self, key: bytes, value: bytes) -> None:
        pass

class MultiplexedDatabase(ExampleDatabase):
    """A wrapper around multiple databases.

    Each ``save``, ``fetch``, ``move``, or ``delete`` operation will be run against
    all of the wrapped databases.  ``fetch`` does not yield duplicate values, even
    if the same value is present in two or more of the wrapped databases.

    This combines well with a :class:`ReadOnlyDatabase`, as follows:

    .. code-block:: python

        local = DirectoryBasedExampleDatabase("/tmp/hypothesis/examples/")
        shared = CustomNetworkDatabase()

        settings.register_profile("ci", database=shared)
        settings.register_profile(
            "dev", database=MultiplexedDatabase(local, ReadOnlyDatabase(shared))
        )
        settings.load_profile("ci" if os.environ.get("CI") else "dev")

    So your CI system or fuzzing runs can populate a central shared database;
    while local runs on development machines can reproduce any failures from CI
    but will only cache their own failures locally and cannot remove examples
    from the shared database.
    """

    def __init__(self, *dbs: ExampleDatabase) -> None:
        assert all((isinstance(db, ExampleDatabase) for db in dbs))
        self._wrapped = dbs

    def __repr__(self) -> str:
        return 'MultiplexedDatabase({})'.format(', '.join(map(repr, self._wrapped)))

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
    """
    A file-based database loaded from a `GitHub Actions <https://docs.github.com/en/actions>`_ artifact.

    You can use this for sharing example databases between CI runs and developers, allowing
    the latter to get read-only access to the former. This is particularly useful for
    continuous fuzzing (i.e. with `HypoFuzz <https://hypofuzz.com/>`_),
    where the CI system can help find new failing examples through fuzzing,
    and developers can reproduce them locally without any manual effort.

    .. note::
        You must provide ``GITHUB_TOKEN`` as an environment variable. In CI, Github Actions provides
        this automatically, but it needs to be set manually for local usage. In a developer machine,
        this would usually be a `Personal Access Token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens>`_.
        If the repository is private, it's necessary for the token to have ``repo`` scope
        in the case of a classic token, or ``actions:read`` in the case of a fine-grained token.


    In most cases, this will be used
    through the :class:`~hypothesis.database.MultiplexedDatabase`,
    by combining a local directory-based database with this one. For example:

    .. code-block:: python

        local = DirectoryBasedExampleDatabase(".hypothesis/examples")
        shared = ReadOnlyDatabase(GitHubArtifactDatabase("user", "repo"))

        settings.register_profile("ci", database=local)
        settings.register_profile("dev", database=MultiplexedDatabase(local, shared))
        # We don't want to use the shared database in CI, only to populate its local one.
        # which the workflow should then upload as an artifact.
        settings.load_profile("ci" if os.environ.get("CI") else "dev")

    .. note::
        Because this database is read-only, you always need to wrap it with the
        :class:`ReadOnlyDatabase`.

    A setup like this can be paired with a GitHub Actions workflow including
    something like the following:

    .. code-block:: yaml

        - name: Download example database
          uses: dawidd6/action-download-artifact@v2.24.3
          with:
            name: hypothesis-example-db
            path: .hypothesis/examples
            if_no_artifact_found: warn
            workflow_conclusion: completed

        - name: Run tests
          run: pytest

        - name: Upload example database
          uses: actions/upload-artifact@v3
          if: always()
          with:
            name: hypothesis-example-db
            path: .hypothesis/examples

    In this workflow, we use `dawidd6/action-download-artifact <https://github.com/dawidd6/action-download-artifact>`_
    to download the latest artifact given that the official `actions/download-artifact <https://github.com/actions/download-artifact>`_
    does not support downloading artifacts from previous workflow runs.

    The database automatically implements a simple file-based cache with a default expiration period
    of 1 day. You can adjust this through the ``cache_timeout`` property.

    For mono-repo support, you can provide a unique ``artifact_name`` (e.g. ``hypofuzz-example-db-frontend``).
    """

    def __init__(
        self, 
        owner: str, 
        repo: str, 
        artifact_name: str = 'hypothesis-example-db', 
        cache_timeout: timedelta = timedelta(days=1), 
        path: Optional[StrPathT] = None
    ) -> None:
        self.owner = owner
        self.repo = repo
        self.artifact_name = artifact_name
        self.cache_timeout = cache_timeout
        self.token = getenv('GITHUB_TOKEN')
        if path is None:
            self.path = Path(storage_directory(f'github-artifacts/{self.artifact_name}/'))
        else:
            self.path = Path(path)
        self._initialized: bool = False
        self._disabled: bool = False
        self._artifact: Optional[Path] = None
        self._access_cache: Optional[Dict[PurePath, Set[PurePath]]] = None
        self._read_only_message: str = 'This database is read-only. Please wrap this class with ReadOnlyDatabasei.e. ReadOnlyDatabase(GitHubArtifactDatabase(...)).'

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
                    if fileinfo.is_dir():
                        self._access_cache[PurePath(filename)] = set()
                    else:
                        keypath = PurePath(filename).parent
                        self._access_cache.setdefault(keypath, set()).add(PurePath(filename))
        except BadZipFile:
            warnings.warn('The downloaded artifact from GitHub is invalid. This could be because the artifact was corrupted, or because the artifact was not created by Hypothesis. ', HypothesisWarning, stacklevel=3)
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
                warnings.warn(f'Using an expired artifact as a fallback for the database: {found_artifact}', HypothesisWarning, stacklevel=2)
                self._artifact = found_artifact
            else:
                warnings.warn("Couldn't acquire a new or existing artifact. Disabling database.", HypothesisWarning, stacklevel=2)
                self._disabled = True
                return
        self._prepare_for_io()

    def _get_bytes(self, url: str) -> Optional[bytes]:
        request = Request(url, headers={'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28 ', 'Authorization': f'Bearer {self.token}'})
        warning_message = None
        response_bytes = None
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
    @lru_cache
    def _key_path(key: bytes) -> PurePath:
        return PurePath(_hash(key) + '/')

    def fetch(self, key: bytes) -> Iterator[bytes]:
        if self._disabled:
            return
        if not self._initialized:
            self._initialize_db()
            if self._disabled:
                return
        assert self._artifact is not None
        assert self._access_cache is not None
        kp = self._key_path(key)
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

    Calls to :meth:`~hypothesis.database.ExampleDatabase.fetch` wait for any
    enqueued writes to finish before fetching from the database.
    """

    def __init__(self, db: ExampleDatabase) -> None:
        self._db = db
        self._queue: Queue[Tuple[str, Tuple[Any, ...]]] = Queue()
        self._thread = Thread(target=self._worker, daemon=True)
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

    def fetch(self, key: bytes) -> Iterator[bytes]:
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
    Serialize an integer into variable-length bytes. For each byte, the first 7
    bits represent (part of) the integer, while the last bit indicates whether the
    integer continues into the next byte.

    https://en.wikipedia.org/wiki/LEB128
    """
    parts = bytearray()
    assert value >= 0
    while True:
        byte = value & (1 << 7) - 1
        value >>= 7
        if value:
            byte |= 1 << 7
        parts.append(byte)
        if not value:
            break
    return bytes(parts)

def _unpack_uleb128(buffer: bytes) -> Tuple[int, int]:
    """
    Inverts _pack_uleb128, and also returns the index at which at which we stopped
    reading.
    """
    value = 0
    for i, byte in enumerate(buffer):
        n = byte & (1 << 7) - 1
        value |= n << i * 7
        if not byte >> 7:
            break
    return (i + 1, value)

def choices_to_bytes(ir: List[Union[bool, float, int, bytes, str]], /) -> bytes:
    """Serialize a list of IR elements to a bytestring.  Inverts choices_from_bytes."""
    parts: List[bytes] = []
    for elem in ir:
        if isinstance(elem, bool):
            parts.append(b'\x01' if elem else b'\x00')
            continue
        if isinstance(elem, float):
            tag = 1 << 5
            elem = struct.pack('!d', elem)
        elif isinstance(elem, int):
            tag = 2 << 5
            elem = elem.to_bytes(1 + elem.bit_length() // 8, 'big', signed=True)
        elif isinstance(elem, bytes):
            tag = 3 << 5
        else:
            assert isinstance(elem, str)
            tag = 4 << 5
            elem = elem.encode(errors='surrogatepass')
        size = len(elem)
        if size < 31:
            parts.append((tag | size).to_bytes(1, 'big'))
        else:
            parts.append((tag | 31).to_bytes(1, 'big'))
            parts.append(_pack_uleb128(size))
        parts.append(elem)
    return b''.join(parts)

def _choices_from_bytes(buffer: bytes, /) -> Tuple[Union[bool, float, int, bytes, str], ...]:
    parts: List[Union[bool, float, int, bytes, str]] = []
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
    return tuple(parts)

def choices_from_bytes(buffer: bytes, /) -> Optional[Tuple[Union[bool, float, int, bytes, str], ...]]:
    """
    Deserialize a bytestring to a tuple of choices. Inverts choices_to_bytes.

    Returns None if the given bytestring is not a valid serialization of choice
    sequences.
    """
    try:
        return _choices_from_bytes(buffer)
    except Exception:
        return None
