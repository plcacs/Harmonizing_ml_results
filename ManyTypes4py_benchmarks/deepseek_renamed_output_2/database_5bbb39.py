import abc
import json
import os
import struct
import sys
import tempfile
import warnings
import weakref
from collections.abc import Iterable, Set
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha384
from os import PathLike, getenv
from pathlib import Path, PurePath
from queue import Queue
from threading import Thread
from typing import (
    TYPE_CHECKING, Any, BinaryIO, Dict, Final, Iterator, List, Optional, Tuple, TypeVar,
    Union, cast, overload
)
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
from hypothesis.configuration import storage_directory
from hypothesis.errors import HypothesisException, HypothesisWarning
from hypothesis.internal.conjecture.choice import ChoiceT
from hypothesis.utils.conventions import UniqueIdentifier, not_set

__all__ = [
    'DirectoryBasedExampleDatabase', 'ExampleDatabase',
    'GitHubArtifactDatabase', 'InMemoryExampleDatabase',
    'MultiplexedDatabase', 'ReadOnlyDatabase'
]

if TYPE_CHECKING:
    from typing import TypeAlias
    StrPathT: TypeAlias = Union[str, PathLike[str]]
else:
    StrPathT = Union[str, PathLike[str]]

T = TypeVar('T')
Example = bytes
Key = bytes

def func_xndsnuji(path: Union[str, PathLike[str]]) -> bool:
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

def func_ek79ubjq(path: Union[UniqueIdentifier, StrPathT, None] = not_set) -> 'ExampleDatabase':
    if path is not_set:
        if os.getenv('HYPOTHESIS_DATABASE_FILE') is not None:
            raise HypothesisException(
                """The $HYPOTHESIS_DATABASE_FILE environment variable no longer has any effect.  Configure your database location via a settings profile instead.
https://hypothesis.readthedocs.io/en/latest/settings.html#settings-profiles"""
                )
        path = storage_directory('examples', intent_to_write=False)
        if not func_xndsnuji(path):
            warnings.warn(
                f'The database setting is not configured, and the default location is unusable - falling back to an in-memory database for this session.  path={path!r}'
                , HypothesisWarning, stacklevel=3)
            return InMemoryExampleDatabase()
    if path in (None, ':memory:'):
        return InMemoryExampleDatabase()
    path = cast(StrPathT, path)
    return DirectoryBasedExampleDatabase(path)

class _EDMeta(abc.ABCMeta):
    def __call__(self, *args: Any, **kwargs: Any) -> 'ExampleDatabase':
        if self is ExampleDatabase:
            return func_ek79ubjq(*args, **kwargs)
        return super().__call__(*args, **kwargs)

if 'sphinx' in sys.modules:
    try:
        from sphinx.ext.autodoc import _METACLASS_CALL_BLACKLIST
        _METACLASS_CALL_BLACKLIST.append('hypothesis.database._EDMeta.__call__')
    except Exception:
        pass

class ExampleDatabase(metaclass=_EDMeta):
    """An abstract base class for storing examples in Hypothesis' internal format."""

    @abc.abstractmethod
    def func_gg0c3wyf(self, key: Key, value: Example) -> None:
        """Save ``value`` under ``key``."""
        raise NotImplementedError(f'{type(self).__name__}.save')

    @abc.abstractmethod
    def func_rmsthu24(self, key: Key) -> Iterator[Example]:
        """Return an iterable over all values matching this key."""
        raise NotImplementedError(f'{type(self).__name__}.fetch')

    @abc.abstractmethod
    def func_kuhh5i2k(self, key: Key, value: Example) -> None:
        """Remove this value from this key."""
        raise NotImplementedError(f'{type(self).__name__}.delete')

    def func_s8vcjllv(self, src: Key, dest: Key, value: Example) -> None:
        """Move ``value`` from key ``src`` to key ``dest``."""
        if src == dest:
            self.func_gg0c3wyf(src, value)
            return
        self.func_kuhh5i2k(src, value)
        self.func_gg0c3wyf(dest, value)

    # Aliases for better readability
    save = func_gg0c3wyf
    fetch = func_rmsthu24
    delete = func_kuhh5i2k
    move = func_s8vcjllv

class InMemoryExampleDatabase(ExampleDatabase):
    def __init__(self) -> None:
        self.data: Dict[Key, Set[Example]] = {}

    def __repr__(self) -> str:
        return f'InMemoryExampleDatabase({self.data!r})'

    def func_rmsthu24(self, key: Key) -> Iterator[Example]:
        yield from self.data.get(key, ())

    def func_gg0c3wyf(self, key: Key, value: Example) -> None:
        self.data.setdefault(key, set()).add(bytes(value))

    def func_kuhh5i2k(self, key: Key, value: Example) -> None:
        self.data.get(key, set()).discard(bytes(value))

def func_ecn9w2xg(key: bytes) -> str:
    return sha384(key).hexdigest()[:16]

class DirectoryBasedExampleDatabase(ExampleDatabase):
    def __init__(self, path: StrPathT) -> None:
        self.path = Path(path)
        self.keypaths: Dict[Key, Path] = {}

    def __repr__(self) -> str:
        return f'DirectoryBasedExampleDatabase({self.path!r})'

    def _key_path(self, key: Key) -> Path:
        try:
            return self.keypaths[key]
        except KeyError:
            pass
        self.keypaths[key] = self.path / func_ecn9w2xg(key)
        return self.keypaths[key]

    def _value_path(self, key: Key, value: Example) -> Path:
        return self._key_path(key) / func_ecn9w2xg(value)

    def func_rmsthu24(self, key: Key) -> Iterator[Example]:
        kp = self._key_path(key)
        if not kp.is_dir():
            return
        for path in os.listdir(kp):
            try:
                yield (kp / path).read_bytes()
            except OSError:
                pass

    def func_gg0c3wyf(self, key: Key, value: Example) -> None:
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

    def func_s8vcjllv(self, src: Key, dest: Key, value: Example) -> None:
        if src == dest:
            self.save(src, value)
            return
        try:
            os.renames(self._value_path(src, value), self._value_path(dest, value))
        except OSError:
            self.delete(src, value)
            self.save(dest, value)

    def func_kuhh5i2k(self, key: Key, value: Example) -> None:
        try:
            self._value_path(key, value).unlink()
        except OSError:
            pass

class ReadOnlyDatabase(ExampleDatabase):
    def __init__(self, db: ExampleDatabase) -> None:
        assert isinstance(db, ExampleDatabase)
        self._wrapped = db

    def __repr__(self) -> str:
        return f'ReadOnlyDatabase({self._wrapped!r})'

    def func_rmsthu24(self, key: Key) -> Iterator[Example]:
        yield from self._wrapped.fetch(key)

    def func_gg0c3wyf(self, key: Key, value: Example) -> None:
        pass

    def func_kuhh5i2k(self, key: Key, value: Example) -> None:
        pass

class MultiplexedDatabase(ExampleDatabase):
    def __init__(self, *dbs: ExampleDatabase) -> None:
        assert all(isinstance(db, ExampleDatabase) for db in dbs)
        self._wrapped = dbs

    def __repr__(self) -> str:
        return 'MultiplexedDatabase({})'.format(', '.join(map(repr, self._wrapped)))

    def func_rmsthu24(self, key: Key) -> Iterator[Example]:
        seen: Set[Example] = set()
        for db in self._wrapped:
            for value in db.fetch(key):
                if value not in seen:
                    yield value
                    seen.add(value)

    def func_gg0c3wyf(self, key: Key, value: Example) -> None:
        for db in self._wrapped:
            db.save(key, value)

    def func_kuhh5i2k(self, key: Key, value: Example) -> None:
        for db in self._wrapped:
            db.delete(key, value)

    def func_s8vcjllv(self, src: Key, dest: Key, value: Example) -> None:
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
        self._initialized = False
        self._disabled = False
        self._artifact: Optional[Path] = None
        self._access_cache: Optional[Dict[PurePath, Set[PurePath]]] = None
        self._read_only_message = (
            'This database is read-only. Please wrap this class with ReadOnlyDatabasei.e. ReadOnlyDatabase(GitHubArtifactDatabase(...)).'
            )

    def __repr__(self) -> str:
        return f'GitHubArtifactDatabase(owner={self.owner!r}, repo={self.repo!r}, artifact_name={self.artifact_name!r})'

    def _initialize_db(self) -> None:
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
                        self._access_cache[keypath].add(PurePath(filename))
        except BadZipFile:
            warnings.warn(
                'The downloaded artifact from GitHub is invalid. This could be because the artifact was corrupted, or because the artifact was not created by Hypothesis. '
                , HypothesisWarning, stacklevel=3)
            self._disabled = True
        self._initialized = True

    def _prepare_for_io(self) -> None:
        storage_directory(self.path.name)
        self.path.mkdir(exist_ok=True, parents=True)
        cached_artifacts = sorted(self.path.glob('*.zip'), key=lambda a:
            datetime.fromisoformat(a.stem.replace('_', ':')))
        for artifact in cached_artifacts[:-1]:
            artifact.unlink()
        try:
            found_artifact = cached_artifacts[-1]
        except IndexError:
            found_artifact = None
        if found_artifact is not None and datetime.now(timezone.utc
            ) - datetime.fromisoformat(found_artifact.stem.replace('_', ':')
            ) < self.cache_timeout:
            self._artifact = found_artifact
        else:
            new_artifact = self._fetch_artifact()
            if new_artifact:
                if found_artifact is not None:
                    found_artifact.unlink()
                self._artifact = new_artifact
            elif found_artifact is not None:
                warnings.warn(
                    f'Using an expired artifact as a fallback for the database: {found_artifact}'
                    , HypothesisWarning, stacklevel=2)
                self._artifact = found_artifact
            else:
                warnings.warn(
                    "Couldn't acquire a new or existing artifact. Disabling database."
                    , HypothesisWarning, stacklevel=2)
                self._disabled = True
                return
        self._initialize_db()

    def _get_bytes(self, url: str) -> Optional[bytes]:
        request = Request(url, headers={'Accept':
            'application/vnd.github+json', 'X-GitHub-Api-Version':
            '2022-11-28 ', 'Authorization': f'Bearer {self.token}'})
        warning_message = None
        response_bytes = None
        try:
            with urlopen(request) as response:
                response_bytes = response.read()
        except HTTPError as e:
            if e.code == 401:
                warning_message = (
                    'Authorization failed when trying to download artifact from GitHub. Check that you have a valid GITHUB_TOKEN set in your environment.'
                    )
            else:
                warning_message = (
                    'Could not get the latest artifact from GitHub. This could be because because the repository or artifact does not exist. '
                    )
        except URLError:
            warning_message = (
                'Could not connect to GitHub to get the latest artifact. ')
        except TimeoutError:
            warning_message = (
                'Could not connect to GitHub to get the latest artifact (connection timed out).'
                )
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
            warnings.warn('Could not save the latest artifact from GitHub. ',
                HypothesisWarning, stacklevel=3)
            return None
        return artifact_path

    @staticmethod
    @lru_cache
    def _key_path(key: Key) -> PurePath:
        return PurePath(func_ecn9w2xg(key) + '/')

    def func_rmsthu24(self, key: Key) -> Iterator[Example]:
        if self._disabled:
            return
        if not self._initialized:
            self._prepare_for_io()
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

    def func_gg0c3wyf(self, key: Key, value: Example) -> None:
        raise RuntimeError(self._read_only_message)

    def func_s8vcjllv(self, src: Key, dest: Key, value: Example) -> None:
        raise RuntimeError(self._read_only_message)

    def func_kuhh5i2k(self, key: Key, value: Example) -> None:
        raise RuntimeError(self._read_only_message)

class BackgroundWriteDatabase(ExampleDatabase):
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

    def func_rmsthu24(self, key: Key)