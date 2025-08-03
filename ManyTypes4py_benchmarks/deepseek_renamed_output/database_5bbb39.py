import abc
import json
import os
import struct
import sys
import tempfile
import warnings
import weakref
from collections.abc import Iterable, Set, Mapping, Iterator
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha384
from os import PathLike, getenv
from pathlib import Path, PurePath
from queue import Queue
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Union,
    cast,
    Dict,
    List,
    Tuple,
    TypeVar,
    Generic,
    Callable,
    AbstractSet,
    Type,
    final,
    overload,
)
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
from hypothesis.configuration import storage_directory
from hypothesis.errors import HypothesisException, HypothesisWarning
from hypothesis.internal.conjecture.choice import ChoiceT
from hypothesis.utils.conventions import UniqueIdentifier, not_set

__all__ = [
    "DirectoryBasedExampleDatabase",
    "ExampleDatabase",
    "GitHubArtifactDatabase",
    "InMemoryExampleDatabase",
    "MultiplexedDatabase",
    "ReadOnlyDatabase",
]

if TYPE_CHECKING:
    from typing import TypeAlias

StrPathT = Union[str, PathLike[str]]
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def func_unz6xnao(path: Union[str, PathLike[str]]) -> bool:
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


def func_i35ipbt6(path: Optional[Union[str, PathLike[str]]] = not_set) -> "ExampleDatabase":
    if path is not_set:
        if os.getenv("HYPOTHESIS_DATABASE_FILE") is not None:
            raise HypothesisException(
                """The $HYPOTHESIS_DATABASE_FILE environment variable no longer has any effect.  Configure your database location via a settings profile instead.
https://hypothesis.readthedocs.io/en/latest/settings.html#settings-profiles"""
            )
        path = storage_directory("examples", intent_to_write=False)
        if not func_unz6xnao(path):
            warnings.warn(
                f"The database setting is not configured, and the default location is unusable - falling back to an in-memory database for this session.  path={path!r}",
                HypothesisWarning,
                stacklevel=3,
            )
            return InMemoryExampleDatabase()
    if path in (None, ":memory:"):
        return InMemoryExampleDatabase()
    path = cast(StrPathT, path)
    return DirectoryBasedExampleDatabase(path)


class _EDMeta(abc.ABCMeta):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self is ExampleDatabase:
            return func_i35ipbt6(*args, **kwargs)
        return super().__call__(*args, **kwargs)


if "sphinx" in sys.modules:
    try:
        from sphinx.ext.autodoc import _METACLASS_CALL_BLACKLIST

        _METACLASS_CALL_BLACKLIST.append("hypothesis.database._EDMeta.__call__")
    except Exception:
        pass


class ExampleDatabase(metaclass=_EDMeta):
    """An abstract base class for storing examples in Hypothesis' internal format.

    An ExampleDatabase maps each ``bytes`` key to many distinct ``bytes``
    values, like a ``Mapping[bytes, AbstractSet[bytes]]``.
    """

    @abc.abstractmethod
    def func_8o3r4li3(self, key: bytes, value: bytes) -> None:
        """Save ``value`` under ``key``.

        If this value is already present for this key, silently do nothing.
        """
        raise NotImplementedError(f"{type(self).__name__}.save")

    @abc.abstractmethod
    def func_pj97zdr0(self, key: bytes) -> Iterator[bytes]:
        """Return an iterable over all values matching this key."""
        raise NotImplementedError(f"{type(self).__name__}.fetch")

    @abc.abstractmethod
    def func_pkr5zmvw(self, key: bytes, value: bytes) -> None:
        """Remove this value from this key.

        If this value is not present, silently do nothing.
        """
        raise NotImplementedError(f"{type(self).__name__}.delete")

    def func_l8j0d4k7(self, src: bytes, dest: bytes, value: bytes) -> None:
        """Move ``value`` from key ``src`` to key ``dest``. Equivalent to
        ``delete(src, value)`` followed by ``save(src, value)``, but may
        have a more efficient implementation.

        Note that ``value`` will be inserted at ``dest`` regardless of whether
        it is currently present at ``src``.
        """
        if src == dest:
            self.func_8o3r4li3(src, value)
            return
        self.func_pkr5zmvw(src, value)
        self.func_8o3r4li3(dest, value)


class InMemoryExampleDatabase(ExampleDatabase):
    """A non-persistent example database, implemented in terms of a dict of sets.

    This can be useful if you call a test function several times in a single
    session, or for testing other database implementations, but because it
    does not persist between runs we do not recommend it for general use.
    """

    def __init__(self) -> None:
        self.data: Dict[bytes, Set[bytes]] = {}

    def __repr__(self) -> str:
        return f"InMemoryExampleDatabase({self.data!r})"

    def func_pj97zdr0(self, key: bytes) -> Iterator[bytes]:
        yield from self.data.get(key, set())

    def func_8o3r4li3(self, key: bytes, value: bytes) -> None:
        self.data.setdefault(key, set()).add(bytes(value))

    def func_pkr5zmvw(self, key: bytes, value: bytes) -> None:
        self.data.get(key, set()).discard(bytes(value))


def func_p8xr1uxy(key: bytes) -> str:
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

    def __init__(self, path: Union[str, PathLike[str]]) -> None:
        self.path = Path(path)
        self.keypaths: Dict[bytes, Path] = {}

    def __repr__(self) -> str:
        return f"DirectoryBasedExampleDatabase({self.path!r})"

    def func_b29f7kcc(self, key: bytes) -> Path:
        try:
            return self.keypaths[key]
        except KeyError:
            pass
        self.keypaths[key] = self.path / func_p8xr1uxy(key)
        return self.keypaths[key]

    def func_0uzuukcs(self, key: bytes, value: bytes) -> Path:
        return self.func_b29f7kcc(key) / func_p8xr1uxy(value)

    def func_pj97zdr0(self, key: bytes) -> Iterator[bytes]:
        kp = self.func_b29f7kcc(key)
        if not kp.is_dir():
            return
        for path in os.listdir(kp):
            try:
                yield (kp / path).read_bytes()
            except OSError:
                pass

    def func_8o3r4li3(self, key: bytes, value: bytes) -> None:
        try:
            self.func_b29f7kcc(key).mkdir(exist_ok=True, parents=True)
            path = self.func_0uzuukcs(key, value)
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

    def func_l8j0d4k7(self, src: bytes, dest: bytes, value: bytes) -> None:
        if src == dest:
            self.func_8o3r4li3(src, value)
            return
        try:
            os.renames(
                self.func_0uzuukcs(src, value), self.func_0uzuukcs(dest, value)
            )
        except OSError:
            self.func_pkr5zmvw(src, value)
            self.func_8o3r4li3(dest, value)

    def func_pkr5zmvw(self, key: bytes, value: bytes) -> None:
        try:
            self.func_0uzuukcs(key, value).unlink()
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
        return f"ReadOnlyDatabase({self._wrapped!r})"

    def func_pj97zdr0(self, key: bytes) -> Iterator[bytes]:
        yield from self._wrapped.func_pj97zdr0(key)

    def func_8o3r4li3(self, key: bytes, value: bytes) -> None:
        pass

    def func_pkr5zmvw(self, key: bytes, value: bytes) -> None:
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
        assert all(isinstance(db, ExampleDatabase) for db in dbs)
        self._wrapped = dbs

    def __repr__(self) -> str:
        return "MultiplexedDatabase({})".format(", ".join(map(repr, self._wrapped)))

    def func_pj97zdr0(self, key: bytes) -> Iterator[bytes]:
        seen: Set[bytes] = set()
        for db in self._wrapped:
            for value in db.func_pj97zdr0(key):
                if value not in seen:
                    yield value
                    seen.add(value)

    def func_8o3r4li3(self, key: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.func_8o3r4li3(key, value)

    def func_pkr5zmvw(self, key: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.func_pkr5zmvw(key, value)

    def func_l8j0d4k7(self, src: bytes, dest: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.func_l8j0d4k7(src, dest, value)


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
        artifact_name: str = "hypothesis-example-db",
        cache_timeout: timedelta = timedelta(days=1),
        path: Optional[Union[str, PathLike[str]]] = None,
    ) -> None:
        self.owner = owner
        self.repo = repo
        self.artifact_name = artifact_name
        self.cache_timeout = cache_timeout
        self.token = getenv("GITHUB_TOKEN")
        if path is None:
            self.path = Path(
                storage_directory(f"github-artifacts/{self.artifact_name}/")
            )
        else:
            self.path = Path(path)
        self._initialized = False
        self._disabled = False
        self._artifact: Optional[Path] = None
        self._access_cache: Optional[Dict[PurePath, Set[PurePath]]] = None
        self._read_only_message = (
            "This database is read-only. Please wrap this class with ReadOnlyDatabasei.e. ReadOnlyDatabase(GitHubArtifactDatabase(...))."
        )

    def __repr__(self) -> str:
        return f"GitHubArtifactDatabase(owner={self.owner!r}, repo={self.repo!r}, artifact_name={self.artifact_name!r})"

    def func_ewo67cga(self) -> None:
        assert self._artifact is not None, "Artifact not loaded."
        if self._initialized:
            return
        try:
