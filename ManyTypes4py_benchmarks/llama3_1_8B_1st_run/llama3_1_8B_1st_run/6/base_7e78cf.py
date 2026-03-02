import abc
import functools
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generic, List, Optional, TypeVar, Type, Protocol

from dbt.contracts.project import ProjectPackageMetadata
from dbt.events.types import DepsSetDownloadDirectory
from dbt_common.clients import system
from dbt_common.events.functions import fire_event
from dbt_common.utils.connection import connection_exception_retry

DOWNLOADS_PATH: Optional[str] = None

def get_downloads_path() -> str:
    return DOWNLOADS_PATH

@contextmanager
def downloads_directory() -> str:
    global DOWNLOADS_PATH
    remove_downloads: bool = False
    if DOWNLOADS_PATH is None:
        DOWNLOADS_PATH = os.getenv('DBT_DOWNLOADS_DIR')
        remove_downloads = False
    if DOWNLOADS_PATH is None:
        DOWNLOADS_PATH = tempfile.mkdtemp(prefix='dbt-downloads-')
        remove_downloads = True
    system.make_directory(DOWNLOADS_PATH)
    fire_event(DepsSetDownloadDirectory(path=DOWNLOADS_PATH))
    yield DOWNLOADS_PATH
    if remove_downloads:
        system.rmtree(DOWNLOADS_PATH)
        DOWNLOADS_PATH = None

class BasePackage(Protocol):
    @property
    def name(self) -> str:
        ...

    def all_names(self) -> List[str]:
        ...

    @property
    def source_type(self) -> str:
        ...

class BasePackageABC(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        ...

    def all_names(self) -> List[str]:
        return [self.name]

    @abc.abstractmethod
    def source_type(self) -> str:
        ...

class PinnedPackage(BasePackageABC):
    def __init__(self) -> None:
        self._cached_metadata: Optional[ProjectPackageMetadata] = None

    def __str__(self) -> str:
        version: Optional[str] = self.get_version()
        if not version:
            return self.name
        return f'{self.name}@{version}'

    @abc.abstractmethod
    def get_version(self) -> Optional[str]:
        ...

    @abc.abstractmethod
    def _fetch_metadata(self, project: 'Project', renderer: 'Renderer') -> ProjectPackageMetadata:
        ...

    @abc.abstractmethod
    def install(self, project: 'Project', renderer: 'Renderer') -> None:
        ...

    @abc.abstractmethod
    def nice_version_name(self) -> str:
        ...

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, str]:
        ...

    def fetch_metadata(self, project: 'Project', renderer: 'Renderer') -> ProjectPackageMetadata:
        if not self._cached_metadata:
            self._cached_metadata = self._fetch_metadata(project, renderer)
        return self._cached_metadata

    def get_project_name(self, project: 'Project', renderer: 'Renderer') -> str:
        metadata: ProjectPackageMetadata = self.fetch_metadata(project, renderer)
        return metadata.name

    def get_installation_path(self, project: 'Project', renderer: 'Renderer') -> str:
        dest_dirname: str = self.get_project_name(project, renderer)
        return os.path.join(project.packages_install_path, dest_dirname)

    def get_subdirectory(self) -> Optional[str]:
        return None

    def _install(self, project: 'Project', renderer: 'Renderer') -> None:
        metadata: ProjectPackageMetadata = self.fetch_metadata(project, renderer)
        tar_name: str = f'{self.package}.{self.version}.tar.gz'
        tar_path: Path = (Path(get_downloads_path()) / tar_name).resolve(strict=False)
        system.make_directory(str(tar_path.parent))
        download_url: str = metadata.downloads.tarball
        deps_path: str = project.packages_install_path
        package_name: str = self.get_project_name(project, renderer)
        download_untar_fn: Callable[[str, str, str, str], None] = functools.partial(
            self.download_and_untar, download_url, str(tar_path), deps_path, package_name
        )
        connection_exception_retry(download_untar_fn, 5)

    def download_and_untar(
        self, download_url: str, tar_path: str, deps_path: str, package_name: str
    ) -> None:
        """
        Sometimes the download of the files fails and we want to retry.  Sometimes the
        download appears successful but the file did not make it through as expected
        (generally due to a github incident).  Either way we want to retry downloading
        and untarring to see if we can get a success.  Call this within
        `connection_exception_retry`
        """
        system.download(download_url, tar_path)
        system.untar_package(tar_path, deps_path, package_name)

SomePinned = TypeVar('SomePinned', bound=PinnedPackage)
SomeUnpinned = TypeVar('SomeUnpinned', bound='UnpinnedPackage')

class UnpinnedPackage(Generic[SomePinned], BasePackageABC):
    @classmethod
    @abc.abstractmethod
    def from_contract(cls: Type[UnpinnedPackage[SomePinned]], contract: 'Contract') -> UnpinnedPackage[SomePinned]:
        ...

    @abc.abstractmethod
    def incorporate(self, other: SomeUnpinned) -> None:
        ...

    @abc.abstractmethod
    def resolved(self) -> SomePinned:
        ...

class Project(Protocol):
    ...

class Renderer(Protocol):
    ...

class Contract(Protocol):
    ...
