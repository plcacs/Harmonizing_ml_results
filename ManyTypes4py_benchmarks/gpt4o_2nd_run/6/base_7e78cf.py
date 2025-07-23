import abc
import functools
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generic, List, Optional, TypeVar, Iterator
from dbt.contracts.project import ProjectPackageMetadata
from dbt.events.types import DepsSetDownloadDirectory
from dbt_common.clients import system
from dbt_common.events.functions import fire_event
from dbt_common.utils.connection import connection_exception_retry

DOWNLOADS_PATH: Optional[str] = None

def get_downloads_path() -> Optional[str]:
    return DOWNLOADS_PATH

@contextmanager
def downloads_directory() -> Iterator[str]:
    global DOWNLOADS_PATH
    remove_downloads = False
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

class BasePackage(metaclass=abc.ABCMeta):

    @abc.abstractproperty
    def name(self) -> str:
        raise NotImplementedError

    def all_names(self) -> List[str]:
        return [self.name]

    @abc.abstractmethod
    def source_type(self) -> str:
        raise NotImplementedError

class PinnedPackage(BasePackage):

    def __init__(self) -> None:
        self._cached_metadata: Optional[ProjectPackageMetadata] = None

    def __str__(self) -> str:
        version = self.get_version()
        if not version:
            return self.name
        return '{}@{}'.format(self.name, version)

    @abc.abstractmethod
    def get_version(self) -> Optional[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def _fetch_metadata(self, project: 'Project', renderer: 'Renderer') -> ProjectPackageMetadata:
        raise NotImplementedError

    @abc.abstractmethod
    def install(self, project: 'Project', renderer: 'Renderer') -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def nice_version_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, str]:
        raise NotImplementedError

    def fetch_metadata(self, project: 'Project', renderer: 'Renderer') -> ProjectPackageMetadata:
        if not self._cached_metadata:
            self._cached_metadata = self._fetch_metadata(project, renderer)
        return self._cached_metadata

    def get_project_name(self, project: 'Project', renderer: 'Renderer') -> str:
        metadata = self.fetch_metadata(project, renderer)
        return metadata.name

    def get_installation_path(self, project: 'Project', renderer: 'Renderer') -> str:
        dest_dirname = self.get_project_name(project, renderer)
        return os.path.join(project.packages_install_path, dest_dirname)

    def get_subdirectory(self) -> Optional[str]:
        return None

    def _install(self, project: 'Project', renderer: 'Renderer') -> None:
        metadata = self.fetch_metadata(project, renderer)
        tar_name = f'{self.package}.{self.version}.tar.gz'
        tar_path = (Path(get_downloads_path()) / tar_name).resolve(strict=False)
        system.make_directory(str(tar_path.parent))
        download_url = metadata.downloads.tarball
        deps_path = project.packages_install_path
        package_name = self.get_project_name(project, renderer)
        download_untar_fn = functools.partial(self.download_and_untar, download_url, str(tar_path), deps_path, package_name)
        connection_exception_retry(download_untar_fn, 5)

    def download_and_untar(self, download_url: str, tar_path: str, deps_path: str, package_name: str) -> None:
        system.download(download_url, tar_path)
        system.untar_package(tar_path, deps_path, package_name)

SomePinned = TypeVar('SomePinned', bound=PinnedPackage)
SomeUnpinned = TypeVar('SomeUnpinned', bound='UnpinnedPackage')

class UnpinnedPackage(Generic[SomePinned], BasePackage):

    @classmethod
    @abc.abstractmethod
    def from_contract(cls, contract: 'Contract') -> 'UnpinnedPackage':
        raise NotImplementedError

    @abc.abstractmethod
    def incorporate(self, other: 'UnpinnedPackage') -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def resolved(self) -> SomePinned:
        raise NotImplementedError
