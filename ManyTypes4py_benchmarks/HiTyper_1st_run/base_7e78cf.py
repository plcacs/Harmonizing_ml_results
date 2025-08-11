import abc
import functools
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generic, List, Optional, TypeVar
from dbt.contracts.project import ProjectPackageMetadata
from dbt.events.types import DepsSetDownloadDirectory
from dbt_common.clients import system
from dbt_common.events.functions import fire_event
from dbt_common.utils.connection import connection_exception_retry
DOWNLOADS_PATH = None

def get_downloads_path() -> Union[pathlib.Path, str]:
    return DOWNLOADS_PATH

@contextmanager
def downloads_directory() -> typing.Generator[typing.Union[pathlib.Path,str]]:
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
    def name(self) -> None:
        raise NotImplementedError

    def all_names(self) -> list:
        return [self.name]

    @abc.abstractmethod
    def source_type(self) -> None:
        raise NotImplementedError

class PinnedPackage(BasePackage):

    def __init__(self) -> None:
        self._cached_metadata = None

    def __str__(self) -> str:
        version = self.get_version()
        if not version:
            return self.name
        return '{}@{}'.format(self.name, version)

    @abc.abstractmethod
    def get_version(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _fetch_metadata(self, project: Union[str, typing.Sequence[str], typing.Iterable[str]], renderer: Union[str, typing.Sequence[str], typing.Iterable[str]]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def install(self, project: Union[str, typing.Iterable[str], typing.Sequence[str]], renderer: Union[str, typing.Iterable[str], typing.Sequence[str]]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def nice_version_name(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def to_dict(self) -> None:
        raise NotImplementedError

    def fetch_metadata(self, project: Union[str, dict[str, typing.Any], list[dict]], renderer: Union[str, dict[str, typing.Any], list[dict]]) -> Union[dict, list, typing.Type]:
        if not self._cached_metadata:
            self._cached_metadata = self._fetch_metadata(project, renderer)
        return self._cached_metadata

    def get_project_name(self, project: Union[list[dict], dict[str, typing.Any], dict], renderer: Union[list[dict], dict[str, typing.Any], dict]):
        metadata = self.fetch_metadata(project, renderer)
        return metadata.name

    def get_installation_path(self, project: Union[str, dict[str, typing.Any], None], renderer: Union[str, None, pathlib.Path]) -> str:
        dest_dirname = self.get_project_name(project, renderer)
        return os.path.join(project.packages_install_path, dest_dirname)

    def get_subdirectory(self) -> None:
        return None

    def _install(self, project: str, renderer: Union[str, dict[str, typing.Any], list[dict]]) -> None:
        metadata = self.fetch_metadata(project, renderer)
        tar_name = f'{self.package}.{self.version}.tar.gz'
        tar_path = (Path(get_downloads_path()) / tar_name).resolve(strict=False)
        system.make_directory(str(tar_path.parent))
        download_url = metadata.downloads.tarball
        deps_path = project.packages_install_path
        package_name = self.get_project_name(project, renderer)
        download_untar_fn = functools.partial(self.download_and_untar, download_url, str(tar_path), deps_path, package_name)
        connection_exception_retry(download_untar_fn, 5)

    def download_and_untar(self, download_url: Union[str, pathlib2.Path], tar_path: Union[str, pathlib.Path], deps_path: Union[str, pathlib.Path, None], package_name: Union[str, pathlib.Path, None]) -> None:
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

class UnpinnedPackage(Generic[SomePinned], BasePackage):

    @classmethod
    @abc.abstractmethod
    def from_contract(cls: list[str], contract: list[str]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def incorporate(self, other: typing.AbstractSet) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def resolved(self) -> None:
        raise NotImplementedError