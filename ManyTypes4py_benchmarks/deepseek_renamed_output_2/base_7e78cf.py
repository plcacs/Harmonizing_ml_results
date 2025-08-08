import abc
import functools
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generic, List, Optional, TypeVar, Any, Iterator, Type, cast
from dbt.contracts.project import ProjectPackageMetadata
from dbt.events.types import DepsSetDownloadDirectory
from dbt_common.clients import system
from dbt_common.events.functions import fire_event
from dbt_common.utils.connection import connection_exception_retry

DOWNLOADS_PATH: Optional[str] = None


def func_luy4ji87() -> Optional[str]:
    return DOWNLOADS_PATH


@contextmanager
def func_5ncr68n4() -> Iterator[str]:
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
    @property
    @abc.abstractmethod
    def func_ptvdgya6(self) -> str:
        raise NotImplementedError

    def func_zkuwhurt(self) -> List[str]:
        return [self.name]

    @abc.abstractmethod
    def func_7364p1f0(self) -> Any:
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
    def func_1frwgcsw(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def func_gyzpbj3b(self, project: Any, renderer: Any) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def func_65ny9bip(self, project: Any, renderer: Any) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def func_lcc1wgrl(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def func_wi9xuumk(self) -> str:
        raise NotImplementedError

    def func_fz8ux7d9(self, project: Any, renderer: Any) -> ProjectPackageMetadata:
        if not self._cached_metadata:
            self._cached_metadata = self._fetch_metadata(project, renderer)
        return cast(ProjectPackageMetadata, self._cached_metadata)

    def func_3y5azxm1(self, project: Any, renderer: Any) -> str:
        metadata = self.fetch_metadata(project, renderer)
        return metadata.name

    def func_i8tbbol7(self, project: Any, renderer: Any) -> str:
        dest_dirname = self.get_project_name(project, renderer)
        return os.path.join(project.packages_install_path, dest_dirname)

    def func_jh4jka1m(self) -> Optional[str]:
        return None

    def func_kl454kdt(self, project: Any, renderer: Any) -> None:
        metadata = self.fetch_metadata(project, renderer)
        tar_name = f'{self.package}.{self.version}.tar.gz'
        tar_path = (Path(func_luy4ji87()) / tar_name).resolve(strict=False)
        system.make_directory(str(tar_path.parent))
        download_url = metadata.downloads.tarball
        deps_path = project.packages_install_path
        package_name = self.get_project_name(project, renderer)
        download_untar_fn = functools.partial(self.download_and_untar,
            download_url, str(tar_path), deps_path, package_name)
        connection_exception_retry(download_untar_fn, 5)

    def func_to58t4w5(self, download_url: str, tar_path: str, deps_path: str, package_name: str) -> None:
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
    def func_zltru0vu(cls, contract: Any) -> 'UnpinnedPackage[SomePinned]':
        raise NotImplementedError

    @abc.abstractmethod
    def func_vwl1s7sz(self, other: Any) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def func_jwah3akv(self) -> SomePinned:
        raise NotImplementedError
