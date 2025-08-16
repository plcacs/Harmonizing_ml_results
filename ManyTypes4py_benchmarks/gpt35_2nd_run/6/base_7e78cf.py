from typing import TypeVar

DOWNLOADS_PATH: str

def get_downloads_path() -> str:
    return DOWNLOADS_PATH

class BasePackage(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def all_names(self) -> List[str]:
        return [self.name]

    @abc.abstractmethod
    def source_type(self) -> str:
        raise NotImplementedError

class PinnedPackage(BasePackage):

    def __str__(self) -> str:
        ...

    @abc.abstractmethod
    def get_version(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _fetch_metadata(self, project, renderer):
        raise NotImplementedError

    @abc.abstractmethod
    def install(self, project, renderer):
        raise NotImplementedError

    @abc.abstractmethod
    def nice_version_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, str]:
        raise NotImplementedError

    def fetch_metadata(self, project, renderer):
        ...

    def get_project_name(self, project, renderer) -> str:
        ...

    def get_installation_path(self, project, renderer) -> str:
        ...

    def get_subdirectory(self) -> Optional[str]:
        return None

    def _install(self, project, renderer):
        ...

    def download_and_untar(self, download_url, tar_path, deps_path, package_name):
        ...

SomePinned = TypeVar('SomePinned', bound=PinnedPackage)
SomeUnpinned = TypeVar('SomeUnpinned', bound='UnpinnedPackage')

class UnpinnedPackage(Generic[SomePinned], BasePackage):

    @classmethod
    @abc.abstractmethod
    def from_contract(cls, contract) -> SomeUnpinned:
        raise NotImplementedError

    @abc.abstractmethod
    def incorporate(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def resolved(self) -> bool:
        raise NotImplementedError
