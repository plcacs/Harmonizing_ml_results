from typing import NamedTuple, Optional, Union, cast, List, Tuple, Any, Generator

PYPI_INSTANCE: str = 'https://pypi.org/pypi'
PYPI_TOP_PACKAGES: str = 'https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json'
INTERNAL_BLACK_REPO: str = f'{tempfile.gettempdir()}/__black'
ArchiveKind: Union[tarfile.TarFile, zipfile.ZipFile]
subprocess.run = partial(subprocess.run, check=True)

class BlackVersion(NamedTuple):
    config: Optional[str]

def get_pypi_download_url(package: str, version: Optional[str]) -> str:
    ...

def get_top_packages() -> List[str]:
    ...

def get_package_source(package: str, version: Optional[str]) -> str:
    ...

def get_archive_manager(local_file: str) -> ArchiveKind:
    ...

def get_first_archive_member(archive: ArchiveKind) -> str:
    ...

def download_and_extract(package: str, version: Optional[str], directory: Path) -> Path:
    ...

def get_package(package: str, version: Optional[str], directory: Path) -> Optional[Path]:
    ...

def download_and_extract_top_packages(directory: Path, workers: int, limit: slice) -> Generator[Path, None, None]:
    ...

def git_create_repository(repo: Path) -> None:
    ...

def git_add_and_commit(msg: str, repo: Path) -> None:
    ...

def git_switch_branch(branch: str, repo: Path, new: bool = False, from_branch: Optional[str] = None) -> None:
    ...

def init_repos(options: Namespace) -> Tuple[Path, ...]:
    ...

@lru_cache(8)
def black_runner(version: str, black_repo: Path) -> Path:
    ...

def format_repo_with_version(repo: Path, from_branch: Optional[str], black_repo: Path, black_version: BlackVersion, input_directory: Path) -> str:
    ...

def format_repos(repos: Tuple[Path, ...], options: Namespace) -> None:
    ...

def main() -> None:
    ...
