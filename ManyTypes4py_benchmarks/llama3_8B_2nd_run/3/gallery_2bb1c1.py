import atexit
import json
import subprocess
import tarfile
import tempfile
import traceback
import venv
import zipfile
from argparse import ArgumentParser, Namespace
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from typing import NamedTuple, Optional, Union, cast

PYPI_INSTANCE: str = 'https://pypi.org/pypi'
PYPI_TOP_PACKAGES: str = 'https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json'
INTERNAL_BLACK_REPO: str = f'{tempfile.gettempdir()}/__black'

ArchiveKind: Union[tarfile.TarFile, zipfile.ZipFile] = Union[tarfile.TarFile, zipfile.ZipFile]
subprocess.run: partial[subprocess.run, check=True] = partial(subprocess.run, check=True)

class BlackVersion(NamedTuple):
    config: Optional[str] = None

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

def git_switch_branch(branch: str, repo: Path, new: bool, from_branch: Optional[str]) -> None:
    ...

def init_repos(options: Namespace) -> Tuple[Path, ...]:
    ...

def format_repo_with_version(repo: Path, from_branch: Optional[str], black_repo: Path, black_version: BlackVersion, input_directory: Path) -> str:
    ...

def format_repos(repos: Tuple[Path, ...], options: Namespace) -> None:
    ...

def main() -> None:
    parser: ArgumentParser = ArgumentParser(description='Black Gallery is a script that\n    automates the process of applying different Black versions to a selected\n    PyPI package and seeing the results between versions.')
    group: ArgumentGroup = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--pypi-package', help='PyPI package to download.')
    group.add_argument('-t', '--top-packages', help='Top n PyPI packages to download.', type=int)
    parser.add_argument('-b', '--black-repo', help="Black's Git repository.", type=Path)
    parser.add_argument('-v', '--version', help='Version for given PyPI package. Will be discarded if used with -t option.')
    parser.add_argument('-w', '--workers', help='Maximum number of threads to download with at the same time. Will be discarded if used with -p option.')
    parser.add_argument('-i', '--input', default=Path('/input'), type=Path, help='Input directory to read configuration.')
    parser.add_argument('-o', '--output', default=Path('/output'), type=Path, help='Output directory to download and put result artifacts.')
    parser.add_argument('versions', nargs='*', default=('main',), help='')
    options: Namespace = parser.parse_args()
    repos: Tuple[Path, ...] = init_repos(options)
    format_repos(repos, options)
