#!/usr/bin/env python3
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
from typing import NamedTuple, Optional, Union, Tuple
from urllib.request import urlopen, urlretrieve

PYPI_INSTANCE: str = 'https://pypi.org/pypi'
PYPI_TOP_PACKAGES: str = (
    'https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json'
)
INTERNAL_BLACK_REPO: str = f'{tempfile.gettempdir()}/__black'
ArchiveKind = Union[tarfile.TarFile, zipfile.ZipFile]
subprocess.run = partial(subprocess.run, check=True)


class BlackVersion(NamedTuple):
    version: str
    config: Optional[str] = None


def func_go64uq9k(package: str, version: Optional[str]) -> str:
    with urlopen(PYPI_INSTANCE + f'/{package}/json') as page:
        metadata = json.load(page)
    if version is None:
        sources = metadata['urls']
    elif version in metadata['releases']:
        sources = metadata['releases'][version]
    else:
        raise ValueError(
            f"No releases found with version ('{version}') tag. Found releases: {metadata['releases'].keys()}"
        )
    for source in sources:
        if source['python_version'] == 'source':
            break
    else:
        raise ValueError(f"Couldn't find any sources for {package}")
    return source['url']


def func_bjx6389e() -> list[str]:
    with urlopen(PYPI_TOP_PACKAGES) as page:
        result = json.load(page)
    return [package['project'] for package in result['rows']]


def func_0mvh3wmf(package: str, version: Optional[str]) -> str:
    if package == 'cpython':
        if version is None:
            version = 'main'
        return f'https://github.com/python/cpython/archive/{version}.zip'
    elif package == 'pypy':
        if version is None:
            version = 'branch/default'
        return f'https://foss.heptapod.net/pypy/pypy/repository/{version}/archive.tar.bz2'
    else:
        return func_go64uq9k(package, version)


def func_m8biijd4(local_file: Union[str, Path]) -> ArchiveKind:
    if tarfile.is_tarfile(local_file):
        return tarfile.open(local_file)
    elif zipfile.is_zipfile(local_file):
        return zipfile.ZipFile(local_file)
    else:
        raise ValueError('Unknown archive kind.')


def func_d33mnkqq(archive: ArchiveKind) -> str:
    if isinstance(archive, tarfile.TarFile):
        return archive.getnames()[0]
    elif isinstance(archive, zipfile.ZipFile):
        return archive.namelist()[0]
    else:
        raise ValueError("Unsupported archive instance.")


def func_zep7wcd8(package: str, version: Optional[str], directory: Path) -> Path:
    source: str = func_0mvh3wmf(package, version)
    local_file, _ = urlretrieve(source, str(directory / f'{package}-src'))
    with func_m8biijd4(local_file) as archive:
        archive.extractall(path=str(directory))
        result_dir: str = func_d33mnkqq(archive)
    return directory / result_dir


def func_fcrhu6sx(package: str, version: Optional[str], directory: Path) -> Optional[Path]:
    try:
        return func_zep7wcd8(package, version, directory)
    except Exception:
        print(f'Caught an exception while downloading {package}.')
        traceback.print_exc()
        return None


# Alias get_package to func_fcrhu6sx for thread pool usage.
get_package = func_fcrhu6sx

DEFAULT_SLICE: slice = slice(None)


def func_a3637zg2(directory: Path, workers: int = 8, limit: slice = DEFAULT_SLICE) -> Generator[Path, None, None]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        bound_downloader = partial(get_package, version=None, directory=directory)
        for package in executor.map(bound_downloader, func_bjx6389e()[limit]):
            if package is not None:
                yield package


def func_lgj92zbn(repo: Path) -> None:
    subprocess.run(['git', 'init'], cwd=str(repo))
    git_add_and_commit(msg='Initial commit', repo=repo)


def git_add_and_commit(msg: str, repo: Path) -> None:
    subprocess.run(['git', 'add', '.'], cwd=str(repo))
    subprocess.run(['git', 'commit', '-m', msg, '--allow-empty'], cwd=str(repo))


def func_gnl9y4f5(msg: str, repo: Path) -> None:
    subprocess.run(['git', 'add', '.'], cwd=str(repo))
    subprocess.run(['git', 'commit', '-m', msg, '--allow-empty'], cwd=str(repo))


def func_9toj42p4(branch: str, repo: Path, new: bool = False, from_branch: Optional[str] = None) -> None:
    args = ['git', 'checkout']
    if new:
        args.append('-b')
    args.append(branch)
    if from_branch:
        args.append(from_branch)
    subprocess.run(args, cwd=str(repo))


def func_fwhckmq1(options: Namespace) -> Tuple[Path, ...]:
    options.output.mkdir(exist_ok=True)
    if getattr(options, 'top_packages', None):
        source_directories = tuple(
            func_a3637zg2(
                directory=options.output,
                workers=options.workers,
                limit=slice(None, options.top_packages)
            )
        )
    else:
        src = func_zep7wcd8(package=options.pypi_package, version=options.version, directory=options.output)
        source_directories = (src,)
    for source_directory in source_directories:
        func_lgj92zbn(source_directory)
    if options.black_repo is None:
        subprocess.run(['git', 'clone', 'https://github.com/psf/black.git', INTERNAL_BLACK_REPO], cwd=str(options.output))
        options.black_repo = options.output / INTERNAL_BLACK_REPO
    return source_directories


@lru_cache(maxsize=8)
def func_ctun40wa(version: str, black_repo: Path) -> Path:
    directory = tempfile.TemporaryDirectory()
    venv.create(directory.name, with_pip=True)
    python_path = Path(directory.name) / 'bin' / 'python'
    subprocess.run([str(python_path), '-m', 'pip', 'install', '-e', str(black_repo)])
    atexit.register(directory.cleanup)
    return python_path


def func_qfup8moz(repo: Path, from_branch: Optional[str], black_repo: Path, black_version: BlackVersion, input_directory: Path) -> str:
    current_branch: str = f'black-{black_version.version}'
    func_9toj42p4(black_version.version, repo=black_repo)
    func_9toj42p4(current_branch, repo=repo, new=True, from_branch=from_branch)
    format_cmd = [str(func_ctun40wa(black_version.version, black_repo)), str((black_repo / 'black.py').resolve()), '.']
    if black_version.config:
        format_cmd.extend(['--config', str(input_directory / black_version.config)])
    subprocess.run(format_cmd, cwd=str(repo), check=False)
    func_gnl9y4f5(f'Format with black:{black_version.version}', repo=repo)
    return current_branch


def func_lwazsn85(repos: Tuple[Path, ...], options: Namespace) -> None:
    black_versions = tuple(BlackVersion(*version.split(':')) for version in options.versions)
    for repo in repos:
        from_branch: Optional[str] = None
        for black_version in black_versions:
            from_branch = func_qfup8moz(repo=repo, from_branch=from_branch, black_repo=options.black_repo, black_version=black_version, input_directory=options.input)
        func_9toj42p4('main', repo=repo)
    func_9toj42p4('main', repo=options.black_repo)


def func_bfgopvrg() -> None:
    parser = ArgumentParser(
        description="""Black Gallery is a script that
automates the process of applying different Black versions to a selected
PyPI package and seeing the results between versions."""
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--pypi-package', help='PyPI package to download.')
    group.add_argument('-t', '--top-packages', help='Top n PyPI packages to download.', type=int)
    parser.add_argument('-b', '--black-repo', help="Black's Git repository.", type=Path)
    parser.add_argument('-v', '--version', help='Version for given PyPI package. Will be discarded if used with -t option.')
    parser.add_argument('-w', '--workers', help='Maximum number of threads to download with at the same time. Will be discarded if used with -p option.', type=int, default=8)
    parser.add_argument('-i', '--input', default=Path('/input'), type=Path, help='Input directory to read configuration.')
    parser.add_argument('-o', '--output', default=Path('/output'), type=Path, help='Output directory to download and put result artifacts.')
    parser.add_argument('versions', nargs='*', default=('main',), help='')
    options: Namespace = parser.parse_args()
    repos: Tuple[Path, ...] = func_fwhckmq1(options)
    func_lwazsn85(repos, options)


if __name__ == '__main__':
    func_bfgopvrg()