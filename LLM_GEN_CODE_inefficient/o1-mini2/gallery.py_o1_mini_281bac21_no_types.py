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
from urllib.request import urlopen, urlretrieve
PYPI_INSTANCE: str = 'https://pypi.org/pypi'
PYPI_TOP_PACKAGES: tuple[str, ...] = (
    'https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json'
    ,)
INTERNAL_BLACK_REPO: str = f'{tempfile.gettempdir()}/__black'
ArchiveKind = Union[tarfile.TarFile, zipfile.ZipFile]
subprocess.run = partial(subprocess.run, check=True)


class BlackVersion(NamedTuple):
    version: str
    config: Optional[str] = None


def get_pypi_download_url(package, version):
    with urlopen(PYPI_INSTANCE + f'/{package}/json') as page:
        metadata: dict = json.load(page)
    if version is None:
        sources: list[dict] = metadata['urls']
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
    return cast(str, source['url'])


def get_top_packages():
    with urlopen(PYPI_TOP_PACKAGES) as page:
        result: dict = json.load(page)
    return [package['project'] for package in result['rows']]


def get_package_source(package, version):
    if package == 'cpython':
        if version is None:
            version = 'main'
        return f'https://github.com/python/cpython/archive/{version}.zip'
    elif package == 'pypy':
        if version is None:
            version = 'branch/default'
        return (
            f'https://foss.heptapod.net/pypy/pypy/repository/{version}/archive.tar.bz2'
            )
    else:
        return get_pypi_download_url(package, version)


def get_archive_manager(local_file):
    if tarfile.is_tarfile(local_file):
        return tarfile.open(local_file)
    elif zipfile.is_zipfile(local_file):
        return zipfile.ZipFile(local_file)
    else:
        raise ValueError('Unknown archive kind.')


def get_first_archive_member(archive):
    if isinstance(archive, tarfile.TarFile):
        return archive.getnames()[0]
    elif isinstance(archive, zipfile.ZipFile):
        return archive.namelist()[0]


def download_and_extract(package, version, directory):
    source: str = get_package_source(package, version)
    local_file: str
    _, _ = urlretrieve(source, directory / f'{package}-src')
    with get_archive_manager(local_file) as archive:
        archive.extractall(path=directory)
        result_dir: str = get_first_archive_member(archive)
    return directory / result_dir


def get_package(package, version, directory):
    try:
        return download_and_extract(package, version, directory)
    except Exception:
        print(f'Caught an exception while downloading {package}.')
        traceback.print_exc()
        return None


DEFAULT_SLICE: slice = slice(None)


def download_and_extract_top_packages(directory, workers=8, limit=DEFAULT_SLICE
    ):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        bound_downloader = partial(get_package, version=None, directory=
            directory)
        for package in executor.map(bound_downloader, get_top_packages()[limit]
            ):
            if package is not None:
                yield package


def git_create_repository(repo):
    subprocess.run(['git', 'init'], cwd=repo)
    git_add_and_commit(msg='Initial commit', repo=repo)


def git_add_and_commit(msg, repo):
    subprocess.run(['git', 'add', '.'], cwd=repo)
    subprocess.run(['git', 'commit', '-m', msg, '--allow-empty'], cwd=repo)


def git_switch_branch(branch, repo, new=False, from_branch=None):
    args: list[Union[str, Path]] = ['git', 'checkout']
    if new:
        args.append('-b')
    args.append(branch)
    if from_branch:
        args.append(from_branch)
    subprocess.run(args, cwd=repo)


def init_repos(options):
    options.output.mkdir(exist_ok=True)
    if options.top_packages:
        source_directories: tuple[Path, ...] = tuple(
            download_and_extract_top_packages(directory=options.output,
            workers=options.workers, limit=slice(None, options.top_packages)))
    else:
        source_directories = download_and_extract(package=options.
            pypi_package, version=options.version, directory=options.output),
    for source_directory in source_directories:
        git_create_repository(source_directory)
    if options.black_repo is None:
        subprocess.run(['git', 'clone', 'https://github.com/psf/black.git',
            INTERNAL_BLACK_REPO], cwd=options.output)
        options.black_repo = Path(INTERNAL_BLACK_REPO)
    return source_directories


@lru_cache(maxsize=8)
def black_runner(version, black_repo):
    directory = tempfile.TemporaryDirectory()
    venv.create(directory.name, with_pip=True)
    python: Path = Path(directory.name) / 'bin' / 'python'
    subprocess.run([python, '-m', 'pip', 'install', '-e', black_repo])
    atexit.register(directory.cleanup)
    return python


def format_repo_with_version(repo, from_branch, black_repo, black_version,
    input_directory):
    current_branch: str = f'black-{black_version.version}'
    git_switch_branch(black_version.version, repo=black_repo)
    git_switch_branch(current_branch, repo=repo, new=True, from_branch=
        from_branch)
    format_cmd: list[Union[Path, str]] = [black_runner(black_version.
        version, black_repo), (black_repo / 'black.py').resolve(), '.']
    if black_version.config:
        format_cmd.extend(['--config', input_directory / black_version.config])
    subprocess.run(format_cmd, cwd=repo, check=False)
    git_add_and_commit(f'Format with black:{black_version.version}', repo=repo)
    return current_branch


def format_repos(repos, options):
    black_versions: tuple[BlackVersion, ...] = tuple(BlackVersion(*version.
        split(':')) for version in options.versions)
    for repo in repos:
        from_branch: Optional[str] = None
        for black_version in black_versions:
            from_branch = format_repo_with_version(repo=repo, from_branch=
                from_branch, black_repo=options.black_repo, black_version=
                black_version, input_directory=options.input)
        git_switch_branch('main', repo=repo)
    git_switch_branch('main', repo=options.black_repo)


def main():
    parser = ArgumentParser(description=
        """Black Gallery is a script that
    automates the process of applying different Black versions to a selected
    PyPI package and seeing the results between versions."""
        )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--pypi-package', help='PyPI package to download.'
        )
    group.add_argument('-t', '--top-packages', help=
        'Top n PyPI packages to download.', type=int)
    parser.add_argument('-b', '--black-repo', help=
        "Black's Git repository.", type=Path)
    parser.add_argument('-v', '--version', help=
        'Version for given PyPI package. Will be discarded if used with -t option.'
        )
    parser.add_argument('-w', '--workers', help=
        'Maximum number of threads to download with at the same time. Will be discarded if used with -p option.'
        , type=int)
    parser.add_argument('-i', '--input', default=Path('/input'), type=Path,
        help='Input directory to read configuration.')
    parser.add_argument('-o', '--output', default=Path('/output'), type=
        Path, help='Output directory to download and put result artifacts.')
    parser.add_argument('versions', nargs='*', default=('main',), help='')
    options: Namespace = parser.parse_args()
    repos: tuple[Path, ...] = init_repos(options)
    format_repos(repos, options)


if __name__ == '__main__':
    main()
