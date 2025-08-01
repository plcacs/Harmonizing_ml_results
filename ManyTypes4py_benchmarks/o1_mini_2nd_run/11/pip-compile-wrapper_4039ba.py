"""
Helper utility to compile / upgrade requirements files from templates.

This only manages dependencies between requirements sources.
The actual compiling is delegated to ``pip-compile`` from the ``pip-tools` package.

NOTE: This utility *must only* use stdlib imports in order to be runnable even
      before the dev requirements are installed.
"""
import os
import re
import shlex
import subprocess
import sys
from argparse import ArgumentParser
from enum import Enum
from itertools import chain, groupby, repeat
from operator import itemgetter
from pathlib import Path
from shutil import which
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

REQUIREMENT_RE: re.Pattern = re.compile(
    r'^([A-Z0-9][A-Z0-9._-]*[A-Z0-9])', re.IGNORECASE
)
REQUIREMENTS_SOURCE_DEV: str = 'requirements-dev.in'
SCRIPT_NAME: str = os.environ.get('_SCRIPT_NAME', sys.argv[0])
REQUIREMENTS_DIR: Path = Path(__file__).parent.parent.joinpath('requirements').resolve()
SOURCE_PATHS: Dict[str, Path] = {
    path.relative_to(REQUIREMENTS_DIR).stem: path.resolve()
    for path in REQUIREMENTS_DIR.glob('*.in')
}
TARGET_PATHS: Dict[str, Path] = {
    name: REQUIREMENTS_DIR.joinpath(name).with_suffix('.txt')
    for name in SOURCE_PATHS.keys()
}
SOURCE_DEPENDENCIES: Dict[str, Set[str]] = {}


class TargetType(Enum):
    SOURCE = 1
    TARGET = 2
    ALL = 3


def _resolve_source_dependencies() -> None:
    """Determine direct dependencies between requirements files

    Dependencies of the form ``-r <other-file>`` are recognized.
    """
    for source_name, source_path in SOURCE_PATHS.items():
        source_path = source_path.resolve()
        SOURCE_DEPENDENCIES[source_name] = set()
        target_dir = source_path.parent
        with source_path.open('rt') as target_file:
            for line in target_file:
                line = line.strip()
                if line.startswith('-r'):
                    required = (
                        target_dir.joinpath(line.lstrip('-r').strip())
                        .resolve()
                        .relative_to(REQUIREMENTS_DIR)
                        .stem
                    )
                    SOURCE_DEPENDENCIES[source_name].add(required)


_resolve_source_dependencies()


def _run_pip_compile(
    source_name: str,
    upgrade_all: bool = False,
    upgrade_packages: Optional[Set[str]] = None,
    verbose: bool = False,
    dry_run: bool = False,
    pre: bool = False
) -> None:
    """Run pip-compile with the given parameters

    This automatically makes sure that packages listed in ``upgrade_packages`` are only passed
    for requirement files that already contain this package either in the source or the target.
    This is necessary since pip-compile will otherwise unconditionally add that package to the
    output.
    """
    assert_msg = 'Only one of `upgrade_all` or `upgrade_packages` may be given.'
    assert not (upgrade_all and upgrade_packages), assert_msg
    pip_compile_exe: Optional[str] = which('pip-compile')
    if not pip_compile_exe:
        raise RuntimeError("pip-compile missing. This shouldn't happen.")
    if not upgrade_packages:
        upgrade_packages = set()
    working_path: Path = Path.cwd()
    source_path: Path = SOURCE_PATHS[source_name]
    target_path: Path = TARGET_PATHS[source_name]
    upgrade_packages_cmd: List[str] = []
    if upgrade_packages:
        packages_in_target: Set[str] = {
            package_name
            for package_name, _ in _get_requirement_packages(source_name, TargetType.ALL)
        }
        upgrade_packages_cmd = list(
            chain.from_iterable(
                zip(
                    repeat('--upgrade-package'),
                    upgrade_packages.intersection(packages_in_target)
                )
            )
        )
    upgrade_all_cmd: List[str] = []
    if upgrade_all:
        upgrade_all_cmd = ['--upgrade']
    dry_run_cmd: List[str] = ['--dry-run'] if dry_run else []
    pre_cmd: List[str] = ['--pre'] if pre else []
    command: List[str] = [
        pip_compile_exe,
        '--verbose' if verbose else '--quiet',
        *dry_run_cmd,
        *pre_cmd,
        '--no-emit-index-url',
        *upgrade_packages_cmd,
        *upgrade_all_cmd,
        '--output-file',
        str(target_path),
        str(source_path.relative_to(working_path))
    ]
    print(f'Compiling {source_path.name}...', end='', flush=True)
    if verbose:
        print(f'\nRunning command: {" ".join((shlex.quote(c) for c in command))}')
    env: Dict[str, str] = os.environ.copy()
    env['CUSTOM_COMPILE_COMMAND'] = "'requirements/deps compile' (for details see requirements/README)"
    process: subprocess.CompletedProcess = subprocess.run(
        command,
        capture_output=not verbose,
        cwd=str(source_path.parent),
        env=env
    )
    if process.returncode == 0:
        print('\x08\x08 Success.')
        return
    print('\x08\x08 Error!')
    if not verbose:
        if process.stdout:
            print(process.stdout.decode())
        if process.stderr:
            print(process.stderr.decode())
    process.check_returncode()


def _resolve_deps(source_names: Iterable[str]) -> List[str]:
    """Partially order source_names based on their dependencies

    Raises an Exception if not possible.
    The resulting list has the following property: Each entry does not depend on a later entry.
    """
    requirements: Dict[str, Set[str]] = {
        source: dependencies.intersection(source_names)
        for source, dependencies in SOURCE_DEPENDENCIES.items()
        if source in source_names
    }
    solution: List[str] = []
    while requirements:
        satisfied: Set[str] = {
            source for source, targets in requirements.items() if not targets
        }
        if not satisfied:
            raise RuntimeError(f'Missing dependencies or circular dependency in: {requirements}')
        for source in satisfied:
            del requirements[source]
        for dependencies in requirements.values():
            dependencies -= satisfied
        solution.extend(satisfied)
    return solution


def _get_requirement_packages(
    source_name: str,
    where: TargetType = TargetType.SOURCE
) -> Iterator[Tuple[str, str]]:
    if where is TargetType.SOURCE:
        source_paths: List[Optional[Path]] = [SOURCE_PATHS.get(source_name)]
    elif where is TargetType.TARGET:
        source_paths = [TARGET_PATHS.get(source_name)]
    elif where is TargetType.ALL:
        source_paths = [
            SOURCE_PATHS.get(source_name),
            TARGET_PATHS.get(source_name)
        ]
    else:
        raise ValueError("Invalid 'where'")
    filtered_source_paths: List[Path] = [
        source_path for source_path in source_paths if source_path
    ]
    if not filtered_source_paths or not all(path.exists() for path in filtered_source_paths):
        yield from []
    for source_path in filtered_source_paths:
        with source_path.open('rt') as source_file:
            for line in source_file:
                line, *_ = line.strip().partition('#')
                line = line.strip()
                if not line or line.startswith('-'):
                    continue
                match = REQUIREMENT_RE.search(line)
                if match:
                    yield (match.group(1), line)


def _get_sources_for_packages(
    package_names: Set[str],
    where: TargetType
) -> Dict[str, Set[str]]:
    """Return source and / or target files concerned by packages"""
    package_to_source: List[Tuple[str, str]] = [
        (package_name, source_name)
        for source_name in SOURCE_PATHS.keys()
        for package_name, _ in _get_requirement_packages(source_name, where)
        if package_name in package_names
    ]
    return {
        key: {source_name for _, source_name in group}
        for key, group in groupby(
            sorted(package_to_source, key=itemgetter(0)),
            key=itemgetter(0)
        )
    }


def _get_requirement_package(
    source_name: str,
    target_package_name: str
) -> Optional[str]:
    for package_name, req_line in _get_requirement_packages(source_name):
        if package_name == target_package_name:
            return req_line
    return None


def _ensure_pip_tools() -> None:
    if not which('pip-compile'):
        print('pip-tools not available.')
        pip_tools_req: Optional[str] = _get_requirement_package(
            REQUIREMENTS_SOURCE_DEV.replace('.in', ''),
            'pip-tools'
        )
        if not pip_tools_req:
            raise RuntimeError(f"Package 'pip-tools' not found in {REQUIREMENTS_SOURCE_DEV}")
        print(f'Installing {pip_tools_req}...', end='', flush=True)
        process: subprocess.CompletedProcess = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', pip_tools_req],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        process.check_returncode()
        print('\x08\x08 Done.')


def compile_source(
    upgrade_all: bool = False,
    verbose: bool = False,
    dry_run: bool = False
) -> None:
    for source_name in _resolve_deps(SOURCE_PATHS.keys()):
        _run_pip_compile(
            source_name,
            upgrade_all=upgrade_all,
            verbose=verbose,
            dry_run=dry_run
        )


def upgrade_source(
    upgrade_package_names: Set[str],
    verbose: bool = False,
    dry_run: bool = False,
    pre: bool = False
) -> None:
    packages_to_sources: Dict[str, Set[str]] = _get_sources_for_packages(
        upgrade_package_names, TargetType.ALL
    )
    _newline: str = '\n  - '
    missing_packages: Set[str] = upgrade_package_names - packages_to_sources.keys()
    if missing_packages:
        print(
            f"Some of the given packages were not found in either source or target files.\n"
            f"Please check that the packages are spelled correctly.\n"
            f"If any of these packages were newly added to any of the source files you need to run '{SCRIPT_NAME} compile' first.\n"
            f"Missing package(s):\n  - {_newline.join(missing_packages)}"
        )
        sys.exit(1)
    grouped_packages_to_sources: List[Tuple[Set[str], str]] = [
        (set((package_name for package_name, _ in group)), key)
        for key, group in groupby(
            sorted(packages_to_sources.items(), key=itemgetter(1)),
            key=itemgetter(1)
        )
    ]
    for package_names, source_names in grouped_packages_to_sources:
        print(f'Upgrading package(s):\n  - {_newline.join(package_names)}')
        for source_name in _resolve_deps(source_names):
            _run_pip_compile(
                source_name,
                upgrade_packages=package_names,
                verbose=verbose,
                dry_run=dry_run,
                pre=pre
            )


def main() -> None:
    parser: ArgumentParser = ArgumentParser(prog=SCRIPT_NAME)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-n', '--dry-run', action='store_true', default=False)
    commands = parser.add_subparsers(title='Sub-commands', required=True, dest='command')
    commands.add_parser(
        'compile',
        help='Compile source files. Keep current versions unless changed requirements force newer versions.'
    )
    upgrade_parser = commands.add_parser(
        'upgrade',
        help='Compile source files and upgrade package versions. Optionally specify package names to upgrade only those.'
    )
    upgrade_parser.add_argument(
        '--pre',
        action='store_true',
        default=False,
        help='Use pre-release versions of packages if available.'
    )
    upgrade_parser.add_argument(
        'packages',
        metavar='package',
        nargs='*'
    )
    parsed = parser.parse_args()
    _ensure_pip_tools()
    if parsed.command == 'compile':
        compile_source(
            verbose=parsed.verbose,
            dry_run=parsed.dry_run
        )
    elif parsed.command == 'upgrade':
        packages: Set[str] = set(parsed.packages)
        if not packages:
            resp: str = input('Are you sure you want to upgrade ALL packages? [y/N] ')
            if resp.lower() != 'y':
                print('Aborting')
                sys.exit(1)
            compile_source(
                upgrade_all=True,
                verbose=parsed.verbose,
                dry_run=parsed.dry_run
            )
        else:
            if parsed.pre:
                print(
                    "Warning: Using the '--pre' option can cause unintended upgrades to prerelease versions of "
                    "unrelated packages. This is due to constraints in the underlying tools (pip-compile / pip) "
                    "that don't currently allow constraining pre-releases to only specific packages.\n"
                    "Please carefully inspect the generated output files!"
                )
            upgrade_source(
                packages,
                verbose=parsed.verbose,
                dry_run=parsed.dry_run,
                pre=parsed.pre
            )


if __name__ == '__main__':
    main()
