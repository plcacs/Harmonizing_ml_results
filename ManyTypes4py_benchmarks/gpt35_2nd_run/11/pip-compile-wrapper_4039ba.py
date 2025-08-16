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

REQUIREMENT_RE: re.Pattern = re.compile('^([A-Z0-9][A-Z0-9._-]*[A-Z0-9])', re.IGNORECASE)
REQUIREMENTS_SOURCE_DEV: str = 'requirements-dev.in'
SCRIPT_NAME: str = os.environ.get('_SCRIPT_NAME', sys.argv[0])
REQUIREMENTS_DIR: Path = Path(__file__).parent.parent.joinpath('requirements').resolve()
SOURCE_PATHS: Dict[str, Path] = {path.relative_to(REQUIREMENTS_DIR).stem: path.resolve() for path in REQUIREMENTS_DIR.glob('*.in')}
TARGET_PATHS: Dict[str, Path] = {name: REQUIREMENTS_DIR.joinpath(name).with_suffix('.txt') for name in SOURCE_PATHS.keys()}
SOURCE_DEPENDENCIES: Dict[str, Set[str]] = {}

class TargetType(Enum):
    SOURCE = 1
    TARGET = 2
    ALL = 3

def _resolve_source_dependencies() -> None:
    ...

def _run_pip_compile(source_name: str, upgrade_all: bool = False, upgrade_packages: Optional[Set[str]] = None, verbose: bool = False, dry_run: bool = False, pre: bool = False) -> None:
    ...

def _resolve_deps(source_names: Set[str]) -> List[str]:
    ...

def _get_requirement_packages(source_name: str, where: TargetType) -> Iterable[Tuple[str, str]]:
    ...

def _get_sources_for_packages(package_names: Set[str], where: TargetType) -> Dict[str, Set[str]]:
    ...

def _get_requirement_package(source_name: str, target_package_name: str) -> Optional[str]:
    ...

def _ensure_pip_tools() -> None:
    ...

def compile_source(upgrade_all: bool = False, verbose: bool = False, dry_run: bool = False) -> None:
    ...

def upgrade_source(upgrade_package_names: Set[str], verbose: bool = False, dry_run: bool = False, pre: bool = False) -> None:
    ...

def main() -> None:
    ...

if __name__ == '__main__':
    main()
