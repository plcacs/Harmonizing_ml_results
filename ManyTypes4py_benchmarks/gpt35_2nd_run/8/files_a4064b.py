import io
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, Any, Optional, Union
from mypy_extensions import mypyc_attr
from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
if sys.version_info >= (3, 11):
    try:
        import tomllib
    except ImportError:
        if not TYPE_CHECKING:
            import tomli as tomllib
else:
    import tomli as tomllib
from black.handle_ipynb_magics import jupyter_dependencies_are_installed
from black.mode import TargetVersion
from black.output import err
from black.report import Report
if TYPE_CHECKING:
    import colorama

@lru_cache
def _load_toml(path: Path) -> Any:
    with open(path, 'rb') as f:
        return tomllib.load(f)

@lru_cache
def _cached_resolve(path: Path) -> Path:
    return path.resolve()

@lru_cache
def find_project_root(srcs: Iterable[str], stdin_filename: Optional[str] = None) -> Tuple[Path, str]:
    ...

def find_pyproject_toml(path_search_start: Iterable[str], stdin_filename: Optional[str] = None) -> Optional[str]:
    ...

@mypyc_attr(patchable=True)
def parse_pyproject_toml(path_config: Path) -> dict:
    ...

def infer_target_version(pyproject_toml: dict) -> Optional[list]:
    ...

def parse_req_python_version(requires_python: str) -> Optional[list]:
    ...

def parse_req_python_specifier(requires_python: str) -> Optional[list]:
    ...

def strip_specifier_set(specifier_set: SpecifierSet) -> SpecifierSet:
    ...

@lru_cache
def find_user_pyproject_toml() -> Path:
    ...

@lru_cache
def get_gitignore(root: Path) -> PathSpec:
    ...

def resolves_outside_root_or_cannot_stat(path: Path, root: Path, report: Optional[Report] = None) -> bool:
    ...

def best_effort_relative_path(path: Path, root: Path) -> Path:
    ...

def _path_is_ignored(root_relative_path: str, root: Path, gitignore_dict: dict) -> bool:
    ...

def path_is_excluded(normalized_path: str, pattern: Pattern) -> bool:
    ...

def gen_python_files(paths: Iterable[Path], root: Path, include: Pattern, exclude: Pattern, extend_exclude: Pattern, force_exclude: Pattern, report: Report, gitignore_dict: dict, *, verbose: bool, quiet: bool) -> Iterable[Path]:
    ...

def wrap_stream_for_windows(f: Any) -> Any:
    ...
