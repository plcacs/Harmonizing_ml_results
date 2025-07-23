import io
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, Any, Optional, Union, Tuple, Dict, List, Generator
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
def _load_toml(path: Union[str, Path]) -> Any:
    with open(path, 'rb') as f:
        return tomllib.load(f)

@lru_cache
def _cached_resolve(path: Path) -> Path:
    return path.resolve()

@lru_cache
def find_project_root(srcs: Sequence[str], stdin_filename: Optional[str] = None) -> Tuple[Path, str]:
    if stdin_filename is not None:
        srcs = tuple((stdin_filename if s == '-' else s for s in srcs))
    if not srcs:
        srcs = [str(_cached_resolve(Path.cwd()))]
    path_srcs = [_cached_resolve(Path(Path.cwd(), src)) for src in srcs]
    src_parents = [list(path.parents) + ([path] if path.is_dir() else []) for path in path_srcs]
    common_base = max(set.intersection(*(set(parents) for parents in src_parents)), key=lambda path: path.parts)
    for directory in (common_base, *common_base.parents):
        if (directory / '.git').exists():
            return (directory, '.git directory')
        if (directory / '.hg').is_dir():
            return (directory, '.hg directory')
        if (directory / 'pyproject.toml').is_file():
            pyproject_toml = _load_toml(directory / 'pyproject.toml')
            if 'black' in pyproject_toml.get('tool', {}):
                return (directory, 'pyproject.toml')
    return (directory, 'file system root')

def find_pyproject_toml(path_search_start: Sequence[str], stdin_filename: Optional[str] = None) -> Optional[str]:
    path_project_root, _ = find_project_root(path_search_start, stdin_filename)
    path_pyproject_toml = path_project_root / 'pyproject.toml'
    if path_pyproject_toml.is_file():
        return str(path_pyproject_toml)
    try:
        path_user_pyproject_toml = find_user_pyproject_toml()
        return str(path_user_pyproject_toml) if path_user_pyproject_toml.is_file() else None
    except (PermissionError, RuntimeError) as e:
        err(f'Ignoring user configuration directory due to {e!r}')
        return None

@mypyc_attr(patchable=True)
def parse_pyproject_toml(path_config: Union[str, Path]) -> Dict[str, Any]:
    pyproject_toml = _load_toml(path_config)
    config = pyproject_toml.get('tool', {}).get('black', {})
    config = {k.replace('--', '').replace('-', '_'): v for k, v in config.items()}
    if 'target_version' not in config:
        inferred_target_version = infer_target_version(pyproject_toml)
        if inferred_target_version is not None:
            config['target_version'] = [v.name.lower() for v in inferred_target_version]
    return config

def infer_target_version(pyproject_toml: Dict[str, Any]) -> Optional[List[TargetVersion]]:
    project_metadata = pyproject_toml.get('project', {})
    requires_python = project_metadata.get('requires-python', None)
    if requires_python is not None:
        try:
            return parse_req_python_version(requires_python)
        except InvalidVersion:
            pass
        try:
            return parse_req_python_specifier(requires_python)
        except (InvalidSpecifier, InvalidVersion):
            pass
    return None

def parse_req_python_version(requires_python: str) -> Optional[List[TargetVersion]]:
    version = Version(requires_python)
    if version.release[0] != 3:
        return None
    try:
        return [TargetVersion(version.release[1])]
    except (IndexError, ValueError):
        return None

def parse_req_python_specifier(requires_python: str) -> Optional[List[TargetVersion]]:
    specifier_set = strip_specifier_set(SpecifierSet(requires_python))
    if not specifier_set:
        return None
    target_version_map = {f'3.{v.value}': v for v in TargetVersion}
    compatible_versions = list(specifier_set.filter(target_version_map))
    if compatible_versions:
        return [target_version_map[v] for v in compatible_versions]
    return None

def strip_specifier_set(specifier_set: SpecifierSet) -> SpecifierSet:
    specifiers = []
    for s in specifier_set:
        if '*' in str(s):
            specifiers.append(s)
        elif s.operator in ['~=', '==', '>=', '===']:
            version = Version(s.version)
            stripped = Specifier(f'{s.operator}{version.major}.{version.minor}')
            specifiers.append(stripped)
        elif s.operator == '>':
            version = Version(s.version)
            if len(version.release) > 2:
                s = Specifier(f'>={version.major}.{version.minor}')
            specifiers.append(s)
        else:
            specifiers.append(s)
    return SpecifierSet(','.join((str(s) for s in specifiers)))

@lru_cache
def find_user_pyproject_toml() -> Path:
    if sys.platform == 'win32':
        user_config_path = Path.home() / '.black'
    else:
        config_root = os.environ.get('XDG_CONFIG_HOME', '~/.config')
        user_config_path = Path(config_root).expanduser() / 'black'
    return _cached_resolve(user_config_path)

@lru_cache
def get_gitignore(root: Path) -> PathSpec:
    gitignore = root / '.gitignore'
    lines = []
    if gitignore.is_file():
        with gitignore.open(encoding='utf-8') as gf:
            lines = gf.readlines()
    try:
        return PathSpec.from_lines('gitwildmatch', lines)
    except GitWildMatchPatternError as e:
        err(f'Could not parse {gitignore}: {e}')
        raise

def resolves_outside_root_or_cannot_stat(path: Path, root: Path, report: Optional[Report] = None) -> bool:
    try:
        resolved_path = _cached_resolve(path)
    except OSError as e:
        if report:
            report.path_ignored(path, f'cannot be read because {e}')
        return True
    try:
        resolved_path.relative_to(root)
    except ValueError:
        if report:
            report.path_ignored(path, f'is a symbolic link that points outside {root}')
        return True
    return False

def best_effort_relative_path(path: Path, root: Path) -> Path:
    try:
        return path.absolute().relative_to(root)
    except ValueError:
        pass
    root_parent = next((p for p in path.parents if _cached_resolve(p) == root), None)
    if root_parent is not None:
        return path.relative_to(root_parent)
    return _cached_resolve(path).relative_to(root)

def _path_is_ignored(root_relative_path: str, root: Path, gitignore_dict: Dict[Path, PathSpec]) -> bool:
    path = root / root_relative_path
    for gitignore_path, pattern in gitignore_dict.items():
        try:
            relative_path = path.relative_to(gitignore_path).as_posix()
            if path.is_dir():
                relative_path = relative_path + '/'
        except ValueError:
            break
        if pattern.match_file(relative_path):
            return True
    return False

def path_is_excluded(normalized_path: str, pattern: Optional[Pattern]) -> bool:
    match = pattern.search(normalized_path) if pattern else None
    return bool(match and match.group(0))

def gen_python_files(paths: Iterable[Path], root: Path, include: Optional[Pattern], exclude: Optional[Pattern], extend_exclude: Optional[Pattern], force_exclude: Optional[Pattern], report: Report, gitignore_dict: Optional[Dict[Path, PathSpec]], *, verbose: bool, quiet: bool) -> Generator[Path, None, None]:
    assert root.is_absolute(), f'INTERNAL ERROR: `root` must be absolute but is {root}'
    for child in paths:
        assert child.is_absolute()
        root_relative_path = child.relative_to(root).as_posix()
        if gitignore_dict and _path_is_ignored(root_relative_path, root, gitignore_dict):
            report.path_ignored(child, 'matches a .gitignore file content')
            continue
        root_relative_path = '/' + root_relative_path
        if child.is_dir():
            root_relative_path += '/'
        if path_is_excluded(root_relative_path, exclude):
            report.path_ignored(child, 'matches the --exclude regular expression')
            continue
        if path_is_excluded(root_relative_path, extend_exclude):
            report.path_ignored(child, 'matches the --extend-exclude regular expression')
            continue
        if path_is_excluded(root_relative_path, force_exclude):
            report.path_ignored(child, 'matches the --force-exclude regular expression')
            continue
        if resolves_outside_root_or_cannot_stat(child, root, report):
            continue
        if child.is_dir():
            if gitignore_dict is not None:
                new_gitignore_dict = {**gitignore_dict, root / child: get_gitignore(child)}
            else:
                new_gitignore_dict = None
            yield from gen_python_files(child.iterdir(), root, include, exclude, extend_exclude, force_exclude, report, new_gitignore_dict, verbose=verbose, quiet=quiet)
        elif child.is_file():
            if child.suffix == '.ipynb' and (not jupyter_dependencies_are_installed(warn=verbose or not quiet)):
                continue
            include_match = include.search(root_relative_path) if include else True
            if include_match:
                yield child

def wrap_stream_for_windows(f: io.TextIOWrapper) -> io.TextIOWrapper:
    try:
        from colorama.initialise import wrap_stream
    except ImportError:
        return f
    else:
        return wrap_stream(f, convert=None, strip=False, autoreset=False, wrap=True)
