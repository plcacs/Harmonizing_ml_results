"""A collection of CLI commands for working with Kedro micro-packages."""
from __future__ import annotations
import logging
import re
import shutil
import sys
import tarfile
import tempfile
import toml
from importlib import import_module
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional, List, Tuple, Set, Dict, Union, cast
import click
from omegaconf import OmegaConf
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from rope.base.project import Project
from rope.contrib import generate
from rope.refactor.move import MoveModule
from rope.refactor.rename import Rename
from build.util import project_wheel_metadata
from kedro.framework.cli.pipeline import _assert_pkg_name_ok, _check_pipeline_name, _get_artifacts_to_package, _sync_dirs
from kedro.framework.cli.utils import KedroCliError, _clean_pycache, call, command_with_verbosity, env_option, python_call

if TYPE_CHECKING:
    from kedro.framework.startup import ProjectMetadata
    from importlib_metadata import PackageMetadata
    from collections.abc import Iterable, Iterator
    from fsspec import AbstractFileSystem

_PYPROJECT_TOML_TEMPLATE = '\n[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "{name}"\nversion = "{version}"\ndescription = "Micro-package `{name}`"\ndependencies = {install_requires}\n\n[tool.setuptools.packages]\nfind = {{}}\n'
logger = logging.getLogger(__name__)

class _EquivalentRequirement(Requirement):
    """Parse a requirement according to PEP 508."""

    def _iter_parts(self, name: str) -> Iterator[str]:
        yield name
        if self.extras:
            formatted_extras = ','.join(sorted(self.extras))
            yield f'[{formatted_extras}]'
        if self.specifier:
            yield str(self.specifier)
        if self.url:
            yield f'@ {self.url}'
            if self.marker:
                yield ' '
        if self.marker:
            yield f'; {self.marker}'

    def __str__(self) -> str:
        return ''.join(self._iter_parts(self.name))

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, *self._iter_parts(canonicalize_name(self.name))))

    def __eq__(self, other: Any) -> bool:
        return (canonicalize_name(self.name) == canonicalize_name(other.name) and 
                self.extras == other.extras and 
                self.specifier == other.specifier and 
                self.url == other.url and 
                self.marker == other.marker)

def _check_module_path(ctx: click.Context, param: click.Parameter, value: Optional[str]) -> Optional[str]:
    if value and (not re.match('^[\\w.]+$', value)):
        message = 'The micro-package location you provided is not a valid Python module path'
        raise KedroCliError(message)
    return value

@click.group(name='Kedro')
def micropkg_cli() -> None:
    pass

@micropkg_cli.group()
def micropkg() -> None:
    """(DEPRECATED) Commands for working with micro-packages."""

@command_with_verbosity(micropkg, 'pull')
@click.argument('package_path', nargs=1, required=False)
@click.option('--all', '-a', 'all_flag', is_flag=True, help='Pull and unpack all micro-packages in the `pyproject.toml` package manifest section.')
@env_option(help='Environment to install the micro-package configuration to. Defaults to `base`.')
@click.option('--alias', type=str, default='', help='Rename the package.')
@click.option('-d', '--destination', type=click.Path(file_okay=False, dir_okay=False), default=None, help='Module location where to unpack under.')
@click.option('--fs-args', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), default=None, help='Location of a configuration file for the fsspec filesystem used to pull the package.')
@click.pass_obj
def pull_package(
    metadata: ProjectMetadata,
    package_path: Optional[str],
    env: Optional[str],
    alias: str,
    destination: Optional[str],
    fs_args: Optional[str],
    all_flag: bool,
    **kwargs: Any
) -> None:
    """(DEPRECATED) Pull and unpack a modular pipeline and other micro-packages in your project."""
    deprecation_message = "DeprecationWarning: Command 'kedro micropkg pull' is deprecated and will not be available from Kedro 0.20.0."
    click.secho(deprecation_message, fg='red')
    if not package_path and (not all_flag):
        click.secho("Please specify a package path or add '--all' to pull all micro-packages in the 'pyproject.toml' package manifest section.")
        sys.exit(1)
    if all_flag:
        _pull_packages_from_manifest(metadata)
        return
    _pull_package(package_path, metadata, env=env, alias=alias, destination=destination, fs_args=fs_args)
    as_alias = f" as '{alias}'" if alias else ''
    message = f'Micro-package {package_path} pulled and unpacked{as_alias}!'
    click.secho(message, fg='green')

def _pull_package(
    package_path: Optional[str],
    metadata: ProjectMetadata,
    env: Optional[str] = None,
    alias: Optional[str] = None,
    destination: Optional[str] = None,
    fs_args: Optional[str] = None
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir).resolve()
        _unpack_sdist(package_path, temp_dir_path, fs_args)
        contents = [member for member in temp_dir_path.iterdir() if member.is_dir()]
        if len(contents) != 1:
            raise KedroCliError(f'Invalid sdist was extracted: exactly one directory was expected, got {contents}')
        project_root_dir = contents[0]
        library_meta = project_wheel_metadata(project_root_dir)
        packages = [project_item.name for project_item in project_root_dir.iterdir() if project_item.is_dir() and project_item.name != 'tests' and (project_item / '__init__.py').exists()]
        if len(packages) != 1:
            raise KedroCliError(f'Invalid package contents: exactly one package was expected, got {packages}')
        package_name = packages[0]
        package_reqs = _get_all_library_reqs(library_meta)
        if package_reqs:
            requirements_txt = metadata.project_path / 'requirements.txt'
            _append_package_reqs(requirements_txt, package_reqs, package_name)
        _clean_pycache(temp_dir_path)
        _install_files(metadata, package_name, project_root_dir, env, alias, destination)

def _pull_packages_from_manifest(metadata: ProjectMetadata) -> None:
    config_dict = toml.load(metadata.config_file)
    config_dict = config_dict['tool']['kedro']
    build_specs = config_dict.get('micropkg', {}).get('pull')
    if not build_specs:
        click.secho("Nothing to pull. Please update the 'pyproject.toml' package manifest section.", fg='yellow')
        return
    for package_path, specs in build_specs.items():
        if 'alias' in specs:
            _assert_pkg_name_ok(specs['alias'].split('.')[-1])
        _pull_package(package_path, metadata, **specs)
        click.secho(f"Pulled and unpacked '{package_path}'!")
    click.secho('Micro-packages pulled and unpacked!', fg='green')

def _package_micropkgs_from_manifest(metadata: ProjectMetadata) -> None:
    config_dict = toml.load(metadata.config_file)
    config_dict = config_dict['tool']['kedro']
    build_specs = config_dict.get('micropkg', {}).get('package')
    if not build_specs:
        click.secho("Nothing to package. Please update the 'pyproject.toml' package manifest section.", fg='yellow')
        return
    for package_name, specs in build_specs.items():
        if 'alias' in specs:
            _assert_pkg_name_ok(specs['alias'])
        _package_micropkg(package_name, metadata, **specs)
        click.secho(f"Packaged '{package_name}' micro-package!")
    click.secho('Micro-packages packaged!', fg='green')

@command_with_verbosity(micropkg, 'package')
@env_option(help='Environment where the micro-package configuration lives. Defaults to `base`.')
@click.option('--alias', type=str, default='', callback=_check_pipeline_name, help='Alternative name to package under.')
@click.option('-d', '--destination', type=click.Path(resolve_path=True, file_okay=False), help='Location where to create the source distribution file. Defaults to `dist/`.')
@click.option('--all', '-a', 'all_flag', is_flag=True, help='Package all micro-packages in the `pyproject.toml` package manifest section.')
@click.argument('module_path', nargs=1, required=False, callback=_check_module_path)
@click.pass_obj
def package_micropkg(
    metadata: ProjectMetadata,
    module_path: Optional[str],
    env: Optional[str],
    alias: str,
    destination: Optional[str],
    all_flag: bool,
    **kwargs: Any
) -> None:
    """(DEPRECATED) Package up a modular pipeline or micro-package as a Python source distribution."""
    deprecation_message = "DeprecationWarning: Command 'kedro micropkg package' is deprecated and will not be available from Kedro 0.20.0."
    click.secho(deprecation_message, fg='red')
    if not module_path and (not all_flag):
        click.secho("Please specify a micro-package name or add '--all' to package all micro-packages in the 'pyproject.toml' package manifest section.")
        sys.exit(1)
    if all_flag:
        _package_micropkgs_from_manifest(metadata)
        return
    result_path = _package_micropkg(module_path, metadata, alias=alias, destination=destination, env=env)
    as_alias = f" as '{alias}'" if alias else ''
    message = f"'{metadata.package_name}.{module_path}' packaged{as_alias}! Location: {result_path}"
    click.secho(message, fg='green')

def _get_fsspec_filesystem(location: str, fs_args: Optional[str]) -> Optional[AbstractFileSystem]:
    import fsspec
    from kedro.io.core import get_protocol_and_path
    protocol, _ = get_protocol_and_path(location)
    fs_args_config = OmegaConf.to_container(OmegaConf.load(fs_args)) if fs_args else {}
    try:
        return fsspec.filesystem(protocol, **fs_args_config)
    except Exception as exc:
        click.secho(str(exc), fg='red')
        click.secho("Trying to use 'pip download'...", fg='red')
        return None

def _is_within_directory(directory: Path, target: Path) -> bool:
    abs_directory = directory.resolve()
    abs_target = target.resolve()
    return abs_directory in abs_target.parents

def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    safe_members = []
    for member in tar.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise Exception('Failed to safely extract tar file.')
        safe_members.append(member)
    tar.extractall(path, members=safe_members)

def _unpack_sdist(location: Optional[str], destination: Path, fs_args: Optional[str]) -> None:
    if location is None:
        raise KedroCliError("Package path cannot be None")
    filesystem = _get_fsspec_filesystem(location, fs_args)
    if location.endswith('.tar.gz') and filesystem and filesystem.exists(location):
        with filesystem.open(location) as fs_file:
            with tarfile.open(fileobj=fs_file, mode='r:gz') as tar_file:
                safe_extract(tar_file, destination)
    else:
        python_call('pip', ['download', '--no-deps', '--no-binary', ':all:', '--dest', str(destination), location])
        sdist_file = list(destination.glob('*.tar.gz'))
        if len(sdist_file) != 1:
            file_names = [sf.name for sf in sdist_file]
            raise KedroCliError(f'More than 1 or no sdist files found: {file_names}. There has to be exactly one source distribution file.')
        with tarfile.open(sdist_file[0], 'r:gz') as fs_file:
            safe_extract(fs_file, destination)

def _rename_files(conf_source: Path, old_name: str, new_name: str) -> None:
    config_files_to_rename = (each for each in conf_source.rglob('*') if each.is_file() and old_name in each.name)
    for config_file in config_files_to_rename:
        new_config_name = config_file.name.replace(old_name, new_name)
        config_file.rename(config_file.parent / new_config_name)

def _refactor_code_for_unpacking(
    project: Project,
    package_path: Path,
    tests_path: Path,
    alias: Optional[str],
    destination: Optional[str],
    project_metadata: ProjectMetadata
) -> Tuple[Path, Path]:
    package_name = package_path.stem
    package_target = Path(project_metadata.package_name)
    tests_target = Path('tests')
    if destination:
        destination_path = Path(destination)
        package_target = package_target / destination_path
        tests_target = tests_target / destination_path
    if alias and alias != package_name:
        _rename_package(project, package_name, alias)
        package_name = alias
    if package_name == project_metadata.package_name:
        full_path = _move_package_with_conflicting_name(project, package_target, package_name)
    else:
        full_path = _create_nested_package(project, package_target)
        _move_package(project, package_name, package_target.as_posix())
    refactored_package_path = full_path / package_name
    if not tests_path.exists():
        return (refactored_package_path, tests_path)
    full_path = _move_package_with_conflicting_name(project, tests_target, original_name='tests', desired_name=package_name)
    refactored_tests_path = full_path / package_name
    return (refactored_package_path, refactored_tests_path)

def _install_files(
    project_metadata: ProjectMetadata,
    package_name: str,
    source_path: Path,
    env: Optional[str] = None,
    alias: Optional[str] = None,
    destination: Optional[str] = None
) -> None:
    env = env or 'base'
    package_source, test_source, conf_source = _get_package_artifacts(source_path, package_name)
    if conf_source.is_dir() and alias:
        _rename_files(conf_source, package_name, alias)
    module_path = alias or package_name
    if destination:
        module_path = f'{destination}.{module_path}'
    package_dest, test_dest, conf_dest = _get_artifacts_to_package(project_metadata, module_path=module_path, env=env)
    if conf_source.is_dir():
        _sync_dirs(conf_source, conf_dest)
        shutil.rmtree(str(conf_source))
    project = Project(source_path)
    refactored_package_source, refactored_test_source = _refactor_code_for_unpacking(project, package_source, test_source, alias, destination, project_metadata)
    project.close()
    if refactored_test_source.is_dir():
        _sync_dirs(refactored_test_source, test_dest)
    if refactored_package_source.is_dir():
        _sync_dirs(refactored_package_source, package_dest)

def _find_config_files(source_config_dir: Path, glob_patterns: List[str]) -> List[Tuple[Path, str]]:
    config_files = []
    if source_config_dir.is_dir():
        config_files = [(path, path.parent.relative_to(source_config_dir).as_posix()) for glob_pattern in glob_patterns for path in source_config_dir.glob(glob_pattern) if path.is_file()]
    return config_files

def _get_default_version(metadata: ProjectMetadata, micropkg_module_path: str) -> str:
    try:
        micropkg_module = import_module(f'{metadata.package_name}.{micropkg_module_path}')
        return micropkg_module.__version__
    except (AttributeError, ModuleNotFoundError):
        logger.warning("Micropackage version not found in '%s.%s', will take the top-level one in '%s'", metadata.package_name, micropkg_module_path, metadata.package_name)
        project_module = import_module(f'{metadata.package_name}')
        return project_module.__version__

def _package_micropkg(
    micropkg_module_path: str,
    metadata: ProjectMetadata,
    alias: Optional[str] = None,
    destination: Optional[str] = None,
    env: Optional[str] = None
) -> Path:
    micropkg_name = micropkg_module_path.split('.')[-1]
    package_dir = metadata.source_dir / metadata.package_name
    env = env or 'base'
    package_source, package_tests, package_conf = _get_artifacts_to_package(metadata, module_path=micropkg_module_path, env=env)
    configs_to_package = _find_config_files(package_conf, [f'**/parameters_{micropkg_name}.yml', f'**/{micropkg_name}/**/*', f'