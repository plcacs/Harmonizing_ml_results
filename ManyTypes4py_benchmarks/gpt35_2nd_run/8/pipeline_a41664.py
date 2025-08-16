from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
import click
import kedro
from kedro.framework.cli.utils import KedroCliError, _clean_pycache, command_with_verbosity, env_option
from kedro.framework.project import settings

if TYPE_CHECKING:
    from kedro.framework.startup import ProjectMetadata

_SETUP_PY_TEMPLATE: str = '# -*- coding: utf-8 -*-\nfrom setuptools import setup, find_packages\n\nsetup(\n    name="{name}",\n    version="{version}",\n    description="Modular pipeline `{name}`",\n    packages=find_packages(),\n    include_package_data=True,\n    install_requires={install_requires},\n)\n'

class PipelineArtifacts(NamedTuple):
    """An ordered collection of source_path, tests_path, config_paths"""
    source_path: Path
    tests_path: Path
    config_paths: Path

def _assert_pkg_name_ok(pkg_name: str) -> None:
    """Check that python package name is in line with PEP8 requirements.

    Args:
        pkg_name: Candidate Python package name.

    Raises:
        KedroCliError: If package name violates the requirements.
    """

def _check_pipeline_name(ctx: Any, param: Any, value: Any) -> Any:

@click.group(name='Kedro')
def pipeline_cli() -> None:

@pipeline_cli.group()
def pipeline() -> None:

@command_with_verbosity(pipeline, 'create')
@click.argument('name', nargs=1, callback=_check_pipeline_name)
@click.option('--skip-config', is_flag=True, help='Skip creation of config files for the new pipeline(s).')
@click.option('template_path', '-t', '--template', type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path), help='Path to cookiecutter template to use for pipeline(s). Will override any local templates.')
@env_option(help='Environment to create pipeline configuration in. Defaults to `base`.')
@click.pass_obj
def create_pipeline(metadata: ProjectMetadata, /, name: str, template_path: Path, skip_config: bool, env: str, **kwargs: Any) -> None:

@command_with_verbosity(pipeline, 'delete')
@click.argument('name', nargs=1, callback=_check_pipeline_name)
@env_option(help="Environment to delete pipeline configuration from. Defaults to 'base'.")
@click.option('-y', '--yes', is_flag=True, help='Confirm deletion of pipeline non-interactively.')
@click.pass_obj
def delete_pipeline(metadata: ProjectMetadata, /, name: str, env: str, yes: bool, **kwargs: Any) -> None:

def _echo_deletion_warning(message: str, **paths: Any) -> None:

def _create_pipeline(name: str, template_path: Path, output_dir: Path) -> Path:

def _sync_dirs(source: Path, target: Path, prefix: str = '', overwrite: bool = False) -> None:

def _get_pipeline_artifacts(project_metadata: ProjectMetadata, pipeline_name: str, env: str) -> PipelineArtifacts:

def _get_artifacts_to_package(project_metadata: ProjectMetadata, module_path: str, env: str) -> tuple[Path, Path, Path]:

def _copy_pipeline_tests(pipeline_name: str, result_path: Path, project_root: Path) -> None:

def _copy_pipeline_configs(result_path: Path, conf_path: Path, skip_config: bool, env: str) -> None:

def _delete_artifacts(*artifacts: Path) -> None:
