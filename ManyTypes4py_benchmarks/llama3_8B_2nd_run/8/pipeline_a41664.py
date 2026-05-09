from __future__ import annotations
import re
import shutil
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, PathType

class PipelineArtifacts(NamedTuple):
    """An ordered collection of source_path, tests_path, config_paths"""

def _assert_pkg_name_ok(pkg_name: str) -> None:
    """Check that python package name is in line with PEP8 requirements.

    Args:
        pkg_name: Candidate Python package name.

    Raises:
        KedroCliError: If package name violates the requirements.
    """
    # ...

def _check_pipeline_name(ctx: click.Context, param: click.Parameter, value: str) -> str:
    if value:
        _assert_pkg_name_ok(value)
    return value

@click.group(name='Kedro')
def pipeline_cli() -> None:
    pass

@pipeline_cli.group()
def pipeline() -> None:
    """Commands for working with pipelines."""

@command_with_verbosity(pipeline, 'create')
@click.argument('name', nargs=1, callback=_check_pipeline_name)
@click.option('--skip-config', is_flag=True, help='Skip creation of config files for the new pipeline(s).')
@click.option('template_path', '-t', '--template', type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=PathType), help='Path to cookiecutter template to use for pipeline(s). Will override any local templates.')
@env_option(help='Environment to create pipeline configuration in. Defaults to `base`.')
@click.pass_obj
def create_pipeline(metadata: ProjectMetadata, /, name: str, template_path: Optional[PathType], skip_config: bool, env: str, **kwargs) -> None:
    # ...

@command_with_verbosity(pipeline, 'delete')
@click.argument('name', nargs=1, callback=_check_pipeline_name)
@env_option(help="Environment to delete pipeline configuration from. Defaults to 'base'.")
@click.option('-y', '--yes', is_flag=True, help='Confirm deletion of pipeline non-interactively.')
@click.pass_obj
def delete_pipeline(metadata: ProjectMetadata, /, name: str, env: str, yes: bool, **kwargs) -> None:
    # ...

def _echo_deletion_warning(message: str, **paths: PathType) -> None:
    # ...

def _create_pipeline(name: str, template_path: PathType, output_dir: PathType) -> PathType:
    # ...

def _sync_dirs(source: PathType, target: PathType, prefix: str = '', overwrite: bool = False) -> None:
    # ...

def _get_pipeline_artifacts(project_metadata: ProjectMetadata, pipeline_name: str, env: str) -> PipelineArtifacts:
    # ...

def _get_artifacts_to_package(project_metadata: ProjectMetadata, module_path: str, env: str) -> tuple[PathType, PathType, PathType]:
    # ...

def _copy_pipeline_tests(pipeline_name: str, result_path: PathType, project_root: PathType) -> None:
    # ...

def _copy_pipeline_configs(result_path: PathType, conf_path: PathType, skip_config: bool, env: str) -> None:
    # ...

def _delete_artifacts(*artifacts: PathType) -> None:
    # ...
