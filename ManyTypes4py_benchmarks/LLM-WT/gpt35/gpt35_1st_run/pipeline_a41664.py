from __future__ import annotations
import re
import shutil
from pathlib import Path
from textwrap import indent
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
    base_message: str = f"'{pkg_name}' is not a valid Python package name."
    if not re.match('^[a-zA-Z_]', pkg_name):
        message: str = base_message + ' It must start with a letter or underscore.'
        raise KedroCliError(message)
    if len(pkg_name) < 2:
        message: str = base_message + ' It must be at least 2 characters long.'
        raise KedroCliError(message)
    if not re.match('^\\w+$', pkg_name[1:]):
        message: str = base_message + ' It must contain only letters, digits, and/or underscores.'
        raise KedroCliError(message)

def _check_pipeline_name(ctx: Any, param: Any, value: Any) -> Any:
    if value:
        _assert_pkg_name_ok(value)
    return value
