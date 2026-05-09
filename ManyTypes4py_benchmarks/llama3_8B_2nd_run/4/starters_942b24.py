from __future__ import annotations
import logging
import os
import re
import shutil
import stat
import sys
import tempfile
import yaml
from collections import OrderedDict
from importlib_metadata import EntryPoints
from typing import TYPE_CHECKING, Any, Callable
import click
from attrs import define, field
from packaging import version
from cookiecutter import main
from cookiecutter.exceptions import RepositoryCloneFailed, RepositoryNotFound
from cookiecutter.generate import generate_context

@define(order=True)
class KedroStarterSpec:
    """Specification of custom kedro starter template
    Args:
        alias: alias of the starter which shows up on `kedro starter list` and is used
        by the starter argument of `kedro new`
        template_path: path to a directory or a URL to a remote VCS repository supported
        by `cookiecutter`
        directory: optional directory inside the repository where the starter resides.
        origin: reserved field used by kedro internally to determine where the starter
        comes from, users do not need to provide this field.
    """
    directory: str | None
    origin: str = field(init=False)

KEDRO_PATH: Path = Path(kedro.__file__).parent
TEMPLATE_PATH: Path = KEDRO_PATH / 'templates' / 'project'

def _get_latest_starters_version() -> str:
    # ...

def _kedro_version_equal_or_lower_to_starters(version: str) -> bool:
    # ...

_STARTERS_REPO: str = 'git+https://github.com/kedro-org/kedro-starters.git'
_OFFICIAL_STARTER_SPECS: list[KedroStarterSpec] = [
    KedroStarterSpec('astro-airflow-iris', _STARTERS_REPO, 'astro-airflow-iris'),
    KedroStarterSpec('spaceflights-pandas', _STARTERS_REPO, 'spaceflights-pandas'),
    # ...
]

def _get_starters_dict() -> dict[str, KedroStarterSpec]:
    # ...

def _get_cookiecutter_dir(template_path: str, checkout: str, directory: str, tmpdir: str) -> Path:
    # ...

def _get_prompts_required_and_clear_from_CLI_provided(cookiecutter_dir: Path, selected_tools: str | None, project_name: str | None, example_pipeline: str | None) -> dict[str, str]:
    # ...

def _get_extra_context(prompts_required: dict[str, str], config_path: str | None, cookiecutter_context: dict[str, str], selected_tools: str | None, project_name: str | None, example_pipeline: str | None, starter_alias: str | None) -> dict[str, str]:
    # ...

def _make_cookiecutter_context_for_prompts(cookiecutter_dir: Path) -> dict[str, str]:
    # ...

def _select_checkout_branch_for_cookiecutter(checkout: str) -> str:
    # ...

def _make_cookiecutter_args_and_fetch_template(config: dict[str, str], checkout: