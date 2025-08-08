from __future__ import annotations
import getpass
import logging
import logging.config
import os
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import click
from kedro import __version__ as kedro_version
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.hooks.manager import _register_hooks, _register_hooks_entry_points
from kedro.framework.project import pipelines, settings, validate_settings
from kedro.io.core import generate_timestamp
from kedro.runner import AbstractRunner, SequentialRunner
from kedro.utils import _find_kedro_project

if TYPE_CHECKING:
    from collections.abc import Iterable
    from kedro.config import AbstractConfigLoader
    from kedro.framework.context import KedroContext
    from kedro.framework.session.store import BaseSessionStore

def _describe_git(project_path: Path) -> Dict[str, Union[str, bool]]:
    ...

def _jsonify_cli_context(ctx: click.Context) -> Dict[str, Union[List[str], str]]:
    ...

class KedroSessionError(Exception):
    ...

class KedroSession:
    def __init__(self, session_id: str, package_name: Optional[str] = None, project_path: Optional[Path] = None, save_on_close: bool = False, conf_source: Optional[str] = None):
        ...

    @classmethod
    def create(cls, project_path: Optional[Path] = None, save_on_close: bool = True, env: Optional[str] = None, extra_params: Optional[Dict[str, Any]] = None, conf_source: Optional[str] = None) -> KedroSession:
        ...

    def _init_store(self) -> BaseSessionStore:
        ...

    def _log_exception(self, exc_type: type, exc_value: Exception, exc_tb: traceback) -> None:
        ...

    @property
    def _logger(self) -> logging.Logger:
        ...

    @property
    def store(self) -> Dict[str, Any]:
        ...

    def load_context(self) -> KedroContext:
        ...

    def _get_config_loader(self) -> AbstractConfigLoader:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> KedroSession:
        ...

    def __exit__(self, exc_type: type, exc_value: Exception, tb_: traceback) -> None:
        ...

    def run(self, pipeline_name: Optional[str] = None, tags: Optional[List[str]] = None, runner: Optional[AbstractRunner] = None, node_names: Optional[List[str]] = None, from_nodes: Optional[List[str]] = None, to_nodes: Optional[List[str]] = None, from_inputs: Optional[List[str]] = None, to_outputs: Optional[List[str]] = None, load_versions: Optional[str] = None, namespace: Optional[str] = None) -> Any:
        ...
