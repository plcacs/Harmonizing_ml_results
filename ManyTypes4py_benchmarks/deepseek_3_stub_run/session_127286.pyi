from __future__ import annotations

import logging
import subprocess
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import click

if TYPE_CHECKING:
    from kedro.config import AbstractConfigLoader
    from kedro.framework.context import KedroContext
    from kedro.framework.session.store import BaseSessionStore
    from kedro.runner import AbstractRunner
    from kedro.pipeline import Pipeline

def _describe_git(project_path: Path) -> dict[str, dict[str, Union[str, bool]]]:
    ...

def _jsonify_cli_context(ctx: click.Context) -> dict[str, Any]:
    ...

class KedroSessionError(Exception):
    ...

class KedroSession:
    _project_path: Path
    session_id: str
    save_on_close: bool
    _package_name: Optional[str]
    _store: BaseSessionStore
    _run_called: bool
    _hook_manager: Any
    _conf_source: str
    _logger: logging.Logger

    def __init__(
        self,
        session_id: str,
        package_name: Optional[str] = None,
        project_path: Optional[Union[str, Path]] = None,
        save_on_close: bool = False,
        conf_source: Optional[str] = None,
    ) -> None:
        ...

    @classmethod
    def create(
        cls,
        project_path: Optional[Union[str, Path]] = None,
        save_on_close: bool = True,
        env: Optional[str] = None,
        extra_params: Optional[dict[str, Any]] = None,
        conf_source: Optional[str] = None,
    ) -> KedroSession:
        ...

    def _init_store(self) -> BaseSessionStore:
        ...

    def _log_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: traceback.TracebackException,
    ) -> None:
        ...

    @property
    def store(self) -> dict[str, Any]:
        ...

    def load_context(self) -> KedroContext:
        ...

    def _get_config_loader(self) -> AbstractConfigLoader:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> KedroSession:
        ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        tb_: Optional[traceback.TracebackException],
    ) -> None:
        ...

    def run(
        self,
        pipeline_name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        runner: Optional[AbstractRunner] = None,
        node_names: Optional[Iterable[str]] = None,
        from_nodes: Optional[Iterable[str]] = None,
        to_nodes: Optional[Iterable[str]] = None,
        from_inputs: Optional[Iterable[str]] = None,
        to_outputs: Optional[Iterable[str]] = None,
        load_versions: Optional[dict[str, str]] = None,
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        ...