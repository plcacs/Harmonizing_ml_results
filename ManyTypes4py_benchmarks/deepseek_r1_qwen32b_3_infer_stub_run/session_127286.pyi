"""This module implements Kedro session responsible for project lifecycle."""

from __future__ import annotations
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)
import logging
import traceback

from kedro.config import AbstractConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks.manager import HookManager
from kedro.framework.session.store import BaseSessionStore
from kedro.runner import AbstractRunner, SequentialRunner
from kedro.utils import RunResult


def _describe_git(project_path: Path) -> Dict[str, Dict[str, Union[str, bool]]]:
    ...


def _jsonify_cli_context(ctx: Any) -> Dict[str, Union[List[str], str]]:
    ...


class KedroSessionError(Exception):
    pass


class KedroSession:
    def __init__(
        self,
        session_id: str,
        package_name: Optional[str] = None,
        project_path: Optional[Path] = None,
        save_on_close: bool = False,
        conf_source: Optional[str] = None,
    ) -> None:
        ...

    @classmethod
    def create(
        cls,
        project_path: Optional[Path] = None,
        save_on_close: bool = True,
        env: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        conf_source: Optional[str] = None,
    ) -> KedroSession:
        ...

    def _init_store(self) -> BaseSessionStore:
        ...

    def _log_exception(
        self,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_tb: traceback.TracebackException,
    ) -> None:
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

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        tb: Optional[traceback.TracebackException],
    ) -> None:
        ...

    def run(
        self,
        pipeline_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        runner: Optional[AbstractRunner] = None,
        node_names: Optional[List[str]] = None,
        from_nodes: Optional[List[str]] = None,
        to_nodes: Optional[List[str]] = None,
        from_inputs: Optional[List[str]] = None,
        to_outputs: Optional[List[str]] = None,
        load_versions: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
    ) -> RunResult:
        ...