"""
Type stub file for session_127286 module.
"""

from __future__ import annotations
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    Iterable,
    ClassVar,
    Tuple,
    TypeVar,
    cast,
)
import logging
import traceback
import click
from kedro.framework.context import KedroContext
from kedro.io.core import generate_timestamp
from kedro.runner import AbstractRunner
from kedro.framework.session.store import BaseSessionStore
from collections.abc import Iterable
from logging import Logger
from types import TracebackType

def _describe_git(project_path: Path) -> Dict[str, Any]: ...

def _jsonify_cli_context(ctx: click.Context) -> Dict[str, Any]: ...

class KedroSessionError(Exception): ...

class KedroSession:
    _project_path: Path
    session_id: str
    save_on_close: bool
    _package_name: Optional[str]
    _store: BaseSessionStore
    _run_called: bool
    _hook_manager: Any  # Inferred from internal usage
    _conf_source: str

    def __init__(self, session_id: str, package_name: Optional[str] = None, project_path: Optional[Path] = None, save_on_close: bool = False, conf_source: Optional[str] = None) -> None: ...

    @classmethod
    def create(cls, project_path: Optional[Path] = None, save_on_close: bool = True, env: Optional[str] = None, extra_params: Optional[Dict[str, Any]] = None, conf_source: Optional[str] = None) -> KedroSession: ...

    def _init_store(self) -> BaseSessionStore: ...

    def _log_exception(self, exc_type: Type[BaseException], exc_value: BaseException, exc_tb: TracebackType) -> None: ...

    @property
    def _logger(self) -> Logger: ...

    @property
    def store(self) -> Dict[str, Any]: ...

    def load_context(self) -> KedroContext: ...

    def _get_config_loader(self) -> AbstractConfigLoader: ...

    def close(self) -> None: ...

    def __enter__(self) -> KedroSession: ...

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], tb: Optional[TracebackType]) -> None: ...

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
        namespace: Optional[str] = None
    ) -> Dict[Any, Any]: ...