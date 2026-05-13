from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Union
from collections.abc import Iterable
import click
import traceback
from types import TracebackType
from kedro.config import AbstractConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.session.store import BaseSessionStore
from kedro.runner import AbstractRunner


class KedroSessionError(Exception):
    pass


class KedroSession:
    session_id: str
    save_on_close: bool

    def __init__(
        self,
        session_id: str,
        package_name: Optional[str] = None,
        project_path: Optional[Union[str, Path]] = None,
        save_on_close: bool = False,
        conf_source: Optional[str] = None,
    ) -> None: ...

    @classmethod
    def create(
        cls,
        project_path: Optional[Union[str, Path]] = None,
        save_on_close: bool = True,
        env: Optional[str] = None,
        extra_params: Optional[dict[str, Any]] = None,
        conf_source: Optional[str] = None,
    ) -> KedroSession: ...

    def _init_store(self) -> BaseSessionStore: ...

    def _log_exception(
        self,
        exc_type: type[BaseException],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...

    @property
    def _logger(self): ...

    @property
    def store(self) -> dict[str, Any]: ...

    def load_context(self) -> KedroContext: ...

    def _get_config_loader(self) -> AbstractConfigLoader: ...

    def close(self) -> None: ...

    def __enter__(self) -> KedroSession: ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...

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
    ) -> dict[str, Any]: ...


def _describe_git(project_path: Path) -> dict[str, dict[str, str | bool]]: ...
def _jsonify_cli_context(ctx: click.Context) -> dict[str, Any]: ...