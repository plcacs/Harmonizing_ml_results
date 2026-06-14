from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from kedro.config import AbstractConfigLoader
    from kedro.framework.context import KedroContext
    from kedro.framework.session.store import BaseSessionStore
    from kedro.runner import AbstractRunner

import click

def _describe_git(project_path: Path | str) -> dict[str, Any]: ...

def _jsonify_cli_context(ctx: click.Context) -> dict[str, Any]: ...

class KedroSessionError(Exception): ...

class KedroSession:
    session_id: str
    save_on_close: bool

    _project_path: Path
    _package_name: str | None
    _store: BaseSessionStore
    _run_called: bool
    _hook_manager: Any
    _conf_source: str

    def __init__(
        self,
        session_id: str,
        package_name: str | None = ...,
        project_path: Path | str | None = ...,
        save_on_close: bool = ...,
        conf_source: str | None = ...,
    ) -> None: ...

    @classmethod
    def create(
        cls,
        project_path: Path | str | None = ...,
        save_on_close: bool = ...,
        env: str | None = ...,
        extra_params: dict[str, Any] | None = ...,
        conf_source: str | None = ...,
    ) -> KedroSession: ...

    def _init_store(self) -> BaseSessionStore: ...

    def _log_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None: ...

    @property
    def _logger(self) -> logging.Logger: ...

    @property
    def store(self) -> dict[str, Any]: ...

    def load_context(self) -> KedroContext: ...

    def _get_config_loader(self) -> AbstractConfigLoader: ...

    def close(self) -> None: ...

    def __enter__(self) -> KedroSession: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb_: TracebackType | None,
    ) -> None: ...

    def run(
        self,
        pipeline_name: str | None = ...,
        tags: Iterable[str] | None = ...,
        runner: AbstractRunner | None = ...,
        node_names: Iterable[str] | None = ...,
        from_nodes: Iterable[str] | None = ...,
        to_nodes: Iterable[str] | None = ...,
        from_inputs: Iterable[str] | None = ...,
        to_outputs: Iterable[str] | None = ...,
        load_versions: dict[str, str] | None = ...,
        namespace: str | None = ...,
    ) -> dict[str, Any]: ...