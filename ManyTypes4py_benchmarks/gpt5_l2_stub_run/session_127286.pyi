from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Optional

from click import Context
from kedro.config import AbstractConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.session.store import BaseSessionStore
from kedro.runner import AbstractRunner, SequentialRunner

kedro_version: str = ...

def _describe_git(project_path: Path) -> dict[str, Any]: ...
def _jsonify_cli_context(ctx: Context) -> dict[str, Any]: ...

class KedroSessionError(Exception): ...

class KedroSession:
    session_id: str
    save_on_close: bool

    def __init__(
        self,
        session_id: str,
        package_name: Optional[str] = ...,
        project_path: Optional[Path | str] = ...,
        save_on_close: bool = ...,
        conf_source: Optional[str] = ...,
    ) -> None: ...
    @classmethod
    def create(
        cls,
        project_path: Optional[Path | str] = ...,
        save_on_close: bool = ...,
        env: Optional[str] = ...,
        extra_params: Optional[dict[str, Any]] = ...,
        conf_source: Optional[str] = ...,
    ) -> KedroSession: ...
    def _init_store(self) -> BaseSessionStore: ...
    def _log_exception(
        self, exc_type: type[BaseException], exc_value: BaseException, exc_tb: TracebackType
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
        pipeline_name: Optional[str] = ...,
        tags: Optional[Iterable[str]] = ...,
        runner: Optional[AbstractRunner] = ...,
        node_names: Optional[Iterable[str]] = ...,
        from_nodes: Optional[Iterable[str]] = ...,
        to_nodes: Optional[Iterable[str]] = ...,
        from_inputs: Optional[Iterable[str]] = ...,
        to_outputs: Optional[Iterable[str]] = ...,
        load_versions: Optional[dict[str, str]] = ...,
        namespace: Optional[str] = ...,
    ) -> dict[str, Any]: ...