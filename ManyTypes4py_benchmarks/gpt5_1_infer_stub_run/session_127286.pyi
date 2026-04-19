from typing import Any, Iterable, Optional
from os import PathLike
from types import TracebackType
import logging
import click
from kedro.config import AbstractConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.session.store import BaseSessionStore
from kedro.runner import AbstractRunner

def _describe_git(project_path: PathLike[str] | str) -> dict[str, dict[str, object]]: ...
def _jsonify_cli_context(ctx: click.Context) -> dict[str, object]: ...

class KedroSessionError(Exception): ...

class KedroSession:
    def __init__(
        self,
        session_id: str,
        package_name: Optional[str] = ...,
        project_path: Optional[PathLike[str] | str] = ...,
        save_on_close: bool = ...,
        conf_source: Optional[PathLike[str] | str] = ...,
    ) -> None: ...
    @classmethod
    def create(
        cls,
        project_path: Optional[PathLike[str] | str] = ...,
        save_on_close: bool = ...,
        env: Optional[str] = ...,
        extra_params: Optional[dict[str, Any]] = ...,
        conf_source: Optional[PathLike[str] | str] = ...,
    ) -> "KedroSession": ...
    def _init_store(self) -> BaseSessionStore: ...
    def _log_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType,
    ) -> None: ...
    @property
    def _logger(self) -> logging.Logger: ...
    @property
    def store(self) -> dict[str, Any]: ...
    def load_context(self) -> KedroContext: ...
    def _get_config_loader(self) -> AbstractConfigLoader: ...
    def close(self) -> None: ...
    def __enter__(self) -> "KedroSession": ...
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