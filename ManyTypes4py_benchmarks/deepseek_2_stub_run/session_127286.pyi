```python
from __future__ import annotations

import logging
import subprocess
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from kedro.config import AbstractConfigLoader
    from kedro.framework.context import KedroContext
    from kedro.framework.session.store import BaseSessionStore
    import click

def _describe_git(project_path: Path) -> dict[str, Any]: ...

def _jsonify_cli_context(ctx: click.Context) -> dict[str, Any]: ...

class KedroSessionError(Exception): ...

class KedroSession:
    def __init__(
        self,
        session_id: str,
        package_name: str | None = ...,
        project_path: str | Path | None = ...,
        save_on_close: bool = ...,
        conf_source: str | None = ...,
    ) -> None: ...
    
    @classmethod
    def create(
        cls,
        project_path: str | Path | None = ...,
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
        exc_tb: traceback.TracebackException | None,
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
        tb_: traceback.TracebackException | None,
    ) -> None: ...
    
    def run(
        self,
        pipeline_name: str | None = ...,
        tags: Iterable[str] | None = ...,
        runner: Any | None = ...,
        node_names: Iterable[str] | None = ...,
        from_nodes: Iterable[str] | None = ...,
        to_nodes: Iterable[str] | None = ...,
        from_inputs: Iterable[str] | None = ...,
        to_outputs: Iterable[str] | None = ...,
        load_versions: dict[str, str] | None = ...,
        namespace: str | None = ...,
    ) -> dict[str, Any]: ...
```