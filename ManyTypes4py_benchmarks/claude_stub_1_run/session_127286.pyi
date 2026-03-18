```pyi
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from kedro.config import AbstractConfigLoader
    from kedro.framework.context import KedroContext
    from kedro.framework.session.store import BaseSessionStore
    from kedro.runner import AbstractRunner

def _describe_git(project_path: Path | str) -> dict[str, Any]: ...
def _jsonify_cli_context(ctx: Any) -> dict[str, Any]: ...

class KedroSessionError(Exception):
    """``KedroSessionError`` raised by ``KedroSession``
    in the case that multiple runs are attempted in one session.
    """
    ...

class KedroSession:
    """``KedroSession`` is the object that is responsible for managing the lifecycle
    of a Kedro run. Use `KedroSession.create()` as
    a context manager to construct a new KedroSession with session data
    provided (see the example below).
    """

    _project_path: Path
    session_id: str
    save_on_close: bool
    _package_name: str | None
    _store: BaseSessionStore
    _run_called: bool
    _hook_manager: Any
    _conf_source: str

    def __init__(
        self,
        session_id: str,
        package_name: str | None = None,
        project_path: Path | str | None = None,
        save_on_close: bool = False,
        conf_source: str | None = None,
    ) -> None: ...

    @classmethod
    def create(
        cls,
        project_path: Path | str | None = None,
        save_on_close: bool = True,
        env: str | None = None,
        extra_params: dict[str, Any] | None = None,
        conf_source: str | None = None,
    ) -> KedroSession: ...

    def _init_store(self) -> BaseSessionStore: ...
    def _log_exception(self, exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> None: ...

    @property
    def _logger(self) -> logging.Logger: ...

    @property
    def store(self) -> dict[str, Any]: ...

    def load_context(self) -> KedroContext: ...
    def _get_config_loader(self) -> AbstractConfigLoader: ...
    def close(self) -> None: ...
    def __enter__(self) -> KedroSession: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb_: Any) -> None: ...

    def run(
        self,
        pipeline_name: str | None = None,
        tags: Iterable[str] | None = None,
        runner: AbstractRunner | None = None,
        node_names: Iterable[str] | None = None,
        from_nodes: Iterable[str] | None = None,
        to_nodes: Iterable[str] | None = None,
        from_inputs: Iterable[str] | None = None,
        to_outputs: Iterable[str] | None = None,
        load_versions: dict[str, str] | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]: ...
```