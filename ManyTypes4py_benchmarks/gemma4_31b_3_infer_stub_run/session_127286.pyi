from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Union, overload
from pathlib import Path
from collections.abc import Iterable

if TYPE_CHECKING:
    from kedro.config import AbstractConfigLoader
    from kedro.framework.context import KedroContext
    from kedro.framework.session.store import BaseSessionStore
    from kedro.runner import AbstractRunner

def _describe_git(project_path: Union[str, Path]) -> dict[str, Any]: ...

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
    _package_name: Optional[str]
    _store: BaseSessionStore
    _run_called: bool
    _hook_manager: Any
    _conf_source: str

    def __init__(
        self,
        session_id: str,
        package_name: Optional[str] = ...,
        project_path: Optional[Union[str, Path]] = ...,
        save_on_close: bool = ...,
        conf_source: Optional[str] = ...,
    ) -> None: ...

    @classmethod
    def create(
        cls,
        project_path: Optional[Union[str, Path]] = ...,
        save_on_close: bool = ...,
        env: Optional[str] = ...,
        extra_params: Optional[dict[str, Any]] = ...,
        conf_source: Optional[str] = ...,
    ) -> KedroSession: ...

    def _init_store(self) -> BaseSessionStore: ...

    def _log_exception(self, exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> None: ...

    @property
    def _logger(self) -> Any: ...

    @property
    def store(self) -> dict[str, Any]: ...

    def load_context(self) -> KedroContext: ...

    def _get_config_loader(self) -> AbstractConfigLoader: ...

    def close(self) -> None: ...

    def __enter__(self) -> KedroSession: ...

    def __exit__(self, exc_type: Optional[type[BaseException]], exc_value: Optional[BaseException], tb_: Any) -> Optional[bool]: ...

    def run(
        self,
        pipeline_name: Optional[str] = ...,
        tags: Optional[Iterable[str]] = ...,
        runner: Optional[Union[AbstractRunner, type[AbstractRunner]]] = ...,
        node_names: Optional[Iterable[str]] = ...,
        from_nodes: Optional[Iterable[str]] = ...,
        to_nodes: Optional[Iterable[str]] = ...,
        from_inputs: Optional[Iterable[str]] = ...,
        to_outputs: Optional[Iterable[str]] = ...,
        load_versions: Optional[Union[str, Iterable[str]]] = ...,
        namespace: Optional[str] = ...,
    ) -> Any: ...