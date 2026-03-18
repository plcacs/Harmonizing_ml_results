from typing import Any, Iterable, Optional, Union, Type
from pathlib import Path
import logging

def _describe_git(project_path: Any) -> dict[str, Any]: ...
def _jsonify_cli_context(ctx: Any) -> dict[str, Any]: ...

class KedroSessionError(Exception): ...

class KedroSession:
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
    ) -> "KedroSession": ...
    def _init_store(self) -> Any: ...
    def _log_exception(self, exc_type: Type[BaseException], exc_value: BaseException, exc_tb: Any) -> None: ...
    @property
    def _logger(self) -> logging.Logger: ...
    @property
    def store(self) -> dict[str, Any]: ...
    def load_context(self) -> Any: ...
    def _get_config_loader(self) -> Any: ...
    def close(self) -> None: ...
    def __enter__(self) -> "KedroSession": ...
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], tb_: Optional[Any]) -> None: ...
    def run(
        self,
        pipeline_name: Optional[str] = ...,
        tags: Optional[Iterable[str]] = ...,
        runner: Any = ...,
        node_names: Optional[Iterable[str]] = ...,
        from_nodes: Optional[Iterable[str]] = ...,
        to_nodes: Optional[Iterable[str]] = ...,
        from_inputs: Optional[Iterable[str]] = ...,
        to_outputs: Optional[Iterable[str]] = ...,
        load_versions: Any = ...,
        namespace: Optional[str] = ...,
    ) -> Any: ...