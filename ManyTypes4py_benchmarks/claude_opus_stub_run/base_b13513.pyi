import os
import threading
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dbt.artifacts.resources.types import NodeType
from dbt.artifacts.schemas.results import NodeStatus, TimingInfo
from dbt.artifacts.schemas.run import RunResult
from dbt.compilation import Compiler
from dbt.config import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.graph import Graph


def read_profiles(profiles_dir: Optional[str] = ...) -> Dict[str, Any]: ...

class BaseTask(metaclass=ABCMeta):
    args: Any
    orig_dir: str

    def __init__(self, args: Any) -> None: ...
    def __enter__(self) -> "BaseTask": ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

    @abstractmethod
    def run(self) -> Any: ...
    def interpret_results(self, results: Any) -> bool: ...

def get_nearest_project_dir(project_dir: Optional[str]) -> Path: ...
def move_to_nearest_project_dir(project_dir: Optional[str]) -> Path: ...

class ConfiguredTask(BaseTask):
    config: RuntimeConfig
    graph: Optional[Graph]
    manifest: Optional[Manifest]
    compiler: Compiler

    def __init__(self, args: Any, config: RuntimeConfig, manifest: Optional[Manifest] = ...) -> None: ...
    def compile_manifest(self) -> None: ...

    @classmethod
    def from_args(cls, args: Any, *pargs: Any, **kwargs: Any) -> "ConfiguredTask": ...

class ExecutionContext:
    timing: List[TimingInfo]
    node: Any

    def __init__(self, node: Any) -> None: ...

class BaseRunner(metaclass=ABCMeta):
    config: RuntimeConfig
    compiler: Compiler
    adapter: Any
    node: Any
    node_index: int
    num_nodes: int
    skip: bool
    skip_cause: Optional[RunResult]
    run_ephemeral_models: bool

    def __init__(
        self,
        config: RuntimeConfig,
        adapter: Any,
        node: Any,
        node_index: int,
        num_nodes: int,
    ) -> None: ...

    @abstractmethod
    def compile(self, manifest: Manifest) -> Any: ...
    def _node_build_path(self) -> Optional[str]: ...
    def get_result_status(self, result: RunResult) -> Dict[str, str]: ...
    def run_with_hooks(self, manifest: Manifest) -> RunResult: ...
    def _build_run_result(
        self,
        node: Any,
        start_time: float,
        status: NodeStatus,
        timing_info: List[TimingInfo],
        message: Optional[str],
        agate_table: Optional[Any] = ...,
        adapter_response: Optional[Dict[str, Any]] = ...,
        failures: Optional[Any] = ...,
        batch_results: Optional[Any] = ...,
    ) -> RunResult: ...
    def error_result(
        self,
        node: Any,
        message: str,
        start_time: float,
        timing_info: List[TimingInfo],
    ) -> RunResult: ...
    def ephemeral_result(
        self,
        node: Any,
        start_time: float,
        timing_info: List[TimingInfo],
    ) -> RunResult: ...
    def from_run_result(
        self,
        result: RunResult,
        start_time: float,
        timing_info: List[TimingInfo],
    ) -> RunResult: ...
    def compile_and_execute(self, manifest: Manifest, ctx: ExecutionContext) -> Optional[RunResult]: ...
    def _handle_catchable_exception(self, e: Any, ctx: ExecutionContext) -> str: ...
    def _handle_internal_exception(self, e: Any, ctx: ExecutionContext) -> str: ...
    def _handle_generic_exception(self, e: Any, ctx: ExecutionContext) -> str: ...
    def handle_exception(self, e: Exception, ctx: ExecutionContext) -> str: ...
    def safe_run(self, manifest: Manifest) -> RunResult: ...
    def _safe_release_connection(self) -> Optional[str]: ...
    def before_execute(self) -> None: ...
    def execute(self, compiled_node: Any, manifest: Manifest) -> RunResult: ...
    def run(self, compiled_node: Any, manifest: Manifest) -> RunResult: ...
    def after_execute(self, result: RunResult) -> None: ...
    def _skip_caused_by_ephemeral_failure(self) -> bool: ...
    def on_skip(self) -> RunResult: ...
    def do_skip(self, cause: Optional[RunResult] = ...) -> None: ...

def resource_types_from_args(
    args: Any,
    all_resource_values: Set[NodeType],
    default_resource_values: Set[NodeType],
) -> Set[NodeType]: ...