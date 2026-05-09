import os
import threading
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import dbt.exceptions
import dbt_common.exceptions.base
from dbt.artifacts.schemas.results import NodeStatus, RunStatus, TimingInfo, RunResult
from dbt.config import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest
from dbt.task.printer import print_run_result_error

def read_profiles(profiles_dir: Optional[str] = None) -> Dict[str, Any]:
    ...

def get_nearest_project_dir(project_dir: Optional[str]) -> str:
    ...

def move_to_nearest_project_dir(project_dir: Optional[str]) -> str:
    ...

def resource_types_from_args(
    args: Any,
    all_resource_values: List[str],
    default_resource_values: List[str]
) -> Set[NodeType]:
    ...

class BaseTask(metaclass=ABCMeta):
    def __init__(self, args: Any):
        ...

    def __enter__(self) -> 'BaseTask':
        ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        ...

    @abstractmethod
    def run(self) -> Any:
        raise NotImplementedError('Not Implemented')

    def interpret_results(self, results: Any) -> bool:
        ...

class ConfiguredTask(BaseTask):
    def __init__(self, args: Any, config: RuntimeConfig, manifest: Optional[Manifest] = None):
        ...

    def compile_manifest(self) -> None:
        ...

    @classmethod
    def from_args(cls, args: Any, *pargs: Any, **kwargs: Any) -> 'ConfiguredTask':
        ...

class ExecutionContext:
    def __init__(self, node: Any):
        ...

class BaseRunner(metaclass=ABCMeta):
    def __init__(self, config: RuntimeConfig, adapter: Any, node: Any, node_index: int, num_nodes: int):
        ...

    @abstractmethod
    def compile(self, manifest: Any) -> Any:
        ...

    def get_result_status(self, result: Any) -> Dict[str, str]:
        ...

    def run_with_hooks(self, manifest: Any) -> Any:
        ...

    def _build_run_result(
        self,
        node: Any,
        start_time: float,
        status: RunStatus,
        timing_info: TimingInfo,
        message: Optional[str],
        agate_table: Optional[Any] = None,
        adapter_response: Optional[Dict[str, Any]] = None,
        failures: Optional[List[Any]] = None,
        batch_results: Optional[List[Any]] = None
    ) -> RunResult:
        ...

    def error_result(self, node: Any, message: str, start_time: float, timing_info: TimingInfo) -> RunResult:
        ...

    def ephemeral_result(self, node: Any, start_time: float, timing_info: TimingInfo) -> RunResult:
        ...

    def from_run_result(self, result: RunResult, start_time: float, timing_info: TimingInfo) -> RunResult:
        ...

    def compile_and_execute(self, manifest: Any, ctx: ExecutionContext) -> Optional[Any]:
        ...

    def _handle_catchable_exception(self, e: Exception, ctx: ExecutionContext) -> str:
        ...

    def _handle_internal_exception(self, e: Exception, ctx: ExecutionContext) -> str:
        ...

    def _handle_generic_exception(self, e: Exception, ctx: ExecutionContext) -> str:
        ...

    def handle_exception(self, e: Exception, ctx: ExecutionContext) -> str:
        ...

    def safe_run(self, manifest: Any) -> RunResult:
        ...

    def _safe_release_connection(self) -> Optional[str]:
        ...

    def before_execute(self) -> None:
        raise NotImplementedError('before_execute is not implemented')

    def execute(self, compiled_node: Any, manifest: Any) -> Any:
        raise NotImplementedError('execute is not implemented')

    def run(self, compiled_node: Any, manifest: Any) -> Any:
        ...

    def after_execute(self, result: Any) -> None:
        raise NotImplementedError('after_execute is not implemented')

    def _skip_caused_by_ephemeral_failure(self) -> bool:
        ...

    def on_skip(self) -> RunResult:
        ...

    def do_skip(self, cause: Optional[Any] = None) -> None:
        ...