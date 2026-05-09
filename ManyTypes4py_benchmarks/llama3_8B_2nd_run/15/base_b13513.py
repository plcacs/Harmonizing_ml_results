import os
import threading
import time
import traceback
from abc import ABCMeta, abstractmethod
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import dbt.exceptions
import dbt_common.exceptions.base
from dbt import tracking
from dbt.artifacts.resources.types import NodeType
from dbt.artifacts.schemas.results import NodeStatus, RunningStatus, RunStatus, TimingInfo, collect_timing_info
from dbt.artifacts.schemas.run import RunResult
from dbt.cli.flags import Flags
from dbt.compilation import Compiler
from dbt.config import RuntimeConfig
from dbt.config.profile import read_profile
from dbt.constants import DBT_PROJECT_FILE_NAME
from dbt.contracts.graph.manifest import Manifest
from dbt.events.types import CatchableExceptionOnRun, GenericExceptionOnRun, InternalErrorOnRun, LogDbtProfileError, LogDbtProjectError, LogDebugStackTrace, LogSkipBecauseError, NodeCompiling, NodeConnectionReleaseError, NodeExecuting, SkippingDetails
from dbt.flags import get_flags
from dbt.graph import Graph
from dbt.task import group_lookup
from dbt.task.printer import print_run_result_error
from dbt_common.events.contextvars import get_node_info
from dbt_common.events.functions import fire_event
from dbt_common.exceptions import DbtInternalError, DbtRuntimeError, NotImplementedError

def read_profiles(profiles_dir: Optional[str] = None) -> Dict[str, Any]:
    ...

class BaseTask(metaclass=ABCMeta):
    def __init__(self, args: Any) -> None:
        ...

    def run(self) -> None:
        ...

    def interpret_results(self, results: List[Any]) -> bool:
        ...

class ConfiguredTask(BaseTask):
    def __init__(self, args: Any, config: RuntimeConfig, manifest: Optional[Manifest] = None) -> None:
        ...

    @classmethod
    def from_args(cls, args: Any, *pargs: Any, **kwargs: Any) -> 'ConfiguredTask':
        ...

class ExecutionContext:
    def __init__(self, node: Any) -> None:
        ...

class BaseRunner(metaclass=ABCMeta):
    def __init__(self, config: RuntimeConfig, adapter: Any, node: Any, node_index: int, num_nodes: int) -> None:
        ...

    def compile(self, manifest: Manifest) -> Any:
        ...

    def run_with_hooks(self, manifest: Manifest) -> RunResult:
        ...

    def _build_run_result(self, node: Any, start_time: float, status: RunStatus, timing_info: TimingInfo, message: str, agate_table: Optional[Any] = None, adapter_response: Optional[Any] = None, failures: Optional[List[Any]] = None, batch_results: Optional[List[Any]] = None) -> RunResult:
        ...

    def error_result(self, node: Any, message: str, start_time: float, timing_info: TimingInfo) -> RunResult:
        ...

    def ephemeral_result(self, node: Any, start_time: float, timing_info: TimingInfo) -> RunResult:
        ...

    def from_run_result(self, result: RunResult, start_time: float, timing_info: TimingInfo) -> RunResult:
        ...

    def compile_and_execute(self, manifest: Manifest, ctx: ExecutionContext) -> RunResult:
        ...

    def _handle_catchable_exception(self, e: DbtRuntimeError, ctx: ExecutionContext) -> str:
        ...

    def _handle_internal_exception(self, e: DbtInternalError, ctx: ExecutionContext) -> str:
        ...

    def _handle_generic_exception(self, e: Exception, ctx: ExecutionContext) -> str:
        ...

    def handle_exception(self, e: Exception, ctx: ExecutionContext) -> str:
        ...

    def safe_run(self, manifest: Manifest) -> RunResult:
        ...

    def _safe_release_connection(self) -> Optional[str]:
        ...

    def before_execute(self) -> None:
        ...

    def execute(self, compiled_node: Any, manifest: Manifest) -> Any:
        ...

    def run(self, compiled_node: Any, manifest: Manifest) -> Any:
        ...

    def after_execute(self, result: RunResult) -> None:
        ...

    def on_skip(self) -> RunResult:
        ...

    def do_skip(self, cause: Optional[Exception] = None) -> None:
        ...

def resource_types_from_args(args: Any, all_resource_values: Set[NodeType], default_resource_values: Set[NodeType]) -> Set[NodeType]:
    ...
