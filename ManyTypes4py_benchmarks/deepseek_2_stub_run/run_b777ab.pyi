```python
import functools
import os
import threading
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from dbt import tracking, utils
from dbt.adapters.base import BaseAdapter, BaseRelation
from dbt.adapters.capability import Capability
from dbt.adapters.events.types import FinishedRunningStats
from dbt.adapters.exceptions import MissingMaterializationError
from dbt.artifacts.resources import Hook
from dbt.artifacts.schemas.batch_results import BatchResults, BatchType
from dbt.artifacts.schemas.results import NodeStatus, RunningStatus, RunStatus, TimingInfo
from dbt.artifacts.schemas.run import RunResult
from dbt.cli.flags import Flags
from dbt.clients.jinja import MacroGenerator
from dbt.config import RuntimeConfig
from dbt.context.providers import generate_runtime_model_context
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import BatchContext, HookNode, ModelNode, ResultNode
from dbt.events.types import (
    GenericExceptionOnRun,
    LogBatchResult,
    LogHookEndLine,
    LogHookStartLine,
    LogModelResult,
    LogStartBatch,
    LogStartLine,
    MicrobatchExecutionDebug,
)
from dbt.exceptions import CompilationError, DbtInternalError, DbtRuntimeError
from dbt.graph import ResourceTypeSelector
from dbt.hooks import get_hook_dict
from dbt.materializations.incremental.microbatch import MicrobatchBuilder
from dbt.node_types import NodeType, RunHookType
from dbt.task import group_lookup
from dbt.task.base import BaseRunner
from dbt.task.compile import CompileRunner, CompileTask
from dbt.task.printer import get_counts, print_run_end_messages
from dbt_common.clients.jinja import MacroProtocol
from dbt_common.dataclass_schema import dbtClassMixin
from dbt_common.events.base_types import EventLevel
from dbt_common.events.contextvars import log_contextvars
from dbt_common.events.functions import fire_event, get_invocation_id
from dbt_common.events.types import Formatting
from dbt_common.exceptions import DbtValidationError


@functools.total_ordering
class BiggestName(str):
    def __lt__(self, other: Any) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...


def _hook_list() -> List[Any]: ...


def get_hooks_by_tags(
    nodes: Iterable[Any], match_tags: Set[str]
) -> List[HookNode]: ...


def get_hook(source: Any, index: int) -> Hook: ...


def get_execution_status(
    sql: str, adapter: BaseAdapter
) -> Tuple[RunStatus, str]: ...


def _get_adapter_info(
    adapter: Optional[BaseAdapter], run_model_result: RunResult
) -> Dict[str, Any]: ...


def track_model_run(
    index: int,
    num_nodes: int,
    run_model_result: RunResult,
    adapter: Optional[BaseAdapter] = None,
) -> None: ...


def _validate_materialization_relations_dict(
    inp: Dict[str, Any], model: ModelNode
) -> List[BaseRelation]: ...


class ModelRunner(CompileRunner):
    def get_node_representation(self) -> str: ...
    def describe_node(self) -> str: ...
    def print_start_line(self) -> None: ...
    def print_result_line(self, result: RunResult) -> None: ...
    def before_execute(self) -> None: ...
    def after_execute(self, result: RunResult) -> None: ...
    def _build_run_model_result(
        self, model: ModelNode, context: Dict[str, Any], elapsed_time: float = 0.0
    ) -> RunResult: ...
    def _materialization_relations(
        self, result: Any, model: ModelNode
    ) -> List[BaseRelation]: ...
    def _execute_model(
        self,
        hook_ctx: Any,
        context_config: Any,
        model: ModelNode,
        context: Dict[str, Any],
        materialization_macro: MacroProtocol,
    ) -> RunResult: ...
    def execute(self, model: ModelNode, manifest: Manifest) -> RunResult: ...


class MicrobatchModelRunner(ModelRunner):
    batch_idx: Optional[int]
    batches: Dict[int, Any]
    relation_exists: bool

    def __init__(
        self,
        config: RuntimeConfig,
        adapter: BaseAdapter,
        node: ModelNode,
        node_index: int,
        num_nodes: int,
    ) -> None: ...
    def compile(self, manifest: Manifest) -> ModelNode: ...
    def set_batch_idx(self, batch_idx: Optional[int]) -> None: ...
    def set_relation_exists(self, relation_exists: bool) -> None: ...
    def set_batches(self, batches: Dict[int, Any]) -> None: ...
    @property
    def batch_start(self) -> Optional[Any]: ...
    def describe_node(self) -> str: ...
    def describe_batch(self) -> str: ...
    def print_batch_result_line(self, result: RunResult) -> None: ...
    def print_batch_start_line(self) -> None: ...
    def before_execute(self) -> None: ...
    def after_execute(self, result: RunResult) -> None: ...
    def merge_batch_results(
        self, result: RunResult, batch_results: List[RunResult]
    ) -> None: ...
    def on_skip(self) -> RunResult: ...
    def _build_succesful_run_batch_result(
        self,
        model: ModelNode,
        context: Dict[str, Any],
        batch: Any,
        elapsed_time: float = 0.0,
    ) -> RunResult: ...
    def _build_failed_run_batch_result(
        self, model: ModelNode, batch: Any, elapsed_time: float = 0.0
    ) -> RunResult: ...
    def _build_run_microbatch_model_result(self, model: ModelNode) -> RunResult: ...
    def _execute_microbatch_materialization(
        self,
        model: ModelNode,
        context: Dict[str, Any],
        materialization_macro: MacroProtocol,
    ) -> RunResult: ...
    def _has_relation(self, model: ModelNode) -> bool: ...
    def should_run_in_parallel(self) -> bool: ...
    def _is_incremental(self, model: ModelNode) -> bool: ...
    def _execute_model(
        self,
        hook_ctx: Any,
        context_config: Any,
        model: ModelNode,
        context: Dict[str, Any],
        materialization_macro: MacroProtocol,
    ) -> RunResult: ...


class RunTask(CompileTask):
    batch_map: Optional[Dict[str, Any]]

    def __init__(
        self,
        args: Flags,
        config: RuntimeConfig,
        manifest: Manifest,
        batch_map: Optional[Dict[str, Any]] = None,
    ) -> None: ...
    def raise_on_first_error(self) -> bool: ...
    def get_hook_sql(
        self,
        adapter: BaseAdapter,
        hook: HookNode,
        idx: int,
        num_hooks: int,
        extra_context: Dict[str, Any],
    ) -> str: ...
    def handle_job_queue(
        self, pool: ThreadPool, callback: Callable[[Any], None]
    ) -> None: ...
    def handle_microbatch_model(
        self, runner: MicrobatchModelRunner, pool: ThreadPool
    ) -> RunResult: ...
    def _submit_batch(
        self,
        node: ModelNode,
        adapter: BaseAdapter,
        relation_exists: bool,
        batches: Dict[int, Any],
        batch_idx: int,
        batch_results: List[RunResult],
        pool: ThreadPool,
        force_sequential_run: bool = False,
        skip: bool = False,
    ) -> bool: ...
    def _hook_keyfunc(self, hook: HookNode) -> Tuple[Union[str, BiggestName], int]: ...
    def get_hooks_by_type(self, hook_type: RunHookType) -> List[HookNode]: ...
    def safe_run_hooks(
        self,
        adapter: BaseAdapter,
        hook_type: RunHookType,
        extra_context: Dict[str, Any],
    ) -> RunStatus: ...
    def print_results_line(
        self, results: List[RunResult], execution_time: Optional[float]
    ) -> None: ...
    def populate_microbatch_batches(self, selected_uids: List[str]) -> None: ...
    def before_run(
        self, adapter: BaseAdapter, selected_uids: List[str]
    ) -> RunStatus: ...
    def after_run(
        self, adapter: BaseAdapter, results: List[RunResult]
    ) -> None: ...
    def get_node_selector(self) -> ResourceTypeSelector: ...
    def get_runner_type(self, node: ModelNode) -> Type[BaseRunner]: ...
    def task_end_messages(self, results: List[RunResult]) -> None: ...
```