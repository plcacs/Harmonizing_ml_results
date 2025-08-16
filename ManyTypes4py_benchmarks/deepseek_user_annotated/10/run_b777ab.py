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
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    Callable,
    cast,
)

from dbt import tracking, utils
from dbt.adapters.base import BaseAdapter, BaseRelation
from dbt.adapters.capability import Capability
from dbt.adapters.events.types import FinishedRunningStats
from dbt.adapters.exceptions import MissingMaterializationError
from dbt.artifacts.resources import Hook
from dbt.artifacts.schemas.batch_results import BatchResults, BatchType
from dbt.artifacts.schemas.results import (
    NodeStatus,
    RunningStatus,
    RunStatus,
    TimingInfo,
    collect_timing_info,
)
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
    LogStartLine,
    LogStartBatch,
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
    def __lt__(self, other: Any) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)


def _hook_list() -> List[HookNode]:
    return []


def get_hooks_by_tags(
    nodes: Iterable[ResultNode],
    match_tags: Set[str],
) -> List[HookNode]:
    matched_nodes: List[HookNode] = []
    for node in nodes:
        if not isinstance(node, HookNode):
            continue
        node_tags = node.tags
        if len(set(node_tags) & match_tags):
            matched_nodes.append(node)
    return matched_nodes


def get_hook(source: str, index: int) -> Hook:
    hook_dict = get_hook_dict(source)
    hook_dict.setdefault("index", index)
    Hook.validate(hook_dict)
    return Hook.from_dict(hook_dict)


def get_execution_status(sql: str, adapter: BaseAdapter) -> Tuple[RunStatus, str]:
    if not sql.strip():
        return RunStatus.Success, "OK"

    try:
        response, _ = adapter.execute(sql, auto_begin=False, fetch=False)
        status = RunStatus.Success
        message = response._message
    except (KeyboardInterrupt, SystemExit):
        raise
    except DbtRuntimeError as exc:
        status = RunStatus.Error
        message = exc.msg
    except Exception as exc:
        status = RunStatus.Error
        message = str(exc)

    return (status, message)


def _get_adapter_info(adapter: Optional[BaseAdapter], run_model_result: RunResult) -> Dict[str, Any]:
    """Each adapter returns a dataclass with a flexible dictionary for
    adapter-specific fields. Only the non-'model_adapter_details' fields
    are guaranteed cross adapter."""
    return asdict(adapter.get_adapter_run_info(run_model_result.node.config)) if adapter else {}


def track_model_run(
    index: int, num_nodes: int, run_model_result: RunResult, adapter: Optional[BaseAdapter] = None
) -> None:
    if tracking.active_user is None:
        raise DbtInternalError("cannot track model run with no active user")
    invocation_id = get_invocation_id()
    node = run_model_result.node
    has_group = True if hasattr(node, "group") and node.group else False
    if node.resource_type == NodeType.Model:
        access = node.access.value if node.access is not None else None
        contract_enforced = node.contract.enforced
        versioned = True if node.version else False
        incremental_strategy = node.config.incremental_strategy
    else:
        access = None
        contract_enforced = False
        versioned = False
        incremental_strategy = None

    tracking.track_model_run(
        {
            "invocation_id": invocation_id,
            "index": index,
            "total": num_nodes,
            "execution_time": run_model_result.execution_time,
            "run_status": str(run_model_result.status).upper(),
            "run_skipped": run_model_result.status == NodeStatus.Skipped,
            "run_error": run_model_result.status == NodeStatus.Error,
            "model_materialization": node.get_materialization(),
            "model_incremental_strategy": incremental_strategy,
            "model_id": utils.get_hash(node),
            "hashed_contents": utils.get_hashed_contents(node),
            "timing": [t.to_dict(omit_none=True) for t in run_model_result.timing],
            "language": str(node.language),
            "has_group": has_group,
            "contract_enforced": contract_enforced,
            "access": access,
            "versioned": versioned,
            "adapter_info": _get_adapter_info(adapter, run_model_result),
        }
    )


def _validate_materialization_relations_dict(inp: Dict[Any, Any], model: ModelNode) -> List[BaseRelation]:
    try:
        relations_value = inp["relations"]
    except KeyError:
        msg = (
            'Invalid return value from materialization, "relations" '
            "not found, got keys: {}".format(list(inp))
        )
        raise CompilationError(msg, node=model) from None

    if not isinstance(relations_value, list):
        msg = (
            'Invalid return value from materialization, "relations" '
            "not a list, got: {}".format(relations_value)
        )
        raise CompilationError(msg, node=model) from None

    relations: List[BaseRelation] = []
    for relation in relations_value:
        if not isinstance(relation, BaseRelation):
            msg = (
                "Invalid return value from materialization, "
                '"relations" contains non-Relation: {}'.format(relation)
            )
            raise CompilationError(msg, node=model)

        assert isinstance(relation, BaseRelation)
        relations.append(relation)
    return relations


class ModelRunner(CompileRunner):
    def get_node_representation(self) -> str:
        display_quote_policy = {"database": False, "schema": False, "identifier": False}
        relation = self.adapter.Relation.create_from(
            self.config, self.node, quote_policy=display_quote_policy
        )
        # exclude the database from output if it's the default
        if self.node.database == self.config.credentials.database:
            relation = relation.include(database=False)
        return str(relation)

    def describe_node(self) -> str:
        return f"{self.node.language} {self.node.get_materialization()} model {self.get_node_representation()}"

    def print_start_line(self) -> None:
        fire_event(
            LogStartLine(
                description=self.describe_node(),
                index=self.node_index,
                total=self.num_nodes,
                node_info=self.node.node_info,
            )
        )

    def print_result_line(self, result: RunResult) -> None:
        description = self.describe_node()
        group = group_lookup.get(self.node.unique_id)
        if result.status == NodeStatus.Error:
            status = result.status
            level = EventLevel.ERROR
        else:
            status = result.message
            level = EventLevel.INFO
        fire_event(
            LogModelResult(
                description=description,
                status=status,
                index=self.node_index,
                total=self.num_nodes,
                execution_time=result.execution_time,
                node_info=self.node.node_info,
                group=group,
            ),
            level=level,
        )

    def before_execute(self) -> None:
        self.print_start_line()

    def after_execute(self, result: RunResult) -> None:
        track_model_run(self.node_index, self.num_nodes, result, adapter=self.adapter)
        self.print_result_line(result)

    def _build_run_model_result(
        self, model: ModelNode, context: Dict[str, Any], elapsed_time: float = 0.0
    ) -> RunResult:
        result = context["load_result"]("main")
        if not result:
            raise DbtRuntimeError("main is not being called during running model")
        adapter_response: Dict[str, Any] = {}
        if isinstance(result.response, dbtClassMixin):
            adapter_response = result.response.to_dict(omit_none=True)
        return RunResult(
            node=model,
            status=RunStatus.Success,
            timing=[],
            thread_id=threading.current_thread().name,
            execution_time=elapsed_time,
            message=str(result.response),
            adapter_response=adapter_response,
            failures=result.get("failures"),
            batch_results=None,
        )

    def _materialization_relations(self, result: Any, model: ModelNode) -> List[BaseRelation]:
        if isinstance(result, str):
            msg = (
                'The materialization ("{}") did not explicitly return a '
                "list of relations to add to the cache.".format(str(model.get_materialization()))
            )
            raise CompilationError(msg, node=model)

        if isinstance(result, dict):
            return _validate_materialization_relations_dict(result, model)

        msg = (
            "Invalid return value from materialization, expected a dict "
            'with key "relations", got: {}'.format(str(result))
        )
        raise CompilationError(msg, node=model)

    def _execute_model(
        self,
        hook_ctx: Any,
        context_config: Any,
        model: ModelNode,
        context: Dict[str, Any],
        materialization_macro: MacroProtocol,
    ) -> RunResult:
        try:
            result = MacroGenerator(
                materialization_macro, context, stack=context["context_macro_stack"]
            )()
        finally:
            self.adapter.post_model_hook(context_config, hook_ctx)

        for relation in self._materialization_relations(result, model):
            self.adapter.cache_added(relation.incorporate(dbt_created=True))

        return self._build_run_model_result(model, context)

    def execute(self, model: ModelNode, manifest: Manifest) -> RunResult:
        context = generate_runtime_model_context(model, self.config, manifest)

        materialization_macro = manifest.find_materialization_macro_by_name(
            self.config.project_name, model.get_materialization(), self.adapter.type()
        )

        if materialization_macro is None:
            raise MissingMaterializationError(
                materialization=model.get_materialization(), adapter_type=self.adapter.type()
            )

        if "config" not in context:
            raise DbtInternalError(
                "Invalid materialization context generated, missing config: {}".format(context)
            )
        context_config = context["config"]

        mat_has_supported_langs = hasattr(materialization_macro, "supported_languages")
        model_lang_supported = model.language in materialization_macro.supported_languages
        if mat_has_supported_langs and not model_lang_supported:
            str_langs = [str(lang) for lang in materialization_macro.supported_languages]
            raise DbtValidationError(
                f'Materialization "{materialization_macro.name}" only supports languages {str_langs}; '
                f'got "{model.language}"'
            )

        hook_ctx = self.adapter.pre_model_hook(context_config)

        return self._execute_model(hook_ctx, context_config, model, context, materialization_macro)


class MicrobatchModelRunner(ModelRunner):
    def __init__(
        self,
        config: RuntimeConfig,
        adapter: BaseAdapter,
        node: ModelNode,
        node_index: int,
        num_nodes: int,
    ):
        super().__init__(config, adapter, node, node_index, num_nodes)

        self.batch_idx: Optional[int] = None
        self.batches: Dict[int, BatchType] = {}
        self.relation_exists: bool = False

    def compile(self, manifest: Manifest) -> ModelNode:
        if self.batch_idx is not None:
            batch = self.batches[self.batch_idx]

            # LEGACY: Set start/end in context prior to re-compiling (Will be removed for 1.10+)
            # TODO: REMOVE before 1.10 GA
            self.node.config["__dbt_internal_microbatch_event_time_start"] = batch[0]
            self.node.config["__dbt_internal_microbatch_event_time_end"] = batch[1]
            # Create batch context on model node prior to re-compiling
            self.node.batch = BatchContext(
                id=MicrobatchBuilder.batch_id(batch[0], self.node.config.batch_size),
                event_time_start=batch[0],
                event_time_end=batch[1],
            )
            # Recompile node to re-resolve refs with event time filters rendered, update context
            self.compiler.compile_node(
                self.node,
                manifest,
                {},
                split_suffix=MicrobatchBuilder.format_batch_start(
                    batch[0], self.node.config.batch_size
                ),
            )

        # Skips compilation for non-batch runs
        return self.node

    def set_batch_idx(self, batch_idx: int) -> None:
        self.batch_idx = batch_idx

    def set_relation_exists(self, relation_exists: bool) -> None:
        self.relation_exists = relation_exists

    def set_batches(self, batches: Dict[int, BatchType]) -> None:
        self.batches = batches

    @property
    def batch_start(self) -> Optional[datetime]:
        if self.batch_idx is None:
            return None
        else:
            return self.batches[self.batch_idx][0]

    def describe_node(self) -> str:
        return f"{self.node.language} microbatch model {self.get_node_representation()}"

    def describe_batch(self) -> str:
        batch_start = self.batch_start
        if batch_start is None:
            return ""

        # Only visualize date if batch_start year/month/day
        formatted_batch_start = MicrobatchBuilder.format_batch_start(
            batch_start, self.node.config.batch_size
        )
        return f"batch {formatted_batch_start} of {self.get_node_representation()}"

    def print_batch_result_line(self, result: RunResult) -> None:
        if self.batch_idx is None:
            return

        description = self.describe_batch()
        group = group_lookup.get(self.node.unique_id)
        if result.status == NodeStatus.Error:
            status = result.status
            level = EventLevel.ERROR
        elif result.status == NodeStatus.Skipped:
            status = result.status
            level = EventLevel.INFO
        else:
            status = result.message
            level = EventLevel.INFO
        fire_event(
            LogBatchResult(
                description=description,
                status=status,
                batch_index=self.batch_idx + 1,
                total_batches=len(self.batches),
                execution_time=result.execution_time,
                node_info=self.node.node_info,
                group=group,
            ),
            level=level,
        )

    def print_batch_start_line(self) -> None:
        if self.batch_idx is None:
            return

        batch_start = self.batches[self.batch_idx][0]
        if batch_start is None:
            return

        batch_description = self.describe_batch()
        fire_event(
            LogStartBatch(
                description=batch_description,
                batch_index=self.batch_idx + 1,
                total_batches=len(self.batches),
                node_info=self.node.node_info,
            )
        )

    def before_execute(self) -> None:
        if self.batch_idx is None:
            self.print_start_line()
        else:
            self.print_batch_start_line()

    def after_execute(self, result: RunResult) -> None:
        if self.batch_idx is not None:
            self.print_batch_result_line(result)

    def merge_batch_results(self, result: RunResult, batch_results: List[RunResult]) -> None:
        """merge batch_results into result"""
        if result.batch_results is None:
            result.batch_results = BatchResults()

        for batch_result in batch_results:
            if batch_result.batch_results is not None:
                result.batch_results += batch_result.batch_results
            result.execution_time += batch_result.execution_time

        num_successes = len(result.batch_results.successful)
        num_failures = len(result.batch_results.failed)
        if num_failures == 0:
            status = RunStatus.Success
            msg = "SUCCESS"
        elif num_successes == 0:
            status = RunStatus.Error
            msg = "ERROR"
        else:
            status = RunStatus.PartialSuccess
            msg = f"PARTIAL SUCCESS ({num_successes}/{num_successes + num_failures})"
        result.status = status
        result.message = msg

        result.batch_results.successful = sorted(result.batch_results.successful)
        result.batch_results.failed = sorted(result.batch_results.failed)

        # If retrying, propagate previously successful batches into final result, even though they were not run in this invocation
