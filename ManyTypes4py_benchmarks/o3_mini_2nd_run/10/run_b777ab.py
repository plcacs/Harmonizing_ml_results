#!/usr/bin/env python3
import functools
import os
import threading
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import AbstractSet, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

from dbt import tracking, utils
from dbt.adapters.base import BaseAdapter, BaseRelation
from dbt.adapters.capability import Capability
from dbt.adapters.events.types import FinishedRunningStats
from dbt.adapters.exceptions import MissingMaterializationError
from dbt.artifacts.resources import Hook
from dbt.artifacts.schemas.batch_results import BatchResults, BatchType
from dbt.artifacts.schemas.results import NodeStatus, RunningStatus, RunStatus, TimingInfo, collect_timing_info
from dbt.artifacts.schemas.run import RunResult
from dbt.cli.flags import Flags
from dbt.clients.jinja import MacroGenerator
from dbt.config import RuntimeConfig
from dbt.context.providers import generate_runtime_model_context
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import BatchContext, HookNode, ModelNode, ResultNode
from dbt.events.types import GenericExceptionOnRun, LogBatchResult, LogHookEndLine, LogHookStartLine, LogModelResult, LogStartBatch, LogStartLine, MicrobatchExecutionDebug
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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)


def _hook_list() -> List[Any]:
    return []


def get_hooks_by_tags(nodes: Iterable[Any], match_tags: AbstractSet[str]) -> List[HookNode]:
    matched_nodes: List[HookNode] = []
    for node in nodes:
        if not isinstance(node, HookNode):
            continue
        node_tags = node.tags
        if len(set(node_tags) & match_tags):
            matched_nodes.append(node)
    return matched_nodes


def get_hook(source: Any, index: int) -> Hook:
    hook_dict: Dict[str, Any] = get_hook_dict(source)
    hook_dict.setdefault('index', index)
    Hook.validate(hook_dict)
    return Hook.from_dict(hook_dict)


def get_execution_status(sql: str, adapter: BaseAdapter) -> Tuple[RunStatus, str]:
    if not sql.strip():
        return (RunStatus.Success, 'OK')
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


def track_model_run(index: int, num_nodes: int, run_model_result: RunResult, adapter: Optional[BaseAdapter] = None) -> None:
    if tracking.active_user is None:
        raise DbtInternalError('cannot track model run with no active user')
    invocation_id: str = get_invocation_id()
    node = run_model_result.node
    has_group: bool = True if hasattr(node, 'group') and node.group else False
    if node.resource_type == NodeType.Model:
        access: Optional[str] = node.access.value if node.access is not None else None
        contract_enforced: bool = node.contract.enforced
        versioned: bool = True if node.version else False
        incremental_strategy: Optional[str] = node.config.incremental_strategy
    else:
        access = None
        contract_enforced = False
        versioned = False
        incremental_strategy = None
    tracking.track_model_run({
        'invocation_id': invocation_id,
        'index': index,
        'total': num_nodes,
        'execution_time': run_model_result.execution_time,
        'run_status': str(run_model_result.status).upper(),
        'run_skipped': run_model_result.status == NodeStatus.Skipped,
        'run_error': run_model_result.status == NodeStatus.Error,
        'model_materialization': node.get_materialization(),
        'model_incremental_strategy': incremental_strategy,
        'model_id': utils.get_hash(node),
        'hashed_contents': utils.get_hashed_contents(node),
        'timing': [t.to_dict(omit_none=True) for t in run_model_result.timing],
        'language': str(node.language),
        'has_group': has_group,
        'contract_enforced': contract_enforced,
        'access': access,
        'versioned': versioned,
        'adapter_info': _get_adapter_info(adapter, run_model_result)
    })


def _validate_materialization_relations_dict(inp: Dict[str, Any], model: ModelNode) -> List[BaseRelation]:
    try:
        relations_value = inp['relations']
    except KeyError:
        msg = 'Invalid return value from materialization, "relations" not found, got keys: {}'.format(list(inp))
        raise CompilationError(msg, node=model) from None
    if not isinstance(relations_value, list):
        msg = 'Invalid return value from materialization, "relations" not a list, got: {}'.format(relations_value)
        raise CompilationError(msg, node=model) from None
    relations: List[BaseRelation] = []
    for relation in relations_value:
        if not isinstance(relation, BaseRelation):
            msg = 'Invalid return value from materialization, "relations" contains non-Relation: {}'.format(relation)
            raise CompilationError(msg, node=model)
        assert isinstance(relation, BaseRelation)
        relations.append(relation)
    return relations


class ModelRunner(CompileRunner):
    def get_node_representation(self) -> str:
        display_quote_policy = {'database': False, 'schema': False, 'identifier': False}
        relation = self.adapter.Relation.create_from(self.config, self.node, quote_policy=display_quote_policy)
        if self.node.database == self.config.credentials.database:
            relation = relation.include(database=False)
        return str(relation)

    def describe_node(self) -> str:
        return f'{self.node.language} {self.node.get_materialization()} model {self.get_node_representation()}'

    def print_start_line(self) -> None:
        fire_event(LogStartLine(description=self.describe_node(), index=self.node_index, total=self.num_nodes, node_info=self.node.node_info))

    def print_result_line(self, result: RunResult) -> None:
        description: str = self.describe_node()
        group = group_lookup.get(self.node.unique_id)
        if result.status == NodeStatus.Error:
            status = result.status
            level = EventLevel.ERROR
        else:
            status = result.message
            level = EventLevel.INFO
        fire_event(LogModelResult(description=description, status=status, index=self.node_index, total=self.num_nodes,
                                  execution_time=result.execution_time, node_info=self.node.node_info, group=group), level=level)

    def before_execute(self) -> None:
        self.print_start_line()

    def after_execute(self, result: RunResult) -> None:
        track_model_run(self.node_index, self.num_nodes, result, adapter=self.adapter)
        self.print_result_line(result)

    def _build_run_model_result(self, model: ModelNode, context: Dict[str, Any], elapsed_time: float = 0.0) -> RunResult:
        result = context['load_result']('main')
        if not result:
            raise DbtRuntimeError('main is not being called during running model')
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
            failures=result.get('failures'),
            batch_results=None
        )

    def _materialization_relations(self, result: Any, model: ModelNode) -> List[BaseRelation]:
        if isinstance(result, str):
            msg = 'The materialization ("{}") did not explicitly return a list of relations to add to the cache.'.format(str(model.get_materialization()))
            raise CompilationError(msg, node=model)
        if isinstance(result, dict):
            return _validate_materialization_relations_dict(result, model)
        msg = 'Invalid return value from materialization, expected a dict with key "relations", got: {}'.format(str(result))
        raise CompilationError(msg, node=model)

    def _execute_model(self, hook_ctx: Any, context_config: Dict[str, Any], model: ModelNode, context: Dict[str, Any],
                       materialization_macro: MacroProtocol) -> RunResult:
        try:
            result = MacroGenerator(materialization_macro, context, stack=context['context_macro_stack'])()
        finally:
            self.adapter.post_model_hook(context_config, hook_ctx)
        for relation in self._materialization_relations(result, model):
            self.adapter.cache_added(relation.incorporate(dbt_created=True))
        return self._build_run_model_result(model, context)

    def execute(self, model: ModelNode, manifest: Manifest) -> RunResult:
        context: Dict[str, Any] = generate_runtime_model_context(model, self.config, manifest)
        materialization_macro = manifest.find_materialization_macro_by_name(self.config.project_name, model.get_materialization(), self.adapter.type())
        if materialization_macro is None:
            raise MissingMaterializationError(materialization=model.get_materialization(), adapter_type=self.adapter.type())
        if 'config' not in context:
            raise DbtInternalError('Invalid materialization context generated, missing config: {}'.format(context))
        context_config: Dict[str, Any] = context['config']
        mat_has_supported_langs: bool = hasattr(materialization_macro, 'supported_languages')
        model_lang_supported: bool = model.language in materialization_macro.supported_languages  # type: ignore
        if mat_has_supported_langs and (not model_lang_supported):
            str_langs = [str(lang) for lang in materialization_macro.supported_languages]  # type: ignore
            raise DbtValidationError(f'Materialization "{materialization_macro.name}" only supports languages {str_langs}; got "{model.language}"')
        hook_ctx = self.adapter.pre_model_hook(context_config)
        return self._execute_model(hook_ctx, context_config, model, context, materialization_macro)


class MicrobatchModelRunner(ModelRunner):
    def __init__(self, config: RuntimeConfig, adapter: BaseAdapter, node: ModelNode, node_index: int, num_nodes: int) -> None:
        super().__init__(config, adapter, node, node_index, num_nodes)
        self.batch_idx: Optional[int] = None
        self.batches: Dict[int, Tuple[float, float]] = {}
        self.relation_exists: bool = False

    def compile(self, manifest: Manifest) -> ModelNode:
        if self.batch_idx is not None:
            batch: Tuple[float, float] = self.batches[self.batch_idx]
            self.node.config['__dbt_internal_microbatch_event_time_start'] = batch[0]
            self.node.config['__dbt_internal_microbatch_event_time_end'] = batch[1]
            self.node.batch = BatchContext(
                id=MicrobatchBuilder.batch_id(batch[0], self.node.config.batch_size),
                event_time_start=batch[0],
                event_time_end=batch[1]
            )
            self.compiler.compile_node(self.node, manifest, {}, split_suffix=MicrobatchBuilder.format_batch_start(batch[0], self.node.config.batch_size))
        return self.node

    def set_batch_idx(self, batch_idx: int) -> None:
        self.batch_idx = batch_idx

    def set_relation_exists(self, relation_exists: bool) -> None:
        self.relation_exists = relation_exists

    def set_batches(self, batches: Dict[int, Tuple[float, float]]) -> None:
        self.batches = batches

    @property
    def batch_start(self) -> Optional[float]:
        if self.batch_idx is None:
            return None
        else:
            return self.batches[self.batch_idx][0]

    def describe_node(self) -> str:
        return f'{self.node.language} microbatch model {self.get_node_representation()}'

    def describe_batch(self) -> str:
        batch_start = self.batch_start
        if batch_start is None:
            return ''
        formatted_batch_start = MicrobatchBuilder.format_batch_start(batch_start, self.node.config.batch_size)
        return f'batch {formatted_batch_start} of {self.get_node_representation()}'

    def print_batch_result_line(self, result: RunResult) -> None:
        if self.batch_idx is None:
            return
        description: str = self.describe_batch()
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
        fire_event(LogBatchResult(description=description, status=status, batch_index=self.batch_idx + 1,
                                  total_batches=len(self.batches), execution_time=result.execution_time,
                                  node_info=self.node.node_info, group=group), level=level)

    def print_batch_start_line(self) -> None:
        if self.batch_idx is None:
            return
        batch_start = self.batches[self.batch_idx][0]
        if batch_start is None:
            return
        batch_description: str = self.describe_batch()
        fire_event(LogStartBatch(description=batch_description, batch_index=self.batch_idx + 1,
                                 total_batches=len(self.batches), node_info=self.node.node_info))

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
        num_successes: int = len(result.batch_results.successful)
        num_failures: int = len(result.batch_results.failed)
        if num_failures == 0:
            status = RunStatus.Success
            msg = 'SUCCESS'
        elif num_successes == 0:
            status = RunStatus.Error
            msg = 'ERROR'
        else:
            status = RunStatus.PartialSuccess
            msg = f'PARTIAL SUCCESS ({num_successes}/{num_successes + num_failures})'
        result.status = status
        result.message = msg
        result.batch_results.successful = sorted(result.batch_results.successful)
        result.batch_results.failed = sorted(result.batch_results.failed)
        if self.node.previous_batch_results is not None:
            result.batch_results.successful += self.node.previous_batch_results.successful

    def on_skip(self) -> RunResult:
        if self.batch_idx is None:
            return super().on_skip()  # type: ignore
        else:
            result = RunResult(
                node=self.node,
                status=RunStatus.Skipped,
                timing=[],
                thread_id=threading.current_thread().name,
                execution_time=0.0,
                message='SKIPPED',
                adapter_response={},
                failures=1,
                batch_results=BatchResults(failed=[self.batches[self.batch_idx]])
            )
            self.print_batch_result_line(result=result)
            return result

    def _build_succesful_run_batch_result(self, model: ModelNode, context: Dict[str, Any],
                                          batch: Tuple[float, float], elapsed_time: float = 0.0) -> RunResult:
        run_result = self._build_run_model_result(model, context, elapsed_time)
        run_result.batch_results = BatchResults(successful=[batch])
        return run_result

    def _build_failed_run_batch_result(self, model: ModelNode, batch: Tuple[float, float], elapsed_time: float = 0.0) -> RunResult:
        return RunResult(
            node=model,
            status=RunStatus.Error,
            timing=[],
            thread_id=threading.current_thread().name,
            execution_time=elapsed_time,
            message='ERROR',
            adapter_response={},
            failures=1,
            batch_results=BatchResults(failed=[batch])
        )

    def _build_run_microbatch_model_result(self, model: ModelNode) -> RunResult:
        return RunResult(
            node=model,
            status=RunStatus.Success,
            timing=[],
            thread_id=threading.current_thread().name,
            execution_time=0,
            message='',
            adapter_response={},
            failures=0,
            batch_results=BatchResults()
        )

    def _execute_microbatch_materialization(self, model: ModelNode, context: Dict[str, Any],
                                              materialization_macro: MacroProtocol) -> RunResult:
        event_time_start: Optional[float] = getattr(self.config.args, 'EVENT_TIME_START', None)
        event_time_end: Optional[float] = getattr(self.config.args, 'EVENT_TIME_END', None)
        if os.environ.get('DBT_EXPERIMENTAL_SAMPLE_MODE') and getattr(self.config.args, 'SAMPLE', None):
            event_time_start = self.config.args.sample.start
            event_time_end = self.config.args.sample.end
        microbatch_builder = MicrobatchBuilder(
            model=model,
            is_incremental=self._is_incremental(model),
            event_time_start=event_time_start,
            event_time_end=event_time_end,
            default_end_time=self.config.invoked_at
        )
        if self.batch_idx is None:
            if model.previous_batch_results is None:
                end = microbatch_builder.build_end_time()
                start = microbatch_builder.build_start_time(end)
                batches = microbatch_builder.build_batches(start, end)
            else:
                batches = model.previous_batch_results.failed  # type: ignore
                if self._has_relation(model=model):
                    self.relation_exists = True
            batch_result = self._build_run_microbatch_model_result(model)
            self.batches = {batch_idx: batches[batch_idx] for batch_idx in range(len(batches))}
        else:
            batch = self.batches[self.batch_idx]
            start_time = time.perf_counter()
            try:
                jinja_context: Dict[str, Any] = microbatch_builder.build_jinja_context_for_batch(incremental_batch=self.relation_exists)
                context.update(jinja_context)
                result = MacroGenerator(materialization_macro, context, stack=context['context_macro_stack'])()
                for relation in self._materialization_relations(result, model):
                    self.adapter.cache_added(relation.incorporate(dbt_created=True))
                batch_run_result = self._build_succesful_run_batch_result(model, context, batch, time.perf_counter() - start_time)
                batch_result = batch_run_result
                self.relation_exists = True
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                fire_event(GenericExceptionOnRun(unique_id=self.node.unique_id, exc=f'Exception on worker thread. {str(e)}', node_info=self.node.node_info))
                batch_run_result = self._build_failed_run_batch_result(model, batch, time.perf_counter() - start_time)
            batch_result = batch_run_result
        return batch_result

    def _has_relation(self, model: ModelNode) -> bool:
        relation_info = self.adapter.Relation.create_from(self.config, model)
        relation = self.adapter.get_relation(relation_info.database, relation_info.schema, relation_info.name)
        return relation is not None

    def should_run_in_parallel(self) -> bool:
        if not self.adapter.supports(Capability.MicrobatchConcurrency):
            run_in_parallel = False
        elif not self.relation_exists:
            run_in_parallel = False
        elif self.node.config.concurrent_batches is not None:
            run_in_parallel = self.node.config.concurrent_batches
        else:
            run_in_parallel = not self.node.has_this
        return run_in_parallel

    def _is_incremental(self, model: ModelNode) -> bool:
        relation_info = self.adapter.Relation.create_from(self.config, model)
        relation = self.adapter.get_relation(relation_info.database, relation_info.schema, relation_info.name)
        if relation is not None and relation.type == 'table' and (model.config.materialized == 'incremental'):
            if model.config.full_refresh is not None:
                return not model.config.full_refresh
            else:
                return not getattr(self.config.args, 'FULL_REFRESH', False)
        else:
            return False

    def _execute_model(self, hook_ctx: Any, context_config: Dict[str, Any], model: ModelNode,
                       context: Dict[str, Any], materialization_macro: MacroProtocol) -> RunResult:
        try:
            batch_result = self._execute_microbatch_materialization(model, context, materialization_macro)
        finally:
            self.adapter.post_model_hook(context_config, hook_ctx)
        return batch_result


class RunTask(CompileTask):
    def __init__(self, args: Flags, config: RuntimeConfig, manifest: Manifest, batch_map: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(args, config, manifest)
        self.batch_map: Optional[Dict[str, Any]] = batch_map

    def raise_on_first_error(self) -> bool:
        return False

    def get_hook_sql(self, adapter: BaseAdapter, hook: HookNode, idx: int, num_hooks: int, extra_context: Dict[str, Any]) -> str:
        if self.manifest is None:
            raise DbtInternalError('compile_node called before manifest was loaded')
        compiled = self.compiler.compile_node(hook, self.manifest, extra_context)
        statement: str = compiled.compiled_code
        hook_index: int = hook.index or num_hooks
        hook.index = hook_index
        hook_obj = get_hook(statement, index=hook_index)
        return hook_obj.sql or ''

    def handle_job_queue(self, pool: ThreadPool, callback: Callable[[RunResult], None]) -> None:
        node = self.job_queue.get()
        self._raise_set_error()
        runner = self.get_runner(node)
        if runner.node.unique_id in self._skipped_children:
            cause = self._skipped_children.pop(runner.node.unique_id)
            runner.do_skip(cause=cause)
        if isinstance(runner, MicrobatchModelRunner):
            callback(self.handle_microbatch_model(runner, pool))
        else:
            args = [runner]
            self._submit(pool, args, callback)

    def handle_microbatch_model(self, runner: MicrobatchModelRunner, pool: ThreadPool) -> RunResult:
        result = self.call_runner(runner)
        batches: Dict[int, Tuple[float, float]] = runner.batches
        node = runner.node
        relation_exists: bool = runner.relation_exists
        if result.status == RunStatus.Skipped:
            return result
        elif len(runner.batches) == 0:
            return result
        batch_results: List[RunResult] = []
        batch_idx: int = 0
        relation_exists = self._submit_batch(node=node, adapter=runner.adapter, relation_exists=relation_exists,
                                               batches=batches, batch_idx=batch_idx, batch_results=batch_results, pool=pool, force_sequential_run=True)
        batch_idx += 1
        skip_batches: bool = batch_results[0].status != RunStatus.Success
        while batch_idx < len(runner.batches) - 1:
            relation_exists = self._submit_batch(node=node, adapter=runner.adapter, relation_exists=relation_exists,
                                                   batches=batches, batch_idx=batch_idx, batch_results=batch_results, pool=pool, skip=skip_batches)
            batch_idx += 1
        while len(batch_results) != batch_idx:
            pass
        if len(batches) != 1:
            self._submit_batch(node=node, adapter=runner.adapter, relation_exists=relation_exists,
                               batches=batches, batch_idx=batch_idx, batch_results=batch_results, pool=pool, force_sequential_run=True, skip=skip_batches)
        runner.merge_batch_results(result, batch_results)
        track_model_run(runner.node_index, runner.num_nodes, result, adapter=runner.adapter)
        runner.print_result_line(result)
        return result

    def _submit_batch(self, node: ModelNode, adapter: BaseAdapter, relation_exists: bool, batches: Dict[int, Tuple[float, float]],
                      batch_idx: int, batch_results: List[RunResult], pool: ThreadPool,
                      force_sequential_run: bool = False, skip: bool = False) -> bool:
        node_copy = deepcopy(node)
        if batch_idx != 0:
            node_copy.config.pre_hook = []
        if batch_idx != len(batches) - 1:
            node_copy.config.post_hook = []
        batch_runner = MicrobatchModelRunner(self.config, adapter, node_copy, self.run_count, self.num_nodes)
        batch_runner.set_batch_idx(batch_idx)
        batch_runner.set_relation_exists(relation_exists)
        batch_runner.set_batches(batches)
        if skip:
            batch_runner.do_skip()
        if not force_sequential_run and batch_runner.should_run_in_parallel():
            fire_event(MicrobatchExecutionDebug(msg=f'{batch_runner.describe_batch()} is being run concurrently'))
            self._submit(pool, [batch_runner], batch_results.append)
        else:
            fire_event(MicrobatchExecutionDebug(msg=f'{batch_runner.describe_batch()} is being run sequentially'))
            batch_results.append(self.call_runner(batch_runner))
            relation_exists = batch_runner.relation_exists
        return relation_exists

    def _hook_keyfunc(self, hook: HookNode) -> Tuple[Any, Any]:
        package_name: Any = hook.package_name
        if package_name == self.config.project_name:
            package_name = BiggestName('')
        return (package_name, hook.index)

    def get_hooks_by_type(self, hook_type: RunHookType) -> List[HookNode]:
        if self.manifest is None:
            raise DbtInternalError('self.manifest was None in get_hooks_by_type')
        nodes = self.manifest.nodes.values()
        hooks = get_hooks_by_tags(nodes, {hook_type})
        hooks.sort(key=self._hook_keyfunc)
        return hooks

    def safe_run_hooks(self, adapter: BaseAdapter, hook_type: RunHookType, extra_context: Dict[str, Any]) -> RunStatus:
        ordered_hooks: List[HookNode] = self.get_hooks_by_type(hook_type)
        if hook_type == RunHookType.End and ordered_hooks:
            fire_event(Formatting(''))
        adapter.clear_transaction()
        if not ordered_hooks:
            return RunStatus.Success
        status: RunStatus = RunStatus.Success
        failed: bool = False
        num_hooks: int = len(ordered_hooks)
        for idx, hook in enumerate(ordered_hooks, 1):
            with log_contextvars(node_info=hook.node_info):
                hook.index = idx
                hook_name: str = f'{hook.package_name}.{hook_type}.{hook.index - 1}'
                execution_time: float = 0.0
                timing: List[TimingInfo] = []
                failures: int = 1
                if not failed:
                    with collect_timing_info('compile', timing.append):
                        sql: str = self.get_hook_sql(adapter, hook, hook.index, num_hooks, extra_context)
                    started_at: datetime = timing[0].started_at or datetime.utcnow()  # type: ignore
                    hook.update_event_status(started_at=started_at.isoformat(), node_status=RunningStatus.Started)
                    fire_event(LogHookStartLine(statement=hook_name, index=hook.index, total=num_hooks, node_info=hook.node_info))
                    with collect_timing_info('execute', timing.append):
                        status, message = get_execution_status(sql, adapter)
                    finished_at: datetime = timing[1].completed_at or datetime.utcnow()  # type: ignore
                    hook.update_event_status(finished_at=finished_at.isoformat())
                    execution_time = (finished_at - started_at).total_seconds()
                    failures = 0 if status == RunStatus.Success else 1
                    if status == RunStatus.Success:
                        message = f'{hook_name} passed'
                    else:
                        message = f'{hook_name} failed, error:\n {message}'
                        failed = True
                else:
                    status = RunStatus.Skipped
                    message = f'{hook_name} skipped'
                hook.update_event_status(node_status=status)
                self.node_results.append(RunResult(
                    status=status,
                    thread_id='main',
                    timing=timing,
                    message=message,
                    adapter_response={},
                    execution_time=execution_time,
                    failures=failures,
                    node=hook
                ))
                fire_event(LogHookEndLine(statement=hook_name, status=status, index=hook.index, total=num_hooks, execution_time=execution_time, node_info=hook.node_info))
        if hook_type == RunHookType.Start and ordered_hooks:
            fire_event(Formatting(''))
        return status

    def print_results_line(self, results: List[RunResult], execution_time: float) -> None:
        nodes = [r.node for r in results if hasattr(r, 'node')]
        stat_line: str = get_counts(nodes)
        execution: str = ''
        if execution_time is not None:
            execution = utils.humanize_execution_time(execution_time=execution_time)
        fire_event(Formatting(''))
        fire_event(FinishedRunningStats(stat_line=stat_line, execution=execution, execution_time=execution_time))

    def populate_microbatch_batches(self, selected_uids: List[str]) -> None:
        if self.batch_map is not None and self.manifest is not None:
            for uid in selected_uids:
                if uid in self.batch_map:
                    node = self.manifest.ref_lookup.perform_lookup(uid, self.manifest)
                    if isinstance(node, ModelNode):
                        node.previous_batch_results = self.batch_map[uid]

    def before_run(self, adapter: BaseAdapter, selected_uids: List[str]) -> RunStatus:
        with adapter.connection_named('master'):
            self.defer_to_manifest()
            required_schemas = self.get_model_schemas(adapter, selected_uids)
            self.create_schemas(adapter, required_schemas)
            self.populate_adapter_cache(adapter, required_schemas)
            self.populate_microbatch_batches(selected_uids)
            group_lookup.init(self.manifest, selected_uids)
            run_hooks_status: RunStatus = self.safe_run_hooks(adapter, RunHookType.Start, {})
            return run_hooks_status

    def after_run(self, adapter: BaseAdapter, results: List[RunResult]) -> None:
        database_schema_set = {(r.node.database, r.node.schema) for r in results if (hasattr(r, 'node') and r.node.is_relational) and r.status not in (NodeStatus.Error, NodeStatus.Fail, NodeStatus.Skipped)}
        extras: Dict[str, Any] = {
            'schemas': list({s for _, s in database_schema_set}),
            'results': [r for r in results if r.thread_id != 'main' or r.status == RunStatus.Error],
            'database_schemas': list(database_schema_set)
        }
        try:
            with adapter.connection_named('master'):
                self.safe_run_hooks(adapter, RunHookType.End, extras)
        except (KeyboardInterrupt, SystemExit, DbtRuntimeError):
            run_result = self.get_result(results=self.node_results, elapsed_time=time.time() - self.started_at, generated_at=datetime.utcnow())
            if self.args.write_json and hasattr(run_result, 'write'):
                run_result.write(self.result_path())
            print_run_end_messages(self.node_results, keyboard_interrupt=True)
            raise

    def get_node_selector(self) -> ResourceTypeSelector:
        if self.manifest is None or self.graph is None:
            raise DbtInternalError('manifest and graph must be set to get perform node selection')
        return ResourceTypeSelector(graph=self.graph, manifest=self.manifest, previous_state=self.previous_state, resource_types=[NodeType.Model])

    def get_runner_type(self, node: ModelNode) -> Type[ModelRunner]:
        if self.manifest is None:
            raise DbtInternalError('manifest must be set prior to calling get_runner_type')
        if node.config.materialized == 'incremental' and node.config.incremental_strategy == 'microbatch' and self.manifest.use_microbatch_batches(project_name=self.config.project_name):
            return MicrobatchModelRunner
        else:
            return ModelRunner

    def task_end_messages(self, results: List[RunResult]) -> None:
        if results:
            print_run_end_messages(results)
