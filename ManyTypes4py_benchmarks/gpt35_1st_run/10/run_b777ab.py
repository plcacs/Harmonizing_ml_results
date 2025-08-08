from typing import AbstractSet, Any, Dict, Iterable, List, Optional, Set, Tuple, Type
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

class BiggestName(str):
    def __lt__(self, other: Any) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

def _hook_list() -> List:
    return []

def get_hooks_by_tags(nodes: List, match_tags: Set) -> List:
    matched_nodes = []
    for node in nodes:
        if not isinstance(node, HookNode):
            continue
        node_tags = node.tags
        if len(set(node_tags) & match_tags):
            matched_nodes.append(node)
    return matched_nodes

def get_hook(source: Any, index: int) -> Hook:
    hook_dict = get_hook_dict(source)
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

def _get_adapter_info(adapter: BaseAdapter, run_model_result: RunResult) -> Dict:
    return asdict(adapter.get_adapter_run_info(run_model_result.node.config)) if adapter else {}

def track_model_run(index: int, num_nodes: int, run_model_result: RunResult, adapter: Optional[BaseAdapter] = None) -> None:
    if tracking.active_user is None:
        raise DbtInternalError('cannot track model run with no active user')
    invocation_id = get_invocation_id()
    node = run_model_result.node
    has_group = True if hasattr(node, 'group') and node.group else False
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
    tracking.track_model_run({'invocation_id': invocation_id, 'index': index, 'total': num_nodes, 'execution_time': run_model_result.execution_time, 'run_status': str(run_model_result.status).upper(), 'run_skipped': run_model_result.status == NodeStatus.Skipped, 'run_error': run_model_result.status == NodeStatus.Error, 'model_materialization': node.get_materialization(), 'model_incremental_strategy': incremental_strategy, 'model_id': utils.get_hash(node), 'hashed_contents': utils.get_hashed_contents(node), 'timing': [t.to_dict(omit_none=True) for t in run_model_result.timing], 'language': str(node.language), 'has_group': has_group, 'contract_enforced': contract_enforced, 'access': access, 'versioned': versioned, 'adapter_info': _get_adapter_info(adapter, run_model_result)})

def _validate_materialization_relations_dict(inp: Dict, model: ModelNode) -> List[BaseRelation]:
    try:
        relations_value = inp['relations']
    except KeyError:
        msg = 'Invalid return value from materialization, "relations" not found, got keys: {}'.format(list(inp))
        raise CompilationError(msg, node=model) from None
    if not isinstance(relations_value, list):
        msg = 'Invalid return value from materialization, "relations" not a list, got: {}'.format(relations_value)
        raise CompilationError(msg, node=model) from None
    relations = []
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
        description = self.describe_node()
        group = group_lookup.get(self.node.unique_id)
        if result.status == NodeStatus.Error:
            status = result.status
            level = EventLevel.ERROR
        else:
            status = result.message
            level = EventLevel.INFO
        fire_event(LogModelResult(description=description, status=status, index=self.node_index, total=self.num_nodes, execution_time=result.execution_time, node_info=self.node.node_info, group=group), level=level)

    def before_execute(self) -> None:
        self.print_start_line()

    def after_execute(self, result: RunResult) -> None:
        track_model_run(self.node_index, self.num_nodes, result, adapter=self.adapter)
        self.print_result_line(result)

    def _build_run_model_result(self, model: ModelNode, context: Dict, elapsed_time: float = 0.0) -> RunResult:
        result = context['load_result']('main')
        if not result:
            raise DbtRuntimeError('main is not being called during running model')
        adapter_response = {}
        if isinstance(result.response, dbtClassMixin):
            adapter_response = result.response.to_dict(omit_none=True)
        return RunResult(node=model, status=RunStatus.Success, timing=[], thread_id=threading.current_thread().name, execution_time=elapsed_time, message=str(result.response), adapter_response=adapter_response, failures=result.get('failures'), batch_results=None)

    def _materialization_relations(self, result: Any, model: ModelNode) -> List[BaseRelation]:
        if isinstance(result, str):
            msg = 'The materialization ("{}") did not explicitly return a list of relations to add to the cache.'.format(str(model.get_materialization()))
            raise CompilationError(msg, node=model)
        if isinstance(result, dict):
            return _validate_materialization_relations_dict(result, model)
        msg = 'Invalid return value from materialization, expected a dict with key "relations", got: {}'.format(str(result))
        raise CompilationError(msg, node=model)

    def _execute_model(self, hook_ctx: Any, context_config: Any, model: ModelNode, context: Dict, materialization_macro: MacroProtocol) -> RunResult:
        try:
            result = MacroGenerator(materialization_macro, context, stack=context['context_macro_stack'])()
        finally:
            self.adapter.post_model_hook(context_config, hook_ctx)
        for relation in self._materialization_relations(result, model):
            self.adapter.cache_added(relation.incorporate(dbt_created=True))
        return self._build_run_model_result(model, context)

    def execute(self, model: ModelNode, manifest: Manifest) -> None:
        context = generate_runtime_model_context(model, self.config, manifest)
        materialization_macro = manifest.find_materialization_macro_by_name(self.config.project_name, model.get_materialization(), self.adapter.type())
        if materialization_macro is None:
            raise MissingMaterializationError(materialization=model.get_materialization(), adapter_type=self.adapter.type())
        if 'config' not in context:
            raise DbtInternalError('Invalid materialization context generated, missing config: {}'.format(context))
        context_config = context['config']
        mat_has_supported_langs = hasattr(materialization_macro, 'supported_languages')
        model_lang_supported = model.language in materialization_macro.supported_languages
        if mat_has_supported_langs and (not model_lang_supported):
            str_langs = [str(lang) for lang in materialization_macro.supported_languages]
            raise DbtValidationError(f'Materialization "{materialization_macro.name}" only supports languages {str_langs}; got "{model.language}"')
        hook_ctx = self.adapter.pre_model_hook(context_config)
        return self._execute_model(hook_ctx, context_config, model, context, materialization_macro)
