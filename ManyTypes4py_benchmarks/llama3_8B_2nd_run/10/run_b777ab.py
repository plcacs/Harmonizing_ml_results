class BiggestName(str):
    def __lt__(self, other: str) -> bool:
        return True

    def __eq__(self, other: str) -> bool:
        return isinstance(other, self.__class__)

def _hook_list() -> List[HookNode]:
    return []

def get_hooks_by_tags(nodes: Iterable[Node], match_tags: Set[str]) -> List[HookNode]:
    matched_nodes: List[HookNode] = []
    for node in nodes:
        if not isinstance(node, HookNode):
            continue
        node_tags: Set[str] = set(node.tags)
        if len(set(node_tags) & match_tags):
            matched_nodes.append(node)
    return matched_nodes

def get_hook(source: str, index: int) -> Hook:
    hook_dict: Dict[str, Any] = get_hook_dict(source)
    hook_dict.setdefault('index', index)
    Hook.validate(hook_dict)
    return Hook.from_dict(hook_dict)

def _get_adapter_info(adapter: BaseAdapter, run_model_result: RunResult) -> Dict[str, Any]:
    return asdict(adapter.get_adapter_run_info(run_model_result.node.config))

class ModelRunner(CompileRunner):
    # ...

    def describe_node(self) -> str:
        return f'{self.node.language} {self.node.get_materialization()} model {self.get_node_representation()}'

    def describe_batch(self) -> str:
        batch_start: Optional[float] = self.batch_start
        if batch_start is None:
            return ''
        formatted_batch_start: str = MicrobatchBuilder.format_batch_start(batch_start, self.node.config.batch_size)
        return f'batch {formatted_batch_start} of {self.get_node_representation()}'

    def print_batch_result_line(self, result: RunResult) -> None:
        if self.batch_idx is None:
            return
        description: str = self.describe_batch()
        group: Optional[str] = group_lookup.get(self.node.unique_id)
        if result.status == NodeStatus.Error:
            status: str = result.status
            level: EventLevel = EventLevel.ERROR
        elif result.status == NodeStatus.Skipped:
            status: str = result.status
            level: EventLevel = EventLevel.INFO
        else:
            status: str = result.message
            level: EventLevel = EventLevel.INFO
        fire_event(LogBatchResult(description=description, status=status, batch_index=self.batch_idx + 1, total_batches=len(self.batches), execution_time=result.execution_time, node_info=self.node.node_info, group=group), level=level)

    # ...

class MicrobatchModelRunner(ModelRunner):
    # ...

    def _execute_microbatch_materialization(self, model: ModelNode, context: Dict[str, Any], materialization_macro: MacroProtocol) -> RunResult:
        # ...

    def _submit_batch(self, node: ModelNode, adapter: BaseAdapter, relation_exists: bool, batches: List[Tuple[float, float]], batch_idx: int, batch_results: List[RunResult], pool: ThreadPool, force_sequential_run: bool = False, skip: bool = False) -> bool:
        node_copy: ModelNode = deepcopy(node)
        # ...

    def handle_microbatch_model(self, runner: MicrobatchModelRunner, pool: ThreadPool) -> RunResult:
        result: RunResult = self.call_runner(runner)
        # ...

class RunTask(CompileTask):
    # ...

    def before_run(self, adapter: BaseAdapter, selected_uids: Set[str]) -> RunStatus:
        # ...

    def after_run(self, adapter: BaseAdapter, results: List[RunResult]) -> None:
        # ...

    def get_node_selector(self) -> ResourceTypeSelector:
        # ...

    def get_runner_type(self, node: ModelNode) -> Type[ModelRunner]:
        # ...
