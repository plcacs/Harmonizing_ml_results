# === Internal dependency: core.dbt.task.base ===
class BaseRunner:
    def compile(self, manifest): ...

# === Internal dependency: core.dbt.task.printer ===
def print_run_result_error(result, newline=..., is_warning=..., group=...): ...

# === Internal dependency: core.dbt.task.run ===
class RunTask(CompileTask):
    ...

# === Third-party dependency: dbt ===
# Used symbols: deprecations

# === Third-party dependency: dbt.adapters.base.impl ===
class FreshnessResponse(TypedDict): ...

# === Third-party dependency: dbt.adapters.base.relation ===
class BaseRelation(FakeAPIObject, Hashable): ...

# === Third-party dependency: dbt.adapters.capability ===
class Capability(str, Enum): ...

# === Third-party dependency: dbt.adapters.contracts.connection ===
class AdapterResponse(dbtClassMixin): ...

# === Third-party dependency: dbt.artifacts.schemas.freshness ===
# Used symbols: FreshnessResult, FreshnessStatus, PartialSourceFreshnessResult, SourceFreshnessResult

# === Third-party dependency: dbt.contracts.graph.nodes ===
class HookNode(HookNodeResource, CompiledNode): ...
class SourceDefinition(NodeInfoMixin, GraphNode, SourceDefinitionResource, HasRelationMetadata): ...

# === Third-party dependency: dbt.contracts.results ===
# Used symbols: RunStatus

# === Third-party dependency: dbt.events.types ===
class FreshnessCheckComplete(InfoLevel): ...
class LogStartLine(InfoLevel): ...
class LogFreshnessResult(DynamicLevel):
    ...

# === Third-party dependency: dbt.graph ===
# Used symbols: ResourceTypeSelector

# === Third-party dependency: dbt.node_types ===
# Used symbols: NodeType

# === Third-party dependency: dbt_common.events.base_types ===
class EventLevel(str, Enum): ...

# === Third-party dependency: dbt_common.events.functions ===
def fire_event(e: BaseEvent, level: Optional[EventLevel] = ...) -> None: ...

# === Third-party dependency: dbt_common.events.types ===
class Note(InfoLevel): ...

# === Third-party dependency: dbt_common.exceptions ===
# Used symbols: DbtInternalError, DbtRuntimeError