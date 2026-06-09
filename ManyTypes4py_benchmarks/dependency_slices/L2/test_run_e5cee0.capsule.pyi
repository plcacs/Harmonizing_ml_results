from typing import Any

# === Third-party dependency: dbt.artifacts.resources.v1.model ===
class ModelConfig(NodeConfig):
    ...

# === Third-party dependency: dbt.artifacts.schemas.batch_results ===
class BatchResults(dbtClassMixin): ...

# === Third-party dependency: dbt.artifacts.schemas.results ===
class RunStatus(StrEnum): ...

# === Third-party dependency: dbt.artifacts.schemas.run ===
# Used symbols: RunResult

# === Third-party dependency: dbt.events.types ===
class LogModelResult(DynamicLevel): ...

# === Third-party dependency: dbt.flags ===
def get_flags() -> Any: ...
def set_from_args(args: Namespace, project_flags) -> Any: ...

# === Third-party dependency: dbt.task.run ===
class ModelRunner(CompileRunner): ...
class RunTask(CompileTask):
    def __init__(self, args: Flags, config: RuntimeConfig, manifest: Manifest, batch_map: Optional[Dict[str, List[BatchType]]] = ...) -> None: ...

# === Third-party dependency: dbt.tests.util ===
def safe_set_invocation_context() -> Any: ...

# === Third-party dependency: dbt_common.events.base_types ===
class EventLevel(str, Enum): ...

# === Third-party dependency: dbt_common.events.event_manager_client ===
def add_callback_to_manager(callback: TCallback) -> None: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Internal dependency: tests.utils ===
class EventCatcher:
    def catch(self, event: EventMsg) -> Any: ...