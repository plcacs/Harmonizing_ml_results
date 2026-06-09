# === Internal dependency: core.dbt.graph.graph ===
class Graph: ...
UniqueId = NewType(...)

# === Internal dependency: core.dbt.graph.queue ===
class GraphQueue: ...

# === Internal dependency: core.dbt.graph.selector_methods ===
class MethodManager:
    ...

# === Internal dependency: core.dbt.graph.selector_spec ===
class IndirectSelection(StrEnum): ...
class SelectionCriteria: ...

# === Third-party dependency: dbt ===
# Used symbols: selected_resources

# === Third-party dependency: dbt.contracts.graph.nodes ===
# Used symbols: GraphMemberNode

# === Third-party dependency: dbt.events.types ===
class SelectorReportInvalidSelector(InfoLevel): ...
class NoNodesForSelectionCriteria(WarnLevel): ...

# === Third-party dependency: dbt.exceptions ===
class InvalidSelectorError(DbtRuntimeError): ...
# re-export: from dbt_common.exceptions import DbtInternalError

# === Third-party dependency: dbt.node_types ===
# Used symbols: NodeType

# === Third-party dependency: dbt_common.events.functions ===
def warn_or_error(event, node = ...) -> None: ...
def fire_event(e: BaseEvent, level: Optional[EventLevel] = ...) -> None: ...