# === Third-party dependency: dbt.adapters ===
# Used symbols: factory, postgres

# === Third-party dependency: dbt.clients.jinja ===
class MacroStack(threading.local): ...

# === Third-party dependency: dbt.config.project ===
class VarProvider: ...

# === Third-party dependency: dbt.context ===
# Used symbols: base, docs, macros, providers, query_header

# === Third-party dependency: dbt.contracts.files ===
# Used symbols: FileHash

# === Third-party dependency: dbt.contracts.graph.nodes ===
class ModelNode(ModelResource, CompiledNode): ...
class UnitTestNode(CompiledNode): ...
class Macro(MacroResource, BaseNode): ...
# re-export: from dbt.artifacts.resources import DependsOn
# re-export: from dbt.artifacts.resources import NodeConfig
# re-export: from dbt.contracts.graph.unparsed import UnitTestOverrides

# === Third-party dependency: dbt.node_types ===
# Used symbols: NodeType

# === Third-party dependency: dbt_common.events.functions ===
def reset_metadata_vars() -> None: ...

# === Third-party dependency: dbt_common.exceptions ===
# Used symbols: CompilationError

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Internal dependency: tests.unit.mock_adapter ===
def adapter_factory(): ...

# === Internal dependency: tests.unit.utils ===
def config_from_parts_or_dicts(project, profile, packages=..., selectors=..., cli_vars=...): ...
def inject_adapter(value, plugin): ...
def clear_plugin(plugin): ...