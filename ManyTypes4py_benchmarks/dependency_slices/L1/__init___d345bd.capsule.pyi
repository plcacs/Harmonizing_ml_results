from typing import Any

# === Third-party dependency: agate ===
# Used symbols: Table, TimeDelta

# === Third-party dependency: dbt.adapters.factory ===
FACTORY: AdapterContainer

# === Third-party dependency: dbt.config ===
# Used symbols: Profile, Project, RuntimeConfig

# === Third-party dependency: dbt.config.project ===
class PartialProject(RenderComponents):
    def render(self, renderer: DbtProjectYamlRenderer) -> 'Project': ...

# === Third-party dependency: dbt.config.renderer ===
class DbtProjectYamlRenderer(BaseRenderer):
    def __init__(self, profile: Optional[HasCredentials] = ..., cli_vars: Optional[Dict[str, Any]] = ...) -> None: ...
class ProfileRenderer(SecretRenderer):
    ...

# === Third-party dependency: dbt.config.utils ===
def parse_cli_vars(var_string: str) -> Dict[str, Any]: ...

# === Third-party dependency: dbt.contracts.graph.manifest ===
class Manifest(MacroMethods, dbtClassMixin): ...

# === Third-party dependency: dbt.contracts.graph.nodes ===
class ModelNode(ModelResource, CompiledNode): ...
class SeedNode(SeedResource, ParsedNode): ...
class Macro(MacroResource, BaseNode):
    ...
class Documentation(DocumentationResource, BaseNode): ...
class SourceDefinition(NodeInfoMixin, GraphNode, SourceDefinitionResource, HasRelationMetadata): ...

# === Third-party dependency: dbt.node_types ===
# Used symbols: NodeType

# === Third-party dependency: dbt.parser.manifest ===
class ManifestLoader: ...

# === Third-party dependency: dbt_common.clients ===
# Used symbols: agate_helper

# === Third-party dependency: dbt_common.dataclass_schema ===
class ValidationError(jsonschema.ValidationError): ...

# === Third-party dependency: pytest ===
# Used symbols: raises