from typing import Any

# === Third-party dependency: dbt.artifacts.resources ===
# Used symbols: Defaults, Dimension, Entity, FileHash, Measure, TestMetadata

# === Third-party dependency: dbt.artifacts.resources.v1.semantic_model ===
class NodeRelation(dbtClassMixin): ...

# === Third-party dependency: dbt.contracts.graph.model_config ===
# Used symbols: TestConfig

# === Third-party dependency: dbt.contracts.graph.nodes ===
class ParsedNode(ParsedResource, NodeInfoMixin, ParsedNodeMandatory, SerializableType): ...
class ModelNode(ModelResource, CompiledNode): ...
class SemanticModel(GraphNode, SemanticModelResource): ...
# re-export: from dbt.artifacts.resources import ColumnInfo

# === Third-party dependency: dbt.node_types ===
# Used symbols: NodeType

# === Third-party dependency: dbt_common.contracts.constraints ===
class ConstraintType(str, Enum): ...
class ColumnLevelConstraint(dbtClassMixin): ...
class ModelLevelConstraint(ColumnLevelConstraint): ...

# === Third-party dependency: dbt_semantic_interfaces.references ===
class MeasureReference(ElementReference): ...

# === Third-party dependency: dbt_semantic_interfaces.type_enums ===
# Used symbols: AggregationType, DimensionType, EntityType

# === Third-party dependency: freezegun ===
# Used symbols: freeze_time

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises

# === Internal dependency: tests.unit.fixtures ===
def model_node() -> Any: ...
def generic_test_node() -> Any: ...