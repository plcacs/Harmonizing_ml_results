# === Third-party dependency: dbt.artifacts.resources ===
# Used symbols: ConstantPropertyInput, ConversionTypeParams, CumulativeTypeParams, Defaults, Dimension, DimensionTypeParams, DimensionValidityParams, Entity, FileSlice, Measure, MeasureAggregationParameters, MetricInput, MetricInputMeasure, MetricTimeWindow, MetricTypeParams, NodeRelation, NonAdditiveDimension, SourceFileMetadata, WhereFilter

# === Third-party dependency: dbt.contracts.graph.nodes ===
class Metric(GraphNode, MetricResource):
    ...
class SemanticModel(GraphNode, SemanticModelResource):
class SavedQuery(NodeInfoMixin, GraphNode, SavedQueryResource): ...

# === Third-party dependency: dbt.node_types ===
# Used symbols: NodeType

# === Third-party dependency: dbt_semantic_interfaces.protocols ===
# Used symbols: WhereFilter, dimension, entity, measure, metadata, metric, saved_query, semantic_model

# === Third-party dependency: dbt_semantic_interfaces.type_enums ===
# Used symbols: AggregationType, DimensionType, EntityType, MetricType, TimeGranularity

# === Third-party dependency: hypothesis ===
# Used symbols: given

# === Third-party dependency: hypothesis.strategies ===
# Used symbols: builds, none, text

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark