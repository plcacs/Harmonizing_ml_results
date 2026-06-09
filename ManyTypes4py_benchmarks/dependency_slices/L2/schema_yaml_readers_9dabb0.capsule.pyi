from typing import Any

# === Third-party dependency: dbt.artifacts.resources ===
# Used symbols: ConversionTypeParams, CumulativeTypeParams, Dimension, DimensionTypeParams, Entity, Export, ExportConfig, ExposureConfig, Measure, MetricConfig, MetricInput, MetricInputMeasure, MetricTimeWindow, MetricTypeParams, NonAdditiveDimension, QueryParams, WhereFilter, WhereFilterIntersection

# === Third-party dependency: dbt.clients.jinja ===
def get_rendered(string: str, ctx: Dict[str, Any], node = ..., capture_macros: bool = ..., native: bool = ...) -> Any: ...

# === Third-party dependency: dbt.context.context_config ===
class BaseContextConfigGenerator(Generic[T]): ...
class ContextConfigGenerator(BaseContextConfigGenerator[C]):
    def __init__(self, active_project: RuntimeConfig) -> Any: ...
class UnrenderedConfigGenerator(BaseContextConfigGenerator[Dict[str, Any]]):
    ...

# === Third-party dependency: dbt.context.providers ===
def generate_parse_exposure(exposure: Exposure, config: RuntimeConfig, manifest: Manifest, package_name: str) -> Dict[str, Any]: ...
def generate_parse_semantic_models(semantic_model: SemanticModel, config: RuntimeConfig, manifest: Manifest, package_name: str) -> Dict[str, Any]: ...

# === Third-party dependency: dbt.contracts.files ===
class SchemaSourceFile(BaseSourceFile): ...

# === Third-party dependency: dbt.contracts.graph.nodes ===
class Exposure(GraphNode, ExposureResource):
    ...
class Metric(GraphNode, MetricResource):
class Group(GroupResource, BaseNode):
class SemanticModel(GraphNode, SemanticModelResource):
class SavedQuery(NodeInfoMixin, GraphNode, SavedQueryResource):

# === Third-party dependency: dbt.contracts.graph.unparsed ===
class UnparsedExposure(dbtClassMixin):
    ...
class UnparsedCumulativeTypeParams(dbtClassMixin):
class UnparsedMetricTypeParams(dbtClassMixin): ...
class UnparsedMetric(dbtClassMixin):
class UnparsedGroup(dbtClassMixin):
class UnparsedSemanticModel(dbtClassMixin):
class UnparsedSavedQuery(dbtClassMixin):

# === Third-party dependency: dbt.exceptions ===
class JSONValidationError(DbtValidationError): ...
class YamlParseDictError(ParsingError): ...

# === Third-party dependency: dbt.node_types ===
# Used symbols: NodeType

# === Third-party dependency: dbt.parser.schemas ===
class ParseResult: ...
class YamlReader:
    def manifest(self) -> Manifest: ...
    def project(self) -> RuntimeConfig: ...
    def default_database(self) -> str: ...
    def root_project(self) -> RuntimeConfig: ...
    def parse(self) -> Optional[ParseResult]: ...

# === Third-party dependency: dbt_common.dataclass_schema ===
class ValidationError(jsonschema.ValidationError): ...

# === Third-party dependency: dbt_common.exceptions ===
# Used symbols: DbtInternalError

# === Third-party dependency: dbt_semantic_interfaces.type_enums ===
# Used symbols: AggregationType, ConversionCalculationType, DimensionType, EntityType, MetricType, PeriodAggregation, TimeGranularity