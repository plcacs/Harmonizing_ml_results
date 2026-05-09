from collections.abc import Sequence
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Set,
    Tuple,
    FrozenSet,
    Type,
    TypeVar,
    overload,
    Literal,
    Protocol,
    runtime_checkable,
)

from dbt.artifacts.resources import (
    ConversionTypeParams,
    CumulativeTypeParams,
    Dimension,
    DimensionTypeParams,
    Entity,
    Export,
    ExportConfig,
    ExposureConfig,
    Measure,
    MetricConfig,
    MetricInput,
    MetricInputMeasure,
    MetricTimeWindow,
    MetricTypeParams,
    NonAdditiveDimension,
    QueryParams,
    SavedQueryConfig,
    WhereFilter,
    WhereFilterIntersection,
)
from dbt.contracts.files import SchemaSourceFile
from dbt.contracts.graph.nodes import (
    Exposure,
    Group,
    Metric,
    SavedQuery,
    SemanticModel,
)
from dbt.contracts.graph.unparsed import (
    UnparsedConversionTypeParams,
    UnparsedCumulativeTypeParams,
    UnparsedDimension,
    UnparsedDimensionTypeParams,
    UnparsedEntity,
    UnparsedExport,
    UnparsedExposure,
    UnparsedGroup,
    UnparsedMeasure,
    UnparsedMetric,
    UnparsedMetricInput,
    UnparsedMetricInputMeasure,
    UnparsedMetricTypeParams,
    UnparsedNonAdditiveDimension,
    UnparsedQueryParams,
    UnparsedSavedQuery,
    UnparsedSemanticModel,
)
from dbt.node_types import NodeType
from dbt.parser.schemas import ParseResult, SchemaParser, YamlReader
from dbt_semantic_interfaces.type_enums import (
    AggregationType,
    ConversionCalculationType,
    DimensionType,
    EntityType,
    MetricType,
    PeriodAggregation,
    TimeGranularity,
)

def parse_where_filter(where: Union[str, Sequence[str], None]) -> Optional[WhereFilterIntersection]: ...

class ExposureParser(YamlReader):
    def __init__(self, schema_parser: Any, yaml: Any) -> None: ...
    def parse_exposure(self, unparsed: UnparsedExposure) -> None: ...
    def _generate_exposure_config(
        self,
        target: Any,
        fqn: List[str],
        package_name: str,
        rendered: bool,
    ) -> Dict[str, Any]: ...

class MetricParser(YamlReader):
    def __init__(self, schema_parser: Any, yaml: Any) -> None: ...
    def _get_input_measure(self, unparsed_input_measure: Union[str, UnparsedMetricInputMeasure]) -> MetricInputMeasure: ...
    def _get_optional_input_measure(self, unparsed_input_measure: Optional[Union[str, UnparsedMetricInputMeasure]]) -> Optional[MetricInputMeasure]: ...
    def _get_input_measures(self, unparsed_input_measures: Optional[List[Union[str, UnparsedMetricInputMeasure]]]) -> List[MetricInputMeasure]: ...
    def _get_period_agg(self, unparsed_period_agg: str) -> PeriodAggregation: ...
    def _get_optional_time_window(self, unparsed_window: Optional[str]) -> Optional[MetricTimeWindow]: ...
    def _get_metric_input(self, unparsed: Union[str, UnparsedMetricInput]) -> MetricInput: ...
    def _get_optional_metric_input(self, unparsed: Optional[Union[str, UnparsedMetricInput]]) -> Optional[MetricInput]: ...
    def _get_metric_inputs(self, unparsed_metric_inputs: Optional[List[Union[str, UnparsedMetricInput]]]) -> List[MetricInput]: ...
    def _get_optional_conversion_type_params(self, unparsed: Optional[UnparsedConversionTypeParams]) -> Optional[ConversionTypeParams]: ...
    def _get_optional_cumulative_type_params(self, unparsed_metric: UnparsedMetric) -> Optional[CumulativeTypeParams]: ...
    def _get_metric_type_params(self, unparsed_metric: UnparsedMetric) -> MetricTypeParams: ...
    def parse_metric(self, unparsed: UnparsedMetric, generated_from: Optional[Any] = None) -> None: ...
    def _generate_metric_config(
        self,
        target: Any,
        fqn: List[str],
        package_name: str,
        rendered: bool,
    ) -> Dict[str, Any]: ...

class GroupParser(YamlReader):
    def __init__(self, schema_parser: Any, yaml: Any) -> None: ...
    def parse_group(self, unparsed: UnparsedGroup) -> None: ...

class SemanticModelParser(YamlReader):
    def __init__(self, schema_parser: Any, yaml: Any) -> None: ...
    def _get_dimension_type_params(self, unparsed: Optional[UnparsedDimensionTypeParams]) -> Optional[DimensionTypeParams]: ...
    def _get_dimensions(self, unparsed_dimensions: List[UnparsedDimension]) -> List[Dimension]: ...
    def _get_entities(self, unparsed_entities: List[UnparsedEntity]) -> List[Entity]: ...
    def _get_non_additive_dimension(self, unparsed: Optional[UnparsedNonAdditiveDimension]) -> Optional[NonAdditiveDimension]: ...
    def _get_measures(self, unparsed_measures: List[UnparsedMeasure]) -> List[Measure]: ...
    def _create_metric(self, measure: Measure, enabled: bool, semantic_model_name: str) -> None: ...
    def _generate_semantic_model_config(
        self,
        target: Any,
        fqn: List[str],
        package_name: str,
        rendered: bool,
    ) -> Dict[str, Any]: ...
    def parse_semantic_model(self, unparsed: UnparsedSemanticModel) -> None: ...

class SavedQueryParser(YamlReader):
    def __init__(self, schema_parser: Any, yaml: Any) -> None: ...
    def _generate_saved_query_config(
        self,
        target: Any,
        fqn: List[str],
        package_name: str,
        rendered: bool,
    ) -> Dict[str, Any]: ...
    def _get_export_config(self, unparsed_export_config: Dict[str, Any], saved_query_config: SavedQueryConfig) -> ExportConfig: ...
    def _get_export(self, unparsed: UnparsedExport, saved_query_config: SavedQueryConfig) -> Export: ...
    def _get_query_params(self, unparsed: UnparsedQueryParams) -> QueryParams: ...
    def parse_saved_query(self, unparsed: UnparsedSavedQuery) -> None: ...