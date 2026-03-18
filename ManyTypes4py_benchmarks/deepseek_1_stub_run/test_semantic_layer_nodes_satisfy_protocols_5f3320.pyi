```python
import pytest
from typing import Any, Protocol, runtime_checkable
from dbt.artifacts.resources import (
    ConstantPropertyInput,
    ConversionTypeParams,
    CumulativeTypeParams,
    Defaults,
    Dimension,
    DimensionTypeParams,
    DimensionValidityParams,
    Entity,
    FileSlice,
    Measure,
    MeasureAggregationParameters,
    MetricInput,
    MetricInputMeasure,
    MetricTimeWindow,
    MetricTypeParams,
    NodeRelation,
    NonAdditiveDimension,
    SourceFileMetadata,
    WhereFilter,
)
from dbt.contracts.graph.nodes import Metric, SavedQuery, SemanticModel
from dbt.node_types import NodeType
from dbt_semantic_interfaces.protocols import (
    WhereFilter as WhereFilterProtocol,
    dimension as DimensionProtocols,
    entity as EntityProtocols,
    measure as MeasureProtocols,
    metadata as MetadataProtocols,
    metric as MetricProtocols,
    saved_query as SavedQueryProtocols,
    semantic_model as SemanticModelProtocols,
)
from dbt_semantic_interfaces.type_enums import (
    AggregationType,
    DimensionType,
    EntityType,
    MetricType,
    TimeGranularity,
)

@runtime_checkable
class RuntimeCheckableSemanticModel(SemanticModelProtocols.SemanticModel, Protocol): ...

@runtime_checkable
class RuntimeCheckableDimension(DimensionProtocols.Dimension, Protocol): ...

@runtime_checkable
class RuntimeCheckableEntity(EntityProtocols.Entity, Protocol): ...

@runtime_checkable
class RuntimeCheckableMeasure(MeasureProtocols.Measure, Protocol): ...

@runtime_checkable
class RuntimeCheckableMetric(MetricProtocols.Metric, Protocol): ...

@runtime_checkable
class RuntimeCheckableMetricInput(MetricProtocols.MetricInput, Protocol): ...

@runtime_checkable
class RuntimeCheckableMetricInputMeasure(MetricProtocols.MetricInputMeasure, Protocol): ...

@runtime_checkable
class RuntimeCheckableMetricTypeParams(MetricProtocols.MetricTypeParams, Protocol): ...

@runtime_checkable
class RuntimeCheckableWhereFilter(WhereFilterProtocol, Protocol): ...

@runtime_checkable
class RuntimeCheckableNonAdditiveDimension(MeasureProtocols.NonAdditiveDimensionParameters, Protocol): ...

@runtime_checkable
class RuntimeCheckableFileSlice(MetadataProtocols.FileSlice, Protocol): ...

@runtime_checkable
class RuntimeCheckableSourceFileMetadata(MetadataProtocols.Metadata, Protocol): ...

@runtime_checkable
class RuntimeCheckableSemanticModelDefaults(SemanticModelProtocols.SemanticModelDefaults, Protocol): ...

@runtime_checkable
class RuntimeCheckableDimensionValidityParams(DimensionProtocols.DimensionValidityParams, Protocol): ...

@runtime_checkable
class RuntimeCheckableDimensionTypeParams(DimensionProtocols.DimensionTypeParams, Protocol): ...

@runtime_checkable
class RuntimeCheckableMeasureAggregationParams(MeasureProtocols.MeasureAggregationParameters, Protocol): ...

@runtime_checkable
class RuntimeCheckableMetricTimeWindow(MetricProtocols.MetricTimeWindow, Protocol): ...

@runtime_checkable
class RuntimeCheckableSavedQuery(SavedQueryProtocols.SavedQuery, Protocol): ...

@pytest.fixture(scope="session")
def file_slice() -> FileSlice: ...

@pytest.fixture(scope="session")
def source_file_metadata(file_slice: FileSlice) -> SourceFileMetadata: ...

@pytest.fixture(scope="session")
def semantic_model_defaults() -> Defaults: ...

@pytest.fixture(scope="session")
def dimension_validity_params() -> DimensionValidityParams: ...

@pytest.fixture(scope="session")
def dimension_type_params() -> DimensionTypeParams: ...

@pytest.fixture(scope="session")
def measure_agg_params() -> MeasureAggregationParameters: ...

@pytest.fixture(scope="session")
def non_additive_dimension() -> NonAdditiveDimension: ...

@pytest.fixture(scope="session")
def where_filter() -> WhereFilter: ...

@pytest.fixture(scope="session")
def metric_time_window() -> MetricTimeWindow: ...

@pytest.fixture(scope="session")
def simple_metric_input() -> MetricInput: ...

@pytest.fixture(scope="session")
def complex_metric_input(
    metric_time_window: MetricTimeWindow, where_filter: WhereFilter
) -> MetricInput: ...

@pytest.fixture(scope="session")
def simple_metric_input_measure() -> MetricInputMeasure: ...

@pytest.fixture(scope="session")
def complex_metric_input_measure(
    where_filter: WhereFilter,
) -> MetricInputMeasure: ...

@pytest.fixture(scope="session")
def conversion_type_params(
    simple_metric_input_measure: MetricInputMeasure,
    metric_time_window: MetricTimeWindow,
) -> ConversionTypeParams: ...

@pytest.fixture(scope="session")
def cumulative_type_params() -> CumulativeTypeParams: ...

@pytest.fixture(scope="session")
def complex_metric_type_params(
    metric_time_window: MetricTimeWindow,
    simple_metric_input: MetricInput,
    simple_metric_input_measure: MetricInputMeasure,
    conversion_type_params: ConversionTypeParams,
    cumulative_type_params: CumulativeTypeParams,
) -> MetricTypeParams: ...

def test_file_slice_obj_satisfies_protocol(file_slice: FileSlice) -> None: ...

def test_metadata_obj_satisfies_protocol(source_file_metadata: SourceFileMetadata) -> None: ...

def test_defaults_obj_satisfies_protocol(semantic_model_defaults: Defaults) -> None: ...

def test_dimension_validity_params_satisfies_protocol(
    dimension_validity_params: DimensionValidityParams,
) -> None: ...

def test_dimension_type_params_satisfies_protocol(
    dimension_type_params: DimensionTypeParams,
    dimension_validity_params: DimensionValidityParams,
) -> None: ...

def test_measure_aggregation_params_satisfies_protocol(
    measure_agg_params: MeasureAggregationParameters,
) -> None: ...

def test_semantic_model_node_satisfies_protocol_optionals_unspecified() -> None: ...

def test_semantic_model_node_satisfies_protocol_optionals_specified(
    semantic_model_defaults: Defaults,
    source_file_metadata: SourceFileMetadata,
) -> None: ...

def test_dimension_satisfies_protocol_optionals_unspecified() -> None: ...

def test_dimension_satisfies_protocol_optionals_specified(
    dimension_type_params: DimensionTypeParams,
    source_file_metadata: SourceFileMetadata,
) -> None: ...

def test_entity_satisfies_protocol_optionals_unspecified() -> None: ...

def test_entity_satisfies_protocol_optionals_specified() -> None: ...

def test_measure_satisfies_protocol_optionals_unspecified() -> None: ...

def test_measure_satisfies_protocol_optionals_specified(
    measure_agg_params: MeasureAggregationParameters,
    non_additive_dimension: NonAdditiveDimension,
) -> None: ...

@pytest.mark.skip(reason="Overly sensitive to non-breaking changes")
def test_metric_node_satisfies_protocol_optionals_unspecified() -> None: ...

@pytest.mark.skip(reason="Overly sensitive to non-breaking changes")
def test_metric_node_satisfies_protocol_optionals_specified(
    complex_metric_type_params: MetricTypeParams,
    source_file_metadata: SourceFileMetadata,
    where_filter: WhereFilter,
) -> None: ...

def test_where_filter_satisfies_protocol(where_filter: WhereFilter) -> None: ...

def test_metric_time_window(metric_time_window: MetricTimeWindow) -> None: ...

def test_metric_input(
    simple_metric_input: MetricInput, complex_metric_input: MetricInput
) -> None: ...

def test_metric_input_measure(
    simple_metric_input_measure: MetricInputMeasure,
    complex_metric_input_measure: MetricInputMeasure,
) -> None: ...

@pytest.mark.skip(reason="Overly sensitive to non-breaking changes")
def test_metric_type_params_satisfies_protocol(
    complex_metric_type_params: MetricTypeParams,
) -> None: ...

def test_non_additive_dimension_satisfies_protocol(
    non_additive_dimension: NonAdditiveDimension,
) -> None: ...

def test_saved_query_satisfies_protocol(saved_query: SavedQuery) -> None: ...
```