import copy
from typing import Protocol, runtime_checkable, Optional, Any
import pytest
from hypothesis import given
from hypothesis.strategies import builds, none, text
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
from dbt_semantic_interfaces.protocols import WhereFilter as WhereFilterProtocol
from dbt_semantic_interfaces.protocols import dimension as DimensionProtocols
from dbt_semantic_interfaces.protocols import entity as EntityProtocols
from dbt_semantic_interfaces.protocols import measure as MeasureProtocols
from dbt_semantic_interfaces.protocols import metadata as MetadataProtocols
from dbt_semantic_interfaces.protocols import metric as MetricProtocols
from dbt_semantic_interfaces.protocols import saved_query as SavedQueryProtocols
from dbt_semantic_interfaces.protocols import semantic_model as SemanticModelProtocols
from dbt_semantic_interfaces.type_enums import AggregationType, DimensionType, EntityType, MetricType, TimeGranularity

@runtime_checkable
class RuntimeCheckableSemanticModel(SemanticModelProtocols.SemanticModel, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableDimension(DimensionProtocols.Dimension, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableEntity(EntityProtocols.Entity, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableMeasure(MeasureProtocols.Measure, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableMetric(MetricProtocols.Metric, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableMetricInput(MetricProtocols.MetricInput, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableMetricInputMeasure(MetricProtocols.MetricInputMeasure, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableMetricTypeParams(MetricProtocols.MetricTypeParams, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableWhereFilter(WhereFilterProtocol, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableNonAdditiveDimension(MeasureProtocols.NonAdditiveDimensionParameters, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableFileSlice(MetadataProtocols.FileSlice, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableSourceFileMetadata(MetadataProtocols.Metadata, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableSemanticModelDefaults(SemanticModelProtocols.SemanticModelDefaults, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableDimensionValidityParams(DimensionProtocols.DimensionValidityParams, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableDimensionTypeParams(DimensionProtocols.DimensionTypeParams, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableMeasureAggregationParams(MeasureProtocols.MeasureAggregationParameters, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableMetricTimeWindow(MetricProtocols.MetricTimeWindow, Protocol):
    pass

@runtime_checkable
class RuntimeCheckableSavedQuery(SavedQueryProtocols.SavedQuery, Protocol):
    pass

@pytest.fixture(scope='session')
def file_slice() -> FileSlice:
    return FileSlice(filename='test_filename', content='test content', start_line_number=0, end_line_number=1)

@pytest.fixture(scope='session')
def source_file_metadata(file_slice: FileSlice) -> SourceFileMetadata:
    return SourceFileMetadata(repo_file_path='test/file/path.yml', file_slice=file_slice)

@pytest.fixture(scope='session')
def semantic_model_defaults() -> Defaults:
    return Defaults(agg_time_dimension='test_time_dimension')

@pytest.fixture(scope='session')
def dimension_validity_params() -> DimensionValidityParams:
    return DimensionValidityParams()

@pytest.fixture(scope='session')
def dimension_type_params() -> DimensionTypeParams:
    return DimensionTypeParams(time_granularity=TimeGranularity.DAY)

@pytest.fixture(scope='session')
def measure_agg_params() -> MeasureAggregationParameters:
    return MeasureAggregationParameters()

@pytest.fixture(scope='session')
def non_additive_dimension() -> NonAdditiveDimension:
    return NonAdditiveDimension(name='dimension_name', window_choice=AggregationType.MIN, window_groupings=['entity_name'])

@pytest.fixture(scope='session')
def where_filter() -> WhereFilter:
    return WhereFilter(where_sql_template="{{ Dimension('enity_name__dimension_name') }} AND {{ TimeDimension('entity_name__time_dimension_name', 'month') }} AND {{ Entity('entity_name') }}")

@pytest.fixture(scope='session')
def metric_time_window() -> MetricTimeWindow:
    return MetricTimeWindow(count=1, granularity=TimeGranularity.DAY.value)

@pytest.fixture(scope='session')
def simple_metric_input() -> MetricInput:
    return MetricInput(name='test_simple_metric_input')

@pytest.fixture(scope='session')
def complex_metric_input(metric_time_window: MetricTimeWindow, where_filter: WhereFilter) -> MetricInput:
    return MetricInput(name='test_complex_metric_input', filter=where_filter, alias='aliased_metric_input', offset_window=metric_time_window, offset_to_grain=TimeGranularity.DAY.value)

@pytest.fixture(scope='session')
def simple_metric_input_measure() -> MetricInputMeasure:
    return MetricInputMeasure(name='test_simple_metric_input_measure')

@pytest.fixture(scope='session')
def complex_metric_input_measure(where_filter: WhereFilter) -> MetricInputMeasure:
    return MetricInputMeasure(name='test_complex_metric_input_measure', filter=where_filter, alias='complex_alias', join_to_timespine=True, fill_nulls_with=0)

@pytest.fixture(scope='session')
def conversion_type_params(simple_metric_input_measure: MetricInputMeasure, metric_time_window: MetricTimeWindow) -> ConversionTypeParams:
    return ConversionTypeParams(
        base_measure=simple_metric_input_measure,
        conversion_measure=simple_metric_input_measure,
        entity='entity',
        window=metric_time_window,
        constant_properties=[ConstantPropertyInput(base_property='base', conversion_property='conversion')]
    )

@pytest.fixture(scope='session')
def cumulative_type_params() -> CumulativeTypeParams:
    return CumulativeTypeParams()

@pytest.fixture(scope='session')
def complex_metric_type_params(metric_time_window: MetricTimeWindow, simple_metric_input: MetricInput, simple_metric_input_measure: MetricInputMeasure, conversion_type_params: ConversionTypeParams, cumulative_type_params: CumulativeTypeParams) -> MetricTypeParams:
    return MetricTypeParams(
        measure=simple_metric_input_measure,
        numerator=simple_metric_input,
        denominator=simple_metric_input,
        expr='1 = 1',
        window=metric_time_window,
        grain_to_date=TimeGranularity.DAY,
        metrics=[simple_metric_input],
        conversion_type_params=conversion_type_params,
        cumulative_type_params=cumulative_type_params
    )

def test_file_slice_obj_satisfies_protocol(file_slice: FileSlice) -> None:
    assert isinstance(file_slice, RuntimeCheckableFileSlice)

def test_metadata_obj_satisfies_protocol(source_file_metadata: SourceFileMetadata) -> None:
    assert isinstance(source_file_metadata, RuntimeCheckableSourceFileMetadata)

def test_defaults_obj_satisfies_protocol(semantic_model_defaults: Defaults) -> None:
    assert isinstance(semantic_model_defaults, RuntimeCheckableSemanticModelDefaults)
    assert isinstance(Defaults(), RuntimeCheckableSemanticModelDefaults)

def test_dimension_validity_params_satisfies_protocol(dimension_validity_params: DimensionValidityParams) -> None:
    assert isinstance(dimension_validity_params, RuntimeCheckableDimensionValidityParams)

def test_dimension_type_params_satisfies_protocol(dimension_type_params: DimensionTypeParams, dimension_validity_params: DimensionValidityParams) -> None:
    assert isinstance(dimension_type_params, RuntimeCheckableDimensionTypeParams)
    optionals_specified_type_params: DimensionTypeParams = copy.deepcopy(dimension_type_params)
    optionals_specified_type_params.validity_params = dimension_validity_params
    assert isinstance(optionals_specified_type_params, RuntimeCheckableDimensionTypeParams)

def test_measure_aggregation_params_satisfies_protocol(measure_agg_params: MeasureAggregationParameters) -> None:
    assert isinstance(measure_agg_params, RuntimeCheckableMeasureAggregationParams)
    optionals_specified_measure_agg_params: MeasureAggregationParameters = copy.deepcopy(measure_agg_params)
    optionals_specified_measure_agg_params.percentile = 0.5  # type: ignore[attr-defined]
    assert isinstance(optionals_specified_measure_agg_params, RuntimeCheckableMeasureAggregationParams)

def test_semantic_model_node_satisfies_protocol_optionals_unspecified() -> None:
    test_semantic_model: SemanticModel = SemanticModel(
        name='test_semantic_model',
        resource_type=NodeType.SemanticModel,
        package_name='package_name',
        path='path.to.semantic_model',
        original_file_path='path/to/file',
        unique_id='not_like_the_other_semantic_models',
        fqn=['fully', 'qualified', 'name'],
        model="ref('a_model')",
        node_relation=NodeRelation(alias='test_alias', schema_name='test_schema_name')
    )
    assert isinstance(test_semantic_model, RuntimeCheckableSemanticModel)

def test_semantic_model_node_satisfies_protocol_optionals_specified(semantic_model_defaults: Defaults, source_file_metadata: SourceFileMetadata) -> None:
    test_semantic_model: SemanticModel = SemanticModel(
        name='test_semantic_model',
        resource_type=NodeType.SemanticModel,
        package_name='package_name',
        path='path.to.semantic_model',
        original_file_path='path/to/file',
        unique_id='not_like_the_other_semantic_models',
        fqn=['fully', 'qualified', 'name'],
        model="ref('a_model')",
        node_relation=NodeRelation(alias='test_alias', schema_name='test_schema_name'),
        description='test_description',
        label='test label',
        defaults=semantic_model_defaults,
        metadata=source_file_metadata,
        primary_entity='test_primary_entity'
    )
    assert isinstance(test_semantic_model, RuntimeCheckableSemanticModel)

def test_dimension_satisfies_protocol_optionals_unspecified() -> None:
    dimension: Dimension = Dimension(name='test_dimension', type=DimensionType.TIME)
    assert isinstance(dimension, RuntimeCheckableDimension)

def test_dimension_satisfies_protocol_optionals_specified(dimension_type_params: DimensionTypeParams, source_file_metadata: SourceFileMetadata) -> None:
    dimension: Dimension = Dimension(
        name='test_dimension',
        type=DimensionType.TIME,
        description='test_description',
        label='test_label',
        type_params=dimension_type_params,
        expr='1',
        metadata=source_file_metadata
    )
    assert isinstance(dimension, RuntimeCheckableDimension)

def test_entity_satisfies_protocol_optionals_unspecified() -> None:
    entity: Entity = Entity(name='test_entity', type=EntityType.PRIMARY)
    assert isinstance(entity, RuntimeCheckableEntity)

def test_entity_satisfies_protocol_optionals_specified() -> None:
    entity: Entity = Entity(
        name='test_entity',
        description='a test entity',
        label='A test entity',
        type=EntityType.PRIMARY,
        expr='id',
        role='a_role'
    )
    assert isinstance(entity, RuntimeCheckableEntity)

def test_measure_satisfies_protocol_optionals_unspecified() -> None:
    measure: Measure = Measure(name='test_measure', agg='sum')
    assert isinstance(measure, RuntimeCheckableMeasure)

def test_measure_satisfies_protocol_optionals_specified(measure_agg_params: MeasureAggregationParameters, non_additive_dimension: NonAdditiveDimension) -> None:
    measure: Measure = Measure(
        name='test_measure',
        description='a test measure',
        label='A test measure',
        agg='sum',
        create_metric=True,
        expr='amount',
        agg_params=measure_agg_params,
        non_additive_dimension=non_additive_dimension,
        agg_time_dimension='a_time_dimension'
    )
    assert isinstance(measure, RuntimeCheckableMeasure)

@pytest.mark.skip(reason='Overly sensitive to non-breaking changes')
def test_metric_node_satisfies_protocol_optionals_unspecified() -> None:
    metric: Metric = Metric(
        name='a_metric',
        resource_type=NodeType.Metric,
        package_name='package_name',
        path='path.to.semantic_model',
        original_file_path='path/to/file',
        unique_id='not_like_the_other_semantic_models',
        fqn=['fully', 'qualified', 'name'],
        description='a test metric',
        label='A test metric',
        type=MetricType.SIMPLE,
        type_params=MetricTypeParams(measure=MetricInputMeasure(name='a_test_measure', filter=WhereFilter(where_sql_template='a_dimension is true')))
    )
    assert isinstance(metric, RuntimeCheckableMetric)

@pytest.mark.skip(reason='Overly sensitive to non-breaking changes')
def test_metric_node_satisfies_protocol_optionals_specified(complex_metric_type_params: MetricTypeParams, source_file_metadata: SourceFileMetadata, where_filter: WhereFilter) -> None:
    metric: Metric = Metric(
        name='a_metric',
        resource_type=NodeType.Metric,
        package_name='package_name',
        path='path.to.semantic_model',
        original_file_path='path/to/file',
        unique_id='not_like_the_other_semantic_models',
        fqn=['fully', 'qualified', 'name'],
        description='a test metric',
        label='A test metric',
        type=MetricType.SIMPLE,
        type_params=complex_metric_type_params,
        filter=where_filter,
        metadata=source_file_metadata,
        group='test_group'
    )
    assert isinstance(metric, RuntimeCheckableMetric)

def test_where_filter_satisfies_protocol(where_filter: WhereFilter) -> None:
    assert isinstance(where_filter, RuntimeCheckableWhereFilter)

def test_metric_time_window(metric_time_window: MetricTimeWindow) -> None:
    assert isinstance(metric_time_window, RuntimeCheckableMetricTimeWindow)

def test_metric_input(simple_metric_input: MetricInput, complex_metric_input: MetricInput) -> None:
    assert isinstance(simple_metric_input, RuntimeCheckableMetricInput)
    assert isinstance(complex_metric_input, RuntimeCheckableMetricInput)

def test_metric_input_measure(simple_metric_input_measure: MetricInputMeasure, complex_metric_input_measure: MetricInputMeasure) -> None:
    assert isinstance(simple_metric_input_measure, RuntimeCheckableMetricInputMeasure)
    assert isinstance(complex_metric_input_measure, RuntimeCheckableMetricInputMeasure)

@pytest.mark.skip(reason='Overly sensitive to non-breaking changes')
def test_metric_type_params_satisfies_protocol(complex_metric_type_params: MetricTypeParams) -> None:
    assert isinstance(MetricTypeParams(), RuntimeCheckableMetricTypeParams)
    assert isinstance(complex_metric_type_params, RuntimeCheckableMetricTypeParams)

def test_non_additive_dimension_satisfies_protocol(non_additive_dimension: NonAdditiveDimension) -> None:
    assert isinstance(non_additive_dimension, RuntimeCheckableNonAdditiveDimension)

@given(builds(SavedQuery, description=text() | none(), label=text() | none(), metadata=builds(SourceFileMetadata) | none()))
def test_saved_query_satisfies_protocol(saved_query: SavedQuery) -> None:
    assert isinstance(saved_query, SavedQuery)