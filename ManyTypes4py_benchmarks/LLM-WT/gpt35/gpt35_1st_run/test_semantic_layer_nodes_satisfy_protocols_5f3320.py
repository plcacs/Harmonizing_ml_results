from dbt_semantic_interfaces.protocols import WhereFilter as WhereFilterProtocol
from dbt_semantic_interfaces.protocols import dimension as DimensionProtocols
from dbt_semantic_interfaces.protocols import entity as EntityProtocols
from dbt_semantic_interfaces.protocols import measure as MeasureProtocols
from dbt_semantic_interfaces.protocols import metadata as MetadataProtocols
from dbt_semantic_interfaces.protocols import metric as MetricProtocols
from dbt_semantic_interfaces.protocols import saved_query as SavedQueryProtocols
from dbt_semantic_interfaces.protocols import semantic_model as SemanticModelProtocols
from dbt_semantic_interfaces.type_enums import AggregationType, DimensionType, EntityType, MetricType, TimeGranularity
from dbt.artifacts.resources import ConstantPropertyInput, ConversionTypeParams, CumulativeTypeParams, Defaults, Dimension, DimensionTypeParams, DimensionValidityParams, Entity, FileSlice, Measure, MeasureAggregationParameters, MetricInput, MetricInputMeasure, MetricTimeWindow, MetricTypeParams, NodeRelation, NonAdditiveDimension, SourceFileMetadata, WhereFilter
from dbt.contracts.graph.nodes import Metric, SavedQuery, SemanticModel
from dbt.node_types import NodeType

from typing import Protocol, runtime_checkable
import copy
import pytest
from hypothesis import given
from hypothesis.strategies import builds, none, text

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
