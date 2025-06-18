from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Union
from dbt.artifacts.resources import ConversionTypeParams, CumulativeTypeParams, Dimension, DimensionTypeParams, Entity, Export, ExportConfig, ExposureConfig, Measure, MetricConfig, MetricInput, MetricInputMeasure, MetricTimeWindow, MetricTypeParams, NonAdditiveDimension, QueryParams, SavedQueryConfig, WhereFilter, WhereFilterIntersection
from dbt.artifacts.resources.v1.semantic_model import SemanticLayerElementConfig
from dbt.clients.jinja import get_rendered
from dbt.context.context_config import BaseContextConfigGenerator, ContextConfigGenerator, UnrenderedConfigGenerator
from dbt.context.providers import generate_parse_exposure, generate_parse_semantic_models
from dbt.contracts.files import SchemaSourceFile
from dbt.contracts.graph.nodes import Exposure, Group, Metric, SavedQuery, SemanticModel
from dbt.contracts.graph.unparsed import UnparsedConversionTypeParams, UnparsedCumulativeTypeParams, UnparsedDimension, UnparsedDimensionTypeParams, UnparsedEntity, UnparsedExport, UnparsedExposure, UnparsedGroup, UnparsedMeasure, UnparsedMetric, UnparsedMetricInput, UnparsedMetricInputMeasure, UnparsedMetricTypeParams, UnparsedNonAdditiveDimension, UnparsedQueryParams, UnparsedSavedQuery, UnparsedSemanticModel
from dbt.exceptions import JSONValidationError, YamlParseDictError
from dbt.node_types import NodeType
from dbt.parser.common import YamlBlock
from dbt.parser.schemas import ParseResult, SchemaParser, YamlReader
from dbt_common.dataclass_schema import ValidationError
from dbt_common.exceptions import DbtInternalError
from dbt_semantic_interfaces.type_enums import AggregationType, ConversionCalculationType, DimensionType, EntityType, MetricType, PeriodAggregation, TimeGranularity

def parse_where_filter(where):
    if where is None:
        return None
    elif isinstance(where, str):
        return WhereFilterIntersection([WhereFilter(where)])
    else:
        return WhereFilterIntersection([WhereFilter(where_str) for where_str in where])

class ExposureParser(YamlReader):

    def __init__(self, schema_parser, yaml):
        super().__init__(schema_parser, yaml, NodeType.Exposure.pluralize())
        self.schema_parser: SchemaParser = schema_parser
        self.yaml: YamlBlock = yaml

    def parse_exposure(self, unparsed):
        package_name: str = self.project.project_name
        unique_id: str = f'{NodeType.Exposure}.{package_name}.{unparsed.name}'
        path: str = self.yaml.path.relative_path
        fqn: List[str] = self.schema_parser.get_fqn_prefix(path)
        fqn.append(unparsed.name)
        config: ExposureConfig = self._generate_exposure_config(target=unparsed, fqn=fqn, package_name=package_name, rendered=True)
        config = config.finalize_and_validate()
        unrendered_config: ExposureConfig = self._generate_exposure_config(target=unparsed, fqn=fqn, package_name=package_name, rendered=False)
        if not isinstance(config, ExposureConfig):
            raise DbtInternalError(f'Calculated a {type(config)} for an exposure, but expected an ExposureConfig')
        parsed: Exposure = Exposure(resource_type=NodeType.Exposure, package_name=package_name, path=path, original_file_path=self.yaml.path.original_file_path, unique_id=unique_id, fqn=fqn, name=unparsed.name, type=unparsed.type, url=unparsed.url, meta=unparsed.meta, tags=unparsed.tags, description=unparsed.description, label=unparsed.label, owner=unparsed.owner, maturity=unparsed.maturity, config=config, unrendered_config=unrendered_config)
        ctx: Dict[str, Any] = generate_parse_exposure(parsed, self.root_project, self.schema_parser.manifest, package_name)
        depends_on_jinja: str = '\n'.join(('{{ ' + line + '}}' for line in unparsed.depends_on))
        get_rendered(depends_on_jinja, ctx, parsed, capture_macros=True)
        assert isinstance(self.yaml.file, SchemaSourceFile)
        if parsed.config.enabled:
            self.manifest.add_exposure(self.yaml.file, parsed)
        else:
            self.manifest.add_disabled(self.yaml.file, parsed)

    def _generate_exposure_config(self, target, fqn, package_name, rendered):
        generator: BaseContextConfigGenerator
        if rendered:
            generator = ContextConfigGenerator(self.root_project)
        else:
            generator = UnrenderedConfigGenerator(self.root_project)
        precedence_configs: Dict[str, Any] = dict()
        precedence_configs.update(target.config)
        return generator.calculate_node_config(config_call_dict={}, fqn=fqn, resource_type=NodeType.Exposure, project_name=package_name, base=False, patch_config_dict=precedence_configs)

    def parse(self):
        for data in self.get_key_dicts():
            try:
                UnparsedExposure.validate(data)
                unparsed: UnparsedExposure = UnparsedExposure.from_dict(data)
            except (ValidationError, JSONValidationError) as exc:
                raise YamlParseDictError(self.yaml.path, self.key, data, exc)
            self.parse_exposure(unparsed)

class MetricParser(YamlReader):

    def __init__(self, schema_parser, yaml):
        super().__init__(schema_parser, yaml, NodeType.Metric.pluralize())
        self.schema_parser: SchemaParser = schema_parser
        self.yaml: YamlBlock = yaml

    def _get_input_measure(self, unparsed_input_measure):
        if isinstance(unparsed_input_measure, str):
            return MetricInputMeasure(name=unparsed_input_measure)
        else:
            return MetricInputMeasure(name=unparsed_input_measure.name, filter=parse_where_filter(unparsed_input_measure.filter), alias=unparsed_input_measure.alias, join_to_timespine=unparsed_input_measure.join_to_timespine, fill_nulls_with=unparsed_input_measure.fill_nulls_with)

    def _get_optional_input_measure(self, unparsed_input_measure):
        if unparsed_input_measure is not None:
            return self._get_input_measure(unparsed_input_measure)
        else:
            return None

    def _get_input_measures(self, unparsed_input_measures):
        input_measures: List[MetricInputMeasure] = []
        if unparsed_input_measures is not None:
            for unparsed_input_measure in unparsed_input_measures:
                input_measures.append(self._get_input_measure(unparsed_input_measure))
        return input_measures

    def _get_period_agg(self, unparsed_period_agg):
        return PeriodAggregation(unparsed_period_agg)

    def _get_optional_time_window(self, unparsed_window):
        if unparsed_window is not None:
            parts: List[str] = unparsed_window.lower().split(' ')
            if len(parts) != 2:
                raise YamlParseDictError(self.yaml.path, 'window', {'window': unparsed_window}, f'Invalid window ({unparsed_window}) in cumulative/conversion metric. Should be of the form `<count> <granularity>`, e.g., `28 days`')
            granularity: str = parts[1]
            if granularity.endswith('s') and granularity[:-1] in [item.value for item in TimeGranularity]:
                granularity = granularity[:-1]
            count: str = parts[0]
            if not count.isdigit():
                raise YamlParseDictError(self.yaml.path, 'window', {'window': unparsed_window}, f'Invalid count ({count}) in cumulative/conversion metric window string: ({unparsed_window})')
            return MetricTimeWindow(count=int(count), granularity=granularity)
        else:
            return None

    def _get_metric_input(self, unparsed):
        if isinstance(unparsed, str):
            return MetricInput(name=unparsed)
        else:
            return MetricInput(name=unparsed.name, filter=parse_where_filter(unparsed.filter), alias=unparsed.alias, offset_window=self._get_optional_time_window(unparsed.offset_window), offset_to_grain=unparsed.offset_to_grain)

    def _get_optional_metric_input(self, unparsed):
        if unparsed is not None:
            return self._get_metric_input(unparsed)
        else:
            return None

    def _get_metric_inputs(self, unparsed_metric_inputs):
        metric_inputs: List[MetricInput] = []
        if unparsed_metric_inputs is not None:
            for unparsed_metric_input in unparsed_metric_inputs:
                metric_inputs.append(self._get_metric_input(unparsed=unparsed_metric_input))
        return metric_inputs

    def _get_optional_conversion_type_params(self, unparsed):
        if unparsed is None:
            return None
        return ConversionTypeParams(base_measure=self._get_input_measure(unparsed.base_measure), conversion_measure=self._get_input_measure(unparsed.conversion_measure), entity=unparsed.entity, calculation=ConversionCalculationType(unparsed.calculation), window=self._get_optional_time_window(unparsed.window), constant_properties=unparsed.constant_properties)

    def _get_optional_cumulative_type_params(self, unparsed_metric):
        unparsed_type_params = unparsed_metric.type_params
        if unparsed_metric.type.lower() == MetricType.CUMULATIVE.value:
            if not unparsed_type_params.cumulative_type_params:
                unparsed_type_params.cumulative_type_params = UnparsedCumulativeTypeParams()
            if unparsed_type_params.window and (not unparsed_type_params.cumulative_type_params.window):
                unparsed_type_params.cumulative_type_params.window = unparsed_type_params.window
            if unparsed_type_params.grain_to_date and (not unparsed_type_params.cumulative_type_params.grain_to_date):
                unparsed_type_params.cumulative_type_params.grain_to_date = unparsed_type_params.grain_to_date
            return CumulativeTypeParams(window=self._get_optional_time_window(unparsed_type_params.cumulative_type_params.window), grain_to_date=unparsed_type_params.cumulative_type_params.grain_to_date, period_agg=self._get_period_agg(unparsed_type_params.cumulative_type_params.period_agg))
        return None

    def _get_metric_type_params(self, unparsed_metric):
        type_params = unparsed_metric.type_params
        grain_to_date: Optional[TimeGranularity] = None
        if type_params.grain_to_date is not None:
            grain_to_date = TimeGranularity(type_params.grain_to_date)
        return MetricTypeParams(measure=self._get_optional_input_measure(type_params.measure), numerator=self._get_optional_metric_input(type_params.numerator), denominator=self._get_optional_metric_input(type_params.denominator), expr=str(type_params.expr) if type_params.expr is not None else None, window=self._get_optional_time_window(type_params.window), grain_to_date=grain_to_date, metrics=self._get_metric_inputs(type_params.metrics), conversion_type_params=self._get_optional_conversion_type_params(type_params.conversion_type_params), cumulative_type_params=self._get_optional_cumulative_type_params(unparsed_metric=unparsed_metric))

    def parse_metric(self, unparsed, generated_from=None):
        package_name: str = self.project.project_name
        unique_id: str = f'{NodeType.Metric}.{package_name}.{unparsed.name}'
        path: str = self.yaml.path.relative_path
        fqn: List[str] = self.schema_parser.get_fqn_prefix(path)
        fqn.append(unparsed.name)
        config: MetricConfig = self._generate_metric_config(target=unparsed, fqn=fqn, package_name=package_name, rendered=True)
        config = config.finalize_and_validate()
        unrendered_config: MetricConfig = self._generate_metric_config(target=unparsed, fqn=fqn, package_name=package_name, rendered=False)
        if not isinstance(config, MetricConfig):
            raise DbtInternalError(f'Calculated a {type(config)} for a metric, but expected a MetricConfig')
        if 'meta' in config and config['meta']:
            unparsed.meta = config['meta']
        parsed: Metric = Metric(resource_type=NodeType.Metric, package_name=package_name, path=path, original_file_path=self.yaml.path.original_file_path, unique_id=unique_id, fqn=fqn, name=unparsed.name, description=unparsed.description, label=unparsed.label, type=MetricType(unparsed.type), type_params=self._get_metric_type_params(unparsed), time_granularity=unparsed.time_granularity, filter=parse_where_filter(unparsed.filter), meta=unparsed.meta, tags=unparsed.tags, config=config, unrendered_config=unrendered_config, group=config.group)
        assert isinstance(self.yaml.file, SchemaSourceFile)
        if parsed.config.enabled:
            self.manifest.add_metric(self.yaml.file, parsed, generated_from)
        else:
            self.manifest.add_disabled(self.yaml.file, parsed)

    def _generate_metric_config(self, target, fqn, package_name, rendered):
        generator: BaseContextConfigGenerator
        if rendered:
            generator = ContextConfigGenerator(self.root_project)
        else:
            generator = UnrenderedConfigGenerator(self.root_project)
        precedence_configs: Dict[str, Any] = dict()
        precedence_configs.update(target.config)
        config: MetricConfig = generator.calculate_node_config(config_call_dict={}, fqn=fqn, resource_type=NodeType.Metric, project_name=package_name, base=False, patch_config_dict=precedence_configs)
        return config

    def parse(self):
        for data in self.get_key_dicts():
            try:
                UnparsedMetric.validate(data)
                unparsed: UnparsedMetric = UnparsedMetric.from_dict(data)
            except (ValidationError, JSONValidationError) as exc:
                raise YamlParseDictError(self.yaml.path, self.key, data, exc)
            self.parse_metric(unparsed)

class GroupParser(YamlReader):

    def __init__(self, schema_parser, yaml):
        super().__init__(schema_parser, yaml, NodeType.Group.pluralize())
        self.schema_parser: SchemaParser = schema_parser
        self.yaml: YamlBlock = yaml

    def parse_group(self, unparsed):
        package_name: str = self.project.project_name
        unique_id: str = f'{NodeType.Group}.{package_name}.{unparsed.name}'
        path: str = self.yaml.path.relative_path
        parsed: Group = Group(resource_type=NodeType.Group, package_name=package_name, path=path, original_file_path=self.yaml.path.original_file_path, unique_id=unique_id, name=unparsed.name, owner=unparsed.owner)
        assert isinstance(self.yaml.file, SchemaSourceFile)
        self.manifest.add_group(self.yaml.file, parsed)

    def parse(self):
        for data in self.get_key_dicts():
            try:
                UnparsedGroup.validate(data)
                unparsed: UnparsedGroup = UnparsedGroup.from_dict(data)
            except (ValidationError, JSONValidationError) as exc:
                raise YamlParseDictError(self.yaml.path, self.key, data, exc)
            self.parse_group(unparsed)

class SemanticModelParser(YamlReader):

    def __init__(self, schema_parser, yaml):
        super().__init__(schema_parser, yaml, 'semantic_models')
        self.schema_parser: SchemaParser = schema_parser
        self.yaml: YamlBlock = yaml

    def _get_dimension_type_params(self, unparsed):
        if unparsed is not None:
            return DimensionTypeParams(time_granularity=TimeGranularity(unparsed.time_granularity), validity_params=unparsed.validity_params)
        else:
            return None

    def _get_dimensions(self, unparsed_dimensions):
        dimensions: List[Dimension] = []
        for unparsed in unparsed_dimensions:
            dimensions.append(Dimension(name=unparsed.name, type=DimensionType(unparsed.type), description=unparsed.description, label=unparsed.label, is_partition=unparsed.is_partition, type_params=self._get_dimension_type_params(unparsed=unparsed.type_params), expr=unparsed.expr, metadata=None, config=SemanticLayerElementConfig(meta=unparsed.config.get('meta', {}))))
        return dimensions

    def _get_entities(self, unparsed_entities):
        entities: List[Entity] = []
        for unparsed in unparsed_entities:
            entities.append(Entity(name=unparsed.name, type=EntityType(unparsed.type), description=unparsed.description, label=unparsed.label, role=unparsed.role, expr=unparsed.expr, config=SemanticLayerElementConfig(meta=unparsed.config.get('meta', {}))))
        return entities

    def _get_non_additive_dimension(self, unparsed):
        if unparsed is not None:
            return NonAdditiveDimension(name=unparsed.name, window_choice=AggregationType(unparsed.window_choice), window_groupings=unparsed.window_groupings)
        else:
            return None

    def _get_measures(self, unparsed_measures):
        measures: List[Measure] = []
        for unparsed in unparsed_measures:
            measures.append(Measure(name=unparsed.name, agg=AggregationType(unparsed.agg), description=unparsed.description, label=unparsed.label, expr=str(unparsed.expr) if unparsed.expr is not None else None, agg_params=unparsed.agg_params, non_additive_dimension=self._get_non_additive_dimension(unparsed.non_additive_dimension), agg_time_dimension=unparsed.agg_time_dimension, config=SemanticLayerElementConfig(meta=unparsed.config.get('meta', {}))))
        return measures

    def _create_metric(self, measure, enabled, semantic_model_name):
        unparsed_metric: UnparsedMetric = UnparsedMetric(name=measure.name, label=measure.label or measure.name, type='simple', type_params=UnparsedMetricTypeParams(measure=measure.name, expr=measure.name), description=measure.description or f'Metric created from measure {measure.name}', config={'enabled': enabled})
        parser: MetricParser = MetricParser(self.schema_parser, yaml=self.yaml)
        parser.parse_metric(unparsed=unparsed_metric, generated_from=semantic_model_name)

    def _generate_semantic_model_config(self, target, fqn, package_name, rendered):
        generator: BaseContextConfigGenerator
        if rendered:
            generator = ContextConfigGenerator(self.root_project)
        else:
            generator = UnrenderedConfigGenerator(self.root_project)
        precedence_configs: Dict[str, Any] = dict()
        precedence_configs.update(target.config)
        config: SemanticLayerElementConfig = generator.calculate_node_config(config_call_dict={}, fqn=fqn, resource_type=NodeType.SemanticModel, project_name=package_name, base=False, patch_config_dict=precedence_configs)
        return config

    def parse_semantic_model(self, unparsed):
        package_name: str = self.project.project_name
        unique_id: str = f'{NodeType.SemanticModel}.{package_name}.{unparsed.name}'
        path: str = self.yaml.path.relative_path
        fqn: List[str] = self.schema_parser.get_fqn_prefix(path)
        fqn.append(unparsed.name)
        entities: List[Entity] = self._get_entities(unparsed.entities)
        measures: List[Measure] = self._get_measures(unparsed.measures)
        dimensions: List[Dimension] = self._get_dimensions(unparsed.dimensions)
        config: SemanticLayerElementConfig = self._generate_semantic_model_config(target=unparsed, fqn=fqn, package_name=package_name, rendered=True)
        elements: Sequence[Union[Dimension, Entity, Measure]] = [*dimensions, *entities, *measures]
        for element in elements:
            if config is not None:
                if element.config is None:
                    element.config = SemanticLayerElementConfig(meta=config.meta)
                else:
                    element.config.meta = {**config.get('meta', {}), **element.config.meta}
        config = config.finalize_and_validate()
        unrendered_config: SemanticLayerElementConfig = self._generate_semantic_model_config(target=unparsed, fqn=fqn, package_name=package_name, rendered=False)
        parsed: SemanticModel = SemanticModel(description=unparsed.description, label=unparsed.label, fqn=fqn, model=unparsed.model, name=unparsed.name, node_relation=None, original_file_path=self.yaml.path.original_file_path, package_name=package_name, path=path, resource_type=NodeType.SemanticModel, unique_id=unique_id, entities=entities, measures=measures, dimensions=dimensions, defaults=unparsed.defaults, primary_entity=unparsed.primary_entity, config=config, unrendered_config=unrendered_config, group=config.group)
        ctx: Dict[str, Any] = generate_parse_semantic_models(parsed, self.root_project, self.schema_parser.manifest, package_name)
        if parsed.model is not None:
            model_ref: str = '{{ ' + parsed.model + ' }}'
            get_rendered(model_ref, ctx, parsed)
        assert isinstance(self.yaml.file, SchemaSourceFile)
        if parsed.config.enabled:
            self.manifest.add_semantic_model(self.yaml.file, parsed)
        else:
            self.manifest.add_disabled(self.yaml.file, parsed)
        for measure in unparsed.measures:
            if measure.create_metric is True:
                self._create_metric(measure=measure, enabled=parsed.config.enabled, semantic_model_name=parsed.name)

    def parse(self):
        for data in self.get_key_dicts():
            try:
                UnparsedSemanticModel.validate(data)
                unparsed: UnparsedSemanticModel = UnparsedSemanticModel.from_dict(data)
            except (ValidationError, JSONValidationError) as exc:
                raise YamlParseDictError(self.yaml.path, self.key, data, exc)
            self.parse_semantic_model(unparsed)

class SavedQueryParser(YamlReader):

    def __init__(self, schema_parser, yaml):
        super().__init__(schema_parser, yaml, 'saved_queries')
        self.schema_parser: SchemaParser = schema_parser
        self.yaml: YamlBlock = yaml

    def _generate_saved_query_config(self, target, fqn, package_name, rendered):
        generator: BaseContextConfigGenerator
        if rendered:
            generator = ContextConfigGenerator(self.root_project)
        else:
            generator = UnrenderedConfigGenerator(self.root_project)
        precedence_configs: Dict[str, Any] = dict()
        precedence_configs.update(target.config)
        config: SavedQueryConfig = generator.calculate_node_config(config_call_dict={}, fqn=fqn, resource_type=NodeType.SavedQuery, project_name=package_name, base=False, patch_config_dict=precedence_configs)
        return config

    def _get_export_config(self, unparsed_export_config, saved_query_config):
        combined: Dict[str, Any] = {**saved_query_config.__dict__, **unparsed_export_config}
        if combined.get('schema') is not None and combined.get('schema_name') is None:
            combined['schema_name'] = combined['schema']
        return ExportConfig.from_dict(combined)

    def _get_export(self, unparsed, saved_query_config):
        return Export(name=unparsed.name, config=self._get_export_config(unparsed.config, saved_query_config), unrendered_config=unparsed.config)

    def _get_query_params(self, unparsed):
        return QueryParams(group_by=unparsed.group_by, metrics=unparsed.metrics, where=parse_where_filter(unparsed.where), order_by=unparsed.order_by, limit=unparsed.limit)

    def parse_saved_query(self, unparsed):
        package_name: str = self.project.project_name
        unique_id: str = f'{NodeType.SavedQuery}.{package_name}.{unparsed.name}'
        path: str = self.yaml.path.relative_path
        fqn: List[str] = self.schema_parser.get_fqn_prefix(path)
        fqn.append(unparsed.name)
        config: SavedQueryConfig = self._generate_saved_query_config(target=unparsed, fqn=fqn, package_name=package_name, rendered=True)
        config = config.finalize_and_validate()
        unrendered_config: SavedQueryConfig = self._generate_saved_query_config(target=unparsed, fqn=fqn, package_name=package_name, rendered=False)

        def wrap_tags(s):
            if s is None:
                return []
            return [s] if isinstance(s, str) else s
        config_tags: List[str] = wrap_tags(config.get('tags'))
        unparsed_tags: List[str] = wrap_tags(unparsed.tags)
        tags: List[str] = list(set([*unparsed_tags, *config_tags]))
        tags.sort()
        parsed: SavedQuery = SavedQuery(description=unparsed.description, label=unparsed.label, fqn=fqn, name=unparsed.name, original_file_path=self.yaml.path.original_file_path, package_name=package_name, path=path, resource_type=NodeType.SavedQuery, unique_id=unique_id, query_params=self._get_query_params(unparsed.query_params), exports=[self._get_export(export, config) for export in unparsed.exports], config=config, unrendered_config=unrendered_config, group=config.group, tags=tags)
        for export in parsed.exports:
            self.schema_parser.update_parsed_node_relation_names(export, export.config.to_dict())
            if not export.config.schema_name:
                export.config.schema_name = getattr(export, 'schema', None)
            delattr(export, 'schema')
            export.config.database = getattr(export, 'database', None) or export.config.database
            delattr(export, 'database')
            if not export.config.alias:
                export.config.alias = getattr(export, 'alias', None)
            delattr(export, 'alias')
            delattr(export, 'relation_name')
        assert isinstance(self.yaml.file, SchemaSourceFile)
        if parsed.config.enabled:
            self.manifest.add_saved_query(self.yaml.file, parsed)
        else:
            self.manifest.add_disabled(self.yaml.file, parsed)

    def parse(self):
        for data in self.get_key_dicts():
            try:
                UnparsedSavedQuery.validate(data)
                unparsed: UnparsedSavedQuery = UnparsedSavedQuery.from_dict(data)
            except (ValidationError, JSONValidationError) as exc:
                raise YamlParseDictError(self.yaml.path, self.key, data, exc)
            self.parse_saved_query(unparsed)
        return ParseResult()