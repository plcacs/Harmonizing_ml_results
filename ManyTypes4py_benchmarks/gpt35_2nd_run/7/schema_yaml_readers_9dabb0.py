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
from typing import Any, Dict, List, Optional, Union

def parse_where_filter(where: Union[str, List[str]]) -> Optional[WhereFilterIntersection]:
    if where is None:
        return None
    elif isinstance(where, str):
        return WhereFilterIntersection([WhereFilter(where)])
    else:
        return WhereFilterIntersection([WhereFilter(where_str) for where_str in where])

class ExposureParser(YamlReader):

    def __init__(self, schema_parser: SchemaParser, yaml: YamlBlock):
        super().__init__(schema_parser, yaml, NodeType.Exposure.pluralize())
        self.schema_parser = schema_parser
        self.yaml = yaml

    def parse_exposure(self, unparsed: UnparsedExposure) -> None:
        package_name: str = self.project.project_name
        unique_id: str = f'{NodeType.Exposure}.{package_name}.{unparsed.name}'
        path: str = self.yaml.path.relative_path
        fqn: List[str] = self.schema_parser.get_fqn_prefix(path)
        fqn.append(unparsed.name)
        config = self._generate_exposure_config(target=unparsed, fqn=fqn, package_name=package_name, rendered=True)
        config = config.finalize_and_validate()
        unrendered_config = self._generate_exposure_config(target=unparsed, fqn=fqn, package_name=package_name, rendered=False)
        if not isinstance(config, ExposureConfig):
            raise DbtInternalError(f'Calculated a {type(config)} for an exposure, but expected an ExposureConfig')
        parsed = Exposure(resource_type=NodeType.Exposure, package_name=package_name, path=path, original_file_path=self.yaml.path.original_file_path, unique_id=unique_id, fqn=fqn, name=unparsed.name, type=unparsed.type, url=unparsed.url, meta=unparsed.meta, tags=unparsed.tags, description=unparsed.description, label=unparsed.label, owner=unparsed.owner, maturity=unparsed.maturity, config=config, unrendered_config=unrendered_config)
        ctx = generate_parse_exposure(parsed, self.root_project, self.schema_parser.manifest, package_name)
        depends_on_jinja = '\n'.join(('{{ ' + line + '}}' for line in unparsed.depends_on))
        get_rendered(depends_on_jinja, ctx, parsed, capture_macros=True)
        assert isinstance(self.yaml.file, SchemaSourceFile)
        if parsed.config.enabled:
            self.manifest.add_exposure(self.yaml.file, parsed)
        else:
            self.manifest.add_disabled(self.yaml.file, parsed)

    def _generate_exposure_config(self, target: UnparsedExposure, fqn: List[str], package_name: str, rendered: bool) -> Union[ExposureConfig, Dict[str, Any]]:
        if rendered:
            generator = ContextConfigGenerator(self.root_project)
        else:
            generator = UnrenderedConfigGenerator(self.root_project)
        precedence_configs: Dict[str, Any] = dict()
        precedence_configs.update(target.config)
        return generator.calculate_node_config(config_call_dict={}, fqn=fqn, resource_type=NodeType.Exposure, project_name=package_name, base=False, patch_config_dict=precedence_configs)

    def parse(self) -> None:
        for data in self.get_key_dicts():
            try:
                UnparsedExposure.validate(data)
                unparsed = UnparsedExposure.from_dict(data)
            except (ValidationError, JSONValidationError) as exc:
                raise YamlParseDictError(self.yaml.path, self.key, data, exc)
            self.parse_exposure(unparsed)
