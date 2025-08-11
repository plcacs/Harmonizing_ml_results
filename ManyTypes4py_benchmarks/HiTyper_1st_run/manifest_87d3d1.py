from typing import Any, Dict, List
import pytest
from dbt.artifacts.resources import ExposureType, MacroDependsOn, MetricInputMeasure, MetricTypeParams, NodeRelation, Owner, QueryParams, RefArgs, TestConfig, TestMetadata, WhereFilter, WhereFilterIntersection
from dbt.artifacts.resources.types import ModelLanguage
from dbt.artifacts.resources.v1.model import ModelConfig
from dbt.contracts.files import AnySourceFile, FileHash
from dbt.contracts.graph.manifest import Manifest, ManifestMetadata
from dbt.contracts.graph.nodes import AccessType, DependsOn, Documentation, Exposure, GenericTestNode, GraphMemberNode, Group, Macro, ManifestNode, Metric, ModelNode, NodeConfig, SavedQuery, SeedNode, SemanticModel, SingularTestNode, SourceDefinition, UnitTestDefinition
from dbt.contracts.graph.unparsed import UnitTestInputFixture, UnitTestOutputFixture
from dbt.node_types import NodeType
from dbt_semantic_interfaces.type_enums import MetricType

def make_model(pkg: Union[str, None, LocalizedString], name: Union[str, None, dict], code: Union[str, None, typing.Iterable[str]], language: typing.Text='sql', refs: Union[None, str, dict[str, str]]=None, sources: Union[None, str, dict[str, str]]=None, tags: Union[None, str]=None, path: Union[None, str, dict]=None, alias: Union[None, str, dict[str, str]]=None, config_kwargs: Union[None, str]=None, fqn_extras: Union[None, str, bool]=None, depends_on_macros: Union[None, str, typing.Any]=None, version: Union[None, str, list]=None, latest_version: Union[None, str, typing.Iterable[str]]=None, access: Union[None, str, typing.Iterable[str]]=None, patch_path: Union[None, str, typing.Iterable[str]]=None) -> ModelNode:
    if refs is None:
        refs = []
    if sources is None:
        sources = []
    if tags is None:
        tags = []
    if path is None:
        if language == 'sql':
            path = f'{name}.sql'
        elif language == 'python':
            path = f'{name}.py'
        else:
            raise ValueError(f'Unknown language: {language}')
    if alias is None:
        alias = name
    if config_kwargs is None:
        config_kwargs = {}
    if depends_on_macros is None:
        depends_on_macros = []
    if fqn_extras is None:
        fqn_extras = []
    fqn = [pkg] + fqn_extras + [name]
    if version:
        fqn.append(f'v{version}')
    depends_on_nodes = []
    source_values = []
    ref_values = []
    for ref in refs:
        ref_version = ref.version if hasattr(ref, 'version') else None
        ref_values.append(RefArgs(name=ref.name, package=ref.package_name, version=ref_version))
        depends_on_nodes.append(ref.unique_id)
    for src in sources:
        source_values.append([src.source_name, src.name])
        depends_on_nodes.append(src.unique_id)
    return ModelNode(language='sql', raw_code=code, database='dbt', schema='dbt_schema', alias=alias, name=name, fqn=fqn, unique_id=f'model.{pkg}.{name}' if not version else f'model.{pkg}.{name}.v{version}', package_name=pkg, path=path, original_file_path=f'models/{path}', config=NodeConfig(**config_kwargs), tags=tags, refs=ref_values, sources=source_values, depends_on=DependsOn(nodes=depends_on_nodes, macros=depends_on_macros), resource_type=NodeType.Model, checksum=FileHash.from_contents(''), version=version, latest_version=latest_version, access=access or AccessType.Protected, patch_path=patch_path)

def make_seed(pkg: Union[str, None, list[str]], name: Union[str, None, bool, list[str]], path: Union[None, str, typing.Any, bool]=None, loader: Union[None, bool, str]=None, alias: Union[None, tuple[str], str, dict[str, str]]=None, tags: Union[None, tuple[str], str]=None, fqn_extras: Union[None, str, dict]=None, checksum: Union[None, str, typing.IO]=None) -> SeedNode:
    if alias is None:
        alias = name
    if tags is None:
        tags = []
    if path is None:
        path = f'{name}.csv'
    if fqn_extras is None:
        fqn_extras = []
    if checksum is None:
        checksum = FileHash.from_contents('')
    fqn = [pkg] + fqn_extras + [name]
    return SeedNode(database='dbt', schema='dbt_schema', alias=alias, name=name, fqn=fqn, unique_id=f'seed.{pkg}.{name}', package_name=pkg, path=path, original_file_path=f'data/{path}', tags=tags, resource_type=NodeType.Seed, checksum=FileHash.from_contents(''))

def make_source(pkg: Union[str, int], source_name: Union[str, int], table_name: Union[str, dict], path: Union[None, str]=None, loader: Union[None, str]=None, identifier: Union[None, str, bool]=None, fqn_extras: Union[None, str]=None) -> SourceDefinition:
    if path is None:
        path = 'models/schema.yml'
    if loader is None:
        loader = 'my_loader'
    if identifier is None:
        identifier = table_name
    if fqn_extras is None:
        fqn_extras = []
    fqn = [pkg] + fqn_extras + [source_name, table_name]
    return SourceDefinition(fqn=fqn, database='dbt', schema='dbt_schema', unique_id=f'source.{pkg}.{source_name}.{table_name}', package_name=pkg, path=path, original_file_path=path, name=table_name, source_name=source_name, loader='my_loader', identifier=identifier, resource_type=NodeType.Source, loaded_at_field='loaded_at', tags=[], source_description='')

def make_macro(pkg: Union[str, bool], name: Union[str, bool], macro_sql: Union[str, bool], path: Union[None, str, bool]=None, depends_on_macros: Union[None, str, list[str], list[dict]]=None) -> Macro:
    if path is None:
        path = 'macros/macros.sql'
    if depends_on_macros is None:
        depends_on_macros = []
    return Macro(name=name, macro_sql=macro_sql, unique_id=f'macro.{pkg}.{name}', package_name=pkg, path=path, original_file_path=path, resource_type=NodeType.Macro, depends_on=MacroDependsOn(macros=depends_on_macros))

def make_unique_test(pkg: Union[str, None, list[str], bool], test_model: Union[str, None, list[str], bool], column_name: Union[str, None, list[str], bool], path: Union[None, str, tuple, list[str]]=None, refs: Union[None, str, tuple, list[str]]=None, sources: Union[None, str, tuple, list[str]]=None, tags: Union[None, str, tuple, list[str]]=None) -> str:
    return make_generic_test(pkg, 'unique', test_model, {}, column_name=column_name)

def make_not_null_test(pkg: Union[str, None, list[str], bool], test_model: Union[str, None, list[str], bool], column_name: Union[str, None, list[str], bool], path: Union[None, str, tuple, list[str]]=None, refs: Union[None, str, tuple, list[str]]=None, sources: Union[None, str, tuple, list[str]]=None, tags: Union[None, str, tuple, list[str]]=None) -> str:
    return make_generic_test(pkg, 'not_null', test_model, {}, column_name=column_name)

def make_generic_test(pkg: Union[str, typing.Mapping, None, bool], test_name: str, test_model: Union[str, dict, typing.MutableSequence], test_kwargs: dict, path: Union[None, str, dict]=None, refs: Union[None, str, dict]=None, sources: Union[None, str, dict]=None, tags: Union[None, str, dict]=None, column_name: Union[None, str]=None) -> GenericTestNode:
    kwargs = test_kwargs.copy()
    ref_values = []
    source_values = []
    if isinstance(test_model, SourceDefinition):
        kwargs['model'] = "{{ source('" + test_model.source_name + "', '" + test_model.name + "') }}"
        source_values.append([test_model.source_name, test_model.name])
    else:
        kwargs['model'] = "{{ ref('" + test_model.name + "')}}"
        ref_values.append(RefArgs(name=test_model.name, package=test_model.package_name, version=test_model.version))
    if column_name is not None:
        kwargs['column_name'] = column_name
    args_name = test_model.search_name.replace('.', '_')
    if column_name is not None:
        args_name += '_' + column_name
    node_name = f'{test_name}_{args_name}'
    raw_code = '{{ config(severity="ERROR") }}{{ test_' + test_name + '(**dbt_schema_test_kwargs) }}'
    name_parts = test_name.split('.')
    if len(name_parts) == 2:
        namespace, test_name = name_parts
        macro_depends = f'macro.{namespace}.test_{test_name}'
    elif len(name_parts) == 1:
        namespace = None
        macro_depends = f'macro.dbt.test_{test_name}'
    else:
        assert False, f'invalid test name: {test_name}'
    if path is None:
        path = 'schema.yml'
    if tags is None:
        tags = ['schema']
    if refs is None:
        refs = []
    if sources is None:
        sources = []
    depends_on_nodes = []
    for ref in refs:
        ref_version = ref.version if hasattr(ref, 'version') else None
        ref_values.append(RefArgs(name=ref.name, package=ref.package_name, version=ref_version))
        depends_on_nodes.append(ref.unique_id)
    for source in sources:
        source_values.append([source.source_name, source.name])
        depends_on_nodes.append(source.unique_id)
    return GenericTestNode(language='sql', raw_code=raw_code, test_metadata=TestMetadata(namespace=namespace, name=test_name, kwargs=kwargs), database='dbt', schema='dbt_postgres', name=node_name, alias=node_name, fqn=['minimal', 'schema_test', node_name], unique_id=f'test.{pkg}.{node_name}', package_name=pkg, path=f'schema_test/{node_name}.sql', original_file_path=f'models/{path}', resource_type=NodeType.Test, tags=tags, refs=ref_values, sources=[], depends_on=DependsOn(macros=[macro_depends], nodes=depends_on_nodes), column_name=column_name, checksum=FileHash.from_contents(''))

def make_unit_test(pkg: Union[str, None], test_name: Union[str, None], test_model: Union[str, None]) -> UnitTestDefinition:
    input_fixture = UnitTestInputFixture(input="ref('table_model')", rows=[{'id': 1, 'string_a': 'a'}])
    output_fixture = UnitTestOutputFixture(rows=[{'id': 1, 'string_a': 'a'}])
    return UnitTestDefinition(name=test_name, model=test_model, package_name=pkg, resource_type=NodeType.Unit, path='unit_tests.yml', original_file_path='models/unit_tests.yml', unique_id=f'unit.{pkg}.{test_model.name}__{test_name}', given=[input_fixture], expect=output_fixture, fqn=[pkg, test_model.name, test_name])

def make_singular_test(pkg: Union[str, None, typing.Iterable[str]], name: Union[str, dict, None, dict[str, str]], sql: Union[str, None, typing.Iterable[str]], refs: Union[None, str, dict[str, str]]=None, sources: Union[None, str, dict[str, str]]=None, tags: Union[None, str, dict[str, str]]=None, path: Union[None, str, dict[str, typing.Any]]=None, config_kwargs: Union[None, str, dict, dict[str, typing.Any]]=None) -> SingularTestNode:
    if refs is None:
        refs = []
    if sources is None:
        sources = []
    if tags is None:
        tags = ['data']
    if path is None:
        path = f'{name}.sql'
    if config_kwargs is None:
        config_kwargs = {}
    fqn = ['minimal', 'data_test', name]
    depends_on_nodes = []
    source_values = []
    ref_values = []
    for ref in refs:
        ref_version = ref.version if hasattr(ref, 'version') else None
        ref_values.append(RefArgs(name=ref.name, package=ref.package_name, version=ref_version))
        depends_on_nodes.append(ref.unique_id)
    for src in sources:
        source_values.append([src.source_name, src.name])
        depends_on_nodes.append(src.unique_id)
    return SingularTestNode(language='sql', raw_code=sql, database='dbt', schema='dbt_schema', name=name, alias=name, fqn=fqn, unique_id=f'test.{pkg}.{name}', package_name=pkg, path=path, original_file_path=f'tests/{path}', config=TestConfig(**config_kwargs), tags=tags, refs=ref_values, sources=source_values, depends_on=DependsOn(nodes=depends_on_nodes, macros=[]), resource_type=NodeType.Test, checksum=FileHash.from_contents(''))

def make_exposure(pkg: Union[str, None], name: Union[str, None], path: Union[None, str]=None, fqn_extras: Union[None, str, list[str]]=None, owner: Union[None, str]=None) -> Exposure:
    if path is None:
        path = 'schema.yml'
    if fqn_extras is None:
        fqn_extras = []
    if owner is None:
        owner = Owner(email='test@example.com')
    fqn = [pkg, 'exposures'] + fqn_extras + [name]
    return Exposure(name=name, resource_type=NodeType.Exposure, type=ExposureType.Notebook, fqn=fqn, unique_id=f'exposure.{pkg}.{name}', package_name=pkg, path=path, original_file_path=path, owner=owner)

def make_metric(pkg: str, name: str, path: Union[None, str]=None) -> Metric:
    if path is None:
        path = 'schema.yml'
    return Metric(name=name, resource_type=NodeType.Metric, path=path, package_name=pkg, original_file_path=path, unique_id=f'metric.{pkg}.{name}', fqn=[pkg, 'metrics', name], label='New Customers', description='New customers', type=MetricType.SIMPLE, type_params=MetricTypeParams(measure=MetricInputMeasure(name='count_cats')), meta={'is_okr': True}, tags=['okrs'])

def make_group(pkg: str, name: str, path: Union[None, str]=None) -> Group:
    if path is None:
        path = 'schema.yml'
    return Group(name=name, resource_type=NodeType.Group, path=path, package_name=pkg, original_file_path=path, unique_id=f'group.{pkg}.{name}', owner='email@gmail.com')

def make_semantic_model(pkg: Union[str, None, types.ModuleType], name: Union[str, None, types.ModuleType], model: Union[str, None, types.ModuleType], path: Union[None, str, list[str]]=None) -> SemanticModel:
    if path is None:
        path = 'schema.yml'
    return SemanticModel(name=name, resource_type=NodeType.SemanticModel, model=model, node_relation=NodeRelation(alias=model.alias, schema_name='dbt', relation_name=model.name), package_name=pkg, path=path, description='Customer entity', primary_entity='customer', unique_id=f'semantic_model.{pkg}.{name}', original_file_path=path, fqn=[pkg, 'semantic_models', name])

def make_saved_query(pkg: str, name: str, metric: str, path: Union[None, str, tuple[str]]=None) -> SavedQuery:
    if path is None:
        path = 'schema.yml'
    return SavedQuery(name=name, resource_type=NodeType.SavedQuery, package_name=pkg, path=path, description='Test Saved Query', query_params=QueryParams(metrics=[metric], group_by=[], where=None), exports=[], unique_id=f'saved_query.{pkg}.{name}', original_file_path=path, fqn=[pkg, 'saved_queries', name])

@pytest.fixture
def macro_test_unique() -> str:
    return make_macro('dbt', 'test_unique', 'blablabla', depends_on_macros=['macro.dbt.default__test_unique'])

@pytest.fixture
def macro_default_test_unique() -> str:
    return make_macro('dbt', 'default__test_unique', 'blablabla')

@pytest.fixture
def macro_test_not_null() -> Union[str, docutils.nodes.Node, list[str]]:
    return make_macro('dbt', 'test_not_null', 'blablabla', depends_on_macros=['macro.dbt.default__test_not_null'])

@pytest.fixture
def macro_materialization_table_default() -> Union[dict[str, str], dict, str]:
    macro = make_macro('dbt', 'materialization_table_default', 'SELECT 1')
    macro.supported_languages = [ModelLanguage.sql]
    return macro

@pytest.fixture
def macro_default_test_not_null() -> Union[str, None, bool]:
    return make_macro('dbt', 'default__test_not_null', 'blabla')

@pytest.fixture
def seed() -> Union[str, None, typing.BinaryIO]:
    return make_seed('pkg', 'seed')

@pytest.fixture
def source() -> Union[str, list[str]]:
    return make_source('pkg', 'raw', 'seed', identifier='seed')

@pytest.fixture
def ephemeral_model(source: str) -> Union[str, int]:
    return make_model('pkg', 'ephemeral_model', 'select * from {{ source("raw", "seed") }}', config_kwargs={'materialized': 'ephemeral'}, sources=[source])

@pytest.fixture
def view_model(ephemeral_model: Union[dict, dict[str, typing.Callable]]):
    return make_model('pkg', 'view_model', 'select * from {{ ref("ephemeral_model") }}', config_kwargs={'materialized': 'view'}, refs=[ephemeral_model], tags=['uses_ephemeral'])

@pytest.fixture
def table_model(ephemeral_model: dict) -> Union[list[dbcontracts.graph.compiled.InjectedCTE], str]:
    return make_model('pkg', 'table_model', 'select * from {{ ref("ephemeral_model") }}', config_kwargs={'materialized': 'table', 'meta': {'string_property': 'some_string', 'truthy_bool_property': True, 'falsy_bool_property': False, 'list_property': ['some_value', True, False]}}, refs=[ephemeral_model], tags=['uses_ephemeral'], path='subdirectory/table_model.sql')

@pytest.fixture
def table_model_py(seed: Union[int, str, None]) -> Union[str, rflx.model.Model, bool]:
    return make_model('pkg', 'table_model_py', 'select * from {{ ref("seed") }}', config_kwargs={'materialized': 'table'}, refs=[seed], tags=[], path='subdirectory/table_model.py')

@pytest.fixture
def table_model_csv(seed: Union[str, None, int]) -> Union[str, rflx.model.Model]:
    return make_model('pkg', 'table_model_csv', 'select * from {{ ref("seed") }}', config_kwargs={'materialized': 'table'}, refs=[seed], tags=[], path='subdirectory/table_model.csv')

@pytest.fixture
def ext_source() -> str:
    return make_source('ext', 'ext_raw', 'ext_source')

@pytest.fixture
def ext_source_2() -> str:
    return make_source('ext', 'ext_raw', 'ext_source_2')

@pytest.fixture
def ext_source_other() -> str:
    return make_source('ext', 'raw', 'ext_source')

@pytest.fixture
def ext_source_other_2() -> str:
    return make_source('ext', 'raw', 'ext_source_2')

@pytest.fixture
def ext_model(ext_source: Union[str, bytes, typing.IO]) -> str:
    return make_model('ext', 'ext_model', 'select * from {{ source("ext_raw", "ext_source") }}', sources=[ext_source])

@pytest.fixture
def union_model(seed: Union[str, None, typing.IO], ext_source: Union[str, None, typing.IO]) -> Union[list, str]:
    return make_model('pkg', 'union_model', 'select * from {{ ref("seed") }} union all select * from {{ source("ext_raw", "ext_source") }}', config_kwargs={'materialized': 'table'}, refs=[seed], sources=[ext_source], fqn_extras=['unions'], path='subdirectory/union_model.sql', tags=['unions'])

@pytest.fixture
def versioned_model_v1(seed: Union[str, int, None]) -> str:
    return make_model('pkg', 'versioned_model', 'select * from {{ ref("seed") }}', config_kwargs={'materialized': 'table'}, refs=[seed], sources=[], path='subdirectory/versioned_model_v1.sql', version=1, latest_version=2)

@pytest.fixture
def versioned_model_v2(seed: Union[str, int, None]) -> str:
    return make_model('pkg', 'versioned_model', 'select * from {{ ref("seed") }}', config_kwargs={'materialized': 'table'}, refs=[seed], sources=[], path='subdirectory/versioned_model_v2.sql', version=2, latest_version=2)

@pytest.fixture
def versioned_model_v3(seed: Union[str, int, None]) -> str:
    return make_model('pkg', 'versioned_model', 'select * from {{ ref("seed") }}', config_kwargs={'materialized': 'table'}, refs=[seed], sources=[], path='subdirectory/versioned_model_v3.sql', version='3', latest_version=2)

@pytest.fixture
def versioned_model_v12_string(seed: Union[str, int]) -> str:
    return make_model('pkg', 'versioned_model', 'select * from {{ ref("seed") }}', config_kwargs={'materialized': 'table'}, refs=[seed], sources=[], path='subdirectory/versioned_model_v12.sql', version='12', latest_version=2)

@pytest.fixture
def versioned_model_v4_nested_dir(seed: Union[str, int]) -> nufb.wrappers.Manifest:
    return make_model('pkg', 'versioned_model', 'select * from {{ ref("seed") }}', config_kwargs={'materialized': 'table'}, refs=[seed], sources=[], path='subdirectory/nested_dir/versioned_model_v3.sql', version='4', latest_version=2, fqn_extras=['nested_dir'])

@pytest.fixture
def table_id_unique(table_model: allennlp.models.model.Model) -> Union[str, tuple]:
    return make_unique_test('pkg', table_model, 'id')

@pytest.fixture
def table_id_not_null(table_model: Union[list[dbcontracts.graph.compiled.InjectedCTE], dbcontracts.graph.compiled.ManifestNode]) -> Union[bool, str]:
    return make_not_null_test('pkg', table_model, 'id')

@pytest.fixture
def view_id_unique(view_model: Any):
    return make_unique_test('pkg', view_model, 'id')

@pytest.fixture
def ext_source_id_unique(ext_source: Union[str, Path, bytes]) -> Union[str, list[str]]:
    return make_unique_test('ext', ext_source, 'id')

@pytest.fixture
def view_test_nothing(view_model: Union[dict[str, typing.Any], dict[str, str], None, dict]):
    return make_singular_test('pkg', 'view_test_nothing', 'select * from {{ ref("view_model") }} limit 0', refs=[view_model])

@pytest.fixture
def unit_test_table_model(table_model: Union[dict[str, dict[str, typing.Any]], allennlp.models.model.Model, typing.Callable]):
    return make_unit_test('pkg', 'unit_test_table_model', table_model)

@pytest.fixture
def namespaced_seed() -> Union[str, typing.IO]:
    return make_seed('pkg', 'mynamespace.seed')

@pytest.fixture
def namespace_model(source: Union[str, bytes]) -> Union[str, human_activities.model.DirectoryViews, None]:
    return make_model('pkg', 'mynamespace.ephemeral_model', 'select * from {{ source("raw", "seed") }}', config_kwargs={'materialized': 'ephemeral'}, sources=[source])

@pytest.fixture
def namespaced_union_model(seed: Union[str, int, typing.IO], ext_source: Union[str, int, typing.IO]) -> Union[str, rflx.model.Model]:
    return make_model('pkg', 'mynamespace.union_model', 'select * from {{ ref("mynamespace.seed") }} union all select * from {{ ref("mynamespace.ephemeral_model") }}', config_kwargs={'materialized': 'table'}, refs=[seed], sources=[ext_source], fqn_extras=['unions'], path='subdirectory/union_model.sql', tags=['unions'])

@pytest.fixture
def metric() -> Metric:
    return Metric(name='my_metric', resource_type=NodeType.Metric, type=MetricType.SIMPLE, type_params=MetricTypeParams(measure=MetricInputMeasure(name='a_measure')), fqn=['test', 'metrics', 'myq_metric'], unique_id='metric.test.my_metric', package_name='test', path='models/metric.yml', original_file_path='models/metric.yml', description='', meta={}, tags=[], label='test_label')

@pytest.fixture
def saved_query() -> SavedQuery:
    pkg = 'test'
    name = 'test_saved_query'
    path = 'test_path'
    return SavedQuery(name=name, resource_type=NodeType.SavedQuery, package_name=pkg, path=path, description='Test Saved Query', query_params=QueryParams(metrics=['my_metric'], group_by=[], where=WhereFilterIntersection(where_filters=[WhereFilter(where_sql_template='1=1')])), exports=[], unique_id=f'saved_query.{pkg}.{name}', original_file_path=path, fqn=[pkg, 'saved_queries', name])

@pytest.fixture
def semantic_model(table_model: Union[str, list[dbcontracts.graph.compiled.InjectedCTE], allennlp.models.model.Model]):
    return make_semantic_model('test', 'test_semantic_model', model=table_model)

@pytest.fixture
def metricflow_time_spine_model() -> ModelNode:
    return ModelNode(name='metricflow_time_spine', database='dbt', schema='analytics', alias='events', resource_type=NodeType.Model, unique_id='model.test.metricflow_time_spine', fqn=['snowplow', 'events'], package_name='snowplow', refs=[], sources=[], metrics=[], depends_on=DependsOn(), config=ModelConfig(), tags=[], path='events.sql', original_file_path='events.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty(), relation_name='events')

@pytest.fixture
def nodes(seed: Union[str, bool], ephemeral_model: Union[str, bool], view_model: Union[str, bool], table_model: Union[str, bool], table_model_py: Union[str, bool], table_model_csv: Union[str, bool], union_model: Union[str, bool], versioned_model_v1: Union[str, bool], versioned_model_v2: Union[str, bool], versioned_model_v3: Union[str, bool], versioned_model_v4_nested_dir: Union[str, bool], versioned_model_v12_string: Union[str, bool], ext_model: Union[str, bool], table_id_unique: Union[str, bool], table_id_not_null: Union[str, bool], view_id_unique: Union[str, bool], ext_source_id_unique: Union[str, bool], view_test_nothing: Union[str, bool], namespaced_seed: Union[str, bool], namespace_model: Union[str, bool], namespaced_union_model: Union[str, bool]) -> list[typing.Union[str,bool]]:
    return [seed, ephemeral_model, view_model, table_model, table_model_py, table_model_csv, union_model, versioned_model_v1, versioned_model_v2, versioned_model_v3, versioned_model_v4_nested_dir, versioned_model_v12_string, ext_model, table_id_unique, table_id_not_null, view_id_unique, ext_source_id_unique, view_test_nothing, namespaced_seed, namespace_model, namespaced_union_model]

@pytest.fixture
def sources(source: Union[str, bool], ext_source: Union[str, bool], ext_source_2: Union[str, bool], ext_source_other: Union[str, bool], ext_source_other_2: Union[str, bool]) -> list[typing.Union[str,bool]]:
    return [source, ext_source, ext_source_2, ext_source_other, ext_source_other_2]

@pytest.fixture
def macros(macro_test_unique: Union[bool, str, typing.Iterable["Entity"]], macro_default_test_unique: Union[bool, str, typing.Iterable["Entity"]], macro_test_not_null: Union[bool, str, typing.Iterable["Entity"]], macro_default_test_not_null: Union[bool, str, typing.Iterable["Entity"]], macro_materialization_table_default: Union[bool, str, typing.Iterable["Entity"]]) -> list[typing.Union[bool,str,typing.Iterable["Entity"]]]:
    return [macro_test_unique, macro_default_test_unique, macro_test_not_null, macro_default_test_not_null, macro_materialization_table_default]

@pytest.fixture
def unit_tests(unit_test_table_model: Union[str, None, bool]) -> list[typing.Union[str,None,bool]]:
    return [unit_test_table_model]

@pytest.fixture
def metrics(metric: Union[float, typing.Iterable[float]]) -> list[typing.Union[float,typing.Iterable[float]]]:
    return [metric]

@pytest.fixture
def semantic_models(semantic_model: typing.AbstractSet) -> list[typing.AbstractSet]:
    return [semantic_model]

@pytest.fixture
def saved_queries(saved_query: Union[bool, list[str], None]) -> list[typing.Union[bool,list[str],None]]:
    return [saved_query]

@pytest.fixture
def files() -> dict:
    return {}

def make_manifest(disabled: dict={}, docs: list=[], exposures: list=[], files: dict={}, groups: list=[], macros: list=[], metrics: list=[], nodes: list=[], saved_queries: list=[], selectors: dict={}, semantic_models: list=[], sources: list=[], unit_tests: list=[]) -> Manifest:
    manifest = Manifest(nodes={n.unique_id: n for n in nodes}, sources={s.unique_id: s for s in sources}, macros={m.unique_id: m for m in macros}, unit_tests={t.unique_id: t for t in unit_tests}, semantic_models={s.unique_id: s for s in semantic_models}, docs={d.unique_id: d for d in docs}, files=files, exposures={e.unique_id: e for e in exposures}, metrics={m.unique_id: m for m in metrics}, disabled=disabled, selectors=selectors, groups={g.unique_id: g for g in groups}, metadata=ManifestMetadata(adapter_type='postgres', project_name='pkg'), saved_queries={s.unique_id: s for s in saved_queries})
    manifest.build_parent_and_child_maps()
    return manifest

@pytest.fixture
def manifest(metric: Union[bool, str, None, list], semantic_model: Union[bool, str, None, list], nodes: Union[dict[str, typing.Any], dict, dbcontracts.graph.manifesManifest], sources: Union[dict[str, typing.Any], dict, dbcontracts.graph.manifesManifest], macros: Union[dict[str, typing.Any], dict, dbcontracts.graph.manifesManifest], unit_tests: Union[dict[str, typing.Any], dict, dbcontracts.graph.manifesManifest], metrics: Union[dict[str, typing.Any], dict, dbcontracts.graph.manifesManifest], semantic_models: Union[dict[str, typing.Any], dict, dbcontracts.graph.manifesManifest], files: Union[dict[str, typing.Any], dict, dbcontracts.graph.manifesManifest], saved_queries: Union[dict[str, typing.Any], dict, dbcontracts.graph.manifesManifest]) -> Union[str, dbcontracts.graph.manifesManifest]:
    return make_manifest(nodes=nodes, sources=sources, macros=macros, unit_tests=unit_tests, semantic_models=semantic_models, files=files, metrics=metrics, saved_queries=saved_queries)