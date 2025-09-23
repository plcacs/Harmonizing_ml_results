from typing import Any, Dict, List, Optional, Union
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

def make_model(pkg: str, name: str, code: str, language: str = 'sql', refs: Optional[List[Union[ModelNode, SeedNode]]] = None, sources: Optional[List[SourceDefinition]] = None, tags: Optional[List[str]] = None, path: Optional[str] = None, alias: Optional[str] = None, config_kwargs: Optional[Dict[str, Any]] = None, fqn_extras: Optional[List[str]] = None, depends_on_macros: Optional[List[str]] = None, version: Optional[Union[int, str]] = None, latest_version: Optional[Union[int, str]] = None, access: Optional[AccessType] = None, patch_path: Optional[str] = None) -> ModelNode:
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
    fqn: List[str] = [pkg] + fqn_extras + [name]
    if version:
        fqn.append(f'v{version}')
    depends_on_nodes: List[str] = []
    source_values: List[List[str]] = []
    ref_values: List[RefArgs] = []
    for ref in refs:
        ref_version = ref.version if hasattr(ref, 'version') else None
        ref_values.append(RefArgs(name=ref.name, package=ref.package_name, version=ref_version))
        depends_on_nodes.append(ref.unique_id)
    for src in sources:
        source_values.append([src.source_name, src.name])
        depends_on_nodes.append(src.unique_id)
    return ModelNode(language='sql', raw_code=code, database='dbt', schema='dbt_schema', alias=alias, name=name, fqn=fqn, unique_id=f'model.{pkg}.{name}' if not version else f'model.{pkg}.{name}.v{version}', package_name=pkg, path=path, original_file_path=f'models/{path}', config=NodeConfig(**config_kwargs), tags=tags, refs=ref_values, sources=source_values, depends_on=DependsOn(nodes=depends_on_nodes, macros=depends_on_macros), resource_type=NodeType.Model, checksum=FileHash.from_contents(''), version=version, latest_version=latest_version, access=access or AccessType.Protected, patch_path=patch_path)

def make_seed(pkg: str, name: str, path: Optional[str] = None, loader: Optional[str] = None, alias: Optional[str] = None, tags: Optional[List[str]] = None, fqn_extras: Optional[List[str]] = None, checksum: Optional[FileHash] = None) -> SeedNode:
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
    fqn: List[str] = [pkg] + fqn_extras + [name]
    return SeedNode(database='dbt', schema='dbt_schema', alias=alias, name=name, fqn=fqn, unique_id=f'seed.{pkg}.{name}', package_name=pkg, path=path, original_file_path=f'data/{path}', tags=tags, resource_type=NodeType.Seed, checksum=FileHash.from_contents(''))

def make_source(pkg: str, source_name: str, table_name: str, path: Optional[str] = None, loader: Optional[str] = None, identifier: Optional[str] = None, fqn_extras: Optional[List[str]] = None) -> SourceDefinition:
    if path is None:
        path = 'models/schema.yml'
    if loader is None:
        loader = 'my_loader'
    if identifier is None:
        identifier = table_name
    if fqn_extras is None:
        fqn_extras = []
    fqn: List[str] = [pkg] + fqn_extras + [source_name, table_name]
    return SourceDefinition(fqn=fqn, database='dbt', schema='dbt_schema', unique_id=f'source.{pkg}.{source_name}.{table_name}', package_name=pkg, path=path, original_file_path=path, name=table_name, source_name=source_name, loader='my_loader', identifier=identifier, resource_type=NodeType.Source, loaded_at_field='loaded_at', tags=[], source_description='')

def make_macro(pkg: str, name: str, macro_sql: str, path: Optional[str] = None, depends_on_macros: Optional[List[str]] = None) -> Macro:
    if path is None:
        path = 'macros/macros.sql'
    if depends_on_macros is None:
        depends_on_macros = []
    return Macro(name=name, macro_sql=macro_sql, unique_id=f'macro.{pkg}.{name}', package_name=pkg, path=path, original_file_path=path, resource_type=NodeType.Macro, depends_on=MacroDependsOn(macros=depends_on_macros))

def make_unique_test(pkg: str, test_model: Union[ModelNode, SourceDefinition], column_name: str, path: Optional[str] = None, refs: Optional[List[Union[ModelNode, SeedNode]]] = None, sources: Optional[List[SourceDefinition]] = None, tags: Optional[List[str]] = None) -> GenericTestNode:
    return make_generic_test(pkg, 'unique', test_model, {}, column_name=column_name)

def make_not_null_test(pkg: str, test_model: Union[ModelNode, SourceDefinition], column_name: str, path: Optional[str] = None, refs: Optional[List[Union[ModelNode, SeedNode]]] = None, sources: Optional[List[SourceDefinition]] = None, tags: Optional[List[str]] = None) -> GenericTestNode:
    return make_generic_test(pkg, 'not_null', test_model, {}, column_name=column_name)

def make_generic_test(pkg: str, test_name: str, test_model: Union[ModelNode, SourceDefinition], test_kwargs: Dict[str, Any], path: Optional[str] = None, refs: Optional[List[Union[ModelNode, SeedNode]]] = None, sources: Optional[List[SourceDefinition]] = None, tags: Optional[List[str]] = None, column_name: Optional[str] = None) -> GenericTestNode:
    kwargs: Dict[str, Any] = test_kwargs.copy()
    ref_values: List[RefArgs] = []
    source_values: List[List[str]] = []
    if isinstance(test_model, SourceDefinition):
        kwargs['model'] = "{{ source('" + test_model.source_name + "', '" + test_model.name + "') }}"
        source_values.append([test_model.source_name, test_model.name])
    else:
        kwargs['model'] = "{{ ref('" + test_model.name + "')}}"
        ref_values.append(RefArgs(name=test_model.name, package=test_model.package_name, version=test_model.version))
    if column_name is not None:
        kwargs['column_name'] = column_name
    args_name: str = test_model.search_name.replace('.', '_')
    if column_name is not None:
        args_name += '_' + column_name
    node_name: str = f'{test_name}_{args_name}'
    raw_code: str = '{{ config(severity="ERROR") }}{{ test_' + test_name + '(**dbt_schema_test_kwargs) }}'
    name_parts: List[str] = test_name.split('.')
    if len(name_parts) == 2:
        (namespace, test_name) = name_parts
        macro_depends: str = f'macro.{namespace}.test_{test_name}'
    elif len(name_parts) == 1:
        namespace = None
        macro_depends: str = f'macro.dbt.test_{test_name}'
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
    depends_on_nodes: List[str] = []
    for ref in refs:
        ref_version = ref.version if hasattr(ref, 'version') else None
        ref_values.append(RefArgs(name=ref.name, package=ref.package_name, version=ref_version))
        depends_on_nodes.append(ref.unique_id)
    for source in sources:
        source_values.append([source.source_name, source.name])
        depends_on_nodes.append(source.unique_id)
    return GenericTestNode(language='sql', raw_code=raw_code, test_metadata=TestMetadata(namespace=namespace, name=test_name, kwargs=kwargs), database='dbt', schema='dbt_postgres', name=node_name, alias=node_name, fqn=['minimal', 'schema_test', node_name], unique_id=f'test.{pkg}.{node_name}', package_name=pkg, path=f'schema_test/{node_name}.sql', original_file_path=f'models/{path}', resource_type=NodeType.Test, tags=tags, refs=ref_values, sources=[], depends_on=DependsOn(macros=[macro_depends], nodes=depends_on_nodes), column_name=column_name, checksum=FileHash.from_contents(''))

def make_unit_test(pkg: str, test_name: str, test_model: ModelNode) -> UnitTestDefinition:
    input_fixture: UnitTestInputFixture = UnitTestInputFixture(input="ref('table_model')", rows=[{'id': 1, 'string_a': 'a'}])
    output_fixture: UnitTestOutputFixture = UnitTestOutputFixture(rows=[{'id': 1, 'string_a': 'a'}])
    return UnitTestDefinition(name=test_name, model=test_model, package_name=pkg, resource_type=NodeType.Unit, path='unit_tests.yml', original_file_path='models/unit_tests.yml', unique_id=f'unit.{pkg}.{test_model.name}__{test_name}', given=[input_fixture], expect=output_fixture, fqn=[pkg, test_model.name, test_name])

def make_singular_test(pkg: str, name: str, sql: str, refs: Optional[List[Union[ModelNode, SeedNode]]] = None, sources: Optional[List[SourceDefinition]] = None, tags: Optional[List[str]] = None, path: Optional[str] = None, config_kwargs: Optional[Dict[str, Any]] = None) -> SingularTestNode:
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
    fqn: List[str] = ['minimal', 'data_test', name]
    depends_on_nodes: List[str] = []
    source_values: List[List[str]] = []
    ref_values: List[RefArgs] = []
    for ref in refs:
        ref_version = ref.version if hasattr(ref, 'version') else None
        ref_values.append(RefArgs(name=ref.name, package=ref.package_name, version=ref_version))
        depends_on_nodes.append(ref.unique_id)
    for src in sources:
        source_values.append([src.source_name, src.name])
        depends_on_nodes.append(src.unique_id)
    return SingularTestNode(language='sql', raw_code=sql, database='dbt', schema='dbt_schema', name=name, alias=name, fqn=fqn, unique_id=f'test.{pkg}.{name}', package_name=pkg, path=path, original_file_path=f'tests/{path}', config=TestConfig(**config_kwargs), tags=tags, refs=ref_values, sources=source_values, depends_on=DependsOn(nodes=depends_on_nodes, macros=[]), resource_type=NodeType.Test, checksum=FileHash.from_contents(''))

def make_exposure(pkg: str, name: str, path: Optional[str] = None, fqn_extras: Optional[List[str]] = None, owner: Optional[Owner] = None) -> Exposure:
    if path is None:
        path = 'schema.yml'
    if fqn_extras is None:
        fqn_extras = []
    if owner is None:
        owner = Owner(email='test@example.com')
    fqn: List[str] = [pkg, 'exposures'] + fqn_extras + [name]
    return Exposure(name=name, resource_type=NodeType.Exposure, type=ExposureType.Notebook, fqn=fqn, unique_id=f'exposure.{pkg}.{name}', package_name=pkg, path=path, original_file_path=path, owner=owner)

def make_metric(pkg: str, name: str, path: Optional[str] = None) -> Metric:
    if path is None:
        path = 'schema.yml'
    return Metric(name=name, resource_type=NodeType.Metric, path=path, package_name=pkg, original_file_path=path, unique_id=f'metric.{pkg}.{name}', fqn=[pkg, 'metrics', name], label='New Customers', description='New customers', type=MetricType.SIMPLE, type_params=MetricTypeParams(measure=MetricInputMeasure(name='count_cats')), meta={'is_okr': True}, tags=['okrs'])

def make_group(pkg: str, name: str, path: Optional[str] = None) -> Group:
    if path is None:
        path = 'schema.yml'
    return Group(name=name, resource_type=NodeType.Group, path=path, package_name=pkg, original_file_path=path, unique_id=f'group.{pkg}.{name}', owner='email@gmail.com')

def make_semantic_model(pkg: str, name: str, model: ModelNode, path: Optional[str] = None) -> SemanticModel:
    if path is None:
        path = 'schema.yml'
    return SemanticModel(name=name, resource_type=NodeType.SemanticModel, model=model, node_relation=NodeRelation(alias=model.alias, schema_name='dbt', relation_name=model.name), package_name=pkg, path=path, description='Customer entity', primary_entity='customer', unique_id=f'semantic_model.{pkg}.{name}', original_file_path=path, fqn=[pkg, 'semantic_models', name])

def make_saved_query(pkg: str, name: str, metric: Union[str, Metric], path: Optional[str] = None) -> SavedQuery:
    if path is None:
        path = 'schema.yml'
    return SavedQuery(name=name, resource_type=NodeType.SavedQuery, package_name=pkg, path=path, description='Test Saved Query', query_params=QueryParams(metrics=[metric], group_by=[], where=None), exports=[], unique_id=f'saved_query.{pkg}.{name}', original_file_path=path, fqn=[pkg, 'saved_queries', name])

@pytest.fixture
def macro_test_unique() -> Macro:
    return make_macro('dbt', 'test_unique', 'blablabla', depends_on_macros=['macro.dbt.default__test_unique'])

@pytest.fixture
def macro_default_test_unique() -> Macro:
    return make_macro('dbt', 'default__test_unique', 'blablabla')

@pytest.fixture
def macro_test_not_null() -> Macro:
    return make_macro('dbt', 'test_not_null', 'blablabla', depends_on_macros=['macro.dbt.default__test_not_null'])

@pytest.fixture
def macro_materialization_table_default() -> Macro:
    macro: Macro = make_macro('dbt', 'materialization_table_default', 'SELECT 1')
    macro.supported_languages = [ModelLanguage.sql]
    return macro

@pytest.fixture
def macro_default_test_not_null() -> Macro:
    return make_macro('dbt', 'default__test_not_null', 'blabla')

@pytest.fixture
def seed() -> SeedNode:
    return make_seed('pkg', 'seed')

@pytest.fixture
def source() -> SourceDefinition:
    return make_source('pkg', 'raw', 'seed', identifier='seed')

@pytest.fixture
def ephemeral_model(source: SourceDefinition) -> ModelNode:
    return make_model('pkg', 'ephemeral_model', 'select * from {{ source("raw", "seed") }}', config_kwargs={'materialized': 'ephemeral'}, sources=[source])

@pytest.fixture
def view_model(ephemer