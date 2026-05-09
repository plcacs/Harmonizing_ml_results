import os
import unittest
from argparse import Namespace
from collections import namedtuple
from copy import deepcopy
from datetime import datetime
from itertools import product
from unittest import mock
import freezegun
import pytest
import dbt.flags
import dbt.version
import dbt_common.invocation
from dbt import tracking
from dbt.adapters.base.plugin import AdapterPlugin
from dbt.artifacts.resources import ExposureType, MaturityType, MetricInputMeasure, MetricTypeParams, Owner, RefArgs, WhereFilter, WhereFilterIntersection
from dbt.contracts.files import FileHash
from dbt.contracts.graph.manifest import DisabledLookup, Manifest, ManifestMetadata
from dbt.contracts.graph.nodes import DependsOn, Exposure, Group, Metric, ModelConfig, ModelNode, SeedNode, SourceDefinition
from dbt.exceptions import AmbiguousResourceNameRefError, ParsingError
from dbt.flags import set_from_args
from dbt.node_types import NodeType
from dbt_common.events.functions import reset_metadata_vars
from dbt_semantic_interfaces.type_enums import MetricType
from tests.unit.utils import MockDocumentation, MockGenerateMacro, MockMacro, MockMaterialization, MockNode, MockSource, inject_plugin, make_manifest
REQUIRED_PARSED_NODE_KEYS: frozenset[str] = frozenset({'alias', 'tags', 'config', 'unique_id', 'refs', 'sources', 'metrics', 'meta', 'depends_on', 'database', 'schema', 'name', 'resource_type', 'group', 'package_name', 'path', 'original_file_path', 'raw_code', 'language', 'description', 'primary_key', 'columns', 'fqn', 'build_path', 'compiled_path', 'patch_path', 'docs', 'doc_blocks', 'checksum', 'unrendered_config', 'unrendered_config_call_dict', 'created_at', 'config_call_dict', 'relation_name', 'contract', 'access', 'version', 'latest_version', 'constraints', 'deprecation_date', 'defer_relation', 'time_spine', 'batch', 'freshness'})
REQUIRED_COMPILED_NODE_KEYS: frozenset[str] = frozenset(REQUIRED_PARSED_NODE_KEYS | {'compiled', 'extra_ctes_injected', 'extra_ctes', 'compiled_code', 'relation_name'})
ENV_KEY_NAME: str = 'KEY' if os.name == 'nt' else 'key'

class ManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_metadata_vars()
        tracking.active_user = None
        self.maxDiff = None
        self.model_config: ModelConfig = ModelConfig.from_dict({'enabled': True, 'materialized': 'view', 'persist_docs': {}, 'post-hook': [], 'pre-hook': [], 'vars': {}, 'quoting': {}, 'column_types': {}, 'tags': []})
        self.exposures: dict[str, Exposure] = {'exposure.root.my_exposure': Exposure(name='my_exposure', type=ExposureType.Dashboard, owner=Owner(email='some@email.com'), resource_type=NodeType.Exposure, description='Test description', maturity=MaturityType.High, url='hhtp://mydashboard.com', depends_on=DependsOn(nodes=['model.root.multi']), refs=[RefArgs(name='multi')], sources=[], fqn=['root', 'my_exposure'], unique_id='exposure.root.my_exposure', package_name='root', path='my_exposure.sql', original_file_path='my_exposure.sql')}
        self.metrics: dict[str, Metric] = {'metric.root.my_metric': Metric(name='new_customers', label='New Customers', description='New customers', meta={'is_okr': True}, tags=['okrs'], type=MetricType.SIMPLE, type_params=MetricTypeParams(measure=MetricInputMeasure(name='customers', filter=WhereFilterIntersection([WhereFilter(where_sql_template='is_new = True')]))), resource_type=NodeType.Metric, depends_on=DependsOn(nodes=['semantic_model.root.customers']), refs=[RefArgs(name='customers')], fqn=['root', 'my_metric'], unique_id='metric.root.my_metric', package_name='root', path='my_metric.yml', original_file_path='my_metric.yml')}
        self.groups: dict[str, Group] = {'group.root.my_group': Group(name='my_group', owner=Owner(email='some@email.com'), resource_type=NodeType.Group, unique_id='group.root.my_group', package_name='root', path='my_metric.yml', original_file_path='my_metric.yml')}
        self.nested_nodes: dict[str, ModelNode] = {'model.snowplow.events': ModelNode(name='events', database='dbt', schema='analytics', alias='events', resource_type=NodeType.Model, unique_id='model.snowplow.events', fqn=['snowplow', 'events'], package_name='snowplow', refs=[], sources=[], metrics=[], depends_on=DependsOn(), config=self.model_config, tags=[], path='events.sql', original_file_path='events.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty()), 'model.root.events': ModelNode(name='events', database='dbt', schema='analytics', alias='events', resource_type=NodeType.Model, unique_id='model.root.events', fqn=['root', 'events'], package_name='root', refs=[], sources=[], metrics=[], depends_on=DependsOn(), config=self.model_config, tags=[], path='events.sql', original_file_path='events.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty()), 'model.root.dep': ModelNode(name='dep', database='dbt', schema='analytics', alias='dep', resource_type=NodeType.Model, unique_id='model.root.dep', fqn=['root', 'dep'], package_name='root', refs=[RefArgs(name='events')], sources=[], metrics=[], depends_on=DependsOn(nodes=['model.root.events']), config=self.model_config, tags=[], path='multi.sql', original_file_path='multi.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty()), 'model.root.nested': ModelNode(name='nested', database='dbt', schema='analytics', alias='nested', resource_type=NodeType.Model, unique_id='model.root.nested', fqn=['root', 'nested'], package_name='root', refs=[RefArgs(name='events')], sources=[], metrics=[], depends_on=DependsOn(nodes=['model.root.dep']), config=self.model_config, tags=[], path='multi.sql', original_file_path='multi.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty()), 'model.root.sibling': ModelNode(name='sibling', database='dbt', schema='analytics', alias='sibling', resource_type=NodeType.Model, unique_id='model.root.sibling', fqn=['root', 'sibling'], package_name='root', refs=[RefArgs(name='events')], sources=[], metrics=[], depends_on=DependsOn(nodes=['model.root.events']), config=self.model_config, tags=[], path='multi.sql', original_file_path='multi.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty()), 'model.root.multi': ModelNode(name='multi', database='dbt', schema='analytics', alias='multi', resource_type=NodeType.Model, unique_id='model.root.multi', fqn=['root', 'multi'], package_name='root', refs=[RefArgs(name='events')], sources=[], depends_on=DependsOn(nodes=['model.root.nested', 'model.root.sibling']), config=self.model_config, tags=[], path='multi.sql', original_file_path='multi.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty())}
        self.sources: dict[str, SourceDefinition] = {'source.root.my_source.my_table': SourceDefinition(database='raw', schema='analytics', resource_type=NodeType.Source, identifier='some_source', name='my_table', source_name='my_source', source_description='My source description', description='Table description', loader='a_loader', unique_id='source.test.my_source.my_table', fqn=['test', 'my_source', 'my_table'], package_name='root', path='schema.yml', original_file_path='schema.yml')}
        self.semantic_models: dict[str, any] = {}
        self.saved_queries: dict[str, any] = {}
        for exposure in self.exposures.values():
            exposure.validate(exposure.to_dict(omit_none=True))
        for metric in self.metrics.values():
            metric.validate(metric.to_dict(omit_none=True))
        for node in self.nested_nodes.values():
            node.validate(node.to_dict(omit_none=True))
        for source in self.sources.values():
            source.validate(source.to_dict(omit_none=True))
        os.environ['DBT_ENV_CUSTOM_ENV_key'] = 'value'

    def tearDown(self) -> None:
        del os.environ['DBT_ENV_CUSTOM_ENV_key']
        reset_metadata_vars()

    @mock.patch.object(tracking, 'active_user')
    @freezegun.freeze_time('2018-02-14T09:15:13Z')
    def test_no_nodes(self, mock_user: any) -> None:
        manifest = Manifest(nodes={}, sources={}, macros={}, docs={}, disabled={}, files={}, exposures={}, metrics={}, selectors={}, metadata=ManifestMetadata(generated_at=datetime.utcnow()), semantic_models={}, saved_queries={})
        invocation_id = dbt_common.invocation._INVOCATION_ID
        mock_user.id = 'cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf'
        set_from_args(Namespace(SEND_ANONYMOUS_USAGE_STATS=False), None)
        self.assertEqual(manifest.writable_manifest().to_dict(omit_none=True), {'nodes': {}, 'sources': {}, 'macros': {}, 'exposures': {}, 'metrics': {}, 'groups': {}, 'selectors': {}, 'parent_map': {}, 'child_map': {}, 'group_map': {}, 'metadata': {'generated_at': '2018-02-14T09:15:13Z', 'dbt_schema_version': 'https://schemas.getdbt.com/dbt/manifest/v12.json', 'dbt_version': dbt.version.__version__, 'env': {ENV_KEY_NAME: 'value'}, 'invocation_id': invocation_id, 'send_anonymous_usage_stats': False, 'user_id': 'cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf'}, 'docs': {}, 'disabled': {}, 'semantic_models': {}, 'unit_tests': {}, 'saved_queries': {}})

    @freezegun.freeze_time('2018-02-14T09:15:13Z')
    @mock.patch.object(tracking, 'active_user')
    def test_nested_nodes(self, mock_user: any) -> None:
        set_from_args(Namespace(SEND_ANONYMOUS_USAGE_STATS=False), None)
        mock_user.id = 'cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf'
        nodes = deepcopy(self.nested_nodes)
        manifest = Manifest(nodes=nodes, sources={}, macros={}, docs={}, disabled={}, files={}, exposures={}, metrics={}, selectors={}, metadata=ManifestMetadata(generated_at=datetime.utcnow()))
        serialized = manifest.writable_manifest().to_dict(omit_none=True)
        self.assertEqual(serialized['metadata']['generated_at'], '2018-02-14T09:15:13Z')
        self.assertEqual(serialized['metadata']['user_id'], mock_user.id)
        self.assertFalse(serialized['metadata']['send_anonymous_usage_stats'])
        self.assertEqual(serialized['docs'], {})
        self.assertEqual(serialized['disabled'], {})
        parent_map = serialized['parent_map']
        child_map = serialized['child_map']
        self.assertEqual(set(parent_map), set(nodes))
        self.assertEqual(set(child_map), set(nodes))
        self.assertEqual(parent_map['model.root.sibling'], ['model.root.events'])
        self.assertEqual(parent_map['model.root.nested'], ['model.root.dep'])
        self.assertEqual(parent_map['model.root.dep'], ['model.root.events'])
        self.assertEqual(set(parent_map['model.root.multi']), set(['model.root.nested', 'model.root.sibling']))
        self.assertEqual(parent_map['model.root.events'], [])
        self.assertEqual(parent_map['model.snowplow.events'], [])
        self.assertEqual(child_map['model.root.sibling'], ['model.root.multi'])
        self.assertEqual(child_map['model.root.nested'], ['model.root.multi'])
        self.assertEqual(child_map['model.root.dep'], ['model.root.nested'])
        self.assertEqual(child_map['model.root.multi'], [])
        self.assertEqual(set(child_map['model.root.events']), set(['model.root.dep', 'model.root.sibling']))
        self.assertEqual(child_map['model.snowplow.events'], [])

    def test_build_flat_graph(self) -> None:
        exposures = deepcopy(self.exposures)
        metrics = deepcopy(self.metrics)
        groups = deepcopy(self.groups)
        nodes = deepcopy(self.nested_nodes)
        sources = deepcopy(self.sources)
        manifest = Manifest(nodes=nodes, sources=sources, macros={}, docs={}, disabled={}, files={}, exposures=exposures, metrics=metrics, groups=groups, selectors={})
        manifest.build_flat_graph()
        flat_graph = manifest.flat_graph
        flat_exposures = flat_graph['exposures']
        flat_groups = flat_graph['groups']
        flat_metrics = flat_graph['metrics']
        flat_nodes = flat_graph['nodes']
        flat_sources = flat_graph['sources']
        flat_semantic_models = flat_graph['semantic_models']
        flat_saved_queries = flat_graph['saved_queries']
        self.assertEqual(set(flat_graph), set(['exposures', 'groups', 'nodes', 'sources', 'metrics', 'semantic_models', 'saved_queries']))
        self.assertEqual(set(flat_exposures), set(self.exposures))
        self.assertEqual(set(flat_groups), set(self.groups))
        self.assertEqual(set(flat_metrics), set(self.metrics))
        self.assertEqual(set(flat_nodes), set(self.nested_nodes))
        self.assertEqual(set(flat_sources), set(self.sources))
        self.assertEqual(set(flat_semantic_models), set(self.semantic_models))
        self.assertEqual(set(flat_saved_queries), set(self.saved_queries))
        for node in flat_nodes.values():
            self.assertEqual(frozenset(node), REQUIRED_PARSED_NODE_KEYS)

    @mock.patch.object(tracking, 'active_user')
    @freezegun.freeze_time('2018-02-14T09:15:13Z')
    def test_no_nodes_with_metadata(self, mock_user: any) -> None:
        mock_user.id = 'cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf'
        dbt_common.invocation._INVOCATION_ID = '01234567-0123-0123-0123-0123456789ab'
        set_from_args(Namespace(SEND_ANONYMOUS_USAGE_STATS=False), None)
        metadata = ManifestMetadata(project_id='098f6bcd4621d373cade4e832627b4f6', adapter_type='postgres', generated_at=datetime.utcnow(), user_id='cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf', send_anonymous_usage_stats=False)
        manifest = Manifest(nodes={}, sources={}, macros={}, docs={}, disabled={}, selectors={}, metadata=metadata, files={}, exposures={}, semantic_models={}, saved_queries={})
        self.assertEqual(manifest.writable_manifest().to_dict(omit_none=True), {'nodes': {}, 'sources': {}, 'macros': {}, 'exposures': {}, 'metrics': {}, 'groups': {}, 'selectors': {}, 'parent_map': {}, 'child_map': {}, 'group_map': {}, 'docs': {}, 'metadata': {'generated_at': '2018-02-14T09:15:13Z', 'dbt_schema_version': 'https://schemas.getdbt.com/dbt/manifest/v12.json', 'dbt_version': dbt.version.__version__, 'project_id': '098f6bcd4621d373cade4e832627b4f6', 'user_id': 'cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf', 'send_anonymous_usage_stats': False, 'adapter_type': 'postgres', 'invocation_id': '01234567-0123-0123-0123-0123456789ab', 'env': {ENV_KEY_NAME: 'value'}}, 'disabled': {}, 'semantic_models': {}, 'unit_tests': {}, 'saved_queries': {}})

    def test_get_resource_fqns_empty(self) -> None:
        manifest = Manifest(nodes={}, sources={}, macros={}, docs={}, disabled={}, files={}, exposures={}, selectors={})
        self.assertEqual(manifest.get_resource_fqns(), {})

    def test_get_resource_fqns(self) -> None:
        nodes = deepcopy(self.nested_nodes)
        nodes['seed.root.seed'] = SeedNode(name='seed', database='dbt', schema='analytics', alias='seed', resource_type=NodeType.Seed, unique_id='seed.root.seed', fqn=['root', 'seed'], package_name='root', config=self.model_config, tags=[], path='seed.csv', original_file_path='seed.csv', checksum=FileHash.empty())
        manifest = Manifest(nodes=nodes, sources=self.sources, macros={}, docs={}, disabled={}, files={}, exposures=self.exposures, metrics=self.metrics, selectors={})
        expect = {'metrics': frozenset([('root', 'my_metric')]), 'exposures': frozenset([('root', 'my_exposure')]), 'models': frozenset([('snowplow', 'events'), ('root', 'events'), ('root', 'dep'), ('root', 'nested'), ('root', 'sibling'), ('root', 'multi')]), 'seeds': frozenset([('root', 'seed')]), 'sources': frozenset([('test', 'my_source', 'my_table')])}
        resource_fqns = manifest.get_resource_fqns()
        self.assertEqual(resource_fqns, expect)

    def test_deepcopy_copies_flat_graph(self) -> None:
        test_node = ModelNode(name='events', database='dbt', schema='analytics', alias='events', resource_type=NodeType.Model, unique_id='model.snowplow.events', fqn=['snowplow', 'events'], package_name='snowplow', refs=[], sources=[], metrics=[], depends_on=DependsOn(), config=self.model_config, tags=[], path='events.sql', original_file_path='events.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty())
        original = make_manifest(nodes=[test_node])
        original.build_flat_graph()
        copy = original.deepcopy()
        self.assertEqual(original.flat_graph, copy.flat_graph)

class MixedManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        self.model_config: ModelConfig = ModelConfig.from_dict({'enabled': True, 'materialized': 'view', 'persist_docs': {}, 'post-hook': [], 'pre-hook': [], 'vars': {}, 'quoting': {}, 'column_types': {}, 'tags': []})
        self.nested_nodes: dict[str, ModelNode] = {'model.snowplow.events': ModelNode(name='events', database='dbt', schema='analytics', alias='events', resource_type=NodeType.Model, unique_id='model.snowplow.events', fqn=['snowplow', 'events'], package_name='snowplow', refs=[], sources=[], depends_on=DependsOn(), config=self.model_config, tags=[], path='events.sql', original_file_path='events.sql', language='sql', raw_code='does not matter', meta={}, compiled=True, compiled_code='also does not matter', extra_ctes_injected=True, relation_name='"dbt"."analytics"."events"', extra_ctes=[], checksum=FileHash.empty()), 'model.root.events': ModelNode(name='events', database='dbt', schema='analytics', alias='events', resource_type=NodeType.Model, unique_id='model.root.events', fqn=['root', 'events'], package_name='root', refs=[], sources=[], depends_on=DependsOn(), config=self.model_config, tags=[], path='events.sql', original_file_path='events.sql', raw_code='does not matter', meta={}, compiled=True, compiled_code='also does not matter', language='sql', extra_ctes_injected=True, relation_name='"dbt"."analytics"."events"', extra_ctes=[], checksum=FileHash.empty()), 'model.root.dep': ModelNode(name='dep', database='dbt', schema='analytics', alias='dep', resource_type=NodeType.Model, unique_id='model.root.dep', fqn=['root', 'dep'], package_name='root', refs=[RefArgs(name='events')], sources=[], depends_on=DependsOn(nodes=['model.root.events']), config=self.model_config, tags=[], path='multi.sql', original_file_path='multi.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty()), 'model.root.versioned.v1': ModelNode(name='versioned', database='dbt', schema='analytics', alias='dep', resource_type=NodeType.Model, unique_id='model.root.versioned.v1', fqn=['root', 'dep'], package_name='root', refs=[], sources=[], depends_on=DependsOn(), config=self.model_config, tags=[], path='versioned.sql', original_file_path='versioned.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty(), version=1), 'model.root.dep_version': ModelNode(name='dep_version', database='dbt', schema='analytics', alias='dep', resource_type=NodeType.Model, unique_id='model.root.dep_version', fqn=['root', 'dep'], package_name='root', refs=[RefArgs(name='versioned', version=1)], sources=[], depends_on=DependsOn(nodes=['model.root.versioned.v1']), config=self.model_config, tags=[], path='dep_version.sql', original_file_path='dep_version.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty()), 'model.root.nested': ModelNode(name='nested', database='dbt', schema='analytics', alias='nested', resource_type=NodeType.Model, unique_id='model.root.nested', fqn=['root', 'nested'], package_name='root', refs=[RefArgs(name='events')], sources=[], depends_on=DependsOn(nodes=['model.root.dep']), config=self.model_config, tags=[], path='multi.sql', original_file_path='multi.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty()), 'model.root.sibling': ModelNode(name='sibling', database='dbt', schema='analytics', alias='sibling', resource_type=NodeType.Model, unique_id='model.root.sibling', fqn=['root', 'sibling'], package_name='root', refs=[RefArgs(name='events')], sources=[], depends_on=DependsOn(nodes=['model.root.events']), config=self.model_config, tags=[], path='multi.sql', original_file_path='multi.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty()), 'model.root.multi': ModelNode(name='multi', database='dbt', schema='analytics', alias='multi', resource_type=NodeType.Model, unique_id='model.root.multi', fqn=['root', 'multi'], package_name='root', refs=[RefArgs(name='events')], sources=[], depends_on=DependsOn(nodes=['model.root.nested', 'model.root.sibling']), config=self.model_config, tags=[], path='multi.sql', original_file_path='multi.sql', meta={}, language='sql', raw_code='does not matter', checksum=FileHash.empty())}
        os.environ['DBT_ENV_CUSTOM_ENV_key'] = 'value'

    def tearDown(self) -> None:
        del os.environ['DBT_ENV_CUSTOM_ENV_key']

    @mock.patch.object(tracking, 'active_user')
    @freezegun.freeze_time('2018-02-14T09:15:13Z')
    def test_no_nodes(self, mock_user: any) -> None:
        mock_user.id = 'cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf'
        set_from_args(Namespace(SEND_ANONYMOUS_USAGE_STATS=False), None)
        metadata = ManifestMetadata(generated_at=datetime.utcnow(), invocation_id='01234567-0123-0123-0123-0123456789ab')
        manifest = Manifest(nodes={}, sources={}, macros={}, docs={}, selectors={}, disabled={}, metadata=metadata, files={}, exposures={}, semantic_models={}, saved_queries={})
        self.assertEqual(manifest.writable_manifest().to_dict(omit_none=True), {'nodes': {}, 'macros': {}, 'sources': {}, 'exposures': {}, 'metrics': {}, 'groups': {}, 'selectors': {}, 'parent_map': {}, 'child_map': {}, 'group_map': {}, 'metadata': {'generated_at': '2018-02-14T09:15:13Z', 'dbt_schema_version': 'https://schemas.getdbt.com/dbt/manifest/v12.json', 'dbt_version': dbt.version.__version__, 'invocation_id': '01234567-0123-0123-0123-0123456789ab', 'env': {ENV_KEY_NAME: 'value'}, 'send_anonymous_usage_stats': False, 'user_id': 'cfc9500f-dc7f-4c83-9ea7-2c581c1b38cf'}, 'docs': {}, 'disabled': {}, 'semantic_models': {}, 'unit_tests': {}, 'saved_queries': {}})

    @freezegun.freeze_time('2018-02-14T09:15:13Z')
    def test_nested_nodes(self) -> None:
        nodes = deepcopy(self.nested_nodes)
        manifest = Manifest(nodes=nodes, sources={}, macros={}, docs={}, disabled={}, selectors={}, metadata=ManifestMetadata(generated_at=datetime.utcnow()), files={}, exposures={})
        serialized = manifest.writable_manifest().to_dict(omit_none=True)
        self.assertEqual(serialized['metadata']['generated_at'], '2018-02-14T09:15:13Z')
        self.assertEqual(serialized['disabled'], {})
        parent_map = serialized['parent_map']
        child_map = serialized['child_map']
        self.assertEqual(set(parent_map), set(nodes))
        self.assertEqual(set(child_map), set(nodes))
        self.assertEqual(parent_map['model.root.sibling'], ['model.root.events'])
        self.assertEqual(parent_map['model.root.nested'], ['model.root.dep'])
        self.assertEqual(parent_map['model.root.dep'], ['model.root.events'])
        self.assertEqual(set(parent_map['model.root.multi']), set(['model.root.nested', 'model.root.sibling']))
        self.assertEqual(parent_map['model.root.events'], [])
        self.assertEqual(parent_map['model.snowplow.events'], [])
        self.assertEqual(child_map['model.root.sibling'], ['model.root.multi'])
        self.assertEqual(child_map['model.root.nested'], ['model.root.multi'])
        self.assertEqual(child_map['model.root.dep'], ['model.root.nested'])
        self.assertEqual(child_map['model.root.multi'], [])
        self.assertEqual(set(child_map['model.root.events']), set(['model.root.dep', 'model.root.sibling']))
        self.assertEqual(child_map['model.snowplow.events'], [])

    def test_build_flat_graph(self) -> None:
        nodes = deepcopy(self.nested_nodes)
        manifest = Manifest(nodes=nodes, sources={}, macros={}, docs={}, disabled={}, selectors={}, files={}, exposures={}, semantic_models={}, saved_queries={})
        manifest.build_flat_graph()
        flat_graph = manifest.flat_graph
        flat_nodes = flat_graph['nodes']
        self.assertEqual(set(flat_graph), set(['exposures', 'groups', 'nodes', 'sources', 'semantic_models', 'saved_queries']))
        self.assertEqual(set(flat_nodes), set(self.nested_nodes))
        compiled_count = 0
        for node in flat_nodes.values():
            if node.get('compiled'):
                self.assertEqual(frozenset(node), REQUIRED_COMPILED_NODE_KEYS)
                compiled_count += 1
            else:
                self.assertEqual(frozenset(node), REQUIRED_PARSED_NODE_KEYS)
        self.assertEqual(compiled_count, 2)

    def test_merge_from_artifact(self) -> None:
        original_nodes = deepcopy(self.nested_nodes)
        other_nodes = deepcopy(self.nested_nodes)
        nested2 = other_nodes.pop('model.root.nested')
        nested2.name = 'nested2'
        nested2.alias = 'nested2'
        nested2.fqn = ['root', 'nested2']
        other_nodes['model.root.nested2'] = nested2
        for k, v in other_nodes.items():
            v.database = 'other_' + v.database
            v.schema = 'other_' + v.schema
            v.alias = 'other_' + v.alias
            if v.relation_name:
                v.relation_name = 'other_' + v.relation_name
            other_nodes[k] = v
        original_manifest = Manifest(nodes=original_nodes)
        other_manifest = Manifest(nodes=other_nodes)
        original_manifest.merge_from_artifact(other_manifest)
        assert 'model.root.nested2' not in original_manifest.nodes
        assert original_manifest.nodes['model.root.nested'].defer_relation is None
        for k, v in original_manifest.nodes.items():
            if v.defer_relation:
                self.assertEqual('other_' + v.database, v.defer_relation.database)
                self.assertEqual('other_' + v.schema, v.defer_relation.schema)
                self.assertEqual('other_' + v.alias, v.defer_relation.alias)
                if v.relation_name:
                    self.assertEqual('other_' + v.relation_name, v.defer_relation.relation_name)

class TestManifestSearch(unittest.TestCase):
    _macros: list[MockMacro] = []
    _nodes: list[MockNode] = []
    _docs: list[MockDocumentation] = []

    @property
    def macros(self) -> list[MockMacro]:
        return self._macros

    @property
    def nodes(self) -> list[MockNode]:
        return self._nodes

    @property
    def docs(self) -> list[MockDocumentation]:
        return self._docs

    def setUp(self) -> None:
        self.manifest = Manifest(nodes={n.unique_id: n for n in self.nodes}, macros={m.unique_id: m for m in self.macros}, docs={d.unique_id: d for d in self.docs}, disabled={}, files={}, exposures={}, metrics={}, selectors={})

class TestManifestFindNodeFromRefOrSource(unittest.TestCase):
    @pytest.fixture
    def mock_node(self) -> MockNode:
        return MockNode('my_package', 'my_model')

    @pytest.fixture
    def mock_disabled_node(self) -> MockNode:
        return MockNode('my_package', 'disabled_node', config={'enabled': False})

    @pytest.fixture
    def mock_source(self) -> MockSource:
        return MockSource('root', 'my_source', 'source_table')

    @pytest.fixture
    def mock_disabled_source(self) -> MockSource:
        return MockSource('root', 'my_source', 'disabled_source_table', config={'enabled': False})

    @pytest.fixture
    def mock_manifest(self, mock_node: MockNode, mock_source: MockSource, mock_disabled_node: MockNode, mock_disabled_source: MockSource) -> Manifest:
        return make_manifest(nodes=[mock_node, mock_disabled_node], sources=[mock_source, mock_disabled_source])

    @pytest.mark.parametrize('expression,expected_node', [("ref('my_package', 'my_model')", 'mock_node'), ("ref('my_package', 'doesnt_exist')", None), ("ref('my_package', 'disabled_node')", 'mock_disabled_node'), ("source('my_source', 'source_table')", 'mock_source'), ("source('my_source', 'doesnt_exist')", None), ("source('my_source', 'disabled_source_table')", 'mock_disabled_source')])
    def test_find_node_from_ref_or_source(self, expression: str, expected_node: str, mock_manifest: Manifest, request: unittest.TestCase) -> None:
        node = mock_manifest.find_node_from_ref_or_source(expression)
        if expected_node is None:
            assert node is None
        else:
            assert node == request.getfixturevalue(expected_node)

    @pytest.mark.parametrize('invalid_expression', ['invalid', "ref(')"])
    def test_find_node_from_ref_or_source_invalid_expression(self, invalid_expression: str, mock_manifest: Manifest) -> None:
        with pytest.raises(ParsingError):
            mock_manifest.find_node_from_ref_or_source(invalid_expression)

class TestDisabledLookup(unittest.TestCase):
    @pytest.fixture(scope='class')
    def manifest(self) -> Manifest:
        return Manifest(nodes={}, sources={}, macros={}, docs={}, disabled={}, files={}, exposures={}, metrics={}, selectors={})

    @pytest.fixture(scope='class')
    def mock_model(self) -> MockNode:
        return MockNode('package', 'name', NodeType.Model)

    @pytest.fixture(scope='class')
    def mock_model_with_version(self) -> MockNode:
        return MockNode('package', 'name', NodeType.Model, version=3)

    @pytest.fixture(scope='class')
    def mock_seed(self) -> MockNode:
        return MockNode('package', 'name', NodeType.Seed)

    def test_find(self, manifest: Manifest, mock_model: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'package') == [mock_model]

    def test_find_wrong_name(self, manifest: Manifest, mock_model: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('missing_name', 'package') is None

    def test_find_wrong_package(self, manifest: Manifest, mock_model: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'missing_package') is None

    def test_find_wrong_version(self, manifest: Manifest, mock_model: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'package', version=3) is None

    def test_find_wrong_resource_types(self, manifest: Manifest, mock_model: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'package', resource_types=[NodeType.Analysis]) is None

    def test_find_no_package(self, manifest: Manifest, mock_model: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', None) == [mock_model]

    def test_find_versioned_node(self, manifest: Manifest, mock_model_with_version: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model_with_version]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'package', version=3) == [mock_model_with_version]

    def test_find_versioned_node_no_package(self, manifest: Manifest, mock_model_with_version: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model_with_version]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', None, version=3) == [mock_model_with_version]

    def test_find_versioned_node_no_version(self, manifest: Manifest, mock_model_with_version: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model_with_version]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'package') is None

    def test_find_versioned_node_wrong_version(self, manifest: Manifest, mock_model_with_version: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model_with_version]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'package', version=2) is None

    def test_find_versioned_node_wrong_name(self, manifest: Manifest, mock_model_with_version: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model_with_version]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('wrong_name', 'package', version=3) is None

    def test_find_versioned_node_wrong_package(self, manifest: Manifest, mock_model_with_version: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model_with_version]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'wrong_package', version=3) is None

    def test_find_multiple_nodes(self, manifest: Manifest, mock_model: MockNode, mock_seed: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model, mock_seed]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'package') == [mock_model, mock_seed]

    def test_find_multiple_nodes_with_resource_types(self, manifest: Manifest, mock_model: MockNode, mock_seed: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model, mock_seed]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'package', resource_types=[NodeType.Model]) == [mock_model]

    def test_find_multiple_nodes_with_wrong_resource_types(self, manifest: Manifest, mock_model: MockNode, mock_seed: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model, mock_seed]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'package', resource_types=[NodeType.Analysis]) is None

    def test_find_multiple_nodes_with_resource_types_empty(self, manifest: Manifest, mock_model: MockNode, mock_seed: MockNode) -> None:
        manifest.disabled = {'model.package.name': [mock_model, mock_seed]}
        lookup = DisabledLookup(manifest)
        assert lookup.find('name', 'package', resource_types=[]) is None
