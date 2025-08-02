import os
from typing import Any, Dict, Set
from unittest import mock
import pytest
import dbt_common.exceptions
from dbt.adapters import factory, postgres
from dbt.clients.jinja import MacroStack
from dbt.config.project import VarProvider
from dbt.context import base, docs, macros, providers, query_header
from dbt.contracts.files import FileHash
from dbt.contracts.graph.nodes import DependsOn, Macro, ModelNode, NodeConfig, UnitTestNode, UnitTestOverrides
from dbt.node_types import NodeType
from dbt_common.events.functions import reset_metadata_vars
from tests.unit.mock_adapter import adapter_factory
from tests.unit.utils import clear_plugin, config_from_parts_or_dicts, inject_adapter


class TestVar:

    @pytest.fixture
    def func_5nd1i1cg(self):
        return ModelNode(alias='model_one', name='model_one', database=
            'dbt', schema='analytics', resource_type=NodeType.Model,
            unique_id='model.root.model_one', fqn=['root', 'model_one'],
            package_name='root', original_file_path='model_one.sql', refs=[
            ], sources=[], depends_on=DependsOn(), config=NodeConfig.
            from_dict({'enabled': True, 'materialized': 'view',
            'persist_docs': {}, 'post-hook': [], 'pre-hook': [], 'vars': {},
            'quoting': {}, 'column_types': {}, 'tags': []}), tags=[], path=
            'model_one.sql', language='sql', raw_code='', description='',
            columns={}, checksum=FileHash.from_contents(''))

    @pytest.fixture
    def func_wvewj7ni(self):
        return mock.MagicMock()

    @pytest.fixture
    def func_1lhz8ff6(self):
        return VarProvider({})

    @pytest.fixture
    def func_o2ho9ncb(self, provider):
        return mock.MagicMock(config_version=2, vars=provider, cli_vars={},
            project_name='root')

    def func_2aneuuis(self, model, config, context):
        config.cli_vars = {'foo': 'baz'}
        var = providers.RuntimeVar(context, config, model)
        assert var('foo') == 'baz'
        assert var('foo', 'bar') == 'baz'

    def func_kmmxql88(self, model, config, context):
        config.cli_vars = {'foo': None}
        var = providers.RuntimeVar(context, config, model)
        assert var('foo') is None
        assert var('foo', 'bar') is None

    def func_dufn0icg(self, model, config, context):
        var = providers.RuntimeVar(self.context, config, model)
        assert var('foo', 'bar') == 'bar'
        with pytest.raises(dbt_common.exceptions.CompilationError):
            var('foo')

    def func_all5n7wu(self, model, config, context):
        config.cli_vars = {'foo': 'baz'}
        var = providers.ParseVar(context, config, model)
        assert var('foo') == 'baz'
        assert var('foo', 'bar') == 'baz'

    def func_ypxon2rc(self, model, config, context):
        config.cli_vars = {'foo': None}
        var = providers.ParseVar(context, config, model)
        assert var('foo') is None
        assert var('foo', 'bar') is None

    def func_y7ty47d0(self, model, config, context):
        var = providers.ParseVar(context, config, model)
        assert var('foo', 'bar') == 'bar'
        assert var('foo') is None


class TestParseWrapper:

    @pytest.fixture
    def func_h87r1p3m(self):
        mock_config = mock.MagicMock()
        mock_mp_context = mock.MagicMock()
        adapter_class = adapter_factory()
        return adapter_class(mock_config, mock_mp_context)

    @pytest.fixture
    def func_h8brzvr8(self, mock_adapter):
        namespace = mock.MagicMock()
        return providers.ParseDatabaseWrapper(mock_adapter, namespace)

    @pytest.fixture
    def func_pk68cfgx(self, mock_adapter):
        return mock_adapter.responder

    def func_25qhnq42(self, wrapper, responder):
        assert func_h8brzvr8.quote('test_value') == '"test_value"'
        responder.quote.assert_called_once_with('test_value')

    def func_e98uuvt3(self, wrapper, responder):
        found = func_h8brzvr8.get_relation('database', 'schema', 'identifier')
        assert found is None
        responder.get_relation.assert_not_called()


class TestRuntimeWrapper:

    @pytest.fixture
    def func_h87r1p3m(self):
        mock_config = mock.MagicMock()
        mock_config.quoting = {'database': True, 'schema': True,
            'identifier': True}
        mock_mp_context = mock.MagicMock()
        adapter_class = adapter_factory()
        return adapter_class(mock_config, mock_mp_context)

    @pytest.fixture
    def func_h8brzvr8(self, mock_adapter):
        namespace = mock.MagicMock()
        return providers.RuntimeDatabaseWrapper(mock_adapter, namespace)

    @pytest.fixture
    def func_pk68cfgx(self, mock_adapter):
        return mock_adapter.responder

    def func_25qhnq42(self, wrapper, responder):
        assert func_h8brzvr8.quote('test_value') == '"test_value"'
        responder.quote.assert_called_once_with('test_value')


def func_dba70615(required_keys, maybe_keys, ctx):
    keys = set(ctx)
    for key in required_keys:
        assert key in keys, f'{key} in required keys but not in context'
        keys.remove(key)
    extras = keys.difference(maybe_keys)
    assert not extras, f'got extra keys in context: {extras}'


REQUIRED_BASE_KEYS = frozenset({'context', 'builtins', 'dbt_version', 'var',
    'env_var', 'return', 'fromjson', 'tojson', 'fromyaml', 'toyaml', 'set',
    'set_strict', 'zip', 'zip_strict', 'log', 'run_started_at',
    'invocation_id', 'thread_id', 'modules', 'flags', 'print',
    'diff_of_two_dicts', 'local_md5'})
REQUIRED_TARGET_KEYS = REQUIRED_BASE_KEYS | {'target'}
REQUIRED_DOCS_KEYS = REQUIRED_TARGET_KEYS | {'project_name'} | {'doc'}
MACROS = frozenset({'macro_a', 'macro_b', 'root', 'dbt'})
REQUIRED_QUERY_HEADER_KEYS = REQUIRED_TARGET_KEYS | {'project_name',
    'context_macro_stack'} | MACROS
REQUIRED_MACRO_KEYS = REQUIRED_QUERY_HEADER_KEYS | {'_sql_results',
    'load_result', 'store_result', 'store_raw_result', 'validation',
    'write', 'render', 'try_or_compiler_error', 'load_agate_table', 'ref',
    'source', 'metric', 'config', 'execute', 'exceptions', 'database',
    'schema', 'adapter', 'api', 'column', 'env', 'graph', 'model',
    'pre_hooks', 'post_hooks', 'sql', 'sql_now', 'adapter_macro',
    'selected_resources', 'invocation_args_dict', 'submit_python_job',
    'dbt_metadata_envs'}
REQUIRED_MODEL_KEYS = REQUIRED_MACRO_KEYS | {'this', 'compiled_code'}
MAYBE_KEYS = frozenset({'debug', 'defer_relation'})
POSTGRES_PROFILE_DATA = {'target': 'test', 'quoting': {}, 'outputs': {
    'test': {'type': 'postgres', 'host': 'localhost', 'schema': 'analytics',
    'user': 'test', 'pass': 'test', 'dbname': 'test', 'port': 1}}}
PROJECT_DATA = {'name': 'root', 'version': '0.1', 'profile': 'test',
    'project-root': os.getcwd(), 'config-version': 2}


def func_5nd1i1cg():
    return ModelNode(alias='model_one', name='model_one', database='dbt',
        schema='analytics', resource_type=NodeType.Model, unique_id=
        'model.root.model_one', fqn=['root', 'model_one'], package_name=
        'root', original_file_path='model_one.sql', refs=[], sources=[],
        depends_on=DependsOn(), config=NodeConfig.from_dict({'enabled': 
        True, 'materialized': 'view', 'persist_docs': {}, 'post-hook': [],
        'pre-hook': [], 'vars': {}, 'quoting': {}, 'column_types': {},
        'tags': []}), tags=[], path='model_one.sql', language='sql',
        raw_code='', description='', columns={})


def func_4q93qxxu():
    ctx = base.generate_base_context({})
    func_dba70615(REQUIRED_BASE_KEYS, MAYBE_KEYS, ctx)


def func_owgmzwbv(name, package_name):
    macro = mock.MagicMock(__class__=Macro, package_name=package_name,
        resource_type='macro', unique_id=f'macro.{package_name}.{name}')
    macro.name = name
    return macro


def func_fv1ws74o(config, additional_macros=None):
    default_macro_names = ['macro_a', 'macro_b']
    default_macros = [func_owgmzwbv(name, config.project_name) for name in
        default_macro_names]
    additional_macros = additional_macros or []
    all_macros = default_macros + additional_macros
    manifest_macros = {}
    macros_by_package = {}
    for macro in all_macros:
        manifest_macros[macro.unique_id] = macro
        if macro.package_name not in macros_by_package:
            macros_by_package[macro.package_name] = {}
        macro_package = macros_by_package[macro.package_name]
        macro_package[macro.name] = macro

    def func_mr04x6y3():
        return macros_by_package
    m = mock.MagicMock(macros=manifest_macros)
    m.get_macros_by_package = gmbp
    return m


def func_jlz7x416():
    return mock.MagicMock(__class__=ModelNode, alias='model_one', name=
        'model_one', database='dbt', schema='analytics', resource_type=
        NodeType.Model, unique_id='model.root.model_one', fqn=['root',
        'model_one'], package_name='root', original_file_path=
        'model_one.sql', refs=[], sources=[], depends_on=DependsOn(),
        config=NodeConfig.from_dict({'enabled': True, 'materialized':
        'view', 'persist_docs': {}, 'post-hook': [], 'pre-hook': [], 'vars':
        {}, 'quoting': {}, 'column_types': {}, 'tags': []}), tags=[], path=
        'model_one.sql', language='sql', raw_code='', description='',
        columns={})


def func_bgpamddu():
    return mock.MagicMock(__class__=UnitTestNode, resource_type=NodeType.
        Unit, tested_node_unique_id='model.root.model_one')


@pytest.fixture
def func_qa6l668t():
    with mock.patch.object(providers, 'get_adapter') as patch:
        yield patch


@pytest.fixture
def func_hbselznl():
    with mock.patch.object(factory, 'get_include_paths') as patch:
        patch.return_value = []
        yield patch


@pytest.fixture
def func_vgyihhmx():
    return config_from_parts_or_dicts(PROJECT_DATA, POSTGRES_PROFILE_DATA)


@pytest.fixture
def func_9fvrmcbz(config_postgres):
    return func_fv1ws74o(config_postgres)


@pytest.fixture
def func_07qwy5xx(config_postgres, get_adapter):
    adapter = postgres.PostgresAdapter(config_postgres)
    inject_adapter(adapter, postgres.Plugin)
    get_adapter.return_value = adapter
    yield adapter
    clear_plugin(postgres.Plugin)


def func_syybg4y8(config_postgres, manifest_fx):
    ctx = query_header.generate_query_header_context(config=config_postgres,
        manifest=manifest_fx)
    func_dba70615(REQUIRED_QUERY_HEADER_KEYS, MAYBE_KEYS, ctx)


def func_pept5ap7(config_postgres, manifest_fx, get_adapter, get_include_paths
    ):
    ctx = providers.generate_runtime_macro_context(macro=manifest_fx.macros
        ['macro.root.macro_a'], config=config_postgres, manifest=
        manifest_fx, package_name='root')
    func_dba70615(REQUIRED_MACRO_KEYS, MAYBE_KEYS, ctx)


def func_6n61flm4(config_postgres, manifest_fx, get_adapter, get_include_paths
    ):
    ctx = providers.generate_runtime_macro_context(macro=manifest_fx.macros
        ['macro.root.macro_a'], config=config_postgres, manifest=
        manifest_fx, package_name='root')
    assert ctx['invocation_args_dict']['printer_width'] == 80
    assert ctx['invocation_args_dict']['profile_dir'] == '/dev/null'
    assert isinstance(ctx['invocation_args_dict']['warn_error_options'], Dict)
    assert ctx['invocation_args_dict']['warn_error_options'] == {'include':
        [], 'exclude': []}


def func_puqk9d5w(config_postgres, manifest_fx, get_adapter, get_include_paths
    ):
    ctx = providers.generate_parser_model_context(model=func_jlz7x416(),
        config=config_postgres, manifest=manifest_fx, context_config=mock.
        MagicMock())
    func_dba70615(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)


def func_27oc7zc3(config_postgres, manifest_fx, get_adapter, get_include_paths
    ):
    ctx = providers.generate_runtime_model_context(model=func_jlz7x416(),
        config=config_postgres, manifest=manifest_fx)
    func_dba70615(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)


def func_ysa1ibys(config_postgres):
    ctx = docs.generate_runtime_docs_context(config_postgres, func_jlz7x416
        (), [], 'root')
    func_dba70615(REQUIRED_DOCS_KEYS, MAYBE_KEYS, ctx)


def func_jr8h93la(config_postgres, manifest_fx):
    mn = macros.MacroNamespaceBuilder('root', 'search', MacroStack(), [
        'dbt_postgres', 'dbt'])
    mn.add_macros(manifest_fx.macros.values(), {})
    with pytest.raises(dbt_common.exceptions.CompilationError):
        mn.add_macro(func_owgmzwbv('macro_a', 'root'), {})
    mn.add_macros(func_owgmzwbv('macro_a', 'dbt'), {})


def func_24a4ccqi(config_postgres, manifest_fx):
    mn = macros.MacroNamespaceBuilder('root', 'search', MacroStack(), [
        'dbt_postgres', 'dbt'])
    mbp = func_9fvrmcbz.get_macros_by_package()
    dbt_macro = func_owgmzwbv('some_macro', 'dbt')
    mbp['dbt'] = {'some_macro': dbt_macro}
    pg_macro = func_owgmzwbv('some_macro', 'dbt_postgres')
    mbp['dbt_postgres'] = {'some_macro': pg_macro}
    package_macro = func_owgmzwbv('some_macro', 'root')
    mbp['root']['some_macro'] = package_macro
    namespace = mn.build_namespace(mbp, {})
    dct = dict(namespace)
    for result in [dct, namespace]:
        assert 'dbt' in result
        assert 'root' in result
        assert 'some_macro' in result
        assert 'dbt_postgres' not in result
        assert len(result) == 5
        assert set(result) == {'dbt', 'root', 'some_macro', 'macro_a',
            'macro_b'}
        assert len(result['dbt']) == 1
        assert len(result['root']) == 3
        assert result['dbt']['some_macro'].macro is pg_macro
        assert result['root']['some_macro'].macro is package_macro
        assert result['some_macro'].macro is package_macro


def func_m6soxxis(monkeypatch, config_postgres, manifest_fx, get_adapter,
    get_include_paths):
    reset_metadata_vars()
    envs = {'DBT_ENV_CUSTOM_ENV_RUN_ID': 1234, 'DBT_ENV_CUSTOM_ENV_JOB_ID':
        5678, 'DBT_ENV_RUN_ID': 91011, 'RANDOM_ENV': 121314}
    monkeypatch.setattr(os, 'environ', envs)
    ctx = providers.generate_runtime_macro_context(macro=manifest_fx.macros
        ['macro.root.macro_a'], config=config_postgres, manifest=
        manifest_fx, package_name='root')
    assert ctx['dbt_metadata_envs'] == {'JOB_ID': 5678, 'RUN_ID': 1234}
    reset_metadata_vars()


def func_8l3kzm6d(config_postgres, manifest_fx, get_adapter, get_include_paths
    ):
    ctx = providers.generate_runtime_unit_test_context(unit_test=
        func_bgpamddu(), config=config_postgres, manifest=manifest_fx)
    func_dba70615(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)


def func_y1zgd9iu(config_postgres, manifest_fx, get_adapter, get_include_paths
    ):
    unit_test = func_bgpamddu()
    unit_test.overrides = UnitTestOverrides(macros={'macro_a': 'override'})
    ctx = providers.generate_runtime_unit_test_context(unit_test=unit_test,
        config=config_postgres, manifest=manifest_fx)
    assert ctx['macro_a']() == 'override'


def func_1vkshryh(config_postgres, manifest_fx, get_adapter, get_include_paths
    ):
    unit_test = func_bgpamddu()
    unit_test.overrides = UnitTestOverrides(macros={
        'some_package.some_macro': 'override'})
    dbt_macro = func_owgmzwbv('some_macro', 'some_package')
    manifest_with_dbt_macro = func_fv1ws74o(config_postgres,
        additional_macros=[dbt_macro])
    ctx = providers.generate_runtime_unit_test_context(unit_test=unit_test,
        config=config_postgres, manifest=manifest_with_dbt_macro)
    assert ctx['some_package']['some_macro']() == 'override'


@pytest.mark.parametrize('overrides,expected_override_value', [({
    'some_macro': 'override'}, 'override'), ({'dbt.some_macro': 'override'},
    'override'), ({'some_macro': 'dbt_global_override', 'dbt.some_macro':
    'dbt_namespaced_override'}, 'dbt_global_override'), ({'dbt.some_macro':
    'dbt_namespaced_override', 'some_macro': 'dbt_global_override'},
    'dbt_global_override')])
def func_66a24jnx(overrides, expected_override_value, config_postgres,
    manifest_fx, get_adapter, get_include_paths):
    unit_test = func_bgpamddu()
    unit_test.overrides = UnitTestOverrides(macros=overrides)
    dbt_macro = func_owgmzwbv('some_macro', 'dbt')
    manifest_with_dbt_macro = func_fv1ws74o(config_postgres,
        additional_macros=[dbt_macro])
    ctx = providers.generate_runtime_unit_test_context(unit_test=unit_test,
        config=config_postgres, manifest=manifest_with_dbt_macro)
    assert ctx['some_macro']() == expected_override_value
    assert ctx['dbt']['some_macro']() == expected_override_value
