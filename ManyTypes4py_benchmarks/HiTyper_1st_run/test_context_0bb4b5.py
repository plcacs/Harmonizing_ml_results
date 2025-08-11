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
    def model(self) -> ModelNode:
        return ModelNode(alias='model_one', name='model_one', database='dbt', schema='analytics', resource_type=NodeType.Model, unique_id='model.root.model_one', fqn=['root', 'model_one'], package_name='root', original_file_path='model_one.sql', refs=[], sources=[], depends_on=DependsOn(), config=NodeConfig.from_dict({'enabled': True, 'materialized': 'view', 'persist_docs': {}, 'post-hook': [], 'pre-hook': [], 'vars': {}, 'quoting': {}, 'column_types': {}, 'tags': []}), tags=[], path='model_one.sql', language='sql', raw_code='', description='', columns={}, checksum=FileHash.from_contents(''))

    @pytest.fixture
    def context(self):
        return mock.MagicMock()

    @pytest.fixture
    def provider(self) -> VarProvider:
        return VarProvider({})

    @pytest.fixture
    def config(self, provider: Union[str, typing.Iterable[str]]):
        return mock.MagicMock(config_version=2, vars=provider, cli_vars={}, project_name='root')

    def test_var_default_something(self, model: Any, config: Union[str, None, dict[str, str]], context: Any) -> None:
        config.cli_vars = {'foo': 'baz'}
        var = providers.RuntimeVar(context, config, model)
        assert var('foo') == 'baz'
        assert var('foo', 'bar') == 'baz'

    def test_var_default_none(self, model: typing.Mapping, config: Union[str, None, dict], context: typing.Mapping) -> None:
        config.cli_vars = {'foo': None}
        var = providers.RuntimeVar(context, config, model)
        assert var('foo') is None
        assert var('foo', 'bar') is None

    def test_var_not_defined(self, model: Union[typing.Mapping, str], config: Union[typing.Mapping, str], context: Union[typing.Mapping, str]) -> None:
        var = providers.RuntimeVar(self.context, config, model)
        assert var('foo', 'bar') == 'bar'
        with pytest.raises(dbt_common.exceptions.CompilationError):
            var('foo')

    def test_parser_var_default_something(self, model: Union[str, None], config: Union[str, None], context: Union[str, None]) -> None:
        config.cli_vars = {'foo': 'baz'}
        var = providers.ParseVar(context, config, model)
        assert var('foo') == 'baz'
        assert var('foo', 'bar') == 'baz'

    def test_parser_var_default_none(self, model: typing.Mapping, config: Union[str, None, dict], context: typing.Mapping) -> None:
        config.cli_vars = {'foo': None}
        var = providers.ParseVar(context, config, model)
        assert var('foo') is None
        assert var('foo', 'bar') is None

    def test_parser_var_not_defined(self, model: str, config: str, context: str) -> None:
        var = providers.ParseVar(context, config, model)
        assert var('foo', 'bar') == 'bar'
        assert var('foo') is None

class TestParseWrapper:

    @pytest.fixture
    def mock_adapter(self):
        mock_config = mock.MagicMock()
        mock_mp_context = mock.MagicMock()
        adapter_class = adapter_factory()
        return adapter_class(mock_config, mock_mp_context)

    @pytest.fixture
    def wrapper(self, mock_adapter: tuple[typing.Union[str,mock.Mock]]):
        namespace = mock.MagicMock()
        return providers.ParseDatabaseWrapper(mock_adapter, namespace)

    @pytest.fixture
    def responder(self, mock_adapter: Any):
        return mock_adapter.responder

    def test_unwrapped_method(self, wrapper: Union[str, bool], responder: Union[app.utils.models.ModelManager, tests.e2e.Helper]) -> None:
        assert wrapper.quote('test_value') == '"test_value"'
        responder.quote.assert_called_once_with('test_value')

    def test_wrapped_method(self, wrapper: Union[str, bool, typing.Callable], responder: Any) -> None:
        found = wrapper.get_relation('database', 'schema', 'identifier')
        assert found is None
        responder.get_relation.assert_not_called()

class TestRuntimeWrapper:

    @pytest.fixture
    def mock_adapter(self):
        mock_config = mock.MagicMock()
        mock_config.quoting = {'database': True, 'schema': True, 'identifier': True}
        mock_mp_context = mock.MagicMock()
        adapter_class = adapter_factory()
        return adapter_class(mock_config, mock_mp_context)

    @pytest.fixture
    def wrapper(self, mock_adapter: tuple[typing.Union[str,mock.Mock]]):
        namespace = mock.MagicMock()
        return providers.RuntimeDatabaseWrapper(mock_adapter, namespace)

    @pytest.fixture
    def responder(self, mock_adapter: Any):
        return mock_adapter.responder

    def test_unwrapped_method(self, wrapper: Union[str, bool], responder: Union[app.utils.models.ModelManager, tests.e2e.Helper]) -> None:
        assert wrapper.quote('test_value') == '"test_value"'
        responder.quote.assert_called_once_with('test_value')

def assert_has_keys(required_keys: Union[set[str], dict], maybe_keys: Union[set[str], list[str]], ctx: Union[list[int], frozenset]) -> None:
    keys = set(ctx)
    for key in required_keys:
        assert key in keys, f'{key} in required keys but not in context'
        keys.remove(key)
    extras = keys.difference(maybe_keys)
    assert not extras, f'got extra keys in context: {extras}'
REQUIRED_BASE_KEYS = frozenset({'context', 'builtins', 'dbt_version', 'var', 'env_var', 'return', 'fromjson', 'tojson', 'fromyaml', 'toyaml', 'set', 'set_strict', 'zip', 'zip_strict', 'log', 'run_started_at', 'invocation_id', 'thread_id', 'modules', 'flags', 'print', 'diff_of_two_dicts', 'local_md5'})
REQUIRED_TARGET_KEYS = REQUIRED_BASE_KEYS | {'target'}
REQUIRED_DOCS_KEYS = REQUIRED_TARGET_KEYS | {'project_name'} | {'doc'}
MACROS = frozenset({'macro_a', 'macro_b', 'root', 'dbt'})
REQUIRED_QUERY_HEADER_KEYS = REQUIRED_TARGET_KEYS | {'project_name', 'context_macro_stack'} | MACROS
REQUIRED_MACRO_KEYS = REQUIRED_QUERY_HEADER_KEYS | {'_sql_results', 'load_result', 'store_result', 'store_raw_result', 'validation', 'write', 'render', 'try_or_compiler_error', 'load_agate_table', 'ref', 'source', 'metric', 'config', 'execute', 'exceptions', 'database', 'schema', 'adapter', 'api', 'column', 'env', 'graph', 'model', 'pre_hooks', 'post_hooks', 'sql', 'sql_now', 'adapter_macro', 'selected_resources', 'invocation_args_dict', 'submit_python_job', 'dbt_metadata_envs'}
REQUIRED_MODEL_KEYS = REQUIRED_MACRO_KEYS | {'this', 'compiled_code'}
MAYBE_KEYS = frozenset({'debug', 'defer_relation'})
POSTGRES_PROFILE_DATA = {'target': 'test', 'quoting': {}, 'outputs': {'test': {'type': 'postgres', 'host': 'localhost', 'schema': 'analytics', 'user': 'test', 'pass': 'test', 'dbname': 'test', 'port': 1}}}
PROJECT_DATA = {'name': 'root', 'version': '0.1', 'profile': 'test', 'project-root': os.getcwd(), 'config-version': 2}

def model() -> ModelNode:
    return ModelNode(alias='model_one', name='model_one', database='dbt', schema='analytics', resource_type=NodeType.Model, unique_id='model.root.model_one', fqn=['root', 'model_one'], package_name='root', original_file_path='model_one.sql', refs=[], sources=[], depends_on=DependsOn(), config=NodeConfig.from_dict({'enabled': True, 'materialized': 'view', 'persist_docs': {}, 'post-hook': [], 'pre-hook': [], 'vars': {}, 'quoting': {}, 'column_types': {}, 'tags': []}), tags=[], path='model_one.sql', language='sql', raw_code='', description='', columns={})

def test_base_context() -> None:
    ctx = base.generate_base_context({})
    assert_has_keys(REQUIRED_BASE_KEYS, MAYBE_KEYS, ctx)

def mock_macro(name: Union[str, dict[str, typing.Any]], package_name: Union[str, dict[str, typing.Any], None]):
    macro = mock.MagicMock(__class__=Macro, package_name=package_name, resource_type='macro', unique_id=f'macro.{package_name}.{name}')
    macro.name = name
    return macro

def mock_manifest(config: Union[list, typing.Mapping, dict[str, typing.Any]], additional_macros: Union[None, list, typing.Mapping, dict[str, typing.Any]]=None):
    default_macro_names = ['macro_a', 'macro_b']
    default_macros = [mock_macro(name, config.project_name) for name in default_macro_names]
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

    def gmbp():
        return macros_by_package
    m = mock.MagicMock(macros=manifest_macros)
    m.get_macros_by_package = gmbp
    return m

def mock_model():
    return mock.MagicMock(__class__=ModelNode, alias='model_one', name='model_one', database='dbt', schema='analytics', resource_type=NodeType.Model, unique_id='model.root.model_one', fqn=['root', 'model_one'], package_name='root', original_file_path='model_one.sql', refs=[], sources=[], depends_on=DependsOn(), config=NodeConfig.from_dict({'enabled': True, 'materialized': 'view', 'persist_docs': {}, 'post-hook': [], 'pre-hook': [], 'vars': {}, 'quoting': {}, 'column_types': {}, 'tags': []}), tags=[], path='model_one.sql', language='sql', raw_code='', description='', columns={})

def mock_unit_test_node():
    return mock.MagicMock(__class__=UnitTestNode, resource_type=NodeType.Unit, tested_node_unique_id='model.root.model_one')

@pytest.fixture
def get_adapter() -> typing.Generator:
    with mock.patch.object(providers, 'get_adapter') as patch:
        yield patch

@pytest.fixture
def get_include_paths() -> typing.Generator:
    with mock.patch.object(factory, 'get_include_paths') as patch:
        patch.return_value = []
        yield patch

@pytest.fixture
def config_postgres() -> Union[dict[str, typing.Any], str, typing.Callable[str, None]]:
    return config_from_parts_or_dicts(PROJECT_DATA, POSTGRES_PROFILE_DATA)

@pytest.fixture
def manifest_fx(config_postgres: Union[dict[str, typing.Any], dict, str]):
    return mock_manifest(config_postgres)

@pytest.fixture
def postgres_adapter(config_postgres: Union[list[str], dict[str, typing.Any]], get_adapter: backup.config.Config) -> typing.Generator:
    adapter = postgres.PostgresAdapter(config_postgres)
    inject_adapter(adapter, postgres.Plugin)
    get_adapter.return_value = adapter
    yield adapter
    clear_plugin(postgres.Plugin)

def test_query_header_context(config_postgres: Union[typing.IO, dict, backup.config.Config], manifest_fx: Union[typing.IO, dict, backup.config.Config]) -> None:
    ctx = query_header.generate_query_header_context(config=config_postgres, manifest=manifest_fx)
    assert_has_keys(REQUIRED_QUERY_HEADER_KEYS, MAYBE_KEYS, ctx)

def test_macro_runtime_context(config_postgres: str, manifest_fx: str, get_adapter: Union[bool, dict[str, typing.Any]], get_include_paths: Union[bool, dict[str, typing.Any]]) -> None:
    ctx = providers.generate_runtime_macro_context(macro=manifest_fx.macros['macro.root.macro_a'], config=config_postgres, manifest=manifest_fx, package_name='root')
    assert_has_keys(REQUIRED_MACRO_KEYS, MAYBE_KEYS, ctx)

def test_invocation_args_to_dict_in_macro_runtime_context(config_postgres: Union[list[str], dict], manifest_fx: Union[list[str], dict], get_adapter: Union[bool, dict[str, typing.Any]], get_include_paths: Union[bool, dict[str, typing.Any]]) -> None:
    ctx = providers.generate_runtime_macro_context(macro=manifest_fx.macros['macro.root.macro_a'], config=config_postgres, manifest=manifest_fx, package_name='root')
    assert ctx['invocation_args_dict']['printer_width'] == 80
    assert ctx['invocation_args_dict']['profile_dir'] == '/dev/null'
    assert isinstance(ctx['invocation_args_dict']['warn_error_options'], Dict)
    assert ctx['invocation_args_dict']['warn_error_options'] == {'include': [], 'exclude': []}

def test_model_parse_context(config_postgres: pyramid.config.Configurator, manifest_fx: pyramid.config.Configurator, get_adapter: Union[bool, dict[str, typing.Any]], get_include_paths: Union[bool, dict[str, typing.Any]]) -> None:
    ctx = providers.generate_parser_model_context(model=mock_model(), config=config_postgres, manifest=manifest_fx, context_config=mock.MagicMock())
    assert_has_keys(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)

def test_model_runtime_context(config_postgres: Union[human_activities.config.Config, core.base.utilities.GeneralUtils, pyramid.config.Configurator], manifest_fx: Union[human_activities.config.Config, core.base.utilities.GeneralUtils, pyramid.config.Configurator], get_adapter: Union[bool, dict[str, typing.Any]], get_include_paths: Union[bool, dict[str, typing.Any]]) -> None:
    ctx = providers.generate_runtime_model_context(model=mock_model(), config=config_postgres, manifest=manifest_fx)
    assert_has_keys(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)

def test_docs_runtime_context(config_postgres: Union[pyramid.config.Configurator, str, meier.config.Config]) -> None:
    ctx = docs.generate_runtime_docs_context(config_postgres, mock_model(), [], 'root')
    assert_has_keys(REQUIRED_DOCS_KEYS, MAYBE_KEYS, ctx)

def test_macro_namespace_duplicates(config_postgres: Union[backup.config.Config, bool], manifest_fx: Any) -> None:
    mn = macros.MacroNamespaceBuilder('root', 'search', MacroStack(), ['dbt_postgres', 'dbt'])
    mn.add_macros(manifest_fx.macros.values(), {})
    with pytest.raises(dbt_common.exceptions.CompilationError):
        mn.add_macro(mock_macro('macro_a', 'root'), {})
    mn.add_macros(mock_macro('macro_a', 'dbt'), {})

def test_macro_namespace(config_postgres: Union[dbcontracts.graph.manifesManifest, backup.config.Config, dict], manifest_fx: Any) -> None:
    mn = macros.MacroNamespaceBuilder('root', 'search', MacroStack(), ['dbt_postgres', 'dbt'])
    mbp = manifest_fx.get_macros_by_package()
    dbt_macro = mock_macro('some_macro', 'dbt')
    mbp['dbt'] = {'some_macro': dbt_macro}
    pg_macro = mock_macro('some_macro', 'dbt_postgres')
    mbp['dbt_postgres'] = {'some_macro': pg_macro}
    package_macro = mock_macro('some_macro', 'root')
    mbp['root']['some_macro'] = package_macro
    namespace = mn.build_namespace(mbp, {})
    dct = dict(namespace)
    for result in [dct, namespace]:
        assert 'dbt' in result
        assert 'root' in result
        assert 'some_macro' in result
        assert 'dbt_postgres' not in result
        assert len(result) == 5
        assert set(result) == {'dbt', 'root', 'some_macro', 'macro_a', 'macro_b'}
        assert len(result['dbt']) == 1
        assert len(result['root']) == 3
        assert result['dbt']['some_macro'].macro is pg_macro
        assert result['root']['some_macro'].macro is package_macro
        assert result['some_macro'].macro is package_macro

def test_dbt_metadata_envs(monkeypatch: dbcontracts.graph.manifesManifest, config_postgres: list[str], manifest_fx: list[str], get_adapter: Union[bool, neuromation.api.config._AuthConfig, dict], get_include_paths: Union[bool, neuromation.api.config._AuthConfig, dict]) -> None:
    reset_metadata_vars()
    envs = {'DBT_ENV_CUSTOM_ENV_RUN_ID': 1234, 'DBT_ENV_CUSTOM_ENV_JOB_ID': 5678, 'DBT_ENV_RUN_ID': 91011, 'RANDOM_ENV': 121314}
    monkeypatch.setattr(os, 'environ', envs)
    ctx = providers.generate_runtime_macro_context(macro=manifest_fx.macros['macro.root.macro_a'], config=config_postgres, manifest=manifest_fx, package_name='root')
    assert ctx['dbt_metadata_envs'] == {'JOB_ID': 5678, 'RUN_ID': 1234}
    reset_metadata_vars()

def test_unit_test_runtime_context(config_postgres: Union[backup.config.Config, dict], manifest_fx: Union[backup.config.Config, dict], get_adapter: Union[bool, dict[str, typing.Any]], get_include_paths: Union[bool, dict[str, typing.Any]]) -> None:
    ctx = providers.generate_runtime_unit_test_context(unit_test=mock_unit_test_node(), config=config_postgres, manifest=manifest_fx)
    assert_has_keys(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)

def test_unit_test_runtime_context_macro_overrides_global(config_postgres: Union[str, tbump.config.Config, dict], manifest_fx: Union[str, tbump.config.Config, dict], get_adapter: Union[backup.config.Config, str, bool], get_include_paths: Union[backup.config.Config, str, bool]) -> None:
    unit_test = mock_unit_test_node()
    unit_test.overrides = UnitTestOverrides(macros={'macro_a': 'override'})
    ctx = providers.generate_runtime_unit_test_context(unit_test=unit_test, config=config_postgres, manifest=manifest_fx)
    assert ctx['macro_a']() == 'override'

def test_unit_test_runtime_context_macro_overrides_package(config_postgres: Union[typing.IO, list[str], str], manifest_fx: Union[int, neuromation.api.config._AuthConfig, dict[str, typing.Any]], get_adapter: Union[int, neuromation.api.config._AuthConfig, dict[str, typing.Any]], get_include_paths: Union[int, neuromation.api.config._AuthConfig, dict[str, typing.Any]]) -> None:
    unit_test = mock_unit_test_node()
    unit_test.overrides = UnitTestOverrides(macros={'some_package.some_macro': 'override'})
    dbt_macro = mock_macro('some_macro', 'some_package')
    manifest_with_dbt_macro = mock_manifest(config_postgres, additional_macros=[dbt_macro])
    ctx = providers.generate_runtime_unit_test_context(unit_test=unit_test, config=config_postgres, manifest=manifest_with_dbt_macro)
    assert ctx['some_package']['some_macro']() == 'override'

@pytest.mark.parametrize('overrides,expected_override_value', [({'some_macro': 'override'}, 'override'), ({'dbt.some_macro': 'override'}, 'override'), ({'some_macro': 'dbt_global_override', 'dbt.some_macro': 'dbt_namespaced_override'}, 'dbt_global_override'), ({'dbt.some_macro': 'dbt_namespaced_override', 'some_macro': 'dbt_global_override'}, 'dbt_global_override')])
def test_unit_test_runtime_context_macro_overrides_dbt_macro(overrides: Union[dict, dict[str, typing.Any]], expected_override_value: Union[dict, pyramid.config.Configurator, str], config_postgres: Union[list[str], utils.CWLObjectType, tbump.config.Config], manifest_fx: dict[str, typing.Any], get_adapter: dict[str, typing.Any], get_include_paths: dict[str, typing.Any]) -> None:
    unit_test = mock_unit_test_node()
    unit_test.overrides = UnitTestOverrides(macros=overrides)
    dbt_macro = mock_macro('some_macro', 'dbt')
    manifest_with_dbt_macro = mock_manifest(config_postgres, additional_macros=[dbt_macro])
    ctx = providers.generate_runtime_unit_test_context(unit_test=unit_test, config=config_postgres, manifest=manifest_with_dbt_macro)
    assert ctx['some_macro']() == expected_override_value
    assert ctx['dbt']['some_macro']() == expected_override_value