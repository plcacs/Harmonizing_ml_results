#!/usr/bin/env python3
import os
from typing import Any, Dict, Set, List, Optional, Iterator
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
        return ModelNode(
            alias='model_one',
            name='model_one',
            database='dbt',
            schema='analytics',
            resource_type=NodeType.Model,
            unique_id='model.root.model_one',
            fqn=['root', 'model_one'],
            package_name='root',
            original_file_path='model_one.sql',
            refs=[],
            sources=[],
            depends_on=DependsOn(),
            config=NodeConfig.from_dict({
                'enabled': True,
                'materialized': 'view',
                'persist_docs': {},
                'post-hook': [],
                'pre-hook': [],
                'vars': {},
                'quoting': {},
                'column_types': {},
                'tags': []
            }),
            tags=[],
            path='model_one.sql',
            language='sql',
            raw_code='',
            description='',
            columns={},
            checksum=FileHash.from_contents('')
        )

    @pytest.fixture
    def context(self) -> mock.MagicMock:
        return mock.MagicMock()

    @pytest.fixture
    def provider(self) -> VarProvider:
        return VarProvider({})

    @pytest.fixture
    def config(self, provider: VarProvider) -> mock.MagicMock:
        return mock.MagicMock(config_version=2, vars=provider, cli_vars={}, project_name='root')

    def test_var_default_something(self, model: ModelNode, config: mock.MagicMock, context: mock.MagicMock) -> None:
        config.cli_vars = {'foo': 'baz'}
        var = providers.RuntimeVar(context, config, model)
        assert var('foo') == 'baz'
        assert var('foo', 'bar') == 'baz'

    def test_var_default_none(self, model: ModelNode, config: mock.MagicMock, context: mock.MagicMock) -> None:
        config.cli_vars = {'foo': None}
        var = providers.RuntimeVar(context, config, model)
        assert var('foo') is None
        assert var('foo', 'bar') is None

    def test_var_not_defined(self, model: ModelNode, config: mock.MagicMock, context: mock.MagicMock) -> None:
        var = providers.RuntimeVar(context, config, model)
        assert var('foo', 'bar') == 'bar'
        with pytest.raises(dbt_common.exceptions.CompilationError):
            var('foo')

    def test_parser_var_default_something(self, model: ModelNode, config: mock.MagicMock, context: mock.MagicMock) -> None:
        config.cli_vars = {'foo': 'baz'}
        var = providers.ParseVar(context, config, model)
        assert var('foo') == 'baz'
        assert var('foo', 'bar') == 'baz'

    def test_parser_var_default_none(self, model: ModelNode, config: mock.MagicMock, context: mock.MagicMock) -> None:
        config.cli_vars = {'foo': None}
        var = providers.ParseVar(context, config, model)
        assert var('foo') is None
        assert var('foo', 'bar') is None

    def test_parser_var_not_defined(self, model: ModelNode, config: mock.MagicMock, context: mock.MagicMock) -> None:
        var = providers.ParseVar(context, config, model)
        assert var('foo', 'bar') == 'bar'
        assert var('foo') is None

class TestParseWrapper:
    @pytest.fixture
    def mock_adapter(self) -> Any:
        mock_config: mock.MagicMock = mock.MagicMock()
        mock_mp_context: mock.MagicMock = mock.MagicMock()
        adapter_class = adapter_factory()
        return adapter_class(mock_config, mock_mp_context)

    @pytest.fixture
    def wrapper(self, mock_adapter: Any) -> providers.ParseDatabaseWrapper:
        namespace: mock.MagicMock = mock.MagicMock()
        return providers.ParseDatabaseWrapper(mock_adapter, namespace)

    @pytest.fixture
    def responder(self, mock_adapter: Any) -> Any:
        return mock_adapter.responder

    def test_unwrapped_method(self, wrapper: providers.ParseDatabaseWrapper, responder: Any) -> None:
        assert wrapper.quote('test_value') == '"test_value"'
        responder.quote.assert_called_once_with('test_value')

    def test_wrapped_method(self, wrapper: providers.ParseDatabaseWrapper, responder: Any) -> None:
        found = wrapper.get_relation('database', 'schema', 'identifier')
        assert found is None
        responder.get_relation.assert_not_called()

class TestRuntimeWrapper:
    @pytest.fixture
    def mock_adapter(self) -> Any:
        mock_config: mock.MagicMock = mock.MagicMock()
        mock_config.quoting = {'database': True, 'schema': True, 'identifier': True}
        mock_mp_context: mock.MagicMock = mock.MagicMock()
        adapter_class = adapter_factory()
        return adapter_class(mock_config, mock_mp_context)

    @pytest.fixture
    def wrapper(self, mock_adapter: Any) -> providers.RuntimeDatabaseWrapper:
        namespace: mock.MagicMock = mock.MagicMock()
        return providers.RuntimeDatabaseWrapper(mock_adapter, namespace)

    @pytest.fixture
    def responder(self, mock_adapter: Any) -> Any:
        return mock_adapter.responder

    def test_unwrapped_method(self, wrapper: providers.RuntimeDatabaseWrapper, responder: Any) -> None:
        assert wrapper.quote('test_value') == '"test_value"'
        responder.quote.assert_called_once_with('test_value')

def assert_has_keys(required_keys: Set[str], maybe_keys: Set[str], ctx: Dict[str, Any]) -> None:
    keys: Set[str] = set(ctx)
    for key in required_keys:
        assert key in keys, f'{key} in required keys but not in context'
        keys.remove(key)
    extras: Set[str] = keys.difference(maybe_keys)
    assert not extras, f'got extra keys in context: {extras}'

REQUIRED_BASE_KEYS: Set[str] = frozenset({
    'context', 'builtins', 'dbt_version', 'var', 'env_var', 'return', 'fromjson', 'tojson',
    'fromyaml', 'toyaml', 'set', 'set_strict', 'zip', 'zip_strict', 'log', 'run_started_at',
    'invocation_id', 'thread_id', 'modules', 'flags', 'print', 'diff_of_two_dicts', 'local_md5'
})
REQUIRED_TARGET_KEYS: Set[str] = REQUIRED_BASE_KEYS.union({'target'})
REQUIRED_DOCS_KEYS: Set[str] = REQUIRED_TARGET_KEYS.union({'project_name'}).union({'doc'})
MACROS: Set[str] = frozenset({'macro_a', 'macro_b', 'root', 'dbt'})
REQUIRED_QUERY_HEADER_KEYS: Set[str] = REQUIRED_TARGET_KEYS.union({'project_name', 'context_macro_stack'}).union(MACROS)
REQUIRED_MACRO_KEYS: Set[str] = REQUIRED_QUERY_HEADER_KEYS.union({
    '_sql_results', 'load_result', 'store_result', 'store_raw_result', 'validation', 'write', 'render',
    'try_or_compiler_error', 'load_agate_table', 'ref', 'source', 'metric', 'config', 'execute',
    'exceptions', 'database', 'schema', 'adapter', 'api', 'column', 'env', 'graph', 'model', 'pre_hooks',
    'post_hooks', 'sql', 'sql_now', 'adapter_macro', 'selected_resources', 'invocation_args_dict',
    'submit_python_job', 'dbt_metadata_envs'
})
REQUIRED_MODEL_KEYS: Set[str] = REQUIRED_MACRO_KEYS.union({'this', 'compiled_code'})
MAYBE_KEYS: Set[str] = frozenset({'debug', 'defer_relation'})
POSTGRES_PROFILE_DATA: Dict[str, Any] = {
    'target': 'test',
    'quoting': {},
    'outputs': {'test': {'type': 'postgres', 'host': 'localhost', 'schema': 'analytics', 'user': 'test', 'pass': 'test', 'dbname': 'test', 'port': 1}}
}
PROJECT_DATA: Dict[str, Any] = {
    'name': 'root',
    'version': '0.1',
    'profile': 'test',
    'project-root': os.getcwd(),
    'config-version': 2
}

def model() -> ModelNode:
    return ModelNode(
        alias='model_one',
        name='model_one',
        database='dbt',
        schema='analytics',
        resource_type=NodeType.Model,
        unique_id='model.root.model_one',
        fqn=['root', 'model_one'],
        package_name='root',
        original_file_path='model_one.sql',
        refs=[],
        sources=[],
        depends_on=DependsOn(),
        config=NodeConfig.from_dict({
            'enabled': True,
            'materialized': 'view',
            'persist_docs': {},
            'post-hook': [],
            'pre-hook': [],
            'vars': {},
            'quoting': {},
            'column_types': {},
            'tags': []
        }),
        tags=[],
        path='model_one.sql',
        language='sql',
        raw_code='',
        description='',
        columns={}
    )

def mock_macro(name: str, package_name: str) -> mock.MagicMock:
    macro = mock.MagicMock(__class__=Macro, package_name=package_name, resource_type='macro', unique_id=f'macro.{package_name}.{name}')
    macro.name = name
    return macro

def mock_manifest(config: Any, additional_macros: Optional[List[Any]] = None) -> mock.MagicMock:
    default_macro_names: List[str] = ['macro_a', 'macro_b']
    default_macros: List[mock.MagicMock] = [mock_macro(name, config.project_name) for name in default_macro_names]
    additional_macros = additional_macros or []
    all_macros: List[mock.MagicMock] = default_macros + additional_macros
    manifest_macros: Dict[str, Any] = {}
    macros_by_package: Dict[str, Dict[str, Any]] = {}
    for macro in all_macros:
        manifest_macros[macro.unique_id] = macro
        if macro.package_name not in macros_by_package:
            macros_by_package[macro.package_name] = {}
        macro_package: Dict[str, Any] = macros_by_package[macro.package_name]
        macro_package[macro.name] = macro

    def gmbp() -> Dict[str, Dict[str, Any]]:
        return macros_by_package
    m = mock.MagicMock(macros=manifest_macros)
    m.get_macros_by_package = gmbp
    return m

def mock_model() -> mock.MagicMock:
    return mock.MagicMock(
        __class__=ModelNode,
        alias='model_one',
        name='model_one',
        database='dbt',
        schema='analytics',
        resource_type=NodeType.Model,
        unique_id='model.root.model_one',
        fqn=['root', 'model_one'],
        package_name='root',
        original_file_path='model_one.sql',
        refs=[],
        sources=[],
        depends_on=DependsOn(),
        config=NodeConfig.from_dict({
            'enabled': True,
            'materialized': 'view',
            'persist_docs': {},
            'post-hook': [],
            'pre-hook': [],
            'vars': {},
            'quoting': {},
            'column_types': {},
            'tags': []
        }),
        tags=[],
        path='model_one.sql',
        language='sql',
        raw_code='',
        description='',
        columns={}
    )

def mock_unit_test_node() -> mock.MagicMock:
    return mock.MagicMock(__class__=UnitTestNode, resource_type=NodeType.Unit, tested_node_unique_id='model.root.model_one')

@pytest.fixture
def get_adapter() -> Iterator[mock.MagicMock]:
    with mock.patch.object(providers, 'get_adapter') as patch:
        yield patch

@pytest.fixture
def get_include_paths() -> Iterator[mock.MagicMock]:
    with mock.patch.object(factory, 'get_include_paths') as patch:
        patch.return_value = []
        yield patch

@pytest.fixture
def config_postgres() -> Any:
    return config_from_parts_or_dicts(PROJECT_DATA, POSTGRES_PROFILE_DATA)

@pytest.fixture
def manifest_fx(config_postgres: Any) -> mock.MagicMock:
    return mock_manifest(config_postgres)

@pytest.fixture
def postgres_adapter(config_postgres: Any, get_adapter: Any) -> Iterator[Any]:
    adapter = postgres.PostgresAdapter(config_postgres)
    inject_adapter(adapter, postgres.Plugin)
    get_adapter.return_value = adapter
    yield adapter
    clear_plugin(postgres.Plugin)

def test_query_header_context(config_postgres: Any, manifest_fx: Any) -> None:
    ctx: Dict[str, Any] = query_header.generate_query_header_context(config=config_postgres, manifest=manifest_fx)
    assert_has_keys(REQUIRED_QUERY_HEADER_KEYS, MAYBE_KEYS, ctx)

def test_macro_runtime_context(config_postgres: Any, manifest_fx: Any, get_adapter: Any, get_include_paths: Any) -> None:
    ctx: Dict[str, Any] = providers.generate_runtime_macro_context(
        macro=manifest_fx.macros['macro.root.macro_a'],
        config=config_postgres,
        manifest=manifest_fx,
        package_name='root'
    )
    assert_has_keys(REQUIRED_MACRO_KEYS, MAYBE_KEYS, ctx)

def test_invocation_args_to_dict_in_macro_runtime_context(config_postgres: Any, manifest_fx: Any, get_adapter: Any, get_include_paths: Any) -> None:
    ctx: Dict[str, Any] = providers.generate_runtime_macro_context(
        macro=manifest_fx.macros['macro.root.macro_a'],
        config=config_postgres,
        manifest=manifest_fx,
        package_name='root'
    )
    assert ctx['invocation_args_dict']['printer_width'] == 80
    assert ctx['invocation_args_dict']['profile_dir'] == '/dev/null'
    assert isinstance(ctx['invocation_args_dict']['warn_error_options'], Dict)
    assert ctx['invocation_args_dict']['warn_error_options'] == {'include': [], 'exclude': []}

def test_model_parse_context(config_postgres: Any, manifest_fx: Any, get_adapter: Any, get_include_paths: Any) -> None:
    ctx: Dict[str, Any] = providers.generate_parser_model_context(
        model=mock_model(),
        config=config_postgres,
        manifest=manifest_fx,
        context_config=mock.MagicMock()
    )
    assert_has_keys(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)

def test_model_runtime_context(config_postgres: Any, manifest_fx: Any, get_adapter: Any, get_include_paths: Any) -> None:
    ctx: Dict[str, Any] = providers.generate_runtime_model_context(
        model=mock_model(),
        config=config_postgres,
        manifest=manifest_fx
    )
    assert_has_keys(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)

def test_docs_runtime_context(config_postgres: Any) -> None:
    ctx: Dict[str, Any] = docs.generate_runtime_docs_context(config_postgres, mock_model(), [], 'root')
    assert_has_keys(REQUIRED_DOCS_KEYS, MAYBE_KEYS, ctx)

def test_macro_namespace_duplicates(config_postgres: Any, manifest_fx: Any) -> None:
    mn = macros.MacroNamespaceBuilder('root', 'search', MacroStack(), ['dbt_postgres', 'dbt'])
    mn.add_macros(manifest_fx.macros.values(), {})
    with pytest.raises(dbt_common.exceptions.CompilationError):
        mn.add_macro(mock_macro('macro_a', 'root'), {})
    mn.add_macros(mock_macro('macro_a', 'dbt'), {})

def test_macro_namespace(config_postgres: Any, manifest_fx: Any) -> None:
    mn = macros.MacroNamespaceBuilder('root', 'search', MacroStack(), ['dbt_postgres', 'dbt'])
    mbp: Dict[str, Dict[str, Any]] = manifest_fx.get_macros_by_package()
    dbt_macro = mock_macro('some_macro', 'dbt')
    mbp['dbt'] = {'some_macro': dbt_macro}
    pg_macro = mock_macro('some_macro', 'dbt_postgres')
    mbp['dbt_postgres'] = {'some_macro': pg_macro}
    package_macro = mock_macro('some_macro', 'root')
    mbp['root']['some_macro'] = package_macro
    namespace = mn.build_namespace(mbp, {})
    dct: Dict[str, Any] = dict(namespace)
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

def test_dbt_metadata_envs(monkeypatch: Any, config_postgres: Any, manifest_fx: Any, get_adapter: Any, get_include_paths: Any) -> None:
    reset_metadata_vars()
    envs: Dict[str, Any] = {
        'DBT_ENV_CUSTOM_ENV_RUN_ID': 1234,
        'DBT_ENV_CUSTOM_ENV_JOB_ID': 5678,
        'DBT_ENV_RUN_ID': 91011,
        'RANDOM_ENV': 121314
    }
    monkeypatch.setattr(os, 'environ', envs)
    ctx: Dict[str, Any] = providers.generate_runtime_macro_context(
        macro=manifest_fx.macros['macro.root.macro_a'],
        config=config_postgres,
        manifest=manifest_fx,
        package_name='root'
    )
    assert ctx['dbt_metadata_envs'] == {'JOB_ID': 5678, 'RUN_ID': 1234}
    reset_metadata_vars()

def test_unit_test_runtime_context(config_postgres: Any, manifest_fx: Any, get_adapter: Any, get_include_paths: Any) -> None:
    ctx: Dict[str, Any] = providers.generate_runtime_unit_test_context(
        unit_test=mock_unit_test_node(),
        config=config_postgres,
        manifest=manifest_fx
    )
    assert_has_keys(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)

def test_unit_test_runtime_context_macro_overrides_global(config_postgres: Any, manifest_fx: Any, get_adapter: Any, get_include_paths: Any) -> None:
    unit_test: mock.MagicMock = mock_unit_test_node()
    unit_test.overrides = UnitTestOverrides(macros={'macro_a': 'override'})
    ctx: Dict[str, Any] = providers.generate_runtime_unit_test_context(
        unit_test=unit_test,
        config=config_postgres,
        manifest=manifest_fx
    )
    assert ctx['macro_a']() == 'override'

def test_unit_test_runtime_context_macro_overrides_package(config_postgres: Any, manifest_fx: Any, get_adapter: Any, get_include_paths: Any) -> None:
    unit_test: mock.MagicMock = mock_unit_test_node()
    unit_test.overrides = UnitTestOverrides(macros={'some_package.some_macro': 'override'})
    dbt_macro = mock_macro('some_macro', 'some_package')
    manifest_with_dbt_macro: mock.MagicMock = mock_manifest(config_postgres, additional_macros=[dbt_macro])
    ctx: Dict[str, Any] = providers.generate_runtime_unit_test_context(
        unit_test=unit_test,
        config=config_postgres,
        manifest=manifest_with_dbt_macro
    )
    assert ctx['some_package']['some_macro']() == 'override'

@pytest.mark.parametrize(
    'overrides,expected_override_value',
    [
        ({'some_macro': 'override'}, 'override'),
        ({'dbt.some_macro': 'override'}, 'override'),
        ({'some_macro': 'dbt_global_override', 'dbt.some_macro': 'dbt_namespaced_override'}, 'dbt_global_override'),
        ({'dbt.some_macro': 'dbt_namespaced_override', 'some_macro': 'dbt_global_override'}, 'dbt_global_override')
    ]
)
def test_unit_test_runtime_context_macro_overrides_dbt_macro(
    overrides: Dict[str, str],
    expected_override_value: str,
    config_postgres: Any,
    manifest_fx: Any,
    get_adapter: Any,
    get_include_paths: Any
) -> None:
    unit_test: mock.MagicMock = mock_unit_test_node()
    unit_test.overrides = UnitTestOverrides(macros=overrides)
    dbt_macro = mock_macro('some_macro', 'dbt')
    manifest_with_dbt_macro: mock.MagicMock = mock_manifest(config_postgres, additional_macros=[dbt_macro])
    ctx: Dict[str, Any] = providers.generate_runtime_unit_test_context(
        unit_test=unit_test,
        config=config_postgres,
        manifest=manifest_with_dbt_macro
    )
    assert ctx['some_macro']() == expected_override_value
    assert ctx['dbt']['some_macro']() == expected_override_value
