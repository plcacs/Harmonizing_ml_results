import os
from typing import Any, Dict, Set, Optional, FrozenSet, List, Tuple, Union, Callable, Iterator
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
    def func_5nd1i1cg(self) -> ModelNode:
        return ModelNode(
            alias='model_one', name='model_one', database='dbt', schema='analytics',
            resource_type=NodeType.Model, unique_id='model.root.model_one',
            fqn=['root', 'model_one'], package_name='root',
            original_file_path='model_one.sql', refs=[], sources=[],
            depends_on=DependsOn(), config=NodeConfig.from_dict({
                'enabled': True, 'materialized': 'view', 'persist_docs': {},
                'post-hook': [], 'pre-hook': [], 'vars': {}, 'quoting': {},
                'column_types': {}, 'tags': []
            }), tags=[], path='model_one.sql', language='sql', raw_code='',
            description='', columns={}, checksum=FileHash.from_contents('')
        )

    @pytest.fixture
    def func_wvewj7ni(self) -> mock.MagicMock:
        return mock.MagicMock()

    @pytest.fixture
    def func_1lhz8ff6(self) -> VarProvider:
        return VarProvider({})

    @pytest.fixture
    def func_o2ho9ncb(self, provider: VarProvider) -> mock.MagicMock:
        return mock.MagicMock(
            config_version=2, vars=provider, cli_vars={}, project_name='root'
        )

    def func_2aneuuis(self, model: ModelNode, config: mock.MagicMock, context: Dict[str, Any]) -> None:
        config.cli_vars = {'foo': 'baz'}
        var = providers.RuntimeVar(context, config, model)
        assert var('foo') == 'baz'
        assert var('foo', 'bar') == 'baz'

    def func_kmmxql88(self, model: ModelNode, config: mock.MagicMock, context: Dict[str, Any]) -> None:
        config.cli_vars = {'foo': None}
        var = providers.RuntimeVar(context, config, model)
        assert var('foo') is None
        assert var('foo', 'bar') is None

    def func_dufn0icg(self, model: ModelNode, config: mock.MagicMock, context: Dict[str, Any]) -> None:
        var = providers.RuntimeVar(context, config, model)
        assert var('foo', 'bar') == 'bar'
        with pytest.raises(dbt_common.exceptions.CompilationError):
            var('foo')

    def func_all5n7wu(self, model: ModelNode, config: mock.MagicMock, context: Dict[str, Any]) -> None:
        config.cli_vars = {'foo': 'baz'}
        var = providers.ParseVar(context, config, model)
        assert var('foo') == 'baz'
        assert var('foo', 'bar') == 'baz'

    def func_ypxon2rc(self, model: ModelNode, config: mock.MagicMock, context: Dict[str, Any]) -> None:
        config.cli_vars = {'foo': None}
        var = providers.ParseVar(context, config, model)
        assert var('foo') is None
        assert var('foo', 'bar') is None

    def func_y7ty47d0(self, model: ModelNode, config: mock.MagicMock, context: Dict[str, Any]) -> None:
        var = providers.ParseVar(context, config, model)
        assert var('foo', 'bar') == 'bar'
        assert var('foo') is None


class TestParseWrapper:
    @pytest.fixture
    def func_h87r1p3m(self) -> Any:
        mock_config = mock.MagicMock()
        mock_mp_context = mock.MagicMock()
        adapter_class = adapter_factory()
        return adapter_class(mock_config, mock_mp_context)

    @pytest.fixture
    def func_h8brzvr8(self, mock_adapter: Any) -> providers.ParseDatabaseWrapper:
        namespace = mock.MagicMock()
        return providers.ParseDatabaseWrapper(mock_adapter, namespace)

    @pytest.fixture
    def func_pk68cfgx(self, mock_adapter: Any) -> mock.MagicMock:
        return mock_adapter.responder

    def func_25qhnq42(self, wrapper: providers.ParseDatabaseWrapper, responder: mock.MagicMock) -> None:
        assert wrapper.quote('test_value') == '"test_value"'
        responder.quote.assert_called_once_with('test_value')

    def func_e98uuvt3(self, wrapper: providers.ParseDatabaseWrapper, responder: mock.MagicMock) -> None:
        found = wrapper.get_relation('database', 'schema', 'identifier')
        assert found is None
        responder.get_relation.assert_not_called()


class TestRuntimeWrapper:
    @pytest.fixture
    def func_h87r1p3m(self) -> Any:
        mock_config = mock.MagicMock()
        mock_config.quoting = {'database': True, 'schema': True, 'identifier': True}
        mock_mp_context = mock.MagicMock()
        adapter_class = adapter_factory()
        return adapter_class(mock_config, mock_mp_context)

    @pytest.fixture
    def func_h8brzvr8(self, mock_adapter: Any) -> providers.RuntimeDatabaseWrapper:
        namespace = mock.MagicMock()
        return providers.RuntimeDatabaseWrapper(mock_adapter, namespace)

    @pytest.fixture
    def func_pk68cfgx(self, mock_adapter: Any) -> mock.MagicMock:
        return mock_adapter.responder

    def func_25qhnq42(self, wrapper: providers.RuntimeDatabaseWrapper, responder: mock.MagicMock) -> None:
        assert wrapper.quote('test_value') == '"test_value"'
        responder.quote.assert_called_once_with('test_value')


def func_dba70615(required_keys: FrozenSet[str], maybe_keys: FrozenSet[str], ctx: Dict[str, Any]) -> None:
    keys = set(ctx)
    for key in required_keys:
        assert key in keys, f'{key} in required keys but not in context'
        keys.remove(key)
    extras = keys.difference(maybe_keys)
    assert not extras, f'got extra keys in context: {extras}'


REQUIRED_BASE_KEYS: FrozenSet[str] = frozenset({
    'context', 'builtins', 'dbt_version', 'var', 'env_var', 'return',
    'fromjson', 'tojson', 'fromyaml', 'toyaml', 'set', 'set_strict',
    'zip', 'zip_strict', 'log', 'run_started_at', 'invocation_id',
    'thread_id', 'modules', 'flags', 'print', 'diff_of_two_dicts', 'local_md5'
})
REQUIRED_TARGET_KEYS: FrozenSet[str] = REQUIRED_BASE_KEYS | {'target'}
REQUIRED_DOCS_KEYS: FrozenSet[str] = REQUIRED_TARGET_KEYS | {'project_name'} | {'doc'}
MACROS: FrozenSet[str] = frozenset({'macro_a', 'macro_b', 'root', 'dbt'})
REQUIRED_QUERY_HEADER_KEYS: FrozenSet[str] = REQUIRED_TARGET_KEYS | {
    'project_name', 'context_macro_stack'
} | MACROS
REQUIRED_MACRO_KEYS: FrozenSet[str] = REQUIRED_QUERY_HEADER_KEYS | {
    '_sql_results', 'load_result', 'store_result', 'store_raw_result',
    'validation', 'write', 'render', 'try_or_compiler_error',
    'load_agate_table', 'ref', 'source', 'metric', 'config', 'execute',
    'exceptions', 'database', 'schema', 'adapter', 'api', 'column', 'env',
    'graph', 'model', 'pre_hooks', 'post_hooks', 'sql', 'sql_now',
    'adapter_macro', 'selected_resources', 'invocation_args_dict',
    'submit_python_job', 'dbt_metadata_envs'
}
REQUIRED_MODEL_KEYS: FrozenSet[str] = REQUIRED_MACRO_KEYS | {'this', 'compiled_code'}
MAYBE_KEYS: FrozenSet[str] = frozenset({'debug', 'defer_relation'})
POSTGRES_PROFILE_DATA: Dict[str, Any] = {
    'target': 'test', 'quoting': {}, 'outputs': {
        'test': {
            'type': 'postgres', 'host': 'localhost', 'schema': 'analytics',
            'user': 'test', 'pass': 'test', 'dbname': 'test', 'port': 1
        }
    }
}
PROJECT_DATA: Dict[str, Any] = {
    'name': 'root', 'version': '0.1', 'profile': 'test',
    'project-root': os.getcwd(), 'config-version': 2
}


def func_5nd1i1cg() -> ModelNode:
    return ModelNode(
        alias='model_one', name='model_one', database='dbt', schema='analytics',
        resource_type=NodeType.Model, unique_id='model.root.model_one',
        fqn=['root', 'model_one'], package_name='root',
        original_file_path='model_one.sql', refs=[], sources=[],
        depends_on=DependsOn(), config=NodeConfig.from_dict({
            'enabled': True, 'materialized': 'view', 'persist_docs': {},
            'post-hook': [], 'pre-hook': [], 'vars': {}, 'quoting': {},
            'column_types': {}, 'tags': []
        }), tags=[], path='model_one.sql', language='sql', raw_code='',
        description='', columns={}
    )


def func_4q93qxxu() -> None:
    ctx = base.generate_base_context({})
    func_dba70615(REQUIRED_BASE_KEYS, MAYBE_KEYS, ctx)


def func_owgmzwbv(name: str, package_name: str) -> mock.MagicMock:
    macro = mock.MagicMock(
        __class__=Macro, package_name=package_name,
        resource_type='macro', unique_id=f'macro.{package_name}.{name}'
    )
    macro.name = name
    return macro


def func_fv1ws74o(config: Any, additional_macros: Optional[List[mock.MagicMock]] = None) -> mock.MagicMock:
    default_macro_names = ['macro_a', 'macro_b']
    default_macros = [func_owgmzwbv(name, config.project_name) for name in default_macro_names]
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

    def gmbp() -> Dict[str, Dict[str, mock.MagicMock]]:
        return macros_by_package
    m = mock.MagicMock(macros=manifest_macros)
    m.get_macros_by_package = gmbp
    return m


def func_jlz7x416() -> mock.MagicMock:
    return mock.MagicMock(
        __class__=ModelNode, alias='model_one', name='model_one',
        database='dbt', schema='analytics', resource_type=NodeType.Model,
        unique_id='model.root.model_one', fqn=['root', 'model_one'],
        package_name='root', original_file_path='model_one.sql', refs=[],
        sources=[], depends_on=DependsOn(), config=NodeConfig.from_dict({
            'enabled': True, 'materialized': 'view', 'persist_docs': {},
            'post-hook': [], 'pre-hook': [], 'vars': {}, 'quoting': {},
            'column_types': {}, 'tags': []
        }), tags=[], path='model_one.sql', language='sql', raw_code='',
        description='', columns={}
    )


def func_bgpamddu() -> mock.MagicMock:
    return mock.MagicMock(
        __class__=UnitTestNode, resource_type=NodeType.Unit,
        tested_node_unique_id='model.root.model_one'
    )


@pytest.fixture
def func_qa6l668t() -> Iterator[mock.MagicMock]:
    with mock.patch.object(providers, 'get_adapter') as patch:
        yield patch


@pytest.fixture
def func_hbselznl() -> Iterator[mock.MagicMock]:
    with mock.patch.object(factory, 'get_include_paths') as patch:
        patch.return_value = []
        yield patch


@pytest.fixture
def func_vgyihhmx() -> Any:
    return config_from_parts_or_dicts(PROJECT_DATA, POSTGRES_PROFILE_DATA)


@pytest.fixture
def func_9fvrmcbz(config_postgres: Any) -> mock.MagicMock:
    return func_fv1ws74o(config_postgres)


@pytest.fixture
def func_07qwy5xx(config_postgres: Any, get_adapter: mock.MagicMock) -> Iterator[postgres.PostgresAdapter]:
    adapter = postgres.PostgresAdapter(config_postgres)
    inject_adapter(adapter, postgres.Plugin)
    get_adapter.return_value = adapter
    yield adapter
    clear_plugin(postgres.Plugin)


def func_syybg4y8(config_postgres: Any, manifest_fx: mock.MagicMock) -> None:
    ctx = query_header.generate_query_header_context(
        config=config_postgres, manifest=manifest_fx
    )
    func_dba70615(REQUIRED_QUERY_HEADER_KEYS, MAYBE_KEYS, ctx)


def func_pept5ap7(
    config_postgres: Any,
    manifest_fx: mock.MagicMock,
    get_adapter: mock.MagicMock,
    get_include_paths: mock.MagicMock
) -> None:
    ctx = providers.generate_runtime_macro_context(
        macro=manifest_fx.macros['macro.root.macro_a'],
        config=config_postgres,
        manifest=manifest_fx,
        package_name='root'
    )
    func_dba70615(REQUIRED_MACRO_KEYS, MAYBE_KEYS, ctx)


def func_6n61flm4(
    config_postgres: Any,
    manifest_fx: mock.MagicMock,
    get_adapter: mock.MagicMock,
    get_include_paths: mock.MagicMock
) -> None:
    ctx = providers.generate_runtime_macro_context(
        macro=manifest_fx.macros['macro.root.macro_a'],
        config=config_postgres,
        manifest=manifest_fx,
        package_name='root'
    )
    assert ctx['invocation_args_dict']['printer_width'] == 80
    assert ctx['invocation_args_dict']['profile_dir'] == '/dev/null'
    assert isinstance(ctx['invocation_args_dict']['warn_error_options'], Dict)
    assert ctx['invocation_args_dict']['warn_error_options'] == {
        'include': [], 'exclude': []
    }


def func_puqk9d5w(
    config_postgres: Any,
    manifest_fx: mock.MagicMock,
    get_adapter: mock.MagicMock,
    get_include_paths: mock.MagicMock
) -> None:
    ctx = providers.generate_parser_model_context(
        model=func_jlz7x416(),
        config=config_postgres,
        manifest=manifest_fx,
        context_config=mock.MagicMock()
    )
    func_dba70615(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)


def func_27oc7zc3(
    config_postgres: Any,
    manifest_fx: mock.MagicMock,
    get_adapter: mock.MagicMock,
    get_include_paths: mock.MagicMock
) -> None:
    ctx = providers.generate_runtime_model_context(
        model=func_jlz7x416(),
        config=config_postgres,
        manifest=manifest_fx
    )
    func_dba70615(REQUIRED_MODEL_KEYS, MAYBE_KEYS, ctx)


def func_ysa1ibys(config_postgres: Any) -> None:
    ctx = docs.generate_runtime_docs_context(
        config_postgres, func_jlz7x416(), [], 'root'
    )
    func_dba70615(REQUIRED_DOCS_KEYS, MAYBE_KEYS, ctx)


def func_jr8h93la(config_postgres: Any, manifest_fx: mock.MagicMock) -> None:
    mn = macros.MacroNamespaceBuilder('root', 'search', MacroStack(), ['dbt_postgres', 'dbt'])
    mn.add_macros(manifest_fx.macros.values(), {})
    with pytest.raises(dbt_common.exceptions.CompilationError):
        mn.add_macro(func_owgmzwbv('macro_a', 'root'), {})
    mn.add_macros(func_owgmzwbv('macro_a', 'dbt'), {})


def