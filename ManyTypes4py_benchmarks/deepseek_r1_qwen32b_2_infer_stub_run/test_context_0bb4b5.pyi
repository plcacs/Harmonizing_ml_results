import os
from typing import Any, Dict, Set, MagicMock, Optional, Union, List, FrozenSet
from unittest.mock import MagicMock
from dbt.adapters.postgres import PostgresAdapter
from dbt.contracts.graph.nodes import ModelNode, Macro, UnitTestNode
from dbt.node_types import NodeType
from tests.unit.mock_adapter import adapter_factory

class TestVar:
    @pytest.fixture
    def model(self) -> ModelNode:
        ...

    @pytest.fixture
    def context(self) -> MagicMock:
        ...

    @pytest.fixture
    def provider(self) -> VarProvider:
        ...

    @pytest.fixture
    def config(self, provider: VarProvider) -> MagicMock:
        ...

    def test_var_default_something(self, model: ModelNode, config: MagicMock, context: MagicMock) -> None:
        ...

    def test_var_default_none(self, model: ModelNode, config: MagicMock, context: MagicMock) -> None:
        ...

    def test_var_not_defined(self, model: ModelNode, config: MagicMock, context: MagicMock) -> None:
        ...

    def test_parser_var_default_something(self, model: ModelNode, config: MagicMock, context: MagicMock) -> None:
        ...

    def test_parser_var_default_none(self, model: ModelNode, config: MagicMock, context: MagicMock) -> None:
        ...

    def test_parser_var_not_defined(self, model: ModelNode, config: MagicMock, context: MagicMock) -> None:
        ...

class TestParseWrapper:
    @pytest.fixture
    def mock_adapter(self) -> PostgresAdapter:
        ...

    @pytest.fixture
    def wrapper(self, mock_adapter: PostgresAdapter) -> providers.ParseDatabaseWrapper:
        ...

    @pytest.fixture
    def responder(self, mock_adapter: PostgresAdapter) -> MagicMock:
        ...

    def test_unwrapped_method(self, wrapper: providers.ParseDatabaseWrapper, responder: MagicMock) -> None:
        ...

    def test_wrapped_method(self, wrapper: providers.ParseDatabaseWrapper, responder: MagicMock) -> None:
        ...

class TestRuntimeWrapper:
    @pytest.fixture
    def mock_adapter(self) -> PostgresAdapter:
        ...

    @pytest.fixture
    def wrapper(self, mock_adapter: PostgresAdapter) -> providers.RuntimeDatabaseWrapper:
        ...

    @pytest.fixture
    def responder(self, mock_adapter: PostgresAdapter) -> MagicMock:
        ...

    def test_unwrapped_method(self, wrapper: providers.RuntimeDatabaseWrapper, responder: MagicMock) -> None:
        ...

def assert_has_keys(required_keys: FrozenSet[str], maybe_keys: FrozenSet[str], ctx: Dict[str, Any]) -> None:
    ...

def model() -> ModelNode:
    ...

def test_base_context() -> Dict[str, Any]:
    ...

def mock_macro(name: str, package_name: str) -> Macro:
    ...

def mock_manifest(config: Any, additional_macros: Optional[List[Macro]] = None) -> MagicMock:
    ...

def mock_model() -> MagicMock:
    ...

def mock_unit_test_node() -> UnitTestNode:
    ...

@pytest.fixture
def get_adapter() -> MagicMock:
    ...

@pytest.fixture
def get_include_paths() -> MagicMock:
    ...

@pytest.fixture
def config_postgres() -> MagicMock:
    ...

@pytest.fixture
def manifest_fx(config_postgres: MagicMock) -> MagicMock:
    ...

@pytest.fixture
def postgres_adapter(config_postgres: MagicMock, get_adapter: MagicMock) -> PostgresAdapter:
    ...

def test_query_header_context(config: MagicMock, manifest: MagicMock) -> Dict[str, Any]:
    ...

def test_macro_runtime_context(config: MagicMock, manifest: MagicMock, get_adapter: MagicMock, get_include_paths: MagicMock) -> Dict[str, Any]:
    ...

def test_invocation_args_to_dict_in_macro_runtime_context(config: MagicMock, manifest: MagicMock, get_adapter: MagicMock, get_include_paths: MagicMock) -> Dict[str, Any]:
    ...

def test_model_parse_context(config: MagicMock, manifest: MagicMock, get_adapter: MagicMock, get_include_paths: MagicMock) -> Dict[str, Any]:
    ...

def test_model_runtime_context(config: MagicMock, manifest: MagicMock, get_adapter: MagicMock, get_include_paths: MagicMock) -> Dict[str, Any]:
    ...

def test_docs_runtime_context(config: MagicMock) -> Dict[str, Any]:
    ...

def test_macro_namespace_duplicates(config: MagicMock, manifest: MagicMock) -> None:
    ...

def test_macro_namespace(config: MagicMock, manifest: MagicMock) -> None:
    ...

def test_dbt_metadata_envs(monkeypatch: Any, config: MagicMock, manifest: MagicMock, get_adapter: MagicMock, get_include_paths: MagicMock) -> Dict[str, Any]:
    ...

def test_unit_test_runtime_context(config: MagicMock, manifest: MagicMock, get_adapter: MagicMock, get_include_paths: MagicMock) -> Dict[str, Any]:
    ...

def test_unit_test_runtime_context_macro_overrides_global(config: MagicMock, manifest: MagicMock, get_adapter: MagicMock, get_include_paths: MagicMock) -> Dict[str, Any]:
    ...

def test_unit_test_runtime_context_macro_overrides_package(config: MagicMock, manifest: MagicMock, get_adapter: MagicMock, get_include_paths: MagicMock) -> Dict[str, Any]:
    ...

@pytest.mark.parametrize('overrides,expected_override_value', [({'some_macro': 'override'}, 'override'), ({'dbt.some_macro': 'override'}, 'override'), ({'some_macro': 'dbt_global_override', 'dbt.some_macro': 'dbt_namespaced_override'}, 'dbt_global_override'), ({'dbt.some_macro': 'dbt_namespaced_override', 'some_macro': 'dbt_global_override'}, 'dbt_global_override')])
def test_unit_test_runtime_context_macro_overrides_dbt_macro(overrides: Dict[str, str], expected_override_value: str, config: MagicMock, manifest: MagicMock, get_adapter: MagicMock, get_include_paths: MagicMock) -> None:
    ...