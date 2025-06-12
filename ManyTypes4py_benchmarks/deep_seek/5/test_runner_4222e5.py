import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import ANY, Mock, patch
import pytest
from dbt.artifacts.resources.types import NodeType
from dbt.artifacts.schemas.catalog import CatalogArtifact
from dbt.artifacts.schemas.results import RunStatus, TestStatus
from dbt.artifacts.schemas.run import RunExecutionResult
from dbt.cli.main import dbtRunnerResult
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import ManifestNode
from dbt_common.events.base_types import EventLevel
from prefect_dbt.core.runner import PrefectDbtRunner
from prefect_dbt.core.settings import PrefectDbtSettings
from prefect import flow
from prefect.events.schemas.events import RelatedResource

@pytest.fixture
def mock_dbt_runner() -> Mock:
    with patch('prefect_dbt.core.runner.dbtRunner') as mock_runner:
        mock_instance = Mock()
        mock_runner.return_value = mock_instance
        result = Mock(spec=dbtRunnerResult)
        result.success = True
        manifest_mock = Mock(spec=Manifest)
        manifest_mock.metadata = Mock()
        manifest_mock.metadata.adapter_type = 'postgres'
        result.result = manifest_mock
        mock_instance.invoke.return_value = result
        yield mock_instance

@pytest.fixture
def settings() -> PrefectDbtSettings:
    return PrefectDbtSettings(
        profiles_dir=Path('./tests/dbt_configs').resolve(),
        project_dir=Path('./tests/dbt_configs').resolve(),
        log_level=EventLevel.DEBUG
    )

@pytest.fixture
def mock_manifest() -> Mock:
    """Create a mock manifest with different node types."""
    manifest = Mock(spec=Manifest)
    manifest.metadata = Mock()
    manifest.metadata.adapter_type = 'postgres'
    return manifest

@pytest.fixture
def mock_nodes() -> Dict[str, Mock]:
    """Create mock nodes of different types."""
    model_node = Mock()
    model_node.relation_name = '"schema"."model_table"'
    model_node.depends_on_nodes = []
    model_node.config.meta = {}
    model_node.resource_type = NodeType.Model
    model_node.unique_id = 'model.test.model'
    model_node.name = 'model'
    
    seed_node = Mock()
    seed_node.relation_name = '"schema"."seed_table"'
    seed_node.depends_on_nodes = []
    seed_node.config.meta = {}
    seed_node.resource_type = NodeType.Seed
    seed_node.unique_id = 'seed.test.seed'
    seed_node.name = 'seed'
    
    exposure_node = Mock()
    exposure_node.relation_name = '"schema"."exposure_table"'
    exposure_node.depends_on_nodes = []
    exposure_node.config.meta = {}
    exposure_node.resource_type = NodeType.Exposure
    exposure_node.unique_id = 'exposure.test.exposure'
    exposure_node.name = 'exposure'
    
    test_node = Mock()
    test_node.relation_name = '"schema"."test_table"'
    test_node.depends_on_nodes = []
    test_node.config.meta = {}
    test_node.resource_type = NodeType.Test
    test_node.unique_id = 'test.test.test'
    test_node.name = 'test'
    
    return {
        'model': model_node,
        'seed': seed_node,
        'exposure': exposure_node,
        'test': test_node
    }

@pytest.fixture
def mock_manifest_with_nodes(mock_manifest: Mock, mock_nodes: Dict[str, Mock]) -> Mock:
    """Create a mock manifest populated with test nodes."""
    mock_manifest.nodes = {
        'model.test.model': mock_nodes['model'],
        'seed.test.seed': mock_nodes['seed'],
        'exposure.test.exposure': mock_nodes['exposure'],
        'test.test.test': mock_nodes['test']
    }
    return mock_manifest

class TestPrefectDbtRunnerInitialization:

    def test_runner_initialization(self, settings: PrefectDbtSettings) -> None:
        manifest = Mock()
        client = Mock()
        runner = PrefectDbtRunner(manifest=manifest, settings=settings, client=client)
        assert runner.manifest == manifest
        assert runner.settings == settings
        assert runner.client == client

    def test_runner_default_initialization(self) -> None:
        runner = PrefectDbtRunner()
        assert runner.manifest is None
        assert isinstance(runner.settings, PrefectDbtSettings)
        assert runner.client is not None

class TestPrefectDbtRunnerParse:

    def test_parse_success(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)
        runner.parse()
        mock_dbt_runner.invoke.assert_called_once_with(
            ['parse'],
            profiles_dir=ANY,
            project_dir=settings.project_dir,
            log_level=EventLevel.DEBUG
        )
        assert runner.manifest is not None

    def test_parse_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        mock_dbt_runner.invoke.return_value.success = False
        mock_dbt_runner.invoke.return_value.exception = 'Parse error'
        runner = PrefectDbtRunner(settings=settings)
        with pytest.raises(ValueError, match='Failed to load manifest'):
            runner.parse()

class TestPrefectDbtRunnerInvoke:

    def test_invoke_with_custom_kwargs(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)
        runner.invoke(['run'], vars={'custom_var': 'value'}, threads=4)
        call_kwargs = mock_dbt_runner.invoke.call_args.kwargs
        assert call_kwargs['vars'] == {'custom_var': 'value'}
        assert call_kwargs['threads'] == 4
        assert call_kwargs['profiles_dir'] == ANY
        assert call_kwargs['project_dir'] == settings.project_dir

    @pytest.mark.asyncio
    async def test_ainvoke_with_custom_kwargs(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)
        await runner.ainvoke(['run'], vars={'custom_var': 'value'}, threads=4)
        call_kwargs = mock_dbt_runner.invoke.call_args.kwargs
        assert call_kwargs['vars'] == {'custom_var': 'value'}
        assert call_kwargs['threads'] == 4
        assert call_kwargs['profiles_dir'] == ANY
        assert call_kwargs['project_dir'] == settings.project_dir

    def test_invoke_with_parsing(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)
        runner.invoke(['run'])
        assert mock_dbt_runner.invoke.call_count == 2
        parse_call = mock_dbt_runner.invoke.call_args_list[0]
        assert parse_call.args[0] == ['parse']
        run_call = mock_dbt_runner.invoke.call_args_list[1]
        assert run_call.args[0] == ['run']

    @pytest.mark.asyncio
    async def test_ainvoke_with_parsing(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)
        await runner.ainvoke(['run'])
        assert mock_dbt_runner.invoke.call_count == 2
        parse_call = mock_dbt_runner.invoke.call_args_list[0]
        assert parse_call.args[0] == ['parse']
        run_call = mock_dbt_runner.invoke.call_args_list[1]
        assert run_call.args[0] == ['run']

    def test_invoke_raises_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None
        run_result.result = Mock(spec=RunExecutionResult)
        node_result = Mock()
        node_result.status = RunStatus.Error
        node_result.node.resource_type = NodeType.Model
        node_result.node.name = 'failed_model'
        node_result.message = 'Something went wrong'
        run_result.result.results = [node_result]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ['parse']:
                return parse_result
            elif args == ['run']:
                return run_result
            return Mock()
        
        mock_dbt_runner.invoke.side_effect = side_effect
        with pytest.raises(ValueError, match="Failures detected during invocation of dbt command 'run'"):
            runner.invoke(['run'])

    @pytest.mark.asyncio
    async def test_ainvoke_raises_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None
        run_result.result = Mock(spec=RunExecutionResult)
        node_result = Mock()
        node_result.status = RunStatus.Error
        node_result.node.resource_type = NodeType.Model
        node_result.node.name = 'failed_model'
        node_result.message = 'Something went wrong'
        run_result.result.results = [node_result]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ['parse']:
                return parse_result
            elif args == ['run']:
                return run_result
            return Mock()
        
        mock_dbt_runner.invoke.side_effect = side_effect
        with pytest.raises(ValueError, match="Failures detected during invocation of dbt command 'run'"):
            await runner.ainvoke(['run'])

    def test_invoke_multiple_failures(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None
        run_result.result = Mock(spec=RunExecutionResult)
        failed_model = Mock()
        failed_model.status = RunStatus.Error
        failed_model.node.resource_type = NodeType.Model
        failed_model.node.name = 'failed_model'
        failed_model.message = 'Model error'
        failed_test = Mock()
        failed_test.status = TestStatus.Fail
        failed_test.node.resource_type = NodeType.Test
        failed_test.node.name = 'failed_test'
        failed_test.message = 'Test failed'
        run_result.result.results = [failed_model, failed_test]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ['parse']:
                return parse_result
            elif args == ['run']:
                return run_result
            return Mock()
        
        mock_dbt_runner.invoke.side_effect = side_effect
        with pytest.raises(ValueError) as exc_info:
            runner.invoke(['run'])
        error_message = str(exc_info.value)
        assert 'Model failed_model errored with message: "Model error"' in error_message
        assert 'Test failed_test failed with message: "Test failed"' in error_message

    @pytest.mark.asyncio
    async def test_ainvoke_multiple_failures(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None
        run_result.result = Mock(spec=RunExecutionResult)
        failed_model = Mock()
        failed_model.status = RunStatus.Error
        failed_model.node.resource_type = NodeType.Model
        failed_model.node.name = 'failed_model'
        failed_model.message = 'Model error'
        failed_test = Mock()
        failed_test.status = TestStatus.Fail
        failed_test.node.resource_type = NodeType.Test
        failed_test.node.name = 'failed_test'
        failed_test.message = 'Test failed'
        run_result.result.results = [failed_model, failed_test]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ['parse']:
                return parse_result
            elif args == ['run']:
                return run_result
            return Mock()
        
        mock_dbt_runner.invoke.side_effect = side_effect
        with pytest.raises(ValueError) as exc_info:
            await runner.ainvoke(['run'])
        error_message = str(exc_info.value)
        assert 'Model failed_model errored with message: "Model error"' in error_message
        assert 'Test failed_test failed with message: "Test failed"' in error_message

    def test_invoke_no_raise_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings, raise_on_failure=False)
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None
        node_result = Mock()
        node_result.status = RunStatus.Error
        node_result.node.resource_type = NodeType.Model
        node_result.node.name = 'failed_model'
        node_result.message = 'Something went wrong'
        run_result.result.results = [node_result]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ['parse']:
                return parse_result
            elif args == ['run']:
                return run_result
            return Mock()
        
        mock_dbt_runner.invoke.side_effect = side_effect
        result = runner.invoke(['run'])
        assert result.success is False

    @pytest.mark.asyncio
    async def test_ainvoke_no_raise_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings, raise_on_failure=False)
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None
        node_result = Mock()
        node_result.status = RunStatus.Error
        node_result.node.resource_type = NodeType.Model
        node_result.node.name = 'failed_model'
        node_result.message = 'Something went wrong'
        run_result.result.results = [node_result]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ['parse']:
                return parse_result
            elif args == ['run']:
                return run_result
            return Mock()
        
        mock_dbt_runner.invoke.side_effect = side_effect
        result = await runner.ainvoke(['run'])
        assert result.success is False

    @pytest.mark.parametrize('command,expected_type,requires_manifest', [
        (['run'], RunExecutionResult, True),
        (['test'], RunExecutionResult, True),
        (['seed'], RunExecutionResult, True),
        (['snapshot'], RunExecutionResult, True),
        (['build'], RunExecutionResult, True),
        (['compile'], RunExecutionResult, True),
        (['run-operation'], RunExecutionResult, True),
        (['parse'], Manifest, False),
        (['docs', 'generate'], CatalogArtifact, True),
        (['list'], list, True),
        (['ls'], list, True),
        (['debug'], bool, False),
        (['clean'], None, False),
        (['deps'], None, False),
        (['init'], None, False),
        (['source'], None, True)
    ])
    def test_invoke_command_return_types(
        self,
        mock_dbt_runner: Mock,
        settings: PrefectDbtSettings,
        command: List[str],
        expected_type: Type[Any],
        requires_manifest: bool
    ) -> None:
        """Test that different dbt commands return the expected result types."""
        runner = PrefectDbtRunner(settings=settings)
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest = Mock(spec=Manifest)
        manifest.metadata = Mock()
        manifest.metadata.adapter_type = 'postgres'
        parse_result.result = manifest
        command_result = Mock(spec=dbtRunnerResult)
        command_result.success = True
        command_result.exception = None
        
        if expected_type is None:
            command_result.result = None
        elif expected_type is bool:
            command_result.result = True
        elif expected_type is list:
            command_result.result = ['item1', 'item2']
        else:
            command_result.result = Mock(spec=expected_type)

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ['parse']