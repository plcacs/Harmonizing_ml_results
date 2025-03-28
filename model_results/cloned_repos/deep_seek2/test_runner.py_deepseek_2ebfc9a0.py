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
    with patch("prefect_dbt.core.runner.dbtRunner") as mock_runner:
        mock_instance = Mock()
        mock_runner.return_value = mock_instance

        # Setup default successful result
        result = Mock(spec=dbtRunnerResult)
        result.success = True
        # Create a Mock that inherits from Manifest
        manifest_mock = Mock(spec=Manifest)
        manifest_mock.metadata = Mock()
        manifest_mock.metadata.adapter_type = "postgres"
        result.result = manifest_mock
        mock_instance.invoke.return_value = result

        yield mock_instance


@pytest.fixture
def settings() -> PrefectDbtSettings:
    return PrefectDbtSettings(
        profiles_dir=Path("./tests/dbt_configs").resolve(),
        project_dir=Path("./tests/dbt_configs").resolve(),
        log_level=EventLevel.DEBUG,
    )


@pytest.fixture
def mock_manifest() -> Mock:
    """Create a mock manifest with different node types."""
    manifest = Mock(spec=Manifest)
    # Create the metadata structure
    manifest.metadata = Mock()
    manifest.metadata.adapter_type = "postgres"
    return manifest


@pytest.fixture
def mock_nodes() -> Dict[str, Mock]:
    """Create mock nodes of different types."""
    model_node = Mock()
    model_node.relation_name = '"schema"."model_table"'
    model_node.depends_on_nodes = []
    model_node.config.meta = {}
    model_node.resource_type = NodeType.Model
    model_node.unique_id = "model.test.model"
    model_node.name = "model"

    seed_node = Mock()
    seed_node.relation_name = '"schema"."seed_table"'
    seed_node.depends_on_nodes = []
    seed_node.config.meta = {}
    seed_node.resource_type = NodeType.Seed
    seed_node.unique_id = "seed.test.seed"
    seed_node.name = "seed"

    exposure_node = Mock()
    exposure_node.relation_name = '"schema"."exposure_table"'
    exposure_node.depends_on_nodes = []
    exposure_node.config.meta = {}
    exposure_node.resource_type = NodeType.Exposure
    exposure_node.unique_id = "exposure.test.exposure"
    exposure_node.name = "exposure"

    test_node = Mock()
    test_node.relation_name = '"schema"."test_table"'
    test_node.depends_on_nodes = []
    test_node.config.meta = {}
    test_node.resource_type = NodeType.Test
    test_node.unique_id = "test.test.test"
    test_node.name = "test"

    return {
        "model": model_node,
        "seed": seed_node,
        "exposure": exposure_node,
        "test": test_node,
    }


@pytest.fixture
def mock_manifest_with_nodes(mock_manifest: Mock, mock_nodes: Dict[str, Mock]) -> Mock:
    """Create a mock manifest populated with test nodes."""
    mock_manifest.nodes = {
        "model.test.model": mock_nodes["model"],
        "seed.test.seed": mock_nodes["seed"],
        "exposure.test.exposure": mock_nodes["exposure"],
        "test.test.test": mock_nodes["test"],
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
            ["parse"],
            profiles_dir=ANY,
            project_dir=settings.project_dir,
            log_level=EventLevel.DEBUG,
        )
        assert runner.manifest is not None

    def test_parse_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        mock_dbt_runner.invoke.return_value.success = False
        mock_dbt_runner.invoke.return_value.exception = "Parse error"

        runner = PrefectDbtRunner(settings=settings)

        with pytest.raises(ValueError, match="Failed to load manifest"):
            runner.parse()


class TestPrefectDbtRunnerInvoke:
    # Basic invocation tests
    def test_invoke_with_custom_kwargs(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)

        runner.invoke(["run"], vars={"custom_var": "value"}, threads=4)

        call_kwargs = mock_dbt_runner.invoke.call_args.kwargs
        assert call_kwargs["vars"] == {"custom_var": "value"}
        assert call_kwargs["threads"] == 4
        assert call_kwargs["profiles_dir"] == ANY
        assert call_kwargs["project_dir"] == settings.project_dir

    @pytest.mark.asyncio
    async def test_ainvoke_with_custom_kwargs(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)

        await runner.ainvoke(["run"], vars={"custom_var": "value"}, threads=4)

        call_kwargs = mock_dbt_runner.invoke.call_args.kwargs
        assert call_kwargs["vars"] == {"custom_var": "value"}
        assert call_kwargs["threads"] == 4
        assert call_kwargs["profiles_dir"] == ANY
        assert call_kwargs["project_dir"] == settings.project_dir

    # Parsing tests
    def test_invoke_with_parsing(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)

        runner.invoke(["run"])

        assert mock_dbt_runner.invoke.call_count == 2
        parse_call = mock_dbt_runner.invoke.call_args_list[0]
        assert parse_call.args[0] == ["parse"]

        run_call = mock_dbt_runner.invoke.call_args_list[1]
        assert run_call.args[0] == ["run"]

    @pytest.mark.asyncio
    async def test_ainvoke_with_parsing(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)

        await runner.ainvoke(["run"])

        assert mock_dbt_runner.invoke.call_count == 2
        parse_call = mock_dbt_runner.invoke.call_args_list[0]
        assert parse_call.args[0] == ["parse"]

        run_call = mock_dbt_runner.invoke.call_args_list[1]
        assert run_call.args[0] == ["run"]

    # Single failure tests
    def test_invoke_raises_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)

        # Setup successful parse result first
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock

        # Mock a failed run result
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None
        run_result.result = Mock(spec=RunExecutionResult)

        # Create a failed node result
        node_result = Mock()
        node_result.status = RunStatus.Error
        node_result.node.resource_type = NodeType.Model
        node_result.node.name = "failed_model"
        node_result.message = "Something went wrong"

        run_result.result.results = [node_result]

        # Set up the side effects to return parse_result for parse command and run_result for run command
        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ["parse"]:
                return parse_result
            elif args == ["run"]:
                return run_result
            return Mock()

        mock_dbt_runner.invoke.side_effect = side_effect

        with pytest.raises(
            ValueError, match="Failures detected during invocation of dbt command 'run'"
        ):
            runner.invoke(["run"])

    @pytest.mark.asyncio
    async def test_ainvoke_raises_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)

        # Setup successful parse result first
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock

        # Mock a failed result
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None
        run_result.result = Mock(spec=RunExecutionResult)
        # Create a failed node result
        node_result = Mock()
        node_result.status = RunStatus.Error
        node_result.node.resource_type = NodeType.Model
        node_result.node.name = "failed_model"
        node_result.message = "Something went wrong"

        run_result.result.results = [node_result]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ["parse"]:
                return parse_result
            elif args == ["run"]:
                return run_result
            return Mock()

        mock_dbt_runner.invoke.side_effect = side_effect

        with pytest.raises(
            ValueError, match="Failures detected during invocation of dbt command 'run'"
        ):
            await runner.ainvoke(["run"])

    # Multiple failures tests
    def test_invoke_multiple_failures(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)

        # Setup successful parse result first
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock

        # Mock a failed result with multiple failures
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None
        run_result.result = Mock(spec=RunExecutionResult)

        # Create multiple failed node results
        failed_model = Mock()
        failed_model.status = RunStatus.Error
        failed_model.node.resource_type = NodeType.Model
        failed_model.node.name = "failed_model"
        failed_model.message = "Model error"

        failed_test = Mock()
        failed_test.status = TestStatus.Fail
        failed_test.node.resource_type = NodeType.Test
        failed_test.node.name = "failed_test"
        failed_test.message = "Test failed"

        run_result.result.results = [failed_model, failed_test]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ["parse"]:
                return parse_result
            elif args == ["run"]:
                return run_result
            return Mock()

        mock_dbt_runner.invoke.side_effect = side_effect

        with pytest.raises(ValueError) as exc_info:
            runner.invoke(["run"])

        error_message = str(exc_info.value)
        assert 'Model failed_model errored with message: "Model error"' in error_message
        assert 'Test failed_test failed with message: "Test failed"' in error_message

    @pytest.mark.asyncio
    async def test_ainvoke_multiple_failures(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings)

        # Setup successful parse result first
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock

        # Mock a failed result with multiple failures
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None
        run_result.result = Mock(spec=RunExecutionResult)

        # Create multiple failed node results
        failed_model = Mock()
        failed_model.status = RunStatus.Error
        failed_model.node.resource_type = NodeType.Model
        failed_model.node.name = "failed_model"
        failed_model.message = "Model error"

        failed_test = Mock()
        failed_test.status = TestStatus.Fail
        failed_test.node.resource_type = NodeType.Test
        failed_test.node.name = "failed_test"
        failed_test.message = "Test failed"

        run_result.result.results = [failed_model, failed_test]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ["parse"]:
                return parse_result
            elif args == ["run"]:
                return run_result
            return Mock()

        mock_dbt_runner.invoke.side_effect = side_effect

        with pytest.raises(ValueError) as exc_info:
            await runner.ainvoke(["run"])

        error_message = str(exc_info.value)
        assert 'Model failed_model errored with message: "Model error"' in error_message
        assert 'Test failed_test failed with message: "Test failed"' in error_message

    # No raise on failure tests
    def test_invoke_no_raise_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings, raise_on_failure=False)

        # Setup successful parse result first
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock

        # Mock a failed result
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None

        # Create a failed node result
        node_result = Mock()
        node_result.status = RunStatus.Error
        node_result.node.resource_type = NodeType.Model
        node_result.node.name = "failed_model"
        node_result.message = "Something went wrong"

        run_result.result.results = [node_result]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ["parse"]:
                return parse_result
            elif args == ["run"]:
                return run_result
            return Mock()

        mock_dbt_runner.invoke.side_effect = side_effect

        # Should not raise an exception
        result = runner.invoke(["run"])
        assert result.success is False

    @pytest.mark.asyncio
    async def test_ainvoke_no_raise_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        runner = PrefectDbtRunner(settings=settings, raise_on_failure=False)

        # Setup successful parse result first
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest_mock = Mock(Manifest)
        parse_result.result = manifest_mock

        # Mock a failed result
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = False
        run_result.exception = None

        # Create a failed node result
        node_result = Mock()
        node_result.status = RunStatus.Error
        node_result.node.resource_type = NodeType.Model
        node_result.node.name = "failed_model"
        node_result.message = "Something went wrong"

        run_result.result.results = [node_result]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ["parse"]:
                return parse_result
            elif args == ["run"]:
                return run_result
            return Mock()

        mock_dbt_runner.invoke.side_effect = side_effect

        # Should not raise an exception
        result = await runner.ainvoke(["run"])
        assert result.success is False

    @pytest.mark.parametrize(
        "command,expected_type,requires_manifest",
        [
            (["run"], RunExecutionResult, True),
            (["test"], RunExecutionResult, True),
            (["seed"], RunExecutionResult, True),
            (["snapshot"], RunExecutionResult, True),
            (["build"], RunExecutionResult, True),
            (["compile"], RunExecutionResult, True),
            (["run-operation"], RunExecutionResult, True),
            (["parse"], Manifest, False),
            (["docs", "generate"], CatalogArtifact, True),
            (["list"], list, True),
            (["ls"], list, True),
            (["debug"], bool, False),
            (["clean"], None, False),
            (["deps"], None, False),
            (["init"], None, False),
            (["source"], None, True),
        ],
    )
    def test_invoke_command_return_types(
        self,
        mock_dbt_runner: Mock,
        settings: PrefectDbtSettings,
        command: List[str],
        expected_type: Optional[Type[Any]],
        requires_manifest: bool,
    ) -> None:
        """Test that different dbt commands return the expected result types."""
        runner = PrefectDbtRunner(settings=settings)

        # Mock parse result if needed
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest = Mock(spec=Manifest)  # Create the manifest
        manifest.metadata = Mock()  # Add required metadata
        manifest.metadata.adapter_type = "postgres"
        parse_result.result = manifest  # Set the actual manifest

        # Mock command result
        command_result = Mock(spec=dbtRunnerResult)
        command_result.success = True
        command_result.exception = None

        # Set appropriate result based on command
        if expected_type is None:
            command_result.result = None
        elif expected_type is bool:
            command_result.result = True
        elif expected_type is list:
            command_result.result = ["item1", "item2"]
        else:
            command_result.result = Mock(spec=expected_type)

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ["parse"]:
                return parse_result
            return command_result

        mock_dbt_runner.invoke.side_effect = side_effect

        result = runner.invoke(command)

        assert result.success
        if expected_type is None:
            assert result.result is None
        elif expected_type is bool:
            assert isinstance(result.result, bool)
        elif expected_type is list:
            assert isinstance(result.result, list)
            assert all(isinstance(item, str) for item in result.result)
        else:
            assert isinstance(result.result, expected_type)

        # Verify call count and order
        if requires_manifest:
            assert mock_dbt_runner.invoke.call_count == 2
            assert mock_dbt_runner.invoke.call_args_list[0].args[0] == ["parse"]
            assert mock_dbt_runner.invoke.call_args_list[1].args[0] == command
        else:
            mock_dbt_runner.invoke.assert_called_once()
            assert mock_dbt_runner.invoke.call_args.args[0] == command

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "command,expected_type,requires_manifest",
        [
            (["run"], RunExecutionResult, True),
            (["test"], RunExecutionResult, True),
            (["seed"], RunExecutionResult, True),
            (["snapshot"], RunExecutionResult, True),
            (["build"], RunExecutionResult, True),
            (["compile"], RunExecutionResult, True),
            (["run-operation"], RunExecutionResult, True),
            (["parse"], Manifest, False),
            (["docs", "generate"], CatalogArtifact, True),
            (["list"], list, True),
            (["ls"], list, True),
            (["debug"], bool, False),
            (["clean"], None, False),
            (["deps"], None, False),
            (["init"], None, False),
            (["source"], None, True),
        ],
    )
    async def test_ainvoke_command_return_types(
        self,
        mock_dbt_runner: Mock,
        settings: PrefectDbtSettings,
        command: List[str],
        expected_type: Optional[Type[Any]],
        requires_manifest: bool,
    ) -> None:
        """Test that different dbt commands return the expected result types when called asynchronously."""
        runner = PrefectDbtRunner(settings=settings)

        # Mock parse result if needed
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        parse_result.result = Mock(spec=Manifest)

        # Mock command result
        command_result = Mock(spec=dbtRunnerResult)
        command_result.success = True
        command_result.exception = None

        # Set appropriate result based on command
        if expected_type is None:
            command_result.result = None
        elif expected_type is bool:
            command_result.result = True
        elif expected_type is list:
            command_result.result = ["item1", "item2"]
        else:
            command_result.result = Mock(spec=expected_type)

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ["parse"]:
                return parse_result
            return command_result

        mock_dbt_runner.invoke.side_effect = side_effect

        result = await runner.ainvoke(command)

        assert result.success
        if expected_type is None:
            assert result.result is None
        elif expected_type is bool:
            assert isinstance(result.result, bool)
        elif expected_type is list:
            assert isinstance(result.result, list)
            assert all(isinstance(item, str) for item in result.result)
        else:
            assert isinstance(result.result, expected_type)

        # Verify call count and order
        if requires_manifest:
            assert mock_dbt_runner.invoke.call_count == 2
            assert mock_dbt_runner.invoke.call_args_list[0].args[0] == ["parse"]
            assert mock_dbt_runner.invoke.call_args_list[1].args[0] == command
        else:
            mock_dbt_runner.invoke.assert_called_once()
            assert mock_dbt_runner.invoke.call_args.args[0] == command

    def test_invoke_with_manifest_requiring_commands(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        """Test that commands requiring manifest trigger parse if manifest not provided."""
        runner = PrefectDbtRunner(settings=settings)

        # Mock parse result
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        parse_result.result = Mock(spec=Manifest)

        # Mock run result
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = True
        run_result.result = Mock(spec=RunExecutionResult)

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ["parse"]:
                return parse_result
            return run_result

        mock_dbt_runner.invoke.side_effect = side_effect

        # Test with command that requires manifest
        runner.invoke(["run"])
        assert mock_dbt_runner.invoke.call_count == 2
        assert mock_dbt_runner.invoke.call_args_list[0].args[0] == ["parse"]
        assert mock_dbt_runner.invoke.call_args_list[1].args[0] == ["run"]

        # Reset mock
        mock_dbt_runner.invoke.reset_mock()
        mock_dbt_runner.invoke.side_effect = side_effect

        # Test with command that doesn't require manifest
        runner.invoke(["clean"])
        assert mock_dbt_runner.invoke.call_count == 1
        assert mock_dbt_runner.invoke.call_args_list[0].args[0] == ["clean"]

    def test_invoke_with_preloaded_manifest(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        """Test that commands don't trigger parse when manifest is preloaded."""
        manifest = Mock(spec=Manifest)
        runner = PrefectDbtRunner(settings=settings, manifest=manifest)

        # Mock run result
        run_result = Mock(spec=dbtRunnerResult)
        run_result.success = True
        run_result.result = Mock(spec=RunExecutionResult)
        mock_dbt_runner.invoke.return_value = run_result

        # Test with command that normally requires manifest
        runner.invoke(["run"])

        # Should not call parse since manifest was preloaded
        mock_dbt_runner.invoke.assert_called_once()
        assert mock_dbt_runner.invoke.call_args.args[0] == ["run"]

    def test_invoke_debug_command(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        """Test the dbt debug command which has unique behavior."""
        runner = PrefectDbtRunner(settings=settings)

        # Mock debug result
        debug_result = Mock(spec=dbtRunnerResult)
        debug_result.success = True
        # debug command doesn't return a specific result type
        debug_result.result = None
        mock_dbt_runner.invoke.return_value = debug_result

        result = runner.invoke(["debug"])

        assert result.success
        assert result.result is None
        mock_dbt_runner.invoke.assert_called_once_with(
            ["debug"],
            project_dir=settings.project_dir,
            profiles_dir=ANY,
            log_level=ANY,
        )

    @pytest.mark.parametrize(
        "command,expected_type",
        [
            (["run"], RunExecutionResult),
            (["test"], RunExecutionResult),
            (["seed"], RunExecutionResult),
            (["snapshot"], RunExecutionResult),
            (["build"], RunExecutionResult),
            (["compile"], RunExecutionResult),
            (["run-operation"], RunExecutionResult),
            (["parse"], Manifest),
            (["docs", "generate"], CatalogArtifact),
            (["list"], list),
            (["ls"], list),
            (["debug"], bool),
            (["clean"], type(None)),
            (["deps"], type(None)),
            (["init"], type(None)),
            (["source"], type(None)),
        ],
    )
    def test_failure_result_types(
        self,
        mock_dbt_runner: Mock,
        settings: PrefectDbtSettings,
        command: List[str],
        expected_type: Optional[Type[Any]],
    ) -> None:
        """Test that failed commands return the expected result types."""
        runner = PrefectDbtRunner(settings=settings, raise_on_failure=False)

        # Mock parse result if needed for manifest loading
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest = Mock(spec=Manifest)
        manifest.metadata = Mock()
        manifest.metadata.adapter_type = "postgres"
        parse_result.result = manifest

        # Mock failed command result
        command_result = Mock(spec=dbtRunnerResult)
        command_result.success = False
        command_result.exception = None

        # Set appropriate result based on command
        if expected_type is type(None):
            command_result.result = None
        elif expected_type is bool:
            command_result.result = False
        elif expected_type is list:
            command_result.result = []
        else:
            command_result.result = Mock(spec=expected_type)
            if expected_type == RunExecutionResult:
                # Add failed results for RunExecutionResult
                failed_node = Mock()
                failed_node.status = RunStatus.Error
                failed_node.node = Mock()
                failed_node.node.resource_type = NodeType.Model
                failed_node.node.name = "failed_model"
                failed_node.message = "Test failure"
                command_result.result.results = [failed_node]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ["parse"] and command != [
                "parse"
            ]:  # Only return successful parse when loading manifest
                return parse_result
            return command_result

        mock_dbt_runner.invoke.side_effect = side_effect

        result = runner.invoke(command)

        assert not result.success
        if expected_type is type(None):
            assert result.result is None
        else:
            assert isinstance(result.result, expected_type)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "command,expected_type",
        [
            (["run"], RunExecutionResult),
            (["test"], RunExecutionResult),
            (["seed"], RunExecutionResult),
            (["snapshot"], RunExecutionResult),
            (["build"], RunExecutionResult),
            (["compile"], RunExecutionResult),
            (["run-operation"], RunExecutionResult),
            (["parse"], Manifest),
            (["docs", "generate"], CatalogArtifact),
            (["list"], list),
            (["ls"], list),
            (["debug"], bool),
            (["clean"], type(None)),
            (["deps"], type(None)),
            (["init"], type(None)),
            (["source"], type(None)),
        ],
    )
    async def test_failure_result_types_async(
        self,
        mock_dbt_runner: Mock,
        settings: PrefectDbtSettings,
        command: List[str],
        expected_type: Optional[Type[Any]],
    ) -> None:
        """Test that failed commands return the expected result types when called asynchronously."""
        runner = PrefectDbtRunner(settings=settings, raise_on_failure=False)

        # Mock parse result if needed for manifest loading
        parse_result = Mock(spec=dbtRunnerResult)
        parse_result.success = True
        manifest = Mock(spec=Manifest)
        parse_result.result = manifest

        # Mock failed command result
        command_result = Mock(spec=dbtRunnerResult)
        command_result.success = False
        command_result.exception = None

        # Set appropriate result based on command
        if expected_type is type(None):
            command_result.result = None
        elif expected_type is bool:
            command_result.result = False
        elif expected_type is list:
            command_result.result = []
        else:
            command_result.result = Mock(spec=expected_type)
            if expected_type == RunExecutionResult:
                # Add failed results for RunExecutionResult
                failed_node = Mock()
                failed_node.status = RunStatus.Error
                failed_node.node = Mock()
                failed_node.node.resource_type = NodeType.Model
                failed_node.node.name = "failed_model"
                failed_node.message = "Test failure"
                command_result.result.results = [failed_node]

        def side_effect(args: List[str], **kwargs: Any) -> Mock:
            if args == ["parse"] and command != [
                "parse"
            ]:  # Only return successful parse when loading manifest
                return parse_result
            return command_result

        mock_dbt_runner.invoke.side_effect = side_effect

        result = await runner.ainvoke(command)

        assert not result.success
        if expected_type is type(None):
            assert result.result is None
        else:
            assert isinstance(result.result, expected_type)


class TestPrefectDbtRunnerLogging:
    def test_logging_callback(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings, caplog: pytest.LogCaptureFixture) -> None:
        runner = PrefectDbtRunner(settings=settings)

        # Create mock events for different log levels
        debug_event = Mock()
        debug_event.info.level = EventLevel.DEBUG
        debug_event.info.msg = "Debug message"

        info_event = Mock()
        info_event.info.level = EventLevel.INFO
        info_event.info.msg = "Info message"

        warn_event = Mock()
        warn_event.info.level = EventLevel.WARN
        warn_event.info.msg = "Warning message"

        error_event = Mock()
        error_event.info.level = EventLevel.ERROR
        error_event.info.msg = "Error message"

        test_event = Mock()
        test_event.info.level = EventLevel.TEST
        test_event.info.msg = "Test message"

        @flow
        def test_flow() -> None:
            callback = runner._create_logging_callback(EventLevel.DEBUG)
            callback(debug_event)
            callback(info_event)
            callback(warn_event)
            callback(error_event)
            callback(test_event)

        caplog.clear()

        with caplog.at_level(logging.DEBUG):
            test_flow()

            assert "Debug message" in caplog.text
            assert "Test message" in caplog.text
            assert "Info message" in caplog.text
            assert "Warning message" in caplog.text
            assert "Error message" in caplog.text

            for record in caplog.records:
                if (
                    "Debug message" in record.message
                    or "Test message" in record.message
                ):
                    assert record.levelno == logging.DEBUG
                elif "Info message" in record.message:
                    assert record.levelno == logging.INFO
                elif "Warning message" in record.message:
                    assert record.levelno == logging.WARNING
                elif "Error message" in record.message:
                    assert record.levelno == logging.ERROR

    def test_logging_callback_no_flow(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings, caplog: pytest.LogCaptureFixture) -> None:
        runner = PrefectDbtRunner(settings=settings)

        event = Mock()
        event.info.level = EventLevel.INFO
        event.info.msg = "Test message"

        # no flow decorator
        def test_flow() -> None:
            callback = runner._create_logging_callback(EventLevel.DEBUG)
            callback(event)

        caplog.clear()

        with caplog.at_level(logging.INFO):
            test_flow()
            assert caplog.text == "", (
                "Expected empty log output when not in flow context"
            )


class TestPrefectDbtRunnerEvents:
    def test_events_callback_node_finished(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        """Test that the events callback correctly handles NodeFinished events."""
        runner = PrefectDbtRunner(settings=settings)
        runner.manifest = Mock()
        runner.manifest.metadata.adapter_type = "postgres"

        # Create a mock node in the manifest
        node = Mock(spec=ManifestNode)
        node.unique_id = "model.test.example"
        node.name = "example"
        node.relation_name = '"schema"."example_table"'
        node.depends_on_nodes = ["model.test.upstream"]
        node.config = Mock()
        node.config.meta = {
            "prefect