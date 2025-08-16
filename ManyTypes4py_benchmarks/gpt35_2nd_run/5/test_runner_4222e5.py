from pathlib import Path
from unittest.mock import ANY, Mock, patch
from typing import Dict, List, Union

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

def mock_dbt_runner() -> Mock:
    pass

def settings() -> PrefectDbtSettings:
    pass

def mock_manifest() -> Mock:
    pass

def mock_nodes() -> Dict[str, Mock]:
    pass

def mock_manifest_with_nodes(mock_manifest: Mock, mock_nodes: Dict[str, Mock]) -> Mock:
    pass

class TestPrefectDbtRunnerInitialization:

    def test_runner_initialization(self, settings: PrefectDbtSettings) -> None:
        pass

    def test_runner_default_initialization(self) -> None:
        pass

class TestPrefectDbtRunnerParse:

    def test_parse_success(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_parse_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

class TestPrefectDbtRunnerInvoke:

    def test_invoke_with_custom_kwargs(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    async def test_ainvoke_with_custom_kwargs(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_invoke_with_parsing(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    async def test_ainvoke_with_parsing(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_invoke_raises_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    async def test_ainvoke_raises_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_invoke_multiple_failures(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    async def test_ainvoke_multiple_failures(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_invoke_no_raise_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    async def test_ainvoke_no_raise_on_failure(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_invoke_command_return_types(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings, command: List[str], expected_type: Union[type, None], requires_manifest: bool) -> None:
        pass

    async def test_ainvoke_command_return_types(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings, command: List[str], expected_type: Union[type, None], requires_manifest: bool) -> None:
        pass

    def test_invoke_with_manifest_requiring_commands(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_invoke_with_preloaded_manifest(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_invoke_debug_command(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_failure_result_types(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings, command: List[str], expected_type: Union[type, None]) -> None:
        pass

    async def test_failure_result_types_async(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings, command: List[str], expected_type: Union[type, None]) -> None:
        pass

class TestPrefectDbtRunnerLogging:

    def test_logging_callback(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings, caplog: Any) -> None:
        pass

    def test_logging_callback_no_flow(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings, caplog: Any) -> None:
        pass

class TestPrefectDbtRunnerEvents:

    def test_events_callback_node_finished(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_events_callback_with_emit_events_false(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_events_callback_with_emit_node_events_false(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_events_callback_with_emit_lineage_events_false(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

    def test_events_callback_with_all_events_disabled(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings) -> None:
        pass

class TestPrefectDbtRunnerLineage:

    def test_emit_lineage_events(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings, mock_manifest_with_nodes: Mock, provide_manifest: bool) -> None:
        pass

    async def test_aemit_lineage_events(self, mock_dbt_runner: Mock, settings: PrefectDbtSettings, mock_manifest_with_nodes: Mock, provide_manifest: bool) -> None:
        pass
