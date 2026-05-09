from __future__ import annotations
from argparse import Namespace
from typing import Any, Optional, Dict, List, Union, Tuple, Set
from unittest.mock import MagicMock
from pytest import fixture, mark
from pytest_mock import MockerFixture
from dbt.adapters.postgres import PostgresAdapter
from dbt.artifacts.resources.base import FileHash
from dbt.config import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest, ManifestStateCheck
from dbt.events.types import InvalidConcurrentBatchesConfig, UnusedResourceConfigPath
from dbt.flags import set_from_args
from dbt.parser.manifest import ManifestLoader, _warn_for_unused_resource_config_paths
from dbt.parser.read_files import FileDiff
from dbt.tracking import User
from tests.unit.fixtures import model_node
from tests.utils import EventCatcher

class TestPartialParse:
    @patch('dbt.parser.manifest.ManifestLoader.build_manifest_state_check')
    @patch('dbt.parser.manifest.os.path.exists')
    @patch('dbt.parser.manifest.open')
    def test_partial_parse_file_path(self, patched_open: Any, patched_os_exist: Any, patched_state_check: Any) -> None: ...

    def test_profile_hash_change(self, mock_project: MagicMock[RuntimeConfig]) -> None: ...

    @patch('dbt.parser.manifest.ManifestLoader.build_manifest_state_check')
    @patch('dbt.parser.manifest.os.path.exists')
    @patch('dbt.parser.manifest.open')
    def test_partial_parse_by_version(self, patched_open: Any, patched_os_exist: Any, patched_state_check: Any, runtime_config: RuntimeConfig, manifest: Manifest) -> None: ...

class TestFailedPartialParse:
    @patch('dbt.tracking.track_partial_parser')
    @patch('dbt.tracking.active_user')
    @patch('dbt.parser.manifest.PartialParsing')
    @patch('dbt.parser.manifest.ManifestLoader.read_manifest_for_partial_parse')
    @patch('dbt.parser.manifest.ManifestLoader.build_manifest_state_check')
    def test_partial_parse_safe_update_project_parser_files_partially(self, patched_state_check: Any, patched_read_manifest_for_partial_parse: Any, patched_partial_parsing: Any, patched_active_user: Any, patched_track_partial_parser: Any) -> None: ...

class TestGetFullManifest:
    @fixture
    def set_required_mocks(self, mocker: MockerFixture, manifest: Manifest, mock_adapter: MagicMock[PostgresAdapter]) -> None: ...

    def test_write_perf_info(self, mock_project: MagicMock[RuntimeConfig], mocker: MockerFixture, set_required_mocks: None) -> None: ...

    def test_reset(self, mock_project: MagicMock[RuntimeConfig], mock_adapter: MagicMock[PostgresAdapter], set_required_mocks: None) -> None: ...

    def test_partial_parse_file_diff_flag(self, mock_project: MagicMock[RuntimeConfig], mocker: MockerFixture, set_required_mocks: None) -> None: ...

class TestWarnUnusedConfigs:
    @mark.parametrize('resource_type,path,expect_used', [
        ('data_tests', 'unused_path', False), 
        ('data_tests', 'minimal', True),
        ('metrics', 'unused_path', False),
        ('metrics', 'test', True),
        ('models', 'unused_path', False),
        ('models', 'pkg', True),
        ('saved_queries', 'unused_path', False),
        ('saved_queries', 'test', True),
        ('seeds', 'unused_path', False),
        ('seeds', 'pkg', True),
        ('semantic_models', 'unused_path', False),
        ('semantic_models', 'test', True),
        ('sources', 'unused_path', False),
        ('sources', 'pkg', True),
        ('unit_tests', 'unused_path', False),
        ('unit_tests', 'pkg', True)
    ])
    def test_warn_for_unused_resource_config_paths(self, resource_type: str, path: str, expect_used: bool, manifest: Manifest, runtime_config: RuntimeConfig) -> None: ...

class TestCheckForcingConcurrentBatches:
    @fixture
    @patch('dbt.parser.manifest.ManifestLoader.build_manifest_state_check')
    @patch('dbt.parser.manifest.os.path.exists')
    @patch('dbt.parser.manifest.open')
    def manifest_loader(self, patched_open: Any, patched_os_exist: Any, patched_state_check: Any) -> ManifestLoader: ...

    @fixture
    def event_catcher(self) -> EventCatcher: ...

    @mark.parametrize('adapter_support,concurrent_batches_config,expect_warning', [
        (False, True, True),
        (False, False, False),
        (False, None, False),
        (True, True, False),
        (True, False, False),
        (True, None, False)
    ])
    def test_check_forcing_concurrent_batches(self, mocker: MockerFixture, manifest_loader: ManifestLoader, postgres_adapter: PostgresAdapter, event_catcher: EventCatcher, adapter_support: bool, concurrent_batches_config: Optional[bool], expect_warning: bool) -> None: ...