from argparse import Namespace
from typing import Any, Dict, Optional, Tuple, Union
from unittest.mock import MagicMock
import pytest
from pytest_mock import MockerFixture
from dbt.adapters.postgres import PostgresAdapter
from dbt.artifacts.resources.base import FileHash
from dbt.config import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest, ManifestStateCheck
from dbt.events.base import EventMsg
from dbt.parser.manifest import ManifestLoader
from dbt.parser.read_files import FileDiff
from dbt.tracking import User
from tests.unit.fixtures import ModelNode
from tests.utils import EventCatcher

class TestPartialParse:
    @pytest.fixture
    def mock_project(self) -> MagicMock: ...
    @pytest.fixture
    def runtime_config(self) -> RuntimeConfig: ...
    @pytest.fixture
    def manifest(self) -> Manifest: ...
    
    def test_partial_parse_file_path(
        self,
        patched_open: MagicMock,
        patched_os_exist: MagicMock,
        patched_state_check: MagicMock
    ) -> None: ...
    
    def test_profile_hash_change(self, mock_project: MagicMock) -> None: ...
    
    def test_partial_parse_by_version(
        self,
        patched_open: MagicMock,
        patched_os_exist: MagicMock,
        patched_state_check: MagicMock,
        runtime_config: RuntimeConfig,
        manifest: Manifest
    ) -> None: ...

class TestFailedPartialParse:
    def test_partial_parse_safe_update_project_parser_files_partially(
        self,
        patched_state_check: MagicMock,
        patched_read_manifest_for_partial_parse: MagicMock,
        patched_partial_parsing: MagicMock,
        patched_active_user: MagicMock,
        patched_track_partial_parser: MagicMock
    ) -> None: ...

class TestGetFullManifest:
    @pytest.fixture
    def set_required_mocks(
        self,
        mocker: MockerFixture,
        manifest: Manifest,
        mock_adapter: MagicMock
    ) -> None: ...
    
    @pytest.fixture
    def mock_project(self) -> MagicMock: ...
    
    @pytest.fixture
    def mock_adapter(self) -> MagicMock: ...
    
    def test_write_perf_info(
        self,
        mock_project: MagicMock,
        mocker: MockerFixture,
        set_required_mocks: None
    ) -> None: ...
    
    def test_reset(
        self,
        mock_project: MagicMock,
        mock_adapter: MagicMock,
        set_required_mocks: None
    ) -> None: ...
    
    def test_partial_parse_file_diff_flag(
        self,
        mock_project: MagicMock,
        mocker: MockerFixture,
        set_required_mocks: None
    ) -> None: ...

class TestWarnUnusedConfigs:
    @pytest.fixture
    def manifest(self) -> Manifest: ...
    
    @pytest.fixture
    def runtime_config(self) -> RuntimeConfig: ...
    
    @pytest.mark.parametrize("resource_type,path,expect_used", [
        ("data_tests", "unused_path", False),
        ("data_tests", "minimal", True),
        ("metrics", "unused_path", False),
        ("metrics", "test", True),
        ("models", "unused_path", False),
        ("models", "pkg", True),
        ("saved_queries", "unused_path", False),
        ("saved_queries", "test", True),
        ("seeds", "unused_path", False),
        ("seeds", "pkg", True),
        ("semantic_models", "unused_path", False),
        ("semantic_models", "test", True),
        ("sources", "unused_path", False),
        ("sources", "pkg", True),
        ("unit_tests", "unused_path", False),
        ("unit_tests", "pkg", True)
    ])
    def test_warn_for_unused_resource_config_paths(
        self,
        resource_type: str,
        path: str,
        expect_used: bool,
        manifest: Manifest,
        runtime_config: RuntimeConfig
    ) -> None: ...

class TestCheckForcingConcurrentBatches:
    @pytest.fixture
    def manifest_loader(self) -> ManifestLoader: ...
    
    @pytest.fixture
    def event_catcher(self) -> EventCatcher: ...
    
    @pytest.fixture
    def postgres_adapter(self) -> PostgresAdapter: ...
    
    @pytest.mark.parametrize("adapter_support,concurrent_batches_config,expect_warning", [
        (False, True, True),
        (False, False, False),
        (False, None, False),
        (True, True, False),
        (True, False, False),
        (True, None, False)
    ])
    def test_check_forcing_concurrent_batches(
        self,
        mocker: MockerFixture,
        manifest_loader: ManifestLoader,
        postgres_adapter: PostgresAdapter,
        event_catcher: EventCatcher,
        adapter_support: bool,
        concurrent_batches_config: Optional[bool],
        expect_warning: bool
    ) -> None: ...