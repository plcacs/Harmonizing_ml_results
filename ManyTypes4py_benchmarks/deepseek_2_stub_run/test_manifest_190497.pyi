```python
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple, Union
from unittest.mock import MagicMock
import pytest
from pytest_mock import MockerFixture
from dbt.adapters.postgres import PostgresAdapter
from dbt.artifacts.resources.base import FileHash
from dbt.config import RuntimeConfig
from dbt.contracts.graph.manifest import Manifest, ManifestStateCheck
from dbt.events.types import InvalidConcurrentBatchesConfig, UnusedResourceConfigPath
from dbt.parser.manifest import ManifestLoader, _warn_for_unused_resource_config_paths
from dbt.parser.read_files import FileDiff
from dbt.tracking import User
from dbt_common.events.event_manager_client import add_callback_to_manager
from tests.unit.fixtures import model_node
from tests.utils import EventCatcher

class TestPartialParse:
    @staticmethod
    def test_partial_parse_file_path(patched_open: Any, patched_os_exist: Any, patched_state_check: Any) -> None: ...
    def test_profile_hash_change(self, mock_project: Any) -> None: ...
    @staticmethod
    def test_partial_parse_by_version(patched_open: Any, patched_os_exist: Any, patched_state_check: Any, runtime_config: Any, manifest: Any) -> None: ...

class TestFailedPartialParse:
    @staticmethod
    def test_partial_parse_safe_update_project_parser_files_partially(patched_state_check: Any, patched_read_manifest_for_partial_parse: Any, patched_partial_parsing: Any, patched_active_user: Any, patched_track_partial_parser: Any) -> None: ...

class TestGetFullManifest:
    @staticmethod
    def set_required_mocks(mocker: MockerFixture, manifest: Manifest, mock_adapter: Any) -> None: ...
    @staticmethod
    def test_write_perf_info(mock_project: Any, mocker: MockerFixture, set_required_mocks: Any) -> None: ...
    @staticmethod
    def test_reset(mock_project: Any, mock_adapter: Any, set_required_mocks: Any) -> None: ...
    @staticmethod
    def test_partial_parse_file_diff_flag(mock_project: Any, mocker: MockerFixture, set_required_mocks: Any) -> None: ...

class TestWarnUnusedConfigs:
    @staticmethod
    def test_warn_for_unused_resource_config_paths(resource_type: str, path: str, expect_used: bool, manifest: Manifest, runtime_config: RuntimeConfig) -> None: ...

class TestCheckForcingConcurrentBatches:
    @staticmethod
    def manifest_loader(patched_open: Any, patched_os_exist: Any, patched_state_check: Any) -> ManifestLoader: ...
    @staticmethod
    def event_catcher() -> EventCatcher: ...
    @staticmethod
    def test_check_forcing_concurrent_batches(mocker: MockerFixture, manifest_loader: ManifestLoader, postgres_adapter: PostgresAdapter, event_catcher: EventCatcher, adapter_support: bool, concurrent_batches_config: Optional[bool], expect_warning: bool) -> None: ...
```