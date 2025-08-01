from argparse import Namespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
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
from dbt_common.events.event_manager_client import add_callback_to_manager
from tests.unit.fixtures import model_node
from tests.utils import EventCatcher


class TestPartialParse:

    @patch('dbt.parser.manifest.ManifestLoader.build_manifest_state_check')
    @patch('dbt.parser.manifest.os.path.exists')
    @patch('dbt.parser.manifest.open')
    def test_partial_parse_file_path(
        self,
        patched_open: MagicMock,
        patched_os_exist: MagicMock,
        patched_state_check: MagicMock,
    ) -> None:
        mock_project: MagicMock = MagicMock(spec=RuntimeConfig)
        mock_project.project_target_path = 'mock_target_path'
        patched_os_exist.return_value = True
        ManifestLoader(mock_project, {})
        patched_open.assert_called_with('mock_target_path/partial_parse.msgpack', 'rb')
        set_from_args(Namespace(partial_parse_file_path='specified_partial_parse_path'), {})
        ManifestLoader(mock_project, {})
        patched_open.assert_called_with('specified_partial_parse_path', 'rb')

    def test_profile_hash_change(self, mock_project: MagicMock) -> None:
        profile_hash: str = '750bc99c1d64ca518536ead26b28465a224be5ffc918bf2a490102faa5a1bcf5'
        mock_project.credentials.connection_info.return_value = 'test'
        manifest_loader: ManifestLoader = ManifestLoader(mock_project, {})
        assert manifest_loader.manifest.state_check.profile_hash.checksum == profile_hash
        mock_project.credentials.connection_info.return_value = 'test1'
        manifest_loader = ManifestLoader(mock_project, {})
        assert manifest_loader.manifest.state_check.profile_hash.checksum != profile_hash

    @patch('dbt.parser.manifest.ManifestLoader.build_manifest_state_check')
    @patch('dbt.parser.manifest.os.path.exists')
    @patch('dbt.parser.manifest.open')
    def test_partial_parse_by_version(
        self,
        patched_open: MagicMock,
        patched_os_exist: MagicMock,
        patched_state_check: MagicMock,
        runtime_config: RuntimeConfig,
        manifest: Manifest,
    ) -> None:
        file_hash: FileHash = FileHash.from_contents('test contests')
        manifest.state_check = ManifestStateCheck(
            vars_hash=file_hash,
            profile_hash=file_hash,
            profile_env_vars_hash=file_hash,
            project_env_vars_hash=file_hash,
        )
        loader: ManifestLoader = ManifestLoader(runtime_config, {runtime_config.project_name: runtime_config})
        loader.manifest = manifest.deepcopy()
        is_partial_parsable, _ = loader.is_partial_parsable(manifest)
        assert is_partial_parsable
        manifest.metadata.dbt_version = '0.0.1a1'
        is_partial_parsable, _ = loader.is_partial_parsable(manifest)
        assert not is_partial_parsable
        manifest.metadata.dbt_version = '99999.99.99'
        is_partial_parsable, _ = loader.is_partial_parsable(manifest)
        assert not is_partial_parsable


class TestFailedPartialParse:

    @patch('dbt.tracking.track_partial_parser')
    @patch('dbt.tracking.active_user')
    @patch('dbt.parser.manifest.PartialParsing')
    @patch('dbt.parser.manifest.ManifestLoader.read_manifest_for_partial_parse')
    @patch('dbt.parser.manifest.ManifestLoader.build_manifest_state_check')
    def test_partial_parse_safe_update_project_parser_files_partially(
        self,
        patched_state_check: MagicMock,
        patched_read_manifest_for_partial_parse: MagicMock,
        patched_partial_parsing: MagicMock,
        patched_active_user: MagicMock,
        patched_track_partial_parser: MagicMock,
    ) -> None:
        mock_instance: MagicMock = MagicMock()
        mock_instance.skip_parsing.return_value = False
        mock_instance.get_parsing_files.side_effect = KeyError('Whoopsie!')
        patched_partial_parsing.return_value = mock_instance
        mock_project: MagicMock = MagicMock(spec=RuntimeConfig)
        mock_project.project_target_path = 'mock_target_path'
        mock_saved_manifest: MagicMock = MagicMock(spec=Manifest)
        mock_saved_manifest.files = {}
        patched_read_manifest_for_partial_parse.return_value = mock_saved_manifest
        loader: ManifestLoader = ManifestLoader(mock_project, {})
        loader.safe_update_project_parser_files_partially({})
        patched_track_partial_parser.assert_called_once()
        exc_info: Any = patched_track_partial_parser.call_args[0][0]
        assert 'traceback' in exc_info
        assert 'exception' in exc_info
        assert 'code' in exc_info
        assert 'location' in exc_info
        assert 'full_reparse_reason' in exc_info
        assert "KeyError: 'Whoopsie!'" == exc_info['exception']
        assert isinstance(exc_info['code'], str) or isinstance(exc_info['code'], type(None))


class TestGetFullManifest:

    @pytest.fixture
    def set_required_mocks(
        self, mocker: MockerFixture, manifest: Manifest, mock_adapter: Any
    ) -> None:
        mocker.patch('dbt.parser.manifest.get_adapter').return_value = mock_adapter
        mocker.patch('dbt.parser.manifest.ManifestLoader.load').return_value = manifest
        mocker.patch('dbt.parser.manifest._check_manifest').return_value = None
        mocker.patch('dbt.parser.manifest.ManifestLoader.save_macros_to_adapter').return_value = None
        mocker.patch('dbt.tracking.active_user').return_value = User(None)

    def test_write_perf_info(
        self, mock_project: RuntimeConfig, mocker: MockerFixture, set_required_mocks: None
    ) -> None:
        write_perf_info_patch: MagicMock = mocker.patch('dbt.parser.manifest.ManifestLoader.write_perf_info')
        ManifestLoader.get_full_manifest(config=mock_project)
        assert not write_perf_info_patch.called
        ManifestLoader.get_full_manifest(config=mock_project, write_perf_info=False)
        assert not write_perf_info_patch.called
        ManifestLoader.get_full_manifest(config=mock_project, write_perf_info=True)
        assert write_perf_info_patch.called

    def test_reset(
        self, mock_project: RuntimeConfig, mock_adapter: Any, set_required_mocks: None
    ) -> None:
        ManifestLoader.get_full_manifest(config=mock_project)
        assert not mock_project.clear_dependencies.called
        assert not mock_adapter.clear_macro_resolver.called
        ManifestLoader.get_full_manifest(config=mock_project, reset=False)
        assert not mock_project.clear_dependencies.called
        assert not mock_adapter.clear_macro_resolver.called
        ManifestLoader.get_full_manifest(config=mock_project, reset=True)
        assert mock_project.clear_dependencies.called
        assert mock_adapter.clear_macro_resolver.called

    def test_partial_parse_file_diff_flag(
        self, mock_project: RuntimeConfig, mocker: MockerFixture, set_required_mocks: None
    ) -> None:
        mock_file_diff_patch: MagicMock = mocker.patch('dbt.parser.read_files.FileDiff.from_dict')
        mock_file_diff_patch.return_value = FileDiff([], [], [])
        ManifestLoader.get_full_manifest(config=mock_project)
        assert not mock_file_diff_patch.called
        set_from_args(Namespace(PARTIAL_PARSE_FILE_DIFF=True), {})
        ManifestLoader.get_full_manifest(config=mock_project)
        assert not mock_file_diff_patch.called
        set_from_args(Namespace(PARTIAL_PARSE_FILE_DIFF=False), {})
        ManifestLoader.get_full_manifest(config=mock_project)
        assert mock_file_diff_patch.called


class TestWarnUnusedConfigs:

    @pytest.mark.parametrize(
        'resource_type,path,expect_used',
        [
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
            ('unit_tests', 'pkg', True),
        ],
    )
    def test_warn_for_unused_resource_config_paths(
        self,
        resource_type: str,
        path: str,
        expect_used: bool,
        manifest: Manifest,
        runtime_config: RuntimeConfig,
    ) -> None:
        catcher: EventCatcher = EventCatcher(UnusedResourceConfigPath)
        add_callback_to_manager(catcher.catch)
        setattr(runtime_config, resource_type, {path: {'+materialized': 'table'}})
        _warn_for_unused_resource_config_paths(manifest=manifest, config=runtime_config)
        if expect_used:
            assert len(catcher.caught_events) == 0
        else:
            assert len(catcher.caught_events) == 1
            assert f'{resource_type}.{path}' in str(catcher.caught_events[0].data)


class TestCheckForcingConcurrentBatches:

    @pytest.fixture
    @patch('dbt.parser.manifest.ManifestLoader.build_manifest_state_check')
    @patch('dbt.parser.manifest.os.path.exists')
    @patch('dbt.parser.manifest.open')
    def manifest_loader(
        self,
        patched_open: MagicMock,
        patched_os_exist: MagicMock,
        patched_state_check: MagicMock,
    ) -> ManifestLoader:
        mock_project: MagicMock = MagicMock(spec=RuntimeConfig)
        mock_project.project_target_path = 'mock_target_path'
        mock_project.project_name = 'mock_project_name'
        return ManifestLoader(mock_project, {})

    @pytest.fixture
    def event_catcher(self) -> EventCatcher:
        return EventCatcher(InvalidConcurrentBatchesConfig)

    @pytest.mark.parametrize(
        'adapter_support,concurrent_batches_config,expect_warning',
        [
            (False, True, True),
            (False, False, False),
            (False, None, False),
            (True, True, False),
            (True, False, False),
            (True, None, False),
        ],
    )
    def test_check_forcing_concurrent_batches(
        self,
        mocker: MockerFixture,
        manifest_loader: ManifestLoader,
        postgres_adapter: PostgresAdapter,
        event_catcher: EventCatcher,
        adapter_support: bool,
        concurrent_batches_config: Optional[bool],
        expect_warning: bool,
    ) -> None:
        add_callback_to_manager(event_catcher.catch)
        model = model_node()
        model.config.concurrent_batches = concurrent_batches_config
        mocker.patch.object(postgres_adapter, 'supports').return_value = adapter_support
        mocker.patch('dbt.parser.manifest.get_adapter').return_value = postgres_adapter
        mocker.patch.object(manifest_loader.manifest, 'use_microbatch_batches').return_value = True
        manifest_loader.manifest.add_node_nofile(model)
        manifest_loader.check_forcing_batch_concurrency()
        if expect_warning:
            assert len(event_catcher.caught_events) == 1
            assert 'Batches will be run sequentially' in event_catcher.caught_events[0].info.msg
        else:
            assert len(event_catcher.caught_events) == 0
