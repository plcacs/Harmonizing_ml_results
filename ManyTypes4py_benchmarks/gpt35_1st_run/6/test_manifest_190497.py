    def test_partial_parse_file_path(self, patched_open: MagicMock, patched_os_exist: MagicMock, patched_state_check: MagicMock) -> None:
    def test_profile_hash_change(self, mock_project: MagicMock) -> None:
    def test_partial_parse_by_version(self, patched_open: MagicMock, patched_os_exist: MagicMock, patched_state_check: MagicMock, runtime_config: RuntimeConfig, manifest: Manifest) -> None:
    def test_partial_parse_safe_update_project_parser_files_partially(self, patched_state_check: MagicMock, patched_read_manifest_for_partial_parse: MagicMock, patched_partial_parsing: MagicMock, patched_active_user: MagicMock, patched_track_partial_parser: MagicMock) -> None:
    def test_write_perf_info(self, mock_project: MagicMock, mocker: MockerFixture, set_required_mocks: None) -> None:
    def test_reset(self, mock_project: MagicMock, mock_adapter: PostgresAdapter, set_required_mocks: None) -> None:
    def test_partial_parse_file_diff_flag(self, mock_project: MagicMock, mocker: MockerFixture, set_required_mocks: None) -> None:
    def test_warn_for_unused_resource_config_paths(self, resource_type: str, path: str, expect_used: bool, manifest: Manifest, runtime_config: RuntimeConfig) -> None:
    def test_check_forcing_concurrent_batches(self, mocker: MockerFixture, manifest_loader: ManifestLoader, postgres_adapter: PostgresAdapter, event_catcher: EventCatcher, adapter_support: bool, concurrent_batches_config: Optional[bool], expect_warning: bool) -> None:
