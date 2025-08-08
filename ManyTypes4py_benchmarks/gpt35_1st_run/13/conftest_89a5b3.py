from __future__ import annotations
from typing import Any

@pytest.fixture(scope='session', autouse=find_spec('zeroconf') is not None)
def patch_zeroconf_multiple_catcher() -> Any:
    ...

@pytest.fixture(scope='session', autouse=True)
def prevent_io() -> Any:
    ...

@pytest.fixture
def entity_registry_enabled_by_default() -> Any:
    ...

@pytest.fixture(name='stub_blueprint_populate')
def stub_blueprint_populate_fixture() -> Any:
    ...

@pytest.fixture(name='mock_tts_get_cache_files')
def mock_tts_get_cache_files_fixture() -> Any:
    ...

@pytest.fixture(name='mock_tts_init_cache_dir')
def mock_tts_init_cache_dir_fixture(init_tts_cache_dir_side_effect) -> Any:
    ...

@pytest.fixture(name='init_tts_cache_dir_side_effect')
def init_tts_cache_dir_side_effect_fixture() -> Any:
    ...

@pytest.fixture(name='mock_tts_cache_dir')
def mock_tts_cache_dir_fixture(tmp_path, mock_tts_init_cache_dir, mock_tts_get_cache_files, request) -> Any:
    ...

@pytest.fixture(name='tts_mutagen_mock')
def tts_mutagen_mock_fixture() -> Any:
    ...

@pytest.fixture(name='mock_conversation_agent')
def mock_conversation_agent_fixture(hass) -> Any:
    ...

@pytest.fixture(scope='session', autouse=find_spec('ffmpeg') is not None)
def prevent_ffmpeg_subprocess() -> Any:
    ...

@pytest.fixture
def mock_light_entities() -> Any:
    ...

@pytest.fixture
def mock_sensor_entities() -> Any:
    ...

@pytest.fixture
def mock_switch_entities() -> Any:
    ...

@pytest.fixture
def mock_legacy_device_scanner() -> Any:
    ...

@pytest.fixture
def mock_legacy_device_tracker_setup() -> Any:
    ...

@pytest.fixture(name='addon_manager')
def addon_manager_fixture(hass, supervisor_client) -> Any:
    ...

@pytest.fixture(name='discovery_info')
def discovery_info_fixture() -> Any:
    ...

@pytest.fixture(name='discovery_info_side_effect')
def discovery_info_side_effect_fixture() -> Any:
    ...

@pytest.fixture(name='get_addon_discovery_info')
def get_addon_discovery_info_fixture(supervisor_client, discovery_info, discovery_info_side_effect) -> Any:
    ...

@pytest.fixture(name='get_discovery_message_side_effect')
def get_discovery_message_side_effect_fixture() -> Any:
    ...

@pytest.fixture(name='get_discovery_message')
def get_discovery_message_fixture(supervisor_client, get_discovery_message_side_effect) -> Any:
    ...

@pytest.fixture(name='addon_store_info_side_effect')
def addon_store_info_side_effect_fixture() -> Any:
    ...

@pytest.fixture(name='addon_store_info')
def addon_store_info_fixture(supervisor_client, addon_store_info_side_effect) -> Any:
    ...

@pytest.fixture(name='addon_info_side_effect')
def addon_info_side_effect_fixture() -> Any:
    ...

@pytest.fixture(name='addon_info')
def addon_info_fixture(supervisor_client, addon_info_side_effect) -> Any:
    ...

@pytest.fixture(name='addon_not_installed')
def addon_not_installed_fixture(addon_store_info, addon_info) -> Any:
    ...

@pytest.fixture(name='addon_installed')
def addon_installed_fixture(addon_store_info, addon_info) -> Any:
    ...

@pytest.fixture(name='addon_running')
def addon_running_fixture(addon_store_info, addon_info) -> Any:
    ...

@pytest.fixture(name='install_addon_side_effect')
def install_addon_side_effect_fixture(addon_store_info, addon_info) -> Any:
    ...

@pytest.fixture(name='install_addon')
def install_addon_fixture(supervisor_client, install_addon_side_effect) -> Any:
    ...

@pytest.fixture(name='start_addon_side_effect')
def start_addon_side_effect_fixture(addon_store_info, addon_info) -> Any:
    ...

@pytest.fixture(name='start_addon')
def start_addon_fixture(supervisor_client, start_addon_side_effect) -> Any:
    ...

@pytest.fixture(name='restart_addon_side_effect')
def restart_addon_side_effect_fixture() -> Any:
    ...

@pytest.fixture(name='restart_addon')
def restart_addon_fixture(supervisor_client, restart_addon_side_effect) -> Any:
    ...

@pytest.fixture(name='stop_addon')
def stop_addon_fixture(supervisor_client) -> Any:
    ...

@pytest.fixture(name='addon_options')
def addon_options_fixture(addon_info) -> Any:
    ...

@pytest.fixture(name='set_addon_options_side_effect')
def set_addon_options_side_effect_fixture(addon_options) -> Any:
    ...

@pytest.fixture(name='set_addon_options')
def set_addon_options_fixture(supervisor_client, set_addon_options_side_effect) -> Any:
    ...

@pytest.fixture(name='uninstall_addon')
def uninstall_addon_fixture(supervisor_client) -> Any:
    ...

@pytest.fixture(name='create_backup')
def create_backup_fixture() -> Any:
    ...

@pytest.fixture(name='update_addon')
def update_addon_fixture(supervisor_client) -> Any:
    ...

@pytest.fixture(name='store_addons')
def store_addons_fixture() -> Any:
    ...

@pytest.fixture(name='store_repositories')
def store_repositories_fixture() -> Any:
    ...

@pytest.fixture(name='store_info')
def store_info_fixture(supervisor_client, store_addons, store_repositories) -> Any:
    ...

@pytest.fixture(name='addon_stats')
def addon_stats_fixture(supervisor_client) -> Any:
    ...

@pytest.fixture(name='addon_changelog')
def addon_changelog_fixture(supervisor_client) -> Any:
    ...

@pytest.fixture(name='supervisor_is_connected')
def supervisor_is_connected_fixture(supervisor_client) -> Any:
    ...

@pytest.fixture(name='resolution_info')
def resolution_info_fixture(supervisor_client) -> Any:
    ...

@pytest.fixture(name='resolution_suggestions_for_issue')
def resolution_suggestions_for_issue_fixture(supervisor_client) -> Any:
    ...

@pytest.fixture(name='supervisor_client')
def supervisor_client() -> Any:
    ...

def _validate_translation_placeholders(full_key, translation, description_placeholders, translation_errors) -> None:
    ...

async def _validate_translation(hass, translation_errors, category, component, key, description_placeholders, *, translation_required=True) -> None:
    ...

def _get_integration_quality_scale(integration) -> Any:
    ...

def _get_integration_quality_scale_rule(integration, rule) -> Any:
    ...

async def _check_step_or_section_translations(hass, translation_errors, category, integration, translation_prefix, description_placeholders, data_schema) -> None:
    ...

async def _check_config_flow_result_translations(manager, flow, result, translation_errors) -> None:
    ...

async def _check_create_issue_translations(issue_registry, issue, translation_errors) -> None:
    ...

def _get_request_quality_scale(request, rule) -> Any:
    ...

async def _check_exception_translation(hass, exception, translation_errors, request) -> None:
    ...

@pytest.fixture(autouse=True)
async def check_translations(ignore_translations, request) -> None:
    ...
