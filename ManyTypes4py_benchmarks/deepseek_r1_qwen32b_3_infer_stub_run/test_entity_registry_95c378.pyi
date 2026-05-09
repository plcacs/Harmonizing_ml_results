"""Stub file for Entity Registry tests."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import MagicMock, Mock

import attr
from freezegun.api import FrozenDateTimeFactory
import pytest
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import (
    EVENT_HOMEASSISTANT_START,
    STATE_UNAVAILABLE,
    EntityCategory,
)
from homeassistant.core import (
    CoreState,
    HomeAssistant,
    callback,
)
from homeassistant.exceptions import MaxLengthExceeded
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.util.dt import utc_from_timestamp

YAML__OPEN_PATH: str

@pytest.fixture
def hass() -> HomeAssistant:
    ...

@pytest.fixture
def entity_registry(hass: HomeAssistant) -> er.EntityRegistry:
    ...

@pytest.fixture
def device_registry(hass: HomeAssistant) -> dr.DeviceRegistry:
    ...

@pytest.fixture
def freezer() -> FrozenDateTimeFactory:
    ...

@pytest.fixture
def config_entry() -> config_entries.ConfigEntry:
    ...

@pytest.fixture
def mock_config() -> MockConfigEntry:
    ...

@pytest.fixture
def update_events() -> List[dict]:
    ...

@pytest.fixture
def known() -> List[str]:
    ...

@pytest.fixture
def caplog() -> pytest.LogCaptureFixture:
    ...

@pytest.fixture
def load_registries() -> bool:
    ...

@pytest.fixture
def hass_storage() -> Dict:
    ...

@pytest.fixture
def config_entry_1() -> config_entries.ConfigEntry:
    ...

@pytest.fixture
def config_entry_2() -> config_entries.ConfigEntry:
    ...

@pytest.fixture
def mock_subentry_id() -> str:
    ...

@pytest.fixture
def mock_config_entry() -> config_entries.ConfigEntry:
    ...

@pytest.fixture
def mock_config_subentry() -> config_entries.ConfigSubentryData:
    ...

@pytest.fixture
def mock_device_entry() -> dr.DeviceEntry:
    ...

@pytest.fixture
def mock_entity_entry() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_id() -> str:
    ...

@pytest.fixture
def mock_unique_id() -> str:
    ...

@pytest.fixture
def mock_label_id() -> str:
    ...

@pytest.fixture
def mock_category_id() -> Tuple[str, str]:
    ...

@pytest.fixture
def mock_config_entry_with_subentries() -> config_entries.ConfigEntry:
    ...

@pytest.fixture
def mock_device_entry_with_config_entry() -> dr.DeviceEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_config_entry() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_subentry() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_device() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_area() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_hidden_by() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_disabled_by() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_entity_category() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_entry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_config_subentry_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_area_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_aliases() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels_and_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options_and_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_all_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_orphaned_timestamp() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_previous_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_unique_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_new_entity_id() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_updated_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_labels() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_categories() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_options() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_original_attributes() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_device_class() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_icon() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_name() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_unit_of_measurement() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_supported_features() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_translation_key() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_cleared_capabilities() -> er.RegistryEntry:
    ...

@pytest.fixture
def mock_entity_entry_with_cleared_cleared_cleared_cleared_cleared_cleared