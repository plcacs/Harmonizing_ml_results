"""Support for Azure DevOps sensors."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any
from aioazuredevops.helper import WorkItemState, WorkItemTypeAndState
from aioazuredevops.models.build import Build
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.util import dt as dt_util
from .coordinator import AzureDevOpsConfigEntry, AzureDevOpsDataUpdateCoordinator
from .entity import AzureDevOpsEntity
_LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True, kw_only=True)
class AzureDevOpsBuildSensorEntityDescription(SensorEntityDescription):
    """Class describing Azure DevOps build sensor entities."""
    attr_fn = lambda _: None

@dataclass(frozen=True, kw_only=True)
class AzureDevOpsWorkItemSensorEntityDescription(SensorEntityDescription):
    """Class describing Azure DevOps work item sensor entities."""
BASE_BUILD_SENSOR_DESCRIPTIONS = (AzureDevOpsBuildSensorEntityDescription(key='latest_build', translation_key='latest_build', attr_fn=lambda build: {'definition_id': build.definition.build_id if build.definition else None, 'definition_name': build.definition.name if build.definition else None, 'id': build.build_id, 'reason': build.reason, 'result': build.result, 'source_branch': build.source_branch, 'source_version': build.source_version, 'status': build.status, 'url': build.links.web if build.links else None, 'queue_time': build.queue_time, 'start_time': build.start_time, 'finish_time': build.finish_time}, value_fn=lambda build: build.build_number), AzureDevOpsBuildSensorEntityDescription(key='build_id', translation_key='build_id', entity_registry_visible_default=False, value_fn=lambda build: build.build_id), AzureDevOpsBuildSensorEntityDescription(key='reason', translation_key='reason', entity_registry_visible_default=False, value_fn=lambda build: build.reason), AzureDevOpsBuildSensorEntityDescription(key='result', translation_key='result', entity_registry_visible_default=False, value_fn=lambda build: build.result), AzureDevOpsBuildSensorEntityDescription(key='source_branch', translation_key='source_branch', entity_registry_enabled_default=False, entity_registry_visible_default=False, value_fn=lambda build: build.source_branch), AzureDevOpsBuildSensorEntityDescription(key='source_version', translation_key='source_version', entity_registry_visible_default=False, value_fn=lambda build: build.source_version), AzureDevOpsBuildSensorEntityDescription(key='queue_time', translation_key='queue_time', device_class=SensorDeviceClass.TIMESTAMP, entity_registry_enabled_default=False, entity_registry_visible_default=False, value_fn=lambda build: parse_datetime(build.queue_time)), AzureDevOpsBuildSensorEntityDescription(key='start_time', translation_key='start_time', device_class=SensorDeviceClass.TIMESTAMP, entity_registry_visible_default=False, value_fn=lambda build: parse_datetime(build.start_time)), AzureDevOpsBuildSensorEntityDescription(key='finish_time', translation_key='finish_time', device_class=SensorDeviceClass.TIMESTAMP, entity_registry_visible_default=False, value_fn=lambda build: parse_datetime(build.finish_time)), AzureDevOpsBuildSensorEntityDescription(key='url', translation_key='url', value_fn=lambda build: build.links.web if build.links else None))
BASE_WORK_ITEM_SENSOR_DESCRIPTIONS = (AzureDevOpsWorkItemSensorEntityDescription(key='work_item_count', translation_key='work_item_count', value_fn=lambda work_item_state: len(work_item_state.work_items)),)

def parse_datetime(value):
    """Parse datetime string."""
    if value is None:
        return None
    return dt_util.parse_datetime(value)

async def async_setup_entry(hass, entry, async_add_entities):
    """Set up Azure DevOps sensor based on a config entry."""
    coordinator = entry.runtime_data
    initial_builds = coordinator.data.builds
    entities = [AzureDevOpsBuildSensor(coordinator, description, key) for description in BASE_BUILD_SENSOR_DESCRIPTIONS for key, build in enumerate(initial_builds) if build.project and build.definition]
    entities.extend((AzureDevOpsWorkItemSensor(coordinator, description, key, state_key) for description in BASE_WORK_ITEM_SENSOR_DESCRIPTIONS for key, work_item_type_state in enumerate(coordinator.data.work_items) for state_key, _ in enumerate(work_item_type_state.state_items)))
    async_add_entities(entities)

class AzureDevOpsBuildSensor(AzureDevOpsEntity, SensorEntity):
    """Define a Azure DevOps build sensor."""

    def __init__(self, coordinator, description, item_key):
        """Initialize."""
        super().__init__(coordinator)
        self.entity_description = description
        self.item_key = item_key
        self._attr_unique_id = f'{coordinator.data.organization}_{coordinator.data.project.id}_{self.build.definition.build_id}_{description.key}'
        self._attr_translation_placeholders = {'definition_name': self.build.definition.name}

    @property
    def build(self):
        """Return the build."""
        return self.coordinator.data.builds[self.item_key]

    @property
    def native_value(self):
        """Return the state."""
        return self.entity_description.value_fn(self.build)

    @property
    def extra_state_attributes(self):
        """Return the state attributes of the entity."""
        return self.entity_description.attr_fn(self.build)

class AzureDevOpsWorkItemSensor(AzureDevOpsEntity, SensorEntity):
    """Define a Azure DevOps work item sensor."""

    def __init__(self, coordinator, description, wits_key, state_key):
        """Initialize."""
        super().__init__(coordinator)
        self.entity_description = description
        self.wits_key = wits_key
        self.state_key = state_key
        self._attr_unique_id = f'{coordinator.data.organization}_{coordinator.data.project.id}_{self.work_item_type.name}_{self.work_item_state.name}_{description.key}'
        self._attr_translation_placeholders = {'item_type': self.work_item_type.name, 'item_state': self.work_item_state.name}

    @property
    def work_item_type(self):
        """Return the work item."""
        return self.coordinator.data.work_items[self.wits_key]

    @property
    def work_item_state(self):
        """Return the work item state."""
        return self.work_item_type.state_items[self.state_key]

    @property
    def native_value(self):
        """Return the state."""
        return self.entity_description.value_fn(self.work_item_state)