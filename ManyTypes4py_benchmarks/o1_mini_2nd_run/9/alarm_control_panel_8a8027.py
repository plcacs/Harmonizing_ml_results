"""Support for Ezviz alarm."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import logging
from typing import Optional, List
from pyezviz import PyEzvizError
from pyezviz.constants import DefenseModeType
from homeassistant.components.alarm_control_panel import (
    AlarmControlPanelEntity,
    AlarmControlPanelEntityDescription,
    AlarmControlPanelEntityFeature,
    AlarmControlPanelState,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import DOMAIN, MANUFACTURER
from .coordinator import EzvizConfigEntry, EzvizDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)
SCAN_INTERVAL: timedelta = timedelta(seconds=60)
PARALLEL_UPDATES: int = 0

@dataclass(frozen=True, kw_only=True)
class EzvizAlarmControlPanelEntityDescription(AlarmControlPanelEntityDescription):
    """Describe an EZVIZ Alarm control panel entity."""
    ezviz_alarm_states: List[Optional[AlarmControlPanelState]]

ALARM_TYPE: EzvizAlarmControlPanelEntityDescription = EzvizAlarmControlPanelEntityDescription(
    key='ezviz_alarm',
    ezviz_alarm_states=[
        None,
        AlarmControlPanelState.DISARMED,
        AlarmControlPanelState.ARMED_AWAY,
        AlarmControlPanelState.ARMED_HOME
    ],
)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: EzvizConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Ezviz alarm control panel."""
    coordinator: EzvizDataUpdateCoordinator = entry.runtime_data
    device_info: DeviceInfo = DeviceInfo(
        identifiers={(DOMAIN, entry.unique_id)},
        name='EZVIZ Alarm',
        model='EZVIZ Alarm',
        manufacturer=MANUFACTURER,
    )
    async_add_entities(
        [EzvizAlarm(coordinator, entry.entry_id, device_info, ALARM_TYPE)]
    )

class EzvizAlarm(AlarmControlPanelEntity):
    """Representation of an Ezviz alarm control panel."""
    _attr_has_entity_name: bool = True
    _attr_name: Optional[str] = None
    _attr_supported_features: AlarmControlPanelEntityFeature = (
        AlarmControlPanelEntityFeature.ARM_AWAY | AlarmControlPanelEntityFeature.ARM_HOME
    )
    _attr_code_arm_required: bool = False

    coordinator: EzvizDataUpdateCoordinator
    entity_description: EzvizAlarmControlPanelEntityDescription

    def __init__(
        self,
        coordinator: EzvizDataUpdateCoordinator,
        entry_id: str,
        device_info: DeviceInfo,
        entity_description: EzvizAlarmControlPanelEntityDescription,
    ) -> None:
        """Initialize alarm control panel entity."""
        super().__init__()
        self._attr_unique_id: str = f'{entry_id}_{entity_description.key}'
        self._attr_device_info: DeviceInfo = device_info
        self.entity_description: EzvizAlarmControlPanelEntityDescription = entity_description
        self.coordinator: EzvizDataUpdateCoordinator = coordinator
        self._attr_alarm_state: Optional[AlarmControlPanelState] = None

    async def async_added_to_hass(self) -> None:
        """Entity added to hass."""
        self.async_schedule_update_ha_state(True)

    def alarm_disarm(self, code: Optional[str] = None) -> None:
        """Send disarm command."""
        try:
            if self.coordinator.ezviz_client.api_set_defence_mode(DefenseModeType.HOME_MODE.value):
                self._attr_alarm_state = AlarmControlPanelState.DISARMED
        except PyEzvizError as err:
            raise HomeAssistantError('Cannot disarm EZVIZ alarm') from err

    def alarm_arm_away(self, code: Optional[str] = None) -> None:
        """Send arm away command."""
        try:
            if self.coordinator.ezviz_client.api_set_defence_mode(DefenseModeType.AWAY_MODE.value):
                self._attr_alarm_state = AlarmControlPanelState.ARMED_AWAY
        except PyEzvizError as err:
            raise HomeAssistantError('Cannot arm EZVIZ alarm') from err

    def alarm_arm_home(self, code: Optional[str] = None) -> None:
        """Send arm home command."""
        try:
            if self.coordinator.ezviz_client.api_set_defence_mode(DefenseModeType.SLEEP_MODE.value):
                self._attr_alarm_state = AlarmControlPanelState.ARMED_HOME
        except PyEzvizError as err:
            raise HomeAssistantError('Cannot arm EZVIZ alarm') from err

    def update(self) -> None:
        """Fetch data from EZVIZ."""
        ezviz_alarm_state_number: str = '0'
        try:
            ezviz_alarm_state_number = self.coordinator.ezviz_client.get_group_defence_mode()
            _LOGGER.debug('Updating EZVIZ alarm with response %s', ezviz_alarm_state_number)
            self._attr_alarm_state = self.entity_description.ezviz_alarm_states[int(ezviz_alarm_state_number)]
        except PyEzvizError as error:
            raise HomeAssistantError(f'Could not fetch EZVIZ alarm status: {error}') from error
