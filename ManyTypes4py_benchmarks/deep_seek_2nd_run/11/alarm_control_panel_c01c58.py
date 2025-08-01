"""Interfaces with TotalConnect alarm control panels."""
from __future__ import annotations
from typing import Any, cast
from total_connect_client import ArmingHelper
from total_connect_client.exceptions import BadResultCodeError, UsercodeInvalid
from total_connect_client.location import TotalConnectLocation
from homeassistant.components.alarm_control_panel import AlarmControlPanelEntity, AlarmControlPanelEntityFeature, AlarmControlPanelState, CodeFormat
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import entity_platform
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from .const import CODE_REQUIRED, DOMAIN
from .coordinator import TotalConnectConfigEntry, TotalConnectDataUpdateCoordinator
from .entity import TotalConnectLocationEntity
SERVICE_ALARM_ARM_AWAY_INSTANT = 'arm_away_instant'
SERVICE_ALARM_ARM_HOME_INSTANT = 'arm_home_instant'

async def async_setup_entry(
    hass: HomeAssistant,
    entry: TotalConnectConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up TotalConnect alarm panels based on a config entry."""
    coordinator = entry.runtime_data
    code_required = entry.options.get(CODE_REQUIRED, False)
    async_add_entities((TotalConnectAlarm(coordinator, location, partition_id, code_required) for location in coordinator.client.locations.values() for partition_id in location.partitions))
    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(SERVICE_ALARM_ARM_AWAY_INSTANT, None, 'async_alarm_arm_away_instant')
    platform.async_register_entity_service(SERVICE_ALARM_ARM_HOME_INSTANT, None, 'async_alarm_arm_home_instant')

class TotalConnectAlarm(TotalConnectLocationEntity, AlarmControlPanelEntity):
    """Represent a TotalConnect alarm panel."""
    _attr_supported_features = AlarmControlPanelEntityFeature.ARM_HOME | AlarmControlPanelEntityFeature.ARM_AWAY | AlarmControlPanelEntityFeature.ARM_NIGHT

    def __init__(
        self,
        coordinator: TotalConnectDataUpdateCoordinator,
        location: TotalConnectLocation,
        partition_id: str,
        require_code: bool,
    ) -> None:
        """Initialize the TotalConnect status."""
        super().__init__(coordinator, location)
        self._partition_id = int(partition_id)
        self._partition = self._location.partitions[partition_id]
        '\n        Set unique_id to location_id for partition 1 to avoid breaking change\n        for most users with new support for partitions.\n        Add _# for partition 2 and beyond.\n        '
        if int(partition_id) == 1:
            self._attr_name = None
            self._attr_unique_id = str(location.location_id)
        else:
            self._attr_translation_key = 'partition'
            self._attr_translation_placeholders = {'partition_id': str(partition_id)}
            self._attr_unique_id = f'{location.location_id}_{partition_id}'
        self._attr_code_arm_required = require_code
        if require_code:
            self._attr_code_format = CodeFormat.NUMBER

    @property
    def alarm_state(self) -> AlarmControlPanelState | None:
        """Return the state of the device."""
        attr: dict[str, Any] = {'location_id': self._location.location_id, 'partition': self._partition_id, 'ac_loss': self._location.ac_loss, 'low_battery': self._location.low_battery, 'cover_tampered': self._location.is_cover_tampered(), 'triggered_source': None, 'triggered_zone': None}
        if self._partition_id == 1:
            attr['location_name'] = self.device.name
        else:
            attr['location_name'] = f'{self.device.name} partition {self._partition_id}'
        state = None
        if self._partition.arming_state.is_disarmed():
            state = AlarmControlPanelState.DISARMED
        elif self._partition.arming_state.is_armed_night():
            state = AlarmControlPanelState.ARMED_NIGHT
        elif self._partition.arming_state.is_armed_home():
            state = AlarmControlPanelState.ARMED_HOME
        elif self._partition.arming_state.is_armed_away():
            state = AlarmControlPanelState.ARMED_AWAY
        elif self._partition.arming_state.is_armed_custom_bypass():
            state = AlarmControlPanelState.ARMED_CUSTOM_BYPASS
        elif self._partition.arming_state.is_arming():
            state = AlarmControlPanelState.ARMING
        elif self._partition.arming_state.is_disarming():
            state = AlarmControlPanelState.DISARMING
        elif self._partition.arming_state.is_triggered_police():
            state = AlarmControlPanelState.TRIGGERED
            attr['triggered_source'] = 'Police/Medical'
        elif self._partition.arming_state.is_triggered_fire():
            state = AlarmControlPanelState.TRIGGERED
            attr['triggered_source'] = 'Fire/Smoke'
        elif self._partition.arming_state.is_triggered_gas():
            state = AlarmControlPanelState.TRIGGERED
            attr['triggered_source'] = 'Carbon Monoxide'
        self._attr_extra_state_attributes = attr
        return state

    async def async_alarm_disarm(self, code: str | None = None) -> None:
        """Send disarm command."""
        self._check_usercode(code)
        try:
            await self.hass.async_add_executor_job(self._disarm)
        except UsercodeInvalid as error:
            self.coordinator.config_entry.async_start_reauth(self.hass)
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='disarm_invalid_code') from error
        except BadResultCodeError as error:
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='disarm_failed', translation_placeholders={'device': self.device.name}) from error
        await self.coordinator.async_request_refresh()

    def _disarm(self) -> None:
        """Disarm synchronous."""
        ArmingHelper(self._partition).disarm()

    async def async_alarm_arm_home(self, code: str | None = None) -> None:
        """Send arm home command."""
        self._check_usercode(code)
        try:
            await self.hass.async_add_executor_job(self._arm_home)
        except UsercodeInvalid as error:
            self.coordinator.config_entry.async_start_reauth(self.hass)
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='arm_home_invalid_code') from error
        except BadResultCodeError as error:
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='arm_home_failed', translation_placeholders={'device': self.device.name}) from error
        await self.coordinator.async_request_refresh()

    def _arm_home(self) -> None:
        """Arm home synchronous."""
        ArmingHelper(self._partition).arm_stay()

    async def async_alarm_arm_away(self, code: str | None = None) -> None:
        """Send arm away command."""
        self._check_usercode(code)
        try:
            await self.hass.async_add_executor_job(self._arm_away)
        except UsercodeInvalid as error:
            self.coordinator.config_entry.async_start_reauth(self.hass)
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='arm_away_invalid_code') from error
        except BadResultCodeError as error:
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='arm_away_failed', translation_placeholders={'device': self.device.name}) from error
        await self.coordinator.async_request_refresh()

    def _arm_away(self) -> None:
        """Arm away synchronous."""
        ArmingHelper(self._partition).arm_away()

    async def async_alarm_arm_night(self, code: str | None = None) -> None:
        """Send arm night command."""
        self._check_usercode(code)
        try:
            await self.hass.async_add_executor_job(self._arm_night)
        except UsercodeInvalid as error:
            self.coordinator.config_entry.async_start_reauth(self.hass)
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='arm_night_invalid_code') from error
        except BadResultCodeError as error:
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='arm_night_failed', translation_placeholders={'device': self.device.name}) from error
        await self.coordinator.async_request_refresh()

    def _arm_night(self) -> None:
        """Arm night synchronous."""
        ArmingHelper(self._partition).arm_stay_night()

    async def async_alarm_arm_home_instant(self) -> None:
        """Send arm home instant command."""
        try:
            await self.hass.async_add_executor_job(self._arm_home_instant)
        except UsercodeInvalid as error:
            self.coordinator.config_entry.async_start_reauth(self.hass)
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='arm_home_instant_invalid_code') from error
        except BadResultCodeError as error:
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='arm_home_instant_failed', translation_placeholders={'device': self.device.name}) from error
        await self.coordinator.async_request_refresh()

    def _arm_home_instant(self) -> None:
        """Arm home instant synchronous."""
        ArmingHelper(self._partition).arm_stay_instant()

    async def async_alarm_arm_away_instant(self) -> None:
        """Send arm away instant command."""
        try:
            await self.hass.async_add_executor_job(self._arm_away_instant)
        except UsercodeInvalid as error:
            self.coordinator.config_entry.async_start_reauth(self.hass)
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='arm_away_instant_invalid_code') from error
        except BadResultCodeError as error:
            raise HomeAssistantError(translation_domain=DOMAIN, translation_key='arm_away_instant_failed', translation_placeholders={'device': self.device.name}) from error
        await self.coordinator.async_request_refresh()

    def _arm_away_instant(self) -> None:
        """Arm away instant synchronous."""
        ArmingHelper(self._partition).arm_away_instant()

    def _check_usercode(self, code: str | None) -> None:
        """Check if the run-time entered code matches configured code."""
        if self._attr_code_arm_required and self.coordinator.client.usercodes[self._location.location_id] != code:
            raise ServiceValidationError(translation_domain=DOMAIN, translation_key='invalid_pin')
