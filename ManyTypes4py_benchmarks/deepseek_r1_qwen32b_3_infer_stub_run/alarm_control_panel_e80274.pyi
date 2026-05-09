"""Support for manual alarms controllable via MQTT."""
from __future__ import annotations
import datetime
import logging
from typing import Any, Callable, Dict, List, Optional, Union
import voluptuous as vol
from homeassistant.components import mqtt
from homeassistant.components.alarm_control_panel import (
    AlarmControlPanelEntity,
    AlarmControlPanelEntityFeature,
    AlarmControlPanelState,
    CodeFormat,
)
from homeassistant.const import (
    CONF_CODE,
    CONF_DELAY_TIME,
    CONF_DISARM_AFTER_TRIGGER,
    CONF_NAME,
    CONF_PENDING_TIME,
    CONF_PLATFORM,
    CONF_TRIGGER_TIME,
)
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)

def _state_validator(config: dict) -> dict: ...
def _state_schema(state: AlarmControlPanelState) -> vol.Schema: ...

PLATFORM_SCHEMA: vol.Schema = ...  # type: ignore[assignment]

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None: ...

class ManualMQTTAlarm(AlarmControlPanelEntity):
    """Representation of an alarm status."""

    _attr_should_poll: bool = ...
    _attr_supported_features: AlarmControlPanelEntityFeature = ...

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        code: Optional[str],
        code_template: Optional[Any],  # Use proper template type if available
        disarm_after_trigger: bool,
        state_topic: str,
        command_topic: str,
        qos: int,
        code_arm_required: bool,
        payload_disarm: str,
        payload_arm_home: str,
        payload_arm_away: str,
        payload_arm_night: str,
        payload_arm_vacation: str,
        payload_arm_custom_bypass: str,
        config: dict,
    ) -> None: ...

    @property
    def alarm_state(self) -> AlarmControlPanelState: ...
    @property
    def _active_state(self) -> AlarmControlPanelState: ...
    @property
    def code_format(self) -> Optional[CodeFormat]: ...

    async def async_alarm_disarm(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_arm_home(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_arm_away(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_arm_night(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_arm_vacation(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_arm_custom_bypass(self, code: Optional[str] = None) -> None: ...
    async def async_alarm_trigger(self, code: Optional[str] = None) -> None: ...

    def _async_update_state(self, state: AlarmControlPanelState) -> None: ...
    def _async_validate_code(self, code: Optional[str], state: AlarmControlPanelState) -> None: ...

    @property
    def extra_state_attributes(self) -> Dict[str, AlarmControlPanelState]: ...

    @callback
    def async_scheduled_update(self, now: datetime.datetime) -> None: ...

    async def async_added_to_hass(self) -> None: ...
    async def _async_state_changed_listener(self, event: Event) -> None: ...