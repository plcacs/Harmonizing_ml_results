"""Support for MQTT vacuums."""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping, Optional, Union, cast, FrozenSet

import voluptuous as vol
from homeassistant.components import vacuum
from homeassistant.components.vacuum import (
    ENTITY_ID_FORMAT,
    StateVacuumEntity,
    VacuumActivity,
    VacuumEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_SUPPORTED_FEATURES, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.json import json_dumps
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, VolSchemaType
from homeassistant.util.json import json_loads_object

from . import subscription
from .config import MQTT_BASE_SCHEMA
from .const import CONF_COMMAND_TOPIC, CONF_RETAIN, CONF_STATE_TOPIC
from .entity import MqttEntity, async_setup_entity_entry_helper
from .models import ReceiveMessage
from .schemas import MQTT_ENTITY_COMMON_SCHEMA
from .util import valid_publish_topic

PARALLEL_UPDATES: int = 0

BATTERY: str = "battery_level"
FAN_SPEED: str = "fan_speed"
STATE: str = "state"

STATE_IDLE: str = "idle"
STATE_DOCKED: str = "docked"
STATE_ERROR: str = "error"
STATE_PAUSED: str = "paused"
STATE_RETURNING: str = "returning"
STATE_CLEANING: str = "cleaning"

POSSIBLE_STATES: dict[str, VacuumActivity] = {
    STATE_IDLE: VacuumActivity.IDLE,
    STATE_DOCKED: VacuumActivity.DOCKED,
    STATE_ERROR: VacuumActivity.ERROR,
    STATE_PAUSED: VacuumActivity.PAUSED,
    STATE_RETURNING: VacuumActivity.RETURNING,
    STATE_CLEANING: VacuumActivity.CLEANING,
}

CONF_SUPPORTED_FEATURES: str = ATTR_SUPPORTED_FEATURES

CONF_PAYLOAD_TURN_ON: str = "payload_turn_on"
CONF_PAYLOAD_TURN_OFF: str = "payload_turn_off"
CONF_PAYLOAD_RETURN_TO_BASE: str = "payload_return_to_base"
CONF_PAYLOAD_STOP: str = "payload_stop"
CONF_PAYLOAD_CLEAN_SPOT: str = "payload_clean_spot"
CONF_PAYLOAD_LOCATE: str = "payload_locate"
CONF_PAYLOAD_START: str = "payload_start"
CONF_PAYLOAD_PAUSE: str = "payload_pause"
CONF_SET_FAN_SPEED_TOPIC: str = "set_fan_speed_topic"
CONF_FAN_SPEED_LIST: str = "fan_speed_list"
CONF_SEND_COMMAND_TOPIC: str = "send_command_topic"

DEFAULT_NAME: str = "MQTT State Vacuum"
DEFAULT_RETAIN: bool = False
DEFAULT_PAYLOAD_RETURN_TO_BASE: str = "return_to_base"
DEFAULT_PAYLOAD_STOP: str = "stop"
DEFAULT_PAYLOAD_CLEAN_SPOT: str = "clean_spot"
DEFAULT_PAYLOAD_LOCATE: str = "locate"
DEFAULT_PAYLOAD_START: str = "start"
DEFAULT_PAYLOAD_PAUSE: str = "pause"

_LOGGER = logging.getLogger(__name__)

SERVICE_TO_STRING: dict[VacuumEntityFeature, str] = {
    VacuumEntityFeature.START: "start",
    VacuumEntityFeature.PAUSE: "pause",
    VacuumEntityFeature.STOP: "stop",
    VacuumEntityFeature.RETURN_HOME: "return_home",
    VacuumEntityFeature.FAN_SPEED: "fan_speed",
    VacuumEntityFeature.BATTERY: "battery",
    VacuumEntityFeature.STATUS: "status",
    VacuumEntityFeature.SEND_COMMAND: "send_command",
    VacuumEntityFeature.LOCATE: "locate",
    VacuumEntityFeature.CLEAN_SPOT: "clean_spot",
}
STRING_TO_SERVICE: dict[str, VacuumEntityFeature] = {v: k for k, v in SERVICE_TO_STRING.items()}

DEFAULT_SERVICES: VacuumEntityFeature = (
    VacuumEntityFeature.START
    | VacuumEntityFeature.STOP
    | VacuumEntityFeature.RETURN_HOME
    | VacuumEntityFeature.BATTERY
    | VacuumEntityFeature.CLEAN_SPOT
)
ALL_SERVICES: VacuumEntityFeature = (
    DEFAULT_SERVICES
    | VacuumEntityFeature.PAUSE
    | VacuumEntityFeature.LOCATE
    | VacuumEntityFeature.FAN_SPEED
    | VacuumEntityFeature.SEND_COMMAND
)


def services_to_strings(
    services: Union[VacuumEntityFeature, int],
    service_to_string: Mapping[VacuumEntityFeature, str],
) -> list[str]:
    """Convert SUPPORT_* service bitmask to list of service strings."""
    return [service_to_string[service] for service in service_to_string if service & services]


DEFAULT_SERVICE_STRINGS: list[str] = services_to_strings(DEFAULT_SERVICES, SERVICE_TO_STRING)

_FEATURE_PAYLOADS: dict[VacuumEntityFeature, str] = {
    VacuumEntityFeature.START: CONF_PAYLOAD_START,
    VacuumEntityFeature.STOP: CONF_PAYLOAD_STOP,
    VacuumEntityFeature.PAUSE: CONF_PAYLOAD_PAUSE,
    VacuumEntityFeature.CLEAN_SPOT: CONF_PAYLOAD_CLEAN_SPOT,
    VacuumEntityFeature.LOCATE: CONF_PAYLOAD_LOCATE,
    VacuumEntityFeature.RETURN_HOME: CONF_PAYLOAD_RETURN_TO_BASE,
}

MQTT_VACUUM_ATTRIBUTES_BLOCKED: FrozenSet[str] = frozenset(
    {vacuum.ATTR_BATTERY_ICON, vacuum.ATTR_BATTERY_LEVEL, vacuum.ATTR_FAN_SPEED}
)

MQTT_VACUUM_DOCS_URL: str = "https://www.home-assistant.io/integrations/vacuum.mqtt/"

PLATFORM_SCHEMA_MODERN: VolSchemaType = MQTT_BASE_SCHEMA.extend(
    {
        vol.Optional(CONF_FAN_SPEED_LIST, default=[]): vol.All(cv.ensure_list, [cv.string]),
        vol.Optional(CONF_NAME): vol.Any(cv.string, None),
        vol.Optional(CONF_PAYLOAD_CLEAN_SPOT, default=DEFAULT_PAYLOAD_CLEAN_SPOT): cv.string,
        vol.Optional(CONF_PAYLOAD_LOCATE, default=DEFAULT_PAYLOAD_LOCATE): cv.string,
        vol.Optional(CONF_PAYLOAD_RETURN_TO_BASE, default=DEFAULT_PAYLOAD_RETURN_TO_BASE): cv.string,
        vol.Optional(CONF_PAYLOAD_START, default=DEFAULT_PAYLOAD_START): cv.string,
        vol.Optional(CONF_PAYLOAD_PAUSE, default=DEFAULT_PAYLOAD_PAUSE): cv.string,
        vol.Optional(CONF_PAYLOAD_STOP, default=DEFAULT_PAYLOAD_STOP): cv.string,
        vol.Optional(CONF_SEND_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_SET_FAN_SPEED_TOPIC): valid_publish_topic,
        vol.Optional(CONF_STATE_TOPIC): valid_publish_topic,
        vol.Optional(CONF_SUPPORTED_FEATURES, default=DEFAULT_SERVICE_STRINGS): vol.All(
            cv.ensure_list, [vol.In(STRING_TO_SERVICE.keys())]
        ),
        vol.Optional(CONF_COMMAND_TOPIC): valid_publish_topic,
        vol.Optional(CONF_RETAIN, default=DEFAULT_RETAIN): cv.boolean,
    }
).extend(MQTT_ENTITY_COMMON_SCHEMA.schema)

DISCOVERY_SCHEMA: VolSchemaType = PLATFORM_SCHEMA_MODERN.extend({}, extra=vol.ALLOW_EXTRA)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up MQTT vacuum through YAML and through MQTT discovery."""
    async_setup_entity_entry_helper(
        hass,
        config_entry,
        MqttStateVacuum,
        vacuum.DOMAIN,
        async_add_entities,
        DISCOVERY_SCHEMA,
        PLATFORM_SCHEMA_MODERN,
    )


class MqttStateVacuum(MqttEntity, StateVacuumEntity):
    """Representation of a MQTT-controlled state vacuum."""

    _default_name: str = DEFAULT_NAME
    _entity_id_format: str = ENTITY_ID_FORMAT
    _attributes_extra_blocked: FrozenSet[str] = MQTT_VACUUM_ATTRIBUTES_BLOCKED

    def __init__(
        self,
        hass: HomeAssistant,
        config: ConfigType,
        config_entry: Optional[ConfigEntry],
        discovery_data: Optional[DiscoveryInfoType],
    ) -> None:
        """Initialize the vacuum."""
        self._state_attrs: dict[str, Any] = {}
        self._command_topic: Optional[str] = None
        self._set_fan_speed_topic: Optional[str] = None
        self._send_command_topic: Optional[str] = None
        self._payloads: dict[str, str] = {}
        MqttEntity.__init__(self, hass, config, config_entry, discovery_data)

    @staticmethod
    def config_schema() -> VolSchemaType:
        """Return the config schema."""
        return DISCOVERY_SCHEMA

    def _setup_from_config(self, config: ConfigType) -> None:
        """(Re)Setup the entity."""

        def _strings_to_services(
            strings: Iterable[str], string_to_service: Mapping[str, VacuumEntityFeature]
        ) -> VacuumEntityFeature:
            """Convert service strings to SUPPORT_* service bitmask."""
            services: VacuumEntityFeature = VacuumEntityFeature.STATE
            for string in strings:
                services |= string_to_service[string]
            return services

        supported_feature_strings = cast(list[str], config[CONF_SUPPORTED_FEATURES])
        self._attr_supported_features = _strings_to_services(
            supported_feature_strings, STRING_TO_SERVICE
        )
        self._attr_fan_speed_list = cast(list[str], config[CONF_FAN_SPEED_LIST])
        self._command_topic = cast(Optional[str], config.get(CONF_COMMAND_TOPIC))
        self._set_fan_speed_topic = cast(Optional[str], config.get(CONF_SET_FAN_SPEED_TOPIC))
        self._send_command_topic = cast(Optional[str], config.get(CONF_SEND_COMMAND_TOPIC))
        self._payloads = {
            key: cast(str, config.get(key))
            for key in (
                CONF_PAYLOAD_START,
                CONF_PAYLOAD_PAUSE,
                CONF_PAYLOAD_STOP,
                CONF_PAYLOAD_RETURN_TO_BASE,
                CONF_PAYLOAD_CLEAN_SPOT,
                CONF_PAYLOAD_LOCATE,
            )
        }

    def _update_state_attributes(self, payload: dict[str, Any]) -> None:
        """Update the entity state attributes."""
        self._state_attrs.update(payload)
        # cast to Any to avoid strict typing issues (value comes from payload)
        self._attr_fan_speed = cast(Any, self._state_attrs.get(FAN_SPEED, 0))
        self._attr_battery_level = max(0, min(100, int(self._state_attrs.get(BATTERY, 0))))

    @callback
    def _state_message_received(self, msg: ReceiveMessage) -> None:
        """Handle state MQTT message."""
        payload = cast(dict[str, Any], json_loads_object(msg.payload))
        if STATE in payload and ((state := payload[STATE]) in POSSIBLE_STATES or state is None):
            self._attr_activity = (
                POSSIBLE_STATES[cast(str, state)] if payload[STATE] else None
            )
            del payload[STATE]
        self._update_state_attributes(payload)

    @callback
    def _prepare_subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        self.add_subscription(
            CONF_STATE_TOPIC,
            self._state_message_received,
            {"_attr_battery_level", "_attr_fan_speed", "_attr_activity"},
        )

    async def _subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        subscription.async_subscribe_topics_internal(self.hass, self._sub_state)

    async def _async_publish_command(self, feature: VacuumEntityFeature) -> None:
        """Publish a command."""
        if self._command_topic is None:
            return
        await self.async_publish_with_config(
            self._command_topic, self._payloads[_FEATURE_PAYLOADS[feature]]
        )
        self.async_write_ha_state()

    async def async_start(self) -> None:
        """Start the vacuum."""
        await self._async_publish_command(VacuumEntityFeature.START)

    async def async_pause(self) -> None:
        """Pause the vacuum."""
        await self._async_publish_command(VacuumEntityFeature.PAUSE)

    async def async_stop(self, **kwargs: Any) -> None:
        """Stop the vacuum."""
        await self._async_publish_command(VacuumEntityFeature.STOP)

    async def async_return_to_base(self, **kwargs: Any) -> None:
        """Tell the vacuum to return to its dock."""
        await self._async_publish_command(VacuumEntityFeature.RETURN_HOME)

    async def async_clean_spot(self, **kwargs: Any) -> None:
        """Perform a spot clean-up."""
        await self._async_publish_command(VacuumEntityFeature.CLEAN_SPOT)

    async def async_locate(self, **kwargs: Any) -> None:
        """Locate the vacuum (usually by playing a song)."""
        await self._async_publish_command(VacuumEntityFeature.LOCATE)

    async def async_set_fan_speed(self, fan_speed: str, **kwargs: Any) -> None:
        """Set fan speed."""
        if (
            self._set_fan_speed_topic is None
            or self.supported_features & VacuumEntityFeature.FAN_SPEED == 0
            or fan_speed not in self.fan_speed_list
        ):
            return
        await self.async_publish_with_config(self._set_fan_speed_topic, fan_speed)

    async def async_send_command(
        self, command: str, params: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Send a command to a vacuum cleaner."""
        if (
            self._send_command_topic is None
            or self.supported_features & VacuumEntityFeature.SEND_COMMAND == 0
        ):
            return
        if isinstance(params, dict):
            message: dict[str, Any] = {"command": command}
            message.update(params)
            payload: str = json_dumps(message)
        else:
            payload = command
        await self.async_publish_with_config(self._send_command_topic, payload)