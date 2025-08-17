"""MQTT (entity) component mixins and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from functools import partial
import logging
from typing import TYPE_CHECKING, Any, Protocol, cast, final, TypedDict, Optional, Union, Dict, List, Set, Tuple, FrozenSet

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_CONFIGURATION_URL,
    ATTR_HW_VERSION,
    ATTR_MANUFACTURER,
    ATTR_MODEL,
    ATTR_MODEL_ID,
    ATTR_NAME,
    ATTR_SERIAL_NUMBER,
    ATTR_SUGGESTED_AREA,
    ATTR_SW_VERSION,
    ATTR_VIA_DEVICE,
    CONF_DEVICE,
    CONF_ENTITY_CATEGORY,
    CONF_ICON,
    CONF_MODEL,
    CONF_MODEL_ID,
    CONF_NAME,
    CONF_UNIQUE_ID,
    CONF_VALUE_TEMPLATE,
)
from homeassistant.core import Event, HassJobType, HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import (
    DeviceEntry,
    DeviceInfo,
    EventDeviceRegistryUpdatedData,
)
from homeassistant.helpers.dispatcher import (
    async_dispatcher_connect,
    async_dispatcher_send,
)
from homeassistant.helpers.entity import Entity, async_generate_entity_id
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_device_registry_updated_event,
    async_track_entity_registry_updated_event,
)
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue
from homeassistant.helpers.service_info.mqtt import ReceivePayloadType
from homeassistant.helpers.typing import (
    UNDEFINED,
    ConfigType,
    DiscoveryInfoType,
    UndefinedType,
    VolSchemaType,
)
from homeassistant.util.json import json_loads
from homeassistant.util.yaml import dump as yaml_dump

from . import debug_info, subscription
from .client import async_publish
from .const import (
    ATTR_DISCOVERY_HASH,
    ATTR_DISCOVERY_PAYLOAD,
    ATTR_DISCOVERY_TOPIC,
    AVAILABILITY_ALL,
    AVAILABILITY_ANY,
    CONF_AVAILABILITY,
    CONF_AVAILABILITY_MODE,
    CONF_AVAILABILITY_TEMPLATE,
    CONF_AVAILABILITY_TOPIC,
    CONF_CONFIGURATION_URL,
    CONF_CONNECTIONS,
    CONF_ENABLED_BY_DEFAULT,
    CONF_ENCODING,
    CONF_ENTITY_PICTURE,
    CONF_HW_VERSION,
    CONF_IDENTIFIERS,
    CONF_JSON_ATTRS_TEMPLATE,
    CONF_JSON_ATTRS_TOPIC,
    CONF_MANUFACTURER,
    CONF_OBJECT_ID,
    CONF_PAYLOAD_AVAILABLE,
    CONF_PAYLOAD_NOT_AVAILABLE,
    CONF_QOS,
    CONF_RETAIN,
    CONF_SCHEMA,
    CONF_SERIAL_NUMBER,
    CONF_SUGGESTED_AREA,
    CONF_SW_VERSION,
    CONF_TOPIC,
    CONF_VIA_DEVICE,
    DEFAULT_ENCODING,
    DOMAIN,
    MQTT_CONNECTION_STATE,
)
from .debug_info import log_message
from .discovery import (
    MQTT_DISCOVERY_DONE,
    MQTT_DISCOVERY_NEW,
    MQTT_DISCOVERY_UPDATED,
    MQTTDiscoveryPayload,
    clear_discovery_hash,
    get_origin_log_string,
    get_origin_support_url,
    set_discovery_hash,
)
from .models import (
    DATA_MQTT,
    MessageCallbackType,
    MqttValueTemplate,
    MqttValueTemplateException,
    PublishPayloadType,
    ReceiveMessage,
)
from .subscription import (
    EntitySubscription,
    async_prepare_subscribe_topics,
    async_subscribe_topics_internal,
    async_unsubscribe_topics,
)
from .util import mqtt_config_entry_enabled

_LOGGER = logging.getLogger(__name__)

MQTT_ATTRIBUTES_BLOCKED: FrozenSet[str] = frozenset({
    "assumed_state",
    "available",
    "device_class",
    "device_info",
    "entity_category",
    "entity_picture",
    "entity_registry_enabled_default",
    "extra_state_attributes",
    "force_update",
    "icon",
    "friendly_name",
    "should_poll",
    "state",
    "supported_features",
    "unique_id",
    "unit_of_measurement",
})


@callback
def async_handle_schema_error(
    discovery_payload: MQTTDiscoveryPayload, err: vol.Invalid
) -> None:
    """Help handling schema errors on MQTT discovery messages."""
    discovery_topic: str = discovery_payload.discovery_data[ATTR_DISCOVERY_TOPIC]
    _LOGGER.error(
        "Error '%s' when processing MQTT discovery message topic: '%s', message: '%s'",
        err,
        discovery_topic,
        discovery_payload,
    )


def _handle_discovery_failure(
    hass: HomeAssistant,
    discovery_payload: MQTTDiscoveryPayload,
) -> None:
    """Handle discovery failure."""
    discovery_hash = discovery_payload.discovery_data[ATTR_DISCOVERY_HASH]
    clear_discovery_hash(hass, discovery_hash)
    async_dispatcher_send(hass, MQTT_DISCOVERY_DONE.format(*discovery_hash), None)


def _verify_mqtt_config_entry_enabled_for_discovery(
    hass: HomeAssistant, domain: str, discovery_payload: MQTTDiscoveryPayload
) -> bool:
    """Verify MQTT config entry is enabled or log warning."""
    if not mqtt_config_entry_enabled(hass):
        _LOGGER.warning(
            (
                "MQTT integration is disabled, skipping setup of discovered item "
                "MQTT %s, payload %s"
            ),
            domain,
            discovery_payload,
        )
        return False
    return True


class _SetupNonEntityHelperCallbackProtocol(Protocol):  # pragma: no cover
    """Callback protocol for async_setup in async_setup_non_entity_entry_helper."""

    async def __call__(
        self, config: ConfigType, discovery_data: DiscoveryInfoType
    ) -> None: ...


@callback
def async_setup_non_entity_entry_helper(
    hass: HomeAssistant,
    domain: str,
    async_setup: _SetupNonEntityHelperCallbackProtocol,
    discovery_schema: vol.Schema,
) -> None:
    """Set up automation or tag creation dynamically through MQTT discovery."""
    mqtt_data = hass.data[DATA_MQTT]

    async def _async_setup_non_entity_entry_from_discovery(
        discovery_payload: MQTTDiscoveryPayload,
    ) -> None:
        """Set up an MQTT entity, automation or tag from discovery."""
        if not _verify_mqtt_config_entry_enabled_for_discovery(
            hass, domain, discovery_payload
        ):
            return
        try:
            config: ConfigType = discovery_schema(discovery_payload)
            await async_setup(config, discovery_data=discovery_payload.discovery_data)
        except vol.Invalid as err:
            _handle_discovery_failure(hass, discovery_payload)
            async_handle_schema_error(discovery_payload, err)
        except Exception:
            _handle_discovery_failure(hass, discovery_payload)
            raise

    mqtt_data.reload_dispatchers.append(
        async_dispatcher_connect(
            hass,
            MQTT_DISCOVERY_NEW.format(domain, "mqtt"),
            _async_setup_non_entity_entry_from_discovery,
        )
    )


@callback
def async_setup_entity_entry_helper(
    hass: HomeAssistant,
    entry: ConfigEntry,
    entity_class: type[MqttEntity] | None,
    domain: str,
    async_add_entities: AddEntitiesCallback,
    discovery_schema: VolSchemaType,
    platform_schema_modern: VolSchemaType,
    schema_class_mapping: dict[str, type[MqttEntity]] | None = None,
) -> None:
    """Set up entity creation dynamically through MQTT discovery."""
    mqtt_data = hass.data[DATA_MQTT]

    @callback
    def _async_setup_entity_entry_from_discovery(
        discovery_payload: MQTTDiscoveryPayload,
    ) -> None:
        """Set up an MQTT entity from discovery."""
        nonlocal entity_class
        if not _verify_mqtt_config_entry_enabled_for_discovery(
            hass, domain, discovery_payload
        ):
            return
        try:
            config: DiscoveryInfoType = discovery_schema(discovery_payload)
            if schema_class_mapping is not None:
                entity_class = schema_class_mapping[config[CONF_SCHEMA]]
            if TYPE_CHECKING:
                assert entity_class is not None
            async_add_entities(
                [entity_class(hass, config, entry, discovery_payload.discovery_data)]
            )
        except vol.Invalid as err:
            _handle_discovery_failure(hass, discovery_payload)
            async_handle_schema_error(discovery_payload, err)
        except Exception:
            _handle_discovery_failure(hass, discovery_payload)
            raise

    mqtt_data.reload_dispatchers.append(
        async_dispatcher_connect(
            hass,
            MQTT_DISCOVERY_NEW.format(domain, "mqtt"),
            _async_setup_entity_entry_from_discovery,
        )
    )

    @callback
    def _async_setup_entities() -> None:
        """Set up MQTT items from configuration.yaml."""
        nonlocal entity_class
        mqtt_data = hass.data[DATA_MQTT]
        if not (config_yaml := mqtt_data.config):
            return
        yaml_configs: list[ConfigType] = [
            config
            for config_item in config_yaml
            for config_domain, configs in config_item.items()
            for config in configs
            if config_domain == domain
        ]
        entities: list[Entity] = []
        for yaml_config in yaml_configs:
            try:
                config = platform_schema_modern(yaml_config)
                if schema_class_mapping is not None:
                    entity_class = schema_class_mapping[config[CONF_SCHEMA]]
                if TYPE_CHECKING:
                    assert entity_class is not None
                entities.append(entity_class(hass, config, entry, None))
            except vol.Invalid as exc:
                error = str(exc)
                config_file = getattr(yaml_config, "__config_file__", "?")
                line = getattr(yaml_config, "__line__", "?")
                issue_id = hex(hash(frozenset(yaml_config)))
                yaml_config_str = yaml_dump(yaml_config)
                learn_more_url = (
                    f"https://www.home-assistant.io/integrations/{domain}.mqtt/"
                )
                async_create_issue(
                    hass,
                    DOMAIN,
                    issue_id,
                    issue_domain=domain,
                    is_fixable=False,
                    severity=IssueSeverity.ERROR,
                    learn_more_url=learn_more_url,
                    translation_placeholders={
                        "domain": domain,
                        "config_file": config_file,
                        "line": line,
                        "config": yaml_config_str,
                        "error": error,
                    },
                    translation_key="invalid_platform_config",
                )
                _LOGGER.error(
                    "%s for manually configured MQTT %s item, in %s, line %s Got %s",
                    error,
                    domain,
                    config_file,
                    line,
                    yaml_config,
                )

        async_add_entities(entities)

    # When reloading we check manual configured items against the schema
    # before reloading
    mqtt_data.reload_schema[domain] = platform_schema_modern
    # discover manual configured MQTT items
    mqtt_data.reload_handlers[domain] = _async_setup_entities
    _async_setup_entities()


def init_entity_id_from_config(
    hass: HomeAssistant, entity: Entity, config: ConfigType, entity_id_format: str
) -> None:
    """Set entity_id from object_id if defined in config."""
    if CONF_OBJECT_ID in config:
        entity.entity_id = async_generate_entity_id(
            entity_id_format, config[CONF_OBJECT_ID], None, hass
        )


class MqttAttributesMixin(Entity):
    """Mixin used for platforms that support JSON attributes."""

    _attributes_extra_blocked: frozenset[str] = frozenset()
    _attr_tpl: Optional[Callable[[ReceivePayloadType], ReceivePayloadType]] = None

    def __init__(self, config: ConfigType) -> None:
        """Initialize the JSON attributes mixin."""
        self._attributes_sub_state: dict[str, EntitySubscription] = {}
        self._attributes_config = config

    async def async_added_to_hass(self) -> None:
        """Subscribe MQTT events."""
        await super().async_added_to_hass()
        self._attributes_prepare_subscribe_topics()
        self._attributes_subscribe_topics()

    def attributes_prepare_discovery_update(self, config: DiscoveryInfoType) -> None:
        """Handle updated discovery message."""
        self._attributes_config = config
        self._attributes_prepare_subscribe_topics()

    async def attributes_discovery_update(self, config: DiscoveryInfoType) -> None:
        """Handle updated discovery message."""
        self._attributes_subscribe_topics()

    def _attributes_prepare_subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        if template := self._attributes_config.get(CONF_JSON_ATTRS_TEMPLATE):
            self._attr_tpl = MqttValueTemplate(
                template, entity=self
            ).async_render_with_possible_json_value
        self._attributes_sub_state = async_prepare_subscribe_topics(
            self.hass,
            self._attributes_sub_state,
            {
                CONF_JSON_ATTRS_TOPIC: {
                    "topic": self._attributes_config.get(CONF_JSON_ATTRS_TOPIC),
                    "msg_callback": partial(
                        self._message_callback,  # type: ignore[attr-defined]
                        self._attributes_message_received,
                        {"_attr_extra_state_attributes"},
                    ),
                    "entity_id": self.entity_id,
                    "qos": self._attributes_config.get(CONF_QOS),
                    "encoding": self._attributes_config[CONF_ENCODING] or None,
                    "job_type": HassJobType.Callback,
                }
            },
        )

    @callback
    def _attributes_subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        async_subscribe_topics_internal(self.hass, self._attributes_sub_state)

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe when removed."""
        self._attributes_sub_state = async_unsubscribe_topics(
            self.hass, self._attributes_sub_state
        )

    @callback
    def _attributes_message_received(self, msg: ReceiveMessage) -> None:
        """Update extra state attributes."""
        payload = (
            self._attr_tpl(msg.payload) if self._attr_tpl is not None else msg.payload
        )
        try:
            json_dict = json_loads(payload) if isinstance(payload, str) else None
        except ValueError:
            _LOGGER.warning("Erroneous JSON: %s", payload)
        else:
            if isinstance(json_dict, dict):
                filtered_dict = {
                    k: v
                    for k, v in json_dict.items()
                    if k not in MQTT_ATTRIBUTES_BLOCKED
                    and k not in self._attributes_extra_blocked
                }
                self._attr_extra_state_attributes = filtered_dict
            else:
                _LOGGER.warning("JSON result was not a dictionary")


class MqttAvailabilityMixin(Entity):
    """Mixin used for platforms that report availability."""

    def __init__(self, config: ConfigType) -> None:
        """Initialize the availability mixin."""
        self._availability_sub_state: dict[str, EntitySubscription] = {}
        self._available: dict[str, str | bool] = {}
        self._available_latest: bool = False
        self._availability_setup_from_config(config)

    async def async_added_to_hass(self) -> None:
        """Subscribe MQTT events."""
        await super().async_added_to_hass()
        self._availability_prepare_subscribe_topics()
        self._availability_subscribe_topics()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                MQTT_CONNECTION_STATE,
                self.async_mqtt_connection_state_changed,
            )
        )

    def availability_prepare_discovery_update(self, config: DiscoveryInfoType) -> None:
        """Handle updated discovery message."""
        self._availability_setup_from_config(config)
        self._availability_prepare_subscribe_topics()

    async def availability_discovery_update(self, config: DiscoveryInfoType) -> None:
        """Handle updated discovery message."""
        self._availability_subscribe_topics()

    def _availability_setup_from_config(self, config: ConfigType) -> None:
        """(Re)Setup."""
        self._avail_topics: dict[str, dict[str, Any]] = {}
        if CONF_AVAILABILITY_TOPIC in config:
            self._avail_topics[config[CONF_AVAILABILITY_TOPIC]] = {
                CONF_PAYLOAD_AVAILABLE: config[CONF_PAYLOAD_AVAILABLE],
                CONF_PAYLOAD_NOT_AVAILABLE: config[CONF_PAYLOAD_NOT_AVAILABLE],
                CONF_AVAILABILITY_TEMPLATE: config.get(CONF_AVAILABILITY_TEMPLATE),
            }

        if CONF_AVAILABILITY in config:
            avail: dict[str, Any]
            for avail in config[CONF_AVAILABILITY]:
                self._avail_topics[avail[CONF_TOPIC]] = {
                    CONF_PAYLOAD_AVAILABLE: avail[CONF_PAY