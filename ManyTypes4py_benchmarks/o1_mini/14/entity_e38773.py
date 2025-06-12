"""MQTT (entity) component mixins and helpers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Iterable
from functools import partial
import logging
from typing import TYPE_CHECKING, Any, Optional, Protocol, Type, cast

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
from homeassistant.helpers.device_registry import DeviceEntry, DeviceInfo, EventDeviceRegistryUpdatedData
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
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
MQTT_ATTRIBUTES_BLOCKED: frozenset[str] = frozenset(
    {'assumed_state', 'available', 'device_class', 'device_info', 'entity_category', 'entity_picture', 'entity_registry_enabled_default', 'extra_state_attributes', 'force_update', 'icon', 'friendly_name', 'should_poll', 'state', 'supported_features', 'unique_id', 'unit_of_measurement'}
)

@callback
def async_handle_schema_error(discovery_payload: MQTTDiscoveryPayload, err: Exception) -> None:
    """Help handling schema errors on MQTT discovery messages."""
    discovery_topic = discovery_payload.discovery_data[ATTR_DISCOVERY_TOPIC]
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
    hass: HomeAssistant,
    domain: str,
    discovery_payload: MQTTDiscoveryPayload,
) -> bool:
    """Verify MQTT config entry is enabled or log warning."""
    if not mqtt_config_entry_enabled(hass):
        _LOGGER.warning(
            'MQTT integration is disabled, skipping setup of discovered item MQTT %s, payload %s',
            domain,
            discovery_payload,
        )
        return False
    return True

class _SetupNonEntityHelperCallbackProtocol(Protocol):
    """Callback protocol for async_setup in async_setup_non_entity_entry_helper."""

    async def __call__(self, config: ConfigType, discovery_data: dict[str, Any]) -> None:
        ...

@callback
def async_setup_non_entity_entry_helper(
    hass: HomeAssistant,
    domain: str,
    async_setup: _SetupNonEntityHelperCallbackProtocol,
    discovery_schema: Callable[[MQTTDiscoveryPayload], ConfigType],
) -> None:
    """Set up automation or tag creation dynamically through MQTT discovery."""
    mqtt_data = hass.data[DATA_MQTT]

    async def _async_setup_non_entity_entry_from_discovery(discovery_payload: MQTTDiscoveryPayload) -> None:
        """Set up an MQTT entity, automation or tag from discovery."""
        if not _verify_mqtt_config_entry_enabled_for_discovery(hass, domain, discovery_payload):
            return
        try:
            config = discovery_schema(discovery_payload)
            await async_setup(config, discovery_data=discovery_payload.discovery_data)
        except vol.Invalid as err:
            _handle_discovery_failure(hass, discovery_payload)
            async_handle_schema_error(discovery_payload, err)
        except Exception:
            _handle_discovery_failure(hass, discovery_payload)
            raise

    mqtt_data.reload_dispatchers.append(
        async_dispatcher_connect(
            hass, MQTT_DISCOVERY_NEW.format(domain, 'mqtt'), _async_setup_non_entity_entry_from_discovery
        )
    )

@callback
def async_setup_entity_entry_helper(
    hass: HomeAssistant,
    entry: ConfigEntry,
    entity_class: Type[Entity],
    domain: str,
    async_add_entities: AddEntitiesCallback,
    discovery_schema: Callable[[MQTTDiscoveryPayload], ConfigType],
    platform_schema_modern: Callable[[Any], ConfigType],
    schema_class_mapping: Optional[dict[str, Type[Entity]]] = None,
) -> None:
    """Set up entity creation dynamically through MQTT discovery."""
    mqtt_data = hass.data[DATA_MQTT]

    @callback
    def _async_setup_entity_entry_from_discovery(discovery_payload: MQTTDiscoveryPayload) -> None:
        """Set up an MQTT entity from discovery."""
        nonlocal entity_class
        if not _verify_mqtt_config_entry_enabled_for_discovery(hass, domain, discovery_payload):
            return
        try:
            config = discovery_schema(discovery_payload)
            if schema_class_mapping is not None:
                entity_class = schema_class_mapping[config[CONF_SCHEMA]]
            if TYPE_CHECKING:
                assert entity_class is not None
            async_add_entities([entity_class(hass, config, entry, discovery_payload.discovery_data)])
        except vol.Invalid as err:
            _handle_discovery_failure(hass, discovery_payload)
            async_handle_schema_error(discovery_payload, err)
        except Exception:
            _handle_discovery_failure(hass, discovery_payload)
            raise

    mqtt_data.reload_dispatchers.append(
        async_dispatcher_connect(
            hass, MQTT_DISCOVERY_NEW.format(domain, 'mqtt'), _async_setup_entity_entry_from_discovery
        )
    )

    @callback
    def _async_setup_entities() -> None:
        """Set up MQTT items from configuration.yaml."""
        nonlocal entity_class
        mqtt_data = hass.data[DATA_MQTT]
        if not (config_yaml := mqtt_data.config):
            return
        yaml_configs = [
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
                config_file = getattr(yaml_config, '__config_file__', '?')
                line = getattr(yaml_config, '__line__', '?')
                issue_id = hex(hash(frozenset(yaml_config)))
                yaml_config_str = yaml_dump(yaml_config)
                learn_more_url = f'https://www.home-assistant.io/integrations/{domain}.mqtt/'
                async_create_issue(
                    hass,
                    DOMAIN,
                    issue_id,
                    issue_domain=domain,
                    is_fixable=False,
                    severity=IssueSeverity.ERROR,
                    learn_more_url=learn_more_url,
                    translation_placeholders={
                        'domain': domain,
                        'config_file': config_file,
                        'line': line,
                        'config': yaml_config_str,
                        'error': error,
                    },
                    translation_key='invalid_platform_config',
                )
                _LOGGER.error(
                    '%s for manually configured MQTT %s item, in %s, line %s Got %s',
                    error,
                    domain,
                    config_file,
                    line,
                    yaml_config,
                )
        async_add_entities(entities)

    mqtt_data.reload_schema[domain] = platform_schema_modern
    mqtt_data.reload_handlers[domain] = _async_setup_entities
    _async_setup_entities()

def init_entity_id_from_config(
    hass: HomeAssistant,
    entity: Entity,
    config: ConfigType,
    entity_id_format: str,
) -> None:
    """Set entity_id from object_id if defined in config."""
    if CONF_OBJECT_ID in config:
        entity.entity_id = async_generate_entity_id(entity_id_format, config[CONF_OBJECT_ID], None, hass)

class MqttAttributesMixin(Entity):
    """Mixin used for platforms that support JSON attributes."""
    _attributes_extra_blocked: frozenset[str] = frozenset()
    _attr_tpl: Optional[Callable[[str], Any]] = None

    def __init__(self, config: ConfigType) -> None:
        """Initialize the JSON attributes mixin."""
        self._attributes_sub_state: dict[str, Any] = {}
        self._attributes_config: ConfigType = config

    async def async_added_to_hass(self) -> None:
        """Subscribe MQTT events."""
        await super().async_added_to_hass()
        self._attributes_prepare_subscribe_topics()
        self._attributes_subscribe_topics()

    def attributes_prepare_discovery_update(self, config: ConfigType) -> None:
        """Handle updated discovery message."""
        self._attributes_config = config
        self._attributes_prepare_subscribe_topics()

    async def attributes_discovery_update(self, config: ConfigType) -> None:
        """Handle updated discovery message."""
        self._attributes_subscribe_topics()

    def _attributes_prepare_subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        if (template := self._attributes_config.get(CONF_JSON_ATTRS_TEMPLATE)):
            self._attr_tpl = MqttValueTemplate(template, entity=self).async_render_with_possible_json_value
        self._attributes_sub_state = async_prepare_subscribe_topics(
            self.hass,
            self._attributes_sub_state,
            {
                CONF_JSON_ATTRS_TOPIC: {
                    'topic': self._attributes_config.get(CONF_JSON_ATTRS_TOPIC),
                    'msg_callback': partial(
                        self._message_callback,
                        self._attributes_message_received,
                        {'_attr_extra_state_attributes'},
                    ),
                    'entity_id': self.entity_id,
                    'qos': self._attributes_config.get(CONF_QOS),
                    'encoding': self._attributes_config[CONF_ENCODING] or None,
                    'job_type': HassJobType.Callback,
                }
            },
        )

    @callback
    def _attributes_subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        async_subscribe_topics_internal(self.hass, self._attributes_sub_state)

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe when removed."""
        self._attributes_sub_state = async_unsubscribe_topics(self.hass, self._attributes_sub_state)

    @callback
    def _attributes_message_received(self, msg: ReceiveMessage) -> None:
        """Update extra state attributes."""
        payload = self._attr_tpl(msg.payload) if self._attr_tpl is not None else msg.payload
        try:
            json_dict = json_loads(payload) if isinstance(payload, str) else None
        except ValueError:
            _LOGGER.warning('Erroneous JSON: %s', payload)
        else:
            if isinstance(json_dict, dict):
                filtered_dict = {
                    k: v
                    for k, v in json_dict.items()
                    if k not in MQTT_ATTRIBUTES_BLOCKED and k not in self._attributes_extra_blocked
                }
                self._attr_extra_state_attributes = filtered_dict
            else:
                _LOGGER.warning('JSON result was not a dictionary')

class MqttAvailabilityMixin(Entity):
    """Mixin used for platforms that report availability."""

    def __init__(self, config: ConfigType) -> None:
        """Initialize the availability mixin."""
        self._availability_sub_state: dict[str, Any] = {}
        self._available: dict[str, bool] = {}
        self._available_latest: bool = False
        self._availability_setup_from_config(config)

    async def async_added_to_hass(self) -> None:
        """Subscribe MQTT events."""
        await super().async_added_to_hass()
        self._availability_prepare_subscribe_topics()
        self._availability_subscribe_topics()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, MQTT_CONNECTION_STATE, self.async_mqtt_connection_state_changed
            )
        )

    def availability_prepare_discovery_update(self, config: ConfigType) -> None:
        """Handle updated discovery message."""
        self._availability_setup_from_config(config)
        self._availability_prepare_subscribe_topics()

    async def availability_discovery_update(self, config: ConfigType) -> None:
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
            for avail in config[CONF_AVAILABILITY]:
                self._avail_topics[avail[CONF_TOPIC]] = {
                    CONF_PAYLOAD_AVAILABLE: avail[CONF_PAYLOAD_AVAILABLE],
                    CONF_PAYLOAD_NOT_AVAILABLE: avail[CONF_PAYLOAD_NOT_AVAILABLE],
                    CONF_AVAILABILITY_TEMPLATE: avail.get(CONF_VALUE_TEMPLATE),
                }
        for avail_topic_conf in self._avail_topics.values():
            if (template := avail_topic_conf[CONF_AVAILABILITY_TEMPLATE]):
                avail_topic_conf[CONF_AVAILABILITY_TEMPLATE] = MqttValueTemplate(
                    template, entity=self
                ).async_render_with_possible_json_value
        self._avail_config: ConfigType = config

    def _availability_prepare_subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        self._available = {topic: self._available.get(topic, False) for topic in self._avail_topics}
        topics: dict[str, dict[str, Any]] = {
            f'availability_{topic}': {
                'topic': topic,
                'msg_callback': partial(
                    self._message_callback,
                    self._availability_message_received,
                    {'available'},
                ),
                'entity_id': self.entity_id,
                'qos': self._avail_config[CONF_QOS],
                'encoding': self._avail_config[CONF_ENCODING] or None,
                'job_type': HassJobType.Callback,
            }
            for topic in self._avail_topics
        }
        self._availability_sub_state = async_prepare_subscribe_topics(
            self.hass,
            self._availability_sub_state,
            topics,
        )

    @callback
    def _availability_message_received(self, msg: ReceiveMessage) -> None:
        """Handle a new received MQTT availability message."""
        topic = msg.topic
        avail_topic = self._avail_topics[topic]
        template = avail_topic[CONF_AVAILABILITY_TEMPLATE]
        payload = template(msg.payload) if template else msg.payload
        if payload == avail_topic[CONF_PAYLOAD_AVAILABLE]:
            self._available[topic] = True
            self._available_latest = True
        elif payload == avail_topic[CONF_PAYLOAD_NOT_AVAILABLE]:
            self._available[topic] = False
            self._available_latest = False

    @callback
    def _availability_subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        async_subscribe_topics_internal(self.hass, self._availability_sub_state)

    @callback
    def async_mqtt_connection_state_changed(self, state: Any) -> None:
        """Update state on connection/disconnection to MQTT broker."""
        if not self.hass.is_stopping:
            self.async_write_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe when removed."""
        self._availability_sub_state = async_unsubscribe_topics(self.hass, self._availability_sub_state)

    @property
    def available(self) -> bool:
        """Return if the device is available."""
        mqtt_data = self.hass.data[DATA_MQTT]
        client = mqtt_data.client
        if not client.connected and (not self.hass.is_stopping):
            return False
        if not self._avail_topics:
            return True
        if self._avail_config[CONF_AVAILABILITY_MODE] == AVAILABILITY_ALL:
            return all(self._available.values())
        if self._avail_config[CONF_AVAILABILITY_MODE] == AVAILABILITY_ANY:
            return any(self._available.values())
        return self._available_latest

async def cleanup_device_registry(
    hass: HomeAssistant,
    device_id: Optional[str],
    config_entry_id: Optional[str],
) -> None:
    """Clean up the device registry after MQTT removal.

    Remove MQTT from the device registry entry if there are no remaining
    entities, triggers or tags.
    """
    from . import device_trigger, tag

    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)
    if (
        device_id
        and device_id not in device_registry.deleted_devices
        and config_entry_id
        and not er.async_entries_for_device(
            entity_registry, device_id, include_disabled_entities=False
        )
        and not await device_trigger.async_get_triggers(hass, device_id)
        and not tag.async_has_tags(hass, device_id)
    ):
        device_registry.async_update_device(device_id, remove_config_entry_id=config_entry_id)

def get_discovery_hash(discovery_data: dict[str, Any]) -> tuple[str, ...]:
    """Get the discovery hash from the discovery data."""
    discovery_hash = discovery_data[ATTR_DISCOVERY_HASH]
    return discovery_hash

def send_discovery_done(
    hass: HomeAssistant,
    discovery_data: dict[str, Any],
) -> None:
    """Acknowledge a discovery message has been handled."""
    discovery_hash = get_discovery_hash(discovery_data)
    async_dispatcher_send(hass, MQTT_DISCOVERY_DONE.format(*discovery_hash), None)

def stop_discovery_updates(
    hass: HomeAssistant,
    discovery_data: dict[str, Any],
    remove_discovery_updated: Optional[Callable[[], None]] = None,
) -> None:
    """Stop discovery updates of being sent."""
    if remove_discovery_updated:
        remove_discovery_updated()
        remove_discovery_updated = None
    discovery_hash = get_discovery_hash(discovery_data)
    clear_discovery_hash(hass, discovery_hash)

async def async_remove_discovery_payload(
    hass: HomeAssistant,
    discovery_data: dict[str, Any],
) -> None:
    """Clear retained discovery payload.

    Remove discovery topic in broker to avoid rediscovery
    after a restart of Home Assistant.
    """
    discovery_topic = discovery_data[ATTR_DISCOVERY_TOPIC]
    await async_publish(hass, discovery_topic, None, retain=True)

async def async_clear_discovery_topic_if_entity_removed(
    hass: HomeAssistant,
    discovery_data: dict[str, Any],
    event: Event,
) -> None:
    """Clear the discovery topic if the entity is removed."""
    if event.data['action'] == 'remove':
        await async_remove_discovery_payload(hass, discovery_data)

class MqttDiscoveryDeviceUpdateMixin(ABC):
    """Add support for auto discovery for platforms without an entity."""

    def __init__(
        self,
        hass: HomeAssistant,
        discovery_data: Optional[MQTTDiscoveryPayload],
        device_id: Optional[str],
        config_entry: ConfigEntry,
        log_name: str,
    ) -> None:
        """Initialize the update service."""
        self.hass = hass
        self.log_name = log_name
        self._discovery_data: Optional[MQTTDiscoveryPayload] = discovery_data
        self._device_id: Optional[str] = device_id
        self._config_entry: ConfigEntry = config_entry
        self._config_entry_id: str = config_entry.entry_id
        self._skip_device_removal: bool = False
        self._migrate_discovery: Optional[str] = None
        discovery_hash = get_discovery_hash(discovery_data) if discovery_data else ()
        self._remove_discovery_updated: Optional[Callable[[], None]] = (
            async_dispatcher_connect(
                hass,
                MQTT_DISCOVERY_UPDATED.format(*discovery_hash) if discovery_data else "",
                self.async_discovery_update,
            )
            if discovery_data
            else None
        )
        config_entry.async_on_unload(self._entry_unload)
        if device_id is not None:
            self._remove_device_updated: Optional[Callable[[], None]] = async_track_device_registry_updated_event(
                hass, device_id, self._async_device_removed
            )
        else:
            self._remove_device_updated = None
        _LOGGER.debug('%s %s has been initialized', self.log_name, discovery_hash)

    @callback
    def _entry_unload(self, *args: Any) -> None:
        """Handle cleanup when the config entry is unloaded."""
        stop_discovery_updates(self.hass, self._discovery_data.discovery_data, self._remove_discovery_updated)
        self._config_entry.async_create_task(self.hass, self.async_tear_down())

    async def async_discovery_update(self, discovery_payload: MQTTDiscoveryPayload) -> None:
        """Handle discovery update."""
        discovery_hash = get_discovery_hash(self._discovery_data) if self._discovery_data else ()
        if (
            discovery_payload.migrate_discovery
            and self._migrate_discovery is None
            and (self._discovery_data.discovery_data[ATTR_DISCOVERY_TOPIC] == discovery_payload.discovery_data[ATTR_DISCOVERY_TOPIC])
            if self._discovery_data else False
        ):
            self._migrate_discovery = self._discovery_data.discovery_data[ATTR_DISCOVERY_TOPIC] if self._discovery_data else None
            discovery_hash = self._discovery_data.discovery_data[ATTR_DISCOVERY_HASH] if self._discovery_data else ()
            origin_info = get_origin_log_string(
                self._discovery_data.discovery_data[ATTR_DISCOVERY_PAYLOAD], include_url=False
            ) if self._discovery_data else ""
            action = 'Rollback' if discovery_payload.device_discovery else 'Migration'
            schema_type = 'platform' if discovery_payload.device_discovery else 'device'
            _LOGGER.info(
                "%s to MQTT %s discovery schema started for %s '%s'%s on topic %s. To complete %s, publish a %s discovery message with %s '%s'. After completed %s, publish an empty (retained) payload to %s",
                action,
                schema_type,
                discovery_hash[0] if len(discovery_hash) > 0 else "",
                discovery_hash[1] if len(discovery_hash) > 1 else "",
                origin_info,
                self._migrate_discovery,
                action.lower(),
                schema_type,
                discovery_hash[0] if len(discovery_hash) > 0 else "",
                discovery_hash[1] if len(discovery_hash) > 1 else "",
                action.lower(),
                self._migrate_discovery,
            )
            await self.async_tear_down()
            stop_discovery_updates(self.hass, self._discovery_data.discovery_data, self._remove_discovery_updated)
            send_discovery_done(self.hass, self._discovery_data.discovery_data)
            return
        _LOGGER.debug(
            "Got update for %s with hash: %s '%s'",
            self.log_name,
            discovery_hash,
            discovery_payload,
        )
        new_discovery_topic = discovery_payload.discovery_data[ATTR_DISCOVERY_TOPIC]
        if self._discovery_data.discovery_data[ATTR_DISCOVERY_TOPIC] != new_discovery_topic:
            old_origin_info = get_origin_log_string(
                self._discovery_data.discovery_data[ATTR_DISCOVERY_PAYLOAD], include_url=False
            ) if self._discovery_data else ""
            new_origin_info = get_origin_log_string(
                discovery_payload.discovery_data[ATTR_DISCOVERY_PAYLOAD], include_url=False
            )
            new_origin_support_url = get_origin_support_url(discovery_payload.discovery_data[ATTR_DISCOVERY_PAYLOAD])
            if new_origin_support_url:
                get_support = f'for support visit {new_origin_support_url}'
            else:
                get_support = 'for documentation on migration to device schema or rollback to discovery schema, visit https://www.home-assistant.io/integrations/mqtt/#migration-from-single-component-to-device-based-discovery'
            _LOGGER.warning(
                "Received a conflicting MQTT discovery message for %s '%s' which was previously discovered on topic %s%s; the conflicting discovery message was received on topic %s%s; %s",
                discovery_hash[0] if len(discovery_hash) > 0 else "",
                discovery_hash[1] if len(discovery_hash) > 1 else "",
                self._discovery_data.discovery_data[ATTR_DISCOVERY_TOPIC] if self._discovery_data else "",
                old_origin_info,
                new_discovery_topic,
                new_origin_info,
                get_support,
            )
            send_discovery_done(self.hass, self._discovery_data.discovery_data)
            return
        if self._discovery_data:
            debug_info.update_entity_discovery_data(self.hass, discovery_payload, self.entity_id)
        if not discovery_payload:
            if self._migrate_discovery is None:
                _LOGGER.info('Removing component: %s', self.entity_id)
            else:
                _LOGGER.info('Unloading component: %s', self.entity_id)
            self.hass.async_create_task(self._async_process_discovery_update_and_remove())
        elif self._discovery_update:
            if self._discovery_data.discovery_data[ATTR_DISCOVERY_PAYLOAD] != discovery_payload:
                _LOGGER.info('Updating component: %s', self.entity_id)
                self.hass.async_create_task(
                    self._async_process_discovery_update(
                        discovery_payload, self._discovery_update, self._discovery_data.discovery_data
                    )
                )
            else:
                _LOGGER.debug('Ignoring unchanged update for: %s', self.entity_id)
                send_discovery_done(self.hass, self._discovery_data.discovery_data)
        else:
            send_discovery_done(self.hass, self._discovery_data.discovery_data)
            _LOGGER.debug('No changes for: %s', self.entity_id)
            return

    async def _async_device_removed(self, event: Event) -> None:
        """Handle the manual removal of a device."""
        if self._skip_device_removal or not async_removed_from_device(
            self.hass, event, cast(str, self._device_id), self._config_entry_id
        ):
            return
        if self._remove_device_updated:
            self._remove_device_updated()
        self._skip_device_removal = True
        stop_discovery_updates(self.hass, self._discovery_data.discovery_data, self._remove_discovery_updated)
        await self._async_tear_down()
        await async_remove_discovery_payload(self.hass, self._discovery_data.discovery_data)

    async def _async_tear_down(self) -> None:
        """Handle the cleanup of the discovery service."""
        await self.async_tear_down()
        if not self._skip_device_removal:
            self._skip_device_removal = True
            await cleanup_device_registry(self.hass, self._device_id, self._config_entry_id)

    @abstractmethod
    async def async_update(self, discovery_data: MQTTDiscoveryPayload) -> None:
        """Handle the update of platform specific parts, extend to the platform."""

    @abstractmethod
    async def async_tear_down(self) -> None:
        """Handle the cleanup of platform specific parts, extend to the platform."""

    @callback
    def _async_discovery_callback(self, payload: MQTTDiscoveryPayload) -> None:
        """Handle discovery update.

        If the payload has changed we will create a task to
        do the discovery update.

        As this callback can fire when nothing has changed, this
        is a normal function to avoid task creation until it is needed.
        """
        if self._discovery_data is None:
            return
        discovery_hash = get_discovery_hash(self._discovery_data)
        if payload.migrate_discovery and self._migrate_discovery is None and (
            self._discovery_data.discovery_data[ATTR_DISCOVERY_TOPIC]
            == payload.discovery_data[ATTR_DISCOVERY_TOPIC]
        ):
            if self.unique_id is None or self.device_info is None:
                _LOGGER.error(
                    'Discovery migration is not possible for entity %s on topic %s. A unique_id and device context is required, got unique_id: %s, device: %s',
                    self.entity_id,
                    self._discovery_data.discovery_data[ATTR_DISCOVERY_TOPIC],
                    self.unique_id,
                    self.device_info,
                )
                send_discovery_done(self.hass, self._discovery_data.discovery_data)
                return
            self._migrate_discovery = self._discovery_data.discovery_data[ATTR_DISCOVERY_TOPIC]
            discovery_hash = self._discovery_data.discovery_data[ATTR_DISCOVERY_HASH]
            origin_info = get_origin_log_string(
                self._discovery_data.discovery_data[ATTR_DISCOVERY_PAYLOAD], include_url=False
            )
            action = 'Rollback' if payload.device_discovery else 'Migration'
            schema_type = 'platform' if payload.device_discovery else 'device'
            _LOGGER.info(
                "%s to MQTT %s discovery schema started for entity %s%s on topic %s. To complete %s, publish a %s discovery message with %s entity '%s'. After completed %s, publish an empty (retained) payload to %s",
                action,
                schema_type,
                self.entity_id,
                origin_info,
                self._migrate_discovery,
                action.lower(),
                schema_type,
                discovery_hash[0],
                discovery_hash[1],
                action.lower(),
                self._migrate_discovery,
            )
        old_payload = self._discovery_data.discovery_data[ATTR_DISCOVERY_PAYLOAD] if self._discovery_data else {}
        _LOGGER.debug(
            "Got update for entity with hash: %s '%s'", discovery_hash, payload
        )
        new_discovery_topic = payload.discovery_data[ATTR_DISCOVERY_TOPIC]
        if self._discovery_data.discovery_data[ATTR_DISCOVERY_TOPIC] != new_discovery_topic:
            old_origin_info = get_origin_log_string(
                self._discovery_data.discovery_data[ATTR_DISCOVERY_PAYLOAD], include_url=False
            ) if self._discovery_data else ""
            new_origin_info = get_origin_log_string(
                payload.discovery_data[ATTR_DISCOVERY_PAYLOAD], include_url=False
            )
            new_origin_support_url = get_origin_support_url(payload.discovery_data[ATTR_DISCOVERY_PAYLOAD])
            if new_origin_support_url:
                get_support = f'for support visit {new_origin_support_url}'
            else:
                get_support = 'for documentation on migration to device schema or rollback to discovery schema, visit https://www.home-assistant.io/integrations/mqtt/#migration-from-single-component-to-device-based-discovery'
            _LOGGER.warning(
                'Received a conflicting MQTT discovery message for entity %s; the entity was previously discovered on topic %s%s; the conflicting discovery message was received on topic %s%s; %s',
                self.entity_id,
                self._discovery_data.discovery_data[ATTR_DISCOVERY_TOPIC],
                old_origin_info,
                new_discovery_topic,
                new_origin_info,
                get_support,
            )
            send_discovery_done(self.hass, self._discovery_data.discovery_data)
            return
        debug_info.update_entity_discovery_data(self.hass, payload, self.entity_id)
        if not payload:
            if self._migrate_discovery is None:
                _LOGGER.info('Removing component: %s', self.entity_id)
            else:
                _LOGGER.info('Unloading component: %s', self.entity_id)
            self.hass.async_create_task(self._async_process_discovery_update_and_remove())
        elif self._discovery_update:
            if old_payload != payload:
                _LOGGER.info('Updating component: %s', self.entity_id)
                self.hass.async_create_task(
                    self._async_process_discovery_update(payload, self._discovery_update, self._discovery_data.discovery_data)
                )
            else:
                _LOGGER.debug('Ignoring unchanged update for: %s', self.entity_id)
                send_discovery_done(self.hass, self._discovery_data.discovery_data)
        else:
            send_discovery_done(self.hass, self._discovery_data.discovery_data)
            _LOGGER.debug('Ignoring unchanged update for: %s', self.entity_id)
            return

    async def async_removed_from_registry(self) -> None:
        """Clear retained discovery topic in broker."""
        if not self._removed_from_hass and self._discovery_data is not None:
            self._cleanup_discovery_on_remove()
            await async_remove_discovery_payload(self.hass, self._discovery_data.discovery_data)

    @final
    async def add_to_platform_finish(self) -> None:
        """Finish adding entity to platform."""
        await super().add_to_platform_finish()
        if self._discovery_data is not None:
            send_discovery_done(self.hass, self._discovery_data.discovery_data)

    @callback
    def add_to_platform_abort(self) -> None:
        """Abort adding an entity to a platform."""
        if self._discovery_data is not None:
            discovery_hash = self._discovery_data.discovery_data[ATTR_DISCOVERY_HASH]
            if self.registry_entry is not None:
                self._registry_hooks[discovery_hash] = async_track_entity_registry_updated_event(
                    self.hass,
                    self.entity_id,
                    partial(
                        async_clear_discovery_topic_if_entity_removed,
                        self.hass,
                        self._discovery_data.discovery_data,
                    ),
                )
            stop_discovery_updates(self.hass, self._discovery_data.discovery_data, self._remove_discovery_updated)
            send_discovery_done(self.hass, self._discovery_data.discovery_data)
        super().add_to_platform_abort()

    async def async_will_remove_from_hass(self) -> None:
        """Stop listening to signal and cleanup discovery data.."""
        self._cleanup_discovery_on_remove()

    def _cleanup_discovery_on_remove(self) -> None:
        """Stop listening to signal and cleanup discovery data."""
        if self._discovery_data and (not self._removed_from_hass):
            stop_discovery_updates(self.hass, self._discovery_data.discovery_data, self._remove_discovery_updated)
            self._removed_from_hass = True

def device_info_from_specifications(specifications: dict[str, Any]) -> Optional[DeviceInfo]:
    """Return a device description for device registry."""
    if not specifications:
        return None
    info = DeviceInfo(
        identifiers={(DOMAIN, id_) for id_ in specifications[CONF_IDENTIFIERS]},
        connections={(conn_[0], conn_[1]) for conn_ in specifications[CONF_CONNECTIONS]},
    )
    if CONF_MANUFACTURER in specifications:
        info[ATTR_MANUFACTURER] = specifications[CONF_MANUFACTURER]
    if CONF_MODEL in specifications:
        info[ATTR_MODEL] = specifications[CONF_MODEL]
    if CONF_MODEL_ID in specifications:
        info[ATTR_MODEL_ID] = specifications[CONF_MODEL_ID]
    if CONF_NAME in specifications:
        info[ATTR_NAME] = specifications[CONF_NAME]
    if CONF_HW_VERSION in specifications:
        info[ATTR_HW_VERSION] = specifications[CONF_HW_VERSION]
    if CONF_SERIAL_NUMBER in specifications:
        info[ATTR_SERIAL_NUMBER] = specifications[CONF_SERIAL_NUMBER]
    if CONF_SW_VERSION in specifications:
        info[ATTR_SW_VERSION] = specifications[CONF_SW_VERSION]
    if CONF_VIA_DEVICE in specifications:
        info[ATTR_VIA_DEVICE] = (DOMAIN, specifications[CONF_VIA_DEVICE])
    if CONF_SUGGESTED_AREA in specifications:
        info[ATTR_SUGGESTED_AREA] = specifications[CONF_SUGGESTED_AREA]
    if CONF_CONFIGURATION_URL in specifications:
        info[ATTR_CONFIGURATION_URL] = specifications[CONF_CONFIGURATION_URL]
    return info

@callback
def ensure_via_device_exists(
    hass: HomeAssistant,
    device_info: Optional[dict[str, Any]],
    config_entry: ConfigEntry,
) -> None:
    """Ensure the via device is in the device registry."""
    if device_info is None or CONF_VIA_DEVICE not in device_info or (
        device_registry := dr.async_get(hass)
    ).async_get_device(identifiers={device_info['via_device']}):
        return
    _LOGGER.debug(
        'Device identifier %s via_device reference from device_info %s not found in the Device Registry, creating new entry',
        device_info['via_device'],
        device_info,
    )
    device_registry.async_get_or_create(
        config_entry_id=config_entry.entry_id, identifiers={device_info['via_device']}
    )

class MqttEntityDeviceInfo(Entity):
    """Mixin used for mqtt platforms that support the device registry."""

    def __init__(
        self,
        specifications: Optional[dict[str, Any]],
        config_entry: ConfigEntry,
    ) -> None:
        """Initialize the device mixin."""
        self._device_specifications: Optional[dict[str, Any]] = specifications
        self._config_entry: ConfigEntry = config_entry

    def device_info_discovery_update(self, config: ConfigType) -> None:
        """Handle updated discovery message."""
        self._device_specifications = config.get(CONF_DEVICE)
        device_registry = dr.async_get(self.hass)
        config_entry_id = self._config_entry.entry_id
        device_info = self.device_info
        if device_info is not None:
            ensure_via_device_exists(self.hass, device_info, self._config_entry)
            device_registry.async_get_or_create(config_entry_id=config_entry_id, **device_info)

    @property
    def device_info(self) -> Optional[DeviceInfo]:
        """Return a device description for device registry."""
        return device_info_from_specifications(self._device_specifications)

class MqttEntity(
    MqttAttributesMixin,
    MqttAvailabilityMixin,
    MqttDiscoveryUpdateMixin,
    MqttEntityDeviceInfo,
):
    """Representation of an MQTT entity."""
    _attr_force_update: bool = False
    _attr_has_entity_name: bool = True
    _attr_should_poll: bool = False

    def __init__(
        self,
        hass: HomeAssistant,
        config: ConfigType,
        config_entry: ConfigEntry,
        discovery_data: Optional[MQTTDiscoveryPayload],
    ) -> None:
        """Init the MQTT Entity."""
        self.hass = hass
        self._config = config
        self._attr_unique_id: Optional[str] = config.get(CONF_UNIQUE_ID)
        self._sub_state: dict[str, Any] = {}
        self._discovery: bool = discovery_data is not None
        self._setup_from_config(self._config)
        self._setup_common_attributes_from_config(self._config)
        self._init_entity_id()
        MqttAttributesMixin.__init__(self, config)
        MqttAvailabilityMixin.__init__(self, config)
        MqttDiscoveryUpdateMixin.__init__(self, hass, discovery_data, self.discovery_update)
        MqttEntityDeviceInfo.__init__(self, config.get(CONF_DEVICE), config_entry)
        ensure_via_device_exists(self.hass, self.device_info, self._config_entry)

    def _init_entity_id(self) -> None:
        """Set entity_id from object_id if defined in config."""
        init_entity_id_from_config(self.hass, self, self._config, self._entity_id_format)

    @final
    async def async_added_to_hass(self) -> None:
        """Subscribe to MQTT events."""
        await super().async_added_to_hass()
        self._subscriptions: dict[str, dict[str, Any]] = {}
        self._prepare_subscribe_topics()
        if self._subscriptions:
            self._sub_state = subscription.async_prepare_subscribe_topics(
                self.hass, self._sub_state, self._subscriptions
            )
        await self._subscribe_topics()
        await self.mqtt_async_added_to_hass()

    async def mqtt_async_added_to_hass(self) -> None:
        """Call before the discovery message is acknowledged.

        To be extended by subclasses.
        """
        pass

    async def discovery_update(self, discovery_payload: MQTTDiscoveryPayload) -> None:
        """Handle updated discovery message."""
        try:
            config = self.config_schema()(discovery_payload)
        except vol.Invalid as err:
            async_handle_schema_error(discovery_payload, err)
            return
        self._config = config
        self._setup_from_config(self._config)
        self._setup_common_attributes_from_config(self._config)
        self.attributes_prepare_discovery_update(config)
        self.availability_prepare_discovery_update(config)
        self.device_info_discovery_update(config)
        self._subscriptions = {}
        self._prepare_subscribe_topics()
        if self._subscriptions:
            self._sub_state = subscription.async_prepare_subscribe_topics(
                self.hass, self._sub_state, self._subscriptions
            )
        await self.attributes_discovery_update(config)
        await self.availability_discovery_update(config)
        await self._subscribe_topics()
        self.async_write_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe when removed."""
        self._sub_state = subscription.async_unsubscribe_topics(self.hass, self._sub_state)
        await MqttAttributesMixin.async_will_remove_from_hass(self)
        await MqttAvailabilityMixin.async_will_remove_from_hass(self)
        await MqttDiscoveryUpdateMixin.async_will_remove_from_hass(self)
        debug_info.remove_entity_data(self.hass, self.entity_id)

    async def async_publish(
        self,
        topic: str,
        payload: PublishPayloadType,
        qos: int = 0,
        retain: bool = False,
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        """Publish message to an MQTT topic."""
        log_message(self.hass, self.entity_id, topic, payload, qos, retain)
        await async_publish(self.hass, topic, payload, qos, retain, encoding)

    async def async_publish_with_config(
        self,
        topic: str,
        payload: PublishPayloadType,
    ) -> None:
        """Publish payload to a topic using config."""
        await self.async_publish(
            topic,
            payload,
            self._config[CONF_QOS],
            self._config[CONF_RETAIN],
            self._config[CONF_ENCODING],
        )

    @staticmethod
    @abstractmethod
    def config_schema() -> Callable[[MQTTDiscoveryPayload], ConfigType]:
        """Return the config schema."""

    def _set_entity_name(self, config: ConfigType) -> None:
        """Help setting the entity name if needed."""
        entity_name = config.get(CONF_NAME, UNDEFINED)
        if entity_name is not UNDEFINED:
            self._attr_name = entity_name
        elif not self._default_to_device_class_name():
            self._attr_name = self._default_name
        elif hasattr(self, '_attr_name'):
            delattr(self, '_attr_name')
        if CONF_DEVICE in config and CONF_NAME not in config[CONF_DEVICE]:
            _LOGGER.info(
                "MQTT device information always needs to include a name, got %s, if device information is shared between multiple entities, the device name must be included in each entity's device configuration",
                config,
            )

    def _setup_common_attributes_from_config(self, config: ConfigType) -> None:
        """(Re)Setup the common attributes for the entity."""
        self._attr_entity_category: Optional[str] = config.get(CONF_ENTITY_CATEGORY)
        self._attr_entity_registry_enabled_default: bool = bool(config.get(CONF_ENABLED_BY_DEFAULT))
        self._attr_icon: Optional[str] = config.get(CONF_ICON)
        self._attr_entity_picture: Optional[str] = config.get(CONF_ENTITY_PICTURE)
        self._set_entity_name(config)

    def _setup_from_config(self, config: ConfigType) -> None:
        """(Re)Setup the entity."""
        pass

    @abstractmethod
    @callback
    def _prepare_subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""

    @abstractmethod
    async def _subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""

    @callback
    def _attrs_have_changed(self, attrs_snapshot: Iterable[tuple[str, Any]]) -> bool:
        """Return True if attributes on entity changed or if update is forced."""
        if self._attr_force_update:
            return True
        for attribute, last_value in attrs_snapshot:
            if getattr(self, attribute, UNDEFINED) != last_value:
                return True
        return False

    @callback
    def _message_callback(
        self,
        msg_callback: Callable[[ReceiveMessage], None],
        attributes: Optional[set[str]],
        msg: ReceiveMessage,
    ) -> None:
        """Process the message callback."""
        if attributes is not None:
            attrs_snapshot = tuple(
                (attribute, getattr(self, attribute, UNDEFINED)) for attribute in attributes
            )
        mqtt_data = self.hass.data[DATA_MQTT]
        messages = mqtt_data.debug_info_entities[self.entity_id]['subscriptions'][msg.subscribed_topic]['messages']
        if msg not in messages:
            messages.append(msg)
        try:
            msg_callback(msg)
        except MqttValueTemplateException as exc:
            _LOGGER.warning(exc)
            return
        if attributes is not None and self._attrs_have_changed(attrs_snapshot):
            mqtt_data.state_write_requests.write_state_request(self)

    def add_subscription(
        self,
        state_topic_config_key: str,
        msg_callback: Callable[[ReceiveMessage], None],
        tracked_attributes: set[str],
        disable_encoding: bool = False,
    ) -> bool:
        """Add a subscription."""
        qos: int = self._config[CONF_QOS]
        encoding: Optional[str] = None
        if not disable_encoding:
            encoding = self._config[CONF_ENCODING] or None
        if state_topic_config_key in self._config and self._config[state_topic_config_key] is not None:
            self._subscriptions[state_topic_config_key] = {
                'topic': self._config[state_topic_config_key],
                'msg_callback': partial(
                    self._message_callback,
                    msg_callback,
                    tracked_attributes,
                ),
                'entity_id': self.entity_id,
                'qos': qos,
                'encoding': encoding,
                'job_type': HassJobType.Callback,
            }
            return True
        return False

def update_device(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    config: ConfigType,
) -> Optional[str]:
    """Update device registry."""
    if CONF_DEVICE not in config:
        return None
    device: Optional[DeviceEntry] = None
    device_registry = dr.async_get(hass)
    config_entry_id = config_entry.entry_id
    device_info = device_info_from_specifications(config[CONF_DEVICE])
    ensure_via_device_exists(hass, device_info, config_entry)
    if config_entry_id is not None and device_info is not None:
        update_device_info: dict[str, Any] = cast(dict[str, Any], device_info)
        update_device_info['config_entry_id'] = config_entry_id
        device = device_registry.async_get_or_create(**update_device_info)
    return device.id if device else None

@callback
def async_removed_from_device(
    hass: HomeAssistant,
    event: Event,
    mqtt_device_id: str,
    config_entry_id: str,
) -> bool:
    """Check if the passed event indicates MQTT was removed from a device."""
    if event.data['action'] == 'update':
        if 'config_entries' not in event.data['changes']:
            return False
        device_registry = dr.async_get(hass)
        if (
            device_entry := device_registry.async_get(mqtt_device_id)
        ) and config_entry_id in device_entry.config_entries:
            return False
    return True
