"""Helper to handle a set of topics to subscribe to."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import datetime as dt
import time
from typing import TYPE_CHECKING, Any
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.typing import DiscoveryInfoType
from homeassistant.util import dt as dt_util
from .const import ATTR_DISCOVERY_PAYLOAD, ATTR_DISCOVERY_TOPIC
from .models import DATA_MQTT, PublishPayloadType
STORED_MESSAGES = 10

@dataclass
class TimestampedPublishMessage:
    """MQTT Message."""

def log_message(hass: Union[str, None, homeassistancore.HomeAssistant], entity_id: Union[str, None, homeassistancore.HomeAssistant], topic: bool, payload: Union[str, int, dict[str, int]], qos: Union[str, int, dict[str, int]], retain: Union[str, int, dict[str, int]]) -> None:
    """Log an outgoing MQTT message."""
    entity_info = hass.data[DATA_MQTT].debug_info_entities.setdefault(entity_id, {'subscriptions': {}, 'discovery_data': {}, 'transmitted': {}})
    if topic not in entity_info['transmitted']:
        entity_info['transmitted'][topic] = {'messages': deque([], STORED_MESSAGES)}
    msg = TimestampedPublishMessage(topic, payload, qos, retain, timestamp=time.monotonic())
    entity_info['transmitted'][topic]['messages'].append(msg)

def add_subscription(hass: homeassistancore.HomeAssistant, subscription: homeassistancore.HomeAssistant, entity_id: homeassistancore.HomeAssistant) -> None:
    """Prepare debug data for subscription."""
    if entity_id:
        entity_info = hass.data[DATA_MQTT].debug_info_entities.setdefault(entity_id, {'subscriptions': {}, 'discovery_data': {}, 'transmitted': {}})
        if subscription not in entity_info['subscriptions']:
            entity_info['subscriptions'][subscription] = {'count': 1, 'messages': deque([], STORED_MESSAGES)}
        else:
            entity_info['subscriptions'][subscription]['count'] += 1

def remove_subscription(hass: Union[str, homeassistancore.HomeAssistant, None], subscription: Union[str, None], entity_id: Union[str, homeassistancore.HomeAssistant, None]) -> None:
    """Remove debug data for subscription if it exists."""
    if entity_id and entity_id in (debug_info_entities := hass.data[DATA_MQTT].debug_info_entities):
        subscriptions = debug_info_entities[entity_id]['subscriptions']
        subscriptions[subscription]['count'] -= 1
        if not subscriptions[subscription]['count']:
            del subscriptions[subscription]

def add_entity_discovery_data(hass: Union[str, homeassistancore.HomeAssistant], discovery_data: Union[str, dict], entity_id: Union[str, homeassistancore.HomeAssistant]) -> None:
    """Add discovery data."""
    entity_info = hass.data[DATA_MQTT].debug_info_entities.setdefault(entity_id, {'subscriptions': {}, 'discovery_data': {}, 'transmitted': {}})
    entity_info['discovery_data'] = discovery_data

def update_entity_discovery_data(hass: Union[homeassistancore.HomeAssistant, str, dict, None], discovery_payload: Union[str, dict, dict[str, str]], entity_id: Union[homeassistancore.HomeAssistant, str, dict, None]) -> None:
    """Update discovery data."""
    discovery_data = hass.data[DATA_MQTT].debug_info_entities[entity_id]['discovery_data']
    if TYPE_CHECKING:
        assert discovery_data is not None
    discovery_data[ATTR_DISCOVERY_PAYLOAD] = discovery_payload

def remove_entity_data(hass: Union[homeassistancore.HomeAssistant, str], entity_id: Union[homeassistancore.HomeAssistant, str]) -> None:
    """Remove discovery data."""
    if entity_id in (debug_info_entities := hass.data[DATA_MQTT].debug_info_entities):
        del debug_info_entities[entity_id]

def add_trigger_discovery_data(hass: homeassistancore.HomeAssistant, discovery_hash: homeassistancore.HomeAssistant, discovery_data: homeassistancore.HomeAssistant, device_id: homeassistancore.HomeAssistant) -> None:
    """Add discovery data."""
    hass.data[DATA_MQTT].debug_info_triggers[discovery_hash] = {'device_id': device_id, 'discovery_data': discovery_data}

def update_trigger_discovery_data(hass: Union[homeassistancore.HomeAssistant, dict], discovery_hash: Union[homeassistancore.HomeAssistant, dict], discovery_payload: Union[homeassistancore.HomeAssistant, dict]) -> None:
    """Update discovery data."""
    hass.data[DATA_MQTT].debug_info_triggers[discovery_hash]['discovery_data'][ATTR_DISCOVERY_PAYLOAD] = discovery_payload

def remove_trigger_discovery_data(hass: Union[int, str, homeassistancore.HomeAssistant], discovery_hash: Union[int, str, homeassistancore.HomeAssistant]) -> None:
    """Remove discovery data."""
    hass.data[DATA_MQTT].debug_info_triggers.pop(discovery_hash, None)

def _info_for_entity(hass: Union[homeassistancore.HomeAssistant, str], entity_id: Union[homeassistancore.HomeAssistant, str]) -> dict[typing.Text, typing.Union[homeassistancore.HomeAssistant,str,list[dict[typing.Text, list[dict[typing.Text, str]]]],dict[typing.Text, ]]]:
    entity_info = hass.data[DATA_MQTT].debug_info_entities[entity_id]
    monotonic_time_diff = time.time() - time.monotonic()
    subscriptions = [{'topic': topic, 'messages': [{'payload': str(msg.payload), 'qos': msg.qos, 'retain': msg.retain, 'time': dt_util.utc_from_timestamp(msg.timestamp + monotonic_time_diff, tz=dt.UTC), 'topic': msg.topic} for msg in subscription['messages']]} for topic, subscription in entity_info['subscriptions'].items()]
    transmitted = [{'topic': topic, 'messages': [{'payload': str(msg.payload), 'qos': msg.qos, 'retain': msg.retain, 'time': dt_util.utc_from_timestamp(msg.timestamp + monotonic_time_diff, tz=dt.UTC), 'topic': msg.topic} for msg in subscription['messages']]} for topic, subscription in entity_info['transmitted'].items()]
    discovery_data = {'topic': entity_info['discovery_data'].get(ATTR_DISCOVERY_TOPIC, ''), 'payload': entity_info['discovery_data'].get(ATTR_DISCOVERY_PAYLOAD, '')}
    return {'entity_id': entity_id, 'subscriptions': subscriptions, 'discovery_data': discovery_data, 'transmitted': transmitted}

def _info_for_trigger(hass: Union[int, str], trigger_key: Union[int, str, homeassistancore.HomeAssistant]) -> dict[typing.Text, typing.Union[None,dict[typing.Text, ],int,str,homeassistancore.HomeAssistant]]:
    trigger = hass.data[DATA_MQTT].debug_info_triggers[trigger_key]
    discovery_data = None
    if trigger['discovery_data'] is not None:
        discovery_data = {'topic': trigger['discovery_data'][ATTR_DISCOVERY_TOPIC], 'payload': trigger['discovery_data'][ATTR_DISCOVERY_PAYLOAD]}
    return {'discovery_data': discovery_data, 'trigger_key': trigger_key}

def info_for_config_entry(hass: homeassistancore.HomeAssistant) -> dict[typing.Text, list]:
    """Get debug info for all entities and triggers."""
    mqtt_data = hass.data[DATA_MQTT]
    mqtt_info = {'entities': [], 'triggers': []}
    mqtt_info['entities'].extend((_info_for_entity(hass, entity_id) for entity_id in mqtt_data.debug_info_entities))
    mqtt_info['triggers'].extend((_info_for_trigger(hass, trigger_key) for trigger_key in mqtt_data.debug_info_triggers))
    return mqtt_info

def info_for_device(hass: Union[str, homeassistancore.HomeAssistant], device_id: Union[str, homeassistancore.HomeAssistant]) -> dict[typing.Text, list]:
    """Get debug info for a device."""
    mqtt_data = hass.data[DATA_MQTT]
    mqtt_info = {'entities': [], 'triggers': []}
    entity_registry = er.async_get(hass)
    entries = er.async_entries_for_device(entity_registry, device_id, include_disabled_entities=True)
    mqtt_info['entities'].extend((_info_for_entity(hass, entry.entity_id) for entry in entries if entry.entity_id in mqtt_data.debug_info_entities))
    mqtt_info['triggers'].extend((_info_for_trigger(hass, trigger_key) for trigger_key, trigger in mqtt_data.debug_info_triggers.items() if trigger['device_id'] == device_id))
    return mqtt_info