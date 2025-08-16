from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import datetime as dt
import time
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Tuple

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.typing import DiscoveryInfoType
from homeassistant.util import dt as dt_util

from .const import ATTR_DISCOVERY_PAYLOAD, ATTR_DISCOVERY_TOPIC
from .models import DATA_MQTT, PublishPayloadType

STORED_MESSAGES: int = 10


@dataclass
class TimestampedPublishMessage:
    topic: str
    payload: PublishPayloadType
    qos: int
    retain: bool
    timestamp: float


def log_message(
    hass: HomeAssistant,
    entity_id: str,
    topic: str,
    payload: PublishPayloadType,
    qos: int,
    retain: bool,
) -> None:
    entity_info: Dict[str, Any] = hass.data[DATA_MQTT].debug_info_entities.setdefault(
        entity_id, {"subscriptions": {}, "discovery_data": {}, "transmitted": {}}
    )
    if topic not in entity_info["transmitted"]:
        entity_info["transmitted"][topic] = {
            "messages": deque([], STORED_MESSAGES),
        }
    msg: TimestampedPublishMessage = TimestampedPublishMessage(
        topic, payload, qos, retain, timestamp=time.monotonic()
    )
    entity_info["transmitted"][topic]["messages"].append(msg)


def add_subscription(
    hass: HomeAssistant, subscription: str, entity_id: str | None
) -> None:
    if entity_id:
        entity_info: Dict[str, Any] = hass.data[DATA_MQTT].debug_info_entities.setdefault(
            entity_id, {"subscriptions": {}, "discovery_data": {}, "transmitted": {}}
        )
        if subscription not in entity_info["subscriptions"]:
            entity_info["subscriptions"][subscription] = {
                "count": 1,
                "messages": deque([], STORED_MESSAGES),
            }
        else:
            entity_info["subscriptions"][subscription]["count"] += 1


def remove_subscription(
    hass: HomeAssistant, subscription: str, entity_id: str | None
) -> None:
    if entity_id and entity_id in (debug_info_entities := hass.data[DATA_MQTT].debug_info_entities):
        subscriptions: Dict[str, Any] = debug_info_entities[entity_id]["subscriptions"]
        subscriptions[subscription]["count"] -= 1
        if not subscriptions[subscription]["count"]:
            del subscriptions[subscription]


def add_entity_discovery_data(
    hass: HomeAssistant, discovery_data: DiscoveryInfoType, entity_id: str
) -> None:
    entity_info: Dict[str, Any] = hass.data[DATA_MQTT].debug_info_entities.setdefault(
        entity_id, {"subscriptions": {}, "discovery_data": {}, "transmitted": {}}
    )
    entity_info["discovery_data"] = discovery_data


def update_entity_discovery_data(
    hass: HomeAssistant, discovery_payload: DiscoveryInfoType, entity_id: str
) -> None:
    discovery_data: Dict[str, Any] = hass.data[DATA_MQTT].debug_info_entities[entity_id]["discovery_data"]
    if TYPE_CHECKING:
        assert discovery_data is not None
    discovery_data[ATTR_DISCOVERY_PAYLOAD] = discovery_payload


def remove_entity_data(hass: HomeAssistant, entity_id: str) -> None:
    if entity_id in (debug_info_entities := hass.data[DATA_MQTT].debug_info_entities):
        del debug_info_entities[entity_id]


def add_trigger_discovery_data(
    hass: HomeAssistant,
    discovery_hash: Tuple[str, str],
    discovery_data: DiscoveryInfoType,
    device_id: str,
) -> None:
    hass.data[DATA_MQTT].debug_info_triggers[discovery_hash] = {
        "device_id": device_id,
        "discovery_data": discovery_data,
    }


def update_trigger_discovery_data(
    hass: HomeAssistant,
    discovery_hash: Tuple[str, str],
    discovery_payload: DiscoveryInfoType,
) -> None:
    hass.data[DATA_MQTT].debug_info_triggers[discovery_hash]["discovery_data"][
        ATTR_DISCOVERY_PAYLOAD
    ] = discovery_payload


def remove_trigger_discovery_data(
    hass: HomeAssistant, discovery_hash: Tuple[str, str]
) -> None:
    hass.data[DATA_MQTT].debug_info_triggers.pop(discovery_hash, None)


def _info_for_entity(hass: HomeAssistant, entity_id: str) -> Dict[str, Any]:
    entity_info: Dict[str, Any] = hass.data[DATA_MQTT].debug_info_entities[entity_id]
    monotonic_time_diff: float = time.time() - time.monotonic()
    subscriptions: List[Dict[str, Any]] = [
        {
            "topic": topic,
            "messages": [
                {
                    "payload": str(msg.payload),
                    "qos": msg.qos,
                    "retain": msg.retain,
                    "time": dt_util.utc_from_timestamp(
                        msg.timestamp + monotonic_time_diff,
                        tz=dt.UTC,
                    ),
                    "topic": msg.topic,
                }
                for msg in subscription["messages"]
            ],
        }
        for topic, subscription in entity_info["subscriptions"].items()
    ]
    transmitted: List[Dict[str, Any]] = [
        {
            "topic": topic,
            "messages": [
                {
                    "payload": str(msg.payload),
                    "qos": msg.qos,
                    "retain": msg.retain,
                    "time": dt_util.utc_from_timestamp(
                        msg.timestamp + monotonic_time_diff,
                        tz=dt.UTC,
                    ),
                    "topic": msg.topic,
                }
                for msg in subscription["messages"]
            ],
        }
        for topic, subscription in entity_info["transmitted"].items()
    ]
    discovery_data: Dict[str, Any] = {
        "topic": entity_info["discovery_data"].get(ATTR_DISCOVERY_TOPIC, ""),
        "payload": entity_info["discovery_data"].get(ATTR_DISCOVERY_PAYLOAD, ""),
    }

    return {
        "entity_id": entity_id,
        "subscriptions": subscriptions,
        "discovery_data": discovery_data,
        "transmitted": transmitted,
    }


def _info_for_trigger(
    hass: HomeAssistant, trigger_key: Tuple[str, str]
) -> Dict[str, Any]:
    trigger: Dict[str, Any] = hass.data[DATA_MQTT].debug_info_triggers[trigger_key]
    discovery_data: Dict[str, Any] | None = None
    if trigger["discovery_data"] is not None:
        discovery_data = {
            "topic": trigger["discovery_data"][ATTR_DISCOVERY_TOPIC],
            "payload": trigger["discovery_data"][ATTR_DISCOVERY_PAYLOAD],
        }
    return {"discovery_data": discovery_data, "trigger_key": trigger_key}


def info_for_config_entry(hass: HomeAssistant) -> Dict[str, List[Any]]:
    mqtt_data: Any = hass.data[DATA_MQTT]
    mqtt_info: Dict[str, List[Any]] = {"entities": [], "triggers": []}

    mqtt_info["entities"].extend(
        _info_for_entity(hass, entity_id) for entity_id in mqtt_data.debug_info_entities
    )

    mqtt_info["triggers"].extend(
        _info_for_trigger(hass, trigger_key)
        for trigger_key in mqtt_data.debug_info_triggers
    )

    return mqtt_info


def info_for_device(hass: HomeAssistant, device_id: str) -> Dict[str, List[Any]]:
    mqtt_data: Any = hass.data[DATA_MQTT]
    mqtt_info: Dict[str, List[Any]] = {"entities": [], "triggers": []}
    entity_registry = er.async_get(hass)

    entries: List[Any] = er.async_entries_for_device(
        entity_registry, device_id, include_disabled_entities=True
    )
    mqtt_info["entities"].extend(
        _info_for_entity(hass, entry.entity_id)
        for entry in entries
        if entry.entity_id in mqtt_data.debug_info_entities
    )

    mqtt_info["triggers"].extend(
        _info_for_trigger(hass, trigger_key)
        for trigger_key, trigger in mqtt_data.debug_info_triggers.items()
        if trigger["device_id"] == device_id
    )

    return mqtt_info