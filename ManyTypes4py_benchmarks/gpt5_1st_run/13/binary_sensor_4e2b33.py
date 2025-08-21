"""Support for exposing Concord232 elements as sensors."""
from __future__ import annotations

import datetime
import logging
from typing import Final, Protocol, TypedDict, cast

from concord232 import client as concord232_client
import requests
import voluptuous as vol

from homeassistant.components.binary_sensor import (
    DEVICE_CLASSES_SCHEMA as BINARY_SENSOR_DEVICE_CLASSES_SCHEMA,
    PLATFORM_SCHEMA as BINARY_SENSOR_PLATFORM_SCHEMA,
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

CONF_EXCLUDE_ZONES: Final[str] = "exclude_zones"
CONF_ZONE_TYPES: Final[str] = "zone_types"

DEFAULT_HOST: Final[str] = "localhost"
DEFAULT_NAME: Final[str] = "Alarm"
DEFAULT_PORT: Final[str] = "5007"
DEFAULT_SSL: Final[bool] = False

SCAN_INTERVAL: datetime.timedelta = datetime.timedelta(seconds=10)

ZONE_TYPES_SCHEMA = vol.Schema({cv.positive_int: BINARY_SENSOR_DEVICE_CLASSES_SCHEMA})

PLATFORM_SCHEMA = BINARY_SENSOR_PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_EXCLUDE_ZONES, default=[]): vol.All(
            cv.ensure_list, [cv.positive_int]
        ),
        vol.Optional(CONF_HOST, default=DEFAULT_HOST): cv.string,
        vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
        vol.Optional(CONF_ZONE_TYPES, default={}): ZONE_TYPES_SCHEMA,
    }
)


class Zone(TypedDict):
    number: int
    name: str
    state: str


class ConcordClientProtocol(Protocol):
    zones: list[Zone]
    last_zone_update: datetime.datetime

    def list_zones(self) -> list[Zone]:
        ...


DeviceClassType = BinarySensorDeviceClass | str


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Concord232 binary sensor platform."""
    host: str = cast(str, config[CONF_HOST])
    port: int | str = cast(int | str, config[CONF_PORT])
    exclude: list[int] = cast(list[int], config[CONF_EXCLUDE_ZONES])
    zone_types: dict[int, DeviceClassType] = cast(
        dict[int, DeviceClassType], config[CONF_ZONE_TYPES]
    )

    sensors: list[Concord232ZoneSensor] = []
    try:
        _LOGGER.debug("Initializing client")
        raw_client = concord232_client.Client(f"http://{host}:{port}")
        client: ConcordClientProtocol = cast(ConcordClientProtocol, raw_client)
        client.zones = client.list_zones()
        client.last_zone_update = dt_util.utcnow()
    except requests.exceptions.ConnectionError as ex:
        _LOGGER.error("Unable to connect to Concord232: %s", str(ex))
        return

    client.zones.sort(key=lambda zone: zone["number"])
    for zone in client.zones:
        _LOGGER.debug("Loading Zone found: %s", zone["name"])
        if zone["number"] not in exclude:
            sensors.append(
                Concord232ZoneSensor(
                    hass,
                    client,
                    zone,
                    zone_types.get(zone["number"], get_opening_type(zone)),
                )
            )
    add_entities(sensors, True)


def get_opening_type(zone: Zone) -> DeviceClassType:
    """Return the result of the type guessing from name."""
    if "MOTION" in zone["name"]:
        return BinarySensorDeviceClass.MOTION
    if "KEY" in zone["name"]:
        return BinarySensorDeviceClass.SAFETY
    if "SMOKE" in zone["name"]:
        return BinarySensorDeviceClass.SMOKE
    if "WATER" in zone["name"]:
        return "water"
    return BinarySensorDeviceClass.OPENING


class Concord232ZoneSensor(BinarySensorEntity):
    """Representation of a Concord232 zone as a sensor."""

    def __init__(
        self,
        hass: HomeAssistant,
        client: ConcordClientProtocol,
        zone: Zone,
        zone_type: DeviceClassType,
    ) -> None:
        """Initialize the Concord232 binary sensor."""
        self._hass: HomeAssistant = hass
        self._client: ConcordClientProtocol = client
        self._zone: Zone = zone
        self._number: int = zone["number"]
        self._zone_type: DeviceClassType = zone_type

    @property
    def device_class(self) -> DeviceClassType:
        """Return the class of this sensor, from DEVICE_CLASSES."""
        return self._zone_type

    @property
    def name(self) -> str:
        """Return the name of the binary sensor."""
        return self._zone["name"]

    @property
    def is_on(self) -> bool:
        """Return true if the binary sensor is on."""
        return bool(self._zone["state"] != "Normal")

    def update(self) -> None:
        """Get updated stats from API."""
        last_update: datetime.timedelta = dt_util.utcnow() - self._client.last_zone_update
        _LOGGER.debug("Zone: %s ", self._zone)
        if last_update > datetime.timedelta(seconds=1):
            self._client.zones = self._client.list_zones()
            self._client.last_zone_update = dt_util.utcnow()
            _LOGGER.debug("Updated from zone: %s", self._zone["name"])
        if hasattr(self._client, "zones"):
            found = next(
                (x for x in self._client.zones if x["number"] == self._number), None
            )
            if found is not None:
                self._zone = found