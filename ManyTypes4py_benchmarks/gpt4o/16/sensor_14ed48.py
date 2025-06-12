"""Support for Huawei LTE sensors."""
from __future__ import annotations
from bisect import bisect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import re
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN, SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfDataRate, UnitOfFrequency, UnitOfInformation, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from . import Router
from .const import DOMAIN, KEY_DEVICE_INFORMATION, KEY_DEVICE_SIGNAL, KEY_MONITORING_CHECK_NOTIFICATIONS, KEY_MONITORING_MONTH_STATISTICS, KEY_MONITORING_STATUS, KEY_MONITORING_TRAFFIC_STATISTICS, KEY_NET_CURRENT_PLMN, KEY_NET_NET_MODE, KEY_SMS_SMS_COUNT, SENSOR_KEYS
from .entity import HuaweiLteBaseEntityWithDevice

_LOGGER = logging.getLogger(__name__)

def format_default(value: StateType) -> tuple[StateType, str | None]:
    """Format value."""
    unit: str | None = None
    if value is not None:
        if (match := re.match(r'((&[gl]t;|[><])=?)?(?P<value>.+?)\s*(?P<unit>[a-zA-Z]+)\s*$', str(value))):
            try:
                value = float(match.group('value'))
                unit = match.group('unit')
            except ValueError:
                pass
    return (value, unit)

def format_freq_mhz(value: StateType) -> tuple[float | None, UnitOfFrequency]:
    """Format a frequency value for which source is in tenths of MHz."""
    return (float(value) / 10 if value is not None else None, UnitOfFrequency.MEGAHERTZ)

def format_last_reset_elapsed_seconds(value: StateType) -> datetime | None:
    """Convert elapsed seconds to last reset datetime."""
    if value is None:
        return None
    try:
        last_reset = datetime.now() - timedelta(seconds=int(value))
        last_reset.replace(microsecond=0)
    except ValueError:
        return None
    return last_reset

def signal_icon(limits: Sequence[int], value: int | None) -> str:
    """Get signal icon."""
    return ('mdi:signal-cellular-outline', 'mdi:signal-cellular-1', 'mdi:signal-cellular-2', 'mdi:signal-cellular-3')[bisect(limits, value if value is not None else -1000)]

def bandwidth_icon(limits: Sequence[int], value: int | None) -> str:
    """Get bandwidth icon."""
    return ('mdi:speedometer-slow', 'mdi:speedometer-medium', 'mdi:speedometer')[bisect(limits, value if value is not None else -1000)]

@dataclass
class HuaweiSensorGroup:
    """Class describing Huawei LTE sensor groups."""
    include: re.Pattern | None = None
    exclude: re.Pattern | None = None

@dataclass(frozen=True)
class HuaweiSensorEntityDescription(SensorEntityDescription):
    """Class describing Huawei LTE sensor entities."""
    name: str = ''
    format_fn: Callable[[StateType], tuple[StateType, str | None]] = format_default
    icon_fn: Callable[[StateType], str] | None = None
    device_class_fn: Callable[[StateType], SensorDeviceClass | None] | None = None
    last_reset_item: str | None = None
    last_reset_format_fn: Callable[[StateType], datetime | None] | None = None

SENSOR_META: dict[str, HuaweiSensorGroup] = {
    KEY_DEVICE_INFORMATION: HuaweiSensorGroup(
        include=re.compile(r'^(WanIP.*Address|uptime)$', re.IGNORECASE),
        descriptions={
            'uptime': HuaweiSensorEntityDescription(
                key='uptime',
                translation_key='uptime',
                icon='mdi:timer-outline',
                native_unit_of_measurement=UnitOfTime.SECONDS,
                device_class=SensorDeviceClass.DURATION,
                entity_category=EntityCategory.DIAGNOSTIC
            ),
            'WanIPAddress': HuaweiSensorEntityDescription(
                key='WanIPAddress',
                translation_key='wan_ip_address',
                icon='mdi:ip',
                entity_category=EntityCategory.DIAGNOSTIC,
                entity_registry_enabled_default=True
            ),
            'WanIPv6Address': HuaweiSensorEntityDescription(
                key='WanIPv6Address',
                translation_key='wan_ipv6_address',
                icon='mdi:ip',
                entity_category=EntityCategory.DIAGNOSTIC
            )
        }
    ),
    # Other keys omitted for brevity
}

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up from config entry."""
    router: Router = hass.data[DOMAIN].routers[config_entry.entry_id]
    sensors: list[HuaweiLteSensor] = []
    for key in SENSOR_KEYS:
        if not (items := router.data.get(key)):
            continue
        if (key_meta := SENSOR_META.get(key)):
            if key_meta.include:
                items = filter(key_meta.include.search, items)
            if key_meta.exclude:
                items = [x for x in items if not key_meta.exclude.search(x)]
        sensors.extend(
            HuaweiLteSensor(
                router,
                key,
                item,
                SENSOR_META[key].descriptions.get(item, HuaweiSensorEntityDescription(key=item))
            ) for item in items
        )
    async_add_entities(sensors, True)

class HuaweiLteSensor(HuaweiLteBaseEntityWithDevice, SensorEntity):
    """Huawei LTE sensor entity."""
    _state: StateType = None
    _unit: str | None = None
    _last_reset: datetime | None = None

    def __init__(self, router: Router, key: str, item: str, entity_description: HuaweiSensorEntityDescription) -> None:
        """Initialize."""
        super().__init__(router)
        self.key = key
        self.item = item
        self.entity_description = entity_description

    async def async_added_to_hass(self) -> None:
        """Subscribe to needed data on add."""
        await super().async_added_to_hass()
        self.router.subscriptions[self.key].append(f'{SENSOR_DOMAIN}/{self.item}')
        if self.entity_description.last_reset_item:
            self.router.subscriptions[self.key].append(f'{SENSOR_DOMAIN}/{self.entity_description.last_reset_item}')

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from needed data on remove."""
        await super().async_will_remove_from_hass()
        self.router.subscriptions[self.key].remove(f'{SENSOR_DOMAIN}/{self.item}')
        if self.entity_description.last_reset_item:
            self.router.subscriptions[self.key].remove(f'{SENSOR_DOMAIN}/{self.entity_description.last_reset_item}')

    @property
    def _device_unique_id(self) -> str:
        return f'{self.key}.{self.item}'

    @property
    def native_value(self) -> StateType:
        """Return sensor state."""
        return self._state

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return sensor's unit of measurement."""
        return self.entity_description.native_unit_of_measurement or self._unit

    @property
    def icon(self) -> str | None:
        """Return icon for sensor."""
        if self.entity_description.icon_fn:
            return self.entity_description.icon_fn(self.state)
        return self.entity_description.icon

    @property
    def device_class(self) -> SensorDeviceClass | None:
        """Return device class for sensor."""
        if self.entity_description.device_class_fn:
            return self.entity_description.device_class_fn(self.native_value)
        return super().device_class

    @property
    def last_reset(self) -> datetime | None:
        """Return the time when the sensor was last reset, if any."""
        return self._last_reset

    async def async_update(self) -> None:
        """Update state."""
        try:
            value = self.router.data[self.key][self.item]
        except KeyError:
            _LOGGER.debug('%s[%s] not in data', self.key, self.item)
            value = None
        last_reset: datetime | None = None
        if self.entity_description.last_reset_item and self.entity_description.last_reset_format_fn:
            try:
                last_reset_value = self.router.data[self.key][self.entity_description.last_reset_item]
            except KeyError:
                _LOGGER.debug('%s[%s] not in data', self.key, self.entity_description.last_reset_item)
            else:
                last_reset = self.entity_description.last_reset_format_fn(last_reset_value)
        self._state, self._unit = self.entity_description.format_fn(value)
        self._last_reset = last_reset
        self._available = value is not None
