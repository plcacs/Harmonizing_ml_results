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

def format_default(value: str) -> tuple:
    """Format value."""
    unit = None
    if value is not None:
        if (match := re.match('((&[gl]t;|[><])=?)?(?P<value>.+?)\\s*(?P<unit>[a-zA-Z]+)\\s*$', str(value))):
            try:
                value = float(match.group('value'))
                unit = match.group('unit')
            except ValueError:
                pass
    return (value, unit)

def format_freq_mhz(value: str) -> tuple:
    """Format a frequency value for which source is in tenths of MHz."""
    return (float(value) / 10 if value is not None else None, UnitOfFrequency.MEGAHERTZ)

def format_last_reset_elapsed_seconds(value: str) -> datetime:
    """Convert elapsed seconds to last reset datetime."""
    if value is None:
        return None
    try:
        last_reset = datetime.now() - timedelta(seconds=int(value))
        last_reset.replace(microsecond=0)
    except ValueError:
        return None
    return last_reset

def signal_icon(limits: Sequence[int], value: int) -> str:
    """Get signal icon."""
    return ('mdi:signal-cellular-outline', 'mdi:signal-cellular-1', 'mdi:signal-cellular-2', 'mdi:signal-cellular-3')[bisect(limits, value if value is not None else -1000]

def bandwidth_icon(limits: Sequence[int], value: int) -> str:
    """Get bandwidth icon."""
    return ('mdi:speedometer-slow', 'mdi:speedometer-medium', 'mdi:speedometer')[bisect(limits, value if value is not None else -1000]

@dataclass
class HuaweiSensorGroup:
    """Class describing Huawei LTE sensor groups."""
    include: re.Pattern = None
    exclude: re.Pattern = None

@dataclass(frozen=True)
class HuaweiSensorEntityDescription(SensorEntityDescription):
    """Class describing Huawei LTE sensor entities."""
    name: str = ''
    format_fn: Callable = format_default
    icon_fn: Callable = None
    device_class_fn: Callable = None
    last_reset_item: str = None
    last_reset_format_fn: Callable = None
