from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from homeassistant.components.sensor import EntityCategory, SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util
from . import NordPoolConfigEntry
from .const import LOGGER
from .coordinator import NordPoolDataUpdateCoordinator
from .entity import NordpoolBaseEntity
from typing import Any, Dict, List, Optional, Tuple

def validate_prices(func: Callable, entity: NordpoolBaseEntity, area: str, index: int) -> Optional[float]:

def get_prices(entity: NordpoolBaseEntity) -> Dict[str, Tuple[Optional[float], float, Optional[float]]]:

def get_min_max_price(entity: NordpoolBaseEntity, func: Callable) -> Tuple[float, datetime, datetime]:

def get_blockprices(entity: NordpoolBaseEntity) -> Dict[str, Dict[str, Tuple[datetime, datetime, float, float, float]]]:

@dataclass(frozen=True, kw_only=True)
class NordpoolDefaultSensorEntityDescription(SensorEntityDescription):

@dataclass(frozen=True, kw_only=True)
class NordpoolPricesSensorEntityDescription(SensorEntityDescription):

@dataclass(frozen=True, kw_only=True)
class NordpoolBlockPricesSensorEntityDescription(SensorEntityDescription):

DEFAULT_SENSOR_TYPES: Tuple[NordpoolDefaultSensorEntityDescription, ...]
PRICES_SENSOR_TYPES: Tuple[NordpoolPricesSensorEntityDescription, ...]
BLOCK_PRICES_SENSOR_TYPES: Tuple[NordpoolBlockPricesSensorEntityDescription, ...]
DAILY_AVERAGE_PRICES_SENSOR_TYPES: Tuple[SensorEntityDescription, ...]

async def async_setup_entry(hass: HomeAssistant, entry: NordPoolConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class NordpoolSensor(NordpoolBaseEntity, SensorEntity):

class NordpoolPriceSensor(NordpoolBaseEntity, SensorEntity):

class NordpoolBlockPriceSensor(NordpoolBaseEntity, SensorEntity):

class NordpoolDailyAveragePriceSensor(NordpoolBaseEntity, SensorEntity):
