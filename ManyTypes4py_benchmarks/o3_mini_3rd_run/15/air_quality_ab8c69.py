from __future__ import annotations
from datetime import timedelta
import logging
from typing import Any, Callable, Dict, Optional
import metno
import voluptuous as vol
from homeassistant.components.air_quality import PLATFORM_SCHEMA as AIR_QUALITY_PLATFORM_SCHEMA, AirQualityEntity
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from aiohttp import ClientSession

_LOGGER = logging.getLogger(__name__)

CONF_FORECAST = 'forecast'
DEFAULT_FORECAST = 0
DEFAULT_NAME = 'Air quality Norway'
OVERRIDE_URL = 'https://aa015h6buqvih86i1.api.met.no/weatherapi/airqualityforecast/0.1/'

PLATFORM_SCHEMA = AIR_QUALITY_PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_FORECAST, default=DEFAULT_FORECAST): vol.Coerce(int),
    vol.Optional(CONF_LATITUDE): cv.latitude,
    vol.Optional(CONF_LONGITUDE): cv.longitude,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string
})

SCAN_INTERVAL = timedelta(minutes=5)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    forecast: int = config.get(CONF_FORECAST)
    latitude: Optional[float] = config.get(CONF_LATITUDE, hass.config.latitude)
    longitude: Optional[float] = config.get(CONF_LONGITUDE, hass.config.longitude)
    name: str = config.get(CONF_NAME)
    if None in (latitude, longitude):
        _LOGGER.error('Latitude or longitude not set in Home Assistant config')
        return
    coordinates: Dict[str, str] = {'lat': str(latitude), 'lon': str(longitude)}
    session: ClientSession = async_get_clientsession(hass)
    async_add_entities([AirSensor(name, coordinates, forecast, session)], True)


def round_state(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def _decorator(self: Any) -> Any:
        res: Any = func(self)
        if isinstance(res, float):
            return round(res, 2)
        return res
    return _decorator


class AirSensor(AirQualityEntity):
    _attr_attribution: str = ('Air quality from https://luftkvalitet.miljostatus.no/, delivered by '
                              'the Norwegian Meteorological Institute.')

    def __init__(self, name: str, coordinates: Dict[str, str], forecast: int, session: ClientSession) -> None:
        self._name: str = name
        self._api = metno.AirQualityData(coordinates, forecast, session, api_url=OVERRIDE_URL)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        return {'level': self._api.data.get('level'), 'location': self._api.data.get('location')}

    @property
    def name(self) -> str:
        return self._name

    @property
    @round_state
    def air_quality_index(self) -> Optional[float]:
        return self._api.data.get('aqi')

    @property
    @round_state
    def nitrogen_dioxide(self) -> Optional[float]:
        return self._api.data.get('no2_concentration')

    @property
    @round_state
    def ozone(self) -> Optional[float]:
        return self._api.data.get('o3_concentration')

    @property
    @round_state
    def particulate_matter_2_5(self) -> Optional[float]:
        return self._api.data.get('pm25_concentration')

    @property
    @round_state
    def particulate_matter_10(self) -> Optional[float]:
        return self._api.data.get('pm10_concentration')

    @property
    def unit_of_measurement(self) -> Optional[str]:
        return self._api.units.get('pm25_concentration')

    async def async_update(self) -> None:
        await self._api.update()