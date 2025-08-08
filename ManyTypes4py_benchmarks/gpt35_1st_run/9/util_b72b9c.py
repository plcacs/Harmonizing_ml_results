from datetime import datetime, timedelta
from http import HTTPStatus
import logging
from typing import Any, Dict, List, Optional
import aiohttp
from buienradar.buienradar import parse_data
from buienradar.constants import ATTRIBUTION, CONDITION, CONTENT, DATA, FORECAST, HUMIDITY, MESSAGE, PRESSURE, STATIONNAME, STATUS_CODE, SUCCESS, TEMPERATURE, VISIBILITY, WINDAZIMUTH, WINDSPEED
from buienradar.urls import JSON_FEED_URL, json_precipitation_forecast_url
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.util import dt as dt_util
from .const import DEFAULT_TIMEOUT, SCHEDULE_NOK, SCHEDULE_OK

__all__: List[str] = ['BrData']
_LOGGER: logging.Logger = logging.getLogger(__name__)
WARN_THRESHOLD: int = 4

def threshold_log(count: int, *args: Any, **kwargs: Any) -> None:
    ...

class BrData:
    load_error_count: int = WARN_THRESHOLD
    rain_error_count: int = WARN_THRESHOLD

    def __init__(self, hass: HomeAssistant, coordinates: Dict[str, float], timeframe: int, devices: List[Any]) -> None:
        ...

    async def update_devices(self) -> None:
        ...

    @callback
    def async_schedule_update(self, minute: int = 1) -> None:
        ...

    async def get_data(self, url: str) -> Dict[str, Any]:
        ...

    async def _async_update(self) -> Optional[Dict[str, Any]]:
        ...

    async def async_update(self, *_: Any) -> None:
        ...

    @property
    def attribution(self) -> Optional[str]:
        ...

    @property
    def stationname(self) -> Optional[str]:
        ...

    @property
    def condition(self) -> Optional[str]:
        ...

    @property
    def temperature(self) -> Optional[float]:
        ...

    @property
    def pressure(self) -> Optional[float]:
        ...

    @property
    def humidity(self) -> Optional[int]:
        ...

    @property
    def visibility(self) -> Optional[int]:
        ...

    @property
    def wind_speed(self) -> Optional[float]:
        ...

    @property
    def wind_bearing(self) -> Optional[int]:
        ...

    @property
    def forecast(self) -> Optional[Dict[str, Any]]:
        ...
