from __future__ import annotations
from datetime import datetime, timedelta
import logging
from aio_geojson_geonetnz_volcano import GeonetnzVolcanoFeedManager
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE, CONF_RADIUS, CONF_SCAN_INTERVAL, CONF_UNIT_SYSTEM, UnitOfLength
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import aiohttp_client, config_validation as cv
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.typing import ConfigType
from homeassistant.util.unit_conversion import DistanceConverter
from .config_flow import configured_instances
from .const import DEFAULT_RADIUS, DEFAULT_SCAN_INTERVAL, DOMAIN, FEED, IMPERIAL_UNITS, PLATFORMS

_LOGGER: logging.Logger

CONFIG_SCHEMA: vol.Schema

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:

class GeonetnzVolcanoFeedEntityManager:

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, radius_in_km: float, unit_system: str):

    async def async_init(self) -> None:

    async def async_update(self) -> None:

    async def async_stop(self) -> None:

    @callback
    def async_event_new_entity(self) -> str:

    def get_entry(self, external_id: str):

    def last_update(self) -> datetime:

    def last_update_successful(self) -> bool:

    async def _generate_entity(self, external_id: str) -> None:

    async def _update_entity(self, external_id: str) -> None:

    async def _remove_entity(self, external_id: str) -> None:
