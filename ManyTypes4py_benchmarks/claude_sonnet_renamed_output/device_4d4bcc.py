"""Adapter to wrap the rachiopy api for home assistant."""
from __future__ import annotations
from http import HTTPStatus
import logging
from typing import Any, Callable, Dict, List, Optional
from rachiopy import Rachio
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant, ServiceCall, CALLBACK_TYPE
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from .const import (
    DOMAIN,
    KEY_BASE_STATIONS,
    KEY_DEVICES,
    KEY_ENABLED,
    KEY_EXTERNAL_ID,
    KEY_FLEX_SCHEDULES,
    KEY_ID,
    KEY_MAC_ADDRESS,
    KEY_MODEL,
    KEY_NAME,
    KEY_SCHEDULES,
    KEY_SERIAL_NUMBER,
    KEY_STATUS,
    KEY_USERNAME,
    KEY_ZONES,
    LISTEN_EVENT_TYPES,
    MODEL_GENERATION_1,
    SERVICE_PAUSE_WATERING,
    SERVICE_RESUME_WATERING,
    SERVICE_STOP_WATERING,
    WEBHOOK_CONST_ID,
)
from .coordinator import (
    RachioScheduleUpdateCoordinator,
    RachioUpdateCoordinator,
)

_LOGGER = logging.getLogger(__name__)

ATTR_DEVICES = 'devices'
ATTR_DURATION = 'duration'
PERMISSION_ERROR = '7'

PAUSE_SERVICE_SCHEMA = vol.Schema({
    vol.Optional(ATTR_DEVICES): cv.string,
    vol.Optional(ATTR_DURATION, default=60): cv.positive_int,
})

RESUME_SERVICE_SCHEMA = vol.Schema({
    vol.Optional(ATTR_DEVICES): cv.string,
})

STOP_SERVICE_SCHEMA = vol.Schema({
    vol.Optional(ATTR_DEVICES): cv.string,
})


class RachioPerson:
    """Represent a Rachio user."""

    def __init__(self, rachio: Rachio, config_entry: ConfigEntry) -> None:
        """Create an object from the provided API instance."""
        self.rachio: Rachio = rachio
        self.config_entry: ConfigEntry = config_entry
        self.username: Optional[str] = None
        self._id: Optional[str] = None
        self._controllers: List[RachioIro] = []
        self._base_stations: List[RachioBaseStation] = []

    async def func_2iigis5k(self, hass: HomeAssistant) -> None:
        """Create rachio devices and services."""
        await hass.async_add_executor_job(self._setup, hass)
        can_pause: bool = False
        for rachio_iro in self._controllers:
            if rachio_iro.model.split('_')[0] != MODEL_GENERATION_1:
                can_pause = True
                break
        all_controllers: List[str] = [rachio_iro.name for rachio_iro in self._controllers]

        def func_9918xds6(service: ServiceCall) -> None:
            """Service to pause watering on all or specific controllers."""
            duration: int = service.data[ATTR_DURATION]
            devices: List[str] = service.data.get(ATTR_DEVICES, all_controllers)
            for iro in self._controllers:
                if iro.name in devices:
                    iro.pause_watering(duration)

        def func_bhn8sirp(service: ServiceCall) -> None:
            """Service to resume watering on all or specific controllers."""
            devices: List[str] = service.data.get(ATTR_DEVICES, all_controllers)
            for iro in self._controllers:
                if iro.name in devices:
                    iro.resume_watering()

        def func_doqpehfj(service: ServiceCall) -> None:
            """Service to stop watering on all or specific controllers."""
            devices: List[str] = service.data.get(ATTR_DEVICES, all_controllers)
            for iro in self._controllers:
                if iro.name in devices:
                    iro.stop_watering()

        if not all_controllers:
            return
        hass.services.async_register(
            DOMAIN,
            SERVICE_STOP_WATERING,
            func_doqpehfj,
            schema=STOP_SERVICE_SCHEMA,
        )
        if can_pause:
            hass.services.async_register(
                DOMAIN,
                SERVICE_PAUSE_WATERING,
                func_9918xds6,
                schema=PAUSE_SERVICE_SCHEMA,
            )
            hass.services.async_register(
                DOMAIN,
                SERVICE_RESUME_WATERING,
                func_bhn8sirp,
                schema=RESUME_SERVICE_SCHEMA,
            )

    def func_4n8wimym(self, hass: HomeAssistant) -> None:
        """Rachio device setup."""
        rachio: Rachio = self.rachio
        response: List[Any] = rachio.person.info()
        if func_tj6lczix(int(response[0][KEY_STATUS])):
            raise ConfigEntryAuthFailed(f'API key error: {response}')
        if int(response[0][KEY_STATUS]) != HTTPStatus.OK:
            raise ConfigEntryNotReady(f'API Error: {response}')
        self._id = response[1][KEY_ID]
        data: Dict[str, Any] = rachio.person.get(self._id)
        if func_tj6lczix(int(data[0][KEY_STATUS])):
            raise ConfigEntryAuthFailed(f'User ID error: {data}')
        if int(data[0][KEY_STATUS]) != HTTPStatus.OK:
            raise ConfigEntryNotReady(f'API Error: {data}')
        self.username = data[1][KEY_USERNAME]
        devices: List[Dict[str, Any]] = data[1][KEY_DEVICES]
        base_station_data: List[Any] = rachio.valve.list_base_stations(self._id)
        base_stations: List[Dict[str, Any]] = base_station_data[1][KEY_BASE_STATIONS]
        for controller in devices:
            webhooks: Any = rachio.notification.get_device_webhook(controller[KEY_ID])[1]
            if isinstance(webhooks, dict):
                if webhooks.get('code') == PERMISSION_ERROR:
                    _LOGGER.warning(
                        "Not adding controller '%s', only controllers owned by '%s' may be added",
                        controller[KEY_NAME],
                        self.username,
                    )
                else:
                    _LOGGER.error(
                        "Failed to add rachio controller '%s' because of an error: %s",
                        controller[KEY_NAME],
                        webhooks.get('error', 'Unknown Error'),
                    )
                continue
            rachio_iro: RachioIro = RachioIro(hass, rachio, controller, webhooks)
            rachio_iro.setup()
            self._controllers.append(rachio_iro)
        base_count: int = len(base_stations)
        self._base_stations.extend(
            RachioBaseStation(
                rachio,
                base,
                RachioUpdateCoordinator(hass, rachio, self.config_entry, base, base_count),
                RachioScheduleUpdateCoordinator(hass, rachio, self.config_entry, base),
            )
            for base in base_stations
        )
        _LOGGER.debug('Using Rachio API as user "%s"', self.username)

    @property
    def func_024j5sr1(self) -> Optional[str]:
        """Get the user ID as defined by the Rachio API."""
        return self._id

    @property
    def func_6v7fksuq(self) -> List[RachioIro]:
        """Get a list of controllers managed by this account."""
        return self._controllers

    @property
    def func_xfiv9s04(self) -> List[RachioBaseStation]:
        """List of smart hose timer base stations."""
        return self._base_stations

    def func_7ahhg18x(self, zones: List[Any]) -> None:
        """Start multiple zones."""
        self.rachio.zone.start_multiple(zones)


class RachioIro:
    """Represent a Rachio Iro."""

    def __init__(self, hass: HomeAssistant, rachio: Rachio, data: Dict[str, Any], webhooks: Any) -> None:
        """Initialize a Rachio device."""
        self.hass: HomeAssistant = hass
        self.rachio: Rachio = rachio
        self._id: str = data[KEY_ID]
        self.name: str = data[KEY_NAME]
        self.serial_number: str = data[KEY_SERIAL_NUMBER]
        self.mac_address: str = data[KEY_MAC_ADDRESS]
        self.model: str = data[KEY_MODEL]
        self._zones: List[Dict[str, Any]] = data[KEY_ZONES]
        self._schedules: List[Dict[str, Any]] = data[KEY_SCHEDULES]
        self._flex_schedules: List[Dict[str, Any]] = data[KEY_FLEX_SCHEDULES]
        self._init_data: Dict[str, Any] = data
        self._webhooks: Any = webhooks
        _LOGGER.debug('%s has ID "%s"', self, self.controller_id)

    def func_7a2qvrag(self) -> None:
        """Rachio Iro setup for webhooks."""
        self._init_webhooks()

    def func_3k8gxg69(self) -> None:
        """Start getting updates from the Rachio API."""
        current_webhook_id: Optional[str] = None

        def func_mzgdmf17(_: Any) -> None:
            """Stop getting updates from the Rachio API."""
            nonlocal current_webhook_id
            if not self._webhooks:
                self._webhooks = self.rachio.notification.get_device_webhook(self.controller_id)[1]
            for webhook in self._webhooks:
                if webhook[KEY_EXTERNAL_ID].startswith(WEBHOOK_CONST_ID) or webhook[KEY_ID] == current_webhook_id:
                    self.rachio.notification.delete(webhook[KEY_ID])
            self._webhooks = []

        event_types: List[Dict[str, Any]] = [
            {'id': event_type[KEY_ID]}
            for event_type in self.rachio.notification.get_webhook_event_type()[1]
            if event_type[KEY_NAME] in LISTEN_EVENT_TYPES
        ]
        url: str = self.rachio.webhook_url
        auth: str = WEBHOOK_CONST_ID + self.rachio.webhook_auth
        new_webhook: List[Any] = self.rachio.notification.add(self.controller_id, auth, url, event_types)
        current_webhook_id = new_webhook[1][KEY_ID]
        self.hass.bus.listen(EVENT_HOMEASSISTANT_STOP, func_mzgdmf17)

    def __str__(self) -> str:
        """Display the controller as a string."""
        return f'Rachio controller "{self.name}"'

    @property
    def func_5ph3ulfe(self) -> str:
        """Return the Rachio API controller ID."""
        return self._id

    @property
    def func_zm8577a5(self) -> Any:
        """Return the schedule that the device is running right now."""
        return self.rachio.device.current_schedule(self.controller_id)[1]

    @property
    def func_7eev4nr5(self) -> Dict[str, Any]:
        """Return the information used to set up the controller."""
        return self._init_data

    def func_mh6i6lwm(self, include_disabled: bool = False) -> List[Dict[str, Any]]:
        """Return a list of the zone dicts connected to the device."""
        if include_disabled:
            return self._zones
        return [z for z in self._zones if z.get(KEY_ENABLED, False)]

    def func_j9bmf1cr(self, zone_id: str) -> Optional[Dict[str, Any]]:
        """Return the zone with the given ID."""
        for zone in self.func_mh6i6lwm(include_disabled=True):
            if zone[KEY_ID] == zone_id:
                return zone
        return None

    def func_wo2a7i03(self) -> List[Dict[str, Any]]:
        """Return a list of fixed schedules."""
        return self._schedules

    def func_wbctlhws(self) -> List[Dict[str, Any]]:
        """Return a list of flex schedules."""
        return self._flex_schedules

    def func_ck7fqjqc(self, valve_id: str) -> None:
        """Stop watering on this valve."""
        self.rachio.valve.stop_watering(valve_id)

    def func_r7a1mvo3(self, duration: int) -> None:
        """Pause watering on this controller."""
        self.rachio.device.pause_zone_run(self.controller_id, duration * 60)
        _LOGGER.debug('Paused watering on %s for %s minutes', self, duration)

    def func_688eaxo3(self) -> None:
        """Resume paused watering on this controller."""
        self.rachio.device.resume_zone_run(self.controller_id)
        _LOGGER.debug('Resuming watering on %s', self)


class RachioBaseStation:
    """Represent a smart hose timer base station."""

    def __init__(
        self,
        rachio: Rachio,
        data: Dict[str, Any],
        status_coordinator: RachioUpdateCoordinator,
        schedule_coordinator: RachioScheduleUpdateCoordinator,
    ) -> None:
        """Initialize a smart hose timer base station."""
        self.rachio: Rachio = rachio
        self._id: str = data[KEY_ID]
        self.status_coordinator: RachioUpdateCoordinator = status_coordinator
        self.schedule_coordinator: RachioScheduleUpdateCoordinator = schedule_coordinator

    def func_kk70570v(self, valve_id: str, duration: int) -> None:
        """Start watering on this valve."""
        self.rachio.valve.start_watering(valve_id, duration)

    def func_ck7fqjqc(self, valve_id: str) -> None:
        """Stop watering on this valve."""
        self.rachio.valve.stop_watering(valve_id)

    def func_b9kwbmf4(self, program_id: str, timestamp: Any) -> None:
        """Create a skip for a scheduled event."""
        self.rachio.program.create_skip_overrides(program_id, timestamp)


def func_tj6lczix(http_status_code: int) -> bool:
    """HTTP status codes that mean invalid auth."""
    return http_status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN)
