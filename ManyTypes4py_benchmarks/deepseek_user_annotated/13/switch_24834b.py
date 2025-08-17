"""Integration with the Rachio Iro sprinkler system controller."""

from abc import abstractmethod
from contextlib import suppress
from datetime import timedelta
import logging
from typing import Any, Dict, List, Optional, Union

import voluptuous as vol

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, ATTR_ID
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.util.dt import as_timestamp, now, parse_datetime, utc_from_timestamp

from .const import (
    CONF_MANUAL_RUN_MINS,
    DEFAULT_MANUAL_RUN_MINS,
    DOMAIN as DOMAIN_RACHIO,
    KEY_CURRENT_STATUS,
    KEY_CUSTOM_CROP,
    KEY_CUSTOM_SHADE,
    KEY_CUSTOM_SLOPE,
    KEY_DEVICE_ID,
    KEY_DURATION,
    KEY_ENABLED,
    KEY_ID,
    KEY_IMAGE_URL,
    KEY_NAME,
    KEY_ON,
    KEY_RAIN_DELAY,
    KEY_RAIN_DELAY_END,
    KEY_REPORTED_STATE,
    KEY_SCHEDULE_ID,
    KEY_STATE,
    KEY_SUBTYPE,
    KEY_SUMMARY,
    KEY_TYPE,
    KEY_ZONE_ID,
    KEY_ZONE_NUMBER,
    SCHEDULE_TYPE_FIXED,
    SCHEDULE_TYPE_FLEX,
    SERVICE_SET_ZONE_MOISTURE,
    SERVICE_START_MULTIPLE_ZONES,
    SERVICE_START_WATERING,
    SIGNAL_RACHIO_CONTROLLER_UPDATE,
    SIGNAL_RACHIO_RAIN_DELAY_UPDATE,
    SIGNAL_RACHIO_SCHEDULE_UPDATE,
    SIGNAL_RACHIO_ZONE_UPDATE,
    SLOPE_FLAT,
    SLOPE_MODERATE,
    SLOPE_SLIGHT,
    SLOPE_STEEP,
)
from .device import RachioPerson
from .entity import RachioDevice, RachioHoseTimerEntity
from .webhooks import (
    SUBTYPE_RAIN_DELAY_OFF,
    SUBTYPE_RAIN_DELAY_ON,
    SUBTYPE_SCHEDULE_COMPLETED,
    SUBTYPE_SCHEDULE_STARTED,
    SUBTYPE_SCHEDULE_STOPPED,
    SUBTYPE_SLEEP_MODE_OFF,
    SUBTYPE_SLEEP_MODE_ON,
    SUBTYPE_ZONE_COMPLETED,
    SUBTYPE_ZONE_PAUSED,
    SUBTYPE_ZONE_STARTED,
    SUBTYPE_ZONE_STOPPED,
)

_LOGGER = logging.getLogger(__name__)

ATTR_DURATION = "duration"
ATTR_PERCENT = "percent"
ATTR_SCHEDULE_SUMMARY = "Summary"
ATTR_SCHEDULE_ENABLED = "Enabled"
ATTR_SCHEDULE_DURATION = "Duration"
ATTR_SCHEDULE_TYPE = "Type"
ATTR_SORT_ORDER = "sortOrder"
ATTR_WATERING_DURATION = "Watering Duration seconds"
ATTR_ZONE_NUMBER = "Zone number"
ATTR_ZONE_SHADE = "Shade"
ATTR_ZONE_SLOPE = "Slope"
ATTR_ZONE_SUMMARY = "Summary"
ATTR_ZONE_TYPE = "Type"

START_MULTIPLE_ZONES_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
        vol.Required(ATTR_DURATION): cv.ensure_list_csv,
    }
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Rachio switches."""
    zone_entities: List[RachioZone] = []
    has_flex_sched = False
    entities = await hass.async_add_executor_job(_create_entities, hass, config_entry)
    for entity in entities:
        if isinstance(entity, RachioZone):
            zone_entities.append(entity)
        if isinstance(entity, RachioSchedule) and entity.type == SCHEDULE_TYPE_FLEX:
            has_flex_sched = True

    async_add_entities(entities)

    def start_multiple(service: ServiceCall) -> None:
        """Service to start multiple zones in sequence."""
        zones_list: List[Dict[str, Any]] = []
        person = hass.data[DOMAIN_RACHIO][config_entry.entry_id]
        entity_id = service.data[ATTR_ENTITY_ID]
        duration = iter(service.data[ATTR_DURATION])
        default_time = service.data[ATTR_DURATION][0]
        entity_to_zone_id = {
            entity.entity_id: entity.zone_id for entity in zone_entities
        }

        for count, data in enumerate(entity_id):
            if data in entity_to_zone_id:
                time = int(next(duration, default_time)) * 60
                zones_list.append(
                    {
                        ATTR_ID: entity_to_zone_id.get(data),
                        ATTR_DURATION: time,
                        ATTR_SORT_ORDER: count,
                    }
                )

        if zones_list:
            person.start_multiple_zones(zones_list)
            _LOGGER.debug("Starting zone(s) %s", entity_id)
        else:
            raise HomeAssistantError("No matching zones found in given entity_ids")

    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(
        SERVICE_START_WATERING,
        {
            vol.Optional(ATTR_DURATION): cv.positive_int,
        },
        "turn_on",
    )

    if not zone_entities:
        return

    hass.services.async_register(
        DOMAIN_RACHIO,
        SERVICE_START_MULTIPLE_ZONES,
        start_multiple,
        schema=START_MULTIPLE_ZONES_SCHEMA,
    )

    if has_flex_sched:
        platform = entity_platform.async_get_current_platform()
        platform.async_register_entity_service(
            SERVICE_SET_ZONE_MOISTURE,
            {vol.Required(ATTR_PERCENT): cv.positive_int},
            "set_moisture_percent",
        )


def _create_entities(hass: HomeAssistant, config_entry: ConfigEntry) -> List[Entity]:
    entities: List[Entity] = []
    person: RachioPerson = hass.data[DOMAIN_RACHIO][config_entry.entry_id]
    for controller in person.controllers:
        entities.append(RachioStandbySwitch(controller))
        entities.append(RachioRainDelay(controller))
        zones = controller.list_zones()
        schedules = controller.list_schedules()
        flex_schedules = controller.list_flex_schedules()
        current_schedule = controller.current_schedule
        entities.extend(
            RachioZone(person, controller, zone, current_schedule) for zone in zones
        )
        entities.extend(
            RachioSchedule(person, controller, schedule, current_schedule)
            for schedule in schedules + flex_schedules
        )
    entities.extend(
        RachioValve(person, base_station, valve, base_station.status_coordinator)
        for base_station in person.base_stations
        for valve in base_station.status_coordinator.data.values()
    )
    return entities


class RachioSwitch(RachioDevice, SwitchEntity):
    """Represent a Rachio state that can be toggled."""

    @callback
    def _async_handle_any_update(self, *args: Any, **kwargs: Any) -> None:
        """Determine whether an update event applies to this device."""
        if args[0][KEY_DEVICE_ID] != self._controller.controller_id:
            return

        self._async_handle_update(args, kwargs)

    @abstractmethod
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None:
        """Handle incoming webhook data."""


class RachioStandbySwitch(RachioSwitch):
    """Representation of a standby status/button."""

    _attr_has_entity_name = True
    _attr_translation_key = "standby"

    @property
    def unique_id(self) -> str:
        """Return a unique id by combining controller id and purpose."""
        return f"{self._controller.controller_id}-standby"

    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None:
        """Update the state using webhook data."""
        if args[0][0][KEY_SUBTYPE] == SUBTYPE_SLEEP_MODE_ON:
            self._attr_is_on = True
        elif args[0][0][KEY_SUBTYPE] == SUBTYPE_SLEEP_MODE_OFF:
            self._attr_is_on = False

        self.async_write_ha_state()

    def turn_on(self, **kwargs: Any) -> None:
        """Put the controller in standby mode."""
        self._controller.rachio.device.turn_off(self._controller.controller_id)

    def turn_off(self, **kwargs: Any) -> None:
        """Resume controller functionality."""
        self._controller.rachio.device.turn_on(self._controller.controller_id)

    async def async_added_to_hass(self) -> None:
        """Subscribe to updates."""
        if KEY_ON in self._controller.init_data:
            self._attr_is_on = not self._controller.init_data[KEY_ON]

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_RACHIO_CONTROLLER_UPDATE,
                self._async_handle_any_update,
            )
        )


class RachioRainDelay(RachioSwitch):
    """Representation of a rain delay status/switch."""

    _attr_has_entity_name = True
    _attr_translation_key = "rain_delay"
    _cancel_update: Optional[CALLBACK_TYPE] = None

    def __init__(self, controller: Any) -> None:
        """Set up a Rachio rain delay switch."""
        super().__init__(controller)

    @property
    def unique_id(self) -> str:
        """Return a unique id by combining controller id and purpose."""
        return f"{self._controller.controller_id}-delay"

    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None:
        """Update the state using webhook data."""
        if self._cancel_update:
            self._cancel_update()
            self._cancel_update = None

        if args[0][0][KEY_SUBTYPE] == SUBTYPE_RAIN_DELAY_ON:
            endtime = parse_datetime(args[0][0][KEY_RAIN_DELAY_END])
            _LOGGER.debug("Rain delay expires at %s", endtime)
            self._attr_is_on = True
            assert endtime is not None
            self._cancel_update = async_track_point_in_utc_time(
                self.hass, self._delay_expiration, endtime
            )
        elif args[0][0][KEY_SUBTYPE] == SUBTYPE_RAIN_DELAY_OFF:
            self._attr_is_on = False

        self.async_write_ha_state()

    @callback
    def _delay_expiration(self, *args: Any) -> None:
        """Trigger when a rain delay expires."""
        self._attr_is_on = False
        self._cancel_update = None
        self.async_write_ha_state()

    def turn_on(self, **kwargs: Any) -> None:
        """Activate a 24 hour rain delay on the controller."""
        self._controller.rachio.device.rain_delay(self._controller.controller_id, 86400)
        _LOGGER.debug("Starting rain delay for 24 hours")

    def turn_off(self, **kwargs: Any) -> None:
        """Resume controller functionality."""
        self._controller.rachio.device.rain_delay(self._controller.controller_id, 0)
        _LOGGER.debug("Canceling rain delay")

    async def async_added_to_hass(self) -> None:
        """Subscribe to updates."""
        if KEY_RAIN_DELAY in self._controller.init_data:
            self._attr_is_on = self._controller.init_data[
                KEY_RAIN_DELAY
            ] / 1000 > as_timestamp(now())

        if self._attr_is_on is True:
            delay_end = utc_from_timestamp(
                self._controller.init_data[KEY_RAIN_DELAY] / 1000
            )
            _LOGGER.debug("Re-setting rain delay timer for %s", delay_end)
            self._cancel_update = async_track_point_in_utc_time(
                self.hass, self._delay_expiration, delay_end
            )

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_RACHIO_RAIN_DELAY_UPDATE,
                self._async_handle_any_update,
            )
        )


class RachioZone(RachioSwitch):
    """Representation of one zone of sprinklers connected to the Rachio Iro."""

    _attr_icon = "mdi:water"
    id: str
    _zone_number: int
    _zone_enabled: bool
    _shade_type: Optional[str]
    _zone_type: Optional[str]
    _slope_type: Optional[str]
    _summary: str
    _current_schedule: Dict[str, Any]

    def __init__(
        self,
        person: RachioPerson,
        controller: Any,
        data: Dict[str, Any],
        current_schedule: Dict[str, Any],
    ) -> None:
        """Initialize a new Rachio Zone."""
        self.id = data[KEY_ID]
        self._attr_name = data[KEY_NAME]
        self._zone_number = data[KEY_ZONE_NUMBER]
        self._zone_enabled = data[KEY_ENABLED]
        self._attr_entity_picture = data.get(KEY_IMAGE_URL)
        self._person = person
        self._shade_type = data.get(KEY_CUSTOM_SHADE, {}).get(KEY_NAME)
        self._zone_type = data.get(KEY_CUSTOM_CROP, {}).get(KEY_NAME)
        self._slope_type = data.get(KEY_CUSTOM_SLOPE, {}).get(KEY_NAME)
        self._summary = ""
        self._current_schedule = current_schedule
        self._attr_unique_id = f"{controller.controller_id}-zone-{self.id}"
        super().__init__(controller)

    def __str__(self) -> str:
        """Display the zone as a string."""
        return f'Rachio Zone "{self.name}" on {self._controller!s}'

    @property
    def zone_id(self) -> str:
        """How the Rachio API refers to the zone."""
        return self.id

    @property
    def zone_is_enabled(self) -> bool:
        """Return whether the zone is allowed to run."""
        return self._zone_enabled

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the optional state attributes."""
        props = {ATTR_ZONE_NUMBER: self._zone_number, ATTR_ZONE_SUMMARY: self._summary}
        if self._shade_type:
            props[ATTR_ZONE_SHADE] = self._shade_type
        if self._zone_type:
            props[ATTR_ZONE_TYPE] = self._zone_type
        if self._slope_type:
            if self._slope_type == SLOPE_FLAT:
                props[ATTR_ZONE_SLOPE] = "Flat"
            elif self._slope_type == SLOPE_SLIGHT:
                props[ATTR_ZONE_SLOPE] = "Slight"
            elif self._slope_type == SLOPE_MODERATE:
                props[ATTR_ZONE_SLOPE] = "Moderate"
            elif self._slope_type == SLOPE_STEEP:
                props[ATTR_ZONE_SLOPE] = "Steep"
        return props

    def turn_on(self, **kwargs: Any) -> None:
        """Start watering this zone."""
        self.turn_off()

        if ATTR_DURATION in kwargs:
            manual_run_time = timedelta(minutes=kwargs[ATTR_DURATION])
        else:
            manual_run_time = timedelta(
                minutes=self._person.config_entry.options.get(
                    CONF_MANUAL_RUN_MINS, DEFAULT_MANUAL_RUN_MINS
                )
            )
        self._controller.rachio.zone.start(self.zone_id, manual_run_time.seconds)
        _LOGGER.debug(
            "Watering %s on %s for %s",
            self.name,
            self._controller.name,
            str(manual_run_time),
        )

    def turn_off(self, **kwargs: Any) -> None:
        """Stop watering all zones."""
        self._controller.stop_watering()

    def set_moisture_percent(self, percent: int) -> None:
        """Set the zone moisture percent."""
        _LOGGER.debug("Setting %s moisture to %s percent", self.name, percent)
        self._controller.rachio.zone.set_moisture_percent(self.id, percent / 100)

    @callback
    def _async_handle_update(self, *args: Any, **kwargs: Any) -> None:
        """Handle incoming webhook zone data."""
        if args[0][KEY_ZONE_ID] != self.zone_id:
            return

        self._summary = args[0][KEY_SUMMARY]

        if args[0][KEY_SUBTYPE] == SUBTYPE_ZONE_STARTED:
            self._attr_is_on = True
        elif args[0][KEY_SUBTYPE] in [
            SUBTYPE_ZONE_STOPPED,
            SUBTYPE_ZONE_COMPLETED,
            SUBTYPE_ZONE_PAUSED,
        ]:
            self._attr_is_on = False

        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Subscribe to updates."""
        self._attr_is_on = self.zone_id == self._current_schedule.get(KEY_ZONE_ID)

        self.async_on_remove(
            async_dispatcher_connect