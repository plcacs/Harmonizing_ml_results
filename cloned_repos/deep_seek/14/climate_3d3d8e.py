"""Support for Schluter thermostats."""
from __future__ import annotations
import logging
from typing import Any, cast
from requests import RequestException
import voluptuous as vol
from homeassistant.components.climate import PLATFORM_SCHEMA as CLIMATE_PLATFORM_SCHEMA, SCAN_INTERVAL, ClimateEntity, ClimateEntityFeature, HVACAction, HVACMode
from homeassistant.const import ATTR_TEMPERATURE, CONF_SCAN_INTERVAL, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator, UpdateFailed
from . import DATA_SCHLUTER_API, DATA_SCHLUTER_SESSION, DOMAIN
_LOGGER = logging.getLogger(__name__)
PLATFORM_SCHEMA = CLIMATE_PLATFORM_SCHEMA.extend({vol.Optional(CONF_SCAN_INTERVAL): vol.All(vol.Coerce(int), vol.Range(min=1))})

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None
) -> None:
    """Set up the Schluter thermostats."""
    if discovery_info is None:
        return
    session_id: str = hass.data[DOMAIN][DATA_SCHLUTER_SESSION]
    api: Any = hass.data[DOMAIN][DATA_SCHLUTER_API]

    async def async_update_data() -> dict[str, Any]:
        try:
            thermostats = await hass.async_add_executor_job(api.get_thermostats, session_id)
        except RequestException as err:
            raise UpdateFailed(f'Error communicating with Schluter API: {err}') from err
        if thermostats is None:
            return {}
        return {thermo.serial_number: thermo for thermo in thermostats}
    coordinator: DataUpdateCoordinator[dict[str, Any]] = DataUpdateCoordinator(
        hass,
        _LOGGER,
        name='schluter',
        update_method=async_update_data,
        update_interval=SCAN_INTERVAL
    )
    await coordinator.async_refresh()
    async_add_entities(
        (SchluterThermostat(coordinator, serial_number, api, session_id) 
        for serial_number, thermostat in coordinator.data.items()
    )

class SchluterThermostat(CoordinatorEntity[DataUpdateCoordinator[dict[str, Any]]], ClimateEntity):
    """Representation of a Schluter thermostat."""
    _attr_hvac_mode: HVACMode = HVACMode.HEAT
    _attr_hvac_modes: list[HVACMode] = [HVACMode.HEAT]
    _attr_supported_features: ClimateEntityFeature = ClimateEntityFeature.TARGET_TEMPERATURE
    _attr_temperature_unit: str = UnitOfTemperature.CELSIUS

    def __init__(
        self,
        coordinator: DataUpdateCoordinator[dict[str, Any]],
        serial_number: str,
        api: Any,
        session_id: str
    ) -> None:
        """Initialize the thermostat."""
        super().__init__(coordinator)
        self._serial_number: str = serial_number
        self._api: Any = api
        self._session_id: str = session_id

    @property
    def unique_id(self) -> str:
        """Return unique ID for this device."""
        return self._serial_number

    @property
    def name(self) -> str:
        """Return the name of the thermostat."""
        return cast(str, self.coordinator.data[self._serial_number].name)

    @property
    def current_temperature(self) -> float:
        """Return the current temperature."""
        return cast(float, self.coordinator.data[self._serial_number].temperature)

    @property
    def hvac_action(self) -> HVACAction:
        """Return current operation. Can only be heating or idle."""
        if self.coordinator.data[self._serial_number].is_heating:
            return HVACAction.HEATING
        return HVACAction.IDLE

    @property
    def target_temperature(self) -> float:
        """Return the temperature we try to reach."""
        return cast(float, self.coordinator.data[self._serial_number].set_point_temp)

    @property
    def min_temp(self) -> float:
        """Identify min_temp in Schluter API."""
        return cast(float, self.coordinator.data[self._serial_number].min_temp)

    @property
    def max_temp(self) -> float:
        """Identify max_temp in Schluter API."""
        return cast(float, self.coordinator.data[self._serial_number].max_temp)

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Mode is always heating, so do nothing."""

    def set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        target_temp: float | None = kwargs.get(ATTR_TEMPERATURE)
        serial_number: str = self.coordinator.data[self._serial_number].serial_number
        _LOGGER.debug('Setting thermostat temperature: %s', target_temp)
        try:
            if target_temp is not None:
                self._api.set_temperature(self._session_id, serial_number, target_temp)
        except RequestException as ex:
            _LOGGER.error('An error occurred while setting temperature: %s', ex)
