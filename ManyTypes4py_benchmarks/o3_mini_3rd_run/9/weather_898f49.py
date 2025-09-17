from __future__ import annotations
from typing import Any, Dict, List, Optional
from homeassistant.components.weather import Forecast, SingleCoordinatorWeatherEntity, WeatherEntityFeature
from homeassistant.const import UnitOfLength, UnitOfPrecipitationDepth, UnitOfPressure, UnitOfSpeed, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.config_entries import ConfigEntry
from . import OpenweathermapConfigEntry
from .const import (
    ATTR_API_CLOUDS,
    ATTR_API_CONDITION,
    ATTR_API_CURRENT,
    ATTR_API_DAILY_FORECAST,
    ATTR_API_DEW_POINT,
    ATTR_API_FEELS_LIKE_TEMPERATURE,
    ATTR_API_HOURLY_FORECAST,
    ATTR_API_HUMIDITY,
    ATTR_API_PRESSURE,
    ATTR_API_TEMPERATURE,
    ATTR_API_VISIBILITY_DISTANCE,
    ATTR_API_WIND_BEARING,
    ATTR_API_WIND_GUST,
    ATTR_API_WIND_SPEED,
    ATTRIBUTION,
    DEFAULT_NAME,
    DOMAIN,
    MANUFACTURER,
    OWM_MODE_FREE_FORECAST,
    OWM_MODE_V25,
    OWM_MODE_V30,
)
from .coordinator import WeatherUpdateCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up OpenWeatherMap weather entity based on a config entry."""
    domain_data: Any = config_entry.runtime_data
    name: str = domain_data.name
    mode: str = domain_data.mode
    weather_coordinator: WeatherUpdateCoordinator = domain_data.coordinator
    unique_id: str = f"{config_entry.unique_id}"
    owm_weather = OpenWeatherMapWeather(name, unique_id, mode, weather_coordinator)
    async_add_entities([owm_weather], False)


class OpenWeatherMapWeather(SingleCoordinatorWeatherEntity[WeatherUpdateCoordinator]):
    """Implementation of an OpenWeatherMap sensor."""
    _attr_attribution: str = ATTRIBUTION
    _attr_should_poll: bool = False
    _attr_native_precipitation_unit: str = UnitOfPrecipitationDepth.MILLIMETERS
    _attr_native_pressure_unit: str = UnitOfPressure.HPA
    _attr_native_temperature_unit: str = UnitOfTemperature.CELSIUS
    _attr_native_wind_speed_unit: str = UnitOfSpeed.METERS_PER_SECOND
    _attr_native_visibility_unit: str = UnitOfLength.METERS

    def __init__(
        self,
        name: str,
        unique_id: str,
        mode: str,
        weather_coordinator: WeatherUpdateCoordinator,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(weather_coordinator)
        self._attr_name: str = name
        self._attr_unique_id: str = unique_id
        self._attr_device_info: DeviceInfo = DeviceInfo(
            entry_type=DeviceEntryType.SERVICE,
            identifiers={(DOMAIN, unique_id)},
            manufacturer=MANUFACTURER,
            name=DEFAULT_NAME,
        )
        if mode in (OWM_MODE_V30, OWM_MODE_V25):
            self._attr_supported_features = WeatherEntityFeature.FORECAST_DAILY | WeatherEntityFeature.FORECAST_HOURLY
        elif mode == OWM_MODE_FREE_FORECAST:
            self._attr_supported_features = WeatherEntityFeature.FORECAST_HOURLY

    @property
    def condition(self) -> Optional[str]:
        """Return the current condition."""
        return self.coordinator.data[ATTR_API_CURRENT].get(ATTR_API_CONDITION)

    @property
    def cloud_coverage(self) -> Optional[int]:
        """Return the Cloud coverage in %."""
        return self.coordinator.data[ATTR_API_CURRENT].get(ATTR_API_CLOUDS)

    @property
    def native_apparent_temperature(self) -> Optional[float]:
        """Return the apparent temperature."""
        return self.coordinator.data[ATTR_API_CURRENT].get(ATTR_API_FEELS_LIKE_TEMPERATURE)

    @property
    def native_temperature(self) -> Optional[float]:
        """Return the temperature."""
        return self.coordinator.data[ATTR_API_CURRENT].get(ATTR_API_TEMPERATURE)

    @property
    def native_pressure(self) -> Optional[float]:
        """Return the pressure."""
        return self.coordinator.data[ATTR_API_CURRENT].get(ATTR_API_PRESSURE)

    @property
    def humidity(self) -> Optional[int]:
        """Return the humidity."""
        return self.coordinator.data[ATTR_API_CURRENT].get(ATTR_API_HUMIDITY)

    @property
    def native_dew_point(self) -> Optional[float]:
        """Return the dew point."""
        return self.coordinator.data[ATTR_API_CURRENT].get(ATTR_API_DEW_POINT)

    @property
    def native_wind_gust_speed(self) -> Optional[float]:
        """Return the wind gust speed."""
        return self.coordinator.data[ATTR_API_CURRENT].get(ATTR_API_WIND_GUST)

    @property
    def native_wind_speed(self) -> Optional[float]:
        """Return the wind speed."""
        return self.coordinator.data[ATTR_API_CURRENT].get(ATTR_API_WIND_SPEED)

    @property
    def wind_bearing(self) -> Optional[int]:
        """Return the wind bearing."""
        return self.coordinator.data[ATTR_API_CURRENT].get(ATTR_API_WIND_BEARING)

    @property
    def visibility(self) -> Optional[float]:
        """Return visibility."""
        return self.coordinator.data[ATTR_API_CURRENT].get(ATTR_API_VISIBILITY_DISTANCE)

    @callback
    def _async_forecast_daily(self) -> List[Forecast]:
        """Return the daily forecast in native units."""
        return self.coordinator.data[ATTR_API_DAILY_FORECAST]

    @callback
    def _async_forecast_hourly(self) -> List[Forecast]:
        """Return the hourly forecast in native units."""
        return self.coordinator.data[ATTR_API_HOURLY_FORECAST]