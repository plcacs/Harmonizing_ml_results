"""Sensor platform integration for ADC ports of Numato USB GPIO expanders."""
from __future__ import annotations

import logging
from typing import Protocol, Sequence

from numato_gpio import NumatoGpioError

from homeassistant.components.sensor import SensorEntity
from homeassistant.const import CONF_ID, CONF_NAME, CONF_SENSORS
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from . import (
    CONF_DEVICES,
    CONF_DST_RANGE,
    CONF_DST_UNIT,
    CONF_PORTS,
    CONF_SRC_RANGE,
    DATA_API,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


class NumatoApi(Protocol):
    def setup_input(self, device_id: str, port: str) -> None: ...
    def read_adc_input(self, device_id: str, port: str) -> float: ...


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the configured Numato USB GPIO ADC sensor ports."""
    if discovery_info is None:
        return
    api: NumatoApi = hass.data[DOMAIN][DATA_API]
    sensors: list[NumatoGpioAdc] = []
    devices = hass.data[DOMAIN][CONF_DEVICES]
    for device in [d for d in devices if CONF_SENSORS in d]:
        device_id: str = device[CONF_ID]
        ports = device[CONF_SENSORS][CONF_PORTS]
        for port, adc_def in ports.items():
            try:
                api.setup_input(device_id, port)
            except NumatoGpioError as err:
                _LOGGER.error(
                    "Failed to initialize sensor '%s' on Numato device %s port %s: %s",
                    adc_def[CONF_NAME],
                    device_id,
                    port,
                    err,
                )
                continue
            sensors.append(
                NumatoGpioAdc(
                    adc_def[CONF_NAME],
                    device_id,
                    port,
                    adc_def[CONF_SRC_RANGE],
                    adc_def[CONF_DST_RANGE],
                    adc_def[CONF_DST_UNIT],
                    api,
                )
            )
    add_entities(sensors, True)


class NumatoGpioAdc(SensorEntity):
    """Represents an ADC port of a Numato USB GPIO expander."""

    _attr_icon: str = "mdi:gauge"
    _attr_name: str
    _attr_native_unit_of_measurement: str | None
    _attr_native_value: float | None

    def __init__(
        self,
        name: str,
        device_id: str,
        port: str,
        src_range: Sequence[float],
        dst_range: Sequence[float],
        dst_unit: str,
        api: NumatoApi,
    ) -> None:
        """Initialize the sensor."""
        self._attr_name = name
        self._device_id: str = device_id
        self._port: str = port
        self._src_range: Sequence[float] = src_range
        self._dst_range: Sequence[float] = dst_range
        self._attr_native_unit_of_measurement = dst_unit
        self._api: NumatoApi = api

    def update(self) -> None:
        """Get the latest data and updates the state."""
        try:
            adc_val = self._api.read_adc_input(self._device_id, self._port)
            adc_val = self._clamp_to_source_range(adc_val)
            self._attr_native_value = self._linear_scale_to_dest_range(adc_val)
        except NumatoGpioError as err:
            self._attr_native_value = None
            _LOGGER.error(
                "Failed to update Numato device %s ADC-port %s: %s",
                self._device_id,
                self._port,
                err,
            )

    def _clamp_to_source_range(self, val: float) -> float:
        val = max(val, self._src_range[0])
        return min(val, self._src_range[1])

    def _linear_scale_to_dest_range(self, val: float) -> float:
        src_len = self._src_range[1] - self._src_range[0]
        adc_val_rel = val - self._src_range[0]
        ratio = float(adc_val_rel) / float(src_len)
        dst_len = self._dst_range[1] - self._dst_range[0]
        return self._dst_range[0] + ratio * dst_len