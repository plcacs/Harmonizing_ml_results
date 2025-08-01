from __future__ import annotations
import logging
from typing import Any, Dict, Tuple, Optional, List
from numato_gpio import NumatoGpioError
from homeassistant.components.sensor import SensorEntity
from homeassistant.const import CONF_ID, CONF_NAME, CONF_SENSORS
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import CONF_DEVICES, CONF_DST_RANGE, CONF_DST_UNIT, CONF_PORTS, CONF_SRC_RANGE, DATA_API, DOMAIN

_LOGGER = logging.getLogger(__name__)


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the configured Numato USB GPIO ADC sensor ports."""
    if discovery_info is None:
        return
    api: Any = hass.data[DOMAIN][DATA_API]
    sensors: List[NumatoGpioAdc] = []
    devices: List[Dict[str, Any]] = hass.data[DOMAIN][CONF_DEVICES]
    for device in [d for d in devices if CONF_SENSORS in d]:
        device_id: str = device[CONF_ID]
        ports: Dict[Any, Any] = device[CONF_SENSORS][CONF_PORTS]
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

    def __init__(
        self,
        name: str,
        device_id: str,
        port: Any,
        src_range: Tuple[float, float],
        dst_range: Tuple[float, float],
        dst_unit: str,
        api: Any,
    ) -> None:
        """Initialize the sensor."""
        self._attr_name: str = name
        self._device_id: str = device_id
        self._port: Any = port
        self._src_range: Tuple[float, float] = src_range
        self._dst_range: Tuple[float, float] = dst_range
        self._attr_native_unit_of_measurement: str = dst_unit
        self._api: Any = api

    def update(self) -> None:
        """Get the latest data and updates the state."""
        try:
            adc_val: float = float(self._api.read_adc_input(self._device_id, self._port))
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
        """Clamp the ADC value to the defined source range."""
        val = max(val, self._src_range[0])
        return min(val, self._src_range[1])

    def _linear_scale_to_dest_range(self, val: float) -> float:
        """Linearly scale the ADC value from the source to the destination range."""
        src_len: float = self._src_range[1] - self._src_range[0]
        adc_val_rel: float = val - self._src_range[0]
        ratio: float = float(adc_val_rel) / float(src_len)
        dst_len: float = self._dst_range[1] - self._dst_range[0]
        return self._dst_range[0] + ratio * dst_len