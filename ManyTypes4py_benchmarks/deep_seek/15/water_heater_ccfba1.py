"""Viessmann ViCare water_heater device."""
from __future__ import annotations
from contextlib import suppress
import logging
from typing import Any, List, Dict, Optional
from PyViCare.PyViCareDevice import Device as PyViCareDevice
from PyViCare.PyViCareDeviceConfig import PyViCareDeviceConfig
from PyViCare.PyViCareHeatingDevice import HeatingCircuit as PyViCareHeatingCircuit
from PyViCare.PyViCareUtils import PyViCareInvalidDataError, PyViCareNotSupportedFeatureError, PyViCareRateLimitError
import requests
from homeassistant.components.water_heater import WaterHeaterEntity, WaterHeaterEntityFeature
from homeassistant.const import ATTR_TEMPERATURE, PRECISION_TENTHS, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .entity import ViCareEntity
from .types import ViCareConfigEntry, ViCareDevice
from .utils import get_circuits, get_device_serial

_LOGGER: logging.Logger = logging.getLogger(__name__)
VICARE_MODE_DHW: str = 'dhw'
VICARE_MODE_HEATING: str = 'heating'
VICARE_MODE_DHWANDHEATING: str = 'dhwAndHeating'
VICARE_MODE_DHWANDHEATINGCOOLING: str = 'dhwAndHeatingCooling'
VICARE_MODE_FORCEDREDUCED: str = 'forcedReduced'
VICARE_MODE_FORCEDNORMAL: str = 'forcedNormal'
VICARE_MODE_OFF: str = 'standby'
VICARE_TEMP_WATER_MIN: int = 10
VICARE_TEMP_WATER_MAX: int = 60
OPERATION_MODE_ON: str = 'on'
OPERATION_MODE_OFF: str = 'off'
VICARE_TO_HA_HVAC_DHW: Dict[str, str] = {
    VICARE_MODE_DHW: OPERATION_MODE_ON,
    VICARE_MODE_DHWANDHEATING: OPERATION_MODE_ON,
    VICARE_MODE_DHWANDHEATINGCOOLING: OPERATION_MODE_ON,
    VICARE_MODE_HEATING: OPERATION_MODE_OFF,
    VICARE_MODE_FORCEDREDUCED: OPERATION_MODE_OFF,
    VICARE_MODE_FORCEDNORMAL: OPERATION_MODE_ON,
    VICARE_MODE_OFF: OPERATION_MODE_OFF
}
HA_TO_VICARE_HVAC_DHW: Dict[str, str] = {
    OPERATION_MODE_OFF: VICARE_MODE_OFF,
    OPERATION_MODE_ON: VICARE_MODE_DHW
}

def _build_entities(device_list: List[ViCareDevice]) -> List[ViCareWater]:
    """Create ViCare domestic hot water entities for a device."""
    return [
        ViCareWater(get_device_serial(device.api), device.config, device.api, circuit)
        for device in device_list
        for circuit in get_circuits(device.api)
    ]

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ViCareConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the ViCare water heater platform."""
    async_add_entities(
        await hass.async_add_executor_job(_build_entities, config_entry.runtime_data.devices)
    )

class ViCareWater(ViCareEntity, WaterHeaterEntity):
    """Representation of the ViCare domestic hot water device."""
    _attr_precision: float = PRECISION_TENTHS
    _attr_supported_features: int = WaterHeaterEntityFeature.TARGET_TEMPERATURE
    _attr_temperature_unit: str = UnitOfTemperature.CELSIUS
    _attr_min_temp: int = VICARE_TEMP_WATER_MIN
    _attr_max_temp: int = VICARE_TEMP_WATER_MAX
    _attr_operation_list: List[str] = list(HA_TO_VICARE_HVAC_DHW)
    _attr_translation_key: str = 'domestic_hot_water'
    _current_mode: Optional[str] = None

    def __init__(
        self,
        device_serial: str,
        device_config: PyViCareDeviceConfig,
        device: PyViCareDevice,
        circuit: PyViCareHeatingCircuit,
    ) -> None:
        """Initialize the DHW water_heater device."""
        super().__init__(circuit.id, device_serial, device_config, device)
        self._circuit: PyViCareHeatingCircuit = circuit
        self._attributes: Dict[str, Any] = {}

    def update(self) -> None:
        """Let HA know there has been an update from the ViCare API."""
        try:
            with suppress(PyViCareNotSupportedFeatureError):
                self._attr_current_temperature = self._api.getDomesticHotWaterStorageTemperature()
            with suppress(PyViCareNotSupportedFeatureError):
                self._attr_target_temperature = self._api.getDomesticHotWaterDesiredTemperature()
            with suppress(PyViCareNotSupportedFeatureError):
                self._current_mode = self._circuit.getActiveMode()
        except requests.exceptions.ConnectionError:
            _LOGGER.error('Unable to retrieve data from ViCare server')
        except PyViCareRateLimitError as limit_exception:
            _LOGGER.error('Vicare API rate limit exceeded: %s', limit_exception)
        except ValueError:
            _LOGGER.error('Unable to decode data from ViCare server')
        except PyViCareInvalidDataError as invalid_data_exception:
            _LOGGER.error('Invalid data from Vicare server: %s', invalid_data_exception)

    def set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperatures."""
        if (temp := kwargs.get(ATTR_TEMPERATURE)) is not None:
            self._api.setDomesticHotWaterTemperature(temp)
            self._attr_target_temperature = temp

    @property
    def current_operation(self) -> Optional[str]:
        """Return current operation ie. heat, cool, idle."""
        if self._current_mode is None:
            return None
        return VICARE_TO_HA_HVAC_DHW.get(self._current_mode, None)
