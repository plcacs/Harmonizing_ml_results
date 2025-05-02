"""Number for ViCare."""
from __future__ import annotations
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
import logging
from typing import Any, Optional, Union, cast
from PyViCare.PyViCareDevice import Device as PyViCareDevice
from PyViCare.PyViCareDeviceConfig import PyViCareDeviceConfig
from PyViCare.PyViCareHeatingDevice import HeatingDeviceWithComponent as PyViCareHeatingDeviceComponent
from PyViCare.PyViCareUtils import PyViCareInvalidDataError, PyViCareNotSupportedFeatureError, PyViCareRateLimitError
from requests.exceptions import ConnectionError as RequestConnectionError
from homeassistant.components.number import NumberDeviceClass, NumberEntity, NumberEntityDescription
from homeassistant.const import EntityCategory, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .entity import ViCareEntity
from .types import HeatingProgram, ViCareConfigEntry, ViCareDevice, ViCareRequiredKeysMixin
from .utils import get_circuits, get_device_serial, is_supported
_LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class ViCareNumberEntityDescription(NumberEntityDescription, ViCareRequiredKeysMixin):
    """Describes ViCare number entity."""
    value_setter: Optional[Callable[[Any, float], None]] = None
    min_value_getter: Optional[Callable[[Any], float]] = None
    max_value_getter: Optional[Callable[[Any], float]] = None
    stepping_getter: Optional[Callable[[Any], float]] = None

DEVICE_ENTITY_DESCRIPTIONS: tuple[ViCareNumberEntityDescription, ...] = (
    ViCareNumberEntityDescription(
        key='dhw_temperature',
        translation_key='dhw_temperature',
        entity_category=EntityCategory.CONFIG,
        device_class=NumberDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        value_getter=lambda api: api.getDomesticHotWaterConfiguredTemperature(),
        value_setter=lambda api, value: api.setDomesticHotWaterTemperature(value),
        min_value_getter=lambda api: api.getDomesticHotWaterMinTemperature(),
        max_value_getter=lambda api: api.getDomesticHotWaterMaxTemperature(),
        native_step=1
    ),
    # ... (other descriptions remain the same)
)

CIRCUIT_ENTITY_DESCRIPTIONS: tuple[ViCareNumberEntityDescription, ...] = (
    ViCareNumberEntityDescription(
        key='heating curve shift',
        translation_key='heating_curve_shift',
        entity_category=EntityCategory.CONFIG,
        device_class=NumberDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        value_getter=lambda api: api.getHeatingCurveShift(),
        value_setter=lambda api, shift: api.setHeatingCurve(shift, api.getHeatingCurveSlope()),
        min_value_getter=lambda api: api.getHeatingCurveShiftMin(),
        max_value_getter=lambda api: api.getHeatingCurveShiftMax(),
        stepping_getter=lambda api: api.getHeatingCurveShiftStepping(),
        native_min_value=-13,
        native_max_value=40,
        native_step=1
    ),
    # ... (other descriptions remain the same)
)

def _build_entities(device_list: list[ViCareDevice]) -> list[ViCareNumber]:
    """Create ViCare number entities for a device."""
    entities: list[ViCareNumber] = []
    for device in device_list:
        entities.extend(
            ViCareNumber(description, get_device_serial(device.api), device.config, device.api)
            for description in DEVICE_ENTITY_DESCRIPTIONS
            if is_supported(description.key, description.value_getter, device.api)
        )
        entities.extend(
            ViCareNumber(description, get_device_serial(device.api), device.config, device.api, circuit)
            for circuit in get_circuits(device.api)
            for description in CIRCUIT_ENTITY_DESCRIPTIONS
            if is_supported(description.key, description.value_getter, circuit)
        )
    return entities

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ViCareConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Create the ViCare number devices."""
    async_add_entities(await hass.async_add_executor_job(_build_entities, config_entry.runtime_data.devices))

class ViCareNumber(ViCareEntity, NumberEntity):
    """Representation of a ViCare number."""

    def __init__(
        self,
        description: ViCareNumberEntityDescription,
        device_serial: str,
        device_config: PyViCareDeviceConfig,
        device: Union[PyViCareDevice, PyViCareHeatingDeviceComponent],
        component: Optional[PyViCareHeatingDeviceComponent] = None,
    ) -> None:
        """Initialize the number."""
        super().__init__(description.key, device_serial, device_config, device, component)
        self.entity_description = description

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self._attr_native_value is not None

    def set_native_value(self, value: float) -> None:
        """Set new value."""
        if self.entity_description.value_setter:
            self.entity_description.value_setter(self._api, value)
        self.schedule_update_ha_state()

    def update(self) -> None:
        """Update state of number."""
        try:
            with suppress(PyViCareNotSupportedFeatureError):
                self._attr_native_value = self.entity_description.value_getter(self._api)
                if (min_value := _get_value(self.entity_description.min_value_getter, self._api)):
                    self._attr_native_min_value = min_value
                if (max_value := _get_value(self.entity_description.max_value_getter, self._api)):
                    self._attr_native_max_value = max_value
                if (stepping_value := _get_value(self.entity_description.stepping_getter, self._api)):
                    self._attr_native_step = stepping_value
        except RequestConnectionError:
            _LOGGER.error('Unable to retrieve data from ViCare server')
        except ValueError:
            _LOGGER.error('Unable to decode data from ViCare server')
        except PyViCareRateLimitError as limit_exception:
            _LOGGER.error('Vicare API rate limit exceeded: %s', limit_exception)
        except PyViCareInvalidDataError as invalid_data_exception:
            _LOGGER.error('Invalid data from Vicare server: %s', invalid_data_exception)

def _get_value(
    fn: Optional[Callable[[Any], float]],
    api: Any
) -> Optional[float]:
    return None if fn is None else fn(api)
