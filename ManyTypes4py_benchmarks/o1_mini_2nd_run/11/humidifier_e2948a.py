"""Support for Xiaomi Mi Air Purifier and Xiaomi Mi Air Humidifier with humidifier entity."""
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Union

from miio.integrations.humidifier.deerma.airhumidifier_mjjsq import (
    OperationMode as AirhumidifierMjjsqOperationMode,
)
from miio.integrations.humidifier.zhimi.airhumidifier import (
    OperationMode as AirhumidifierOperationMode,
)
from miio.integrations.humidifier.zhimi.airhumidifier_miot import (
    OperationMode as AirhumidifierMiotOperationMode,
)
from homeassistant.components.humidifier import (
    ATTR_HUMIDITY,
    HumidifierDeviceClass,
    HumidifierEntity,
    HumidifierEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_MODE, CONF_DEVICE, CONF_MODEL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.percentage import percentage_to_ranged_value

from .const import (
    CONF_FLOW_TYPE,
    DOMAIN,
    KEY_COORDINATOR,
    KEY_DEVICE,
    MODEL_AIRHUMIDIFIER_CA1,
    MODEL_AIRHUMIDIFIER_CA4,
    MODEL_AIRHUMIDIFIER_CB1,
    MODELS_HUMIDIFIER_MIOT,
    MODELS_HUMIDIFIER_MJJSQ,
)
from .entity import XiaomiCoordinatedMiioEntity

_LOGGER: logging.Logger = logging.getLogger(__name__)

ATTR_TARGET_HUMIDITY: str = "target_humidity"
AVAILABLE_ATTRIBUTES: Dict[str, str] = {
    ATTR_MODE: "mode",
    ATTR_TARGET_HUMIDITY: "target_humidity",
    ATTR_HUMIDITY: "humidity",
}
AVAILABLE_MODES_CA1_CB1: List[str] = [
    mode.name for mode in AirhumidifierOperationMode if mode is not AirhumidifierOperationMode.Strong
]
AVAILABLE_MODES_CA4: List[str] = [mode.name for mode in AirhumidifierMiotOperationMode]
AVAILABLE_MODES_MJJSQ: List[str] = [
    mode.name
    for mode in AirhumidifierMjjsqOperationMode
    if mode is not AirhumidifierMjjsqOperationMode.WetAndProtect
]
AVAILABLE_MODES_OTHER: List[str] = [
    mode.name for mode in AirhumidifierOperationMode if mode is not AirhumidifierOperationMode.Auto
]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Humidifier from a config entry."""
    if config_entry.data[CONF_FLOW_TYPE] != CONF_DEVICE:
        return
    entities: List[HumidifierEntity] = []
    model: str = config_entry.data[CONF_MODEL]
    unique_id: Optional[str] = config_entry.unique_id
    coordinator: Any = hass.data[DOMAIN][config_entry.entry_id][KEY_COORDINATOR]
    if model in MODELS_HUMIDIFIER_MIOT:
        air_humidifier: Any = hass.data[DOMAIN][config_entry.entry_id][KEY_DEVICE]
        entity: HumidifierEntity = XiaomiAirHumidifierMiot(
            air_humidifier, config_entry, unique_id, coordinator
        )
    elif model in MODELS_HUMIDIFIER_MJJSQ:
        air_humidifier = hass.data[DOMAIN][config_entry.entry_id][KEY_DEVICE]
        entity = XiaomiAirHumidifierMjjsq(
            air_humidifier, config_entry, unique_id, coordinator
        )
    else:
        air_humidifier = hass.data[DOMAIN][config_entry.entry_id][KEY_DEVICE]
        entity = XiaomiAirHumidifier(
            air_humidifier, config_entry, unique_id, coordinator
        )
    entities.append(entity)
    async_add_entities(entities)


class XiaomiGenericHumidifier(XiaomiCoordinatedMiioEntity, HumidifierEntity):
    """Representation of a generic Xiaomi humidifier device."""

    _attr_device_class: HumidifierDeviceClass = HumidifierDeviceClass.HUMIDIFIER
    _attr_supported_features: HumidifierEntityFeature = HumidifierEntityFeature.MODES
    _attr_name: Optional[str] = None

    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: Optional[str],
        coordinator: Any,
    ) -> None:
        """Initialize the generic Xiaomi device."""
        super().__init__(device, entry, unique_id, coordinator=coordinator)
        self._state: Optional[bool] = None
        self._attributes: Dict[str, Any] = {}
        self._mode: Optional[str] = None
        self._humidity_steps: int = 100
        self._target_humidity: Optional[float] = None

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        return self._state or False

    @property
    def mode(self) -> Optional[str]:
        """Get the current mode."""
        return self._mode

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the device on."""
        result: bool = await self._try_command(
            "Turning the miio device on failed.", self._device.on
        )
        if result:
            self._state = True
            self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the device off."""
        result: bool = await self._try_command(
            "Turning the miio device off failed.", self._device.off
        )
        if result:
            self._state = False
            self.async_write_ha_state()

    def translate_humidity(self, humidity: float) -> Optional[float]:
        """Translate the target humidity to the first valid step."""
        if 0 < humidity <= 100:
            ranged_value = percentage_to_ranged_value((1, self._humidity_steps), humidity)
            return math.ceil(ranged_value) * 100 / self._humidity_steps
        return None


class XiaomiAirHumidifier(XiaomiGenericHumidifier, HumidifierEntity):
    """Representation of a Xiaomi Air Humidifier."""

    def __init__(
        self,
        device: Any,
        entry: ConfigEntry,
        unique_id: Optional[str],
        coordinator: Any,
    ) -> None:
        """Initialize the plug switch."""
        super().__init__(device, entry, unique_id, coordinator)
        self._attr_min_humidity: int = 30
        self._attr_max_humidity: int = 80
        if self._model in [MODEL_AIRHUMIDIFIER_CA1, MODEL_AIRHUMIDIFIER_CB1]:
            self._attr_available_modes = AVAILABLE_MODES_CA1_CB1
            self._humidity_steps = 10
        elif self._model in [MODEL_AIRHUMIDIFIER_CA4]:
            self._attr_available_modes = AVAILABLE_MODES_CA4
            self._humidity_steps = 100
        elif self._model in MODELS_HUMIDIFIER_MJJSQ:
            self._attr_available_modes = AVAILABLE_MODES_MJJSQ
            self._humidity_steps = 100
        else:
            self._attr_available_modes = AVAILABLE_MODES_OTHER
            self._humidity_steps = 10
        self._state: Optional[bool] = self.coordinator.data.is_on
        self._attributes.update(
            {
                key: self._extract_value_from_attribute(
                    self.coordinator.data, value
                )
                for key, value in AVAILABLE_ATTRIBUTES.items()
            }
        )
        self._target_humidity: Optional[float] = self._attributes[
            ATTR_TARGET_HUMIDITY
        ]
        self._attr_current_humidity: Optional[float] = self._attributes[
            ATTR_HUMIDITY
        ]
        self._mode: Optional[str] = self._attributes[ATTR_MODE]

    @property
    def is_on(self) -> bool:
        """Return true if device is on."""
        return self._state or False

    @callback
    def _handle_coordinator_update(self) -> None:
        """Fetch state from the device."""
        self._state = self.coordinator.data.is_on
        self._attributes.update(
            {
                key: self._extract_value_from_attribute(
                    self.coordinator.data, value
                )
                for key, value in AVAILABLE_ATTRIBUTES.items()
            }
        )
        self._target_humidity = self._attributes[ATTR_TARGET_HUMIDITY]
        self._attr_current_humidity = self._attributes[ATTR_HUMIDITY]
        self._mode = self._attributes[ATTR_MODE]
        self.async_write_ha_state()

    @property
    def mode(self) -> Optional[str]:
        """Return the current mode."""
        if self._mode is not None:
            try:
                return AirhumidifierOperationMode(self._mode).name
            except ValueError:
                return None
        return None

    @property
    def target_humidity(self) -> Optional[float]:
        """Return the target humidity."""
        if self._mode is None:
            return None
        if (
            self._mode == AirhumidifierOperationMode.Auto.value
            or AirhumidifierOperationMode.Auto.name not in self.available_modes
        ):
            return self._target_humidity
        return None

    async def async_set_humidity(self, humidity: float) -> None:
        """Set the target humidity of the humidifier and set the mode to auto."""
        target_humidity = self.translate_humidity(humidity)
        if not target_humidity:
            return
        _LOGGER.debug("Setting the target humidity to: %s", target_humidity)
        if await self._try_command(
            "Setting target humidity of the miio device failed.",
            self._device.set_target_humidity,
            target_humidity,
        ):
            self._target_humidity = target_humidity
        if (
            self.supported_features & HumidifierEntityFeature.MODES == 0
            or AirhumidifierOperationMode(self._attributes[ATTR_MODE])
            == AirhumidifierOperationMode.Auto
            or AirhumidifierOperationMode.Auto.name not in self.available_modes
        ):
            self.async_write_ha_state()
            return
        _LOGGER.debug("Setting the operation mode to: Auto")
        if await self._try_command(
            "Setting operation mode of the miio device to MODE_AUTO failed.",
            self._device.set_mode,
            AirhumidifierOperationMode.Auto,
        ):
            self._mode = AirhumidifierOperationMode.Auto.value
            self.async_write_ha_state()

    async def async_set_mode(self, mode: Optional[str]) -> None:
        """Set the mode of the humidifier."""
        if (
            self.supported_features & HumidifierEntityFeature.MODES == 0
            or not mode
        ):
            return
        if mode not in self.available_modes:
            _LOGGER.warning("Mode %s is not a valid operation mode", mode)
            return
        _LOGGER.debug("Setting the operation mode to: %s", mode)
        try:
            mode_enum = AirhumidifierOperationMode[mode]
        except KeyError:
            _LOGGER.warning("Mode %s is not a valid operation mode", mode)
            return
        if await self._try_command(
            "Setting operation mode of the miio device failed.",
            self._device.set_mode,
            mode_enum,
        ):
            self._mode = mode_enum.value
            self.async_write_ha_state()


class XiaomiAirHumidifierMiot(XiaomiAirHumidifier):
    """Representation of a Xiaomi Air Humidifier (MiOT protocol)."""

    MODE_MAPPING: Dict[AirhumidifierMiotOperationMode, str] = {
        AirhumidifierMiotOperationMode.Auto: "Auto",
        AirhumidifierMiotOperationMode.Low: "Low",
        AirhumidifierMiotOperationMode.Mid: "Mid",
        AirhumidifierMiotOperationMode.High: "High",
    }
    REVERSE_MODE_MAPPING: Dict[str, AirhumidifierMiotOperationMode] = {
        v: k for k, v in MODE_MAPPING.items()
    }

    @property
    def mode(self) -> Optional[str]:
        """Return the current mode."""
        if self._mode is not None:
            try:
                return AirhumidifierMiotOperationMode(self._mode).name
            except ValueError:
                return None
        return None

    @property
    def target_humidity(self) -> Optional[float]:
        """Return the target humidity."""
        if self._state:
            try:
                mode_enum = AirhumidifierMiotOperationMode(self._mode)
            except ValueError:
                return None
            if mode_enum == AirhumidifierMiotOperationMode.Auto:
                return self._target_humidity
        return None

    async def async_set_humidity(self, humidity: float) -> None:
        """Set the target humidity of the humidifier and set the mode to auto."""
        target_humidity = self.translate_humidity(humidity)
        if not target_humidity:
            return
        _LOGGER.debug("Setting the humidity to: %s", target_humidity)
        if await self._try_command(
            "Setting operation mode of the miio device failed.",
            self._device.set_target_humidity,
            target_humidity,
        ):
            self._target_humidity = target_humidity
        try:
            current_mode = AirhumidifierMiotOperationMode(self._attributes[ATTR_MODE])
        except ValueError:
            current_mode = None
        if (
            self.supported_features & HumidifierEntityFeature.MODES == 0
            or current_mode == AirhumidifierMiotOperationMode.Auto
        ):
            self.async_write_ha_state()
            return
        _LOGGER.debug("Setting the operation mode to: Auto")
        if await self._try_command(
            "Setting operation mode of the miio device to MODE_AUTO failed.",
            self._device.set_mode,
            AirhumidifierMiotOperationMode.Auto,
        ):
            self._mode = 0
            self.async_write_ha_state()

    async def async_set_mode(self, mode: Optional[str]) -> None:
        """Set the mode of the fan."""
        if (
            self.supported_features & HumidifierEntityFeature.MODES == 0
            or not mode
        ):
            return
        if mode not in self.REVERSE_MODE_MAPPING:
            _LOGGER.warning("Mode %s is not a valid operation mode", mode)
            return
        _LOGGER.debug("Setting the operation mode to: %s", mode)
        if self._state:
            mode_enum = self.REVERSE_MODE_MAPPING.get(mode)
            if mode_enum is None:
                _LOGGER.warning("Mode %s is not a valid operation mode", mode)
                return
            if await self._try_command(
                "Setting operation mode of the miio device failed.",
                self._device.set_mode,
                mode_enum,
            ):
                self._mode = mode_enum.value
                self.async_write_ha_state()


class XiaomiAirHumidifierMjjsq(XiaomiAirHumidifier):
    """Representation of a Xiaomi Air MJJSQ Humidifier."""

    MODE_MAPPING: Dict[str, AirhumidifierMjjsqOperationMode] = {
        "Low": AirhumidifierMjjsqOperationMode.Low,
        "Medium": AirhumidifierMjjsqOperationMode.Medium,
        "High": AirhumidifierMjjsqOperationMode.High,
        "Humidity": AirhumidifierMjjsqOperationMode.Humidity,
    }

    @property
    def mode(self) -> Optional[str]:
        """Return the current mode."""
        if self._mode is not None:
            try:
                return AirhumidifierMjjsqOperationMode(self._mode).name
            except ValueError:
                return None
        return None

    @property
    def target_humidity(self) -> Optional[float]:
        """Return the target humidity."""
        if self._state:
            try:
                mode_enum = AirhumidifierMjjsqOperationMode(self._mode)
            except ValueError:
                return None
            if mode_enum == AirhumidifierMjjsqOperationMode.Humidity:
                return self._target_humidity
        return None

    async def async_set_humidity(self, humidity: float) -> None:
        """Set the target humidity of the humidifier and set the mode to Humidity."""
        target_humidity = self.translate_humidity(humidity)
        if not target_humidity:
            return
        _LOGGER.debug("Setting the humidity to: %s", target_humidity)
        if await self._try_command(
            "Setting operation mode of the miio device failed.",
            self._device.set_target_humidity,
            target_humidity,
        ):
            self._target_humidity = target_humidity
        try:
            current_mode = AirhumidifierMjjsqOperationMode(
                self._attributes[ATTR_MODE]
            )
        except ValueError:
            current_mode = None
        if (
            self.supported_features & HumidifierEntityFeature.MODES == 0
            or current_mode == AirhumidifierMjjsqOperationMode.Humidity
        ):
            self.async_write_ha_state()
            return
        _LOGGER.debug("Setting the operation mode to: Humidity")
        if await self._try_command(
            "Setting operation mode of the miio device to MODE_HUMIDITY failed.",
            self._device.set_mode,
            AirhumidifierMjjsqOperationMode.Humidity,
        ):
            self._mode = AirhumidifierMjjsqOperationMode.Humidity.value
            self.async_write_ha_state()

    async def async_set_mode(self, mode: Optional[str]) -> None:
        """Set the mode of the fan."""
        if mode not in self.MODE_MAPPING:
            _LOGGER.warning("Mode %s is not a valid operation mode", mode)
            return
        _LOGGER.debug("Setting the operation mode to: %s", mode)
        mode_enum = self.MODE_MAPPING.get(mode)
        if mode_enum is None:
            _LOGGER.warning("Mode %s is not a valid operation mode", mode)
            return
        if self._state:
            if await self._try_command(
                "Setting operation mode of the miio device failed.",
                self._device.set_mode,
                mode_enum,
            ):
                self._mode = mode_enum.value
                self.async_write_ha_state()
