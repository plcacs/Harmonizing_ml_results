from __future__ import annotations
from contextlib import suppress
import logging
from typing import Any, List

class ViCareClimate(ViCareEntity, ClimateEntity):
    """Representation of the ViCare heating climate device."""
    _attr_precision: float = PRECISION_TENTHS
    _attr_supported_features: ClimateEntityFeature = ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.PRESET_MODE | ClimateEntityFeature.TURN_OFF | ClimateEntityFeature.TURN_ON
    _attr_temperature_unit: UnitOfTemperature = UnitOfTemperature.CELSIUS
    _attr_min_temp: float = VICARE_TEMP_HEATING_MIN
    _attr_max_temp: float = VICARE_TEMP_HEATING_MAX
    _attr_target_temperature_step: int = PRECISION_WHOLE
    _attr_translation_key: str = 'heating'
    _current_action: bool | None = None
    _current_mode: str | None = None
    _current_program: str | None = None

    def __init__(self, device_serial: str, device_config: dict, device: PyViCareDevice, circuit: PyViCareHeatingCircuit) -> None:
        """Initialize the climate device."""
        super().__init__(self._attr_translation_key, device_serial, device_config, device, circuit)
        self._device = device
        self._attributes: dict[str, Any] = {}
        self._attributes['vicare_programs'] = self._api.getPrograms()
        self._attr_preset_modes: List[str] = [preset for heating_program in self._attributes['vicare_programs'] if (preset := HeatingProgram.to_ha_preset(heating_program)) is not None]

    def update(self) -> None:
        """Let HA know there has been an update from the ViCare API."""
        try:
            _room_temperature: float | None = None
            with suppress(PyViCareNotSupportedFeatureError):
                self._attributes['room_temperature'] = _room_temperature = self._api.getRoomTemperature()
            _supply_temperature: float | None = None
            with suppress(PyViCareNotSupportedFeatureError):
                _supply_temperature = self._api.getSupplyTemperature()
            if _room_temperature is not None:
                self._attr_current_temperature = _room_temperature
            elif _supply_temperature is not None:
                self._attr_current_temperature = _supply_temperature
            else:
                self._attr_current_temperature = None
            with suppress(PyViCareNotSupportedFeatureError):
                self._attributes['active_vicare_program'] = self._current_program = self._api.getActiveProgram()
            with suppress(PyViCareNotSupportedFeatureError):
                self._attr_target_temperature = self._api.getCurrentDesiredTemperature()
            with suppress(PyViCareNotSupportedFeatureError):
                self._attributes['active_vicare_mode'] = self._current_mode = self._api.getActiveMode()
            with suppress(PyViCareNotSupportedFeatureError):
                self._attributes['heating_curve_slope'] = self._api.getHeatingCurveSlope()
            with suppress(PyViCareNotSupportedFeatureError):
                self._attributes['heating_curve_shift'] = self._api.getHeatingCurveShift()
            with suppress(PyViCareNotSupportedFeatureError):
                self._attributes['vicare_modes'] = self._api.getModes()
            self._current_action = False
            with suppress(PyViCareNotSupportedFeatureError):
                for burner in get_burners(self._device):
                    self._current_action = self._current_action or burner.getActive()
            with suppress(PyViCareNotSupportedFeatureError):
                for compressor in get_compressors(self._device):
                    self._current_action = self._current_action or compressor.getActive()
        except requests.exceptions.ConnectionError:
            _LOGGER.error('Unable to retrieve data from ViCare server')
        except PyViCareRateLimitError as limit_exception:
            _LOGGER.error('Vicare API rate limit exceeded: %s', limit_exception)
        except ValueError:
            _LOGGER.error('Unable to decode data from ViCare server')
        except PyViCareInvalidDataError as invalid_data_exception:
            _LOGGER.error('Invalid data from Vicare server: %s', invalid_data_exception)

    @property
    def hvac_mode(self) -> HVACMode | None:
        """Return current hvac mode."""
        if self._current_mode is None:
            return None
        return VICARE_TO_HA_HVAC_HEATING.get(self._current_mode, None)

    def set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set a new hvac mode on the ViCare API."""
        if 'vicare_modes' not in self._attributes:
            raise ValueError('Cannot set hvac mode when vicare_modes are not known')
        vicare_mode = self.vicare_mode_from_hvac_mode(hvac_mode)
        if vicare_mode is None:
            raise ValueError(f'Cannot set invalid hvac mode: {hvac_mode}')
        _LOGGER.debug('Setting hvac mode to %s / %s', hvac_mode, vicare_mode)
        self._api.setMode(vicare_mode)

    def vicare_mode_from_hvac_mode(self, hvac_mode: HVACMode) -> str | None:
        """Return the corresponding vicare mode for an hvac_mode."""
        if 'vicare_modes' not in self._attributes:
            return None
        supported_modes = self._attributes['vicare_modes']
        for key, value in VICARE_TO_HA_HVAC_HEATING.items():
            if key in supported_modes and value == hvac_mode:
                return key
        return None

    @property
    def hvac_modes(self) -> List[HVACMode]:
        """Return the list of available hvac modes."""
        if 'vicare_modes' not in self._attributes:
            return []
        supported_modes = self._attributes['vicare_modes']
        hvac_modes = []
        for key, value in VICARE_TO_HA_HVAC_HEATING.items():
            if value in hvac_modes:
                continue
            if key in supported_modes:
                hvac_modes.append(value)
        return hvac_modes

    @property
    def hvac_action(self) -> HVACAction:
        """Return the current hvac action."""
        if self._current_action:
            return HVACAction.HEATING
        return HVACAction.IDLE

    def set_temperature(self, **kwargs) -> None:
        """Set new target temperatures."""
        if (temp := kwargs.get(ATTR_TEMPERATURE)) is not None:
            self._api.setProgramTemperature(self._current_program, temp)
            self._attr_target_temperature = temp

    @property
    def preset_mode(self) -> str | None:
        """Return the current preset mode, e.g., home, away, temp."""
        return HeatingProgram.to_ha_preset(self._current_program)

    def set_preset_mode(self, preset_mode: str) -> None:
        """Set new preset mode and deactivate any existing programs."""
        target_program = HeatingProgram.from_ha_preset(preset_mode, self._attributes['vicare_programs'])
        if target_program is None:
            raise ServiceValidationError(translation_domain=DOMAIN, translation_key='program_unknown', translation_placeholders={'preset': preset_mode})
        _LOGGER.debug('Current preset %s', self._current_program)
        if self._current_program and self._current_program in CHANGABLE_HEATING_PROGRAMS:
            _LOGGER.debug('deactivating %s', self._current_program)
            try:
                self._api.deactivateProgram(self._current_program)
            except PyViCareCommandError as err:
                raise ServiceValidationError(translation_domain=DOMAIN, translation_key='program_not_deactivated', translation_placeholders={'program': self._current_program}) from err
        _LOGGER.debug('Setting preset to %s / %s', preset_mode, target_program)
        if target_program in CHANGABLE_HEATING_PROGRAMS:
            _LOGGER.debug('activating %s', target_program)
            try:
                self._api.activateProgram(target_program)
            except PyViCareCommandError as err:
                raise ServiceValidationError(translation_domain=DOMAIN, translation_key='program_not_activated', translation_placeholders={'program': target_program}) from err

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Show Device Attributes."""
        return self._attributes

    def set_vicare_mode(self, vicare_mode: str) -> None:
        """Service function to set vicare modes directly."""
        if vicare_mode not in self._attributes['vicare_modes']:
            raise ValueError(f'Cannot set invalid vicare mode: {vicare_mode}.')
        self._api.setMode(vicare_mode)
