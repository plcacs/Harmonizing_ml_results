from typing import Optional

async def async_set_preset_mode(hass: HomeAssistant, preset_mode: str, entity_id: str = ENTITY_MATCH_ALL) -> None:

def set_preset_mode(hass: HomeAssistant, preset_mode: str, entity_id: str = ENTITY_MATCH_ALL) -> None:

async def async_set_aux_heat(hass: HomeAssistant, aux_heat: bool, entity_id: str = ENTITY_MATCH_ALL) -> None:

def set_aux_heat(hass: HomeAssistant, aux_heat: bool, entity_id: str = ENTITY_MATCH_ALL) -> None:

async def async_set_temperature(hass: HomeAssistant, temperature: Optional[float] = None, entity_id: str = ENTITY_MATCH_ALL, target_temp_high: Optional[float] = None, target_temp_low: Optional[float] = None, hvac_mode: Optional[HVACMode] = None) -> None:

def set_temperature(hass: HomeAssistant, temperature: Optional[float] = None, entity_id: str = ENTITY_MATCH_ALL, target_temp_high: Optional[float] = None, target_temp_low: Optional[float] = None, hvac_mode: Optional[HVACMode] = None) -> None:

async def async_set_humidity(hass: HomeAssistant, humidity: float, entity_id: str = ENTITY_MATCH_ALL) -> None:

def set_humidity(hass: HomeAssistant, humidity: float, entity_id: str = ENTITY_MATCH_ALL) -> None:

async def async_set_fan_mode(hass: HomeAssistant, fan: str, entity_id: str = ENTITY_MATCH_ALL) -> None:

def set_fan_mode(hass: HomeAssistant, fan: str, entity_id: str = ENTITY_MATCH_ALL) -> None:

async def async_set_hvac_mode(hass: HomeAssistant, hvac_mode: HVACMode, entity_id: str = ENTITY_MATCH_ALL) -> None:

def set_operation_mode(hass: HomeAssistant, hvac_mode: HVACMode, entity_id: str = ENTITY_MATCH_ALL) -> None:

async def async_set_swing_mode(hass: HomeAssistant, swing_mode: str, entity_id: str = ENTITY_MATCH_ALL) -> None:

def set_swing_mode(hass: HomeAssistant, swing_mode: str, entity_id: str = ENTITY_MATCH_ALL) -> None:

async def async_turn_on(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:

async def async_turn_off(hass: HomeAssistant, entity_id: str = ENTITY_MATCH_ALL) -> None:
