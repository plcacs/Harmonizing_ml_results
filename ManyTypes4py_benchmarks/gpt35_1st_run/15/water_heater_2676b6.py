from homeassistant.components.water_heater import WaterHeaterEntity, WaterHeaterEntityFeature
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import VolDictType
from .const import CONST_HVAC_HEAT, CONST_MODE_AUTO, CONST_MODE_HEAT, CONST_MODE_OFF, CONST_MODE_SMART_SCHEDULE, CONST_OVERLAY_MANUAL, CONST_OVERLAY_TADO_MODE, CONST_OVERLAY_TIMER, TYPE_HOT_WATER
from .coordinator import TadoDataUpdateCoordinator
from .entity import TadoZoneEntity
from .repairs import manage_water_heater_fallback_issue

async def async_setup_entry(hass: HomeAssistant, entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

async def _generate_entities(coordinator: TadoDataUpdateCoordinator) -> List[TadoWaterHeater]:
    ...

async def create_water_heater_entity(coordinator: TadoDataUpdateCoordinator, name: str, zone_id: str, zone: str) -> TadoWaterHeater:
    ...

class TadoWaterHeater(TadoZoneEntity, WaterHeaterEntity):
    def __init__(self, coordinator: TadoDataUpdateCoordinator, zone_name: str, zone_id: str, supports_temperature_control: bool, min_temp: float, max_temp: float) -> None:
        ...

    @callback
    def _handle_coordinator_update(self) -> None:
        ...

    @property
    def current_operation(self) -> str:
        ...

    @property
    def target_temperature(self) -> float:
        ...

    @property
    def is_away_mode_on(self) -> bool:
        ...

    @property
    def min_temp(self) -> float:
        ...

    @property
    def max_temp(self) -> float:
        ...

    async def async_set_operation_mode(self, operation_mode: str) -> None:
        ...

    async def set_timer(self, time_period: str, temperature: float = None) -> None:
        ...

    async def async_set_temperature(self, **kwargs: Any) -> None:
        ...

    @callback
    def _async_update_callback(self) -> None:
        ...

    @callback
    def _async_update_data(self) -> None:
        ...

    async def _control_heater(self, hvac_mode: str = None, target_temp: float = None, duration: float = None) -> None:
        ...
