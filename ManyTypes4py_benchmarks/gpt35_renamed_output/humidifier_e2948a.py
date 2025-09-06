from homeassistant.helpers.entity_platform import EntityPlatform
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.typing import ConfigType, HomeAssistantType
from homeassistant.helpers.update_coordinator import Coordinator
from miio.device import Device

async def func_te30lkhr(hass: HomeAssistantType, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class XiaomiGenericHumidifier(XiaomiCoordinatedMiioEntity, HumidifierEntity):

    def __init__(self, device: Device, entry: ConfigEntry, unique_id: str, coordinator: Coordinator) -> None:

    @property
    def func_m5t3bmon(self) -> bool:

    @property
    def func_kdapfxyn(self) -> Any:

    async def func_k0vmi51z(self, **kwargs) -> None:

    async def func_hjp8cvxi(self, **kwargs) -> None:

    def func_t6us1li1(self, humidity: int) -> float:

class XiaomiAirHumidifier(XiaomiGenericHumidifier, HumidifierEntity):

    def __init__(self, device: Device, entry: ConfigEntry, unique_id: str, coordinator: Coordinator) -> None:

    @property
    def func_m5t3bmon(self) -> bool:

    @callback
    def func_1zj298v3(self) -> None:

    @property
    def func_kdapfxyn(self) -> str:

    @property
    def func_bktkkovz(self) -> Any:

    async def func_2n13i8gs(self, humidity: int) -> None:

    async def func_16qkhgxc(self, mode: str) -> None:

class XiaomiAirHumidifierMiot(XiaomiAirHumidifier):

    @property
    def func_kdapfxyn(self) -> str:

    @property
    def func_bktkkovz(self) -> Any:

    async def func_2n13i8gs(self, humidity: int) -> None:

    async def func_16qkhgxc(self, mode: str) -> None:

class XiaomiAirHumidifierMjjsq(XiaomiAirHumidifier):

    @property
    def func_kdapfxyn(self) -> str:

    @property
    def func_bktkkovz(self) -> Any:

    async def func_2n13i8gs(self, humidity: int) -> None:

    async def func_16qkhgxc(self, mode: str) -> None:
