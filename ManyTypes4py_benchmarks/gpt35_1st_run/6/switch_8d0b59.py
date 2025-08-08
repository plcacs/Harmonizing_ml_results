from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class XiaomiGenericSwitch(XiaomiDevice, SwitchEntity):
    def __init__(self, device: dict, name: str, data_key: str, supports_power_consumption: bool, xiaomi_hub: Any, config_entry: ConfigEntry) -> None:
        ...

    @property
    def icon(self) -> str:
        ...

    @property
    def is_on(self) -> bool:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    def turn_on(self, **kwargs) -> None:
        ...

    def turn_off(self, **kwargs) -> None:
        ...

    def parse_data(self, data: dict, raw_data: dict) -> bool:
        ...

    def update(self) -> None:
        ...
