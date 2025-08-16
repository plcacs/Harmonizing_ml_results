from homeassistant.components.light import LightEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from typing import Any

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    entities: list[XiaomiGatewayLight] = []
    gateway = hass.data[DOMAIN][GATEWAYS_KEY][config_entry.entry_id]
    for device in gateway.devices['light']:
        model = device['model']
        if model in ('gateway', 'gateway.v3'):
            entities.append(XiaomiGatewayLight(device, 'Gateway Light', gateway, config_entry))
    async_add_entities(entities)

class XiaomiGatewayLight(XiaomiDevice, LightEntity):
    _attr_color_mode: ColorMode = ColorMode.HS
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.HS}

    def __init__(self, device: dict, name: str, xiaomi_hub: Any, config_entry: ConfigEntry) -> None:
        self._data_key: str = 'rgb'
        self._hs: tuple[int, int] = (0, 0)
        self._brightness: int = 100
        super().__init__(device, name, xiaomi_hub, config_entry)

    @property
    def is_on(self) -> bool:
        return self._state

    def parse_data(self, data: dict, raw_data: bytes) -> bool:
        value = data.get(self._data_key)
        if value is None:
            return False
        if value == 0:
            self._state = False
            return True
        rgbhexstr = f'{value:x}'
        if len(rgbhexstr) > 8:
            _LOGGER.error("Light RGB data error. Can't be more than 8 characters. Received: %s", rgbhexstr)
            return False
        rgbhexstr = rgbhexstr.zfill(8)
        rgbhex = bytes.fromhex(rgbhexstr)
        rgba = struct.unpack('BBBB', rgbhex)
        brightness = rgba[0]
        rgb = rgba[1:]
        self._brightness = brightness
        self._hs = color_util.color_RGB_to_hs(*rgb)
        self._state = True
        return True

    @property
    def brightness(self) -> int:
        return int(255 * self._brightness / 100)

    @property
    def hs_color(self) -> tuple[int, int]:
        return self._hs

    def turn_on(self, **kwargs) -> None:
        if ATTR_HS_COLOR in kwargs:
            self._hs = kwargs[ATTR_HS_COLOR]
        if ATTR_BRIGHTNESS in kwargs:
            self._brightness = int(100 * kwargs[ATTR_BRIGHTNESS] / 255)
        rgb = color_util.color_hs_to_RGB(*self._hs)
        rgba = (self._brightness, *rgb)
        rgbhex = binascii.hexlify(struct.pack('BBBB', *rgba)).decode('ASCII')
        rgbhex = int(rgbhex, 16)
        if self._write_to_hub(self._sid, **{self._data_key: rgbhex}):
            self._state = True
            self.schedule_update_ha_state()

    def turn_off(self, **kwargs) -> None:
        if self._write_to_hub(self._sid, **{self._data_key: 0}):
            self._state = False
            self.schedule_update_ha_state()
