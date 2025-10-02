from pylutron import Keypad, Lutron, LutronEntity, LutronEvent
from homeassistant.const import ATTR_IDENTIFIERS, ATTR_VIA_DEVICE
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity
from .const import DOMAIN

class LutronBaseEntity(Entity):
    _attr_should_poll: bool = False
    _attr_has_entity_name: bool = True

    def __init__(self, area_name: str, lutron_device: LutronEntity, controller: Lutron):
        self._lutron_device: LutronEntity = lutron_device
        self._controller: Lutron = controller
        self._area_name: str = area_name

    async def async_added_to_hass(self):
        self._lutron_device.subscribe(self._update_callback, None)

    def _request_state(self) -> None:
        pass

    def _update_attrs(self) -> None:
        pass

    def _update_callback(self, _device, _context, _event, _params) -> None:
        self._update_attrs()
        self.schedule_update_ha_state()

    @property
    def unique_id(self) -> str:
        if self._lutron_device.uuid is None:
            return f'{self._controller.guid}_{self._lutron_device.legacy_uuid}'
        return f'{self._controller.guid}_{self._lutron_device.uuid}'

    def update(self) -> None:
        self._request_state()
        self._update_attrs()

class LutronDevice(LutronBaseEntity):
    def __init__(self, area_name: str, lutron_device: LutronEntity, controller: Lutron):
        super().__init__(area_name, lutron_device, controller)
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, self.unique_id)}, manufacturer='Lutron', name=lutron_device.name, suggested_area=area_name, via_device=(DOMAIN, controller.guid))

class LutronKeypad(LutronBaseEntity):
    def __init__(self, area_name: str, lutron_device: LutronEntity, controller: Lutron, keypad: Keypad):
        super().__init__(area_name, lutron_device, controller)
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, keypad.id)}, manufacturer='Lutron', name=keypad.name)
        if keypad.type == 'MAIN_REPEATER':
            self._attr_device_info[ATTR_IDENTIFIERS].add((DOMAIN, controller.guid))
        else:
            self._attr_device_info[ATTR_VIA_DEVICE] = (DOMAIN, controller.guid)
