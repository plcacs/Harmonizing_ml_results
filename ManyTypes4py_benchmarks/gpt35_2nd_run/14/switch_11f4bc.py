async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    entry_data: LutronData = hass.data[DOMAIN][config_entry.entry_id]
    entities: list[LutronDevice] = []
    for area_name, device in entry_data.switches:
        entities.append(LutronSwitch(area_name, device, entry_data.client))
    for area_name, keypad, scene, led in entry_data.scenes:
        if led is not None:
            entities.append(LutronLed(area_name, keypad, scene, led, entry_data.client))
    async_add_entities(entities, True)

class LutronSwitch(LutronDevice, SwitchEntity):
    _attr_name: str = None

    def turn_on(self, **kwargs: Any) -> None:
        self._lutron_device.level = 100

    def turn_off(self, **kwargs: Any) -> None:
        self._lutron_device.level = 0

    @property
    def extra_state_attributes(self) -> Mapping[str, Any]:
        return {'lutron_integration_id': self._lutron_device.id}

    def _request_state(self) -> None:
        _ = self._lutron_device.level

    def _update_attrs(self) -> None:
        self._attr_is_on = self._lutron_device.last_level() > 0

class LutronLed(LutronKeypad, SwitchEntity):
    def __init__(self, area_name: str, keypad: Keypad, scene_device: Output, led_device: Led, controller: Lutron) -> None:
        super().__init__(area_name, led_device, controller, keypad)
        self._keypad_name: str = keypad.name
        self._attr_name: str = scene_device.name

    def turn_on(self, **kwargs: Any) -> None:
        self._lutron_device.state = 1

    def turn_off(self, **kwargs: Any) -> None:
        self._lutron_device.state = 0

    @property
    def extra_state_attributes(self) -> Mapping[str, Any]:
        return {'keypad': self._keypad_name, 'scene': self._attr_name, 'led': self._lutron_device.name}

    def _request_state(self) -> None:
        _ = self._lutron_device.state

    def _update_attrs(self) -> None:
        self._attr_is_on = self._lutron_device.last_state
