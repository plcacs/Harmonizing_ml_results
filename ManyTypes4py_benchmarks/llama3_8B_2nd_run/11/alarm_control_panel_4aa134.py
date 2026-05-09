async def async_setup_entry(
    hass: HomeAssistant, 
    entry: AlarmDecoderConfigEntry, 
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    # ...

class AlarmDecoderAlarmPanel(AlarmDecoderEntity, AlarmControlPanelEntity):
    # ...

    def __init__(
        self, 
        client: object, 
        auto_bypass: bool, 
        code_arm_required: bool, 
        alt_night_mode: bool
    ) -> None:
        # ...

    async def async_added_to_hass(self) -> None:
        # ...

    def _message_callback(self, message: object) -> None:
        # ...

    def alarm_disarm(self, code: str | None = None) -> None:
        # ...

    def alarm_arm_away(self, code: str | None = None) -> None:
        # ...

    def alarm_arm_home(self, code: str | None = None) -> None:
        # ...

    def alarm_arm_night(self, code: str | None = None) -> None:
        # ...

    def alarm_toggle_chime(self, code: str | None = None) -> None:
        # ...

    def alarm_keypress(self, keypress: str) -> None:
        # ...
