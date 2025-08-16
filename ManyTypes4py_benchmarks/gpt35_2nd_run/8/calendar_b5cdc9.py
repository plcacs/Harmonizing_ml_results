def _get_obj_holidays_and_language(country: str, province: str, language: str, selected_categories: list[str] | None) -> tuple[HolidayBase, str]:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class HolidayCalendarEntity(CalendarEntity):
    _attr_name: str | None = None
    _attr_event: CalendarEvent | None = None
    _attr_should_poll: bool = False
    unsub: CALLBACK_TYPE | None = None

    def __init__(self, name: str, country: str, province: str | None, language: str, categories: list[str] | None, obj_holidays: HolidayBase, unique_id: str) -> None:
        ...

    def get_next_interval(self, now: datetime) -> datetime:
        ...

    def _update_state_and_setup_listener(self) -> None:
        ...

    @callback
    def point_in_time_listener(self, time_date: datetime) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def update_event(self, now: datetime) -> CalendarEvent | None:
        ...

    @property
    def event(self) -> CalendarEvent | None:
        ...

    async def async_get_events(self, hass: HomeAssistant, start_date: datetime, end_date: datetime) -> list[CalendarEvent]:
        ...
