    def __init__(self, hass: HomeAssistant, entry: CalDavConfigEntry, calendar: caldav.Calendar, days: int, include_all_day: bool, search: str) -> None:
    async def async_get_events(self, hass: HomeAssistant, start_date: datetime, end_date: datetime) -> List[CalendarEvent]:
    async def _async_update_data(self) -> CalendarEvent | None:
    @staticmethod
    def is_matching(vevent: Any, search: str) -> bool:
    @staticmethod
    def is_all_day(vevent: Any) -> bool:
    @staticmethod
    def is_over(vevent: Any) -> bool:
    @staticmethod
    def to_datetime(obj: Any) -> datetime:
    @staticmethod
    def to_local(obj: Any) -> datetime:
    @staticmethod
    def get_end_date(obj: Any) -> datetime:
