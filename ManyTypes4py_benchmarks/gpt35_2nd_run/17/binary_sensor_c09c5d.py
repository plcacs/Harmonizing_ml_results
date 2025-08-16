def validate_dates(holiday_list: list[str]) -> list[str]:
def _get_obj_holidays(country: str, province: str, year: int, language: str, categories: list[str]) -> HolidayBase:
async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
class IsWorkdaySensor(BinarySensorEntity):
    def __init__(self, obj_holidays: HolidayBase, workdays: list[str], excludes: list[str], days_offset: int, name: str, entry_id: str) -> None:
    def is_include(self, day: str, now: datetime) -> bool:
    def is_exclude(self, day: str, now: datetime) -> bool:
    def get_next_interval(self, now: datetime) -> datetime:
    def _update_state_and_setup_listener(self) -> None:
    def point_in_time_listener(self, time_date: datetime) -> None:
    async def async_added_to_hass(self) -> None:
    def update_data(self, now: datetime) -> None:
    def check_date(self, check_date: date) -> dict[str, bool]:
    def date_is_workday(self, check_date: date) -> bool:
