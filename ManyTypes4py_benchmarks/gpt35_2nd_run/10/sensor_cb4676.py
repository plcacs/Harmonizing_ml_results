def validate_prices(func: Callable, entity: NordpoolBaseEntity, area: str, index: int) -> float:
def get_prices(entity: NordpoolBaseEntity) -> dict[str, tuple[float, float, float]]:
def get_min_max_price(entity: NordpoolBaseEntity, func: Callable) -> tuple[float, datetime, datetime]:
def get_blockprices(entity: NordpoolBaseEntity) -> dict[str, dict[str, tuple[datetime, datetime, float, float, float]]]:
async def async_setup_entry(hass: HomeAssistant, entry: NordPoolConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
class NordpoolSensor(NordpoolBaseEntity, SensorEntity):
class NordpoolPriceSensor(NordpoolBaseEntity, SensorEntity):
class NordpoolBlockPriceSensor(NordpoolBaseEntity, SensorEntity):
class NordpoolDailyAveragePriceSensor(NordpoolBaseEntity, SensorEntity):
