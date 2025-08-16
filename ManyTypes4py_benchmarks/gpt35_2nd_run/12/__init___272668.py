def validate_set_datetime_attrs(config: VolDictType) -> VolDictType:
    ...

class DateTimeStorageCollection(collection.DictStorageCollection):
    CREATE_UPDATE_SCHEMA: vol.Schema = vol.Schema(vol.All(STORAGE_FIELDS, has_date_or_time)

    async def _process_create_data(self, data: VolDictType) -> VolDictType:
        ...

    @callback
    def _get_suggested_id(self, info: VolDictType) -> str:
        ...

    async def _update_data(self, item: VolDictType, update_data: VolDictType) -> VolDictType:
        ...

class InputDatetime(collection.CollectionEntity, RestoreEntity):
    _unrecorded_attributes: frozenset = frozenset({ATTR_EDITABLE, CONF_HAS_DATE, CONF_HAS_TIME})
    _attr_should_poll: bool = False

    def __init__(self, config: VolDictType) -> None:
        ...

    @classmethod
    def from_storage(cls, config: VolDictType) -> InputDatetime:
        ...

    @classmethod
    def from_yaml(cls, config: VolDictType) -> InputDatetime:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def has_date(self) -> bool:
        ...

    @property
    def has_time(self) -> bool:
        ...

    @property
    def icon(self) -> str:
        ...

    @property
    def state(self) -> str:
        ...

    @property
    def capability_attributes(self) -> VolDictType:
        ...

    @property
    def extra_state_attributes(self) -> VolDictType:
        ...

    @property
    def unique_id(self) -> str:
        ...

    @callback
    def async_set_datetime(self, date: py_datetime.date = None, time: py_datetime.time = None, datetime: py_datetime.datetime = None, timestamp: float = None) -> None:
        ...

    async def async_update_config(self, config: VolDictType) -> None:
        ...
