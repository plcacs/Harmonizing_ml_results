class RachioPerson:
    """Represent a Rachio user."""

    def __init__(self, rachio: Rachio, config_entry: ConfigEntry):
        """Create an object from the provided API instance."""
        self.rachio: Rachio = rachio
        self.config_entry: ConfigEntry = config_entry
        self.username: str | None = None
        self._id: str | None = None
        self._controllers: list[RachioIro] = []
        self._base_stations: list[RachioBaseStation] = []

    # ...

class RachioIro:
    """Represent a Rachio Iro."""

    def __init__(self, hass: HomeAssistant, rachio: Rachio, data: dict, webhooks: dict):
        """Initialize a Rachio device."""
        self.hass: HomeAssistant = hass
        self.rachio: Rachio = rachio
        self._id: str = data[KEY_ID]
        self.name: str = data[KEY_NAME]
        self.serial_number: str = data[KEY_SERIAL_NUMBER]
        self.mac_address: str = data[KEY_MAC_ADDRESS]
        self.model: str = data[KEY_MODEL]
        self._zones: list[dict] = data[KEY_ZONES]
        self._schedules: list[dict] = data[KEY_SCHEDULES]
        self._flex_schedules: list[dict] = data[KEY_FLEX_SCHEDULES]
        self._init_data: dict = data
        self._webhooks: dict = webhooks
        _LOGGER.debug('%s has ID "%s"', self, self.controller_id)

    # ...

class RachioBaseStation:
    """Represent a smart hose timer base station."""

    def __init__(self, rachio: Rachio, data: dict, status_coordinator: RachioUpdateCoordinator, schedule_coordinator: RachioScheduleUpdateCoordinator):
        """Initialize a smart hose timer base station."""
        self.rachio: Rachio = rachio
        self._id: str = data[KEY_ID]
        self.status_coordinator: RachioUpdateCoordinator = status_coordinator
        self.schedule_coordinator: RachioScheduleUpdateCoordinator = schedule_coordinator
