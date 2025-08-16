def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

class RedditSensor(SensorEntity):
    def __init__(self, reddit: praw.Reddit, subreddit: str, limit: int, sort_by: str) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def native_value(self) -> int:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    @property
    def icon(self) -> str:
        ...

    def update(self) -> None:
        ...
