class ProtectMediaSource(MediaSource):
    """Represents all UniFi Protect NVRs."""
    name: str = 'UniFi Protect'

    def __init__(self, hass: HomeAssistant, data_sources: dict[str, ProtectData]):
        """Initialize the UniFi Protect media source."""
        super().__init__(DOMAIN)
        self.hass = hass
        self.data_sources = data_sources
        self._registry: Optional[entity_registry.EntityRegistry] = None

    async def async_resolve_media(self, item: MediaSourceItem) -> PlayMedia:
        """Return a streamable URL and associated mime type for a UniFi Protect event."""
        # ...

    async def async_browse_media(self, item: MediaSourceItem) -> BrowseMediaSource:
        """Return a browsable UniFi Protect media source."""
        # ...

    async def _build_event(self, data: ProtectData, event: Union[Event, dict], thumbnail_only: bool) -> BrowseMediaSource:
        """Build media source for an individual event."""
        # ...

    async def _build_events(self, data: ProtectData, start: datetime, end: datetime, camera_id: str, event_types: set[EventType], reserve: bool) -> list[BrowseMediaSource]:
        """Build media source for a given range of time and event type."""
        # ...

    async def _build_recent(self, data: ProtectData, camera_id: str, event_type: SimpleEventType, days: int, build_children: bool) -> BrowseMediaSource:
        """Build media source for events in relative days."""
        # ...

    async def _build_month(self, data: ProtectData, camera_id: str, event_type: SimpleEventType, start: date, build_children: bool) -> BrowseMediaSource:
        """Build media source for selectors for a given month."""
        # ...

    async def _build_days(self, data: ProtectData, camera_id: str, event_type: SimpleEventType, start: date, is_all: bool, build_children: bool) -> BrowseMediaSource:
        """Build media source for events for a given day or whole month."""
        # ...

    async def _build_events_type(self, data: ProtectData, camera_id: str, event_type: SimpleEventType, build_children: bool) -> BrowseMediaSource:
        """Build folder media source for a selectors for a given event type."""
        # ...

    async def _build_camera(self, data: ProtectData, camera_id: str, build_children: bool) -> BrowseMediaSource:
        """Build media source for selectors for a UniFi Protect camera."""
        # ...

    async def _build_cameras(self, data: ProtectData) -> list[BrowseMediaSource]:
        """Build media source for a single UniFi Protect NVR."""
        # ...

    async def _build_console(self, data: ProtectData) -> BrowseMediaSource:
        """Build media source for a single UniFi Protect NVR."""
        # ...

    async def _build_sources(self) -> BrowseMediaSource:
        """Return all media source for all UniFi Protect NVRs."""
        # ...
