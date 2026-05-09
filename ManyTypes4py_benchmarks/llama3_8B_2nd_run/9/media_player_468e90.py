async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up MusicCast sensor based on a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    name = coordinator.data.network_name
    media_players = []
    for zone in coordinator.data.zones:
        zone_name = name if zone == DEFAULT_ZONE else f'{name} {zone}'
        media_players.append(MusicCastMediaPlayer(zone, zone_name, entry.entry_id, coordinator))  # type: MusicCastMediaPlayer
    async_add_entities(media_players)  # type: list[MusicCastMediaPlayer]

class MusicCastMediaPlayer(MusicCastDeviceEntity, MediaPlayerEntity):
    """The musiccast media player."""
    _attr_media_content_type: MediaType
    _attr_should_poll: bool

    def __init__(self, zone_id: str, name: str, entry_id: str, coordinator: MusicCastDataUpdateCoordinator) -> None:
        """Initialize the musiccast device."""
        self._player_state: MediaPlayerState
        self._volume_muted: bool
        self._shuffle: bool
        self._zone_id: str
        super().__init__(name=name, icon='mdi:speaker', coordinator=coordinator)  # type: MusicCastDeviceEntity
        self._volume_min: float
        self._volume_max: float
        self._cur_track: int
        self._repeat: RepeatMode

    async def async_added_to_hass(self) -> None:
        """Run when this Entity has been added to HA."""
        await super().async_added_to_hass()
        self.coordinator.entities.append(self)  # type: list[MusicCastMediaPlayer]
        self.coordinator.musiccast.register_group_update_callback(self.update_all_mc_entities)
        self.async_on_remove(self.coordinator.async_add_listener(self.async_schedule_check_client_list))  # type: callable

    # ... rest of the code ...
