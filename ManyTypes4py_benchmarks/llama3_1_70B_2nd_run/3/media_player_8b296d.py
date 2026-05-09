class YamahaConfigInfo:
    """Configuration Info for Yamaha Receivers."""

    def __init__(self, config: ConfigType, discovery_info: DiscoveryInfoType) -> None:
        """Initialize the Configuration Info for Yamaha Receiver."""
        self.name: str = config.get(CONF_NAME)
        self.host: str = config.get(CONF_HOST)
        self.ctrl_url: str = f'http://{self.host}:80/YamahaRemoteControl/ctrl'
        self.source_ignore: list[str] = config.get(CONF_SOURCE_IGNORE)
        self.source_names: dict[str, str] = config.get(CONF_SOURCE_NAMES)
        self.zone_ignore: list[str] = config.get(CONF_ZONE_IGNORE)
        self.zone_names: dict[str, str] = config.get(CONF_ZONE_NAMES)
        self.from_discovery: bool = False
        _LOGGER.debug('Discovery Info: %s', discovery_info)
        if discovery_info is not None:
            self.name: str = discovery_info.get('name')
            self.model: str = discovery_info.get('model_name')
            self.ctrl_url: str = discovery_info.get('control_url')
            self.desc_url: str = discovery_info.get('description_url')
            self.zone_ignore: list[str] = []
            self.from_discovery: bool = True

def _discovery(config_info: YamahaConfigInfo) -> list:
    """Discover list of zone controllers from configuration in the network."""
    if config_info.from_discovery:
        _LOGGER.debug('Discovery Zones')
        zones: list = rxv.RXV(config_info.ctrl_url, model_name=config_info.model, friendly_name=config_info.name, unit_desc_url=config_info.desc_url).zone_controllers()
    elif config_info.host is None:
        _LOGGER.debug('Config No Host Supplied Zones')
        zones: list = []
        for recv in rxv.find(DISCOVER_TIMEOUT):
            zones.extend(recv.zone_controllers())
    else:
        _LOGGER.debug('Config Zones')
        zones: list = rxv.RXV(config_info.ctrl_url, config_info.name).zone_controllers()
    _LOGGER.debug('Returned _discover zones: %s', zones)
    return zones

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    """Set up the Yamaha platform."""
    known_zones: set[str] = hass.data.setdefault(DOMAIN, {KNOWN_ZONES: set()})[KNOWN_ZONES]
    _LOGGER.debug('Known receiver zones: %s', known_zones)
    config_info: YamahaConfigInfo = YamahaConfigInfo(config=config, discovery_info=discovery_info)
    try:
        zone_ctrls: list = await hass.async_add_executor_job(_discovery, config_info)
    except requests.exceptions.ConnectionError as ex:
        raise PlatformNotReady(f'Issue while connecting to {config_info.name}') from ex
    entities: list[YamahaDeviceZone] = []
    for zctrl in zone_ctrls:
        _LOGGER.debug('Receiver zone: %s serial %s', zctrl.zone, zctrl.serial_number)
        if config_info.zone_ignore and zctrl.zone in config_info.zone_ignore:
            _LOGGER.debug('Ignore receiver zone: %s %s', config_info.name, zctrl.zone)
            continue
        assert config_info.name
        entity: YamahaDeviceZone = YamahaDeviceZone(config_info.name, zctrl, config_info.source_ignore, config_info.source_names, config_info.zone_names)
        if entity.zone_id not in known_zones:
            known_zones.add(entity.zone_id)
            entities.append(entity)
        else:
            _LOGGER.debug('Ignoring duplicate zone: %s %s', config_info.name, zctrl.zone)
    async_add_entities(entities)
    platform: entity_platform.AsyncServiceRegistry = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(SERVICE_SELECT_SCENE, {vol.Required(ATTR_SCENE): cv.string}, 'set_scene')
    platform.async_register_entity_service(SERVICE_ENABLE_OUTPUT, {vol.Required(ATTR_ENABLED): cv.boolean, vol.Required(ATTR_PORT): cv.string}, 'enable_output')
    platform.async_register_entity_service(SERVICE_MENU_CURSOR, {vol.Required(ATTR_CURSOR): vol.In(CURSOR_TYPE_MAP)}, YamahaDeviceZone.menu_cursor.__name__)

class YamahaDeviceZone(MediaPlayerEntity):
    """Representation of a Yamaha device zone."""

    def __init__(self, name: str, zctrl: Any, source_ignore: list[str], source_names: dict[str, str], zone_names: dict[str, str]) -> None:
        """Initialize the Yamaha Receiver."""
        self.zctrl: Any = zctrl
        self._attr_is_volume_muted: bool = False
        self._attr_volume_level: float = 0
        self._attr_state: MediaPlayerState = MediaPlayerState.OFF
        self._source_ignore: list[str] = source_ignore or []
        self._source_names: dict[str, str] = source_names or {}
        self._zone_names: dict[str, str] = zone_names or {}
        self._playback_support: Any = None
        self._is_playback_supported: bool = False
        self._play_status: Any = None
        self._name: str = name
        self._zone: str = zctrl.zone
        if self.zctrl.serial_number is not None:
            self._attr_unique_id: str = f'{self.zctrl.serial_number}_{self._zone}'

    def update(self) -> None:
        """Get the latest details from the device."""
        try:
            self._play_status = self.zctrl.play_status()
        except requests.exceptions.ConnectionError:
            _LOGGER.debug('Receiver is offline: %s', self._name)
            self._attr_available = False
            return
        self._attr_available = True
        if self.zctrl.on:
            if self._play_status is None:
                self._attr_state = MediaPlayerState.ON
            elif self._play_status.playing:
                self._attr_state = MediaPlayerState.PLAYING
            else:
                self._attr_state = MediaPlayerState.IDLE
        else:
            self._attr_state = MediaPlayerState.OFF
        self._attr_is_volume_muted = self.zctrl.mute
        self._attr_volume_level = self.zctrl.volume / 100 + 1
        if self.source_list is None:
            self.build_source_list()
        current_source = self.zctrl.input
        self._attr_source = self._source_names.get(current_source, current_source)
        self._playback_support = self.zctrl.get_playback_support()
        self._is_playback_supported = self.zctrl.is_playback_supported(self._attr_source)
        surround_programs = self.zctrl.surround_programs()
        if surround_programs:
            self._attr_sound_mode = self.zctrl.surround_program
            self._attr_sound_mode_list = surround_programs
        else:
            self._attr_sound_mode = None
            self._attr_sound_mode_list = None

    def build_source_list(self) -> None:
        """Build the source list."""
        self._reverse_mapping: dict[str, str] = {alias: source for source, alias in self._source_names.items()}
        self._attr_source_list: list[str] = sorted((self._source_names.get(source, source) for source in self.zctrl.inputs() if source not in self._source_ignore))

    @property
    def name(self) -> str:
        """Return the name of the device."""
        name: str = self._name
        zone_name: str = self._zone_names.get(self._zone, self._zone)
        if zone_name != 'Main_Zone':
            name += f' {zone_name.replace("_", " ")}'
        return name

    @property
    def zone_id(self) -> str:
        """Return a zone_id to ensure 1 media player per zone."""
        return f'{self.zctrl.ctrl_url}:{self._zone}'

    @property
    def supported_features(self) -> int:
        """Flag media player features that are supported."""
        supported_features: int = SUPPORT_YAMAHA
        supports: Any = self._playback_support
        mapping: dict[str, int] = {'play': MediaPlayerEntityFeature.PLAY | MediaPlayerEntityFeature.PLAY_MEDIA, 'pause': MediaPlayerEntityFeature.PAUSE, 'stop': MediaPlayerEntityFeature.STOP, 'skip_f': MediaPlayerEntityFeature.NEXT_TRACK, 'skip_r': MediaPlayerEntityFeature.PREVIOUS_TRACK}
        for attr, feature in mapping.items():
            if getattr(supports, attr, False):
                supported_features |= feature
        return supported_features

    def turn_off(self) -> None:
        """Turn off media player."""
        self.zctrl.on = False

    def set_volume_level(self, volume: float) -> None:
        """Set volume level, range 0..1."""
        zone_vol: float = 100 - volume * 100
        negative_zone_vol: float = -zone_vol
        self.zctrl.volume = negative_zone_vol

    def mute_volume(self, mute: bool) -> None:
        """Mute (true) or unmute (false) media player."""
        self.zctrl.mute = mute

    def turn_on(self) -> None:
        """Turn the media player on."""
        self.zctrl.on = True
        self._attr_volume_level = self.zctrl.volume / 100 + 1

    def media_play(self) -> None:
        """Send play command."""
        self._call_playback_function(self.zctrl.play, 'play')

    def media_pause(self) -> None:
        """Send pause command."""
        self._call_playback_function(self.zctrl.pause, 'pause')

    def media_stop(self) -> None:
        """Send stop command."""
        self._call_playback_function(self.zctrl.stop, 'stop')

    def media_previous_track(self) -> None:
        """Send previous track command."""
        self._call_playback_function(self.zctrl.previous, 'previous track')

    def media_next_track(self) -> None:
        """Send next track command."""
        self._call_playback_function(self.zctrl.next, 'next track')

    def _call_playback_function(self, function: Any, function_text: str) -> None:
        try:
            function()
        except rxv.exceptions.ResponseException:
            _LOGGER.warning('Failed to execute %s on %s', function_text, self._name)

    def select_source(self, source: str) -> None:
        """Select input source."""
        self.zctrl.input = self._reverse_mapping.get(source, source)

    def play_media(self, media_type: str, media_id: str, **kwargs: Any) -> None:
        """Play media from an ID.

        This exposes a pass through for various input sources in the
        Yamaha to direct play certain kinds of media. media_type is
        treated as the input type that we are setting, and media id is
        specific to it.
        For the NET RADIO mediatype the format for ``media_id`` is a
        "path" in your vtuner hierarchy. For instance:
        ``Bookmarks>Internet>Radio Paradise``. The separators are
        ``>`` and the parts of this are navigated by name behind the
        scenes. There is a looping construct built into the yamaha
        library to do this with a fallback timeout if the vtuner
        service is unresponsive.
        NOTE: this might take a while, because the only API interface
        for setting the net radio station emulates button pressing and
        navigating through the net radio menu hierarchy. And each sub
        menu must be fetched by the receiver from the vtuner service.
        """
        if media_type == 'NET RADIO':
            self.zctrl.net_radio(media_id)

    def enable_output(self, port: str, enabled: bool) -> None:
        """Enable or disable an output port.."""
        self.zctrl.enable_output(port, enabled)

    def menu_cursor(self, cursor: str) -> None:
        """Press a menu cursor button."""
        getattr(self.zctrl, CURSOR_TYPE_MAP[cursor])()

    def set_scene(self, scene: str) -> None:
        """Set the current scene."""
        try:
            self.zctrl.scene = scene
        except AssertionError:
            _LOGGER.warning("Scene '%s' does not exist!", scene)

    def select_sound_mode(self, sound_mode: str) -> None:
        """Set Sound Mode for Receiver.."""
        self.zctrl.surround_program = sound_mode

    @property
    def media_artist(self) -> str | None:
        """Artist of current playing media."""
        if self._play_status is not None:
            return self._play_status.artist
        return None

    @property
    def media_album_name(self) -> str | None:
        """Album of current playing media."""
        if self._play_status is not None:
            return self._play_status.album
        return None

    @property
    def media_content_type(self) -> str | None:
        """Content type of current playing media."""
        if self._is_playback_supported:
            return MediaType.MUSIC
        return None

    @property
    def media_title(self) -> str | None:
        """Artist of current playing media."""
        if self._play_status is not None:
            song: str = self._play_status.song
            station: str = self._play_status.station
            if song and station:
                return f'{station}: {song}'
            return song or station
        return None
