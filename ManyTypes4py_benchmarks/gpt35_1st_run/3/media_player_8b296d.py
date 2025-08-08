def _discovery(config_info: YamahaConfigInfo) -> list:
async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
class YamahaConfigInfo:
    def __init__(self, config: ConfigType, discovery_info: DiscoveryInfoType) -> None:
    def update(self) -> None:
    def build_source_list(self) -> None:
    @property
    def name(self) -> str:
    @property
    def zone_id(self) -> str:
    @property
    def supported_features(self) -> int:
    def turn_off(self) -> None:
    def set_volume_level(self, volume: float) -> None:
    def mute_volume(self, mute: bool) -> None:
    def turn_on(self) -> None:
    def media_play(self) -> None:
    def media_pause(self) -> None:
    def media_stop(self) -> None:
    def media_previous_track(self) -> None:
    def media_next_track(self) -> None:
    def _call_playback_function(self, function, function_text) -> None:
    def select_source(self, source: str) -> None:
    def play_media(self, media_type: str, media_id: str, **kwargs) -> None:
    def enable_output(self, port: str, enabled: bool) -> None:
    def menu_cursor(self, cursor: str) -> None:
    def set_scene(self, scene: str) -> None:
    def select_sound_mode(self, sound_mode: str) -> None:
    @property
    def media_artist(self) -> str:
    @property
    def media_album_name(self) -> str:
    @property
    def media_content_type(self) -> str:
    @property
    def media_title(self) -> str:
