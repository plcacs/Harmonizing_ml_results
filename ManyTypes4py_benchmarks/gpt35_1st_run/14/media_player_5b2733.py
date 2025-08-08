    def _request(self, method: str, path: str, params: dict = None) -> dict:
    def _command(self, named_command: str) -> dict:
    def now_playing(self) -> dict:
    def set_volume(self, level: int) -> dict:
    def set_muted(self, muted: bool) -> dict:
    def set_shuffle(self, shuffle: bool) -> dict:
    def play(self) -> dict:
    def pause(self) -> dict:
    def next(self) -> dict:
    def previous(self) -> dict:
    def stop(self) -> dict:
    def play_playlist(self, playlist_id_or_name: str) -> dict:
    def artwork_url(self) -> str:
    def airplay_devices(self) -> dict:
    def airplay_device(self, device_id: str) -> dict:
    def toggle_airplay_device(self, device_id: str, toggle: bool) -> dict:
    def set_volume_airplay_device(self, device_id: str, level: int) -> dict:

    def __init__(self, host: str, port: int, use_ssl: bool):
    def _base_url(self) -> str:
    def update_state(self, state_hash: dict):
    def set_volume_level(self, volume: float):
    def mute_volume(self, mute: bool):
    def set_shuffle(self, shuffle: bool):
    def play_media(self, media_type: str, media_id: str, **kwargs):
    def set_volume_level(self, volume: float):
    def turn_on(self):
    def turn_off(self):

    def __init__(self, name: str, host: str, port: int, use_ssl: bool, add_entities: AddEntitiesCallback):
    def update(self):
    def update_state(self, state_hash: dict):
    def set_volume_level(self, volume: float):
    def turn_on(self):
    def turn_off(self):
