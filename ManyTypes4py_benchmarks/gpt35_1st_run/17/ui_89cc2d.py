from typing import List, Tuple

class Ui:
    def __init__(self) -> None:
    def addstr(self, *args: Tuple) -> None:
    def update_margin(self) -> None:
    def build_playinfo(self, song_name: str, artist: str, album_name: str, quality: str, start: int, pause: bool = False) -> None:
    def update_lyrics(self, now_playing: int, lyrics: List[str], tlyrics: List[str]) -> None:
    def build_process_bar(self, song: dict, now_playing: int, total_length: int, playing_flag: bool, playing_mode: int) -> None:
    def build_loading(self) -> None:
    def build_submenu(self, data: dict) -> None:
    def build_menu(self, datatype: str, title: str, datalist: List[dict], offset: int, index: int, step: int, start: int) -> None:
    def build_login(self) -> Tuple[str, str]:
    def build_login_bar(self) -> None:
    def build_login_error(self) -> int:
    def build_search_error(self) -> int:
    def build_timing(self) -> str:
    def get_account(self) -> str:
    def get_password(self) -> str:
    def get_param(self, prompt_string: str) -> str:
    def update_size(self) -> None:
    def update_space(self) -> None:
