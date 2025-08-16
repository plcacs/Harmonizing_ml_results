import curses
import datetime
import os
import re
from shutil import get_terminal_size
from . import logger
from .config import Config
from .scrollstring import scrollstring
from .scrollstring import truelen
from .scrollstring import truelen_cut
from .storage import Storage
from .utils import md5
from typing import List, Tuple, Union

class Ui(object):

    def __init__(self) -> None:
        self.screen = curses.initscr()
        self.x: int
        self.y: int
        self.playerX: int
        self.playerY: int
        self.lyric: str
        self.now_lyric: str
        self.post_lyric: str
        self.now_lyric_index: int
        self.now_tlyric_index: int
        self.tlyric: str
        self.storage: Storage
        self.newversion: bool

    def addstr(self, *args: Union[str, int]) -> None:
        pass

    def update_margin(self) -> None:
        pass

    def build_playinfo(self, song_name: str, artist: str, album_name: str, quality: str, start: int, pause: bool = False) -> None:
        pass

    def update_lyrics(self, now_playing: int, lyrics: List[str], tlyrics: List[str]) -> None:
        pass

    def build_process_bar(self, song: dict, now_playing: int, total_length: int, playing_flag: bool, playing_mode: int) -> None:
        pass

    def build_loading(self) -> None:
        pass

    def build_submenu(self, data: dict) -> None:
        pass

    def build_menu(self, datatype: str, title: str, datalist: List[dict], offset: int, index: int, step: int, start: int) -> None:
        pass

    def build_login(self) -> Tuple[str, str]:
        pass

    def build_login_bar(self) -> None:
        pass

    def build_login_error(self) -> int:
        pass

    def build_search_error(self) -> int:
        pass

    def build_timing(self) -> str:
        pass

    def get_account(self) -> str:
        pass

    def get_password(self) -> str:
        pass

    def get_param(self, prompt_string: str) -> str:
        pass

    def update_size(self) -> None:
        pass

    def update_space(self) -> None:
        pass
