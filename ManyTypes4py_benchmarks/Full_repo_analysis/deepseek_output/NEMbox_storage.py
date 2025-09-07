import json
from typing import Any, Dict, List, TypedDict, Union
from .const import Constant
from .singleton import Singleton
from .utils import utf8_data_to_file

class UserInfo(TypedDict):
    username: str
    password: str
    user_id: str
    nickname: str

class CollectionInfo(TypedDict):
    collection_name: str
    collection_type: str
    collection_describe: str
    collection_songs: List[int]

class SongInfo(TypedDict):
    song_id: int
    artist: str
    song_name: str
    mp3_url: str
    album_name: str
    album_id: str
    quality: str
    lyric: str
    tlyric: str

class PlayerInfo(TypedDict):
    player_list: List[Dict[str, Any]]
    player_list_type: str
    player_list_title: str
    playing_order: List[int]
    playing_mode: int
    idx: int
    ridx: int
    playing_volume: int

class Database(TypedDict):
    user: UserInfo
    collections: List[CollectionInfo]
    songs: Dict[int, SongInfo]
    player_info: PlayerInfo

class Storage(Singleton):

    def __init__(self) -> None:
        if hasattr(self, '_init'):
            return
        self._init: bool = True
        self.database: Database = {
            'user': {'username': '', 'password': '', 'user_id': '', 'nickname': ''}, 
            'collections': [], 
            'songs': {}, 
            'player_info': {
                'player_list': [], 
                'player_list_type': '', 
                'player_list_title': '', 
                'playing_order': [], 
                'playing_mode': 0, 
                'idx': 0, 
                'ridx': 0, 
                'playing_volume': 60
            }
        }
        self.storage_path: str = Constant.storage_path
        self.cookie_path: str = Constant.cookie_path

    def login(self, username: str, password: str, userid: str, nickname: str) -> None:
        self.database['user'] = UserInfo(username=username, password=password, user_id=userid, nickname=nickname)

    def logout(self) -> None:
        self.database['user'] = UserInfo(username='', password='', user_id='', nickname='')

    def load(self) -> None:
        try:
            with open(self.storage_path, 'r') as f:
                loaded_data: Dict[str, Any] = json.load(f)
                for (k, v) in loaded_data.items():
                    if isinstance(self.database[k], dict):
                        self.database[k].update(v)
                    else:
                        self.database[k] = v
        except (OSError, KeyError, ValueError):
            pass
        self.save()

    def save(self) -> None:
        with open(self.storage_path, 'w') as f:
            data: str = json.dumps(self.database)
            utf8_data_to_file(f, data)
