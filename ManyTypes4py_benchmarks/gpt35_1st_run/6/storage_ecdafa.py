import json
from .const import Constant
from .singleton import Singleton
from .utils import utf8_data_to_file

class Storage(Singleton):

    def __init__(self) -> None:
        self._init: bool = False
        self.database: dict = {'user': {'username': '', 'password': '', 'user_id': '', 'nickname': ''}, 'collections': [], 'songs': {}, 'player_info': {'player_list': [], 'player_list_type': '', 'player_list_title': '', 'playing_order': [], 'playing_mode': 0, 'idx': 0, 'ridx': 0, 'playing_volume': 60}}
        self.storage_path: str = Constant.storage_path
        self.cookie_path: str = Constant.cookie_path

    def login(self, username: str, password: str, userid: str, nickname: str) -> None:
        self.database['user'] = dict(username=username, password=password, user_id=userid, nickname=nickname)

    def logout(self) -> None:
        self.database['user'] = {'username': '', 'password': '', 'user_id': '', 'nickname': ''}

    def load(self) -> None:
        try:
            with open(self.storage_path, 'r') as f:
                for k, v in json.load(f).items():
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
