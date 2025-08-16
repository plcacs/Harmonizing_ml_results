from typing import Any, List, Union
from unittest.mock import patch
from pylast import PyLastError, Track
from homeassistant.components.lastfm.const import CONF_MAIN_USER, CONF_USERS
from homeassistant.const import CONF_API_KEY
from homeassistant.helpers.typing import UNDEFINED, UndefinedType

API_KEY: str = 'asdasdasdasdasd'
USERNAME_1: str = 'testaccount1'
USERNAME_2: str = 'testaccount2'
CONF_DATA: dict = {CONF_API_KEY: API_KEY, CONF_MAIN_USER: USERNAME_1, CONF_USERS: [USERNAME_1, USERNAME_2]}
CONF_USER_DATA: dict = {CONF_API_KEY: API_KEY, CONF_MAIN_USER: USERNAME_1}
CONF_FRIENDS_DATA: dict = {CONF_USERS: [USERNAME_2]}

class MockNetwork:
    def __init__(self, username: str) -> None:
        self.username: str = username

class MockTopTrack:
    def __init__(self, item: Any) -> None:
        self.item: Any = item

class MockLastTrack:
    def __init__(self, track: Track) -> None:
        self.track: Track = track

class MockUser:
    def __init__(self, username: str = USERNAME_1, now_playing_result: Any = None, thrown_error: Exception = None, friends: Union[List[str], UndefinedType] = UNDEFINED, recent_tracks: Union[List[Track], UndefinedType] = UNDEFINED, top_tracks: Union[List[Track], UndefinedType] = UNDEFINED) -> None:
        self._now_playing_result: Any = now_playing_result
        self._thrown_error: Exception = thrown_error
        self._friends: List[str] = [] if friends is UNDEFINED else friends
        self._recent_tracks: List[Track] = [] if recent_tracks is UNDEFINED else recent_tracks
        self._top_tracks: List[Track] = [] if top_tracks is UNDEFINED else top_tracks
        self.name: str = username

    def get_name(self, capitalized: bool) -> str:
        return self.name

    def get_playcount(self) -> int:
        if self._thrown_error:
            raise self._thrown_error
        return len(self._recent_tracks)

    def get_image(self) -> str:
        return 'image'

    def get_recent_tracks(self, limit: int) -> List[MockLastTrack]:
        return [MockLastTrack(track) for track in self._recent_tracks]

    def get_top_tracks(self, limit: int) -> List[MockTopTrack]:
        return [MockTopTrack(track) for track in self._recent_tracks]

    def get_now_playing(self) -> Any:
        return self._now_playing_result

    def get_friends(self) -> List[str]:
        if len(self._friends) == 0:
            raise PyLastError('network', 'status', 'Page not found')
        return self._friends

def patch_user(user: MockUser) -> patch:
    return patch('pylast.User', return_value=user)

def patch_setup_entry() -> patch:
    return patch('homeassistant.components.lastfm.async_setup_entry', return_value=True)
