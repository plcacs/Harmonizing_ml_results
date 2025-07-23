"""The tests for lastfm."""
from typing import Any, Dict, List, Optional, Union
from unittest.mock import patch, MagicMock
from pylast import PyLastError, Track
from homeassistant.components.lastfm.const import CONF_MAIN_USER, CONF_USERS
from homeassistant.const import CONF_API_KEY
from homeassistant.helpers.typing import UNDEFINED, UndefinedType

API_KEY: str = 'asdasdasdasdasd'
USERNAME_1: str = 'testaccount1'
USERNAME_2: str = 'testaccount2'
CONF_DATA: Dict[str, Any] = {
    CONF_API_KEY: API_KEY,
    CONF_MAIN_USER: USERNAME_1,
    CONF_USERS: [USERNAME_1, USERNAME_2],
}
CONF_USER_DATA: Dict[str, Any] = {
    CONF_API_KEY: API_KEY,
    CONF_MAIN_USER: USERNAME_1,
}
CONF_FRIENDS_DATA: Dict[str, Any] = {
    CONF_USERS: [USERNAME_2],
}

class MockNetwork:
    """Mock _Network object for pylast."""

    username: str

    def __init__(self, username: str) -> None:
        """Initialize the mock."""
        self.username = username

class MockTopTrack:
    """Mock TopTrack object for pylast."""

    item: Any

    def __init__(self, item: Any) -> None:
        """Initialize the mock."""
        self.item = item

class MockLastTrack:
    """Mock LastTrack object for pylast."""

    track: Any

    def __init__(self, track: Any) -> None:
        """Initialize the mock."""
        self.track = track

class MockUser:
    """Mock User object for pylast."""

    _now_playing_result: Optional[Any]
    _thrown_error: Optional[Exception]
    _friends: List[Any]
    _recent_tracks: List[Any]
    _top_tracks: List[Any]
    name: str

    def __init__(
        self,
        username: str = USERNAME_1,
        now_playing_result: Optional[Any] = None,
        thrown_error: Optional[Exception] = None,
        friends: Union[UndefinedType, List[Any]] = UNDEFINED,
        recent_tracks: Union[UndefinedType, List[Any]] = UNDEFINED,
        top_tracks: Union[UndefinedType, List[Any]] = UNDEFINED,
    ) -> None:
        """Initialize the mock."""
        self._now_playing_result = now_playing_result
        self._thrown_error = thrown_error
        self._friends = [] if friends is UNDEFINED else friends
        self._recent_tracks = [] if recent_tracks is UNDEFINED else recent_tracks
        self._top_tracks = [] if top_tracks is UNDEFINED else top_tracks
        self.name = username

    def get_name(self, capitalized: bool) -> str:
        """Get name of the user."""
        return self.name

    def get_playcount(self) -> int:
        """Get mock play count."""
        if self._thrown_error:
            raise self._thrown_error
        return len(self._recent_tracks)

    def get_image(self) -> str:
        """Get mock image."""
        return 'image'

    def get_recent_tracks(self, limit: int) -> List[MockLastTrack]:
        """Get mock recent tracks."""
        return [MockLastTrack(track) for track in self._recent_tracks]

    def get_top_tracks(self, limit: int) -> List[MockTopTrack]:
        """Get mock top tracks."""
        return [MockTopTrack(track) for track in self._recent_tracks]

    def get_now_playing(self) -> Optional[Any]:
        """Get mock now playing."""
        return self._now_playing_result

    def get_friends(self) -> List[Any]:
        """Get mock friends."""
        if len(self._friends) == 0:
            raise PyLastError('network', 'status', 'Page not found')
        return self._friends

def patch_user(user: MockUser) -> MagicMock:
    """Patch interface."""
    return patch('pylast.User', return_value=user)

def patch_setup_entry() -> MagicMock:
    """Patch interface."""
    return patch('homeassistant.components.lastfm.async_setup_entry', return_value=True)
