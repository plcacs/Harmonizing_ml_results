"""The tests for lastfm."""
from typing import Any
from unittest.mock import patch
from pylast import PyLastError, Track
from homeassistant.components.lastfm.const import CONF_MAIN_USER, CONF_USERS
from homeassistant.const import CONF_API_KEY
from homeassistant.helpers.typing import UNDEFINED, UndefinedType
API_KEY = 'asdasdasdasdasd'
USERNAME_1 = 'testaccount1'
USERNAME_2 = 'testaccount2'
CONF_DATA = {CONF_API_KEY: API_KEY, CONF_MAIN_USER: USERNAME_1, CONF_USERS: [USERNAME_1, USERNAME_2]}
CONF_USER_DATA = {CONF_API_KEY: API_KEY, CONF_MAIN_USER: USERNAME_1}
CONF_FRIENDS_DATA = {CONF_USERS: [USERNAME_2]}

class MockNetwork:
    """Mock _Network object for pylast."""

    def __init__(self, username):
        """Initialize the mock."""
        self.username = username

class MockTopTrack:
    """Mock TopTrack object for pylast."""

    def __init__(self, item):
        """Initialize the mock."""
        self.item = item

class MockLastTrack:
    """Mock LastTrack object for pylast."""

    def __init__(self, track):
        """Initialize the mock."""
        self.track = track

class MockUser:
    """Mock User object for pylast."""

    def __init__(self, username=USERNAME_1, now_playing_result=None, thrown_error=None, friends=UNDEFINED, recent_tracks=UNDEFINED, top_tracks=UNDEFINED):
        """Initialize the mock."""
        self._now_playing_result = now_playing_result
        self._thrown_error = thrown_error
        self._friends = [] if friends is UNDEFINED else friends
        self._recent_tracks = [] if recent_tracks is UNDEFINED else recent_tracks
        self._top_tracks = [] if top_tracks is UNDEFINED else top_tracks
        self.name = username

    def get_name(self, capitalized):
        """Get name of the user."""
        return self.name

    def get_playcount(self):
        """Get mock play count."""
        if self._thrown_error:
            raise self._thrown_error
        return len(self._recent_tracks)

    def get_image(self):
        """Get mock image."""
        return 'image'

    def get_recent_tracks(self, limit):
        """Get mock recent tracks."""
        return [MockLastTrack(track) for track in self._recent_tracks]

    def get_top_tracks(self, limit):
        """Get mock top tracks."""
        return [MockTopTrack(track) for track in self._recent_tracks]

    def get_now_playing(self):
        """Get mock now playing."""
        return self._now_playing_result

    def get_friends(self):
        """Get mock friends."""
        if len(self._friends) == 0:
            raise PyLastError('network', 'status', 'Page not found')
        return self._friends

def patch_user(user):
    """Patch interface."""
    return patch('pylast.User', return_value=user)

def patch_setup_entry():
    """Patch interface."""
    return patch('homeassistant.components.lastfm.async_setup_entry', return_value=True)