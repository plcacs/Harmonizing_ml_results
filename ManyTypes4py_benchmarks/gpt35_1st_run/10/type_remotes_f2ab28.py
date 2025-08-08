from abc import ABC, abstractmethod
import logging
from typing import Any
from pyhap.const import CATEGORY_TELEVISION
from homeassistant.components.remote import ATTR_ACTIVITY, ATTR_ACTIVITY_LIST, ATTR_CURRENT_ACTIVITY, DOMAIN as REMOTE_DOMAIN, RemoteEntityFeature
from homeassistant.const import ATTR_ENTITY_ID, ATTR_SUPPORTED_FEATURES, SERVICE_TURN_OFF, SERVICE_TURN_ON, STATE_ON
from homeassistant.core import State, callback
from .accessories import TYPES, HomeAccessory
from .const import ATTR_KEY_NAME, CHAR_ACTIVE, CHAR_ACTIVE_IDENTIFIER, CHAR_CONFIGURED_NAME, CHAR_CURRENT_VISIBILITY_STATE, CHAR_IDENTIFIER, CHAR_INPUT_SOURCE_TYPE, CHAR_IS_CONFIGURED, CHAR_NAME, CHAR_REMOTE_KEY, CHAR_SLEEP_DISCOVER_MODE, EVENT_HOMEKIT_TV_REMOTE_KEY_PRESSED, KEY_ARROW_DOWN, KEY_ARROW_LEFT, KEY_ARROW_RIGHT, KEY_ARROW_UP, KEY_BACK, KEY_EXIT, KEY_FAST_FORWARD, KEY_INFORMATION, KEY_NEXT_TRACK, KEY_PLAY_PAUSE, KEY_PREVIOUS_TRACK, KEY_REWIND, KEY_SELECT, SERV_INPUT_SOURCE, SERV_TELEVISION
from .util import cleanup_name_for_homekit
MAXIMUM_SOURCES = 90
_LOGGER = logging.getLogger(__name__)
REMOTE_KEYS = {0: KEY_REWIND, 1: KEY_FAST_FORWARD, 2: KEY_NEXT_TRACK, 3: KEY_PREVIOUS_TRACK, 4: KEY_ARROW_UP, 5: KEY_ARROW_DOWN, 6: KEY_ARROW_LEFT, 7: KEY_ARROW_RIGHT, 8: KEY_SELECT, 9: KEY_BACK, 10: KEY_EXIT, 11: KEY_PLAY_PAUSE, 15: KEY_INFORMATION}

class RemoteInputSelectAccessory(HomeAccessory, ABC):
    def __init__(self, required_feature: int, source_key: str, source_list_key: str, *args: Any, category: int = CATEGORY_TELEVISION, **kwargs: Any) -> None:
    def _get_mapped_sources(self, state: State) -> dict:
    def _get_ordered_source_list_from_state(self, state: State) -> list:
    def set_on_off(self, value: bool) -> None:
    def set_input_source(self, value: int) -> None:
    def set_remote_key(self, value: int) -> None:
    def _async_update_input_state(self, hk_state: int, new_state: State) -> None:

@TYPES.register('ActivityRemote')
class ActivityRemote(RemoteInputSelectAccessory):
    def __init__(self, *args: Any) -> None:
    def set_on_off(self, value: bool) -> None:
    def set_input_source(self, value: int) -> None:
    def set_remote_key(self, value: int) -> None:
    def async_update_state(self, new_state: State) -> None:
