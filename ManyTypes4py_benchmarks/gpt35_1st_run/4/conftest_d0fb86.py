from unittest.mock import AsyncMock, MagicMock, patch
import pychromecast
from pychromecast.controllers import multizone
import pytest

def get_multizone_status_mock() -> MagicMock:
    mock: MagicMock = MagicMock(spec_set=pychromecast.dial.get_multizone_status)
    mock.return_value.dynamic_groups = []
    return mock

def get_cast_type_mock() -> MagicMock:
    return MagicMock(spec_set=pychromecast.dial.get_cast_type)

def castbrowser_mock() -> MagicMock:
    return MagicMock(spec=pychromecast.discovery.CastBrowser)

def mz_mock() -> MagicMock:
    return MagicMock(spec_set=multizone.MultizoneManager)

def quick_play_mock() -> MagicMock:
    return MagicMock()

def get_chromecast_mock() -> MagicMock:
    return MagicMock()

def ha_controller_mock() -> MagicMock:
    with patch('homeassistant.components.cast.media_player.HomeAssistantController', MagicMock()) as ha_controller_mock:
        yield ha_controller_mock

def cast_mock(mz_mock: MagicMock, quick_play_mock: MagicMock, castbrowser_mock: MagicMock, get_cast_type_mock: MagicMock, get_chromecast_mock: MagicMock, get_multizone_status_mock: MagicMock):
    ignore_cec_orig = list(pychromecast.IGNORE_CEC)
    with patch('homeassistant.components.cast.discovery.pychromecast.discovery.CastBrowser', castbrowser_mock), patch('homeassistant.components.cast.helpers.dial.get_cast_type', get_cast_type_mock), patch('homeassistant.components.cast.helpers.dial.get_multizone_status', get_multizone_status_mock), patch('homeassistant.components.cast.media_player.MultizoneManager', return_value=mz_mock), patch('homeassistant.components.cast.media_player.zeroconf.async_get_instance', AsyncMock()), patch('homeassistant.components.cast.media_player.quick_play', quick_play_mock), patch('homeassistant.components.cast.media_player.pychromecast.get_chromecast_from_cast_info', get_chromecast_mock):
        yield
    pychromecast.IGNORE_CEC = list(ignore_cec_orig)
