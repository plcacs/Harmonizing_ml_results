"""Test fixtures for the cast integration."""

from typing import Any, AsyncMock as TypedAsyncMock, List, MagicMock as TypedMagicMock
from unittest.mock import AsyncMock, MagicMock, patch

import pychromecast
from pychromecast.controllers import multizone
import pytest


@pytest.fixture
def get_multizone_status_mock() -> MagicMock:
    """Mock pychromecast dial."""
    mock = MagicMock(spec_set=pychromecast.dial.get_multizone_status)
    mock.return_value.dynamic_groups = []
    return mock


@pytest.fixture
def get_cast_type_mock() -> MagicMock:
    """Mock pychromecast dial."""
    return MagicMock(spec_set=pychromecast.dial.get_cast_type)


@pytest.fixture
def castbrowser_mock() -> MagicMock:
    """Mock pychromecast CastBrowser."""
    return MagicMock(spec=pychromecast.discovery.CastBrowser)


@pytest.fixture
def mz_mock() -> MagicMock:
    """Mock pychromecast MultizoneManager."""
    return MagicMock(spec_set=multizone.MultizoneManager)


@pytest.fixture
def quick_play_mock() -> MagicMock:
    """Mock pychromecast quick_play."""
    return MagicMock()


@pytest.fixture
def get_chromecast_mock() -> MagicMock:
    """Mock pychromecast get_chromecast_from_cast_info."""
    return MagicMock()


@pytest.fixture
def ha_controller_mock() -> MagicMock:
    """Mock HomeAssistantController."""
    with patch(
        "homeassistant.components.cast.media_player.HomeAssistantController",
        MagicMock(),
    ) as ha_controller_mock:
        yield ha_controller_mock


@pytest.fixture(autouse=True)
def cast_mock(
    mz_mock: MagicMock,
    quick_play_mock: MagicMock,
    castbrowser_mock: MagicMock,
    get_cast_type_mock: MagicMock,
    get_chromecast_mock: MagicMock,
    get_multizone_status_mock: MagicMock,
) -> None:
    """Mock pychromecast."""
    ignore_cec_orig: List[str] = list(pychromecast.IGNORE_CEC)

    with (
        patch(
            "homeassistant.components.cast.discovery.pychromecast.discovery.CastBrowser",
            castbrowser_mock,
        ),
        patch(
            "homeassistant.components.cast.helpers.dial.get_cast_type",
            get_cast_type_mock,
        ),
        patch(
            "homeassistant.components.cast.helpers.dial.get_multizone_status",
            get_multizone_status_mock,
        ),
        patch(
            "homeassistant.components.cast.media_player.MultizoneManager",
            return_value=mz_mock,
        ),
        patch(
            "homeassistant.components.cast.media_player.zeroconf.async_get_instance",
            AsyncMock(),
        ),
        patch(
            "homeassistant.components.cast.media_player.quick_play",
            quick_play_mock,
        ),
        patch(
            "homeassistant.components.cast.media_player.pychromecast.get_chromecast_from_cast_info",
            get_chromecast_mock,
        ),
    ):
        yield

    pychromecast.IGNORE_CEC = list(ignore_cec_orig)
