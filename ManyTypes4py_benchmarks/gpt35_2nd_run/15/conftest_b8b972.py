from homeassistant.core import HomeAssistant
from homeassistant.components.calendar import CalendarEntity, CalendarEvent
from homeassistant.config_entries import ConfigEntry, ConfigFlow
from homeassistant.const import Platform
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from typing import Any
from unittest.mock import AsyncMock
import pytest
from tests.common import MockConfigEntry, MockModule, MockPlatform, mock_config_flow, mock_integration, mock_platform

async def set_time_zone(hass: HomeAssistant) -> None:

class MockFlow(ConfigFlow):

class MockCalendarEntity(CalendarEntity):

    def __init__(self, name: str, events: list = None) -> None:

    @property
    def event(self) -> CalendarEvent:

    def create_event(self, start: datetime, end: datetime, summary: str = None, description: str = None, location: str = None) -> dict:

    async def async_get_events(self, hass: HomeAssistant, start_date: datetime, end_date: datetime) -> list:

def config_flow_fixture(hass: HomeAssistant) -> None:

async def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:

def mock_setup_integration(hass: HomeAssistant, config_flow_fixture: Any, test_entities: list) -> None:

async def create_test_entities() -> list:
