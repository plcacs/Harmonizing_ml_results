from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.components.calendar import CalendarEntity, CalendarEvent
from homeassistant.config_entries import ConfigEntry, ConfigFlow
from homeassistant.const import Platform
from typing import Any
from unittest.mock import AsyncMock
import pytest
from tests.common import MockConfigEntry, MockModule, MockPlatform, mock_config_flow, mock_integration, mock_platform
from datetime import timedelta
from homeassistant.util import dt as dt_util

async def set_time_zone(hass: HomeAssistant):
    ...

class MockFlow(ConfigFlow):
    ...

class MockCalendarEntity(CalendarEntity):
    def __init__(self, name: str, events: list = None):
        ...

    @property
    def event(self) -> CalendarEvent:
        ...

    def create_event(self, start: datetime, end: datetime, summary: str = None, description: str = None, location: str = None) -> dict:
        ...

    async def async_get_events(self, hass: HomeAssistant, start_date: datetime, end_date: datetime) -> list[CalendarEvent]:
        ...

def config_flow_fixture(hass: HomeAssistant):
    ...

async def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    ...

def mock_setup_integration(hass: HomeAssistant, config_flow_fixture, test_entities):
    ...

def create_test_entities() -> list[MockCalendarEntity]:
    ...
