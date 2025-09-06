from __future__ import annotations
from datetime import date, datetime, timedelta
import logging
from typing import Any, List, Dict, Optional
import uuid
from todoist_api_python.api_async import TodoistAPIAsync
from todoist_api_python.endpoints import get_sync_url
from todoist_api_python.headers import create_headers
from todoist_api_python.models import Due, Label, Task
import voluptuous as vol
from homeassistant.components.calendar import PLATFORM_SCHEMA as CALENDAR_PLATFORM_SCHEMA, CalendarEntity, CalendarEvent
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ID, CONF_NAME, CONF_TOKEN, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import Event, HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util
from .const import ALL_DAY, ALL_TASKS, ASSIGNEE, COMPLETED, CONF_EXTRA_PROJECTS, CONF_PROJECT_DUE_DATE, CONF_PROJECT_LABEL_WHITELIST, CONF_PROJECT_WHITELIST, CONTENT, DESCRIPTION, DOMAIN, DUE_DATE, DUE_DATE_LANG, DUE_DATE_STRING, DUE_DATE_VALID_LANGS, DUE_TODAY, END, LABELS, OVERDUE, PRIORITY, PROJECT_NAME, REMINDER_DATE, REMINDER_DATE_LANG, REMINDER_DATE_STRING, SECTION_NAME, SERVICE_NEW_TASK, START, SUMMARY
from .coordinator import TodoistCoordinator
from .types import CalData, CustomProject, ProjectData, TodoistEvent

_LOGGER: logging.Logger

NEW_TASK_SERVICE_SCHEMA: vol.Schema
PLATFORM_SCHEMA: vol.Schema
SCAN_INTERVAL: timedelta

async def func_kapmitfq(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    ...

async def func_fupznv9k(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

def func_khxg1s5u(hass: HomeAssistant, coordinator: TodoistCoordinator) -> None:
    ...

class TodoistProjectEntity(CoordinatorEntity[TodoistCoordinator], CalendarEntity):
    ...

class TodoistProjectData:
    ...

def func_mdh32ofi(due: Due) -> Optional[Union[date, datetime]]:
    ...
