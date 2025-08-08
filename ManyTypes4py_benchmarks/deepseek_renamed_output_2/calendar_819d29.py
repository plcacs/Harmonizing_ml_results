"""Support for Todoist task management (https://todoist.com)."""
from __future__ import annotations
from datetime import date, datetime, timedelta
import logging
from typing import Any, Optional, List, Dict, Union, Callable, cast
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

_LOGGER = logging.getLogger(__name__)

NEW_TASK_SERVICE_SCHEMA = vol.Schema({
    vol.Required(CONTENT): cv.string,
    vol.Optional(DESCRIPTION): cv.string,
    vol.Optional(PROJECT_NAME, default='inbox'): vol.All(cv.string, vol.Lower),
    vol.Optional(SECTION_NAME): vol.All(cv.string, vol.Lower),
    vol.Optional(LABELS): cv.ensure_list_csv,
    vol.Optional(ASSIGNEE): cv.string,
    vol.Optional(PRIORITY): vol.All(vol.Coerce(int), vol.Range(min=1, max=4)),
    vol.Exclusive(DUE_DATE_STRING, 'due_date'): cv.string,
    vol.Optional(DUE_DATE_LANG): vol.All(cv.string, vol.In(DUE_DATE_VALID_LANGS)),
    vol.Exclusive(DUE_DATE, 'due_date'): cv.string,
    vol.Exclusive(REMINDER_DATE_STRING, 'reminder_date'): cv.string,
    vol.Optional(REMINDER_DATE_LANG): vol.All(cv.string, vol.In(DUE_DATE_VALID_LANGS)),
    vol.Exclusive(REMINDER_DATE, 'reminder_date'): cv.string
})

PLATFORM_SCHEMA = CALENDAR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_TOKEN): cv.string,
    vol.Optional(CONF_EXTRA_PROJECTS, default=[]): vol.All(
        cv.ensure_list,
        vol.Schema([
            vol.Schema({
                vol.Required(CONF_NAME): cv.string,
                vol.Optional(CONF_PROJECT_DUE_DATE): vol.Coerce(int),
                vol.Optional(CONF_PROJECT_WHITELIST, default=[]): vol.All(
                    cv.ensure_list,
                    [vol.All(cv.string, vol.Lower)]
                ),
                vol.Optional(CONF_PROJECT_LABEL_WHITELIST, default=[]): vol.All(
                    cv.ensure_list,
                    [vol.All(cv.string)]
                )
            })
        ])
    )
})

SCAN_INTERVAL = timedelta(minutes=1)

async def func_kapmitfq(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the Todoist calendar platform config entry."""
    coordinator: TodoistCoordinator = hass.data[DOMAIN][entry.entry_id]
    projects: List[ProjectData] = await coordinator.async_get_projects()
    labels: List[Label] = await coordinator.async_get_labels()
    entities: List[TodoistProjectEntity] = []
    for project in projects:
        project_data: Dict[str, Any] = {CONF_NAME: project.name, CONF_ID: project.id}
        entities.append(TodoistProjectEntity(coordinator, project_data, labels))
    async_add_entities(entities)
    func_khxg1s5u(hass, coordinator)

async def func_fupznv9k(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Todoist platform."""
    token: str = config[CONF_TOKEN]
    project_id_lookup: Dict[str, str] = {}
    api: TodoistAPIAsync = TodoistAPIAsync(token)
    coordinator: TodoistCoordinator = TodoistCoordinator(hass, _LOGGER, None, SCAN_INTERVAL, api, token)
    await coordinator.async_refresh()

    async def func_bw423dl8(_: Event) -> None:
        await coordinator.async_shutdown()
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, func_bw423dl8)
    projects: List[ProjectData] = await api.get_projects()
    labels: List[Label] = await api.get_labels()
    project_devices: List[TodoistProjectEntity] = []
    for project in projects:
        project_data: Dict[str, Any] = {CONF_NAME: project.name, CONF_ID: project.id}
        project_devices.append(TodoistProjectEntity(coordinator, project_data, labels))
        project_id_lookup[project.name.lower()] = project.id
    extra_projects: List[CustomProject] = config[CONF_EXTRA_PROJECTS]
    for extra_project in extra_projects:
        project_due_date: Optional[int] = extra_project.get(CONF_PROJECT_DUE_DATE)
        project_label_filter: List[str] = extra_project[CONF_PROJECT_LABEL_WHITELIST]
        project_name_filter: List[str] = extra_project[CONF_PROJECT_WHITELIST]
        project_id_filter: Optional[List[str]] = None
        if project_name_filter is not None:
            project_id_filter = [project_id_lookup[project_name.lower()] for project_name in project_name_filter]
        project_devices.append(
            TodoistProjectEntity(
                coordinator,
                {'id': None, 'name': extra_project['name']},
                labels,
                due_date_days=project_due_date,
                whitelisted_labels=project_label_filter,
                whitelisted_projects=project_id_filter
            )
        )
    async_add_entities(project_devices, update_before_add=True)
    func_khxg1s5u(hass, coordinator)

def func_khxg1s5u(hass: HomeAssistant, coordinator: TodoistCoordinator) -> None:
    """Register services."""
    if hass.services.has_service(DOMAIN, SERVICE_NEW_TASK):
        return
    session = async_get_clientsession(hass)

    async def func_qr93r1c8(call: ServiceCall) -> None:
        """Call when a user creates a new Todoist Task from Home Assistant."""
        project_name: str = call.data[PROJECT_NAME]
        projects: List[ProjectData] = await coordinator.async_get_projects()
        project_id: Optional[str] = None
        for project in projects:
            if project_name == project.name.lower():
                project_id = project.id
                break
        if project_id is None:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key='project_invalid',
                translation_placeholders={'project': project_name}
            )
        section_id: Optional[str] = None
        if SECTION_NAME in call.data:
            section_name: str = call.data[SECTION_NAME]
            sections: List[Any] = await coordinator.async_get_sections(project_id)
            for section in sections:
                if section_name == section.name.lower():
                    section_id = section.id
                    break
            if section_id is None:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key='section_invalid',
                    translation_placeholders={'section': section_name, 'project': project_name}
                )
        content: str = call.data[CONTENT]
        data: Dict[str, Any] = {'project_id': project_id}
        if (description := call.data.get(DESCRIPTION)):
            data['description'] = description
        if section_id is not None:
            data['section_id'] = section_id
        if (task_labels := call.data.get(LABELS)):
            data['labels'] = task_labels
        if ASSIGNEE in call.data:
            collaborators: List[Any] = await coordinator.api.get_collaborators(project_id)
            collaborator_id_lookup: Dict[str, str] = {collab.name.lower(): collab.id for collab in collaborators}
            task_assignee: str = call.data[ASSIGNEE].lower()
            if task_assignee in collaborator_id_lookup:
                data['assignee_id'] = collaborator_id_lookup[task_assignee]
            else:
                raise ValueError(f'User is not part of the shared project. user: {task_assignee}')
        if PRIORITY in call.data:
            data['priority'] = call.data[PRIORITY]
        if DUE_DATE_STRING in call.data:
            data['due_string'] = call.data[DUE_DATE_STRING]
        if DUE_DATE_LANG in call.data:
            data['due_lang'] = call.data[DUE_DATE_LANG]
        if DUE_DATE in call.data:
            due_date: Optional[datetime] = dt_util.parse_datetime(call.data[DUE_DATE])
            if due_date is None:
                due: Optional[date] = dt_util.parse_date(call.data[DUE_DATE])
                if due is None:
                    raise ValueError(f'Invalid due_date: {call.data[DUE_DATE]}')
                due_date = datetime(due.year, due.month, due.day)
            due_date = dt_util.as_utc(due_date)
            date_format: str = '%Y-%m-%dT%H:%M:%S'
            data['due_datetime'] = datetime.strftime(due_date, date_format)
        api_task: Task = await coordinator.api.add_task(content, **data)
        sync_url: str = get_sync_url('sync')
        _reminder_due: Dict[str, Any] = {}
        if REMINDER_DATE_STRING in call.data:
            _reminder_due['string'] = call.data[REMINDER_DATE_STRING]
        if REMINDER_DATE_LANG in call.data:
            _reminder_due['lang'] = call.data[REMINDER_DATE_LANG]
        if REMINDER_DATE in call.data:
            due_date = dt_util.parse_datetime(call.data[REMINDER_DATE])
            if due_date is None:
                due = dt_util.parse_date(call.data[REMINDER_DATE])
                if due is None:
                    raise ValueError(f'Invalid reminder_date: {call.data[REMINDER_DATE]}')
                due_date = datetime(due.year, due.month, due.day)
            due_date = dt_util.as_utc(due_date)
            date_format = '%Y-%m-%dT%H:%M:%S'
            _reminder_due['date'] = datetime.strftime(due_date, date_format)

        async def func_r3b6wppf(reminder_due: Dict[str, Any]) -> Any:
            reminder_data: Dict[str, Any] = {
                'commands': [{
                    'type': 'reminder_add',
                    'temp_id': str(uuid.uuid1()),
                    'uuid': str(uuid.uuid1()),
                    'args': {
                        'item_id': api_task.id,
                        'type': 'absolute',
                        'due': reminder_due
                    }
                }]
            }
            headers: Dict[str, str] = create_headers(token=coordinator.token, with_content=True)
            return await session.post(sync_url, headers=headers, json=reminder_data)

        if _reminder_due:
            await func_r3b6wppf(_reminder_due)
        _LOGGER.debug('Created Todoist task: %s', call.data[CONTENT])

    hass.services.async_register(DOMAIN, SERVICE_NEW_TASK, func_qr93r1c8, schema=NEW_TASK_SERVICE_SCHEMA)

class TodoistProjectEntity(CoordinatorEntity[TodoistCoordinator], CalendarEntity):
    """A device for getting the next Task from a Todoist Project."""

    def __init__(
        self,
        coordinator: TodoistCoordinator,
        data: Dict[str, Any],
        labels: List[Label],
        due_date_days: Optional[int] = None,
        whitelisted_labels: Optional[List[str]] = None,
        whitelisted_projects: Optional[List[str]] = None
    ) -> None:
        """Create the Todoist Calendar Entity."""
        super().__init__(coordinator=coordinator)
        self.data: TodoistProjectData = TodoistProjectData(
            data,
            labels,
            coordinator,
            due_date_days=due_date_days,
            whitelisted_labels=whitelisted_labels,
            whitelisted_projects=whitelisted_projects
        )
        self._cal_data: Dict[str, Any] = {}
        self._name: str = data[CONF_NAME]
        self._attr_unique_id: Optional[str] = str(data[CONF_ID]) if data.get(CONF_ID) is not None else None

    @callback
    def func_knd5xkps(self) -> None:
        """Handle updated data from the coordinator."""
        self.data.update()
        super()._handle_coordinator_update()

    @property
    def func_00o3s5pu(self) -> Optional[CalendarEvent]:
        """Return the next upcoming event."""
        return self.data.calendar_event

    @property
    def func_m9qyntae(self) -> str:
        """Return the name of the entity."""
        return self._name

    async def func_7vi05cp6(self) -> None:
        """Update all Todoist Calendars."""
        await super().async_update()
        self.data.update()

    async def func_v2jaq83e(self, hass: HomeAssistant, start_date: datetime, end_date: datetime) -> List[CalendarEvent]:
        """Get all events in a specific time frame."""
        return await self.data.async_get_events(start_date, end_date)

    @property
    def func_lf3c0kzd(self) -> Optional[Dict[str, Any]]:
        """Return the device state attributes."""
        if self.data.event is None:
            return None
        return {
            DUE_TODAY: self.data.event[DUE_TODAY],
            OVERDUE: self.data.event[OVERDUE],
            ALL_TASKS: [task[SUMMARY] for task in self.data.all_project_tasks],
            PRIORITY: self.data.event[PRIORITY],
            LABELS: self.data.event[LABELS]
        }

class TodoistProjectData:
    """Class used by the Task Entity service object to hold all Todoist Tasks."""

    def __init__(
        self,
        project_data: Dict[str, Any],
        labels: List[Label],
        coordinator: TodoistCoordinator,
        due_date_days: Optional[int] = None,
        whitelisted_labels: Optional[List[str]] = None,
        whitelisted_projects: Optional[List[str]] = None
    ) -> None:
        """Initialize a Todoist Project."""
        self.event: Optional[Dict[str, Any]] = None
        self._coordinator: TodoistCoordinator = coordinator
        self._name: str = project_data[CONF_NAME]
        self._id: Optional[str] = project_data.get(CONF_ID)
        self._labels: List[Label] = labels
        self.all_project_tasks: List[Dict[str, Any]] = []
        self._due_date_days: Optional[timedelta] = None
        if due_date_days is not None:
            self._due_date_days = timedelta(days=due_date_days)
        self._label_whitelist: List[str] = []
        if whitelisted_labels is not None:
            self._label_whitelist = whitelisted_labels
        self._project_id_whitelist: List[str] = []
        if whitelisted_projects is not None:
            self._project_id_whitelist = whitelisted_projects

    @property
    def calendar_event(self) -> Optional[CalendarEvent]:
        """Return the next upcoming calendar event."""
        if not self.event:
            return None
        start: datetime = self.event[START]
        if self.event.get(ALL_DAY) or self.event[END] is None:
            return CalendarEvent(
                summary=self.event[SUMMARY],
                start=start.date(),
                end=start.date() + timedelta(days=1)
            )
        return CalendarEvent(
            summary=self.event[SUMMARY],
            start=start,
            end=self.event[END]
        )

    def create_todoist_task(self, data: Task) -> Optional[Dict[str, Any]]:
        """Create a dictionary based on a Task passed from the Todoist API."""
        task: Dict[str, Any] = {
            ALL_DAY: False,
            COMPLETED: data.is_completed,
            DESCRIPTION: f'https://todoist.com/showTask?id={data.id}',
            DUE_TODAY: False,
            END: None,
            LABELS: [],
            OVERDUE: False,
            PRIORITY: data.priority,
            START: dt_util.now(),
           