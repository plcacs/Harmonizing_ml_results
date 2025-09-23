"""Support for Todoist task management (https://todoist.com)."""
from __future__ import annotations
from datetime import date, datetime, timedelta
import logging
from typing import Any, cast
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
_LOGGER: logging.Logger = logging.getLogger(__name__)
NEW_TASK_SERVICE_SCHEMA: vol.Schema = vol.Schema({vol.Required(CONTENT): cv.string, vol.Optional(DESCRIPTION): cv.string, vol.Optional(PROJECT_NAME, default='inbox'): vol.All(cv.string, vol.Lower), vol.Optional(SECTION_NAME): vol.All(cv.string, vol.Lower), vol.Optional(LABELS): cv.ensure_list_csv, vol.Optional(ASSIGNEE): cv.string, vol.Optional(PRIORITY): vol.All(vol.Coerce(int), vol.Range(min=1, max=4)), vol.Exclusive(DUE_DATE_STRING, 'due_date'): cv.string, vol.Optional(DUE_DATE_LANG): vol.All(cv.string, vol.In(DUE_DATE_VALID_LANGS)), vol.Exclusive(DUE_DATE, 'due_date'): cv.string, vol.Exclusive(REMINDER_DATE_STRING, 'reminder_date'): cv.string, vol.Optional(REMINDER_DATE_LANG): vol.All(cv.string, vol.In(DUE_DATE_VALID_LANGS)), vol.Exclusive(REMINDER_DATE, 'reminder_date'): cv.string})
PLATFORM_SCHEMA: vol.Schema = CALENDAR_PLATFORM_SCHEMA.extend({vol.Required(CONF_TOKEN): cv.string, vol.Optional(CONF_EXTRA_PROJECTS, default=[]): vol.All(cv.ensure_list, vol.Schema([vol.Schema({vol.Required(CONF_NAME): cv.string, vol.Optional(CONF_PROJECT_DUE_DATE): vol.Coerce(int), vol.Optional(CONF_PROJECT_WHITELIST, default=[]): vol.All(cv.ensure_list, [vol.All(cv.string, vol.Lower)]), vol.Optional(CONF_PROJECT_LABEL_WHITELIST, default=[]): vol.All(cv.ensure_list, [vol.All(cv.string)])})]))})
SCAN_INTERVAL: timedelta = timedelta(minutes=1)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the Todoist calendar platform config entry."""
    coordinator: TodoistCoordinator = hass.data[DOMAIN][entry.entry_id]
    projects: list[Any] = await coordinator.async_get_projects()
    labels: list[Label] = await coordinator.async_get_labels()
    entities: list[TodoistProjectEntity] = []
    for project in projects:
        project_data: ProjectData = {CONF_NAME: project.name, CONF_ID: project.id}
        entities.append(TodoistProjectEntity(coordinator, project_data, labels))
    async_add_entities(entities)
    async_register_services(hass, coordinator)

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType | None=None) -> None:
    """Set up the Todoist platform."""
    token: str = config[CONF_TOKEN]
    project_id_lookup: dict[str, str] = {}
    api: TodoistAPIAsync = TodoistAPIAsync(token)
    coordinator: TodoistCoordinator = TodoistCoordinator(hass, _LOGGER, None, SCAN_INTERVAL, api, token)
    await coordinator.async_refresh()

    async def _shutdown_coordinator(_: Event) -> None:
        await coordinator.async_shutdown()
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _shutdown_coordinator)
    projects: list[Any] = await api.get_projects()
    labels: list[Label] = await api.get_labels()
    project_devices: list[TodoistProjectEntity] = []
    for project in projects:
        project_data: ProjectData = {CONF_NAME: project.name, CONF_ID: project.id}
        project_devices.append(TodoistProjectEntity(coordinator, project_data, labels))
        project_id_lookup[project.name.lower()] = project.id
    extra_projects: list[CustomProject] = config[CONF_EXTRA_PROJECTS]
    for extra_project in extra_projects:
        project_due_date: int | None = extra_project.get(CONF_PROJECT_DUE_DATE)
        project_label_filter: list[str] = extra_project[CONF_PROJECT_LABEL_WHITELIST]
        project_name_filter: list[str] = extra_project[CONF_PROJECT_WHITELIST]
        project_id_filter: list[str] | None = None
        if project_name_filter is not None:
            project_id_filter = [project_id_lookup[project_name.lower()] for project_name in project_name_filter]
        project_devices.append(TodoistProjectEntity(coordinator, {'id': None, 'name': extra_project['name']}, labels, due_date_days=project_due_date, whitelisted_labels=project_label_filter, whitelisted_projects=project_id_filter))
    async_add_entities(project_devices, update_before_add=True)
    async_register_services(hass, coordinator)

def async_register_services(hass: HomeAssistant, coordinator: TodoistCoordinator) -> None:
    """Register services."""
    if hass.services.has_service(DOMAIN, SERVICE_NEW_TASK):
        return
    session = async_get_clientsession(hass)

    async def handle_new_task(call: ServiceCall) -> None:
        """Call when a user creates a new Todoist Task from Home Assistant."""
        project_name: str = call.data[PROJECT_NAME]
        projects: list[Any] = await coordinator.async_get_projects()
        project_id: str | None = None
        for project in projects:
            if project_name == project.name.lower():
                project_id = project.id
                break
        if project_id is None:
            raise ServiceValidationError(translation_domain=DOMAIN, translation_key='project_invalid', translation_placeholders={'project': project_name})
        section_id: str | None = None
        if SECTION_NAME in call.data:
            section_name: str = call.data[SECTION_NAME]
            sections: list[Any] = await coordinator.async_get_sections(project_id)
            for section in sections:
                if section_name == section.name.lower():
                    section_id = section.id
                    break
            if section_id is None:
                raise ServiceValidationError(translation_domain=DOMAIN, translation_key='section_invalid', translation_placeholders={'section': section_name, 'project': project_name})
        content: str = call.data[CONTENT]
        data: dict[str, Any] = {'project_id': project_id}
        if (description := call.data.get(DESCRIPTION)):
            data['description'] = description
        if section_id is not None:
            data['section_id'] = section_id
        if (task_labels := call.data.get(LABELS)):
            data['labels'] = task_labels
        if ASSIGNEE in call.data:
            collaborators: list[Any] = await coordinator.api.get_collaborators(project_id)
            collaborator_id_lookup: dict[str, str] = {collab.name.lower(): collab.id for collab in collaborators}
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
            due_date: datetime | None = dt_util.parse_datetime(call.data[DUE_DATE])
            if due_date is None:
                due: date | None = dt_util.parse_date(call.data[DUE_DATE])
                if due is None:
                    raise ValueError(f'Invalid due_date: {call.data[DUE_DATE]}')
                due_date = datetime(due.year, due.month, due.day)
            due_date = dt_util.as_utc(due_date)
            date_format: str = '%Y-%m-%dT%H:%M:%S'
            data['due_datetime'] = datetime.strftime(due_date, date_format)
        api_task: Task = await coordinator.api.add_task(content, **data)
        sync_url: str = get_sync_url('sync')
        _reminder_due: dict[str, str] = {}
        if REMINDER_DATE_STRING in call.data:
            _reminder_due['string'] = call.data[REMINDER_DATE_STRING]
        if REMINDER_DATE_LANG in call.data:
            _reminder_due['lang'] = call.data[REMINDER_DATE_LANG]
        if REMINDER_DATE in call.data:
            due_date: datetime | None = dt_util.parse_datetime(call.data[REMINDER_DATE])
            if due_date is None:
                due: date | None = dt_util.parse_date(call.data[REMINDER_DATE])
                if due is None:
                    raise ValueError(f'Invalid reminder_date: {call.data[REMINDER_DATE]}')
                due_date = datetime(due.year, due.month, due.day)
            due_date = dt_util.as_utc(due_date)
            date_format: str = '%Y-%m-%dT%H:%M:%S'
            _reminder_due['date'] = datetime.strftime(due_date, date_format)

        async def add_reminder(reminder_due: dict[str, str]) -> Any:
            reminder_data: dict[str, list[dict[str, Any]]] = {'commands': [{'type': 'reminder_add', 'temp_id': str(uuid.uuid1()), 'uuid': str(uuid.uuid1()), 'args': {'item_id': api_task.id, 'type': 'absolute', 'due': reminder_due}}]}
            headers: dict[str, str] = create_headers(token=coordinator.token, with_content=True)
            return await session.post(sync_url, headers=headers, json=reminder_data)
        if _reminder_due:
            await add_reminder(_reminder_due)
        _LOGGER.debug('Created Todoist task: %s', call.data[CONTENT])
    hass.services.async_register(DOMAIN, SERVICE_NEW_TASK, handle_new_task, schema=NEW_TASK_SERVICE_SCHEMA)

class TodoistProjectEntity(CoordinatorEntity[TodoistCoordinator], CalendarEntity):
    """A device for getting the next Task from a Todoist Project."""

    def __init__(self, coordinator: TodoistCoordinator, data: ProjectData, labels: list[Label], due_date_days: int | None=None, whitelisted_labels: list[str] | None=None, whitelisted_projects: list[str] | None=None) -> None:
        """Create the Todoist Calendar Entity."""
        super().__init__(coordinator=coordinator)
        self.data: TodoistProjectData = TodoistProjectData(data, labels, coordinator, due_date_days=due_date_days, whitelisted_labels=whitelisted_labels, whitelisted_projects=whitelisted_projects)
        self._cal_data: CalData = {}
        self._name: str = data[CONF_NAME]
        self._attr_unique_id: str | None = str(data[CONF_ID]) if data.get(CONF_ID) is not None else None

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self.data.update()
        super()._handle_coordinator_update()

    @property
    def event(self) -> CalendarEvent | None:
        """Return the next upcoming event."""
        return self.data.calendar_event

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        return self._name

    async def async_update(self) -> None:
        """Update all Todoist Calendars."""
        await super().async_update()
        self.data.update()

    async def async_get_events(self, hass: HomeAssistant, start_date: datetime, end_date: datetime) -> list[CalendarEvent]:
        """Get all events in a specific time frame."""
        return await self.data.async_get_events(start_date, end_date)

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the device state attributes."""
        if self.data.event is None:
            return None
        return {DUE_TODAY: self.data.event[DUE_TODAY], OVERDUE: self.data.event[OVERDUE], ALL_TASKS: [task[SUMMARY] for task in self.data.all_project_tasks], PRIORITY: self.data.event[PRIORITY], LABELS: self.data.event[LABELS]}

class TodoistProjectData:
    """Class used by the Task Entity service object to hold all Todoist Tasks.

    This is analogous to the GoogleCalendarData found in the Google Calendar
    component.

    Takes an object with a 'name' field and optionally an 'id' field (either
    user-defined or from the Todoist API), a Todoist API token, and an optional
    integer specifying the latest number of days from now a task can be due (7
    means everything due in the next week, 0 means today, etc.).

    This object has an exposed 'event' property (used by the Calendar platform
    to determine the next calendar event) and an exposed 'update' method (used
    by the Calendar platform to poll for new calendar events).

    The 'event' is a representation of a Todoist Task, with defined parameters
    of 'due_today' (is the task due today?), 'all_day' (does the task have a
    due date?), 'task_labels' (all labels assigned to the task), 'message'
    (the content of the task, e.g. 'Fetch Mail'), 'description' (a URL pointing
    to the task on the Todoist website), 'end_time' (what time the event is
    due), 'start_time' (what time this event was last updated), 'overdue' (is
    the task past its due date?), 'priority' (1-4, how important the task is,
    with 4 being the most important), and 'all_tasks' (all tasks in this
    project, sorted by how important they are).

    'offset_reached', 'location', and 'friendly_name' are defined by the
    platform itself, but are not used by this component at all.

    The 'update' method polls the Todoist API for new projects/tasks, as well
    as any updates to current projects/tasks. This occurs every SCAN_INTERVAL minutes.
    """

    def __init__(self, project_data: ProjectData, labels: list[Label], coordinator: TodoistCoordinator, due_date_days: int | None=None, whitelisted_labels: list[str] | None=None, whitelisted_projects: list[str] | None=None) -> None:
        """Initialize a Todoist Project."""
        self.event: TodoistEvent | None = None
        self._coordinator: TodoistCoordinator = coordinator
        self._name: str = project_data[CONF_NAME]
        self._id: str | None = project_data.get(CONF_ID)
        self._labels: list[Label] = labels
        self.all_project_tasks: list[TodoistEvent] = []
        self._due_date_days: timedelta | None = None
        if due_date_days is not None:
            self._due_date_days = timedelta(days=due_date_days)
        self._label_whitelist: list[str] = []
        if whitelisted_labels is not None:
            self._label_whitelist = whitelisted_labels
        self._project_id_whitelist: list[str] = []
        if whitelisted_projects is not None:
            self._project_id_whitelist = whitelisted_projects

    @property
    def calendar_event(self) -> CalendarEvent | None:
        """Return the next upcoming calendar event."""
        if not self.event:
            return None
        start: datetime |