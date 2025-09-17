from __future__ import annotations
from datetime import date, datetime, timedelta
import logging
from typing import Any, Optional, Union, List, Dict
import uuid

from todoist_api_python.api_async import TodoistAPIAsync
from todoist_api_python.endpoints import get_sync_url
from todoist_api_python.headers import create_headers
from todoist_api_python.models import Due, Label, Task
import voluptuous as vol

from homeassistant.components.calendar import (
    PLATFORM_SCHEMA as CALENDAR_PLATFORM_SCHEMA,
    CalendarEntity,
    CalendarEvent,
)
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

from .const import (
    ALL_DAY,
    ALL_TASKS,
    ASSIGNEE,
    COMPLETED,
    CONF_EXTRA_PROJECTS,
    CONF_PROJECT_DUE_DATE,
    CONF_PROJECT_LABEL_WHITELIST,
    CONF_PROJECT_WHITELIST,
    CONTENT,
    DESCRIPTION,
    DOMAIN,
    DUE_DATE,
    DUE_DATE_LANG,
    DUE_DATE_STRING,
    DUE_DATE_VALID_LANGS,
    DUE_TODAY,
    END,
    LABELS,
    OVERDUE,
    PRIORITY,
    PROJECT_NAME,
    REMINDER_DATE,
    REMINDER_DATE_LANG,
    REMINDER_DATE_STRING,
    SECTION_NAME,
    SERVICE_NEW_TASK,
    START,
    SUMMARY,
)
from .coordinator import TodoistCoordinator
from .types import CalData, CustomProject, ProjectData, TodoistEvent

_LOGGER = logging.getLogger(__name__)

NEW_TASK_SERVICE_SCHEMA = vol.Schema(
    {
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
        vol.Exclusive(REMINDER_DATE, 'reminder_date'): cv.string,
    }
)
PLATFORM_SCHEMA = CALENDAR_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_TOKEN): cv.string,
        vol.Optional(CONF_EXTRA_PROJECTS, default=[]): vol.All(
            cv.ensure_list,
            vol.Schema(
                [
                    vol.Schema(
                        {
                            vol.Required(CONF_NAME): cv.string,
                            vol.Optional(CONF_PROJECT_DUE_DATE): vol.Coerce(int),
                            vol.Optional(CONF_PROJECT_WHITELIST, default=[]): vol.All(
                                cv.ensure_list, [vol.All(cv.string, vol.Lower)]
                            ),
                            vol.Optional(CONF_PROJECT_LABEL_WHITELIST, default=[]): vol.All(
                                cv.ensure_list, [vol.All(cv.string)]
                            ),
                        }
                    )
                ]
            ),
        ),
    }
)
SCAN_INTERVAL = timedelta(minutes=1)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: TodoistCoordinator = hass.data[DOMAIN][entry.entry_id]
    projects = await coordinator.async_get_projects()
    labels: List[Label] = await coordinator.async_get_labels()
    entities: List[TodoistProjectEntity] = []
    for project in projects:
        project_data: Dict[str, Any] = {CONF_NAME: project.name, CONF_ID: project.id}
        entities.append(TodoistProjectEntity(coordinator, project_data, labels))
    async_add_entities(entities)
    async_register_services(hass, coordinator)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    token: str = config[CONF_TOKEN]
    project_id_lookup: Dict[str, Any] = {}
    api: TodoistAPIAsync = TodoistAPIAsync(token)
    coordinator: TodoistCoordinator = TodoistCoordinator(hass, _LOGGER, None, SCAN_INTERVAL, api, token)
    await coordinator.async_refresh()

    async def _shutdown_coordinator(event: Event) -> None:
        await coordinator.async_shutdown()

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _shutdown_coordinator)
    projects = await api.get_projects()
    labels: List[Label] = await api.get_labels()
    project_devices: List[TodoistProjectEntity] = []
    for project in projects:
        project_data: Dict[str, Any] = {CONF_NAME: project.name, CONF_ID: project.id}
        project_devices.append(TodoistProjectEntity(coordinator, project_data, labels))
        project_id_lookup[project.name.lower()] = project.id
    extra_projects: List[Dict[str, Any]] = config[CONF_EXTRA_PROJECTS]
    for extra_project in extra_projects:
        project_due_date: Optional[int] = extra_project.get(CONF_PROJECT_DUE_DATE)
        project_label_filter: List[str] = extra_project[CONF_PROJECT_LABEL_WHITELIST]
        project_name_filter: List[str] = extra_project[CONF_PROJECT_WHITELIST]
        project_id_filter: Optional[List[Any]] = None
        if project_name_filter is not None:
            project_id_filter = [project_id_lookup[project_name.lower()] for project_name in project_name_filter]
        project_devices.append(
            TodoistProjectEntity(
                coordinator,
                {'id': None, 'name': extra_project['name']},
                labels,
                due_date_days=project_due_date,
                whitelisted_labels=project_label_filter,
                whitelisted_projects=project_id_filter,
            )
        )
    async_add_entities(project_devices, update_before_add=True)
    async_register_services(hass, coordinator)


def async_register_services(hass: HomeAssistant, coordinator: TodoistCoordinator) -> None:
    if hass.services.has_service(DOMAIN, SERVICE_NEW_TASK):
        return
    session = async_get_clientsession(hass)

    async def handle_new_task(call: ServiceCall) -> None:
        project_name: str = call.data[PROJECT_NAME]
        projects = await coordinator.async_get_projects()
        project_id: Optional[Any] = None
        for project in projects:
            if project_name == project.name.lower():
                project_id = project.id
                break
        if project_id is None:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key='project_invalid',
                translation_placeholders={'project': project_name},
            )
        section_id: Optional[Any] = None
        if SECTION_NAME in call.data:
            section_name: str = call.data[SECTION_NAME]
            sections = await coordinator.async_get_sections(project_id)
            for section in sections:
                if section_name == section.name.lower():
                    section_id = section.id
                    break
            if section_id is None:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key='section_invalid',
                    translation_placeholders={'section': section_name, 'project': project_name},
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
            collaborators = await coordinator.api.get_collaborators(project_id)
            collaborator_id_lookup: Dict[str, Any] = {collab.name.lower(): collab.id for collab in collaborators}
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
                due = dt_util.parse_date(call.data[DUE_DATE])
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

        async def add_reminder(reminder_due: Dict[str, Any]) -> Any:
            reminder_data: Dict[str, Any] = {
                'commands': [
                    {
                        'type': 'reminder_add',
                        'temp_id': str(uuid.uuid1()),
                        'uuid': str(uuid.uuid1()),
                        'args': {'item_id': api_task.id, 'type': 'absolute', 'due': reminder_due},
                    }
                ]
            }
            headers = create_headers(token=coordinator.token, with_content=True)
            return await session.post(sync_url, headers=headers, json=reminder_data)

        if _reminder_due:
            await add_reminder(_reminder_due)
        _LOGGER.debug('Created Todoist task: %s', call.data[CONTENT])

    hass.services.async_register(DOMAIN, SERVICE_NEW_TASK, handle_new_task, schema=NEW_TASK_SERVICE_SCHEMA)


class TodoistProjectEntity(CoordinatorEntity[TodoistCoordinator], CalendarEntity):
    def __init__(
        self,
        coordinator: TodoistCoordinator,
        data: Dict[str, Any],
        labels: List[Label],
        due_date_days: Optional[int] = None,
        whitelisted_labels: Optional[List[str]] = None,
        whitelisted_projects: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(coordinator=coordinator)
        self.data = TodoistProjectData(data, labels, coordinator, due_date_days=due_date_days, whitelisted_labels=whitelisted_labels, whitelisted_projects=whitelisted_projects)
        self._cal_data: Dict[str, Any] = {}
        self._name: str = data[CONF_NAME]
        self._attr_unique_id: Optional[str] = str(data[CONF_ID]) if data.get(CONF_ID) is not None else None

    @callback
    def _handle_coordinator_update(self) -> None:
        self.data.update()
        super()._handle_coordinator_update()

    @property
    def event(self) -> Optional[CalendarEvent]:
        return self.data.calendar_event

    @property
    def name(self) -> str:
        return self._name

    async def async_update(self) -> None:
        await super().async_update()
        self.data.update()

    async def async_get_events(
        self, hass: HomeAssistant, start_date: Union[datetime, date], end_date: Union[datetime, date]
    ) -> List[CalendarEvent]:
        return await self.data.async_get_events(start_date, end_date)

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        if self.data.event is None:
            return None
        return {
            DUE_TODAY: self.data.event[DUE_TODAY],
            OVERDUE: self.data.event[OVERDUE],
            ALL_TASKS: [task[SUMMARY] for task in self.data.all_project_tasks],
            PRIORITY: self.data.event[PRIORITY],
            LABELS: self.data.event[LABELS],
        }


class TodoistProjectData:
    def __init__(
        self,
        project_data: Dict[str, Any],
        labels: List[Label],
        coordinator: TodoistCoordinator,
        due_date_days: Optional[int] = None,
        whitelisted_labels: Optional[List[str]] = None,
        whitelisted_projects: Optional[List[Any]] = None,
    ) -> None:
        self.event: Optional[Dict[str, Any]] = None
        self._coordinator: TodoistCoordinator = coordinator
        self._name: str = project_data[CONF_NAME]
        self._id: Optional[Any] = project_data.get(CONF_ID)
        self._labels: List[Label] = labels
        self.all_project_tasks: List[Dict[str, Any]] = []
        self._due_date_days: Optional[timedelta] = timedelta(days=due_date_days) if due_date_days is not None else None
        self._label_whitelist: List[str] = whitelisted_labels if whitelisted_labels is not None else []
        self._project_id_whitelist: List[Any] = whitelisted_projects if whitelisted_projects is not None else []

    @property
    def calendar_event(self) -> Optional[CalendarEvent]:
        if not self.event:
            return None
        start = self.event[START]
        if self.event.get(ALL_DAY) or self.event[END] is None:
            return CalendarEvent(summary=self.event[SUMMARY], start=start.date(), end=start.date() + timedelta(days=1))
        return CalendarEvent(summary=self.event[SUMMARY], start=start, end=self.event[END])

    def create_todoist_task(self, data: Task) -> Optional[Dict[str, Any]]:
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
            SUMMARY: data.content,
        }
        if self._project_id_whitelist and data.project_id not in self._project_id_whitelist:
            return None
        labels = data.labels or []
        task[LABELS] = [label.name for label in self._labels if label.name in labels]
        if self._label_whitelist and (not any((label in task[LABELS] for label in self._label_whitelist))):
            return None
        if data.due is not None:
            end = dt_util.parse_datetime(data.due.datetime if data.due.datetime else data.due.date)
            task[END] = dt_util.as_local(end) if end is not None else end
            if task[END] is not None:
                if self._due_date_days is not None and task[END] > dt_util.now() + self._due_date_days:
                    return None
                task[DUE_TODAY] = task[END].date() == dt_util.now().date()
                if task[END] <= task[START]:
                    task[OVERDUE] = True
                    task[END] = task[START] + timedelta(hours=1)
                else:
                    task[OVERDUE] = False
        else:
            if self._due_date_days is not None:
                return None
            task[END] = None
            task[ALL_DAY] = True
            task[DUE_TODAY] = False
            task[OVERDUE] = False
        return task

    @staticmethod
    def select_best_task(project_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        event = project_tasks[-1]
        for proposed_event in project_tasks:
            if event == proposed_event:
                continue
            if proposed_event[COMPLETED]:
                continue
            if proposed_event[END] is None:
                if event[END] is None and proposed_event[PRIORITY] < event[PRIORITY]:
                    event = proposed_event
                continue
            if event[END] is None:
                event = proposed_event
                continue
            if proposed_event[END].date() > event[END].date():
                continue
            if proposed_event[END].date() < event[END].date():
                event = proposed_event
                continue
            if proposed_event[PRIORITY] > event[PRIORITY]:
                event = proposed_event
                continue
            if proposed_event[PRIORITY] == event[PRIORITY] and (event[END] is not None and proposed_event[END] < event[END]):
                event = proposed_event
                continue
        return event

    async def async_get_events(self, start_date: Union[datetime, date], end_date: Union[datetime, date]) -> List[CalendarEvent]:
        tasks: List[Task] = self._coordinator.data
        if self._id is None:
            project_task_data: List[Task] = [task for task in tasks if self.create_todoist_task(task) is not None]
        else:
            project_task_data = [task for task in tasks if task.project_id == self._id]
        events: List[CalendarEvent] = []
        for task in project_task_data:
            if task.due is None:
                continue
            start = get_start(task.due)
            if start is None:
                continue
            event = CalendarEvent(summary=task.content, start=start, end=start + timedelta(days=1))
            if event.start_datetime_local >= end_date:
                continue
            if event.end_datetime_local < start_date:
                continue
            events.append(event)
        return events

    def update(self) -> None:
        tasks: List[Task] = self._coordinator.data
        if self._id is None:
            project_task_data: List[Task] = [task for task in tasks if not self._project_id_whitelist or task.project_id in self._project_id_whitelist]
        else:
            project_task_data = [task for task in tasks if task.project_id == self._id]
        if not project_task_data:
            _LOGGER.debug('No data for %s', self._name)
            self.event = None
            return
        project_tasks: List[Dict[str, Any]] = []
        for task in project_task_data:
            todoist_task = self.create_todoist_task(task)
            if todoist_task is not None:
                project_tasks.append(todoist_task)
        if not project_tasks:
            _LOGGER.debug('No valid tasks for %s', self._name)
            self.event = None
            return
        self.all_project_tasks.clear()
        while project_tasks:
            best_task = self.select_best_task(project_tasks)
            _LOGGER.debug('Found Todoist Task: %s', best_task[SUMMARY])
            project_tasks.remove(best_task)
            self.all_project_tasks.append(best_task)
        event = self.all_project_tasks[0]
        if event is None or event[START] is None:
            _LOGGER.debug('No valid event or event start for %s', self._name)
            self.event = None
            return
        self.event = event
        _LOGGER.debug('Updated %s', self._name)


def get_start(due: Due) -> Optional[Union[datetime, date]]:
    if due.datetime:
        start = dt_util.parse_datetime(due.datetime)
        if not start:
            return None
        return dt_util.as_local(start)
    if due.date:
        return dt_util.parse_date(due.date)
    return None