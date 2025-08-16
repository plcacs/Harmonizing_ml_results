def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    ...

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    ...

def async_register_services(hass: HomeAssistant, coordinator: TodoistCoordinator) -> None:
    ...

class TodoistProjectEntity(CoordinatorEntity[TodoistCoordinator], CalendarEntity):
    def __init__(self, coordinator: TodoistCoordinator, data: dict, labels: List[Label], due_date_days: Optional[int] = None, whitelisted_labels: Optional[List[str]] = None, whitelisted_projects: Optional[List[int]] = None) -> None:
        ...

    def _handle_coordinator_update(self) -> None:
        ...

    @property
    def event(self) -> Optional[CalendarEvent]:
        ...

    @property
    def name(self) -> str:
        ...

    async def async_update(self) -> None:
        ...

    async def async_get_events(self, hass: HomeAssistant, start_date: date, end_date: date) -> List[CalendarEvent]:
        ...

    @property
    def extra_state_attributes(self) -> Optional[dict]:
        ...

class TodoistProjectData:
    def __init__(self, project_data: dict, labels: List[Label], coordinator: TodoistCoordinator, due_date_days: Optional[int] = None, whitelisted_labels: Optional[List[str]] = None, whitelisted_projects: Optional[List[int]] = None) -> None:
        ...

    @property
    def calendar_event(self) -> Optional[CalendarEvent]:
        ...

    def create_todoist_task(self, data: Task) -> Optional[dict]:
        ...

    @staticmethod
    def select_best_task(project_tasks: List[dict]) -> dict:
        ...

    async def async_get_events(self, start_date: date, end_date: date) -> List[CalendarEvent]:
        ...

    def update(self) -> None:
        ...

def get_start(due: Due) -> Optional[Union[date, datetime]]:
    ...
