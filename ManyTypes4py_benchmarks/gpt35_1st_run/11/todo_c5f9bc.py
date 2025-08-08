from __future__ import annotations
from datetime import datetime
from typing import Any, cast, List, Dict
from homeassistant.components.todo import TodoItem, TodoItemStatus, TodoListEntity, TodoListEntityFeature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util
from .coordinator import GoogleTasksConfigEntry, TaskUpdateCoordinator

PARALLEL_UPDATES: int = 0
TODO_STATUS_MAP: Dict[str, TodoItemStatus] = {'needsAction': TodoItemStatus.NEEDS_ACTION, 'completed': TodoItemStatus.COMPLETED}
TODO_STATUS_MAP_INV: Dict[TodoItemStatus, str] = {v: k for k, v in TODO_STATUS_MAP.items()}

def _convert_todo_item(item: TodoItem) -> Dict[str, Any]:
    ...

def _convert_api_item(item: Dict[str, Any]) -> TodoItem:
    ...

async def async_setup_entry(hass: HomeAssistant, entry: GoogleTasksConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class GoogleTaskTodoListEntity(CoordinatorEntity[TaskUpdateCoordinator], TodoListEntity):
    ...

    def __init__(self, coordinator: TaskUpdateCoordinator, name: str, config_entry_id: str, task_list_id: str) -> None:
        ...

    @property
    def todo_items(self) -> List[TodoItem]:
        ...

    async def async_create_todo_item(self, item: TodoItem) -> None:
        ...

    async def async_update_todo_item(self, item: TodoItem) -> None:
        ...

    async def async_delete_todo_items(self, uids: List[str]) -> None:
        ...

    async def async_move_todo_item(self, uid: str, previous_uid: str = None) -> None:
        ...

def _order_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ...
