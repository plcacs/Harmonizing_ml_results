from __future__ import annotations
from collections.abc import Callable
from functools import wraps
from typing import Any, Awaitable, Callable as TypingCallable, Optional, Union, cast, TYPE_CHECKING
import voluptuous as vol

from homeassistant.const import HASSIO_USER_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import Unauthorized
from homeassistant.helpers.typing import VolDictType
from . import const, messages
from .connection import ActiveConnection

# _handle_async_response expects an async function that returns an Awaitable[Any]
async def _handle_async_response(
    func: TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Awaitable[Any]],
    hass: HomeAssistant,
    connection: ActiveConnection,
    msg: VolDictType,
) -> None:
    try:
        await func(hass, connection, msg)
    except Exception as err:
        connection.async_handle_exception(msg, err)

def async_response(
    func: TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Awaitable[Any]]
) -> TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], None]:
    """Decorate an async function to handle WebSocket API messages."""
    task_name = f"websocket_api.async:{func.__name__}"

    @callback
    @wraps(func)
    def schedule_handler(
        hass: HomeAssistant, connection: ActiveConnection, msg: VolDictType
    ) -> None:
        """Schedule the handler."""
        hass.async_create_background_task(
            _handle_async_response(func, hass, connection, msg), task_name, eager_start=True
        )

    return schedule_handler

def require_admin(
    func: TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Any]
) -> TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Any]:
    """Websocket decorator to require user to be an admin."""
    @wraps(func)
    def with_admin(hass: HomeAssistant, connection: ActiveConnection, msg: VolDictType) -> Any:
        user = connection.user
        if user is None or not user.is_admin:
            raise Unauthorized
        return func(hass, connection, msg)
    return with_admin

def ws_require_user(
    only_owner: bool = False,
    only_system_user: bool = False,
    allow_system_user: bool = True,
    only_active_user: bool = True,
    only_inactive_user: bool = False,
    only_supervisor: bool = False,
) -> Callable[
    [TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Any]],
    TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Any],
]:
    """Decorate function validating login user exist in current WS connection.

    Will write out error message if not authenticated.
    """
    def validator(
        func: TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Any]
    ) -> TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Any]:
        """Decorate func."""
        @wraps(func)
        def check_current_user(
            hass: HomeAssistant, connection: ActiveConnection, msg: VolDictType
        ) -> Optional[Any]:
            """Check current user."""
            def output_error(message_id: str, message: str) -> None:
                """Output error message."""
                connection.send_message(messages.error_message(msg["id"], message_id, message))
            if only_owner and (not connection.user.is_owner):
                output_error("only_owner", "Only allowed as owner")
                return None
            if only_system_user and (not connection.user.system_generated):
                output_error("only_system_user", "Only allowed as system user")
                return None
            if not allow_system_user and connection.user.system_generated:
                output_error("not_system_user", "Not allowed as system user")
                return None
            if only_active_user and (not connection.user.is_active):
                output_error("only_active_user", "Only allowed as active user")
                return None
            if only_inactive_user and connection.user.is_active:
                output_error("only_inactive_user", "Not allowed as active user")
                return None
            if only_supervisor and connection.user.name != HASSIO_USER_NAME:
                output_error("only_supervisor", "Only allowed as Supervisor")
                return None
            return func(hass, connection, msg)
        return check_current_user
    return validator

def websocket_command(
    schema: Union[dict[str, Any], vol.Schema]
) -> Callable[
    [TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Any]],
    TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Any],
]:
    """Tag a function as a websocket command.

    The schema must be either a dictionary where the keys are voluptuous markers, or
    a voluptuous.All schema where the first item is a voluptuous Mapping schema.
    """
    is_dict: bool = isinstance(schema, dict)
    if is_dict:
        command = cast(dict, schema)["type"]
    else:
        command = schema.validators[0].schema["type"]

    def decorate(
        func: TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Any]
    ) -> TypingCallable[[HomeAssistant, ActiveConnection, VolDictType], Any]:
        """Decorate ws command function."""
        if is_dict and len(cast(dict, schema)) == 1:
            func._ws_schema = False  # type: ignore
        elif is_dict:
            func._ws_schema = messages.BASE_COMMAND_MESSAGE_SCHEMA.extend(cast(dict, schema))  # type: ignore
        else:
            if TYPE_CHECKING:
                assert not isinstance(schema, dict)
            extended_schema = vol.All(
                schema.validators[0].extend(messages.BASE_COMMAND_MESSAGE_SCHEMA.schema),  # type: ignore
                *schema.validators[1:]
            )
            func._ws_schema = extended_schema  # type: ignore
        func._ws_command = command  # type: ignore
        return func
    return decorate