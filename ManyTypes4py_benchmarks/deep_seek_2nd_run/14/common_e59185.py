"""Collection of helper methods.

All containing methods are legacy helpers that should not be used by new
components. Instead call the service directly.
"""
from typing import Optional, Any, Dict, List
from homeassistant.components.group import ATTR_ADD_ENTITIES, ATTR_ENTITIES, ATTR_OBJECT_ID, DOMAIN, SERVICE_REMOVE, SERVICE_SET
from homeassistant.const import ATTR_ICON, ATTR_NAME, SERVICE_RELOAD
from homeassistant.core import HomeAssistant, callback
from homeassistant.loader import bind_hass

@bind_hass
def reload(hass: HomeAssistant) -> None:
    """Reload the automation from config."""
    hass.add_job(async_reload, hass)

@callback
@bind_hass
def async_reload(hass: HomeAssistant) -> None:
    """Reload the automation from config."""
    hass.async_create_task(hass.services.async_call(DOMAIN, SERVICE_RELOAD))

@bind_hass
def set_group(
    hass: HomeAssistant,
    object_id: str,
    name: Optional[str] = None,
    entity_ids: Optional[List[str]] = None,
    icon: Optional[str] = None,
    add: Optional[bool] = None
) -> None:
    """Create/Update a group."""
    hass.add_job(async_set_group, hass, object_id, name, entity_ids, icon, add)

@callback
@bind_hass
def async_set_group(
    hass: HomeAssistant,
    object_id: str,
    name: Optional[str] = None,
    entity_ids: Optional[List[str]] = None,
    icon: Optional[str] = None,
    add: Optional[bool] = None
) -> None:
    """Create/Update a group."""
    data: Dict[str, Any] = {key: value for key, value in (
        (ATTR_OBJECT_ID, object_id),
        (ATTR_NAME, name),
        (ATTR_ENTITIES, entity_ids),
        (ATTR_ICON, icon),
        (ATTR_ADD_ENTITIES, add)
    ) if value is not None}
    hass.async_create_task(hass.services.async_call(DOMAIN, SERVICE_SET, data))

@callback
@bind_hass
def async_remove(hass: HomeAssistant, object_id: str) -> None:
    """Remove a user group."""
    data: Dict[str, str] = {ATTR_OBJECT_ID: object_id}
    hass.async_create_task(hass.services.async_call(DOMAIN, SERVICE_REMOVE, data))
