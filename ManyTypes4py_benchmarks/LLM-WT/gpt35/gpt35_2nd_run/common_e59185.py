from homeassistant.core import HomeAssistant
from typing import Optional, List

def reload(hass: HomeAssistant) -> None:
    ...

def async_reload(hass: HomeAssistant) -> None:
    ...

def set_group(hass: HomeAssistant, object_id: str, name: Optional[str] = None, entity_ids: Optional[List[str]] = None, icon: Optional[str] = None, add: Optional[bool] = None) -> None:
    ...

def async_set_group(hass: HomeAssistant, object_id: str, name: Optional[str] = None, entity_ids: Optional[List[str]] = None, icon: Optional[str] = None, add: Optional[bool] = None) -> None:
    ...

def async_remove(hass: HomeAssistant, object_id: str) -> None:
    ...
