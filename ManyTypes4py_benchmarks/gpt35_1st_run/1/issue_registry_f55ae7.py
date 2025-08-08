from __future__ import annotations
import dataclasses
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict, cast, Dict, List, Optional

class StrEnum(str, Enum):
    pass

class EventIssueRegistryUpdatedData(TypedDict):
    action: str
    domain: str
    issue_id: str

class IssueEntry:
    created: datetime
    dismissed_version: Optional[str]
    domain: str
    is_persistent: bool
    issue_id: str
    breaks_in_ha_version: Optional[str]
    data: Any
    is_fixable: bool
    issue_domain: Optional[str]
    learn_more_url: Optional[str]
    severity: IssueSeverity
    translation_key: str
    translation_placeholders: Optional[Dict[str, Any]]

    def to_json(self) -> Dict[str, Any]:
        pass

class IssueRegistryStore(Store[Dict[str, List[Dict[str, Any]]]):
    async def _async_migrate_func(self, old_major_version: int, old_minor_version: int, old_data: dict) -> dict:
        pass

class IssueRegistry(BaseRegistry):
    def __init__(self, hass: HomeAssistant):
        pass

    def async_get_issue(self, domain: str, issue_id: str) -> Optional[IssueEntry]:
        pass

    def async_get_or_create(self, domain: str, issue_id: str, *, breaks_in_ha_version: Optional[str] = None, data: Any = None, is_fixable: bool, is_persistent: bool, issue_domain: Optional[str] = None, learn_more_url: Optional[str] = None, severity: IssueSeverity, translation_key: str, translation_placeholders: Optional[Dict[str, Any]] = None) -> IssueEntry:
        pass

    def async_delete(self, domain: str, issue_id: str) -> None:
        pass

    def async_ignore(self, domain: str, issue_id: str, ignore: bool) -> IssueEntry:
        pass

    def make_read_only(self) -> None:
        pass

    async def async_load(self) -> None:
        pass

    def _data_to_save(self) -> dict:
        pass

def async_get(hass: HomeAssistant) -> IssueRegistry:
    pass

async def async_load(hass: HomeAssistant, *, read_only: bool = False) -> None:
    pass

def async_create_issue(hass: HomeAssistant, domain: str, issue_id: str, *, breaks_in_ha_version: Optional[str] = None, data: Any = None, is_fixable: bool, is_persistent: bool = False, issue_domain: Optional[str] = None, learn_more_url: Optional[str] = None, severity: IssueSeverity, translation_key: str, translation_placeholders: Optional[Dict[str, Any]] = None) -> None:
    pass

def create_issue(hass: HomeAssistant, domain: str, issue_id: str, *, breaks_in_ha_version: Optional[str] = None, data: Any = None, is_fixable: bool, is_persistent: bool = False, issue_domain: Optional[str] = None, learn_more_url: Optional[str] = None, severity: IssueSeverity, translation_key: str, translation_placeholders: Optional[Dict[str, Any]] = None) -> None:
    pass

def async_delete_issue(hass: HomeAssistant, domain: str, issue_id: str) -> None:
    pass

def delete_issue(hass: HomeAssistant, domain: str, issue_id: str) -> None:
    pass

def async_ignore_issue(hass: HomeAssistant, domain: str, issue_id: str, ignore: bool) -> None:
    pass
