from __future__ import annotations
import dataclasses
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict, cast

class EventIssueRegistryUpdatedData(TypedDict):
    """Event data for when the issue registry is updated."""

class IssueSeverity(Enum):
    """Issue severity."""
    CRITICAL = 'critical'
    ERROR = 'error'
    WARNING = 'warning'

@dataclasses.dataclass(slots=True, frozen=True)
class IssueEntry:
    """Issue Registry Entry."""

    def to_json(self) -> dict:
        """Return a JSON serializable representation for storage."""
        result: dict[str, Any] = {'created': self.created.isoformat(), 'dismissed_version': self.dismissed_version, 'domain': self.domain, 'is_persistent': False, 'issue_id': self.issue_id}
        if not self.is_persistent:
            return result
        return {**result, 'breaks_in_ha_version': self.breaks_in_ha_version, 'data': self.data, 'is_fixable': self.is_fixable, 'is_persistent': True, 'issue_domain': self.issue_domain, 'issue_id': self.issue_id, 'learn_more_url': self.learn_more_url, 'severity': self.severity, 'translation_key': self.translation_key, 'translation_placeholders': self.translation_placeholders}

class IssueRegistryStore(Store[dict[str, list[dict[str, Any]]]):
    """Store entity registry data."""

    async def _async_migrate_func(self, old_major_version: int, old_minor_version: int, old_data: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
        """Migrate to the new version."""
        if old_major_version == 1 and old_minor_version < 2:
            for issue in old_data['issues']:
                issue['is_persistent'] = False
        return old_data

class IssueRegistry(BaseRegistry):
    """Class to hold a registry of issues."""

    def __init__(self, hass: HomeAssistant):
        """Initialize the issue registry."""
        self.hass = hass
        self.issues: dict[tuple[str, str], IssueEntry] = {}
        self._store = IssueRegistryStore(hass, STORAGE_VERSION_MAJOR, STORAGE_KEY, atomic_writes=True, minor_version=STORAGE_VERSION_MINOR)

    @callback
    def async_get_issue(self, domain: str, issue_id: str) -> IssueEntry:
        """Get issue by id."""
        return self.issues.get((domain, issue_id))

    @callback
    def async_get_or_create(self, domain: str, issue_id: str, *, breaks_in_ha_version: str = None, data: Any = None, is_fixable: bool, is_persistent: bool, issue_domain: str = None, learn_more_url: str = None, severity: IssueSeverity, translation_key: str, translation_placeholders: dict = None) -> IssueEntry:
        """Get issue. Create if it doesn't exist."""
        self.hass.verify_event_loop_thread('issue_registry.async_get_or_create')
        if (issue := self.async_get_issue(domain, issue_id)) is None:
            issue = IssueEntry(active=True, breaks_in_ha_version=breaks_in_ha_version, created=dt_util.utcnow(), data=data, dismissed_version=None, domain=domain, is_fixable=is_fixable, is_persistent=is_persistent, issue_domain=issue_domain, issue_id=issue_id, learn_more_url=learn_more_url, severity=severity, translation_key=translation_key, translation_placeholders=translation_placeholders)
            self.issues[domain, issue_id] = issue
            self.async_schedule_save()
            self.hass.bus.async_fire_internal(EVENT_REPAIRS_ISSUE_REGISTRY_UPDATED, EventIssueRegistryUpdatedData(action='create', domain=domain, issue_id=issue_id))
        else:
            replacement = dataclasses.replace(issue, active=True, breaks_in_ha_version=breaks_in_ha_version, data=data, is_fixable=is_fixable, is_persistent=is_persistent, issue_domain=issue_domain, learn_more_url=learn_more_url, severity=severity, translation_key=translation_key, translation_placeholders=translation_placeholders)
            if replacement != issue:
                issue = self.issues[domain, issue_id] = replacement
                self.async_schedule_save()
                self.hass.bus.async_fire_internal(EVENT_REPAIRS_ISSUE_REGISTRY_UPDATED, EventIssueRegistryUpdatedData(action='update', domain=domain, issue_id=issue_id))
        return issue

    @callback
    def async_delete(self, domain: str, issue_id: str) -> None:
        """Delete issue."""
        self.hass.verify_event_loop_thread('issue_registry.async_delete')
        if self.issues.pop((domain, issue_id), None) is None:
            return
        self.async_schedule_save()
        self.hass.bus.async_fire_internal(EVENT_REPAIRS_ISSUE_REGISTRY_UPDATED, EventIssueRegistryUpdatedData(action='remove', domain=domain, issue_id=issue_id))

    @callback
    def async_ignore(self, domain: str, issue_id: str, ignore: bool) -> IssueEntry:
        """Ignore issue."""
        self.hass.verify_event_loop_thread('issue_registry.async_ignore')
        old = self.issues[domain, issue_id]
        dismissed_version = ha_version if ignore else None
        if old.dismissed_version == dismissed_version:
            return old
        issue = self.issues[domain, issue_id] = dataclasses.replace(old, dismissed_version=dismissed_version)
        self.async_schedule_save()
        self.hass.bus.async_fire_internal(EVENT_REPAIRS_ISSUE_REGISTRY_UPDATED, EventIssueRegistryUpdatedData(action='update', domain=domain, issue_id=issue_id))
        return issue

    @callback
    def make_read_only(self) -> None:
        """Make the registry read-only.

        This method is irreversible.
        """

    async def async_load(self) -> None:
        """Load the issue registry."""

    @callback
    def _data_to_save(self) -> dict:
        """Return data of issue registry to store in a file."""

@callback
@singleton(DATA_REGISTRY)
def async_get(hass: HomeAssistant) -> IssueRegistry:
    """Get issue registry."""
    return IssueRegistry(hass)

async def async_load(hass: HomeAssistant, *, read_only: bool = False) -> None:
    """Load issue registry."""
    ir = async_get(hass)
    if read_only:
        ir.make_read_only()
    return await ir.async_load()

@callback
def async_create_issue(hass: HomeAssistant, domain: str, issue_id: str, *, breaks_in_ha_version: str = None, data: Any = None, is_fixable: bool, is_persistent: bool = False, issue_domain: str = None, learn_more_url: str = None, severity: IssueSeverity, translation_key: str, translation_placeholders: dict = None) -> None:
    """Create an issue, or replace an existing one."""

def create_issue(hass: HomeAssistant, domain: str, issue_id: str, *, breaks_in_ha_version: str = None, data: Any = None, is_fixable: bool, is_persistent: bool = False, issue_domain: str = None, learn_more_url: str = None, severity: IssueSeverity, translation_key: str, translation_placeholders: dict = None) -> None:
    """Create an issue, or replace an existing one."""

@callback
def async_delete_issue(hass: HomeAssistant, domain: str, issue_id: str) -> None:
    """Delete an issue.

    It is not an error to delete an issue that does not exist.
    """

def delete_issue(hass: HomeAssistant, domain: str, issue_id: str) -> None:
    """Delete an issue.

    It is not an error to delete an issue that does not exist.
    """

@callback
def async_ignore_issue(hass: HomeAssistant, domain: str, issue_id: str, ignore: bool) -> IssueEntry:
    """Ignore an issue.

    Will raise if the issue does not exist.
    """
