"""Persistently store issues raised by integrations."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from enum import StrEnum
import functools as ft
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, Union, cast

from awesomeversion import AwesomeVersion, AwesomeVersionStrategy

from homeassistant.const import __version__ as ha_version
from homeassistant.core import HomeAssistant, callback
from homeassistant.util import dt as dt_util
from homeassistant.util.async_ import run_callback_threadsafe
from homeassistant.util.event_type import EventType
from homeassistant.util.hass_dict import HassKey

from .registry import BaseRegistry
from .singleton import singleton
from .storage import Store

DATA_REGISTRY: HassKey[IssueRegistry] = HassKey("issue_registry")
EVENT_REPAIRS_ISSUE_REGISTRY_UPDATED: EventType[EventIssueRegistryUpdatedData] = (
    EventType("repairs_issue_registry_updated")
)
STORAGE_KEY: str = "repairs.issue_registry"
STORAGE_VERSION_MAJOR: int = 1
STORAGE_VERSION_MINOR: int = 2


class EventIssueRegistryUpdatedData(TypedDict):
    """Event data for when the issue registry is updated."""

    action: Literal["create", "remove", "update"]
    domain: str
    issue_id: str


class IssueSeverity(StrEnum):
    """Issue severity."""

    CRITICAL: str = "critical"
    ERROR: str = "error"
    WARNING: str = "warning"


@dataclasses.dataclass(slots=True, frozen=True)
class IssueEntry:
    """Issue Registry Entry."""

    active: bool
    breaks_in_ha_version: Optional[str]
    created: datetime
    data: Optional[Dict[str, Union[str, int, float, None]]]
    dismissed_version: Optional[str]
    domain: str
    is_fixable: Optional[bool]
    is_persistent: bool
    issue_domain: Optional[str]
    issue_id: str
    learn_more_url: Optional[str]
    severity: Optional[IssueSeverity]
    translation_key: Optional[str]
    translation_placeholders: Optional[Dict[str, str]]

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON serializable representation for storage."""
        result: Dict[str, Any] = {
            "created": self.created.isoformat(),
            "dismissed_version": self.dismissed_version,
            "domain": self.domain,
            "is_persistent": False,
            "issue_id": self.issue_id,
        }
        if not self.is_persistent:
            return result
        return {
            **result,
            "breaks_in_ha_version": self.breaks_in_ha_version,
            "data": self.data,
            "is_fixable": self.is_fixable,
            "is_persistent": True,
            "issue_domain": self.issue_domain,
            "issue_id": self.issue_id,
            "learn_more_url": self.learn_more_url,
            "severity": self.severity.value if self.severity else None,
            "translation_key": self.translation_key,
            "translation_placeholders": self.translation_placeholders,
        }


class IssueRegistryStore(Store[Dict[str, List[Dict[str, Any]]]]):
    """Store entity registry data."""

    async def _async_migrate_func(
        self, old_major_version: int, old_minor_version: int, old_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Migrate to the new version."""
        if old_major_version == 1 and old_minor_version < 2:
            # Version 1.2 adds is_persistent
            for issue in old_data["issues"]:
                issue["is_persistent"] = False
        return old_data


class IssueRegistry(BaseRegistry):
    """Class to hold a registry of issues."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the issue registry."""
        self.hass: HomeAssistant = hass
        self.issues: Dict[Tuple[str, str], IssueEntry] = {}
        self._store: IssueRegistryStore = IssueRegistryStore(
            hass,
            STORAGE_VERSION_MAJOR,
            STORAGE_KEY,
            atomic_writes=True,
            minor_version=STORAGE_VERSION_MINOR,
        )

    @callback
    def async_get_issue(self, domain: str, issue_id: str) -> Optional[IssueEntry]:
        """Get issue by id."""
        return self.issues.get((domain, issue_id))

    @callback
    def async_get_or_create(
        self,
        domain: str,
        issue_id: str,
        *,
        breaks_in_ha_version: Optional[str] = None,
        data: Optional[Dict[str, Union[str, int, float, None]]] = None,
        is_fixable: bool,
        is_persistent: bool,
        issue_domain: Optional[str] = None,
        learn_more_url: Optional[str] = None,
        severity: IssueSeverity,
        translation_key: str,
        translation_placeholders: Optional[Dict[str, str]] = None,
    ) -> IssueEntry:
        """Get issue. Create if it doesn't exist."""
        self.hass.verify_event_loop_thread("issue_registry.async_get_or_create")
        existing_issue: Optional[IssueEntry] = self.async_get_issue(domain, issue_id)
        if existing_issue is None:
            issue = IssueEntry(
                active=True,
                breaks_in_ha_version=breaks_in_ha_version,
                created=dt_util.utcnow(),
                data=data,
                dismissed_version=None,
                domain=domain,
                is_fixable=is_fixable,
                is_persistent=is_persistent,
                issue_domain=issue_domain,
                issue_id=issue_id,
                learn_more_url=learn_more_url,
                severity=severity,
                translation_key=translation_key,
                translation_placeholders=translation_placeholders,
            )
            self.issues[(domain, issue_id)] = issue
            self.async_schedule_save()
            self.hass.bus.async_fire_internal(
                EVENT_REPAIRS_ISSUE_REGISTRY_UPDATED,
                EventIssueRegistryUpdatedData(
                    action="create",
                    domain=domain,
                    issue_id=issue_id,
                ),
            )
        else:
            replacement: IssueEntry = dataclasses.replace(
                existing_issue,
                active=True,
                breaks_in_ha_version=breaks_in_ha_version,
                data=data,
                is_fixable=is_fixable,
                is_persistent=is_persistent,
                issue_domain=issue_domain,
                learn_more_url=learn_more_url,
                severity=severity,
                translation_key=translation_key,
                translation_placeholders=translation_placeholders,
            )
            # Only fire if something changed
            if replacement != existing_issue:
                self.issues[(domain, issue_id)] = replacement
                self.async_schedule_save()
                self.hass.bus.async_fire_internal(
                    EVENT_REPAIRS_ISSUE_REGISTRY_UPDATED,
                    EventIssueRegistryUpdatedData(
                        action="update",
                        domain=domain,
                        issue_id=issue_id,
                    ),
                )
            issue = replacement

        return issue

    @callback
    def async_delete(self, domain: str, issue_id: str) -> None:
        """Delete issue."""
        self.hass.verify_event_loop_thread("issue_registry.async_delete")
        if self.issues.pop((domain, issue_id), None) is None:
            return

        self.async_schedule_save()
        self.hass.bus.async_fire_internal(
            EVENT_REPAIRS_ISSUE_REGISTRY_UPDATED,
            EventIssueRegistryUpdatedData(
                action="remove",
                domain=domain,
                issue_id=issue_id,
            ),
        )

    @callback
    def async_ignore(
        self, domain: str, issue_id: str, ignore: bool
    ) -> IssueEntry:
        """Ignore issue."""
        self.hass.verify_event_loop_thread("issue_registry.async_ignore")
        old: IssueEntry = self.issues[(domain, issue_id)]
        dismissed_version: Optional[str] = ha_version if ignore else None
        if old.dismissed_version == dismissed_version:
            return old

        issue: IssueEntry = dataclasses.replace(
            old,
            dismissed_version=dismissed_version,
        )
        self.issues[(domain, issue_id)] = issue

        self.async_schedule_save()
        self.hass.bus.async_fire_internal(
            EVENT_REPAIRS_ISSUE_REGISTRY_UPDATED,
            EventIssueRegistryUpdatedData(
                action="update",
                domain=domain,
                issue_id=issue_id,
            ),
        )

        return issue

    @callback
    def make_read_only(self) -> None:
        """Make the registry read-only.

        This method is irreversible.
        """
        self._store.make_read_only()

    async def async_load(self) -> None:
        """Load the issue registry."""
        data: Optional[Dict[str, Any]] = await self._store.async_load()

        issues: Dict[Tuple[str, str], IssueEntry] = {}

        if isinstance(data, dict):
            for issue in data.get("issues", []):
                created: Optional[datetime] = dt_util.parse_datetime(issue["created"])
                if created is None:
                    continue  # Skip invalid entries
                if issue.get("is_persistent", False):
                    issues[(issue["domain"], issue["issue_id"])] = IssueEntry(
                        active=True,
                        breaks_in_ha_version=issue.get("breaks_in_ha_version"),
                        created=created,
                        data=issue.get("data"),
                        dismissed_version=issue.get("dismissed_version"),
                        domain=issue["domain"],
                        is_fixable=issue.get("is_fixable"),
                        is_persistent=issue["is_persistent"],
                        issue_id=issue["issue_id"],
                        issue_domain=issue.get("issue_domain"),
                        learn_more_url=issue.get("learn_more_url"),
                        severity=IssueSeverity(issue["severity"]) if issue.get("severity") else None,
                        translation_key=issue.get("translation_key"),
                        translation_placeholders=issue.get("translation_placeholders"),
                    )
                else:
                    issues[(issue["domain"], issue["issue_id"])] = IssueEntry(
                        active=False,
                        breaks_in_ha_version=None,
                        created=created,
                        data=None,
                        dismissed_version=issue.get("dismissed_version"),
                        domain=issue["domain"],
                        is_fixable=None,
                        is_persistent=issue["is_persistent"],
                        issue_id=issue["issue_id"],
                        issue_domain=None,
                        learn_more_url=None,
                        severity=None,
                        translation_key=None,
                        translation_placeholders=None,
                    )

        self.issues = issues

    @callback
    def _data_to_save(self) -> Dict[str, List[Dict[str, Union[str, None]]]]:
        """Return data of issue registry to store in a file."""
        data: Dict[str, List[Dict[str, Any]]] = {}

        data["issues"] = [entry.to_json() for entry in self.issues.values()]

        return data


@callback
@singleton(DATA_REGISTRY)
def async_get(hass: HomeAssistant) -> IssueRegistry:
    """Get issue registry."""
    return IssueRegistry(hass)


async def async_load(hass: HomeAssistant, *, read_only: bool = False) -> None:
    """Load issue registry."""
    ir: IssueRegistry = async_get(hass)
    if read_only:  # only used in for check config script
        ir.make_read_only()
    return await ir.async_load()


@callback
def async_create_issue(
    hass: HomeAssistant,
    domain: str,
    issue_id: str,
    *,
    breaks_in_ha_version: Optional[str] = None,
    data: Optional[Dict[str, Union[str, int, float, None]]] = None,
    is_fixable: bool,
    is_persistent: bool = False,
    issue_domain: Optional[str] = None,
    learn_more_url: Optional[str] = None,
    severity: IssueSeverity,
    translation_key: str,
    translation_placeholders: Optional[Dict[str, str]] = None,
) -> None:
    """Create an issue, or replace an existing one."""
    # Verify the breaks_in_ha_version is a valid version string
    if breaks_in_ha_version:
        AwesomeVersion(
            breaks_in_ha_version,
            ensure_strategy=AwesomeVersionStrategy.CALVER,
        )

    issue_registry: IssueRegistry = async_get(hass)
    issue_registry.async_get_or_create(
        domain,
        issue_id,
        breaks_in_ha_version=breaks_in_ha_version,
        data=data,
        is_fixable=is_fixable,
        is_persistent=is_persistent,
        issue_domain=issue_domain,
        learn_more_url=learn_more_url,
        severity=severity,
        translation_key=translation_key,
        translation_placeholders=translation_placeholders,
    )


def create_issue(
    hass: HomeAssistant,
    domain: str,
    issue_id: str,
    *,
    breaks_in_ha_version: Optional[str] = None,
    data: Optional[Dict[str, Union[str, int, float, None]]] = None,
    is_fixable: bool,
    is_persistent: bool = False,
    issue_domain: Optional[str] = None,
    learn_more_url: Optional[str] = None,
    severity: IssueSeverity,
    translation_key: str,
    translation_placeholders: Optional[Dict[str, str]] = None,
) -> None:
    """Create an issue, or replace an existing one."""
    return run_callback_threadsafe(
        hass.loop,
        ft.partial(
            async_create_issue,
            hass,
            domain,
            issue_id,
            breaks_in_ha_version=breaks_in_ha_version,
            data=data,
            is_fixable=is_fixable,
            is_persistent=is_persistent,
            issue_domain=issue_domain,
            learn_more_url=learn_more_url,
            severity=severity,
            translation_key=translation_key,
            translation_placeholders=translation_placeholders,
        ),
    ).result()


@callback
def async_delete_issue(hass: HomeAssistant, domain: str, issue_id: str) -> None:
    """Delete an issue.

    It is not an error to delete an issue that does not exist.
    """
    issue_registry: IssueRegistry = async_get(hass)
    issue_registry.async_delete(domain, issue_id)


def delete_issue(hass: HomeAssistant, domain: str, issue_id: str) -> None:
    """Delete an issue.

    It is not an error to delete an issue that does not exist.
    """
    return run_callback_threadsafe(
        hass.loop, async_delete_issue, hass, domain, issue_id
    ).result()


@callback
def async_ignore_issue(
    hass: HomeAssistant, domain: str, issue_id: str, ignore: bool
) -> None:
    """Ignore an issue.

    Will raise if the issue does not exist.
    """
    issue_registry: IssueRegistry = async_get(hass)
    issue_registry.async_ignore(domain, issue_id, ignore)
