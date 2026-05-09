from __future__ import annotations
from contextlib import suppress
import dataclasses
from enum import IntFlag
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import attr
import attrs
import pytest
from syrupy.constants import EXIT_STATUS_FAIL_UNUSED
from syrupy.data import Snapshot, SnapshotCollection, SnapshotCollections
from syrupy.extensions.amber import AmberDataSerializer, AmberSnapshotExtension
from syrupy.location import PyTestLocation
from syrupy.report import SnapshotReport
from syrupy.session import ItemStatus, SnapshotSession
from syrupy.types import PropertyFilter, PropertyMatcher, PropertyPath, SerializableData
from syrupy.utils import is_xdist_controller, is_xdist_worker
import voluptuous as vol
import voluptuous_serialize
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import State
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import area_registry as ar, device_registry as dr, entity_registry as er, issue_registry as ir

class _ANY:
    """Represent any value."""

    def __repr__(self):
        return '<ANY>'
ANY: Any = _ANY()
__all__: List[str] = ['HomeAssistantSnapshotExtension']

class AreaRegistryEntrySnapshot(Dict):
    """Tiny wrapper to represent an area registry entry in snapshots."""

class ConfigEntrySnapshot(Dict):
    """Tiny wrapper to represent a config entry in snapshots."""

class DeviceRegistryEntrySnapshot(Dict):
    """Tiny wrapper to represent a device registry entry in snapshots."""

class EntityRegistryEntrySnapshot(Dict):
    """Tiny wrapper to represent an entity registry entry in snapshots."""

class FlowResultSnapshot(Dict):
    """Tiny wrapper to represent a flow result in snapshots."""

class IssueRegistryItemSnapshot(Dict):
    """Tiny wrapper to represent an entity registry entry in snapshots."""

class StateSnapshot(Dict):
    """Tiny wrapper to represent an entity state in snapshots."""

class HomeAssistantSnapshotSerializer(AmberDataSerializer):
    """Home Assistant snapshot serializer for Syrupy.

    Handles special cases for Home Assistant data structures.
    """

    @classmethod
    def _serialize(cls, data: Any, *, depth: int, exclude: Dict[str, Any], include: Dict[str, Any], matcher: Any, path: Tuple[str, ...], visited: Dict[str, ...]) -> Any:
        ...

    @classmethod
    def _serializable_area_registry_entry(cls, data: ar.AreaEntry) -> Dict[str, Any]:
        ...

    @classmethod
    def _serializable_config_entry(cls, data: ConfigEntry) -> Dict[str, Any]:
        ...

    @classmethod
    def _serializable_device_registry_entry(cls, data: dr.DeviceEntry) -> Dict[str, Any]:
        ...

    @classmethod
    def _serializable_entity_registry_entry(cls, data: er.RegistryEntry) -> Dict[str, Any]:
        ...

    @classmethod
    def _serializable_flow_result(cls, data: FlowResult) -> Dict[str, Any]:
        ...

    @classmethod
    def _serializable_conversation_result(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @classmethod
    def _serializable_issue_registry_entry(cls, data: ir.IssueEntry) -> Dict[str, Any]:
        ...

    @classmethod
    def _serializable_state(cls, data: State) -> Dict[str, Any]:
        ...

class _IntFlagWrapper:
    """Wrapper for IntFlag."""

    def __init__(self, flag: IntFlag):
        self._flag: IntFlag = flag

    def __repr__(self) -> str:
        return f'<{self._flag.__class__.__name__}: {self._flag.value}>'

class HomeAssistantSnapshotExtension(AmberSnapshotExtension):
    """Home Assistant extension for Syrupy."""
    VERSION: str = '1'
    'Current version of serialization format.\n\n    Need to be bumped when we change the HomeAssistantSnapshotSerializer.\n    '
    serializer_class: type[HomeAssistantSnapshotSerializer] = HomeAssistantSnapshotSerializer

    @classmethod
    def dirname(cls, *, test_location: PyTestLocation) -> str:
        """Return the directory for the snapshot files."""
        test_dir: Path = Path(test_location.filepath).parent
        return str(test_dir.joinpath('snapshots'))

class _FakePytestObject:
    """Fake object."""

    def __init__(self, collected_item: Dict[str, ...]):
        """Initialise fake object."""
        self.__module__: str = collected_item['modulename']
        self.__name__: str = collected_item['methodname']

class _FakePytestItem:
    """Fake pytest.Item object."""

    def __init__(self, collected_item: Dict[str, ...]):
        """Initialise fake pytest.Item object."""
        self.nodeid: str = collected_item['nodeid']
        self.name: str = collected_item['name']
        self.path: Path = Path(collected_item['path'])
        self.obj: _FakePytestObject = _FakePytestObject(collected_item)

def _serialize_collections(collections: SnapshotCollections) -> Dict[str, List[str]]:
    ...

def _serialize_report(report: SnapshotReport, collected_items: List[_FakePytestItem], selected_items: Dict[str, ItemStatus]) -> Dict[str, Any]:
    ...

def _merge_serialized_collections(collections: SnapshotCollections, json_data: Dict[str, List[str]]) -> None:
    ...

def _merge_serialized_report(report: SnapshotReport, json_data: Dict[str, Any]) -> None:
    ...

def override_syrupy_finish(self) -> int:
    """Override the finish method to allow for custom handling."""
    exitstatus: int = 0
    self.flush_snapshot_write_queue()
    self.report: SnapshotReport = SnapshotReport(base_dir=self.pytest_session.config.rootpath, collected_items=self._collected_items, selected_items=self._selected_items, assertions=self._assertions, options=self.pytest_session.config.option)
    needs_xdist_merge: bool = self.update_snapshots or bool(self.pytest_session.config.option.include_snapshot_details)
    if is_xdist_worker():
        ...
    elif is_xdist_controller():
        ...
    elif needs_xdist_merge:
        ...
    if self.report.num_unused:
        ...
    return exitstatus
