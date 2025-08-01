"""Home Assistant extension for Syrupy."""
from __future__ import annotations
from contextlib import suppress
import dataclasses
from enum import IntFlag
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
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
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    issue_registry as ir,
)


class _ANY:
    """Represent any value."""

    def __repr__(self) -> str:
        return '<ANY>'


ANY: _ANY = _ANY()
__all__ = ['HomeAssistantSnapshotExtension']


class AreaRegistryEntrySnapshot(dict):
    """Tiny wrapper to represent an area registry entry in snapshots."""


class ConfigEntrySnapshot(dict):
    """Tiny wrapper to represent a config entry in snapshots."""


class DeviceRegistryEntrySnapshot(dict):
    """Tiny wrapper to represent a device registry entry in snapshots."""


class EntityRegistryEntrySnapshot(dict):
    """Tiny wrapper to represent an entity registry entry in snapshots."""


class FlowResultSnapshot(dict):
    """Tiny wrapper to represent a flow result in snapshots."""


class IssueRegistryItemSnapshot(dict):
    """Tiny wrapper to represent an entity registry entry in snapshots."""


class StateSnapshot(dict):
    """Tiny wrapper to represent an entity state in snapshots."""


class HomeAssistantSnapshotSerializer(AmberDataSerializer):
    """Home Assistant snapshot serializer for Syrupy.

    Handles special cases for Home Assistant data structures.
    """

    @classmethod
    def _serialize(
        cls,
        data: Any,
        *,
        depth: int = 0,
        exclude: Optional[PropertyFilter] = None,
        include: Optional[PropertyFilter] = None,
        matcher: Optional[PropertyMatcher] = None,
        path: PropertyPath = (),
        visited: Optional[Set[int]] = None,
    ) -> Any:
        """Pre-process data before serializing.

        This allows us to handle specific cases for Home Assistant data structures.
        """
        if isinstance(data, State):
            serializable_data = cls._serializable_state(data)
        elif isinstance(data, ar.AreaEntry):
            serializable_data = cls._serializable_area_registry_entry(data)
        elif isinstance(data, dr.DeviceEntry):
            serializable_data = cls._serializable_device_registry_entry(data)
        elif isinstance(data, er.RegistryEntry):
            serializable_data = cls._serializable_entity_registry_entry(data)
        elif isinstance(data, ir.IssueEntry):
            serializable_data = cls._serializable_issue_registry_entry(data)
        elif isinstance(data, dict) and 'flow_id' in data and ('handler' in data):
            serializable_data = cls._serializable_flow_result(data)
        elif isinstance(data, dict) and set(data) == {'conversation_id', 'response'}:
            serializable_data = cls._serializable_conversation_result(data)
        elif isinstance(data, vol.Schema):
            serializable_data = voluptuous_serialize.convert(data)
        elif isinstance(data, ConfigEntry):
            serializable_data = cls._serializable_config_entry(data)
        elif dataclasses.is_dataclass(type(data)):
            serializable_data = dataclasses.asdict(data)
        elif isinstance(data, IntFlag):
            serializable_data = _IntFlagWrapper(data)
        else:
            serializable_data = data
            with suppress(TypeError):
                if attr.has(type(data)):
                    serializable_data = attrs.asdict(data)
        return super()._serialize(
            serializable_data,
            depth=depth,
            exclude=exclude,
            include=include,
            matcher=matcher,
            path=path,
            visited=visited,
        )

    @classmethod
    def _serializable_area_registry_entry(cls, data: ar.AreaEntry) -> AreaRegistryEntrySnapshot:
        """Prepare a Home Assistant area registry entry for serialization."""
        serialized: AreaRegistryEntrySnapshot = AreaRegistryEntrySnapshot(
            dataclasses.asdict(data) | {'id': ANY}
        )
        serialized.pop('_json_repr', None)
        serialized.pop('_cache', None)
        return serialized

    @classmethod
    def _serializable_config_entry(cls, data: ConfigEntry) -> ConfigEntrySnapshot:
        """Prepare a Home Assistant config entry for serialization."""
        entry: ConfigEntrySnapshot = ConfigEntrySnapshot(data.as_dict() | {'entry_id': ANY})
        return cls._remove_created_and_modified_at(entry)

    @classmethod
    def _serializable_device_registry_entry(cls, data: dr.DeviceEntry) -> DeviceRegistryEntrySnapshot:
        """Prepare a Home Assistant device registry entry for serialization."""
        serialized: DeviceRegistryEntrySnapshot = DeviceRegistryEntrySnapshot(
            attrs.asdict(data)
            | {
                'config_entries': ANY,
                'config_entries_subentries': ANY,
                'id': ANY,
            }
        )
        if serialized.get('via_device_id') is not None:
            serialized['via_device_id'] = ANY
        if serialized.get('primary_config_entry') is not None:
            serialized['primary_config_entry'] = ANY
        serialized.pop('_cache', None)
        return cls._remove_created_and_modified_at(serialized)

    @classmethod
    def _remove_created_and_modified_at(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove created_at and modified_at from the data."""
        data.pop('created_at', None)
        data.pop('modified_at', None)
        return data

    @classmethod
    def _serializable_entity_registry_entry(cls, data: er.RegistryEntry) -> EntityRegistryEntrySnapshot:
        """Prepare a Home Assistant entity registry entry for serialization."""
        serialized: EntityRegistryEntrySnapshot = EntityRegistryEntrySnapshot(
            attrs.asdict(data)
            | {
                'config_entry_id': ANY,
                'config_subentry_id': ANY,
                'device_id': ANY,
                'id': ANY,
                'options': {k: dict(v) for k, v in data.options.items()},
            }
        )
        serialized.pop('categories', None)
        serialized.pop('_cache', None)
        return cls._remove_created_and_modified_at(serialized)

    @classmethod
    def _serializable_flow_result(cls, data: Dict[str, Any]) -> FlowResultSnapshot:
        """Prepare a Home Assistant flow result for serialization."""
        return FlowResultSnapshot(data | {'flow_id': ANY})

    @classmethod
    def _serializable_conversation_result(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a Home Assistant conversation result for serialization."""
        return data | {'conversation_id': ANY}

    @classmethod
    def _serializable_issue_registry_entry(cls, data: ir.IssueEntry) -> IssueRegistryItemSnapshot:
        """Prepare a Home Assistant issue registry entry for serialization."""
        return IssueRegistryItemSnapshot(dataclasses.asdict(data) | {'created': ANY})

    @classmethod
    def _serializable_state(cls, data: State) -> StateSnapshot:
        """Prepare a Home Assistant State for serialization."""
        return StateSnapshot(
            data.as_dict()
            | {
                'context': ANY,
                'last_changed': ANY,
                'last_reported': ANY,
                'last_updated': ANY,
            }
        )


class _IntFlagWrapper:
    def __init__(self, flag: IntFlag) -> None:
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
        """Return the directory for the snapshot files.

        Syrupy, by default, uses the `__snapshosts__` directory in the same
        folder as the test file. For Home Assistant, this is changed to just
        `snapshots` in the same folder as the test file, to match our `fixtures`
        folder structure.
        """
        test_dir: Path = Path(test_location.filepath).parent
        return str(test_dir.joinpath('snapshots'))


class _FakePytestObject:
    """Fake object."""

    def __init__(self, collected_item: Dict[str, Any]) -> None:
        """Initialize fake object."""
        self.__module__: str = collected_item['modulename']
        self.__name__: str = collected_item['methodname']


class _FakePytestItem:
    """Fake pytest.Item object."""

    def __init__(self, collected_item: Dict[str, Any]) -> None:
        """Initialize fake pytest.Item object."""
        self.nodeid: str = collected_item['nodeid']
        self.name: str = collected_item['name']
        self.path: Path = Path(collected_item['path'])
        self.obj: _FakePytestObject = _FakePytestObject(collected_item)


def _serialize_collections(collections: SnapshotCollections) -> Dict[str, List[str]]:
    return {
        k: [c.name for c in v]
        for k, v in collections._snapshot_collections.items()
    }


def _serialize_report(
    report: SnapshotReport,
    collected_items: Set[_FakePytestItem],
    selected_items: Dict[str, ItemStatus],
) -> Dict[str, Any]:
    return {
        'discovered': _serialize_collections(report.discovered),
        'created': _serialize_collections(report.created),
        'failed': _serialize_collections(report.failed),
        'matched': _serialize_collections(report.matched),
        'updated': _serialize_collections(report.updated),
        'used': _serialize_collections(report.used),
        '_collected_items': [
            {
                'nodeid': c.nodeid,
                'name': c.name,
                'path': str(c.path),
                'modulename': c.obj.__module__,
                'methodname': c.obj.__name__,
            }
            for c in list(collected_items)
        ],
        '_selected_items': {
            key: status.value
            for key, status in selected_items.items()
        },
    }


def _merge_serialized_collections(
    collections: SnapshotCollections,
    json_data: Optional[Dict[str, List[str]]],
) -> None:
    if not json_data:
        return
    for location, names in json_data.items():
        snapshot_collection = SnapshotCollection(location=location)
        for name in names:
            snapshot_collection.add(Snapshot(name))
        collections.update(snapshot_collection)


def _merge_serialized_report(report: SnapshotReport, json_data: Dict[str, Any]) -> None:
    _merge_serialized_collections(report.discovered, json_data.get('discovered'))
    _merge_serialized_collections(report.created, json_data.get('created'))
    _merge_serialized_collections(report.failed, json_data.get('failed'))
    _merge_serialized_collections(report.matched, json_data.get('matched'))
    _merge_serialized_collections(report.updated, json_data.get('updated'))
    _merge_serialized_collections(report.used, json_data.get('used'))
    for collected_item in json_data.get('_collected_items', []):
        custom_item = _FakePytestItem(collected_item)
        if not any(
            t.nodeid == custom_item.nodeid and t.name == custom_item.name
            for t in report.collected_items
        ):
            report.collected_items.add(custom_item)
    for key, selected_item in json_data.get('_selected_items', {}).items():
        if key in report.selected_items:
            status = ItemStatus(selected_item)
            if status != ItemStatus.NOT_RUN:
                report.selected_items[key] = status
        else:
            report.selected_items[key] = ItemStatus(selected_item)


def override_syrupy_finish(self: SnapshotSession) -> int:
    """Override the finish method to allow for custom handling."""
    exitstatus: int = 0
    self.flush_snapshot_write_queue()
    self.report = SnapshotReport(
        base_dir=self.pytest_session.config.rootpath,
        collected_items=self._collected_items,
        selected_items=self._selected_items,
        assertions=self._assertions,
        options=self.pytest_session.config.option,
    )
    needs_xdist_merge: bool = self.update_snapshots or bool(
        self.pytest_session.config.option.include_snapshot_details
    )
    if is_xdist_worker():
        if not needs_xdist_merge:
            return exitstatus
        with open('.pytest_syrupy_worker_count', 'w', encoding='utf-8') as f:
            worker_count = os.getenv('PYTEST_XDIST_WORKER_COUNT', '')
            f.write(worker_count)
        worker = os.getenv('PYTEST_XDIST_WORKER', '')
        with open(f'.pytest_syrupy_{worker}_result', 'w', encoding='utf-8') as f:
            json.dump(
                _serialize_report(self.report, self._collected_items, self._selected_items),
                f,
                indent=2,
            )
        return exitstatus
    if is_xdist_controller():
        return exitstatus
    if needs_xdist_merge:
        worker_count: Optional[str] = None
        try:
            with open('.pytest_syrupy_worker_count', encoding='utf-8') as f:
                worker_count = f.read()
            os.remove('.pytest_syrupy_worker_count')
        except FileNotFoundError:
            pass
        if worker_count:
            for i in range(int(worker_count)):
                worker_result_file: str = f'.pytest_syrupy_gw{i}_result'
                with open(worker_result_file, encoding='utf-8') as f:
                    worker_json = json.load(f)
                    _merge_serialized_report(self.report, worker_json)
                os.remove(worker_result_file)
    if self.report.num_unused:
        if self.update_snapshots:
            self.remove_unused_snapshots(
                unused_snapshot_collections=self.report.unused,
                used_snapshot_collections=self.report.used,
            )
        elif not self.warn_unused_snapshots:
            exitstatus |= EXIT_STATUS_FAIL_UNUSED
    return exitstatus
