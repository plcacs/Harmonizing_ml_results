from __future__ import annotations
from contextlib import suppress
import dataclasses
from enum import IntFlag
import json
import os
from pathlib import Path
from typing import Any
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
    def __repr__(self) -> str:
        return '<ANY>'
ANY: _ANY = _ANY()
__all__: list[str] = ['HomeAssistantSnapshotExtension']

class AreaRegistryEntrySnapshot(dict):
    pass

class ConfigEntrySnapshot(dict):
    pass

class DeviceRegistryEntrySnapshot(dict):
    pass

class EntityRegistryEntrySnapshot(dict):
    pass

class FlowResultSnapshot(dict):
    pass

class IssueRegistryItemSnapshot(dict):
    pass

class StateSnapshot(dict):
    pass

class HomeAssistantSnapshotSerializer(AmberDataSerializer):
    @classmethod
    def _serialize(cls, data: Any, *, depth: int = 0, exclude: Any = None, include: Any = None, matcher: Any = None, path: Any = (), visited: Any = None) -> Any:
        pass

    @classmethod
    def _serializable_area_registry_entry(cls, data: Any) -> Any:
        pass

    @classmethod
    def _serializable_config_entry(cls, data: Any) -> Any:
        pass

    @classmethod
    def _serializable_device_registry_entry(cls, data: Any) -> Any:
        pass

    @classmethod
    def _remove_created_and_modified_at(cls, data: Any) -> Any:
        pass

    @classmethod
    def _serializable_entity_registry_entry(cls, data: Any) -> Any:
        pass

    @classmethod
    def _serializable_flow_result(cls, data: Any) -> Any:
        pass

    @classmethod
    def _serializable_conversation_result(cls, data: Any) -> Any:
        pass

    @classmethod
    def _serializable_issue_registry_entry(cls, data: Any) -> Any:
        pass

    @classmethod
    def _serializable_state(cls, data: Any) -> Any:
        pass

class _IntFlagWrapper:
    def __init__(self, flag: IntFlag) -> None:
        pass

    def __repr__(self) -> str:
        pass

class HomeAssistantSnapshotExtension(AmberSnapshotExtension):
    VERSION: str = '1'
    serializer_class: Any = HomeAssistantSnapshotSerializer

    @classmethod
    def dirname(cls, *, test_location: PyTestLocation) -> str:
        pass

class _FakePytestObject:
    def __init__(self, collected_item: dict) -> None:
        pass

class _FakePytestItem:
    def __init__(self, collected_item: dict) -> None:
        pass

def _serialize_collections(collections: SnapshotCollections) -> dict:
    pass

def _serialize_report(report: SnapshotReport, collected_items: Any, selected_items: Any) -> dict:
    pass

def _merge_serialized_collections(collections: SnapshotCollections, json_data: dict) -> None:
    pass

def _merge_serialized_report(report: SnapshotReport, json_data: dict) -> None:
    pass

def override_syrupy_finish(self: SnapshotSession) -> int:
    pass
