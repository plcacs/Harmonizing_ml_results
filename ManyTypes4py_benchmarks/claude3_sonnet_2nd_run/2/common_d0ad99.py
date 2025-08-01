"""Provide common test tools."""
from __future__ import annotations
from functools import cache
import json
from typing import Any, Dict, List, Optional, Union, Callable
from unittest.mock import MagicMock
from matter_server.client.models.node import MatterNode
from matter_server.common.helpers.util import dataclass_from_dict
from matter_server.common.models import EventType, MatterNodeData
from syrupy import SnapshotAssertion
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from tests.common import MockConfigEntry, load_fixture

@cache
def load_node_fixture(fixture: str) -> str:
    """Load a fixture."""
    return load_fixture(f'matter/nodes/{fixture}.json')

def load_and_parse_node_fixture(fixture: str) -> Dict[str, Any]:
    """Load and parse a node fixture."""
    return json.loads(load_node_fixture(fixture))

async def setup_integration_with_node_fixture(
    hass: HomeAssistant, 
    node_fixture: str, 
    client: MagicMock, 
    override_attributes: Optional[Dict[str, Any]] = None
) -> MatterNode:
    """Set up Matter integration with fixture as node."""
    node = create_node_from_fixture(node_fixture, override_attributes)
    client.get_nodes.return_value = [node]
    client.get_node.return_value = node
    config_entry = MockConfigEntry(domain='matter', data={'url': 'http://mock-matter-server-url'})
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    return node

def create_node_from_fixture(
    node_fixture: str, 
    override_attributes: Optional[Dict[str, Any]] = None
) -> MatterNode:
    """Create a node from a fixture."""
    node_data = load_and_parse_node_fixture(node_fixture)
    if override_attributes:
        node_data['attributes'].update(override_attributes)
    return MatterNode(dataclass_from_dict(MatterNodeData, node_data))

def set_node_attribute(
    node: MatterNode, 
    endpoint: int, 
    cluster_id: int, 
    attribute_id: int, 
    value: Any
) -> None:
    """Set a node attribute."""
    attribute_path = f'{endpoint}/{cluster_id}/{attribute_id}'
    node.endpoints[endpoint].set_attribute_value(attribute_path, value)

async def trigger_subscription_callback(
    hass: HomeAssistant, 
    client: MagicMock, 
    event: EventType = EventType.ATTRIBUTE_UPDATED, 
    data: Optional[Any] = None
) -> None:
    """Trigger a subscription callback."""
    for sub in client.subscribe_events.call_args_list:
        callback: Callable[[EventType, Any], None] = sub.kwargs['callback']
        event_filter: Optional[EventType] = sub.kwargs.get('event_filter')
        if event_filter in (None, event):
            callback(event, data)
    await hass.async_block_till_done()

def snapshot_matter_entities(
    hass: HomeAssistant, 
    entity_registry: er.EntityRegistry, 
    snapshot: SnapshotAssertion, 
    platform: Platform
) -> None:
    """Snapshot Matter entities."""
    entities = hass.states.async_all(platform)
    for entity_state in entities:
        entity_entry = entity_registry.async_get(entity_state.entity_id)
        assert entity_entry == snapshot(name=f'{entity_entry.entity_id}-entry')
        assert entity_state == snapshot(name=f'{entity_entry.entity_id}-state')
