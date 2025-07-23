"""The tests for the MQTT client."""
import asyncio
from datetime import timedelta
import socket
import ssl
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, call, patch
import certifi
import paho.mqtt.client as paho_mqtt
import pytest
from homeassistant.components import mqtt
from homeassistant.components.mqtt.client import RECONNECT_INTERVAL_SECONDS
from homeassistant.components.mqtt.const import SUPPORTED_COMPONENTS
from homeassistant.components.mqtt.models import MessageCallbackType, ReceiveMessage
from homeassistant.config_entries import ConfigEntryDisabler, ConfigEntryState
from homeassistant.const import CONF_PROTOCOL, EVENT_HOMEASSISTANT_STARTED, EVENT_HOMEASSISTANT_STOP, UnitOfTemperature
from homeassistant.core import CALLBACK_TYPE, CoreState, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util.dt import utcnow
from .conftest import ENTRY_DEFAULT_BIRTH_MESSAGE
from .test_common import help_all_subscribe_calls
from tests.common import MockConfigEntry, MockMqttReasonCode, async_fire_mqtt_message, async_fire_time_changed
from tests.typing import MqttMockHAClient, MqttMockHAClientGenerator, MqttMockPahoClient

def help_assert_message(msg: ReceiveMessage, topic: Optional[str] = None, payload: Optional[Union[str, bytes]] = None, qos: Optional[int] = None, retain: Optional[bool] = None) -> bool:
    """Return True if all of the given attributes match with the message."""
    match = True
    if topic is not None:
        match &= msg.topic == topic
    if payload is not None:
        match &= msg.payload == payload
    if qos is not None:
        match &= msg.qos == qos
    if retain is not None:
        match &= msg.retain == retain
    return match

async def test_mqtt_connects_on_home_assistant_mqtt_setup(hass: HomeAssistant, setup_with_birth_msg_client_mock: MagicMock) -> None:
    """Test if client is connected after mqtt init on bootstrap."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    assert mqtt_client_mock.connect.call_count == 1

async def test_mqtt_does_not_disconnect_on_home_assistant_stop(hass: HomeAssistant, mock_debouncer: MagicMock, setup_with_birth_msg_client_mock: MagicMock) -> None:
    """Test if client is not disconnected on HA stop."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    hass.bus.fire(EVENT_HOMEASSISTANT_STOP)
    await mock_debouncer.wait()
    assert mqtt_client_mock.disconnect.call_count == 0

async def test_mqtt_await_ack_at_disconnect(hass: HomeAssistant) -> None:
    """Test if ACK is awaited correctly when disconnecting."""

    class FakeInfo:
        """Returns a simulated client publish response."""
        mid: int = 100
        rc: int = 0
        
    with patch('homeassistant.components.mqtt.async_client.AsyncMQTTClient') as mock_client:
        mqtt_client = mock_client.return_value
        mqtt_client.connect = MagicMock(return_value=0, side_effect=lambda *args, **kwargs: hass.loop.call_soon_threadsafe(mqtt_client.on_connect, mqtt_client, None, 0, MockMqttReasonCode()))
        mqtt_client.publish = MagicMock(return_value=FakeInfo())
        entry = MockConfigEntry(domain=mqtt.DOMAIN, data={'certificate': 'auto', mqtt.CONF_BROKER: 'test-broker', mqtt.CONF_DISCOVERY: False}, version=mqtt.CONFIG_ENTRY_VERSION, minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION)
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        mqtt_client = mock_client.return_value
        hass.async_create_task(mqtt.async_publish(hass, 'test-topic', 'some-payload', 0, False))
        await asyncio.sleep(0)
        mqtt_client.on_publish(0, 0, 100, MockMqttReasonCode(), None)
        await hass.async_stop()
        await hass.async_block_till_done()
        assert mqtt_client.publish.called
        assert mqtt_client.publish.call_args[0] == ('test-topic', 'some-payload', 0, False)
        await hass.async_block_till_done(wait_background_tasks=True)

@pytest.mark.parametrize('mqtt_config_entry_options', [ENTRY_DEFAULT_BIRTH_MESSAGE])
async def test_publish(hass: HomeAssistant, setup_with_birth_msg_client_mock: MagicMock) -> None:
    """Test the publish function."""
    publish_mock = setup_with_birth_msg_client_mock.publish
    await mqtt.async_publish(hass, 'test-topic', 'test-payload')
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ('test-topic', 'test-payload', 0, False)
    publish_mock.reset_mock()
    await mqtt.async_publish(hass, 'test-topic', 'test-payload', 2, True)
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ('test-topic', 'test-payload', 2, True)
    publish_mock.reset_mock()
    mqtt.publish(hass, 'test-topic2', 'test-payload2')
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ('test-topic2', 'test-payload2', 0, False)
    publish_mock.reset_mock()
    mqtt.publish(hass, 'test-topic2', 'test-payload2', 2, True)
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ('test-topic2', 'test-payload2', 2, True)
    publish_mock.reset_mock()
    mqtt.publish(hass, 'test-topic3', b'\xde\xad\xbe\xef', 0, False)
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ('test-topic3', b'\xde\xad\xbe\xef', 0, False)
    publish_mock.reset_mock()
    mqtt.publish(hass, 'test-topic3', None, 0, False)
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ('test-topic3', None, 0, False)
    publish_mock.reset_mock()

async def test_convert_outgoing_payload(hass: HomeAssistant) -> None:
    """Test the converting of outgoing MQTT payloads without template."""
    command_template = mqtt.MqttCommandTemplate(None)
    assert command_template.async_render(b'\xde\xad\xbe\xef') == b'\xde\xad\xbe\xef'
    assert command_template.async_render("b'\\xde\\xad\\xbe\\xef'") == "b'\\xde\\xad\\xbe\\xef'"
    assert command_template.async_render(1234) == 1234
    assert command_template.async_render(1234.56) == 1234.56
    assert command_template.async_render(None) is None

async def test_all_subscriptions_run_when_decode_fails(hass: HomeAssistant, mqtt_mock_entry: Callable, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test all other subscriptions still run when decode fails for one."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic', record_calls, encoding='ascii')
    await mqtt.async_subscribe(hass, 'test-topic', record_calls)
    async_fire_mqtt_message(hass, 'test-topic', UnitOfTemperature.CELSIUS)
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1

async def test_subscribe_topic(hass: HomeAssistant, mqtt_mock_entry: Callable, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of a topic."""
    await mqtt_mock_entry()
    unsub = await mqtt.async_subscribe(hass, 'test-topic', record_calls)
    async_fire_mqtt_message(hass, 'test-topic', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == 'test-topic'
    assert recorded_calls[0].payload == 'test-payload'
    unsub()
    async_fire_mqtt_message(hass, 'test-topic', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    with pytest.raises(HomeAssistantError):
        unsub()

@pytest.mark.usefixtures('mqtt_mock_entry')
async def test_subscribe_topic_not_initialize(hass: HomeAssistant, record_calls: MessageCallbackType) -> None:
    """Test the subscription of a topic when MQTT was not initialized."""
    with pytest.raises(HomeAssistantError, match='.*make sure MQTT is set up correctly'):
        await mqtt.async_subscribe(hass, 'test-topic', record_calls)

async def test_subscribe_mqtt_config_entry_disabled(hass: HomeAssistant, mqtt_mock: MagicMock, record_calls: MessageCallbackType) -> None:
    """Test the subscription of a topic when MQTT config entry is disabled."""
    mqtt_mock.connected = True
    mqtt_config_entry = hass.config_entries.async_entries(mqtt.DOMAIN)[0]
    mqtt_config_entry_state = mqtt_config_entry.state
    assert mqtt_config_entry_state is ConfigEntryState.LOADED
    assert await hass.config_entries.async_unload(mqtt_config_entry.entry_id)
    mqtt_config_entry_state = mqtt_config_entry.state
    assert mqtt_config_entry_state is ConfigEntryState.NOT_LOADED
    await hass.config_entries.async_set_disabled_by(mqtt_config_entry.entry_id, ConfigEntryDisabler.USER)
    mqtt_mock.connected = False
    with pytest.raises(HomeAssistantError, match='.*MQTT is not enabled'):
        await mqtt.async_subscribe(hass, 'test-topic', record_calls)

async def test_subscribe_and_resubscribe(hass: HomeAssistant, mock_debouncer: MagicMock, setup_with_birth_msg_client_mock: MagicMock, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test resubscribing within the debounce time."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    with patch('homeassistant.components.mqtt.client.SUBSCRIBE_COOLDOWN', 0.4), patch('homeassistant.components.mqtt.client.UNSUBSCRIBE_COOLDOWN', 0.4):
        mock_debouncer.clear()
        unsub = await mqtt.async_subscribe(hass, 'test-topic', record_calls)
        unsub()
        unsub = await mqtt.async_subscribe(hass, 'test-topic', record_calls)
        await mock_debouncer.wait()
        mock_debouncer.clear()
        async_fire_mqtt_message(hass, 'test-topic', 'test-payload')
        assert len(recorded_calls) == 1
        assert recorded_calls[0].topic == 'test-topic'
        assert recorded_calls[0].payload == 'test-payload'
        mqtt_client_mock.unsubscribe.assert_not_called()
        mock_debouncer.clear()
        unsub()
        await mock_debouncer.wait()
        mqtt_client_mock.unsubscribe.assert_called_once_with(['test-topic'])

async def test_subscribe_topic_non_async(hass: HomeAssistant, mock_debouncer: MagicMock, mqtt_mock_entry: Callable, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of a topic using the non-async function."""
    await mqtt_mock_entry()
    await mock_debouncer.wait()
    mock_debouncer.clear()
    unsub = await hass.async_add_executor_job(mqtt.subscribe, hass, 'test-topic', record_calls)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, 'test-topic', 'test-payload')
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == 'test-topic'
    assert recorded_calls[0].payload == 'test-payload'
    mock_debouncer.clear()
    await hass.async_add_executor_job(unsub)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, 'test-topic', 'test-payload')
    assert len(recorded_calls) == 1

async def test_subscribe_bad_topic(hass: HomeAssistant, mqtt_mock_entry: Callable, record_calls: MessageCallbackType) -> None:
    """Test the subscription of a topic."""
    await mqtt_mock_entry()
    with pytest.raises(HomeAssistantError):
        await mqtt.async_subscribe(hass, 55, record_calls)

async def test_subscribe_topic_not_match(hass: HomeAssistant, mqtt_mock_entry: Callable, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test if subscribed topic is not a match."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic', record_calls)
    async_fire_mqtt_message(hass, 'another-test-topic', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0

async def test_subscribe_topic_level_wildcard(hass: HomeAssistant, mqtt_mock_entry: Callable, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic/+/on', record_calls)
    async_fire_mqtt_message(hass, 'test-topic/bier/on', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == 'test-topic/bier/on'
    assert recorded_calls[0].payload == 'test-payload'

async def test_subscribe_topic_level_wildcard_no_subtree_match(hass: HomeAssistant, mqtt_mock_entry: Callable, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic/+/on', record_calls)
    async_fire_mqtt_message(hass, 'test-topic/bier', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0

async def test_subscribe_topic_level_wildcard_root_topic_no_subtree_match(hass: HomeAssistant, mqtt_mock_entry: Callable, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic/#', record_calls)
    async_fire_mqtt_message(hass, 'test-topic-123', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0

async def test_subscribe_topic_subtree_wildcard_subtree_topic(hass: HomeAssistant, mqtt_mock_entry: Callable, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic/#', record_calls)
    async_fire_mqtt_message(hass, 'test-topic/bier/on', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == 'test-topic/bier/on'
    assert recorded_calls[0].payload == 'test-payload'

async def test_subscribe_topic_subtree_wildcard_root_topic(hass: HomeAssistant, mqtt_mock_entry: Callable, recorded_calls: List[ReceiveMessage], record_calls: Message