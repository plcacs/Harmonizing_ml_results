"""The tests for the MQTT client."""
import asyncio
from datetime import timedelta
import socket
import ssl
import time
from typing import Any, Callable, List, Optional, Tuple, Union
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

def help_assert_message(msg, topic=None, payload=None, qos=None, retain=None):
    """Return True if all of the given attributes match with the message."""
    match: bool = True
    if topic is not None:
        match &= msg.topic == topic
    if payload is not None:
        match &= msg.payload == payload
    if qos is not None:
        match &= msg.qos == qos
    if retain is not None:
        match &= msg.retain == retain
    return match

async def test_mqtt_connects_on_home_assistant_mqtt_setup(hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test if client is connected after mqtt init on bootstrap."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    assert mqtt_client_mock.connect.call_count == 1

async def test_mqtt_does_not_disconnect_on_home_assistant_stop(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
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
async def test_publish(hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test the publish function."""
    publish_mock: MagicMock = setup_with_birth_msg_client_mock.publish
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

async def test_all_subscriptions_run_when_decode_fails(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test all other subscriptions still run when decode fails for one."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic', record_calls, encoding='ascii')
    await mqtt.async_subscribe(hass, 'test-topic', record_calls)
    async_fire_mqtt_message(hass, 'test-topic', UnitOfTemperature.CELSIUS)
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1

async def test_subscribe_topic(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
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

async def test_subscribe_mqtt_config_entry_disabled(hass: HomeAssistant, mqtt_mock: MqttMockHAClient, record_calls: MessageCallbackType) -> None:
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

async def test_subscribe_and_resubscribe(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
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

async def test_subscribe_topic_non_async(hass: HomeAssistant, mock_debouncer: asyncio.Event, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
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

async def test_subscribe_bad_topic(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, record_calls: MessageCallbackType) -> None:
    """Test the subscription of a topic."""
    await mqtt_mock_entry()
    with pytest.raises(HomeAssistantError):
        await mqtt.async_subscribe(hass, 55, record_calls)

async def test_subscribe_topic_not_match(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test if subscribed topic is not a match."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic', record_calls)
    async_fire_mqtt_message(hass, 'another-test-topic', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0

async def test_subscribe_topic_level_wildcard(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic/+/on', record_calls)
    async_fire_mqtt_message(hass, 'test-topic/bier/on', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == 'test-topic/bier/on'
    assert recorded_calls[0].payload == 'test-payload'

async def test_subscribe_topic_level_wildcard_no_subtree_match(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic/+/on', record_calls)
    async_fire_mqtt_message(hass, 'test-topic/bier', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0

async def test_subscribe_topic_level_wildcard_root_topic_no_subtree_match(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic/#', record_calls)
    async_fire_mqtt_message(hass, 'test-topic-123', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0

async def test_subscribe_topic_subtree_wildcard_subtree_topic(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic/#', record_calls)
    async_fire_mqtt_message(hass, 'test-topic/bier/on', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == 'test-topic/bier/on'
    assert recorded_calls[0].payload == 'test-payload'

async def test_subscribe_topic_subtree_wildcard_root_topic(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic/#', record_calls)
    async_fire_mqtt_message(hass, 'test-topic', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == 'test-topic'
    assert recorded_calls[0].payload == 'test-payload'

async def test_subscribe_topic_subtree_wildcard_no_match(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, 'test-topic/#', record_calls)
    async_fire_mqtt_message(hass, 'another-test-topic', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0

async def test_subscribe_topic_level_wildcard_and_wildcard_root_topic(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, '+/test-topic/#', record_calls)
    async_fire_mqtt_message(hass, 'hi/test-topic', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == 'hi/test-topic'
    assert recorded_calls[0].payload == 'test-payload'

async def test_subscribe_topic_level_wildcard_and_wildcard_subtree_topic(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, '+/test-topic/#', record_calls)
    async_fire_mqtt_message(hass, 'hi/test-topic/here-iam', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == 'hi/test-topic/here-iam'
    assert recorded_calls[0].payload == 'test-payload'

async def test_subscribe_topic_level_wildcard_and_wildcard_level_no_match(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, '+/test-topic/#', record_calls)
    async_fire_mqtt_message(hass, 'hi/here-iam/test-topic', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0

async def test_subscribe_topic_level_wildcard_and_wildcard_no_match(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, '+/test-topic/#', record_calls)
    async_fire_mqtt_message(hass, 'hi/another-test-topic', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0

async def test_subscribe_topic_sys_root(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of $ root topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, '$test-topic/subtree/on', record_calls)
    async_fire_mqtt_message(hass, '$test-topic/subtree/on', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == '$test-topic/subtree/on'
    assert recorded_calls[0].payload == 'test-payload'

async def test_subscribe_topic_sys_root_and_wildcard_topic(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of $ root and wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, '$test-topic/#', record_calls)
    async_fire_mqtt_message(hass, '$test-topic/some-topic', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == '$test-topic/some-topic'
    assert recorded_calls[0].payload == 'test-payload'

async def test_subscribe_topic_sys_root_and_wildcard_subtree_topic(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription of $ root and wildcard subtree topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, '$test-topic/subtree/#', record_calls)
    async_fire_mqtt_message(hass, '$test-topic/subtree/some-topic', 'test-payload')
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == '$test-topic/subtree/some-topic'
    assert recorded_calls[0].payload == 'test-payload'

async def test_subscribe_special_characters(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test the subscription to topics with special characters."""
    await mqtt_mock_entry()
    topic = '/test-topic/$(.)[^]{-}'
    payload = 'p4y.l[]a|> ?'
    await mqtt.async_subscribe(hass, topic, record_calls)
    async_fire_mqtt_message(hass, topic, payload)
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == topic
    assert recorded_calls[0].payload == payload

async def test_subscribe_same_topic(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test subscribing to same topic twice and simulate retained messages.

    When subscribing to the same topic again, SUBSCRIBE must be sent to the broker again
    for it to resend any retained messages.
    """
    mqtt_client_mock = setup_with_birth_msg_client_mock
    calls_a: List[ReceiveMessage] = []
    calls_b: List[ReceiveMessage] = []

    @callback
    def _callback_a(msg):
        calls_a.append(msg)

    @callback
    def _callback_b(msg):
        calls_b.append(msg)
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'test/state', _callback_a, qos=0)
    async_fire_mqtt_message(hass, 'test/state', 'online', qos=0, retain=False)
    await mock_debouncer.wait()
    assert len(calls_a) == 1
    mqtt_client_mock.subscribe.assert_called()
    calls_a = []
    mqtt_client_mock.reset_mock()
    await hass.async_block_till_done()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'test/state', _callback_b, qos=1)
    async_fire_mqtt_message(hass, 'test/state', 'online', qos=0, retain=False)
    await mock_debouncer.wait()
    assert len(calls_a) == 1
    assert len(calls_b) == 1
    mqtt_client_mock.subscribe.assert_called()

async def test_replaying_payload_same_topic(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test replaying retained messages.

    When subscribing to the same topic again, SUBSCRIBE must be sent to the broker again
    for it to resend any retained messages for new subscriptions.
    Retained messages must only be replayed for new subscriptions, except
    when the MQTT client is reconnecting.
    """
    mqtt_client_mock = setup_with_birth_msg_client_mock
    calls_a: List[ReceiveMessage] = []
    calls_b: List[ReceiveMessage] = []

    @callback
    def _callback_a(msg):
        calls_a.append(msg)

    @callback
    def _callback_b(msg):
        calls_b.append(msg)
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'test/state', _callback_a)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, 'test/state', 'online', qos=0, retain=True)
    assert len(calls_a) == 1
    mqtt_client_mock.subscribe.assert_called()
    calls_a = []
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'test/state', _callback_b)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, 'test/state', 'online', qos=0, retain=False)
    async_fire_mqtt_message(hass, 'test/state', 'online', qos=0, retain=True)
    assert len(calls_a) == 1
    assert help_assert_message(calls_a[0], 'test/state', 'online', qos=0, retain=False)
    assert len(calls_b) == 2
    assert help_assert_message(calls_b[0], 'test/state', 'online', qos=0, retain=False)
    assert help_assert_message(calls_b[1], 'test/state', 'online', qos=0, retain=True)
    mqtt_client_mock.subscribe.assert_called()
    calls_a = []
    calls_b = []
    mqtt_client_mock.reset_mock()
    async_fire_mqtt_message(hass, 'test/state', 'online', qos=0, retain=False)
    assert len(calls_a) == 1
    assert help_assert_message(calls_a[0], 'test/state', 'online', qos=0, retain=False)
    assert len(calls_b) == 1
    assert help_assert_message(calls_b[0], 'test/state', 'online', qos=0, retain=False)
    calls_a = []
    calls_b = []
    mqtt_client_mock.reset_mock()
    mqtt_client_mock.on_disconnect(None, None, 0, MockMqttReasonCode())
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(None, None, None, MockMqttReasonCode())
    await mock_debouncer.wait()
    mqtt_client_mock.subscribe.assert_called()
    async_fire_mqtt_message(hass, 'test/state', 'online', qos=0, retain=True)
    assert len(calls_a) == 1
    assert help_assert_message(calls_a[0], 'test/state', 'online', qos=0, retain=True)
    assert len(calls_b) == 1
    assert help_assert_message(calls_b[0], 'test/state', 'online', qos=0, retain=True)

async def test_replaying_payload_after_resubscribing(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test replaying and filtering retained messages after resubscribing.

    When subscribing to the same topic again, SUBSCRIBE must be sent to the broker again
    for it to resend any retained messages for new subscriptions.
    Retained messages must only be replayed for new subscriptions, except
    when the MQTT client is reconnection.
    """
    mqtt_client_mock = setup_with_birth_msg_client_mock
    calls_a: List[ReceiveMessage] = []

    @callback
    def _callback_a(msg):
        calls_a.append(msg)
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    unsub = await mqtt.async_subscribe(hass, 'test/state', _callback_a)
    await mock_debouncer.wait()
    mqtt_client_mock.subscribe.assert_called()
    async_fire_mqtt_message(hass, 'test/state', 'online', qos=0, retain=True)
    assert help_assert_message(calls_a[0], 'test/state', 'online', qos=0, retain=True)
    calls_a.clear()
    async_fire_mqtt_message(hass, 'test/state', 'offline', qos=0, retain=False)
    assert help_assert_message(calls_a[0], 'test/state', 'offline', qos=0, retain=False)
    calls_a.clear()
    async_fire_mqtt_message(hass, 'test/state', 'offline', qos=0, retain=True)
    await hass.async_block_till_done()
    assert len(calls_a) == 0
    mock_debouncer.clear()
    unsub()
    unsub = await mqtt.async_subscribe(hass, 'test/state', _callback_a)
    await mock_debouncer.wait()
    mqtt_client_mock.subscribe.assert_called()
    async_fire_mqtt_message(hass, 'test/state', 'online', qos=0, retain=True)
    assert help_assert_message(calls_a[0], 'test/state', 'online', qos=0, retain=True)

async def test_replaying_payload_wildcard_topic(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test replaying retained messages.

    When we have multiple subscriptions to the same wildcard topic,
    SUBSCRIBE must be sent to the broker again
    for it to resend any retained messages for new subscriptions.
    Retained messages should only be replayed for new subscriptions, except
    when the MQTT client is reconnection.
    """
    mqtt_client_mock = setup_with_birth_msg_client_mock
    calls_a: List[ReceiveMessage] = []
    calls_b: List[ReceiveMessage] = []

    @callback
    def _callback_a(msg):
        calls_a.append(msg)

    @callback
    def _callback_b(msg):
        calls_b.append(msg)
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'test/#', _callback_a)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, 'test/state1', 'new_value_1', qos=0, retain=True)
    async_fire_mqtt_message(hass, 'test/state2', 'new_value_2', qos=0, retain=True)
    assert len(calls_a) == 2
    mqtt_client_mock.subscribe.assert_called()
    calls_a = []
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'test/#', _callback_b)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, 'test/state1', 'initial_value_1', qos=0, retain=True)
    async_fire_mqtt_message(hass, 'test/state2', 'initial_value_2', qos=0, retain=True)
    assert len(calls_a) == 0
    assert len(calls_b) == 2
    mqtt_client_mock.subscribe.assert_called()
    calls_a = []
    calls_b = []
    mqtt_client_mock.reset_mock()
    async_fire_mqtt_message(hass, 'test/state1', 'update_value_1', qos=0, retain=False)
    async_fire_mqtt_message(hass, 'test/state2', 'update_value_2', qos=0, retain=False)
    assert len(calls_a) == 2
    assert len(calls_b) == 2
    calls_a = []
    calls_b = []
    mqtt_client_mock.reset_mock()
    mqtt_client_mock.on_disconnect(None, None, 0, MockMqttReasonCode())
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(None, None, None, MockMqttReasonCode())
    await mock_debouncer.wait()
    mqtt_client_mock.subscribe.assert_called()
    async_fire_mqtt_message(hass, 'test/state1', 'update_value_1', qos=0, retain=True)
    async_fire_mqtt_message(hass, 'test/state2', 'update_value_2', qos=0, retain=True)
    assert len(calls_a) == 2
    assert len(calls_b) == 2

async def test_not_calling_unsubscribe_with_active_subscribers(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient, record_calls: MessageCallbackType) -> None:
    """Test not calling unsubscribe() when other subscribers are active."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    unsub = await mqtt.async_subscribe(hass, 'test/state', record_calls, 2)
    await mqtt.async_subscribe(hass, 'test/state', record_calls, 1)
    await mock_debouncer.wait()
    assert mqtt_client_mock.subscribe.called
    mock_debouncer.clear()
    unsub()
    await hass.async_block_till_done()
    await hass.async_block_till_done(wait_background_tasks=True)
    async_fire_time_changed(hass, utcnow() + timedelta(seconds=3))
    assert not mqtt_client_mock.unsubscribe.called
    assert not mock_debouncer.is_set()

async def test_not_calling_subscribe_when_unsubscribed_within_cooldown(hass: HomeAssistant, mock_debouncer: asyncio.Event, mqtt_mock_entry: MqttMockHAClientGenerator, record_calls: MessageCallbackType) -> None:
    """Test not calling subscribe() when it is unsubscribed.

    Make sure subscriptions are cleared if unsubscribed before
    the subscribe cool down period has ended.
    """
    mqtt_mock = await mqtt_mock_entry()
    mqtt_client_mock = mqtt_mock._mqttc
    await mock_debouncer.wait()
    mock_debouncer.clear()
    mqtt_client_mock.subscribe.reset_mock()
    unsub = await mqtt.async_subscribe(hass, 'test/state', record_calls)
    unsub()
    await mock_debouncer.wait()
    assert not mqtt_client_mock.subscribe.called

async def test_unsubscribe_race(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test not calling unsubscribe() when other subscribers are active."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    calls_a: List[ReceiveMessage] = []
    calls_b: List[ReceiveMessage] = []

    @callback
    def _callback_a(msg):
        calls_a.append(msg)

    @callback
    def _callback_b(msg):
        calls_b.append(msg)
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    unsub = await mqtt.async_subscribe(hass, 'test/state', _callback_a)
    unsub()
    await mqtt.async_subscribe(hass, 'test/state', _callback_b)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, 'test/state', 'online')
    assert not calls_a
    assert len(calls_b) == 1
    expected_calls_1: List[Tuple[str, Any]] = [call.subscribe([('test/state', 0)]), call.unsubscribe('test/state'), call.subscribe([('test/state', 0)])]
    expected_calls_2: List[Tuple[str, Any]] = [call.subscribe([('test/state', 0)]), call.subscribe([('test/state', 0)])]
    expected_calls_3: List[Tuple[str, Any]] = [call.subscribe([('test/state', 0)])]
    assert mqtt_client_mock.mock_calls in (expected_calls_1, expected_calls_2, expected_calls_3)

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), [({'broker': 'test-broker', 'certificate': 'auto'}, {'not set'})])
async def test_restore_subscriptions_on_reconnect(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient, record_calls: MessageCallbackType) -> None:
    """Test subscriptions are restored on reconnect."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'test/state', record_calls)
    async_fire_time_changed(hass, utcnow() + timedelta(seconds=3))
    await mock_debouncer.wait()
    assert ('test/state', 0) in help_all_subscribe_calls(mqtt_client_mock)
    mqtt_client_mock.reset_mock()
    mqtt_client_mock.on_disconnect(None, None, 0, MockMqttReasonCode())
    await mqtt.async_subscribe(hass, 'test/other', record_calls)
    async_fire_time_changed(hass, utcnow() + timedelta(seconds=3))
    assert ('test/other', 0) not in help_all_subscribe_calls(mqtt_client_mock)
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(None, None, None, MockMqttReasonCode())
    await mock_debouncer.wait()
    assert ('test/state', 0) in help_all_subscribe_calls(mqtt_client_mock)
    assert ('test/other', 0) in help_all_subscribe_calls(mqtt_client_mock)

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), [({'broker': 'test-broker'}, {'conf_discovery': False})])
async def test_restore_all_active_subscriptions_on_reconnect(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient, record_calls: MessageCallbackType) -> None:
    """Test active subscriptions are restored correctly on reconnect."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    unsub = await mqtt.async_subscribe(hass, 'test/state', record_calls, qos=2)
    await mqtt.async_subscribe(hass, 'test/state', record_calls, qos=1)
    await mqtt.async_subscribe(hass, 'test/state', record_calls, qos=0)
    await mock_debouncer.wait()
    expected = [call([('test/state', 2)])]
    assert mqtt_client_mock.subscribe.mock_calls == expected
    unsub()
    assert mqtt_client_mock.unsubscribe.call_count == 0
    mqtt_client_mock.on_disconnect(None, None, 0, MockMqttReasonCode())
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(None, None, None, MockMqttReasonCode())
    await mock_debouncer.wait()
    expected.append(call([('test/state', 1)]))
    for expected_call in expected:
        assert mqtt_client_mock.subscribe.call_args_list and expected_call in mqtt_client_mock.subscribe.call_args_list

@pytest.mark.parametrize(('mqtt_config_entry_data', 'insecure_param'), [({'broker': 'test-broker', 'certificate': 'auto'}, 'not set'), ({'broker': 'test-broker', 'certificate': 'auto', 'tls_insecure': False}, False), ({'broker': 'test-broker', 'certificate': 'auto', 'tls_insecure': True}, True)])
async def test_setup_uses_certificate_on_certificate_set_to_auto_and_insecure(hass: HomeAssistant, mqtt_mock_entry: MqttMockHAClientGenerator, insecure_param: Union[bool, str]) -> None:
    """Test setup uses bundled certs when certificate is set to auto and insecure."""
    calls: List[Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]] = []
    insecure_check: dict[str, Union[bool, str]] = {'insecure': 'not set'}

    def mock_tls_set(certificate, certfile=None, keyfile=None, tls_version=None):
        calls.append((certificate, certfile, keyfile, tls_version))

    def mock_tls_insecure_set(insecure):
        insecure_check['insecure'] = insecure
    with patch('homeassistant.components.mqtt.async_client.AsyncMQTTClient') as mock_client:
        mock_client().tls_set = mock_tls_set
        mock_client().tls_insecure_set = mock_tls_insecure_set
        await mqtt_mock_entry()
        await hass.async_block_till_done()
    assert calls
    expected_certificate = certifi.where()
    assert calls[0][0] == expected_certificate
    assert insecure_check['insecure'] == insecure_param

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), [({mqtt.CONF_BROKER: 'mock-broker', mqtt.CONF_CERTIFICATE: 'auto'}, {})])
async def test_tls_version(hass: HomeAssistant, mqtt_client_mock: MqttMockPahoClient, mqtt_mock_entry: MqttMockHAClientGenerator) -> None:
    """Test setup defaults for tls."""
    await mqtt_mock_entry()
    await hass.async_block_till_done()
    assert mqtt_client_mock.tls_set.mock_calls[0][2]['tls_version'] == ssl.PROTOCOL_TLS_CLIENT

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), [({mqtt.CONF_BROKER: 'mock-broker'}, {mqtt.CONF_WILL_MESSAGE: {mqtt.ATTR_TOPIC: 'death', mqtt.ATTR_PAYLOAD: 'death', mqtt.ATTR_QOS: 0, mqtt.ATTR_RETAIN: False}})])
async def test_custom_will_message(hass: HomeAssistant, mqtt_config_entry_data: dict[str, Any], mqtt_config_entry_options: dict[str, Any], mqtt_client_mock: MqttMockPahoClient) -> None:
    """Test will message."""
    entry = MockConfigEntry(domain=mqtt.DOMAIN, data=mqtt_config_entry_data, options=mqtt_config_entry_options, version=mqtt.CONFIG_ENTRY_VERSION, minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION)
    entry.add_to_hass(hass)
    hass.config.components.add(mqtt.DOMAIN)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    mqtt_client_mock.will_set.assert_called_with(topic='death', payload='death', qos=0, retain=False)

@pytest.mark.parametrize('mqtt_config_entry_options', [ENTRY_DEFAULT_BIRTH_MESSAGE])
async def test_default_will_message(setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test will message."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    mqtt_client_mock.will_set.assert_called_with(topic='homeassistant/status', payload='offline', qos=0, retain=False)

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), [({'broker': 'mock-broker'}, {mqtt.CONF_WILL_MESSAGE: {}})])
async def test_no_will_message(hass: HomeAssistant, mqtt_config_entry_data: dict[str, Any], mqtt_config_entry_options: dict[str, Any], mqtt_client_mock: MqttMockPahoClient) -> None:
    """Test will message."""
    entry = MockConfigEntry(domain=mqtt.DOMAIN, data=mqtt_config_entry_data, options=mqtt_config_entry_options, version=mqtt.CONFIG_ENTRY_VERSION, minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION)
    entry.add_to_hass(hass)
    hass.config.components.add(mqtt.DOMAIN)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    mqtt_client_mock.will_set.assert_not_called()

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), [({'broker': 'mock-broker'}, {'conf_discovery': False})])
async def test_mqtt_subscribes_topics_on_connect(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient, record_calls: MessageCallbackType) -> None:
    """Test subscription to topic on connect."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'test/state', record_calls)
    await mqtt.async_subscribe(hass, 'home/sensor', record_calls, 2)
    await mqtt.async_subscribe(hass, 'still/pending', record_calls)
    await mqtt.async_subscribe(hass, 'still/pending', record_calls, 1)
    await mock_debouncer.wait()
    subscribe_calls = help_all_subscribe_calls(mqtt_client_mock)
    assert ('test/state', 0) in subscribe_calls
    assert ('home/sensor', 2) in subscribe_calls
    assert ('still/pending', 1) in subscribe_calls

@pytest.mark.parametrize('mqtt_config_entry_options', [ENTRY_DEFAULT_BIRTH_MESSAGE | {'conf_discovery': False}])
async def test_mqtt_subscribes_wildcard_topics_in_correct_order(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient, record_calls: MessageCallbackType) -> None:
    """Test subscription to wildcard topics on connect in the order of subscription."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'integration/test#', record_calls)
    await mqtt.async_subscribe(hass, 'integration/kitchen_sink#', record_calls)
    await mock_debouncer.wait()

    def _assert_subscription_order():
        discovery_subscribes: List[str] = [f'homeassistant/{platform}/+/config' for platform in SUPPORTED_COMPONENTS]
        discovery_subscribes.extend([f'homeassistant/{platform}/+/+/config' for platform in SUPPORTED_COMPONENTS])
        discovery_subscribes.extend(['homeassistant/device/+/config', 'homeassistant/device/+/+/config'])
        discovery_subscribes.extend(['integration/test#', 'integration/kitchen_sink#'])
        expected_discovery_subscribes = discovery_subscribes.copy()
        actual_subscribes: List[str] = [discovery_subscribes.pop(0) for call in help_all_subscribe_calls(mqtt_client_mock) if discovery_subscribes and discovery_subscribes[0] == call[0]]
        assert len(discovery_subscribes) == 0
        assert actual_subscribes == expected_discovery_subscribes
    _assert_subscription_order()
    mqtt_client_mock.on_disconnect(Mock(), None, 0, MockMqttReasonCode())
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(Mock(), None, 0, MockMqttReasonCode())
    await mock_debouncer.wait()
    _assert_subscription_order()

@pytest.mark.parametrize('mqtt_config_entry_options', [ENTRY_DEFAULT_BIRTH_MESSAGE | {'conf_discovery': False}])
async def test_mqtt_discovery_not_subscribes_when_disabled(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test discovery subscriptions not performend when discovery is disabled."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    await mock_debouncer.wait()
    subscribe_calls = help_all_subscribe_calls(mqtt_client_mock)
    for component in SUPPORTED_COMPONENTS:
        assert (f'homeassistant/{component}/+/config', 0) not in subscribe_calls
        assert (f'homeassistant/{component}/+/+/config', 0) not in subscribe_calls
    mqtt_client_mock.on_disconnect(Mock(), None, 0, MockMqttReasonCode())
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(Mock(), None, 0, MockMqttReasonCode())
    await mock_debouncer.wait()
    subscribe_calls = help_all_subscribe_calls(mqtt_client_mock)
    for component in SUPPORTED_COMPONENTS:
        assert (f'homeassistant/{component}/+/config', 0) not in subscribe_calls
        assert (f'homeassistant/{component}/+/+/config', 0) not in subscribe_calls

@pytest.mark.parametrize('mqtt_config_entry_options', [ENTRY_DEFAULT_BIRTH_MESSAGE])
@patch('homeassistant.components.mqtt.client.MAX_SUBSCRIBES_PER_CALL', 2)
@patch('homeassistant.components.mqtt.client.MAX_UNSUBSCRIBES_PER_CALL', 2)
async def test_mqtt_subscribes_and_unsubscribes_in_chunks(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient, record_calls: MessageCallbackType) -> None:
    """Test chunked client subscriptions."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    mqtt_client_mock.subscribe.reset_mock()
    unsub_tasks: List[Callable[[], None]] = []
    mock_debouncer.clear()
    unsub_tasks.append(await mqtt.async_subscribe(hass, 'topic/test1', record_calls))
    unsub_tasks.append(await mqtt.async_subscribe(hass, 'home/sensor1', record_calls))
    unsub_tasks.append(await mqtt.async_subscribe(hass, 'topic/test2', record_calls))
    unsub_tasks.append(await mqtt.async_subscribe(hass, 'home/sensor2', record_calls))
    await mock_debouncer.wait()
    assert mqtt_client_mock.subscribe.call_count == 2
    assert len(mqtt_client_mock.subscribe.mock_calls[0][1][0]) == 2
    assert len(mqtt_client_mock.subscribe.mock_calls[1][1][0]) == 2
    mock_debouncer.clear()
    for task in unsub_tasks:
        task()
    await mock_debouncer.wait()
    assert mqtt_client_mock.unsubscribe.call_count == 2
    assert len(mqtt_client_mock.unsubscribe.mock_calls[0][1][0]) == 2
    assert len(mqtt_client_mock.unsubscribe.mock_calls[1][1][0]) == 2

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), [({'broker': 'mock-broker'}, {'birth_message': {'topic': 'birth', 'payload': 'birth', 'qos': 0, 'retain': False}})])
@patch('homeassistant.components.mqtt.client.INITIAL_SUBSCRIBE_COOLDOWN', 0.0)
@patch('homeassistant.components.mqtt.client.DISCOVERY_COOLDOWN', 0.0)
@patch('homeassistant.components.mqtt.client.SUBSCRIBE_COOLDOWN', 0.0)
async def test_custom_birth_message(hass: HomeAssistant, mock_debouncer: asyncio.Event, mqtt_config_entry_data: dict[str, Any], mqtt_config_entry_options: dict[str, Any], mqtt_client_mock: MqttMockPahoClient) -> None:
    """Test sending birth message."""
    entry = MockConfigEntry(domain=mqtt.DOMAIN, data=mqtt_config_entry_data, options=mqtt_config_entry_options, version=mqtt.CONFIG_ENTRY_VERSION, minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION)
    entry.add_to_hass(hass)
    hass.config.components.add(mqtt.DOMAIN)
    assert await hass.config_entries.async_setup(entry.entry_id)
    mock_debouncer.clear()
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await mock_debouncer.wait()
    await hass.async_block_till_done(wait_background_tasks=True)
    mqtt_client_mock.publish.assert_called_with('birth', 'birth', 0, False)

@pytest.mark.parametrize('mqtt_config_entry_options', [ENTRY_DEFAULT_BIRTH_MESSAGE])
async def test_default_birth_message(hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test sending birth message."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    await hass.async_block_till_done(wait_background_tasks=True)
    mqtt_client_mock.publish.assert_called_with('homeassistant/status', 'online', 0, False)

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), [({'broker': 'mock-broker'}, {'birth_message': {}})])
@patch('homeassistant.components.mqtt.client.INITIAL_SUBSCRIBE_COOLDOWN', 0.0)
@patch('homeassistant.components.mqtt.client.DISCOVERY_COOLDOWN', 0.0)
@patch('homeassistant.components.mqtt.client.SUBSCRIBE_COOLDOWN', 0.0)
async def test_no_birth_message(hass: HomeAssistant, record_calls: MessageCallbackType, mock_debouncer: asyncio.Event, mqtt_config_entry_data: dict[str, Any], mqtt_config_entry_options: dict[str, Any], mqtt_client_mock: MqttMockPahoClient) -> None:
    """Test disabling birth message."""
    entry = MockConfigEntry(domain=mqtt.DOMAIN, data=mqtt_config_entry_data, options=mqtt_config_entry_options, version=mqtt.CONFIG_ENTRY_VERSION, minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION)
    entry.add_to_hass(hass)
    hass.config.components.add(mqtt.DOMAIN)
    mock_debouncer.clear()
    assert await hass.config_entries.async_setup(entry.entry_id)
    await mock_debouncer.wait()
    await hass.async_block_till_done(wait_background_tasks=True)
    mqtt_client_mock.publish.assert_not_called()
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'homeassistant/some-topic', record_calls)
    await mock_debouncer.wait()
    mqtt_client_mock.subscribe.assert_called()

@pytest.mark.parametrize('mqtt_config_entry_options', [ENTRY_DEFAULT_BIRTH_MESSAGE])
async def test_subscribed_at_highest_qos(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient, record_calls: MessageCallbackType) -> None:
    """Test the highest qos as assigned when subscribing to the same topic."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'test/state', record_calls, qos=0)
    await hass.async_block_till_done()
    await mock_debouncer.wait()
    assert ('test/state', 0) in help_all_subscribe_calls(mqtt_client_mock)
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'test/state', record_calls, qos=1)
    await mqtt.async_subscribe(hass, 'test/state', record_calls, qos=2)
    await mock_debouncer.wait()
    assert help_all_subscribe_calls(mqtt_client_mock) == [('test/state', 2)]

async def test_initial_setup_logs_error(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, mqtt_client_mock: MqttMockPahoClient) -> None:
    """Test for setup failure if initial client connection fails."""
    entry = MockConfigEntry(domain=mqtt.DOMAIN, data={mqtt.CONF_BROKER: 'test-broker'}, version=mqtt.CONFIG_ENTRY_VERSION, minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION)
    entry.add_to_hass(hass)
    mqtt_client_mock.connect.side_effect = MagicMock(return_value=1)
    try:
        assert await hass.config_entries.async_setup(entry.entry_id)
    except HomeAssistantError:
        assert True
    assert 'Failed to connect to MQTT server:' in caplog.text

async def test_logs_error_if_no_connect_broker(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test for setup failure if connection to broker is missing."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    mqtt_client_mock.on_disconnect(Mock(), None, None, MockMqttReasonCode())
    mqtt_client_mock.on_connect(Mock(), None, None, MockMqttReasonCode(value=136, is_failure=True, name='Server unavailable'))
    await hass.async_block_till_done()
    assert 'Unable to connect to the MQTT broker: Server unavailable' in caplog.text

@pytest.mark.parametrize('reason_code', [MockMqttReasonCode(value=134, is_failure=True, name='Bad user name or password'), MockMqttReasonCode(value=135, is_failure=True, name='Not authorized')])
async def test_triggers_reauth_flow_if_auth_fails(hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockPahoClient, reason_code: MockMqttReasonCode) -> None:
    """Test re-auth is triggered if authentication is failing."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    mqtt_client_mock.on_disconnect(Mock(), None, 0, MockMqttReasonCode(), None)
    mqtt_client_mock.on_connect(Mock(), None, None, reason_code)
    await hass.async_block_till_done()
    flows = hass.config_entries.flow.async_progress()
    assert len(flows) == 1
    assert flows[0]['context']['source'] == 'reauth'

@patch('homeassistant.components.mqtt.client.TIMEOUT_ACK', 0.3)
async def test_handle_mqtt_on_callback(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test receiving an ACK callback before waiting for it."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    with patch.object(mqtt_client_mock, 'get_mid', return_value=100):
        mqtt_client_mock.on_publish(mqtt_client_mock, None, 100, MockMqttReasonCode(), None)
        await hass.async_block_till_done()
        await hass.async_block_till_done()
        await mqtt.async_publish(hass, 'no_callback/test-topic', 'test-payload')
        await hass.async_block_till_done()
        assert 'No ACK from MQTT server' not in caplog.text

async def test_handle_mqtt_on_callback_after_cancellation(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, mqtt_mock_entry: MqttMockHAClientGenerator, mqtt_client_mock: MqttMockPahoClient) -> None:
    """Test receiving an ACK after a cancellation."""
    mqtt_mock = await mqtt_mock_entry()
    mqtt_mock()._async_get_mid_future(101).cancel()
    mqtt_client_mock.on_publish(mqtt_client_mock, None, 101, MockMqttReasonCode(), None)
    await hass.async_block_till_done()
    assert 'No ACK from MQTT server' not in caplog.text
    assert 'InvalidStateError' not in caplog.text

async def test_handle_mqtt_on_callback_after_timeout(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, mqtt_mock_entry: MqttMockHAClientGenerator, mqtt_client_mock: MqttMockPahoClient) -> None:
    """Test receiving an ACK after a timeout."""
    mqtt_mock = await mqtt_mock_entry()
    mqtt_mock()._async_get_mid_future(101).set_exception(asyncio.TimeoutError)
    mqtt_client_mock.on_publish(mqtt_client_mock, None, 101, MockMqttReasonCode(), None)
    await hass.async_block_till_done()
    assert 'No ACK from MQTT server' not in caplog.text
    assert 'InvalidStateError' not in caplog.text

async def test_publish_error(hass: HomeAssistant, caplog: pytest.LogCaptureFixture) -> None:
    """Test publish error."""
    entry = MockConfigEntry(domain=mqtt.DOMAIN, data={mqtt.CONF_BROKER: 'test-broker'}, version=mqtt.CONFIG_ENTRY_VERSION, minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION)
    entry.add_to_hass(hass)
    with patch('homeassistant.components.mqtt.async_client.AsyncMQTTClient') as mock_client:
        mock_client().connect = lambda *args: 1
        mock_client().publish.return_value.rc = 1
        assert await hass.config_entries.async_setup(entry.entry_id)
        with pytest.raises(HomeAssistantError):
            await mqtt.async_publish(hass, 'some-topic', b'test-payload', qos=0, retain=False, encoding=None)
        assert 'Failed to connect to MQTT server: Out of memory.' in caplog.text

async def test_subscribe_error(hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockPahoClient, record_calls: MessageCallbackType, caplog: pytest.LogCaptureFixture) -> None:
    """Test publish error."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    mqtt_client_mock.reset_mock()
    mqtt_client_mock.subscribe.side_effect = lambda *args: (4, None)
    await mqtt.async_subscribe(hass, 'some-topic', record_calls)
    while mqtt_client_mock.subscribe.call_count == 0:
        await hass.async_block_till_done()
    await hass.async_block_till_done()
    assert 'Error talking to MQTT: The client is not currently connected.' in caplog.text

async def test_handle_message_callback(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient) -> None:
    """Test for handling an incoming message callback."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    callbacks: List[ReceiveMessage] = []

    @callback
    def _callback(args):
        callbacks.append(args)
    msg = ReceiveMessage('some-topic', b'test-payload', 1, False, 'some-topic', time.monotonic())
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, 'some-topic', _callback)
    await mock_debouncer.wait()
    mqtt_client_mock.reset_mock()
    mqtt_client_mock.on_message(None, None, msg)
    assert len(callbacks) == 1
    assert callbacks[0].topic == 'some-topic'
    assert callbacks[0].qos == 1
    assert callbacks[0].payload == 'test-payload'

@pytest.mark.parametrize(('mqtt_config_entry_data', 'protocol'), [({mqtt.CONF_BROKER: 'mock-broker', CONF_PROTOCOL: '3.1'}, 3), ({mqtt.CONF_BROKER: 'mock-broker', CONF_PROTOCOL: '3.1.1'}, 4), ({mqtt.CONF_BROKER: 'mock-broker', CONF_PROTOCOL: '5'}, 5)])
async def test_setup_mqtt_client_protocol(mqtt_mock_entry: MqttMockHAClientGenerator, protocol: int) -> None:
    """Test MQTT client protocol setup."""
    with patch('homeassistant.components.mqtt.async_client.AsyncMQTTClient') as mock_client:
        await mqtt_mock_entry()
    assert mock_client.call_args[1]['protocol'] == protocol

@patch('homeassistant.components.mqtt.client.TIMEOUT_ACK', 0.2)
async def test_handle_mqtt_timeout_on_callback(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, mock_debouncer: asyncio.Event) -> None:
    """Test publish without receiving an ACK callback."""
    mid = 0

    class FakeInfo:
        """Returns a simulated client publish response."""
        mid: int = 102
        rc: int = 0
    with patch('homeassistant.components.mqtt.async_client.AsyncMQTTClient') as mock_client:

        def _mock_ack(topic, qos=0):
            nonlocal mid
            mid += 1
            mock_client.on_subscribe(0, 0, mid)
            return (0, mid)
        mock_client_instance = mock_client.return_value
        mock_client_instance.publish.return_value = FakeInfo()
        mock_client_instance.subscribe.side_effect = _mock_ack
        mock_client_instance.unsubscribe.side_effect = _mock_ack
        mock_client_instance.connect = MagicMock(return_value=0, side_effect=lambda *args, **kwargs: hass.loop.call_soon_threadsafe(mock_client_instance.on_connect, mock_client_instance, None, 0, MockMqttReasonCode()))
        entry = MockConfigEntry(domain=mqtt.DOMAIN, data={mqtt.CONF_BROKER: 'test-broker'}, version=mqtt.CONFIG_ENTRY_VERSION, minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION)
        entry.add_to_hass(hass)
        mock_debouncer.clear()
        assert await hass.config_entries.async_setup(entry.entry_id)
        await mqtt.async_publish(hass, 'no_callback/test-topic', 'test-payload')
        await hass.async_block_till_done()
        assert len(mock_client_instance.publish.mock_calls) == 1
        assert 'No ACK from MQTT server' in caplog.text
        await hass.config_entries.async_unload(entry.entry_id)
        assert not mock_debouncer.is_set()

@pytest.mark.parametrize('exception', [OSError('Connection error'), paho_mqtt.WebsocketConnectionError('Connection error')])
async def test_setup_raises_config_entry_not_ready_if_no_connect_broker(hass: HomeAssistant, caplog: pytest.LogCaptureFixture, exception: Exception) -> None:
    """Test for setup failure if connection to broker is missing."""
    entry = MockConfigEntry(domain=mqtt.DOMAIN, data={mqtt.CONF_BROKER: 'test-broker'}, version=mqtt.CONFIG_ENTRY_VERSION, minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION)
    entry.add_to_hass(hass)
    with patch('homeassistant.components.mqtt.async_client.AsyncMQTTClient') as mock_client:
        mock_client().connect = MagicMock(side_effect=exception)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        assert 'Failed to connect to MQTT server due to exception:' in caplog.text

@pytest.mark.parametrize('mqtt_config_entry_options', [ENTRY_DEFAULT_BIRTH_MESSAGE])
async def test_subscribe_and_resubscribe(hass: HomeAssistant, mock_debouncer: asyncio.Event, setup_with_birth_msg_client_mock: MqttMockPahoClient, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType) -> None:
    """Test resubscribing within the debounce time."""