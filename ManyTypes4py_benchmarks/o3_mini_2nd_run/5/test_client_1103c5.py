#!/usr/bin/env python3
"""The tests for the MQTT client."""
import asyncio
from datetime import timedelta
import socket
import ssl
import time
from typing import Any, Optional, Callable, Awaitable, List
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


def help_assert_message(
    msg: Any,
    topic: Optional[str] = None,
    payload: Optional[Any] = None,
    qos: Optional[int] = None,
    retain: Optional[bool] = None,
) -> bool:
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


async def test_mqtt_connects_on_home_assistant_mqtt_setup(
    hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockHAClient
) -> None:
    """Test if client is connected after mqtt init on bootstrap."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    assert mqtt_client_mock.connect.call_count == 1


async def test_mqtt_does_not_disconnect_on_home_assistant_stop(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient
) -> None:
    """Test if client is not disconnected on HA stop."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    hass.bus.fire(EVENT_HOMEASSISTANT_STOP)
    await mock_debouncer.wait()
    assert mqtt_client_mock.disconnect.call_count == 0


async def test_mqtt_await_ack_at_disconnect(hass: HomeAssistant) -> None:
    """Test if ACK is awaited correctly when disconnecting."""

    class FakeInfo:
        """Returns a simulated client publish response."""
        mid: int = 100
        rc: int = 0

    with patch("homeassistant.components.mqtt.async_client.AsyncMQTTClient") as mock_client:
        mqtt_client: Any = mock_client.return_value
        mqtt_client.connect = MagicMock(
            return_value=0,
            side_effect=lambda *args, **kwargs: hass.loop.call_soon_threadsafe(
                mqtt_client.on_connect, mqtt_client, None, 0, MockMqttReasonCode()
            ),
        )
        mqtt_client.publish = MagicMock(return_value=FakeInfo())
        entry = MockConfigEntry(
            domain=mqtt.DOMAIN,
            data={ "certificate": "auto", mqtt.CONF_BROKER: "test-broker", mqtt.CONF_DISCOVERY: False },
            version=mqtt.CONFIG_ENTRY_VERSION,
            minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION,
        )
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        mqtt_client = mock_client.return_value
        hass.async_create_task(mqtt.async_publish(hass, "test-topic", "some-payload", 0, False))
        await asyncio.sleep(0)
        mqtt_client.on_publish(0, 0, 100, MockMqttReasonCode(), None)
        await hass.async_stop()
        await hass.async_block_till_done()
        assert mqtt_client.publish.called
        assert mqtt_client.publish.call_args[0] == ("test-topic", "some-payload", 0, False)
        await hass.async_block_till_done(wait_background_tasks=True)


@pytest.mark.parametrize("mqtt_config_entry_options", [ENTRY_DEFAULT_BIRTH_MESSAGE])
async def test_publish(
    hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockHAClient
) -> None:
    """Test the publish function."""
    publish_mock: Any = setup_with_birth_msg_client_mock.publish
    await mqtt.async_publish(hass, "test-topic", "test-payload")
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ("test-topic", "test-payload", 0, False)
    publish_mock.reset_mock()
    await mqtt.async_publish(hass, "test-topic", "test-payload", 2, True)
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ("test-topic", "test-payload", 2, True)
    publish_mock.reset_mock()
    mqtt.publish(hass, "test-topic2", "test-payload2")
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ("test-topic2", "test-payload2", 0, False)
    publish_mock.reset_mock()
    mqtt.publish(hass, "test-topic2", "test-payload2", 2, True)
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ("test-topic2", "test-payload2", 2, True)
    publish_mock.reset_mock()
    mqtt.publish(hass, "test-topic3", b"\xde\xad\xbe\xef", 0, False)
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ("test-topic3", b"\xde\xad\xbe\xef", 0, False)
    publish_mock.reset_mock()
    mqtt.publish(hass, "test-topic3", None, 0, False)
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == ("test-topic3", None, 0, False)
    publish_mock.reset_mock()


async def test_convert_outgoing_payload(hass: HomeAssistant) -> None:
    """Test the converting of outgoing MQTT payloads without template."""
    command_template: mqtt.MqttCommandTemplate = mqtt.MqttCommandTemplate(None)
    assert command_template.async_render(b"\xde\xad\xbe\xef") == b"\xde\xad\xbe\xef"
    assert command_template.async_render("b'\\xde\\xad\\xbe\\xef'") == "b'\\xde\\xad\\xbe\\xef'"
    assert command_template.async_render(1234) == 1234
    assert command_template.async_render(1234.56) == 1234.56
    assert command_template.async_render(None) is None


async def test_all_subscriptions_run_when_decode_fails(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test all other subscriptions still run when decode fails for one."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic", record_calls, encoding="ascii")
    await mqtt.async_subscribe(hass, "test-topic", record_calls)
    async_fire_mqtt_message(hass, "test-topic", UnitOfTemperature.CELSIUS)
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1


async def test_subscribe_topic(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of a topic."""
    await mqtt_mock_entry()
    unsub: Callable[[], None] = await mqtt.async_subscribe(hass, "test-topic", record_calls)
    async_fire_mqtt_message(hass, "test-topic", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == "test-topic"
    assert recorded_calls[0].payload == "test-payload"
    unsub()
    async_fire_mqtt_message(hass, "test-topic", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    with pytest.raises(HomeAssistantError):
        unsub()


@pytest.mark.usefixtures("mqtt_mock_entry")
async def test_subscribe_topic_not_initialize(hass: HomeAssistant, record_calls: MessageCallbackType) -> None:
    """Test the subscription of a topic when MQTT was not initialized."""
    with pytest.raises(HomeAssistantError, match=".*make sure MQTT is set up correctly"):
        await mqtt.async_subscribe(hass, "test-topic", record_calls)


async def test_subscribe_mqtt_config_entry_disabled(
    hass: HomeAssistant, mqtt_mock: Any, record_calls: MessageCallbackType
) -> None:
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
    with pytest.raises(HomeAssistantError, match=".*MQTT is not enabled"):
        await mqtt.async_subscribe(hass, "test-topic", record_calls)


async def test_subscribe_and_resubscribe(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient, recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test resubscribing within the debounce time."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    with patch("homeassistant.components.mqtt.client.SUBSCRIBE_COOLDOWN", 0.4), patch("homeassistant.components.mqtt.client.UNSUBSCRIBE_COOLDOWN", 0.4):
        mock_debouncer.clear()
        unsub: Callable[[], None] = await mqtt.async_subscribe(hass, "test-topic", record_calls)
        unsub()
        unsub = await mqtt.async_subscribe(hass, "test-topic", record_calls)
        await mock_debouncer.wait()
        mock_debouncer.clear()
        async_fire_mqtt_message(hass, "test-topic", "test-payload")
        assert len(recorded_calls) == 1
        assert recorded_calls[0].topic == "test-topic"
        assert recorded_calls[0].payload == "test-payload"
        mqtt_client_mock.unsubscribe.assert_not_called()
        mock_debouncer.clear()
        unsub()
        await mock_debouncer.wait()
        mqtt_client_mock.unsubscribe.assert_called_once_with(["test-topic"])


async def test_subscribe_topic_non_async(
    hass: HomeAssistant, mock_debouncer: Any, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of a topic using the non-async function."""
    await mqtt_mock_entry()
    await mock_debouncer.wait()
    mock_debouncer.clear()
    unsub: Callable[[], None] = await hass.async_add_executor_job(mqtt.subscribe, hass, "test-topic", record_calls)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, "test-topic", "test-payload")
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == "test-topic"
    assert recorded_calls[0].payload == "test-payload"
    mock_debouncer.clear()
    await hass.async_add_executor_job(unsub)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, "test-topic", "test-payload")
    assert len(recorded_calls) == 1


async def test_subscribe_bad_topic(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of a topic."""
    await mqtt_mock_entry()
    with pytest.raises(HomeAssistantError):
        await mqtt.async_subscribe(hass, 55, record_calls)


async def test_subscribe_topic_not_match(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test if subscribed topic is not a match."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic", record_calls)
    async_fire_mqtt_message(hass, "another-test-topic", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_level_wildcard(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic/+/on", record_calls)
    async_fire_mqtt_message(hass, "test-topic/bier/on", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == "test-topic/bier/on"
    assert recorded_calls[0].payload == "test-payload"


async def test_subscribe_topic_level_wildcard_no_subtree_match(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic/+/on", record_calls)
    async_fire_mqtt_message(hass, "test-topic/bier", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_level_wildcard_root_topic_no_subtree_match(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic/#", record_calls)
    async_fire_mqtt_message(hass, "test-topic-123", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_subtree_wildcard_subtree_topic(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic/#", record_calls)
    async_fire_mqtt_message(hass, "test-topic/bier/on", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == "test-topic/bier/on"
    assert recorded_calls[0].payload == "test-payload"


async def test_subscribe_topic_subtree_wildcard_root_topic(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic/#", record_calls)
    async_fire_mqtt_message(hass, "test-topic", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == "test-topic"
    assert recorded_calls[0].payload == "test-payload"


async def test_subscribe_topic_subtree_wildcard_no_match(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic/#", record_calls)
    async_fire_mqtt_message(hass, "another-test-topic", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_level_wildcard_and_wildcard_root_topic(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "+/test-topic/#", record_calls)
    async_fire_mqtt_message(hass, "hi/test-topic", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == "hi/test-topic"
    assert recorded_calls[0].payload == "test-payload"


async def test_subscribe_topic_level_wildcard_and_wildcard_subtree_topic(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "+/test-topic/#", record_calls)
    async_fire_mqtt_message(hass, "hi/test-topic/here-iam", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == "hi/test-topic/here-iam"
    assert recorded_calls[0].payload == "test-payload"


async def test_subscribe_topic_level_wildcard_and_wildcard_level_no_match(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "+/test-topic/#", record_calls)
    async_fire_mqtt_message(hass, "hi/here-iam/test-topic", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_level_wildcard_and_wildcard_no_match(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "+/test-topic/#", record_calls)
    async_fire_mqtt_message(hass, "hi/another-test-topic", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_sys_root(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of $ root topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "$test-topic/subtree/on", record_calls)
    async_fire_mqtt_message(hass, "$test-topic/subtree/on", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == "$test-topic/subtree/on"
    assert recorded_calls[0].payload == "test-payload"


async def test_subscribe_topic_sys_root_and_wildcard_topic(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of $ root and wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "$test-topic/#", record_calls)
    async_fire_mqtt_message(hass, "$test-topic/some-topic", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == "$test-topic/some-topic"
    assert recorded_calls[0].payload == "test-payload"


async def test_subscribe_topic_sys_root_and_wildcard_subtree_topic(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription of $ root and wildcard subtree topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "$test-topic/subtree/#", record_calls)
    async_fire_mqtt_message(hass, "$test-topic/subtree/some-topic", "test-payload")
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == "$test-topic/subtree/some-topic"
    assert recorded_calls[0].payload == "test-payload"


async def test_subscribe_special_characters(
    hass: HomeAssistant, mqtt_mock_entry: Callable[[], Awaitable[Any]], recorded_calls: List[ReceiveMessage], record_calls: MessageCallbackType
) -> None:
    """Test the subscription to topics with special characters."""
    await mqtt_mock_entry()
    topic: str = "/test-topic/$(.)[^]{-}"
    payload: str = "p4y.l[]a|> ?"
    await mqtt.async_subscribe(hass, topic, record_calls)
    async_fire_mqtt_message(hass, topic, payload)
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == topic
    assert recorded_calls[0].payload == payload


async def test_subscribe_same_topic(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient
) -> None:
    """Test subscribing to same topic twice and simulate retained messages.

    When subscribing to the same topic again, SUBSCRIBE must be sent to the broker again
    for it to resend any retained messages.
    """
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    calls_a: List[Any] = []
    calls_b: List[Any] = []

    @callback
    def _callback_a(msg: Any) -> None:
        calls_a.append(msg)

    @callback
    def _callback_b(msg: Any) -> None:
        calls_b.append(msg)

    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "test/state", _callback_a, qos=0)
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=False)
    await mock_debouncer.wait()
    assert len(calls_a) == 1
    mqtt_client_mock.subscribe.assert_called()
    calls_a = []
    mqtt_client_mock.reset_mock()
    await hass.async_block_till_done()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "test/state", _callback_b, qos=1)
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=False)
    await mock_debouncer.wait()
    assert len(calls_a) == 1
    assert len(calls_b) == 1
    mqtt_client_mock.subscribe.assert_called()


async def test_replaying_payload_same_topic(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient
) -> None:
    """Test replaying retained messages.

    When subscribing to the same topic again, SUBSCRIBE must be sent to the broker again
    for it to resend any retained messages for new subscriptions.
    Retained messages must only be replayed for new subscriptions, except
    when the MQTT client is reconnecting.
    """
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    calls_a: List[Any] = []
    calls_b: List[Any] = []

    @callback
    def _callback_a(msg: Any) -> None:
        calls_a.append(msg)

    @callback
    def _callback_b(msg: Any) -> None:
        calls_b.append(msg)

    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "test/state", _callback_a)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=True)
    assert len(calls_a) == 1
    mqtt_client_mock.subscribe.assert_called()
    calls_a.clear()
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "test/state", _callback_b)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=False)
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=True)
    assert len(calls_a) == 1
    assert help_assert_message(calls_a[0], "test/state", "online", qos=0, retain=False)
    assert len(calls_b) == 2
    assert help_assert_message(calls_b[0], "test/state", "online", qos=0, retain=False)
    assert help_assert_message(calls_b[1], "test/state", "online", qos=0, retain=True)
    mqtt_client_mock.subscribe.assert_called()
    calls_a = []
    calls_b = []
    mqtt_client_mock.reset_mock()
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=False)
    assert len(calls_a) == 1
    assert help_assert_message(calls_a[0], "test/state", "online", qos=0, retain=False)
    assert len(calls_b) == 1
    assert help_assert_message(calls_b[0], "test/state", "online", qos=0, retain=False)
    calls_a = []
    calls_b = []
    mqtt_client_mock.reset_mock()
    mqtt_client_mock.on_disconnect(None, None, 0, MockMqttReasonCode())
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(None, None, None, MockMqttReasonCode())
    await mock_debouncer.wait()
    mqtt_client_mock.subscribe.assert_called()
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=True)
    assert len(calls_a) == 1
    assert help_assert_message(calls_a[0], "test/state", "online", qos=0, retain=True)
    assert len(calls_b) == 1
    assert help_assert_message(calls_b[0], "test/state", "online", qos=0, retain=True)


async def test_replaying_payload_after_resubscribing(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient
) -> None:
    """Test replaying and filtering retained messages after resubscribing.

    When subscribing to the same topic again, SUBSCRIBE must be sent to the broker again
    for it to resend any retained messages for new subscriptions.
    Retained messages must only be replayed for new subscriptions, except
    when the MQTT client is reconnection.
    """
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    calls_a: List[Any] = []

    @callback
    def _callback_a(msg: Any) -> None:
        calls_a.append(msg)

    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    unsub: Callable[[], None] = await mqtt.async_subscribe(hass, "test/state", _callback_a)
    await mock_debouncer.wait()
    mqtt_client_mock.subscribe.assert_called()
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=True)
    assert help_assert_message(calls_a[0], "test/state", "online", qos=0, retain=True)
    calls_a.clear()
    async_fire_mqtt_message(hass, "test/state", "offline", qos=0, retain=False)
    assert help_assert_message(calls_a[0], "test/state", "offline", qos=0, retain=False)
    calls_a.clear()
    async_fire_mqtt_message(hass, "test/state", "offline", qos=0, retain=True)
    await hass.async_block_till_done()
    assert len(calls_a) == 0
    mock_debouncer.clear()
    unsub()
    unsub = await mqtt.async_subscribe(hass, "test/state", _callback_a)
    await mock_debouncer.wait()
    mqtt_client_mock.subscribe.assert_called()
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=True)
    assert help_assert_message(calls_a[0], "test/state", "online", qos=0, retain=True)


async def test_replaying_payload_wildcard_topic(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient
) -> None:
    """Test replaying retained messages.

    When we have multiple subscriptions to the same wildcard topic,
    SUBSCRIBE must be sent to the broker again
    for it to resend any retained messages for new subscriptions.
    Retained messages should only be replayed for new subscriptions, except
    when the MQTT client is reconnection.
    """
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    calls_a: List[Any] = []
    calls_b: List[Any] = []

    @callback
    def _callback_a(msg: Any) -> None:
        calls_a.append(msg)

    @callback
    def _callback_b(msg: Any) -> None:
        calls_b.append(msg)

    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "test/#", _callback_a)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, "test/state1", "new_value_1", qos=0, retain=True)
    async_fire_mqtt_message(hass, "test/state2", "new_value_2", qos=0, retain=True)
    assert len(calls_a) == 2
    mqtt_client_mock.subscribe.assert_called()
    calls_a = []
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "test/#", _callback_b)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, "test/state1", "initial_value_1", qos=0, retain=True)
    async_fire_mqtt_message(hass, "test/state2", "initial_value_2", qos=0, retain=True)
    assert len(calls_a) == 0
    assert len(calls_b) == 2
    mqtt_client_mock.subscribe.assert_called()
    calls_a = []
    calls_b = []
    mqtt_client_mock.reset_mock()
    async_fire_mqtt_message(hass, "test/state1", "update_value_1", qos=0, retain=False)
    async_fire_mqtt_message(hass, "test/state2", "update_value_2", qos=0, retain=False)
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
    async_fire_mqtt_message(hass, "test/state1", "update_value_1", qos=0, retain=True)
    async_fire_mqtt_message(hass, "test/state2", "update_value_2", qos=0, retain=True)
    assert len(calls_a) == 2
    assert len(calls_b) == 2


async def test_not_calling_unsubscribe_with_active_subscribers(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient, record_calls: MessageCallbackType
) -> None:
    """Test not calling unsubscribe() when other subscribers are active."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    unsub: Callable[[], None] = await mqtt.async_subscribe(hass, "test/state", record_calls, 2)
    await mqtt.async_subscribe(hass, "test/state", record_calls, 1)
    await mock_debouncer.wait()
    assert mqtt_client_mock.subscribe.called
    mock_debouncer.clear()
    unsub()
    await hass.async_block_till_done()
    await hass.async_block_till_done(wait_background_tasks=True)
    async_fire_time_changed(hass, utcnow() + timedelta(seconds=3))
    assert not mqtt_client_mock.unsubscribe.called
    assert not mock_debouncer.is_set()


async def test_not_calling_subscribe_when_unsubscribed_within_cooldown(
    hass: HomeAssistant, mock_debouncer: Any, mqtt_mock_entry: Callable[[], Awaitable[Any]], record_calls: MessageCallbackType
) -> None:
    """Test not calling subscribe() when it is unsubscribed.

    Make sure subscriptions are cleared if unsubscribed before
    the subscribe cool down period has ended.
    """
    mqtt_mock: Any = await mqtt_mock_entry()
    mqtt_client_mock: Any = mqtt_mock._mqttc
    await mock_debouncer.wait()
    mock_debouncer.clear()
    mqtt_client_mock.subscribe.reset_mock()
    unsub: Callable[[], None] = await mqtt.async_subscribe(hass, "test/state", record_calls)
    unsub()
    await mock_debouncer.wait()
    assert not mqtt_client_mock.subscribe.called


async def test_unsubscribe_race(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient
) -> None:
    """Test not calling unsubscribe() when other subscribers are active."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    calls_a: List[Any] = []
    calls_b: List[Any] = []

    @callback
    def _callback_a(msg: Any) -> None:
        calls_a.append(msg)

    @callback
    def _callback_b(msg: Any) -> None:
        calls_b.append(msg)

    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    unsub: Callable[[], None] = await mqtt.async_subscribe(hass, "test/state", _callback_a)
    unsub()
    await mqtt.async_subscribe(hass, "test/state", _callback_b)
    await mock_debouncer.wait()
    async_fire_mqtt_message(hass, "test/state", "online")
    assert not calls_a
    assert calls_b
    expected_calls_1 = [call.subscribe([("test/state", 0)]), call.unsubscribe("test/state"), call.subscribe([("test/state", 0)])]
    expected_calls_2 = [call.subscribe([("test/state", 0)]), call.subscribe([("test/state", 0)])]
    expected_calls_3 = [call.subscribe([("test/state", 0)])]
    assert mqtt_client_mock.mock_calls in (expected_calls_1, expected_calls_2, expected_calls_3)


@pytest.mark.parametrize(
    ("mqtt_config_entry_data", "mqtt_config_entry_options"),
    [({mqtt.CONF_BROKER: "mock-broker"}, {mqtt.CONF_DISCOVERY: False})],
)
async def test_restore_subscriptions_on_reconnect(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient, record_calls: MessageCallbackType
) -> None:
    """Test subscriptions are restored on reconnect."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "test/state", record_calls)
    async_fire_time_changed(hass, utcnow() + timedelta(seconds=3))
    await mock_debouncer.wait()
    assert ("test/state", 0) in help_all_subscribe_calls(mqtt_client_mock)
    mqtt_client_mock.reset_mock()
    mqtt_client_mock.on_disconnect(None, None, 0, MockMqttReasonCode())
    await mqtt.async_subscribe(hass, "test/other", record_calls)
    async_fire_time_changed(hass, utcnow() + timedelta(seconds=3))
    assert ("test/other", 0) not in help_all_subscribe_calls(mqtt_client_mock)
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(None, None, None, MockMqttReasonCode())
    await mock_debouncer.wait()
    assert ("test/state", 0) in help_all_subscribe_calls(mqtt_client_mock)
    assert ("test/other", 0) in help_all_subscribe_calls(mqtt_client_mock)


@pytest.mark.parametrize(
    ("mqtt_config_entry_data", "mqtt_config_entry_options"),
    [({mqtt.CONF_BROKER: "mock-broker"}, {mqtt.CONF_DISCOVERY: False})],
)
async def test_restore_all_active_subscriptions_on_reconnect(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient, record_calls: MessageCallbackType
) -> None:
    """Test active subscriptions are restored correctly on reconnect."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    unsub: Callable[[], None] = await mqtt.async_subscribe(hass, "test/state", record_calls, qos=2)
    await mqtt.async_subscribe(hass, "test/state", record_calls, qos=1)
    await mqtt.async_subscribe(hass, "test/state", record_calls, qos=0)
    await mock_debouncer.wait()
    expected = [call([("test/state", 2)])]
    assert mqtt_client_mock.subscribe.mock_calls == expected
    unsub()
    assert mqtt_client_mock.unsubscribe.call_count == 0
    mqtt_client_mock.on_disconnect(None, None, 0, MockMqttReasonCode())
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(None, None, None, MockMqttReasonCode())
    await mock_debouncer.wait()
    expected.append(call([("test/state", 1)]))
    for expected_call in expected:
        assert mqtt_client_mock.subscribe.hass_call(expected_call)


@pytest.mark.parametrize(
    ("mqtt_config_entry_data", "mqtt_config_entry_options"),
    [({mqtt.CONF_BROKER: "mock-broker"}, {mqtt.CONF_BIRTH_MESSAGE: {mqtt.ATTR_TOPIC: "birth", mqtt.ATTR_PAYLOAD: "birth", mqtt.ATTR_QOS: 0, mqtt.ATTR_RETAIN: False}})],
)
@patch("homeassistant.components.mqtt.client.INITIAL_SUBSCRIBE_COOLDOWN", 0.0)
@patch("homeassistant.components.mqtt.client.DISCOVERY_COOLDOWN", 0.0)
@patch("homeassistant.components.mqtt.client.SUBSCRIBE_COOLDOWN", 0.0)
async def test_custom_birth_message(
    hass: HomeAssistant,
    mock_debouncer: Any,
    mqtt_config_entry_data: dict[str, Any],
    mqtt_config_entry_options: dict[str, Any],
    mqtt_client_mock: Any,
) -> None:
    """Test sending birth message."""
    entry = MockConfigEntry(
        domain=mqtt.DOMAIN,
        data=mqtt_config_entry_data,
        options=mqtt_config_entry_options,
        version=mqtt.CONFIG_ENTRY_VERSION,
        minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    hass.config.components.add(mqtt.DOMAIN)
    assert await hass.config_entries.async_setup(entry.entry_id)
    mock_debouncer.clear()
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await mock_debouncer.wait()
    await hass.async_block_till_done(wait_background_tasks=True)
    mqtt_client_mock.publish.assert_called_with("birth", "birth", 0, False)


@pytest.mark.parametrize("mqtt_config_entry_options", [ENTRY_DEFAULT_BIRTH_MESSAGE])
async def test_default_birth_message(hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockHAClient) -> None:
    """Test sending birth message."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    await hass.async_block_till_done(wait_background_tasks=True)
    mqtt_client_mock.publish.assert_called_with("homeassistant/status", "online", 0, False)


@pytest.mark.parametrize(
    ("mqtt_config_entry_data", "mqtt_config_entry_options"),
    [({mqtt.CONF_BROKER: "mock-broker"}, {mqtt.CONF_BIRTH_MESSAGE: {}})],
)
@patch("homeassistant.components.mqtt.client.INITIAL_SUBSCRIBE_COOLDOWN", 0.0)
@patch("homeassistant.components.mqtt.client.DISCOVERY_COOLDOWN", 0.0)
@patch("homeassistant.components.mqtt.client.SUBSCRIBE_COOLDOWN", 0.0)
async def test_no_birth_message(
    hass: HomeAssistant,
    record_calls: MessageCallbackType,
    mock_debouncer: Any,
    mqtt_config_entry_data: dict[str, Any],
    mqtt_config_entry_options: dict[str, Any],
    mqtt_client_mock: Any,
) -> None:
    """Test disabling birth message."""
    entry = MockConfigEntry(
        domain=mqtt.DOMAIN,
        data=mqtt_config_entry_data,
        options=mqtt_config_entry_options,
        version=mqtt.CONFIG_ENTRY_VERSION,
        minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    hass.config.components.add(mqtt.DOMAIN)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await mock_debouncer.wait()
    await hass.async_block_till_done(wait_background_tasks=True)
    mqtt_client_mock.publish.assert_not_called()
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "homeassistant/some-topic", record_calls)
    await mock_debouncer.wait()
    mqtt_client_mock.subscribe.assert_called()


@pytest.mark.parametrize(
    ("mqtt_config_entry_data", "mqtt_config_entry_options"),
    [({mqtt.CONF_BROKER: "mock-broker"}, ENTRY_DEFAULT_BIRTH_MESSAGE)],
)
@patch("homeassistant.components.mqtt.client.DISCOVERY_COOLDOWN", 0.2)
async def test_delayed_birth_message(
    hass: HomeAssistant,
    mqtt_config_entry_data: dict[str, Any],
    mqtt_config_entry_options: dict[str, Any],
    mqtt_client_mock: Any,
) -> None:
    """Test sending birth message does not happen until Home Assistant starts."""
    hass.set_state(CoreState.starting)
    await hass.async_block_till_done()
    birth: asyncio.Event = asyncio.Event()
    entry = MockConfigEntry(
        domain=mqtt.DOMAIN,
        data=mqtt_config_entry_data,
        options=mqtt_config_entry_options,
        version=mqtt.CONFIG_ENTRY_VERSION,
        minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    hass.config.components.add(mqtt.DOMAIN)
    assert await hass.config_entries.async_setup(entry.entry_id)

    @callback
    def wait_birth(msg: Any) -> None:
        """Handle birth message."""
        birth.set()

    await mqtt.async_subscribe(hass, "homeassistant/status", wait_birth)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(birth.wait(), 0.05)
    assert not mqtt_client_mock.publish.called
    assert not birth.is_set()
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STARTED)
    await birth.wait()
    mqtt_client_mock.publish.assert_called_with("homeassistant/status", "online", 0, False)


@pytest.mark.parametrize("mqtt_config_entry_options", [ENTRY_DEFAULT_BIRTH_MESSAGE])
async def test_subscription_done_when_birth_message_is_sent(
    setup_with_birth_msg_client_mock: MqttMockHAClient,
) -> None:
    """Test sending birth message until initial subscription has been completed."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    subscribe_calls: List[Any] = help_all_subscribe_calls(mqtt_client_mock)
    for component in SUPPORTED_COMPONENTS:
        assert (f"homeassistant/{component}/+/config", 0) in subscribe_calls
        assert (f"homeassistant/{component}/+/+/config", 0) in subscribe_calls
    mqtt_client_mock.publish.assert_called_with("homeassistant/status", "online", 0, False)


@pytest.mark.parametrize(
    ("mqtt_config_entry_data", "mqtt_config_entry_options"),
    [({mqtt.CONF_BROKER: "mock-broker"}, {mqtt.CONF_WILL_MESSAGE: {mqtt.ATTR_TOPIC: "death", mqtt.ATTR_PAYLOAD: "death", mqtt.ATTR_QOS: 0, mqtt.ATTR_RETAIN: False}})],
)
async def test_custom_will_message(
    hass: HomeAssistant,
    mqtt_config_entry_data: dict[str, Any],
    mqtt_config_entry_options: dict[str, Any],
    mqtt_client_mock: Any,
) -> None:
    """Test will message."""
    entry = MockConfigEntry(
        domain=mqtt.DOMAIN,
        data=mqtt_config_entry_data,
        options=mqtt_config_entry_options,
        version=mqtt.CONFIG_ENTRY_VERSION,
        minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    hass.config.components.add(mqtt.DOMAIN)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    mqtt_client_mock.will_set.assert_called_with(topic="death", payload="death", qos=0, retain=False)


async def test_default_will_message(setup_with_birth_msg_client_mock: MqttMockHAClient) -> None:
    """Test will message."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    mqtt_client_mock.will_set.assert_called_with(topic="homeassistant/status", payload="offline", qos=0, retain=False)


@pytest.mark.parametrize(
    ("mqtt_config_entry_data", "mqtt_config_entry_options"),
    [({mqtt.CONF_BROKER: "mock-broker"}, {mqtt.CONF_WILL_MESSAGE: {}})],
)
async def test_no_will_message(
    hass: HomeAssistant,
    mqtt_config_entry_data: dict[str, Any],
    mqtt_config_entry_options: dict[str, Any],
    mqtt_client_mock: Any,
) -> None:
    """Test will message."""
    entry = MockConfigEntry(
        domain=mqtt.DOMAIN,
        data=mqtt_config_entry_data,
        options=mqtt_config_entry_options,
        version=mqtt.CONFIG_ENTRY_VERSION,
        minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION,
    )
    entry.add_to_hass(hass)
    hass.config.components.add(mqtt.DOMAIN)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    mqtt_client_mock.will_set.assert_not_called()


@pytest.mark.parametrize("mqtt_config_entry_options", [ENTRY_DEFAULT_BIRTH_MESSAGE | {mqtt.CONF_DISCOVERY: False}])
async def test_mqtt_subscribes_topics_on_connect(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient, record_calls: MessageCallbackType
) -> None:
    """Test subscription to topic on connect."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "topic/test", record_calls)
    await mqtt.async_subscribe(hass, "home/sensor", record_calls, 2)
    await mqtt.async_subscribe(hass, "still/pending", record_calls)
    await mqtt.async_subscribe(hass, "still/pending", record_calls, 1)
    await mock_debouncer.wait()
    mqtt_client_mock.on_disconnect(Mock(), None, 0, MockMqttReasonCode())
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(Mock(), None, 0, MockMqttReasonCode())
    await mock_debouncer.wait()
    subscribe_calls: List[Any] = help_all_subscribe_calls(mqtt_client_mock)
    assert ("topic/test", 0) in subscribe_calls
    assert ("home/sensor", 2) in subscribe_calls
    assert ("still/pending", 1) in subscribe_calls


@pytest.mark.parametrize("mqtt_config_entry_options", [ENTRY_DEFAULT_BIRTH_MESSAGE])
async def test_mqtt_subscribes_wildcard_topics_in_correct_order(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient, record_calls: MessageCallbackType
) -> None:
    """Test subscription to wildcard topics on connect in the order of subscription."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "integration/test#", record_calls)
    await mqtt.async_subscribe(hass, "integration/kitchen_sink#", record_calls)
    await mock_debouncer.wait()

    def _assert_subscription_order() -> None:
        discovery_subscribes: List[str] = [f"homeassistant/{platform}/+/config" for platform in SUPPORTED_COMPONENTS]
        discovery_subscribes.extend([f"homeassistant/{platform}/+/+/config" for platform in SUPPORTED_COMPONENTS])
        discovery_subscribes.extend(["homeassistant/device/+/config", "homeassistant/device/+/+/config"])
        discovery_subscribes.extend(["integration/test#", "integration/kitchen_sink#"])
        expected_discovery_subscribes: List[str] = discovery_subscribes.copy()
        actual_subscribes: List[str] = [sub for sub, _ in help_all_subscribe_calls(mqtt_client_mock) if discovery_subscribes and discovery_subscribes[0] == sub and discovery_subscribes.pop(0)]
        assert len(discovery_subscribes) == 0
        assert actual_subscribes == expected_discovery_subscribes

    _assert_subscription_order()
    mqtt_client_mock.on_disconnect(Mock(), None, 0, MockMqttReasonCode())
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(Mock(), None, 0, MockMqttReasonCode())
    await mock_debouncer.wait()
    _assert_subscription_order()


@pytest.mark.parametrize("mqtt_config_entry_options", [ENTRY_DEFAULT_BIRTH_MESSAGE | {mqtt.CONF_DISCOVERY: False}])
async def test_mqtt_discovery_not_subscribes_when_disabled(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient
) -> None:
    """Test discovery subscriptions not performend when discovery is disabled."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    await mock_debouncer.wait()
    subscribe_calls: List[Any] = help_all_subscribe_calls(mqtt_client_mock)
    for component in SUPPORTED_COMPONENTS:
        assert (f"homeassistant/{component}/+/config", 0) not in subscribe_calls
        assert (f"homeassistant/{component}/+/+/config", 0) not in subscribe_calls
    mqtt_client_mock.on_disconnect(Mock(), None, 0, MockMqttReasonCode())
    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    mqtt_client_mock.on_connect(Mock(), None, 0, MockMqttReasonCode())
    await mock_debouncer.wait()
    subscribe_calls = help_all_subscribe_calls(mqtt_client_mock)
    for component in SUPPORTED_COMPONENTS:
        assert (f"homeassistant/{component}/+/config", 0) not in subscribe_calls
        assert (f"homeassistant/{component}/+/+/config", 0) not in subscribe_calls


@pytest.mark.parametrize("mqtt_config_entry_options", [ENTRY_DEFAULT_BIRTH_MESSAGE])
async def test_mqtt_subscribes_in_single_call(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient, record_calls: MessageCallbackType
) -> None:
    """Test bundled client subscription to topic."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    mqtt_client_mock.subscribe.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "topic/test", record_calls)
    await mqtt.async_subscribe(hass, "home/sensor", record_calls)
    await mock_debouncer.wait()
    assert mqtt_client_mock.subscribe.call_count == 1
    assert mqtt_client_mock.subscribe.mock_calls[0][1][0] in [[("topic/test", 0), ("home/sensor", 0)], [("home/sensor", 0), ("topic/test", 0)]]


@pytest.mark.parametrize("mqtt_config_entry_options", [ENTRY_DEFAULT_BIRTH_MESSAGE])
@patch("homeassistant.components.mqtt.client.MAX_SUBSCRIBES_PER_CALL", 2)
@patch("homeassistant.components.mqtt.client.MAX_UNSUBSCRIBES_PER_CALL", 2)
async def test_mqtt_subscribes_and_unsubscribes_in_chunks(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient, record_calls: MessageCallbackType
) -> None:
    """Test chunked client subscriptions."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    mqtt_client_mock.subscribe.reset_mock()
    unsub_tasks: List[Callable[[], None]] = []
    mock_debouncer.clear()
    unsub_tasks.append(await mqtt.async_subscribe(hass, "topic/test1", record_calls))
    unsub_tasks.append(await mqtt.async_subscribe(hass, "home/sensor1", record_calls))
    unsub_tasks.append(await mqtt.async_subscribe(hass, "topic/test2", record_calls))
    unsub_tasks.append(await mqtt.async_subscribe(hass, "home/sensor2", record_calls))
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


@pytest.mark.parametrize("exception", [OSError("Connection error"), paho_mqtt.WebsocketConnectionError("Connection error")])
async def test_auto_reconnect(
    hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockHAClient, caplog: Any, exception: Exception
) -> None:
    """Test reconnection is automatically done."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    assert mqtt_client_mock.connect.call_count == 1
    mqtt_client_mock.reconnect.reset_mock()
    mqtt_client_mock.disconnect()
    mqtt_client_mock.on_disconnect(None, None, 0, MockMqttReasonCode())
    await hass.async_block_till_done()
    mqtt_client_mock.reconnect.side_effect = exception("foo")
    async_fire_time_changed(hass, utcnow() + timedelta(seconds=RECONNECT_INTERVAL_SECONDS))
    await hass.async_block_till_done()
    assert len(mqtt_client_mock.reconnect.mock_calls) == 1
    assert "Error re-connecting to MQTT server due to exception: foo" in caplog.text
    mqtt_client_mock.reconnect.side_effect = None
    async_fire_time_changed(hass, utcnow() + timedelta(seconds=RECONNECT_INTERVAL_SECONDS))
    await hass.async_block_till_done()
    assert len(mqtt_client_mock.reconnect.mock_calls) == 2
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    mqtt_client_mock.disconnect()
    mqtt_client_mock.on_disconnect(None, None, 0, MockMqttReasonCode())
    await hass.async_block_till_done()
    async_fire_time_changed(hass, utcnow() + timedelta(seconds=RECONNECT_INTERVAL_SECONDS))
    await hass.async_block_till_done()
    assert len(mqtt_client_mock.reconnect.mock_calls) == 2


async def test_server_sock_connect_and_disconnect(
    hass: HomeAssistant, mock_debouncer: Any, setup_with_birth_msg_client_mock: MqttMockHAClient, recorded_calls: List[Any], record_calls: MessageCallbackType
) -> None:
    """Test handling the socket connected and disconnected."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    assert mqtt_client_mock.connect.call_count == 1
    mqtt_client_mock.loop_misc.return_value = paho_mqtt.MQTT_ERR_SUCCESS
    client, server = socket.socketpair(family=socket.AF_UNIX, type=socket.SOCK_STREAM, proto=0)
    client.setblocking(False)
    server.setblocking(False)
    mqtt_client_mock.on_socket_open(mqtt_client_mock, None, client)
    mqtt_client_mock.on_socket_register_write(mqtt_client_mock, None, client)
    await hass.async_block_till_done()
    server.close()
    mock_debouncer.clear()
    unsub: Callable[[], None] = await mqtt.async_subscribe(hass, "test-topic", record_calls)
    await mock_debouncer.wait()
    mqtt_client_mock.loop_misc.return_value = paho_mqtt.MQTT_ERR_CONN_LOST
    mqtt_client_mock.on_socket_unregister_write(mqtt_client_mock, None, client)
    mqtt_client_mock.on_socket_close(mqtt_client_mock, None, client)
    mqtt_client_mock.on_disconnect(mqtt_client_mock, None, None, MockMqttReasonCode())
    await hass.async_block_till_done()
    mock_debouncer.clear()
    unsub()
    await hass.async_block_till_done()
    assert not mock_debouncer.is_set()
    assert len(recorded_calls) == 0


async def test_server_sock_buffer_size(
    hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockHAClient, caplog: Any
) -> None:
    """Test handling the socket buffer size fails."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    assert mqtt_client_mock.connect.call_count == 1
    mqtt_client_mock.loop_misc.return_value = paho_mqtt.MQTT_ERR_SUCCESS
    client, server = socket.socketpair(family=socket.AF_UNIX, type=socket.SOCK_STREAM, proto=0)
    client.setblocking(False)
    server.setblocking(False)
    with patch.object(client, "setsockopt", side_effect=OSError("foo")):
        mqtt_client_mock.on_socket_open(mqtt_client_mock, None, client)
        mqtt_client_mock.on_socket_register_write(mqtt_client_mock, None, client)
        await hass.async_block_till_done()
    assert "Unable to increase the socket buffer size" in caplog.text


async def test_server_sock_buffer_size_with_websocket(
    hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockHAClient, caplog: Any
) -> None:
    """Test handling the socket buffer size fails."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    assert mqtt_client_mock.connect.call_count == 1
    mqtt_client_mock.loop_misc.return_value = paho_mqtt.MQTT_ERR_SUCCESS
    client, server = socket.socketpair(family=socket.AF_UNIX, type=socket.SOCK_STREAM, proto=0)
    client.setblocking(False)
    server.setblocking(False)

    class FakeWebsocket(paho_mqtt._WebsocketWrapper):
        def _do_handshake(self, *args: Any, **kwargs: Any) -> None:
            pass

    wrapped_socket = FakeWebsocket(client, "127.0.01", 1, False, "/", None)
    with patch.object(client, "setsockopt", side_effect=OSError("foo")):
        mqtt_client_mock.on_socket_open(mqtt_client_mock, None, wrapped_socket)
        mqtt_client_mock.on_socket_register_write(mqtt_client_mock, None, wrapped_socket)
        await hass.async_block_till_done()
    assert "Unable to increase the socket buffer size" in caplog.text


async def test_client_sock_failure_after_connect(
    hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockHAClient, recorded_calls: List[Any], record_calls: MessageCallbackType
) -> None:
    """Test handling the socket connected and disconnected."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    assert mqtt_client_mock.connect.call_count == 1
    mqtt_client_mock.loop_misc.return_value = paho_mqtt.MQTT_ERR_SUCCESS
    client, server = socket.socketpair(family=socket.AF_UNIX, type=socket.SOCK_STREAM, proto=0)
    client.setblocking(False)
    server.setblocking(False)
    mqtt_client_mock.on_socket_open(mqtt_client_mock, None, client)
    mqtt_client_mock.on_socket_register_writer(mqtt_client_mock, None, client)
    await hass.async_block_till_done()
    mqtt_client_mock.loop_write.side_effect = OSError("foo")
    client.close()
    assert mqtt_client_mock.connect.call_count == 1
    unsub: Callable[[], None] = await mqtt.async_subscribe(hass, "test-topic", record_calls)
    async_fire_time_changed(hass, utcnow() + timedelta(seconds=5))
    await hass.async_block_till_done()
    unsub()
    assert len(recorded_calls) == 0


async def test_loop_write_failure(
    hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockHAClient, caplog: Any
) -> None:
    """Test handling the socket connected and disconnected."""
    mqtt_client_mock: Any = setup_with_birth_msg_client_mock
    assert mqtt_client_mock.connect.call_count == 1
    mqtt_client_mock.loop_misc.return_value = paho_mqtt.MQTT_ERR_SUCCESS
    client, server = socket.socketpair(family=socket.AF_UNIX, type=socket.SOCK_STREAM, proto=0)
    client.setblocking(False)
    server.setblocking(False)
    mqtt_client_mock.on_socket_open(mqtt_client_mock, None, client)
    mqtt_client_mock.on_socket_register_write(mqtt_client_mock, None, client)
    mqtt_client_mock.loop_write.return_value = paho_mqtt.MQTT_ERR_CONN_LOST
    mqtt_client_mock.loop_read.return_value = paho_mqtt.MQTT_ERR_CONN_LOST
    try:
        for _ in range(1000):
            server.send(b"long" * 100)
    except BlockingIOError:
        pass
    server.close()
    await hass.async_block_till_done()
    await hass.async_block_till_done()
    await hass.async_block_till_done()
    assert "Error returned from MQTT server: The connection was lost." in caplog.text
