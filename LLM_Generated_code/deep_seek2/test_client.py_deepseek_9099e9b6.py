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
from homeassistant.const import (
    CONF_PROTOCOL,
    EVENT_HOMEASSISTANT_STARTED,
    EVENT_HOMEASSISTANT_STOP,
    UnitOfTemperature,
)
from homeassistant.core import CALLBACK_TYPE, CoreState, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util.dt import utcnow

from .conftest import ENTRY_DEFAULT_BIRTH_MESSAGE
from .test_common import help_all_subscribe_calls

from tests.common import (
    MockConfigEntry,
    MockMqttReasonCode,
    async_fire_mqtt_message,
    async_fire_time_changed,
)
from tests.typing import MqttMockHAClient, MqttMockHAClientGenerator, MqttMockPahoClient


def help_assert_message(
    msg: ReceiveMessage,
    topic: Optional[str] = None,
    payload: Optional[str] = None,
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
    hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockPahoClient
) -> None:
    """Test if client is connected after mqtt init on bootstrap."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    assert mqtt_client_mock.connect.call_count == 1


async def test_mqtt_does_not_disconnect_on_home_assistant_stop(
    hass: HomeAssistant,
    mock_debouncer: asyncio.Event,
    setup_with_birth_msg_client_mock: MqttMockPahoClient,
) -> None:
    """Test if client is not disconnected on HA stop."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    hass.bus.fire(EVENT_HOMEASSISTANT_STOP)
    await mock_debouncer.wait()
    assert mqtt_client_mock.disconnect.call_count == 0


async def test_mqtt_await_ack_at_disconnect(hass: HomeAssistant) -> None:
    """Test if ACK is awaited correctly when disconnecting."""

    class FakeInfo:
        """Returns a simulated client publish response."""

        mid = 100
        rc = 0

    with patch(
        "homeassistant.components.mqtt.async_client.AsyncMQTTClient"
    ) as mock_client:
        mqtt_client = mock_client.return_value
        mqtt_client.connect = MagicMock(
            return_value=0,
            side_effect=lambda *args, **kwargs: hass.loop.call_soon_threadsafe(
                mqtt_client.on_connect, mqtt_client, None, 0, MockMqttReasonCode()
            ),
        )
        mqtt_client.publish = MagicMock(return_value=FakeInfo())
        entry = MockConfigEntry(
            domain=mqtt.DOMAIN,
            data={
                "certificate": "auto",
                mqtt.CONF_BROKER: "test-broker",
                mqtt.CONF_DISCOVERY: False,
            },
            version=mqtt.CONFIG_ENTRY_VERSION,
            minor_version=mqtt.CONFIG_ENTRY_MINOR_VERSION,
        )
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)

        mqtt_client = mock_client.return_value

        # publish from MQTT client without awaiting
        hass.async_create_task(
            mqtt.async_publish(hass, "test-topic", "some-payload", 0, False)
        )
        await asyncio.sleep(0)
        # Simulate late ACK callback from client with mid 100
        mqtt_client.on_publish(0, 0, 100, MockMqttReasonCode(), None)
        # disconnect the MQTT client
        await hass.async_stop()
        await hass.async_block_till_done()
        # assert the payload was sent through the client
        assert mqtt_client.publish.called
        assert mqtt_client.publish.call_args[0] == (
            "test-topic",
            "some-payload",
            0,
            False,
        )
        await hass.async_block_till_done(wait_background_tasks=True)


@pytest.mark.parametrize("mqtt_config_entry_options", [ENTRY_DEFAULT_BIRTH_MESSAGE])
async def test_publish(
    hass: HomeAssistant, setup_with_birth_msg_client_mock: MqttMockPahoClient
) -> None:
    """Test the publish function."""
    publish_mock: MagicMock = setup_with_birth_msg_client_mock.publish
    await mqtt.async_publish(hass, "test-topic", "test-payload")
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == (
        "test-topic",
        "test-payload",
        0,
        False,
    )
    publish_mock.reset_mock()

    await mqtt.async_publish(hass, "test-topic", "test-payload", 2, True)
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == (
        "test-topic",
        "test-payload",
        2,
        True,
    )
    publish_mock.reset_mock()

    mqtt.publish(hass, "test-topic2", "test-payload2")
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == (
        "test-topic2",
        "test-payload2",
        0,
        False,
    )
    publish_mock.reset_mock()

    mqtt.publish(hass, "test-topic2", "test-payload2", 2, True)
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == (
        "test-topic2",
        "test-payload2",
        2,
        True,
    )
    publish_mock.reset_mock()

    # test binary pass-through
    mqtt.publish(
        hass,
        "test-topic3",
        b"\xde\xad\xbe\xef",
        0,
        False,
    )
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == (
        "test-topic3",
        b"\xde\xad\xbe\xef",
        0,
        False,
    )
    publish_mock.reset_mock()

    # test null payload
    mqtt.publish(
        hass,
        "test-topic3",
        None,
        0,
        False,
    )
    await hass.async_block_till_done()
    assert publish_mock.called
    assert publish_mock.call_args[0] == (
        "test-topic3",
        None,
        0,
        False,
    )

    publish_mock.reset_mock()


async def test_convert_outgoing_payload(hass: HomeAssistant) -> None:
    """Test the converting of outgoing MQTT payloads without template."""
    command_template = mqtt.MqttCommandTemplate(None)
    assert command_template.async_render(b"\xde\xad\xbe\xef") == b"\xde\xad\xbe\xef"
    assert (
        command_template.async_render("b'\\xde\\xad\\xbe\\xef'")
        == "b'\\xde\\xad\\xbe\\xef'"
    )
    assert command_template.async_render(1234) == 1234
    assert command_template.async_render(1234.56) == 1234.56
    assert command_template.async_render(None) is None


async def test_all_subscriptions_run_when_decode_fails(
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
) -> None:
    """Test all other subscriptions still run when decode fails for one."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic", record_calls, encoding="ascii")
    await mqtt.async_subscribe(hass, "test-topic", record_calls)

    async_fire_mqtt_message(hass, "test-topic", UnitOfTemperature.CELSIUS)

    await hass.async_block_till_done()
    assert len(recorded_calls) == 1


async def test_subscribe_topic(
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
) -> None:
    """Test the subscription of a topic."""
    await mqtt_mock_entry()
    unsub = await mqtt.async_subscribe(hass, "test-topic", record_calls)

    async_fire_mqtt_message(hass, "test-topic", "test-payload")

    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == "test-topic"
    assert recorded_calls[0].payload == "test-payload"

    unsub()

    async_fire_mqtt_message(hass, "test-topic", "test-payload")

    await hass.async_block_till_done()
    assert len(recorded_calls) == 1

    # Cannot unsubscribe twice
    with pytest.raises(HomeAssistantError):
        unsub()


@pytest.mark.usefixtures("mqtt_mock_entry")
async def test_subscribe_topic_not_initialize(
    hass: HomeAssistant, record_calls: MessageCallbackType
) -> None:
    """Test the subscription of a topic when MQTT was not initialized."""
    with pytest.raises(
        HomeAssistantError, match=r".*make sure MQTT is set up correctly"
    ):
        await mqtt.async_subscribe(hass, "test-topic", record_calls)


async def test_subscribe_mqtt_config_entry_disabled(
    hass: HomeAssistant, mqtt_mock: MqttMockHAClient, record_calls: MessageCallbackType
) -> None:
    """Test the subscription of a topic when MQTT config entry is disabled."""
    mqtt_mock.connected = True

    mqtt_config_entry = hass.config_entries.async_entries(mqtt.DOMAIN)[0]

    mqtt_config_entry_state = mqtt_config_entry.state
    assert mqtt_config_entry_state is ConfigEntryState.LOADED

    assert await hass.config_entries.async_unload(mqtt_config_entry.entry_id)
    mqtt_config_entry_state = mqtt_config_entry.state
    assert mqtt_config_entry_state is ConfigEntryState.NOT_LOADED

    await hass.config_entries.async_set_disabled_by(
        mqtt_config_entry.entry_id, ConfigEntryDisabler.USER
    )
    mqtt_mock.connected = False

    with pytest.raises(HomeAssistantError, match=r".*MQTT is not enabled"):
        await mqtt.async_subscribe(hass, "test-topic", record_calls)


async def test_subscribe_and_resubscribe(
    hass: HomeAssistant,
    mock_debouncer: asyncio.Event,
    setup_with_birth_msg_client_mock: MqttMockPahoClient,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
) -> None:
    """Test resubscribing within the debounce time."""
    mqtt_client_mock = setup_with_birth_msg_client_mock
    with (
        patch("homeassistant.components.mqtt.client.SUBSCRIBE_COOLDOWN", 0.4),
        patch("homeassistant.components.mqtt.client.UNSUBSCRIBE_COOLDOWN", 0.4),
    ):
        mock_debouncer.clear()
        unsub = await mqtt.async_subscribe(hass, "test-topic", record_calls)
        # This unsub will be un-done with the following subscribe
        # unsubscribe should not be called at the broker
        unsub()
        unsub = await mqtt.async_subscribe(hass, "test-topic", record_calls)
        await mock_debouncer.wait()
        mock_debouncer.clear()

        async_fire_mqtt_message(hass, "test-topic", "test-payload")

        assert len(recorded_calls) == 1
        assert recorded_calls[0].topic == "test-topic"
        assert recorded_calls[0].payload == "test-payload"
        # assert unsubscribe was not called
        mqtt_client_mock.unsubscribe.assert_not_called()

        mock_debouncer.clear()
        unsub()

        await mock_debouncer.wait()
        mqtt_client_mock.unsubscribe.assert_called_once_with(["test-topic"])


async def test_subscribe_topic_non_async(
    hass: HomeAssistant,
    mock_debouncer: asyncio.Event,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
) -> None:
    """Test the subscription of a topic using the non-async function."""
    await mqtt_mock_entry()
    await mock_debouncer.wait()
    mock_debouncer.clear()
    unsub = await hass.async_add_executor_job(
        mqtt.subscribe, hass, "test-topic", record_calls
    )
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
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    record_calls: MessageCallbackType,
) -> None:
    """Test the subscription of a topic."""
    await mqtt_mock_entry()
    with pytest.raises(HomeAssistantError):
        await mqtt.async_subscribe(hass, 55, record_calls)  # type: ignore[arg-type]


async def test_subscribe_topic_not_match(
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
) -> None:
    """Test if subscribed topic is not a match."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic", record_calls)

    async_fire_mqtt_message(hass, "another-test-topic", "test-payload")

    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_level_wildcard(
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
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
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic/+/on", record_calls)

    async_fire_mqtt_message(hass, "test-topic/bier", "test-payload")

    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_level_wildcard_root_topic_no_subtree_match(
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic/#", record_calls)

    async_fire_mqtt_message(hass, "test-topic-123", "test-payload")

    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_subtree_wildcard_subtree_topic(
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
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
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
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
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "test-topic/#", record_calls)

    async_fire_mqtt_message(hass, "another-test-topic", "test-payload")

    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_level_wildcard_and_wildcard_root_topic(
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
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
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
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
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "+/test-topic/#", record_calls)

    async_fire_mqtt_message(hass, "hi/here-iam/test-topic", "test-payload")

    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_level_wildcard_and_wildcard_no_match(
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
) -> None:
    """Test the subscription of wildcard topics."""
    await mqtt_mock_entry()
    await mqtt.async_subscribe(hass, "+/test-topic/#", record_calls)

    async_fire_mqtt_message(hass, "hi/another-test-topic", "test-payload")

    await hass.async_block_till_done()
    assert len(recorded_calls) == 0


async def test_subscribe_topic_sys_root(
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
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
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
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
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
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
    hass: HomeAssistant,
    mqtt_mock_entry: MqttMockHAClientGenerator,
    recorded_calls: List[ReceiveMessage],
    record_calls: MessageCallbackType,
) -> None:
    """Test the subscription to topics with special characters."""
    await mqtt_mock_entry()
    topic = "/test-topic/$(.)[^]{-}"
    payload = "p4y.l[]a|> ?"

    await mqtt.async_subscribe(hass, topic, record_calls)

    async_fire_mqtt_message(hass, topic, payload)
    await hass.async_block_till_done()
    assert len(recorded_calls) == 1
    assert recorded_calls[0].topic == topic
    assert recorded_calls[0].payload == payload


async def test_subscribe_same_topic(
    hass: HomeAssistant,
    mock_debouncer: asyncio.Event,
    setup_with_birth_msg_client_mock: MqttMockPahoClient,
) -> None:
    """Test subscribing to same topic twice and simulate retained messages.

    When subscribing to the same topic again, SUBSCRIBE must be sent to the broker again
    for it to resend any retained messages.
    """
    mqtt_client_mock = setup_with_birth_msg_client_mock
    calls_a: List[ReceiveMessage] = []
    calls_b: List[ReceiveMessage] = []

    @callback
    def _callback_a(msg: ReceiveMessage) -> None:
        calls_a.append(msg)

    @callback
    def _callback_b(msg: ReceiveMessage) -> None:
        calls_b.append(msg)

    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "test/state", _callback_a, qos=0)
    # Simulate a non retained message after the first subscription
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=False)
    await mock_debouncer.wait()
    assert len(calls_a) == 1
    mqtt_client_mock.subscribe.assert_called()
    calls_a = []
    mqtt_client_mock.reset_mock()

    await hass.async_block_till_done()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "test/state", _callback_b, qos=1)
    # Simulate an other non retained message after the second subscription
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=False)
    await mock_debouncer.wait()
    # Both subscriptions should receive updates
    assert len(calls_a) == 1
    assert len(calls_b) == 1
    mqtt_client_mock.subscribe.assert_called()


async def test_replaying_payload_same_topic(
    hass: HomeAssistant,
    mock_debouncer: asyncio.Event,
    setup_with_birth_msg_client_mock: MqttMockPahoClient,
) -> None:
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
    def _callback_a(msg: ReceiveMessage) -> None:
        calls_a.append(msg)

    @callback
    def _callback_b(msg: ReceiveMessage) -> None:
        calls_b.append(msg)

    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "test/state", _callback_a)
    await mock_debouncer.wait()
    async_fire_mqtt_message(
        hass, "test/state", "online", qos=0, retain=True
    )  # Simulate a (retained) message played back
    assert len(calls_a) == 1
    mqtt_client_mock.subscribe.assert_called()
    calls_a = []
    mqtt_client_mock.reset_mock()

    mock_debouncer.clear()
    await mqtt.async_subscribe(hass, "test/state", _callback_b)
    await mock_debouncer.wait()

    # Simulate edge case where non retained message was received
    # after subscription at HA but before the debouncer delay was passed.
    # The message without retain flag directly after a subscription should
    # be processed by both subscriptions.
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=False)

    # Simulate a (retained) message played back on new subscriptions
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=True)

    # The current subscription only received the message without retain flag
    assert len(calls_a) == 1
    assert help_assert_message(calls_a[0], "test/state", "online", qos=0, retain=False)
    # The retained message playback should only be processed by the new subscription.
    # The existing subscription already got the latest update, hence the existing
    # subscription should not receive the replayed (retained) message.
    # Messages without retain flag are received on both subscriptions.
    assert len(calls_b) == 2
    assert help_assert_message(calls_b[0], "test/state", "online", qos=0, retain=False)
    assert help_assert_message(calls_b[1], "test/state", "online", qos=0, retain=True)
    mqtt_client_mock.subscribe.assert_called()

    calls_a = []
    calls_b = []
    mqtt_client_mock.reset_mock()

    # Simulate new message played back on new subscriptions
    # After connecting the retain flag will not be set, even if the
    # payload published was retained, we cannot see that
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=False)
    assert len(calls_a) == 1
    assert help_assert_message(calls_a[0], "test/state", "online", qos=0, retain=False)
    assert len(calls_b) == 1
    assert help_assert_message(calls_b[0], "test/state", "online", qos=0, retain=False)

    # Now simulate the broker was disconnected shortly
    calls_a = []
    calls_b = []
    mqtt_client_mock.reset_mock()
    mqtt_client_mock.on_disconnect(None, None, 0, MockMqttReasonCode())

    mock_debouncer.clear()
    mqtt_client_mock.on_connect(None, None, None, MockMqttReasonCode())
    await mock_debouncer.wait()
    mqtt_client_mock.subscribe.assert_called()
    # Simulate a (retained) message played back after reconnecting
    async_fire_mqtt_message(hass, "test/state", "online", qos=0, retain=True)
    # Both subscriptions now should replay the retained message
    assert len(calls_a) == 1
    assert help_assert_message(calls_a[0], "test/state", "online", qos=0, retain=True)
    assert len(calls_b) == 1
    assert help_assert_message(calls_b[0], "test/state", "online", qos=0, retain=True)


async def test_replaying_payload_after_resubscribing(
    hass: HomeAssistant,
    mock_debouncer: asyncio.Event,
    setup_with_birth_msg_client_mock: MqttMockPahoClient,
) -> None:
    """Test replaying and filtering retained messages after resubscribing.

    When subscribing to the same topic again, SUBSCRIBE must be sent to the broker again
    for it to resend any retained messages for new subscriptions.
    Retained messages must only be replayed for new subscriptions, except
    when the MQTT client is reconnection.
    """
    mqtt_client_mock = setup_with_birth_msg_client_mock
    calls_a: List[ReceiveMessage] = []

    @callback
    def _callback_a(msg: ReceiveMessage) -> None:
        calls_a.append(msg)

    mqtt_client_mock.reset_mock()
    mock_debouncer.clear()
    unsub = await mqtt.async_subscribe(hass, "test/state", _callback_a)
    await mock_debouncer.wait()
    mqtt_client_mock.subscribe.assert