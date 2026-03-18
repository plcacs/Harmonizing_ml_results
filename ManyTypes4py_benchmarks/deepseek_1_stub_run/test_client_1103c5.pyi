```python
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union
from unittest.mock import MagicMock
from unittest.mock import Mock
import asyncio
import paho.mqtt.client as paho_mqtt
import pytest
from homeassistant.components import mqtt
from homeassistant.components.mqtt.models import MessageCallbackType
from homeassistant.components.mqtt.models import ReceiveMessage
from homeassistant.config_entries import ConfigEntryDisabler
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_PROTOCOL
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.const import UnitOfTemperature
from homeassistant.core import CALLBACK_TYPE
from homeassistant.core import CoreState
from homeassistant.core import HomeAssistant
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util.dt import datetime
from tests.common import MockConfigEntry
from tests.common import MockMqttReasonCode
from tests.typing import MqttMockHAClient
from tests.typing import MqttMockHAClientGenerator
from tests.typing import MqttMockPahoClient

def help_assert_message(
    msg: Any,
    topic: Optional[str] = None,
    payload: Optional[Any] = None,
    qos: Optional[int] = None,
    retain: Optional[bool] = None
) -> bool: ...

async def test_mqtt_connects_on_home_assistant_mqtt_setup(
    hass: HomeAssistant,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

async def test_mqtt_does_not_disconnect_on_home_assistant_stop(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

async def test_mqtt_await_ack_at_disconnect(
    hass: HomeAssistant
) -> None: ...

@pytest.mark.parametrize('mqtt_config_entry_options', ...)
async def test_publish(
    hass: HomeAssistant,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

async def test_convert_outgoing_payload(
    hass: HomeAssistant
) -> None: ...

async def test_all_subscriptions_run_when_decode_fails(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

@pytest.mark.usefixtures('mqtt_mock_entry')
async def test_subscribe_topic_not_initialize(
    hass: HomeAssistant,
    record_calls: Any
) -> None: ...

async def test_subscribe_mqtt_config_entry_disabled(
    hass: HomeAssistant,
    mqtt_mock: Any,
    record_calls: Any
) -> None: ...

async def test_subscribe_and_resubscribe(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_non_async(
    hass: HomeAssistant,
    mock_debouncer: Any,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_bad_topic(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_not_match(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_level_wildcard(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_level_wildcard_no_subtree_match(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_level_wildcard_root_topic_no_subtree_match(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_subtree_wildcard_subtree_topic(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_subtree_wildcard_root_topic(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_subtree_wildcard_no_match(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_level_wildcard_and_wildcard_root_topic(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_level_wildcard_and_wildcard_subtree_topic(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_level_wildcard_and_wildcard_level_no_match(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_level_wildcard_and_wildcard_no_match(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_sys_root(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_sys_root_and_wildcard_topic(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_topic_sys_root_and_wildcard_subtree_topic(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_special_characters(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    recorded_calls: List[Any],
    record_calls: Any
) -> None: ...

async def test_subscribe_same_topic(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

async def test_replaying_payload_same_topic(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

async def test_replaying_payload_after_resubscribing(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

async def test_replaying_payload_wildcard_topic(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

async def test_not_calling_unsubscribe_with_active_subscribers(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any,
    record_calls: Any
) -> None: ...

async def test_not_calling_subscribe_when_unsubscribed_within_cooldown(
    hass: HomeAssistant,
    mock_debouncer: Any,
    mqtt_mock_entry: Any,
    record_calls: Any
) -> None: ...

async def test_unsubscribe_race(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), ...)
async def test_restore_subscriptions_on_reconnect(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any,
    record_calls: Any
) -> None: ...

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), ...)
async def test_restore_all_active_subscriptions_on_reconnect(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any,
    record_calls: Any
) -> None: ...

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), ...)
async def test_subscribed_at_highest_qos(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any,
    record_calls: Any
) -> None: ...

async def test_initial_setup_logs_error(
    hass: HomeAssistant,
    caplog: Any,
    mqtt_client_mock: Any
) -> None: ...

async def test_logs_error_if_no_connect_broker(
    hass: HomeAssistant,
    caplog: Any,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

@pytest.mark.parametrize('reason_code', ...)
async def test_triggers_reauth_flow_if_auth_fails(
    hass: HomeAssistant,
    setup_with_birth_msg_client_mock: Any,
    reason_code: Any
) -> None: ...

@patch('homeassistant.components.mqtt.client.TIMEOUT_ACK', 0.3)
async def test_handle_mqtt_on_callback(
    hass: HomeAssistant,
    caplog: Any,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

async def test_handle_mqtt_on_callback_after_cancellation(
    hass: HomeAssistant,
    caplog: Any,
    mqtt_mock_entry: Any,
    mqtt_client_mock: Any
) -> None: ...

async def test_handle_mqtt_on_callback_after_timeout(
    hass: HomeAssistant,
    caplog: Any,
    mqtt_mock_entry: Any,
    mqtt_client_mock: Any
) -> None: ...

async def test_publish_error(
    hass: HomeAssistant,
    caplog: Any
) -> None: ...

async def test_subscribe_error(
    hass: HomeAssistant,
    setup_with_birth_msg_client_mock: Any,
    record_calls: Any,
    caplog: Any
) -> None: ...

async def test_handle_message_callback(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

@pytest.mark.parametrize(('mqtt_config_entry_data', 'protocol'), ...)
async def test_setup_mqtt_client_protocol(
    mqtt_mock_entry: Any,
    protocol: int
) -> None: ...

@patch('homeassistant.components.mqtt.client.TIMEOUT_ACK', 0.2)
async def test_handle_mqtt_timeout_on_callback(
    hass: HomeAssistant,
    caplog: Any,
    mock_debouncer: Any
) -> None: ...

@pytest.mark.parametrize('exception', ...)
async def test_setup_raises_config_entry_not_ready_if_no_connect_broker(
    hass: HomeAssistant,
    caplog: Any,
    exception: Any
) -> None: ...

@pytest.mark.parametrize(('mqtt_config_entry_data', 'insecure_param'), ...)
async def test_setup_uses_certificate_on_certificate_set_to_auto_and_insecure(
    hass: HomeAssistant,
    mqtt_mock_entry: Any,
    insecure_param: Any
) -> None: ...

@pytest.mark.parametrize('mqtt_config_entry_data', ...)
async def test_tls_version(
    hass: HomeAssistant,
    mqtt_client_mock: Any,
    mqtt_mock_entry: Any
) -> None: ...

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), ...)
@patch('homeassistant.components.mqtt.client.INITIAL_SUBSCRIBE_COOLDOWN', 0.0)
@patch('homeassistant.components.mqtt.client.DISCOVERY_COOLDOWN', 0.0)
@patch('homeassistant.components.mqtt.client.SUBSCRIBE_COOLDOWN', 0.0)
async def test_custom_birth_message(
    hass: HomeAssistant,
    mock_debouncer: Any,
    mqtt_config_entry_data: Any,
    mqtt_config_entry_options: Any,
    mqtt_client_mock: Any
) -> None: ...

@pytest.mark.parametrize('mqtt_config_entry_options', ...)
async def test_default_birth_message(
    hass: HomeAssistant,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), ...)
@patch('homeassistant.components.mqtt.client.INITIAL_SUBSCRIBE_COOLDOWN', 0.0)
@patch('homeassistant.components.mqtt.client.DISCOVERY_COOLDOWN', 0.0)
@patch('homeassistant.components.mqtt.client.SUBSCRIBE_COOLDOWN', 0.0)
async def test_no_birth_message(
    hass: HomeAssistant,
    record_calls: Any,
    mock_debouncer: Any,
    mqtt_config_entry_data: Any,
    mqtt_config_entry_options: Any,
    mqtt_client_mock: Any
) -> None: ...

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), ...)
@patch('homeassistant.components.mqtt.client.DISCOVERY_COOLDOWN', 0.2)
async def test_delayed_birth_message(
    hass: HomeAssistant,
    mqtt_config_entry_data: Any,
    mqtt_config_entry_options: Any,
    mqtt_client_mock: Any
) -> None: ...

@pytest.mark.parametrize('mqtt_config_entry_options', ...)
async def test_subscription_done_when_birth_message_is_sent(
    setup_with_birth_msg_client_mock: Any
) -> None: ...

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), ...)
async def test_custom_will_message(
    hass: HomeAssistant,
    mqtt_config_entry_data: Any,
    mqtt_config_entry_options: Any,
    mqtt_client_mock: Any
) -> None: ...

async def test_default_will_message(
    setup_with_birth_msg_client_mock: Any
) -> None: ...

@pytest.mark.parametrize(('mqtt_config_entry_data', 'mqtt_config_entry_options'), ...)
async def test_no_will_message(
    hass: HomeAssistant,
    mqtt_config_entry_data: Any,
    mqtt_config_entry_options: Any,
    mqtt_client_mock: Any
) -> None: ...

@pytest.mark.parametrize('mqtt_config_entry_options', ...)
async def test_mqtt_subscribes_topics_on_connect(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any,
    record_calls: Any
) -> None: ...

@pytest.mark.parametrize('mqtt_config_entry_options', ...)
async def test_mqtt_subscribes_wildcard_topics_in_correct_order(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any,
    record_calls: Any
) -> None: ...

@pytest.mark.parametrize('mqtt_config_entry_options', ...)
async def test_mqtt_discovery_not_subscribes_when_disabled(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any
) -> None: ...

@pytest.mark.parametrize('mqtt_config_entry_options', ...)
async def test_mqtt_subscribes_in_single_call(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any,
    record_calls: Any
) -> None: ...

@pytest.mark.parametrize('mqtt_config_entry_options', ...)
@patch('homeassistant.components.mqtt.client.MAX_SUBSCRIBES_PER_CALL', 2)
@patch('homeassistant.components.mqtt.client.MAX_UNSUBSCRIBES_PER_CALL', 2)
async def test_mqtt_subscribes_and_unsubscribes_in_chunks(
    hass: HomeAssistant,
    mock_debouncer: Any,
    setup_with_birth_msg_client_mock: Any,
    record_calls: Any
) -> None: ...

@pytest.mark.parametrize('exception', ...)
async def test_auto_reconnect(
    hass: HomeAssistant,
    setup_with_birth_msg_client_mock: Any,
    caplog: Any,
   