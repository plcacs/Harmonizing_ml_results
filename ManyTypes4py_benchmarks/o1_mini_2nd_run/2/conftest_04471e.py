"""Test configuration for Shelly."""
from unittest.mock import AsyncMock, Mock, PropertyMock, patch
from aioshelly.ble.const import (
    BLE_CODE,
    BLE_SCAN_RESULT_EVENT,
    BLE_SCAN_RESULT_VERSION,
    BLE_SCRIPT_NAME,
    VAR_ACTIVE,
    VAR_EVENT_TYPE,
    VAR_VERSION,
)
from aioshelly.block_device import BlockDevice, BlockUpdateType
from aioshelly.const import MODEL_1, MODEL_25, MODEL_PLUS_2PM
from aioshelly.rpc_device import RpcDevice, RpcUpdateType
import pytest
from homeassistant.components.shelly.const import (
    EVENT_SHELLY_CLICK,
    REST_SENSORS_UPDATE_INTERVAL,
)
from homeassistant.core import HomeAssistant
from typing import Any, Callable, Coroutine, Dict, Generator, List, Optional, Union
from . import MOCK_MAC
from tests.common import async_capture_events

MOCK_SETTINGS: Dict[str, Any] = {
    'name': 'Test name',
    'mode': 'relay',
    'device': {
        'mac': MOCK_MAC,
        'hostname': 'test-host',
        'type': MODEL_25,
        'num_outputs': 2,
    },
    'coiot': {'update_period': 15},
    'fw': '20201124-092159/v1.9.0@57ac4ad8',
    'relays': [{'btn_type': 'momentary'}, {'btn_type': 'toggle'}],
    'rollers': [{'positioning': True}],
    'external_power': 0,
    'thermostats': [{'schedule_profile_names': ['Profile1', 'Profile2']}],
}

def mock_light_set_state(
    turn: str = 'on',
    mode: str = 'color',
    red: int = 45,
    green: int = 55,
    blue: int = 65,
    white: int = 70,
    gain: int = 19,
    temp: int = 4050,
    brightness: int = 50,
    effect: int = 0,
    transition: int = 0
) -> Dict[str, Union[str, int, bool]]:
    """Mock light block set_state."""
    return {
        'ison': turn == 'on',
        'mode': mode,
        'red': red,
        'green': green,
        'blue': blue,
        'white': white,
        'gain': gain,
        'temp': temp,
        'brightness': brightness,
        'effect': effect,
        'transition': transition,
    }

def mock_white_light_set_state(
    turn: str = 'on',
    temp: int = 4050,
    gain: int = 19,
    brightness: int = 128,
    transition: int = 0
) -> Dict[str, Union[str, int, bool]]:
    """Mock white light block set_state."""
    return {
        'ison': turn == 'on',
        'mode': 'white',
        'gain': gain,
        'temp': temp,
        'brightness': brightness,
        'transition': transition,
    }

MOCK_BLOCKS: List[Mock] = [
    Mock(
        sensor_ids={'inputEvent': 'S', 'inputEventCnt': 2, 'overpower': 0, 'power': 53.4, 'energy': 1234567.89},
        channel='0',
        type='relay',
        overpower=0,
        power=53.4,
        energy=1234567.89,
        description='relay_0',
        set_state=AsyncMock(side_effect=lambda turn: {'ison': turn == 'on'})
    ),
    Mock(
        sensor_ids={'roller': 'stop', 'rollerPos': 0},
        channel='1',
        type='roller',
        description='roller_0',
        set_state=AsyncMock(side_effect=lambda go, roller_pos=0: {'current_pos': roller_pos, 'state': go})
    ),
    Mock(
        sensor_ids={'mode': 'color', 'effect': 0},
        channel='0',
        output=mock_light_set_state()['ison'],
        colorTemp=mock_light_set_state()['temp'],
        **mock_light_set_state(),
        type='light',
        description='light_0',
        set_state=AsyncMock(side_effect=mock_light_set_state)
    ),
    Mock(
        sensor_ids={'motion': 0, 'temp': 22.1, 'gas': 'mild', 'motionActive': 1},
        channel='0',
        motion=0,
        temp=22.1,
        gas='mild',
        targetTemp=4,
        description='sensor_0',
        type='sensor'
    ),
    Mock(
        sensor_ids={'battery': 98, 'valvePos': 50},
        channel='0',
        battery=98,
        cfgChanged=0,
        mode=0,
        valvePos=50,
        inputEvent='S',
        wakeupEvent=['button'],
        description='device_0',
        type='device'
    ),
    Mock(
        sensor_ids={'powerFactor': 0.98},
        channel='0',
        powerFactor=0.98,
        targetTemp=4,
        temp=22.1,
        description='emeter_0',
        type='emeter'
    ),
    Mock(
        sensor_ids={'valve': 'closed'},
        valve='closed',
        channel='0',
        description='valve_0',
        type='valve',
        set_state=AsyncMock(side_effect=lambda go: {'state': 'opening' if go == 'open' else 'closing'})
    ),
]

MOCK_CONFIG: Dict[str, Any] = {
    'input:0': {'id': 0, 'name': 'Test name input 0', 'type': 'button'},
    'input:1': {
        'id': 1,
        'type': 'analog',
        'enable': True,
        'xpercent': {'expr': None, 'unit': None}
    },
    'input:2': {
        'id': 2,
        'name': 'Gas',
        'type': 'count',
        'enable': True,
        'xcounts': {'expr': None, 'unit': None},
        'xfreq': {'expr': None, 'unit': None}
    },
    'flood:0': {'id': 0, 'name': 'Test name'},
    'light:0': {'name': 'test light_0'},
    'light:1': {'name': 'test light_1'},
    'light:2': {'name': 'test light_2'},
    'light:3': {'name': 'test light_3'},
    'rgb:0': {'name': 'test rgb_0'},
    'rgbw:0': {'name': 'test rgbw_0'},
    'switch:0': {'name': 'test switch_0'},
    'cover:0': {'name': 'test cover_0'},
    'thermostat:0': {'id': 0, 'enable': True, 'type': 'heating'},
    'sys': {'ui_data': {}, 'device': {'name': 'Test name'}},
    'wifi': {'sta': {'enable': True}, 'sta1': {'enable': False}},
    'ws': {'enable': False, 'server': None},
    'voltmeter:100': {'xvoltage': {'unit': 'ppm'}},
    'script:1': {'id': 1, 'name': 'test_script.js', 'enable': True},
    'script:2': {'id': 2, 'name': 'test_script_2.js', 'enable': False},
    'script:3': {'id': 3, 'name': BLE_SCRIPT_NAME, 'enable': False},
}

MOCK_BLU_TRV_REMOTE_CONFIG: Dict[str, Any] = {
    'components': [{
        'key': 'blutrv:200',
        'status': {
            'id': 200,
            'target_C': 17.1,
            'current_C': 17.1,
            'pos': 0,
            'rssi': -60,
            'battery': 100,
            'packet_id': 58,
            'last_updated_ts': 1734967725,
            'paired': True,
            'rpc': True,
            'rsv': 61,
        },
        'config': {
            'id': 200,
            'addr': 'f8:44:77:25:f0:dd',
            'name': 'TRV-Name',
            'key': None,
            'trv': 'bthomedevice:200',
            'temp_sensors': [],
            'dw_sensors': [],
            'override_delay': 30,
            'meta': {},
        },
    }],
    'blutrv:200': {
        'id': 0,
        'enable': True,
        'min_valve_position': 0,
        'default_boost_duration': 1800,
        'default_override_duration': 2147483647,
        'default_override_target_C': 8,
        'addr': 'f8:44:77:25:f0:dd',
        'name': 'TRV-Name',
        'local_name': 'SBTR-001AEU',
    },
}

MOCK_BLU_TRV_REMOTE_STATUS: Dict[str, Any] = {
    'blutrv:200': {
        'id': 0,
        'pos': 0,
        'steps': 0,
        'current_C': 15.2,
        'target_C': 17.1,
        'schedule_rev': 0,
        'rssi': -60,
        'battery': 100,
        'errors': [],
    }
}

MOCK_SHELLY_COAP: Dict[str, Any] = {
    'mac': MOCK_MAC,
    'auth': False,
    'fw': '20210715-092854/v1.11.0@57ac4ad8',
    'num_outputs': 2,
}

MOCK_SHELLY_RPC: Dict[str, Any] = {
    'name': 'Test Gen2',
    'id': 'shellyplus2pm-123456789abc',
    'mac': MOCK_MAC,
    'model': MODEL_PLUS_2PM,
    'gen': 2,
    'fw_id': '20230803-130540/1.0.0-gfa1bc37',
    'ver': '1.0.0',
    'app': 'Plus2PM',
    'auth_en': False,
    'auth_domain': None,
    'profile': 'cover',
}

MOCK_STATUS_COAP: Dict[str, Any] = {
    'update': {
        'status': 'pending',
        'has_update': True,
        'beta_version': '20231107-162609/v1.14.1-rc1-g0617c15',
        'new_version': '20230913-111730/v1.14.0-gcb84623',
        'old_version': '20230913-111730/v1.14.0-gcb84623',
    },
    'uptime': 5 * REST_SENSORS_UPDATE_INTERVAL,
    'wifi_sta': {'rssi': -64},
}

MOCK_STATUS_RPC: Dict[str, Any] = {
    'switch:0': {'output': True},
    'input:0': {'id': 0, 'state': None},
    'input:1': {'id': 1, 'percent': 89, 'xpercent': 8.9},
    'input:2': {
        'id': 2,
        'counts': {'total': 56174, 'xtotal': 561.74},
        'freq': 208.0,
        'xfreq': 6.11,
    },
    'light:0': {'output': True, 'brightness': 53.0},
    'light:1': {'output': True, 'brightness': 53.0},
    'light:2': {'output': True, 'brightness': 53.0},
    'light:3': {'output': True, 'brightness': 53.0},
    'rgb:0': {'output': True, 'brightness': 53.0, 'rgb': [45, 55, 65]},
    'rgbw:0': {'output': True, 'brightness': 53.0, 'rgb': [21, 22, 23], 'white': 120},
    'cloud': {'connected': False},
    'cover:0': {'state': 'stopped', 'pos_control': True, 'current_pos': 50, 'apower': 85.3},
    'devicepower:0': {'external': {'present': True}},
    'temperature:0': {'tC': 22.9},
    'illuminance:0': {'lux': 345},
    'em1:0': {'act_power': 85.3},
    'em1:1': {'act_power': 123.3},
    'em1data:0': {'total_act_energy': 123456.4},
    'em1data:1': {'total_act_energy': 987654.3},
    'flood:0': {'id': 0, 'alarm': False, 'mute': False},
    'thermostat:0': {'id': 0, 'enable': True, 'target_C': 23, 'current_C': 12.3, 'output': True},
    'script:1': {'id': 1, 'running': True, 'mem_used': 826, 'mem_peak': 1666, 'mem_free': 24360},
    'script:2': {'id': 2, 'running': False},
    'script:3': {'id': 3, 'running': False},
    'humidity:0': {'rh': 44.4},
    'sys': {
        'available_updates': {
            'beta': {'version': 'some_beta_version'},
            'stable': {'version': 'some_beta_version'},
        },
        'relay_in_thermostat': True,
    },
    'voltmeter:100': {'voltage': 4.321, 'xvoltage': 12.34},
    'wifi': {'rssi': -63},
}

MOCK_SCRIPTS: List[str] = [
    '\nfunction eventHandler(event, userdata) {\n  if (typeof event.component !== "string")\n    return;\n\n  let component = event.component.substring(0, 5);\n  if (component === "input") {\n    let id = Number(event.component.substring(6));\n    Shelly.emitEvent("input_event", { id: id });\n  }\n}\n\nShelly.addEventHandler(eventHandler);\nShelly.emitEvent("script_start");\n',
    'console.log("Hello World!")',
    BLE_CODE.replace(VAR_ACTIVE, 'true').replace(VAR_EVENT_TYPE, BLE_SCAN_RESULT_EVENT).replace(VAR_VERSION, str(BLE_SCAN_RESULT_VERSION)),
]

@pytest.fixture(autouse=True)
def mock_coap() -> Generator[Any, Any, None]:
    """Mock out coap."""
    with patch(
        'homeassistant.components.shelly.utils.COAP',
        return_value=Mock(initialize=AsyncMock(), close=Mock())
    ):
        yield

@pytest.fixture(autouse=True)
def mock_ws_server() -> Generator[Any, Any, None]:
    """Mock out ws_server."""
    with patch('homeassistant.components.shelly.utils.get_ws_context'):
        yield

@pytest.fixture
def events(hass: HomeAssistant) -> Callable[[], List[Any]]:
    """Yield caught shelly_click events."""
    return async_capture_events(hass, EVENT_SHELLY_CLICK)

@pytest.fixture
async def mock_block_device() -> Mock:
    """Mock block (Gen1, CoAP) device."""
    with patch('aioshelly.block_device.BlockDevice.create') as block_device_mock:

        def update() -> None:
            block_device_mock.return_value.subscribe_updates.call_args[0][0]({}, BlockUpdateType.COAP_PERIODIC)

        def update_reply() -> None:
            block_device_mock.return_value.subscribe_updates.call_args[0][0]({}, BlockUpdateType.COAP_REPLY)

        def online() -> None:
            block_device_mock.return_value.subscribe_updates.call_args[0][0]({}, BlockUpdateType.ONLINE)

        device: Mock = Mock(
            spec=BlockDevice,
            blocks=MOCK_BLOCKS,
            settings=MOCK_SETTINGS,
            shelly=MOCK_SHELLY_COAP,
            version='1.11.0',
            status=MOCK_STATUS_COAP,
            firmware_version='some fw string',
            initialized=True,
            model=MODEL_1,
            gen=1,
        )
        type(device).name = PropertyMock(return_value='Test name')
        block_device_mock.return_value = device
        block_device_mock.return_value.mock_update = Mock(side_effect=update)
        block_device_mock.return_value.mock_update_reply = Mock(side_effect=update_reply)
        block_device_mock.return_value.mock_online = Mock(side_effect=online)
        yield block_device_mock.return_value

def _mock_rpc_device(version: Optional[str] = None) -> Mock:
    """Mock rpc (Gen2, Websocket) device."""
    device: Mock = Mock(
        spec=RpcDevice,
        config=MOCK_CONFIG,
        event={},
        shelly=MOCK_SHELLY_RPC,
        version=version or '1.0.0',
        hostname='test-host',
        status=MOCK_STATUS_RPC,
        firmware_version='some fw string',
        initialized=True,
        connected=True,
        script_getcode=AsyncMock(side_effect=lambda script_id: {'data': MOCK_SCRIPTS[script_id - 1]}),
    )
    type(device).name = PropertyMock(return_value='Test name')
    return device

def _mock_blu_rtv_device(version: Optional[str] = None) -> Mock:
    """Mock BLU TRV device."""
    device: Mock = Mock(
        spec=RpcDevice,
        config={**MOCK_CONFIG, **MOCK_BLU_TRV_REMOTE_CONFIG},
        event={},
        shelly=MOCK_SHELLY_RPC,
        version=version or '1.0.0',
        hostname='test-host',
        status={**MOCK_STATUS_RPC, **MOCK_BLU_TRV_REMOTE_STATUS},
        firmware_version='some fw string',
        initialized=True,
        connected=True,
    )
    type(device).name = PropertyMock(return_value='Test name')
    return device

@pytest.fixture
async def mock_rpc_device() -> Mock:
    """Mock rpc (Gen2, Websocket) device with BLE support."""
    with patch('aioshelly.rpc_device.RpcDevice.create') as rpc_device_mock, \
         patch('homeassistant.components.shelly.bluetooth.async_start_scanner'):

        def update() -> None:
            rpc_device_mock.return_value.subscribe_updates.call_args[0][0]({}, RpcUpdateType.STATUS)

        def event() -> None:
            rpc_device_mock.return_value.subscribe_updates.call_args[0][0]({}, RpcUpdateType.EVENT)

        def online() -> None:
            rpc_device_mock.return_value.subscribe_updates.call_args[0][0]({}, RpcUpdateType.ONLINE)

        def disconnected() -> None:
            rpc_device_mock.return_value.subscribe_updates.call_args[0][0]({}, RpcUpdateType.DISCONNECTED)

        def initialized() -> None:
            rpc_device_mock.return_value.subscribe_updates.call_args[0][0]({}, RpcUpdateType.INITIALIZED)

        device: Mock = _mock_rpc_device()
        rpc_device_mock.return_value = device
        rpc_device_mock.return_value.mock_disconnected = Mock(side_effect=disconnected)
        rpc_device_mock.return_value.mock_update = Mock(side_effect=update)
        rpc_device_mock.return_value.mock_event = Mock(side_effect=event)
        rpc_device_mock.return_value.mock_online = Mock(side_effect=online)
        rpc_device_mock.return_value.mock_initialized = Mock(side_effect=initialized)
        yield rpc_device_mock.return_value

@pytest.fixture(autouse=True)
def mock_bluetooth(enable_bluetooth: Callable[[], None]) -> None:
    """Auto mock bluetooth."""
    enable_bluetooth()

@pytest.fixture(autouse=True)
async def mock_blu_trv() -> Mock:
    """Mock BLU TRV."""
    with patch('aioshelly.rpc_device.RpcDevice.create') as blu_trv_device_mock, \
         patch('homeassistant.components.shelly.bluetooth.async_start_scanner'):

        def update() -> None:
            blu_trv_device_mock.return_value.subscribe_updates.call_args[0][0]({}, RpcUpdateType.STATUS)

        device: Mock = _mock_blu_rtv_device()
        blu_trv_device_mock.return_value = device
        blu_trv_device_mock.return_value.mock_update = Mock(side_effect=update)
        yield blu_trv_device_mock.return_value
