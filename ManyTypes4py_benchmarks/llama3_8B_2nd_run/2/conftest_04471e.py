from typing import AsyncMock, Mock, PropertyMock, patch

def mock_light_set_state(turn: str = 'on', mode: str = 'color', red: int = 45, green: int = 55, blue: int = 65, white: int = 70, gain: int = 19, temp: int = 4050, brightness: int = 50, effect: int = 0, transition: int = 0) -> dict:
    """Mock light block set_state."""
    return {'ison': turn == 'on', 'mode': mode, 'red': red, 'green': green, 'blue': blue, 'white': white, 'gain': gain, 'temp': temp, 'brightness': brightness, 'effect': effect, 'transition': transition}

def mock_white_light_set_state(turn: str = 'on', temp: int = 4050, gain: int = 19, brightness: int = 128, transition: int = 0) -> dict:
    """Mock white light block set_state."""
    return {'ison': turn == 'on', 'mode': 'white', 'gain': gain, 'temp': temp, 'brightness': brightness, 'transition': transition}

@pytest.fixture(autouse=True)
async def mock_block_device() -> Mock:
    """Mock block (Gen1, CoAP) device."""
    with patch('aioshelly.block_device.BlockDevice.create') as block_device_mock:
        # ... (rest of the code)

@pytest.fixture
async def mock_rpc_device() -> Mock:
    """Mock rpc (Gen2, Websocket) device with BLE support."""
    with patch('aioshelly.rpc_device.RpcDevice.create') as rpc_device_mock, patch('homeassistant.components.shelly.bluetooth.async_start_scanner'):
        # ... (rest of the code)

@pytest.fixture(autouse=True)
async def mock_bluetooth() -> None:
    """Auto mock bluetooth."""
    # ... (rest of the code)

@pytest.fixture
async def mock_blu_trv() -> Mock:
    """Mock BLU TRV."""
    with patch('aioshelly.rpc_device.RpcDevice.create') as blu_trv_device_mock, patch('homeassistant.components.shelly.bluetooth.async_start_scanner'):
        # ... (rest of the code)
