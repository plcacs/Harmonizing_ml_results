from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, PropertyMock, patch
from homeassistant.core import HomeAssistant

MOCK_MAC: str

MOCK_SETTINGS: Dict[str, Any]
MOCK_BLOCKS: List[Mock]
MOCK_CONFIG: Dict[str, Any]
MOCK_BLU_TRV_REMOTE_CONFIG: Dict[str, Any]
MOCK_BLU_TRV_REMOTE_STATUS: Dict[str, Any]
MOCK_SHELLY_COAP: Dict[str, Any]
MOCK_SHELLY_RPC: Dict[str, Any]
MOCK_STATUS_COAP: Dict[str, Any]
MOCK_STATUS_RPC: Dict[str, Any]
MOCK_SCRIPTS: List[str]

def mock_light_set_state(turn: str = 'on', mode: str = 'color', red: int = 45, green: int = 55, blue: int = 65, white: int = 70, gain: int = 19, temp: int = 4050, brightness: int = 50, effect: int = 0, transition: int = 0) -> Dict[str, Any]:
    ...

def mock_white_light_set_state(turn: str = 'on', temp: int = 4050, gain: int = 19, brightness: int = 128, transition: int = 0) -> Dict[str, Any]:
    ...

@pytest.fixture(autouse=True)
def mock_coap() -> None:
    ...

@pytest.fixture(autouse=True)
def mock_ws_server() -> None:
    ...

@pytest.fixture
def events(hass: HomeAssistant) -> Any:
    ...

@pytest.fixture
async def mock_block_device() -> Mock:
    ...

def _mock_rpc_device(version: str = None) -> Mock:
    ...

def _mock_blu_rtv_device(version: str = None) -> Mock:
    ...

@pytest.fixture
async def mock_rpc_device() -> Mock:
    ...

@pytest.fixture(autouse=True)
def mock_bluetooth(enable_bluetooth: Any) -> None:
    ...

@pytest.fixture(autouse=True)
async def mock_blu_trv() -> Mock:
    ...
