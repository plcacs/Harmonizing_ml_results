from typing import Any, Dict
from unittest.mock import patch

ADB_SERVER_HOST: str = '127.0.0.1'
KEY_PYTHON: str = 'python'
KEY_SERVER: str = 'server'
ADB_DEVICE_TCP_ASYNC_FAKE: str = 'AdbDeviceTcpAsyncFake'
DEVICE_ASYNC_FAKE: str = 'DeviceAsyncFake'
PROPS_DEV_INFO: str = 'fake\nfake\n0123456\nfake'
PROPS_DEV_MAC: str = 'ether ab:cd:ef:gh:ij:kl brd'

class AdbDeviceTcpAsyncFake:
    def __init__(self, *args, **kwargs) -> None:
        self.available: bool = False

    async def close(self) -> None:
        self.available = False

    async def connect(self, *args, **kwargs) -> None:
        raise NotImplementedError

    async def shell(self, cmd: str, *args, **kwargs) -> Any:
        return None

class ClientAsyncFakeSuccess:
    def __init__(self, host: str = ADB_SERVER_HOST, port: int = DEFAULT_ADB_SERVER_PORT) -> None:
        self._devices: List[DeviceAsyncFake] = []

    async def device(self, serial: str) -> DeviceAsyncFake:
        device = DeviceAsyncFake(serial)
        self._devices.append(device)
        return device

class ClientAsyncFakeFail:
    def __init__(self, host: str = ADB_SERVER_HOST, port: int = DEFAULT_ADB_SERVER_PORT) -> None:
        self._devices: List[DeviceAsyncFake] = []

    async def device(self, serial: str) -> None:
        self._devices = []
        return None

class DeviceAsyncFake:
    def __init__(self, host: str) -> None:
        self.host: str = host

    async def shell(self, cmd: str) -> None:
        raise NotImplementedError

def patch_connect(success: bool) -> Dict[str, patch]:
    async def connect_success_python(self, *args, **kwargs) -> None:
        self.available = True

    async def connect_fail_python(self, *args, **kwargs) -> None:
        raise OSError

    if success:
        return {KEY_PYTHON: patch(f'{__name__}.{ADB_DEVICE_TCP_ASYNC_FAKE}.connect', connect_success_python), KEY_SERVER: patch('androidtv.adb_manager.adb_manager_async.ClientAsync', ClientAsyncFakeSuccess)}
    return {KEY_PYTHON: patch(f'{__name__}.{ADB_DEVICE_TCP_ASYNC_FAKE}.connect', connect_fail_python), KEY_SERVER: patch('androidtv.adb_manager.adb_manager_async.ClientAsync', ClientAsyncFakeFail)}

def patch_shell(response: Any = None, error: bool = False, mac_eth: bool = False, exc: Exception = None) -> Dict[str, patch]:
    async def shell_success(self, cmd: str, *args, **kwargs) -> Any:
        self.shell_cmd = cmd
        if cmd == CMD_DEVICE_PROPERTIES:
            return PROPS_DEV_INFO
        if cmd == CMD_MAC_WLAN0:
            return PROPS_DEV_MAC
        if cmd == CMD_MAC_ETH0:
            return PROPS_DEV_MAC if mac_eth else None
        return response

    async def shell_fail_python(self, cmd: str, *args, **kwargs) -> Any:
        self.shell_cmd = cmd
        raise exc or ValueError

    async def shell_fail_server(self, cmd: str) -> Any:
        self.shell_cmd = cmd
        raise ConnectionResetError

    if not error:
        return {KEY_PYTHON: patch(f'{__name__}.{ADB_DEVICE_TCP_ASYNC_FAKE}.shell', shell_success), KEY_SERVER: patch(f'{__name__}.{DEVICE_ASYNC_FAKE}.shell', shell_success)}
    return {KEY_PYTHON: patch(f'{__name__}.{ADB_DEVICE_TCP_ASYNC_FAKE}.shell', shell_fail_python), KEY_SERVER: patch(f'{__name__}.{DEVICE_ASYNC_FAKE}.shell', shell_fail_server)}

def patch_androidtv_update(state: str, current_app: str, running_apps: List[str], device: str, is_volume_muted: bool, volume_level: float, hdmi_input: str) -> Dict[str, patch]:
    return {DEVICE_ANDROIDTV: patch('androidtv.androidtv.androidtv_async.AndroidTVAsync.update', return_value=(state, current_app, running_apps, device, is_volume_muted, volume_level, hdmi_input)), DEVICE_FIRETV: patch('androidtv.firetv.firetv_async.FireTVAsync.update', return_value=(state, current_app, running_apps, hdmi_input))}

def isfile(filepath: str) -> bool:
    return filepath.endswith('adbkey')

PATCH_SCREENCAP: patch = patch('androidtv.basetv.basetv_async.BaseTVAsync.adb_screencap', return_value=b'image')
PATCH_SETUP_ENTRY: patch = patch('homeassistant.components.androidtv.async_setup_entry', return_value=True)
PATCH_ACCESS: patch = patch('homeassistant.components.androidtv.os.access', return_value=True)
PATCH_ISFILE: patch = patch('homeassistant.components.androidtv.os.path.isfile', isfile)
PATCH_LAUNCH_APP: patch = patch('androidtv.basetv.basetv_async.BaseTVAsync.launch_app')
PATCH_STOP_APP: patch = patch('androidtv.basetv.basetv_async.BaseTVAsync.stop_app')
PATCH_ANDROIDTV_UPDATE_EXCEPTION: patch = patch('androidtv.androidtv.androidtv_async.AndroidTVAsync.update', side_effect=ZeroDivisionError)
