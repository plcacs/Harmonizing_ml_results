"""Define patches used for androidtv tests."""
from typing import Any, Optional, Dict, Coroutine
from unittest.mock import patch
from androidtv.adb_manager.adb_manager_async import DeviceAsync
from androidtv.constants import CMD_DEVICE_PROPERTIES, CMD_MAC_ETH0, CMD_MAC_WLAN0
from homeassistant.components.androidtv.const import DEFAULT_ADB_SERVER_PORT, DEVICE_ANDROIDTV, DEVICE_FIRETV

ADB_SERVER_HOST: str = '127.0.0.1'
KEY_PYTHON: str = 'python'
KEY_SERVER: str = 'server'
ADB_DEVICE_TCP_ASYNC_FAKE: str = 'AdbDeviceTcpAsyncFake'
DEVICE_ASYNC_FAKE: str = 'DeviceAsyncFake'
PROPS_DEV_INFO: str = 'fake\nfake\n0123456\nfake'
PROPS_DEV_MAC: str = 'ether ab:cd:ef:gh:ij:kl brd'


class AdbDeviceTcpAsyncFake:
    """A fake of the `adb_shell.adb_device_async.AdbDeviceTcpAsync` class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a fake `adb_shell.adb_device_async.AdbDeviceTcpAsync` instance."""
        self.available: bool = False

    async def close(self) -> None:
        """Close the socket connection."""
        self.available = False

    async def connect(self, *args: Any, **kwargs: Any) -> None:
        """Try to connect to a device."""
        raise NotImplementedError

    async def shell(self, cmd: str, *args: Any, **kwargs: Any) -> Any:
        """Send an ADB shell command."""
        return None


class ClientAsyncFakeSuccess:
    """A fake of the `ClientAsync` class when the connection and shell commands succeed."""

    def __init__(self, host: str = ADB_SERVER_HOST, port: int = DEFAULT_ADB_SERVER_PORT) -> None:
        """Initialize a `ClientAsyncFakeSuccess` instance."""
        self._devices: list[Any] = []

    async def device(self, serial: str) -> 'DeviceAsyncFake':
        """Mock the `ClientAsync.device` method when the device is connected via ADB."""
        device = DeviceAsyncFake(serial)
        self._devices.append(device)
        return device


class ClientAsyncFakeFail:
    """A fake of the `ClientAsync` class when the connection and shell commands fail."""

    def __init__(self, host: str = ADB_SERVER_HOST, port: int = DEFAULT_ADB_SERVER_PORT) -> None:
        """Initialize a `ClientAsyncFakeFail` instance."""
        self._devices: list[Any] = []

    async def device(self, serial: str) -> Optional[Any]:
        """Mock the `ClientAsync.device` method when the device is not connected via ADB."""
        self._devices = []
        return None


class DeviceAsyncFake:
    """A fake of the `DeviceAsync` class."""

    def __init__(self, host: str) -> None:
        """Initialize a `DeviceAsyncFake` instance."""
        self.host: str = host

    async def shell(self, cmd: str) -> Any:
        """Send an ADB shell command."""
        raise NotImplementedError


def patch_connect(success: bool) -> Dict[str, Any]:
    """Mock the `adb_shell.adb_device_async.AdbDeviceTcpAsync` and `ClientAsync` classes."""

    async def connect_success_python(self: Any, *args: Any, **kwargs: Any) -> None:
        """Mock the `AdbDeviceTcpAsyncFake.connect` method when it succeeds."""
        self.available = True

    async def connect_fail_python(self: Any, *args: Any, **kwargs: Any) -> None:
        """Mock the `AdbDeviceTcpAsyncFake.connect` method when it fails."""
        raise OSError

    if success:
        return {
            KEY_PYTHON: patch(f'{__name__}.{ADB_DEVICE_TCP_ASYNC_FAKE}.connect', connect_success_python),
            KEY_SERVER: patch('androidtv.adb_manager.adb_manager_async.ClientAsync', ClientAsyncFakeSuccess),
        }
    return {
        KEY_PYTHON: patch(f'{__name__}.{ADB_DEVICE_TCP_ASYNC_FAKE}.connect', connect_fail_python),
        KEY_SERVER: patch('androidtv.adb_manager.adb_manager_async.ClientAsync', ClientAsyncFakeFail),
    }


def patch_shell(response: Optional[Any] = None, error: bool = False, mac_eth: bool = False, exc: Optional[Exception] = None) -> Dict[str, Any]:
    """Mock the `AdbDeviceTcpAsyncFake.shell` and `DeviceAsyncFake.shell` methods."""

    async def shell_success(self: Any, cmd: str, *args: Any, **kwargs: Any) -> Any:
        """Mock the `AdbDeviceTcpAsyncFake.shell` and `DeviceAsyncFake.shell` methods when they are successful."""
        self.shell_cmd = cmd
        if cmd == CMD_DEVICE_PROPERTIES:
            return PROPS_DEV_INFO
        if cmd == CMD_MAC_WLAN0:
            return PROPS_DEV_MAC
        if cmd == CMD_MAC_ETH0:
            return PROPS_DEV_MAC if mac_eth else None
        return response

    async def shell_fail_python(self: Any, cmd: str, *args: Any, **kwargs: Any) -> Any:
        """Mock the `AdbDeviceTcpAsyncFake.shell` method when it fails."""
        self.shell_cmd = cmd
        raise exc or ValueError

    async def shell_fail_server(self: Any, cmd: str) -> Any:
        """Mock the `DeviceAsyncFake.shell` method when it fails."""
        self.shell_cmd = cmd
        raise ConnectionResetError

    if not error:
        return {
            KEY_PYTHON: patch(f'{__name__}.{ADB_DEVICE_TCP_ASYNC_FAKE}.shell', shell_success),
            KEY_SERVER: patch(f'{__name__}.{DEVICE_ASYNC_FAKE}.shell', shell_success),
        }
    return {
        KEY_PYTHON: patch(f'{__name__}.{ADB_DEVICE_TCP_ASYNC_FAKE}.shell', shell_fail_python),
        KEY_SERVER: patch(f'{__name__}.{DEVICE_ASYNC_FAKE}.shell', shell_fail_server),
    }


def patch_androidtv_update(state: Any, current_app: Any, running_apps: Any, device: Any, is_volume_muted: Any, volume_level: Any, hdmi_input: Any) -> Dict[str, Any]:
    """Patch the `AndroidTV.update()` method."""
    return {
        DEVICE_ANDROIDTV: patch('androidtv.androidtv.androidtv_async.AndroidTVAsync.update', return_value=(state, current_app, running_apps, device, is_volume_muted, volume_level, hdmi_input)),
        DEVICE_FIRETV: patch('androidtv.firetv.firetv_async.FireTVAsync.update', return_value=(state, current_app, running_apps, hdmi_input)),
    }


def isfile(filepath: str) -> bool:
    """Mock `os.path.isfile`."""
    return filepath.endswith('adbkey')


PATCH_SCREENCAP: Any = patch('androidtv.basetv.basetv_async.BaseTVAsync.adb_screencap', return_value=b'image')
PATCH_SETUP_ENTRY: Any = patch('homeassistant.components.androidtv.async_setup_entry', return_value=True)
PATCH_ACCESS: Any = patch('homeassistant.components.androidtv.os.access', return_value=True)
PATCH_ISFILE: Any = patch('homeassistant.components.androidtv.os.path.isfile', isfile)
PATCH_LAUNCH_APP: Any = patch('androidtv.basetv.basetv_async.BaseTVAsync.launch_app')
PATCH_STOP_APP: Any = patch('androidtv.basetv.basetv_async.BaseTVAsync.stop_app')
PATCH_ANDROIDTV_UPDATE_EXCEPTION: Any = patch('androidtv.androidtv.androidtv_async.AndroidTVAsync.update', side_effect=ZeroDivisionError)