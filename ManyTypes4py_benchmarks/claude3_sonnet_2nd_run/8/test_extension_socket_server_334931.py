from typing import Any, Dict, List, Optional, Set
from unittest import mock
import pytest
from ulauncher.modes.extensions.extension_socket_server import ExtensionSocketServer

class TestExtensionSocketServer:

    @pytest.fixture(autouse=True)
    def socket_service(self, mocker: Any) -> Any:
        return mocker.patch('ulauncher.modes.extensions.extension_socket_server.Gio.SocketService')

    @pytest.fixture(autouse=True)
    def inix_socket_address(self, mocker: Any) -> Any:
        return mocker.patch('ulauncher.modes.extensions.extension_socket_server.Gio.UnixSocketAddress')

    @pytest.fixture(autouse=True)
    def extension_socket_controller(self, mocker: Any) -> Any:
        return mocker.patch('ulauncher.modes.extensions.extension_socket_server.ExtensionSocketController')

    @pytest.fixture(autouse=True)
    def path_exists(self, mocker: Any) -> Any:
        exists = mocker.patch('ulauncher.modes.extensions.extension_socket_server.os.path.exists')
        exists.return_value = False
        return exists

    @pytest.fixture(autouse=True)
    def gobject(self, mocker: Any) -> Any:
        return mocker.patch('ulauncher.modes.extensions.extension_socket_server.GObject')

    @pytest.fixture(autouse=True)
    def unlink(self, mocker: Any) -> Any:
        return mocker.patch('ulauncher.modes.extensions.extension_socket_server.os.unlink')

    @pytest.fixture(autouse=True)
    def jsonframer(self, mocker: Any) -> Any:
        return mocker.patch('ulauncher.modes.extensions.extension_socket_server.JSONFramer')

    @pytest.fixture
    def server(self) -> ExtensionSocketServer:
        return ExtensionSocketServer()

    def test_start(self, server: ExtensionSocketServer) -> None:
        server.start()
        server.service.connect.assert_called_once()
        server.service.add_address.assert_called_once()

    def test_start__clean_socket(self, server: ExtensionSocketServer, path_exists: Any, unlink: Any) -> None:
        path_exists.return_value = True
        server.start()
        unlink.assert_called_once()

    def test_handle_incoming(self, server: ExtensionSocketServer, jsonframer: Any) -> None:
        conn = mock.Mock()
        source = mock.Mock()
        server.start()
        server.handle_incoming(server.service, conn, source)
        assert id(jsonframer.return_value) in server.pending
        jsonframer.return_value.set_connection.assert_called_with(conn)

    def test_handle_registration(self, server: ExtensionSocketServer, jsonframer: Any, gobject: Any, extension_socket_controller: Any) -> None:
        conn = mock.Mock()
        source = mock.Mock()
        server.start()
        server.handle_incoming(server.service, conn, source)
        extid = 'id'
        event: Dict[str, str] = {'type': 'extension:socket_connected', 'ext_id': extid}
        assert id(jsonframer.return_value) in server.pending
        server.handle_registration(jsonframer.return_value, event)
        assert id(jsonframer.return_value) not in server.pending
        assert gobject.signal_handler_disconnect.call_count == 2
        extension_socket_controller.assert_called_once()

    def test_stop(self, server: ExtensionSocketServer) -> None:
        server.start()
        assert server.service
        service = server.service
        server.stop()
        assert not server.service
        assert service.stop.call_count == 1
        assert service.close.call_count == 1
