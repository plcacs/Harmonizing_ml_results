from contextlib import closing
import getpass
import socket
import unittest
from tornado.concurrent import Future
from tornado.netutil import bind_sockets, Resolver
from tornado.queues import Queue
from tornado.tcpclient import TCPClient, _Connector
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, gen_test
from tornado.test.util import skipIfNoIPv6, refusing_port, skipIfNonUnix
from tornado.gen import TimeoutError
import typing
if typing.TYPE_CHECKING:
    from tornado.iostream import IOStream
    from typing import List, Dict, Tuple

AF1, AF2 = (1, 2)

class TestTCPServer(TCPServer):
    """A test TCP server."""

    def __init__(self, family: int) -> None:
        """Initializes the test TCP server.

        Args:
            family: The address family to use.
        """
        super().__init__()
        self.streams: List[IOStream] = []
        self.queue: Queue = Queue()
        sockets = bind_sockets(0, 'localhost', family)
        self.add_sockets(sockets)
        self.port: int = sockets[0].getsockname()[1]

    def handle_stream(self, stream: IOStream, address: Tuple[int, str]) -> None:
        """Handles an incoming stream.

        Args:
            stream: The incoming stream.
            address: The address of the incoming stream.
        """
        self.streams.append(stream)
        self.queue.put(stream)

    def stop(self) -> None:
        """Stops the test TCP server."""
        super().stop()
        for stream in self.streams:
            stream.close()

class TCPClientTest(AsyncTestCase):
    """Tests for the TCP client."""

    def setUp(self) -> None:
        """Sets up the test."""
        super().setUp()
        self.server: TestTCPServer = None
        self.client: TCPClient = TCPClient()

    def start_server(self, family: int) -> int:
        """Starts the test server.

        Args:
            family: The address family to use.

        Returns:
            The port number of the test server.
        """
        self.server = TestTCPServer(family)
        return self.server.port

    def stop_server(self) -> None:
        """Stops the test server."""
        if self.server is not None:
            self.server.stop()
            self.server = None

    def tearDown(self) -> None:
        """Tears down the test."""
        self.client.close()
        self.stop_server()
        super().tearDown()

    def skipIfLocalhostV4(self) -> None:
        """Skips the test if localhost does not resolve to IPv6."""
        addrinfo = self.io_loop.run_sync(lambda: Resolver().resolve('localhost', 80))
        families = {addr[0] for addr in addrinfo}
        if socket.AF_INET6 not in families:
            self.skipTest('localhost does not resolve to ipv6')

    @gen_test
    def do_test_connect(self, family: int, host: str, source_ip: str = None, source_port: int = None) -> None:
        """Tests connecting to a host.

        Args:
            family: The address family to use.
            host: The host to connect to.
            source_ip: The source IP address to use.
            source_port: The source port to use.
        """
        port = self.start_server(family)
        stream = (yield self.client.connect(host, port, source_ip=source_ip, source_port=source_port, af=family))
        assert self.server is not None
        server_stream = (yield self.server.queue.get())
        with closing(stream):
            stream.write(b'hello')
            data = (yield server_stream.read_bytes(5))
            self.assertEqual(data, b'hello')

    def test_connect_ipv4_ipv4(self) -> None:
        """Tests connecting to an IPv4 host."""
        self.do_test_connect(socket.AF_INET, '127.0.0.1')

    def test_connect_ipv4_dual(self) -> None:
        """Tests connecting to a dual-stacked IPv4 host."""
        self.do_test_connect(socket.AF_INET, 'localhost')

    @skipIfNoIPv6
    def test_connect_ipv6_ipv6(self) -> None:
        """Tests connecting to an IPv6 host."""
        self.skipIfLocalhostV4()
        self.do_test_connect(socket.AF_INET6, '::1')

    @skipIfNoIPv6
    def test_connect_ipv6_dual(self) -> None:
        """Tests connecting to a dual-stacked IPv6 host."""
        self.skipIfLocalhostV4()
        self.do_test_connect(socket.AF_INET6, 'localhost')

    def test_connect_unspec_ipv4(self) -> None:
        """Tests connecting to an unspecified IPv4 host."""
        self.do_test_connect(socket.AF_UNSPEC, '127.0.0.1')

    @skipIfNoIPv6
    def test_connect_unspec_ipv6(self) -> None:
        """Tests connecting to an unspecified IPv6 host."""
        self.skipIfLocalhostV4()
        self.do_test_connect(socket.AF_UNSPEC, '::1')

    def test_connect_unspec_dual(self) -> None:
        """Tests connecting to an unspecified dual-stacked host."""
        self.do_test_connect(socket.AF_UNSPEC, 'localhost')

    @gen_test
    def test_refused_ipv4(self) -> None:
        """Tests connecting to a refused IPv4 host."""
        cleanup_func, port = refusing_port()
        self.addCleanup(cleanup_func)
        with self.assertRaises(IOError):
            yield self.client.connect('127.0.0.1', port)

    def test_source_ip_fail(self) -> None:
        """Tests failing to use a source IP address."""
        self.assertRaises(socket.error, self.do_test_connect, socket.AF_INET, '127.0.0.1', source_ip='8.8.8.8')

    def test_source_ip_success(self) -> None:
        """Tests successfully using a source IP address."""
        self.do_test_connect(socket.AF_INET, '127.0.0.1', source_ip='127.0.0.1')

    @skipIfNonUnix
    def test_source_port_fail(self) -> None:
        """Tests failing to use a source port."""
        if getpass.getuser() == 'root':
            self.skipTest('running as root')
        self.assertRaises(socket.error, self.do_test_connect, socket.AF_INET, '127.0.0.1', source_port=1)

    @gen_test
    def test_connect_timeout(self) -> None:
        """Tests connecting to a host with a timeout."""
        timeout = 0.05

        class TimeoutResolver(Resolver):
            """A resolver that times out."""

            def resolve(self, *args, **kwargs) -> Future:
                """Resolves the host with a timeout."""
                return Future()

        with self.assertRaises(TimeoutError):
            yield TCPClient(resolver=TimeoutResolver()).connect('1.2.3.4', 12345, timeout=timeout)

class TestConnectorSplit(unittest.TestCase):
    """Tests for the connector splitter."""

    def test_one_family(self) -> None:
        """Tests splitting a list with one family."""
        primary, secondary = _Connector.split([(AF1, 'a'), (AF1, 'b')])
        self.assertEqual(primary, [(AF1, 'a'), (AF1, 'b')])
        self.assertEqual(secondary, [])

    def test_mixed(self) -> None:
        """Tests splitting a list with mixed families."""
        primary, secondary = _Connector.split([(AF1, 'a'), (AF2, 'b'), (AF1, 'c'), (AF2, 'd')])
        self.assertEqual(primary, [(AF1, 'a'), (AF1, 'c')])
        self.assertEqual(secondary, [(AF2, 'b'), (AF2, 'd')])

class ConnectorTest(AsyncTestCase):
    """Tests for the connector."""

    class FakeStream:
        """A fake stream."""

        def __init__(self) -> None:
            self.closed: bool = False

        def close(self) -> None:
            """Closes the stream."""
            self.closed = True

    def setUp(self) -> None:
        """Sets up the test."""
        super().setUp()
        self.connect_futures: Dict[Tuple[int, str], Future] = {}
        self.streams: Dict[str, FakeStream] = {}
        self.addrinfo: List[Tuple[int, str]] = [(AF1, 'a'), (AF1, 'b'), (AF2, 'c'), (AF2, 'd')]

    def tearDown(self) -> None:
        """Tears down the test."""
        for stream in self.streams.values():
            self.assertFalse(stream.closed)
        super().tearDown()

    def create_stream(self, af: int, addr: str) -> Tuple[FakeStream, Future]:
        """Creates a fake stream.

        Args:
            af: The address family to use.
            addr: The address to use.

        Returns:
            A tuple of the fake stream and its future.
        """
        stream = ConnectorTest.FakeStream()
        self.streams[addr] = stream
        future = Future()
        self.connect_futures[af, addr] = future
        return (stream, future)

    def assert_pending(self, *keys: Tuple[int, str]) -> None:
        """Asserts that the futures are pending.

        Args:
            *keys: The keys to check.
        """
        self.assertEqual(sorted(self.connect_futures.keys()), sorted(keys))

    def resolve_connect(self, af: int, addr: str, success: bool) -> None:
        """Resolves a connect future.

        Args:
            af: The address family to use.
            addr: The address to use.
            success: Whether the connect was successful.
        """
        future = self.connect_futures.pop((af, addr))
        if success:
            future.set_result(self.streams[addr])
        else:
            self.streams.pop(addr)
            future.set_exception(IOError())
        self.io_loop.add_callback(self.stop)
        self.wait()

    def assert_connector_streams_closed(self, conn: _Connector) -> None:
        """Asserts that the connector streams are closed.

        Args:
            conn: The connector to check.
        """
        for stream in conn.streams:
            self.assertTrue(stream.closed)

    def start_connect(self, addrinfo: List[Tuple[int, str]]) -> Tuple[_Connector, Future]:
        """Starts a connect.

        Args:
            addrinfo: The address information to use.

        Returns:
            A tuple of the connector and its future.
        """
        conn = _Connector(addrinfo, self.create_stream)
        future = conn.start(3600, connect_timeout=self.io_loop.time() + 3600)
        return (conn, future)

    def test_immediate_success(self) -> None:
        """Tests a successful connect with immediate success."""
        conn, future = self.start_connect(self.addrinfo)
        self.assertEqual(list(self.connect_futures.keys()), [(AF1, 'a')])
        self.resolve_connect(AF1, 'a', True)
        self.assertEqual(future.result(), (AF1, 'a', self.streams['a']))

    def test_immediate_failure(self) -> None:
        """Tests a failed connect with immediate failure."""
        conn, future = self.start_connect([(AF1, 'a')])
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assertRaises(IOError, future.result)

    def test_one_family_second_try(self) -> None:
        """Tests a successful connect with a second try."""
        conn, future = self.start_connect([(AF1, 'a'), (AF1, 'b')])
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending((AF1, 'b'))
        self.resolve_connect(AF1, 'b', True)
        self.assertEqual(future.result(), (AF1, 'b', self.streams['b']))

    def test_one_family_second_try_failure(self) -> None:
        """Tests a failed connect with a second try."""
        conn, future = self.start_connect([(AF1, 'a'), (AF1, 'b')])
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending((AF1, 'b'))
        self.resolve_connect(AF1, 'b', False)
        self.assertRaises(IOError, future.result)

    def test_one_family_second_try_timeout(self) -> None:
        """Tests a successful connect with a second try and a timeout."""
        conn, future = self.start_connect([(AF1, 'a'), (AF1, 'b')])
        self.assert_pending((AF1, 'a'))
        conn.on_timeout()
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending((AF1, 'b'))
        self.resolve_connect(AF1, 'b', True)
        self.assertEqual(future.result(), (AF1, 'b', self.streams['b']))

    def test_two_families_immediate_failure(self) -> None:
        """Tests a failed connect with two families."""
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending((AF1, 'b'), (AF2, 'c'))
        self.resolve_connect(AF1, 'b', False)
        self.resolve_connect(AF2, 'c', True)
        self.assertEqual(future.result(), (AF2, 'c', self.streams['c']))

    def test_two_families_timeout(self) -> None:
        """Tests a successful connect with two families and a timeout."""
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, 'a'))
        conn.on_timeout()
        self.assert_pending((AF1, 'a'), (AF2, 'c'))
        self.resolve_connect(AF2, 'c', True)
        self.assertEqual(future.result(), (AF2, 'c', self.streams['c']))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending()

    def test_success_after_timeout(self) -> None:
        """Tests a successful connect after a timeout."""
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, 'a'))
        conn.on_timeout()
        self.assert_pending((AF1, 'a'), (AF2, 'c'))
        self.resolve_connect(AF1, 'a', True)
        self.assertEqual(future.result(), (AF1, 'a', self.streams['a']))
        self.resolve_connect(AF2, 'c', True)
        self.assertTrue(self.streams.pop('c').closed)

    def test_all_fail(self) -> None:
        """Tests a failed connect with all families."""
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, 'a'))
        conn.on_timeout()
        self.assert_pending((AF1, 'a'), (AF2, 'c'))
        self.resolve_connect(AF2, 'c', False)
        self.assert_pending((AF1, 'a'), (AF2, 'd'))
        self.resolve_connect(AF2, 'd', False)
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending((AF1, 'b'))
        self.assertFalse(future.done())
        self.resolve_connect(AF1, 'b', False)
        self.assertRaises(IOError, future.result)

    def test_one_family_timeout_after_connect_timeout(self) -> None:
        """Tests a failed connect with a timeout after a connect timeout."""
        conn, future = self.start_connect([(AF1, 'a'), (AF1, 'b')])
        self.assert_pending((AF1, 'a'))
        conn.on_connect_timeout()
        self.connect_futures.pop((AF1, 'a'))
        self.assertTrue(self.streams.pop('a').closed)
        conn.on_timeout()
        self.assert_pending()
        self.assertEqual(len(conn.streams), 1)
        self.assert_connector_streams_closed(conn)
        self.assertRaises(TimeoutError, future.result)

    def test_one_family_success_before_connect_timeout(self) -> None:
        """Tests a successful connect before a connect timeout."""
        conn, future = self.start_connect([(AF1, 'a'), (AF1, 'b')])
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', True)
        conn.on_connect_timeout()
        self.assert_pending()
        self.assertFalse(self.streams['a'].closed)
        self.assertEqual(len(conn.streams), 0)
        self.assert_connector_streams_closed(conn)
        self.assertEqual(future.result(), (AF1, 'a', self.streams['a']))

    def test_one_family_second_try_after_connect_timeout(self) -> None:
        """Tests a failed connect with a second try after a connect timeout."""
        conn, future = self.start_connect([(AF1, 'a'), (AF1, 'b')])
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending((AF1, 'b'))
        conn.on_connect_timeout()
        self.connect_futures.pop((AF1, 'b'))
        self.assertTrue(self.streams.pop('b').closed)
        self.assert_pending()
        self.assertEqual(len(conn.streams), 2)
        self.assert_connector_streams_closed(conn)
        self.assertRaises(TimeoutError, future.result)

    def test_one_family_second_try_failure_before_connect_timeout(self) -> None:
        """Tests a failed connect with a second try before a connect timeout."""
        conn, future = self.start_connect([(AF1, 'a'), (AF1, 'b')])
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending((AF1, 'b'))
        self.resolve_connect(AF1, 'b', False)
        conn.on_connect_timeout()
        self.assert_pending()
        self.assertEqual(len(conn.streams), 2)
        self.assert_connector_streams_closed(conn)
        self.assertRaises(IOError, future.result)

    def test_two_family_timeout_before_connect_timeout(self) -> None:
        """Tests a failed connect with two families and a timeout before a connect timeout."""
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, 'a'))
        conn.on_timeout()
        self.assert_pending((AF1, 'a'), (AF2, 'c'))
        conn.on_connect_timeout()
        self.connect_futures.pop((AF1, 'a'))
        self.assertTrue(self.streams.pop('a').closed)
        self.connect_futures.pop((AF2, 'c'))
        self.assertTrue(self.streams.pop('c').closed)
        self.assert_pending()
        self.assertEqual(len(conn.streams), 2)
        self.assert_connector_streams_closed(conn)
        self.assertRaises(TimeoutError, future.result)

    def test_two_family_success_after_timeout(self) -> None:
        """Tests a successful connect with two families and a timeout."""
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, 'a'))
        conn.on_timeout()
        self.assert_pending((AF1, 'a'), (AF2, 'c'))
        self.resolve_connect(AF1, 'a', True)
        self.connect_futures.pop((AF2, 'c'))
        self.assertTrue(self.streams.pop('c').closed)
        self.assert_pending()
        self.assertEqual(len(conn.streams), 1)
        self.assert_connector_streams_closed(conn)
        self.assertEqual(future.result(), (AF1, 'a', self.streams['a']))

    def test_two_family_timeout_after_connect_timeout(self) -> None:
        """Tests a failed connect with two families and a timeout after a connect timeout."""
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, 'a'))
        conn.on_connect_timeout()
        self.connect_futures.pop((AF1, 'a'))
        self.assertTrue(self.streams.pop('a').closed)
        self.assert_pending()
        conn.on_timeout()
        self.assert_pending()
        self.assertEqual(len(conn.streams), 1)
        self.assert_connector_streams_closed(conn)
        self.assertRaises(TimeoutError, future.result)
