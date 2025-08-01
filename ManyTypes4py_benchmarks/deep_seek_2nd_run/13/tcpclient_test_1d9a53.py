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
from typing import Any, Optional, Set, Sequence, Tuple, Union, cast
if typing.TYPE_CHECKING:
    from tornado.iostream import IOStream
    from typing import List, Dict, Tuple, Set

AF1: int = 1
AF2: int = 2

class TestTCPServer(TCPServer):
    def __init__(self, family: int) -> None:
        super().__init__()
        self.streams: List[IOStream] = []
        self.queue: Queue[IOStream] = Queue()
        sockets = bind_sockets(0, 'localhost', family)
        self.add_sockets(sockets)
        self.port: int = sockets[0].getsockname()[1]

    def handle_stream(self, stream: IOStream, address: Any) -> None:
        self.streams.append(stream)
        self.queue.put(stream)

    def stop(self) -> None:
        super().stop()
        for stream in self.streams:
            stream.close()

class TCPClientTest(AsyncTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.server: Optional[TestTCPServer] = None
        self.client: TCPClient = TCPClient()

    def start_server(self, family: int) -> int:
        self.server = TestTCPServer(family)
        return self.server.port

    def stop_server(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None

    def tearDown(self) -> None:
        self.client.close()
        self.stop_server()
        super().tearDown()

    def skipIfLocalhostV4(self) -> None:
        addrinfo = self.io_loop.run_sync(lambda: Resolver().resolve('localhost', 80))
        families: Set[int] = {addr[0] for addr in addrinfo}
        if socket.AF_INET6 not in families:
            self.skipTest('localhost does not resolve to ipv6')

    @gen_test
    def do_test_connect(self, family: int, host: str, source_ip: Optional[str] = None, source_port: Optional[int] = None) -> None:
        port = self.start_server(family)
        stream = (yield self.client.connect(host, port, source_ip=source_ip, source_port=source_port, af=family))
        assert self.server is not None
        server_stream = (yield self.server.queue.get())
        with closing(stream):
            stream.write(b'hello')
            data = (yield server_stream.read_bytes(5))
            self.assertEqual(data, b'hello')

    def test_connect_ipv4_ipv4(self) -> None:
        self.do_test_connect(socket.AF_INET, '127.0.0.1')

    def test_connect_ipv4_dual(self) -> None:
        self.do_test_connect(socket.AF_INET, 'localhost')

    @skipIfNoIPv6
    def test_connect_ipv6_ipv6(self) -> None:
        self.skipIfLocalhostV4()
        self.do_test_connect(socket.AF_INET6, '::1')

    @skipIfNoIPv6
    def test_connect_ipv6_dual(self) -> None:
        self.skipIfLocalhostV4()
        self.do_test_connect(socket.AF_INET6, 'localhost')

    def test_connect_unspec_ipv4(self) -> None:
        self.do_test_connect(socket.AF_UNSPEC, '127.0.0.1')

    @skipIfNoIPv6
    def test_connect_unspec_ipv6(self) -> None:
        self.skipIfLocalhostV4()
        self.do_test_connect(socket.AF_UNSPEC, '::1')

    def test_connect_unspec_dual(self) -> None:
        self.do_test_connect(socket.AF_UNSPEC, 'localhost')

    @gen_test
    def test_refused_ipv4(self) -> None:
        cleanup_func, port = refusing_port()
        self.addCleanup(cleanup_func)
        with self.assertRaises(IOError):
            yield self.client.connect('127.0.0.1', port)

    def test_source_ip_fail(self) -> None:
        """Fail when trying to use the source IP Address '8.8.8.8'."""
        self.assertRaises(socket.error, self.do_test_connect, socket.AF_INET, '127.0.0.1', source_ip='8.8.8.8')

    def test_source_ip_success(self) -> None:
        """Success when trying to use the source IP Address '127.0.0.1'."""
        self.do_test_connect(socket.AF_INET, '127.0.0.1', source_ip='127.0.0.1')

    @skipIfNonUnix
    def test_source_port_fail(self) -> None:
        """Fail when trying to use source port 1."""
        if getpass.getuser() == 'root':
            self.skipTest('running as root')
        self.assertRaises(socket.error, self.do_test_connect, socket.AF_INET, '127.0.0.1', source_port=1)

    @gen_test
    def test_connect_timeout(self) -> None:
        timeout = 0.05

        class TimeoutResolver(Resolver):
            def resolve(self, *args: Any, **kwargs: Any) -> Future:
                return Future()
        with self.assertRaises(TimeoutError):
            yield TCPClient(resolver=TimeoutResolver()).connect('1.2.3.4', 12345, timeout=timeout)

class TestConnectorSplit(unittest.TestCase):
    def test_one_family(self) -> None:
        primary, secondary = _Connector.split([(AF1, 'a'), (AF1, 'b')])
        self.assertEqual(primary, [(AF1, 'a'), (AF1, 'b')])
        self.assertEqual(secondary, [])

    def test_mixed(self) -> None:
        primary, secondary = _Connector.split([(AF1, 'a'), (AF2, 'b'), (AF1, 'c'), (AF2, 'd')])
        self.assertEqual(primary, [(AF1, 'a'), (AF1, 'c')])
        self.assertEqual(secondary, [(AF2, 'b'), (AF2, 'd')])

class ConnectorTest(AsyncTestCase):
    class FakeStream:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def setUp(self) -> None:
        super().setUp()
        self.connect_futures: Dict[Tuple[int, str], Future] = {}
        self.streams: Dict[str, ConnectorTest.FakeStream] = {}
        self.addrinfo: List[Tuple[int, str]] = [(AF1, 'a'), (AF1, 'b'), (AF2, 'c'), (AF2, 'd')]

    def tearDown(self) -> None:
        for stream in self.streams.values():
            self.assertFalse(stream.closed)
        super().tearDown()

    def create_stream(self, af: int, addr: str) -> Tuple[ConnectorTest.FakeStream, Future]:
        stream = ConnectorTest.FakeStream()
        self.streams[addr] = stream
        future = Future()
        self.connect_futures[af, addr] = future
        return (stream, future)

    def assert_pending(self, *keys: Tuple[int, str]) -> None:
        self.assertEqual(sorted(self.connect_futures.keys()), sorted(keys))

    def resolve_connect(self, af: int, addr: str, success: bool) -> None:
        future = self.connect_futures.pop((af, addr))
        if success:
            future.set_result(self.streams[addr])
        else:
            self.streams.pop(addr)
            future.set_exception(IOError())
        self.io_loop.add_callback(self.stop)
        self.wait()

    def assert_connector_streams_closed(self, conn: _Connector) -> None:
        for stream in conn.streams:
            self.assertTrue(stream.closed)

    def start_connect(self, addrinfo: Sequence[Tuple[int, str]]) -> Tuple[_Connector, Future]:
        conn = _Connector(addrinfo, self.create_stream)
        future = conn.start(3600, connect_timeout=self.io_loop.time() + 3600)
        return (conn, future)

    def test_immediate_success(self) -> None:
        conn, future = self.start_connect(self.addrinfo)
        self.assertEqual(list(self.connect_futures.keys()), [(AF1, 'a')])
        self.resolve_connect(AF1, 'a', True)
        self.assertEqual(future.result(), (AF1, 'a', self.streams['a']))

    def test_immediate_failure(self) -> None:
        conn, future = self.start_connect([(AF1, 'a')])
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assertRaises(IOError, future.result)

    def test_one_family_second_try(self) -> None:
        conn, future = self.start_connect([(AF1, 'a'), (AF1, 'b')])
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending((AF1, 'b'))
        self.resolve_connect(AF1, 'b', True)
        self.assertEqual(future.result(), (AF1, 'b', self.streams['b']))

    def test_one_family_second_try_failure(self) -> None:
        conn, future = self.start_connect([(AF1, 'a'), (AF1, 'b')])
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending((AF1, 'b'))
        self.resolve_connect(AF1, 'b', False)
        self.assertRaises(IOError, future.result)

    def test_one_family_second_try_timeout(self) -> None:
        conn, future = self.start_connect([(AF1, 'a'), (AF1, 'b')])
        self.assert_pending((AF1, 'a'))
        conn.on_timeout()
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending((AF1, 'b'))
        self.resolve_connect(AF1, 'b', True)
        self.assertEqual(future.result(), (AF1, 'b', self.streams['b']))

    def test_two_families_immediate_failure(self) -> None:
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, 'a'))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending((AF1, 'b'), (AF2, 'c'))
        self.resolve_connect(AF1, 'b', False)
        self.resolve_connect(AF2, 'c', True)
        self.assertEqual(future.result(), (AF2, 'c', self.streams['c']))

    def test_two_families_timeout(self) -> None:
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, 'a'))
        conn.on_timeout()
        self.assert_pending((AF1, 'a'), (AF2, 'c'))
        self.resolve_connect(AF2, 'c', True)
        self.assertEqual(future.result(), (AF2, 'c', self.streams['c']))
        self.resolve_connect(AF1, 'a', False)
        self.assert_pending()

    def test_success_after_timeout(self) -> None:
        conn, future = self.start_connect(self.addrinfo)
        self.assert_pending((AF1, 'a'))
        conn.on_timeout()
        self.assert_pending((AF1, 'a'), (AF2, 'c'))
        self.resolve_connect(AF1, 'a', True)
        self.assertEqual(future.result(), (AF1, 'a', self.streams['a']))
        self.resolve_connect(AF2, 'c', True)
        self.assertTrue(self.streams.pop('c').closed)

    def test_all_fail(self) -> None:
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
        self.assertRaises(