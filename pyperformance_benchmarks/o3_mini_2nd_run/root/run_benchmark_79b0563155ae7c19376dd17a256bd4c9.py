import sys
import socket
import pyperf
from typing import Dict, Tuple, Any, Generator
from tornado.httpclient import AsyncHTTPClient, HTTPResponse
from tornado.httpserver import HTTPServer
from tornado.gen import coroutine
from tornado.ioloop import IOLoop
from tornado.netutil import bind_sockets
from tornado.web import RequestHandler, Application

HOST: str = '127.0.0.1'
FAMILY: int = socket.AF_INET
CHUNK: bytes = (b'Hello world\n' * 1000)
NCHUNKS: int = 5
CONCURRENCY: int = 150

class MainHandler(RequestHandler):
    @coroutine
    def get(self) -> Generator[Any, Any, None]:
        for i in range(NCHUNKS):
            self.write(CHUNK)
            yield self.flush()

    def compute_etag(self) -> None:
        return None

def make_application() -> Application:
    return Application([('/', MainHandler)])

def make_http_server(request_handler: Application) -> Tuple[HTTPServer, socket.socket]:
    server: HTTPServer = HTTPServer(request_handler)
    sockets: list = bind_sockets(0, HOST, family=FAMILY)
    assert len(sockets) == 1
    server.add_sockets(sockets)
    sock: socket.socket = sockets[0]
    return server, sock

def bench_tornado(loops: int) -> float:
    server, sock = make_http_server(make_application())
    host, port = sock.getsockname()
    url: str = 'http://%s:%s/' % (host, port)
    namespace: Dict[str, float] = {}

    @coroutine
    def run_client() -> Generator[Any, Any, None]:
        client: AsyncHTTPClient = AsyncHTTPClient()
        range_it = range(loops)
        t0: float = pyperf.perf_counter()
        for _ in range_it:
            futures = [client.fetch(url) for j in range(CONCURRENCY)]
            for fut in futures:
                resp: HTTPResponse = yield fut
                buf = resp.buffer
                buf.seek(0, 2)
                assert buf.tell() == (len(CHUNK) * NCHUNKS)
        namespace['dt'] = pyperf.perf_counter() - t0
        client.close()

    IOLoop.current().run_sync(run_client)
    server.stop()
    return namespace['dt']

if __name__ == '__main__':
    if sys.platform == 'win32' and sys.version_info[:2] >= (3, 8):
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Test the performance of HTTP requests with Tornado.'
    runner.bench_time_func('tornado_http', bench_tornado)