
'Test the performance of simple HTTP serving and client using the Tornado\nframework.\n\nA trivial "application" is generated which generates a number of chunks of\ndata as a HTTP response\'s body.\n'
import sys
import socket
import pyperf
from tornado.httpclient import AsyncHTTPClient
from tornado.httpserver import HTTPServer
from tornado.gen import coroutine
from tornado.ioloop import IOLoop
from tornado.netutil import bind_sockets
from tornado.web import RequestHandler, Application
HOST = '127.0.0.1'
FAMILY = socket.AF_INET
CHUNK = (b'Hello world\n' * 1000)
NCHUNKS = 5
CONCURRENCY = 150

class MainHandler(RequestHandler):

    @coroutine
    def get(self):
        for i in range(NCHUNKS):
            self.write(CHUNK)
            (yield self.flush())

    def compute_etag(self):
        return None

def make_application():
    return Application([('/', MainHandler)])

def make_http_server(request_handler):
    server = HTTPServer(request_handler)
    sockets = bind_sockets(0, HOST, family=FAMILY)
    assert (len(sockets) == 1)
    server.add_sockets(sockets)
    sock = sockets[0]
    return (server, sock)

def bench_tornado(loops):
    (server, sock) = make_http_server(make_application())
    (host, port) = sock.getsockname()
    url = ('http://%s:%s/' % (host, port))
    namespace = {}

    @coroutine
    def run_client():
        client = AsyncHTTPClient()
        range_it = range(loops)
        t0 = pyperf.perf_counter()
        for _ in range_it:
            futures = [client.fetch(url) for j in range(CONCURRENCY)]
            for fut in futures:
                resp = (yield fut)
                buf = resp.buffer
                buf.seek(0, 2)
                assert (buf.tell() == (len(CHUNK) * NCHUNKS))
        namespace['dt'] = (pyperf.perf_counter() - t0)
        client.close()
    IOLoop.current().run_sync(run_client)
    server.stop()
    return namespace['dt']
if (__name__ == '__main__'):
    if ((sys.platform == 'win32') and (sys.version_info[:2] >= (3, 8))):
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Test the performance of HTTP requests with Tornado.'
    runner.bench_time_func('tornado_http', bench_tornado)
