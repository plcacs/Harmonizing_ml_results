"""
Benchmark for asyncio TCP server and client performance
transferring 10MB of data.

Author: Kumar Aditya
"""
import asyncio
from pyperf import Runner
import ssl
import os
from typing import Optional, Any, List, Tuple

CHUNK_SIZE: int = ((1024 ** 2) * 10)
SSL_CERT: str = os.path.join(os.path.dirname(__file__), 'ssl_cert.pem')
SSL_KEY: str = os.path.join(os.path.dirname(__file__), 'ssl_key.pem')

async def handle_echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    data: bytes = (b'x' * CHUNK_SIZE)
    for _ in range(100):
        writer.write(data)
        await writer.drain()
    writer.close()
    await writer.wait_closed()

async def main(use_ssl: bool) -> None:
    server_context: Optional[ssl.SSLContext]
    client_context: Optional[ssl.SSLContext]
    if use_ssl:
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(SSL_CERT, SSL_KEY)
        server_context.check_hostname = False
        server_context.verify_mode = ssl.CERT_NONE
        client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        client_context.load_cert_chain(SSL_CERT, SSL_KEY)
        client_context.check_hostname = False
        client_context.verify_mode = ssl.CERT_NONE
    else:
        server_context = None
        client_context = None
    server: asyncio.base_events.Server = await asyncio.start_server(
        handle_echo, '127.0.0.1', 8882, ssl=server_context
    )
    async with server:
        asyncio.create_task(server.start_serving())
        reader: asyncio.StreamReader
        writer: asyncio.StreamWriter
        reader, writer = await asyncio.open_connection('127.0.0.1', 8882, ssl=client_context)
        data_len: int = 0
        while True:
            data: bytes = await reader.read(CHUNK_SIZE)
            if not data:
                break
            data_len += len(data)
        assert (data_len == (CHUNK_SIZE * 100))
        writer.close()
        await writer.wait_closed()

def add_cmdline_args(cmd: List[str], args: Any) -> None:
    if args.ssl:
        cmd.append('--ssl')

if __name__ == '__main__':
    runner: Runner = Runner(add_cmdline_args=add_cmdline_args)
    parser = runner.argparser
    parser.add_argument('--ssl', action='store_true', default=False)
    args: Any = runner.parse_args()
    name: str = ('asyncio_tcp' + ('_ssl' if args.ssl else ''))
    runner.bench_async_func(name, main, args.ssl)
