#!/usr/bin/env python3
"""
Benchmark for asyncio websocket server and client performance
transferring 1MB of data.

Author: Kumar Aditya
"""
import asyncio
from typing import Any
import pyperf
import websockets.server
import websockets.client
import websockets.exceptions

CHUNK_SIZE: int = (1024 ** 2)
DATA: bytes = b'x' * CHUNK_SIZE
stop: asyncio.Event  = None  # will be initialized in main()

async def handler(websocket: websockets.server.WebSocketServerProtocol) -> None:
    for _ in range(100):
        await websocket.recv()
    stop.set()

async def send(ws: websockets.client.WebSocketClientProtocol) -> None:
    try:
        await ws.send(DATA)
    except websockets.exceptions.ConnectionClosedOK:
        pass

async def main() -> float:
    global stop
    t0: float = pyperf.perf_counter()
    stop = asyncio.Event()
    async with websockets.server.serve(handler, '', 8001):
        async with websockets.client.connect('ws://localhost:8001') as ws:
            await asyncio.gather(*[send(ws) for _ in range(100)])
        await stop.wait()
    return pyperf.perf_counter() - t0

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark asyncio websockets'
    runner.bench_async_func('asyncio_websockets', main)