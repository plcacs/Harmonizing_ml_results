import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, List
from uuid import uuid4
from fastapi import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
from freqtrade.rpc.api_server.ws.proxy import WebSocketProxy
from freqtrade.rpc.api_server.ws.serializer import HybridJSONWebSocketSerializer, WebSocketSerializer
from freqtrade.rpc.api_server.ws.ws_types import WebSocketType
from freqtrade.rpc.api_server.ws_schemas import WSMessageSchemaType

logger: logging.Logger = logging.getLogger(__name__)

class WebSocketChannel:
    def __init__(self, websocket: Any, channel_id: str = None, serializer_cls: Any = HybridJSONWebSocketSerializer, send_throttle: float = 0.01) -> None:
    def __repr__(self) -> str:
    @property
    def raw_websocket(self) -> Any:
    @property
    def remote_addr(self) -> Any:
    @property
    def avg_send_time(self) -> float:
    def _calc_send_limit(self) -> None:
    async def send(self, message: Any, use_timeout: bool = False) -> None:
    async def recv(self) -> Any:
    async def ping(self) -> Any:
    async def accept(self) -> Any:
    async def close(self) -> None:
    def is_closed(self) -> bool:
    def set_subscriptions(self, subscriptions: List[str]) -> None:
    def subscribed_to(self, message_type: str) -> bool:
    async def run_channel_tasks(self, *tasks: Any, **kwargs: Any) -> Any:
    async def cancel_channel_tasks(self) -> None:
    async def __aiter__(self) -> AsyncIterator:
    
@asynccontextmanager
async def create_channel(websocket: Any, **kwargs: Any) -> Any:
