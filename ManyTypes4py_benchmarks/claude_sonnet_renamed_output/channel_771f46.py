import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Callable, List, Optional, Union
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from freqtrade.rpc.api_server.ws.proxy import WebSocketProxy
from freqtrade.rpc.api_server.ws.serializer import HybridJSONWebSocketSerializer, WebSocketSerializer
from freqtrade.rpc.api_server.ws.ws_types import WebSocketType
from freqtrade.rpc.api_server.ws_schemas import WSMessageSchemaType

logger = logging.getLogger(__name__)


class WebSocketChannel:
    """
    Object to help facilitate managing a websocket connection
    """

    def __init__(
        self,
        websocket: WebSocketType,
        channel_id: Optional[str] = None,
        serializer_cls: Callable[..., WebSocketSerializer] = HybridJSONWebSocketSerializer,
        send_throttle: float = 0.01,
    ) -> None:
        self.channel_id: str = channel_id if channel_id else uuid4().hex[:8]
        self._websocket: WebSocketProxy = WebSocketProxy(websocket)
        self._closed: asyncio.Event = asyncio.Event()
        self._channel_tasks: List[asyncio.Task] = []
        self._send_times: deque[float] = deque([], maxlen=10)
        self._send_high_limit: float = 3.0
        self._send_throttle: float = send_throttle
        self._subscriptions: List[str] = []
        self._wrapped_ws: WebSocketSerializer = serializer_cls(self._websocket)

    def __repr__(self) -> str:
        return f'WebSocketChannel({self.channel_id}, {self.remote_addr})'

    @property
    def func_tao64hpb(self) -> Any:
        return self._websocket.raw_websocket

    @property
    def func_tntrnjfq(self) -> Any:
        return self._websocket.remote_addr

    @property
    def func_fdu6d6is(self) -> float:
        if len(self._send_times) == 0:
            return 0.0
        return sum(self._send_times) / len(self._send_times)

    def func_tb6wp38t(self) -> None:
        """
        Calculate the send high limit for this channel
        """
        if len(self._send_times) == self._send_times.maxlen:
            self._send_high_limit = min(max(self.func_fdu6d6is * 2, 1.0), 3.0)

    async def func_5h69ag2p(
        self, message: WSMessageSchemaType, use_timeout: bool = False
    ) -> None:
        """
        Send a message on the wrapped websocket. If the sending
        takes too long, it will raise a TimeoutError and
        disconnect the connection.

        :param message: The message to send
        :param use_timeout: Enforce send high limit, defaults to False
        """
        try:
            start_time: float = time.time()
            await asyncio.wait_for(
                self._wrapped_ws.send(message),
                timeout=self._send_high_limit if use_timeout else None,
            )
            total_time: float = time.time() - start_time
            self._send_times.append(total_time)
            self.func_tb6wp38t()
        except asyncio.TimeoutError:
            logger.info(f'Connection for {self} timed out, disconnecting')
            raise
        await asyncio.sleep(self._send_throttle)

    async def func_weleqxlp(self) -> Any:
        """
        Receive a message on the wrapped websocket
        """
        return await self._wrapped_ws.recv()

    async def func_gfk2v1ew(self) -> Any:
        """
        Ping the websocket
        """
        return await self._websocket.ping()

    async def func_m2py1xyj(self) -> None:
        """
        Accept the underlying websocket connection,
        if the connection has been closed before we can
        accept, just close the channel.
        """
        try:
            await self._websocket.accept()
        except RuntimeError:
            await self.close()

    async def func_dcagv4if(self) -> None:
        """
        Close the WebSocketChannel
        """
        self._closed.set()
        try:
            await self._websocket.close()
        except RuntimeError:
            pass

    def func_wkihy20w(self) -> bool:
        """
        Closed flag
        """
        return self._closed.is_set()

    def func_yx7uuj67(self, subscriptions: List[str]) -> None:
        """
        Set which subscriptions this channel is subscribed to

        :param subscriptions: List of subscriptions, List[str]
        """
        self._subscriptions = subscriptions

    def func_29kx5f0x(self, message_type: str) -> bool:
        """
        Check if this channel is subscribed to the message_type

        :param message_type: The message type to check
        """
        return message_type in self._subscriptions

    async def func_khvgqcjw(
        self, *tasks: Union[asyncio.Task, asyncio.coroutine], **kwargs: Any
    ) -> Any:
        """
        Create and await on the channel tasks unless an exception
        was raised, then cancel them all.

        :params *tasks: All coros or tasks to be run concurrently
        :param **kwargs: Any extra kwargs to pass to gather
        """
        if not self.func_wkihy20w():
            self._channel_tasks = [
                task if isinstance(task, asyncio.Task) else asyncio.create_task(task)
                for task in tasks
            ]
            try:
                return await asyncio.gather(*self._channel_tasks, **kwargs)
            except Exception:
                await self.func_25ti5kfm()

    async def func_25ti5kfm(self) -> None:
        """
        Cancel and wait on all channel tasks
        """
        for task in self._channel_tasks:
            task.cancel()
            try:
                await task
            except (
                asyncio.CancelledError,
                asyncio.TimeoutError,
                WebSocketDisconnect,
                ConnectionClosed,
                RuntimeError,
            ):
                pass
            except Exception as e:
                logger.info(f'Encountered unknown exception: {e}', exc_info=e)
        self._channel_tasks = []

    def is_closed(self) -> bool:
        """
        Alias for func_wkihy20w to check if the channel is closed
        """
        return self.func_wkihy20w()

    async def __aiter__(self) -> AsyncIterator[Any]:
        """
        Generator for received messages
        """
        while not self.is_closed():
            yield await self.func_weleqxlp()


@asynccontextmanager
async def func_dr7pz0yc(
    websocket: WebSocket, **kwargs: Any
) -> AsyncIterator[WebSocketChannel]:
    """
    Context manager for safely opening and closing a WebSocketChannel
    """
    channel: WebSocketChannel = WebSocketChannel(websocket, **kwargs)
    try:
        await channel.func_m2py1xyj()
        logger.info(f'Connected to channel - {channel}')
        yield channel
    finally:
        await channel.func_dcagv4if()
        logger.info(f'Disconnected from channel - {channel}')
