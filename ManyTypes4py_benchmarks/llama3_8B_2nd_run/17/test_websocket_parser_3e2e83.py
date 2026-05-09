import asyncio
import pickle
import struct
import zlib
from typing import Union
from unittest import mock
import pytest
from aiohttp._websocket import helpers as _websocket_helpers
from aiohttp._websocket.helpers import PACK_CLOSE_CODE, PACK_LEN1, PACK_LEN2
from aiohttp._websocket.models import WS_DEFLATE_TRAILING
from aiohttp._websocket.reader import WebSocketDataQueue
from aiohttp.base_protocol import BaseProtocol
from aiohttp.http import WebSocketError, WSCloseCode, WSMsgType
from aiohttp.http_websocket import WebSocketReader, WSMessageBinary, WSMessageClose, WSMessagePing, WSMessagePong, WSMessageText

class PatchableWebSocketReader(WebSocketReader):
    """WebSocketReader subclass that allows for patching parse_frame."""

def build_frame(message: bytes, opcode: int, noheader: bool = False, is_fin: bool = True, compress: bool = False) -> bytes:
    ...

@pytest.fixture()
def protocol(loop: asyncio.AbstractEventLoop) -> BaseProtocol:
    transport = mock.Mock(spec_set=asyncio.Transport)
    protocol = BaseProtocol(loop)
    protocol.connection_made(transport)
    return protocol

@pytest.fixture()
def out(loop: asyncio.AbstractEventLoop) -> WebSocketDataQueue:
    return WebSocketDataQueue(mock.Mock(_reading_paused=False), 2 ** 16, loop=loop)

@pytest.fixture()
def out_low_limit(protocol: BaseProtocol, loop: asyncio.AbstractEventLoop) -> WebSocketDataQueue:
    return WebSocketDataQueue(protocol, 16, loop=loop)

@pytest.fixture()
def parser_low_limit(out_low_limit: WebSocketDataQueue) -> PatchableWebSocketReader:
    return PatchableWebSocketReader(out_low_limit, 4 * 1024 * 1024)

def test_feed_data_remembers_exception(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_length0(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_length2(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_length2_multi_packet(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_length4(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_mask(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_header_reversed_bits(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_header_control_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_header_payload_size(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_ping_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader, data: bytes) -> None:
    ...

def test_pong_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_close_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_close_frame_info(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_close_frame_invalid(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_close_frame_invalid_2(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_close_frame_unicode_err(parser: PatchableWebSocketReader) -> None:
    ...

def test_unknown_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_simple_text(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_simple_text_unicode_err(parser: PatchableWebSocketReader) -> None:
    ...

def test_simple_binary(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_fragmentation_header(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_continuation(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_continuation_err(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_continuation_with_close(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_continuation_with_close_unicode_err(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_websocket_mask_python() -> None:
    ...

def test_websocket_mask_cython() -> None:
    ...

def test_parse_compress_frame_single(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_compress_frame_multi(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_compress_error_frame(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_no_compress_frame_single(out: WebSocketDataQueue) -> None:
    ...

def test_msg_too_large(out: WebSocketDataQueue) -> None:
    ...

def test_msg_too_large_not_fin(out: WebSocketDataQueue) -> None:
    ...

def test_compressed_msg_too_large(out: WebSocketDataQueue) -> None:
    ...

class TestWebSocketError:
    def test_ctor(self) -> None:
        ...

    def test_pickle(self) -> None:
        ...
