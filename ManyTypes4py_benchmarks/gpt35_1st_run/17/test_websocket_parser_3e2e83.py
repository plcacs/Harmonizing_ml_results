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
    if compress:
        compressobj = zlib.compressobj(wbits=-9)
        message = compressobj.compress(message)
        message = message + compressobj.flush(zlib.Z_SYNC_FLUSH)
        if message.endswith(WS_DEFLATE_TRAILING):
            message = message[:-4]
    msg_length = len(message)
    if is_fin:
        header_first_byte = 128 | opcode
    else:
        header_first_byte = opcode
    if compress:
        header_first_byte |= 64
    if msg_length < 126:
        header = PACK_LEN1(header_first_byte, msg_length)
    else:
        assert msg_length < 1 << 16
        header = PACK_LEN2(header_first_byte, 126, msg_length)
    if noheader:
        return message
    else:
        return header + message

def build_close_frame(code: int = 1000, message: bytes = b'', noheader: bool = False) -> bytes:
    return build_frame(PACK_CLOSE_CODE(code) + message, opcode=WSMsgType.CLOSE, noheader=noheader)

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
def out_low_limit(loop: asyncio.AbstractEventLoop, protocol: BaseProtocol) -> WebSocketDataQueue:
    return WebSocketDataQueue(protocol, 16, loop=loop)

@pytest.fixture()
def parser_low_limit(out_low_limit: WebSocketDataQueue) -> PatchableWebSocketReader:
    return PatchableWebSocketReader(out_low_limit, 4 * 1024 * 1024)

@pytest.fixture()
def parser(out: WebSocketDataQueue) -> PatchableWebSocketReader:
    return PatchableWebSocketReader(out, 4 * 1024 * 1024)

def test_feed_data_remembers_exception(parser: PatchableWebSocketReader) -> None:
    """Verify that feed_data remembers an exception was already raised internally."""
    error, data = parser.feed_data(struct.pack('!BB', 96, 0))
    assert error is True
    assert data == b''
    error, data = parser.feed_data(b'')
    assert error is True
    assert data == b''

def test_parse_frame(parser: PatchableWebSocketReader) -> None:
    parser.parse_frame(struct.pack('!BB', 1, 1))
    res = parser.parse_frame(b'1')
    fin, opcode, payload, compress = res[0]
    assert (0, 1, b'1', False) == (fin, opcode, payload, not not compress)

# More test functions with type annotations
