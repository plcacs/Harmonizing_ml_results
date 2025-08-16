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

def build_close_frame(code: int = 1000, message: bytes = b'', noheader: bool = False) -> bytes:
    ...

@pytest.fixture()
def protocol(loop: asyncio.AbstractEventLoop) -> BaseProtocol:
    ...

@pytest.fixture()
def out(loop: asyncio.AbstractEventLoop) -> WebSocketDataQueue:
    ...

@pytest.fixture()
def out_low_limit(loop: asyncio.AbstractEventLoop, protocol: BaseProtocol) -> WebSocketDataQueue:
    ...

@pytest.fixture()
def parser_low_limit(out_low_limit: WebSocketDataQueue) -> PatchableWebSocketReader:
    ...

@pytest.fixture()
def parser(out: WebSocketDataQueue) -> PatchableWebSocketReader:
    ...

def test_feed_data_remembers_exception(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_length0(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_length2(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_length2_multi_byte(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_length2_multi_byte_multi_packet(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_length4(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_mask(parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_header_reversed_bits(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_header_control_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

@pytest.mark.xfail()
def test_parse_frame_header_new_data_err(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_header_payload_size(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

@pytest.mark.parametrize(argnames='data', argvalues=[b'', bytearray(b''), memoryview(b'')], ids=['bytes', 'bytearray', 'memoryview'])
def test_ping_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader, data: Union[bytes, bytearray, memoryview]) -> None:
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

def test_continuation_with_ping(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_continuation_err(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_continuation_with_close(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_continuation_with_close_unicode_err(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_continuation_with_close_bad_code(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_continuation_with_close_bad_payload(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_continuation_with_close_empty(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

websocket_mask_data = b'some very long data for masking by websocket'
websocket_mask_mask = b'1234'
websocket_mask_masked = b'B]^Q\x11DVFH\x12_[_U\x13PPFR\x14W]A\x14\\S@_X\\T\x14SK\x13CTP@[RYV@'

def test_websocket_mask_python() -> None:
    ...

@pytest.mark.skipif(not hasattr(_websocket_helpers, '_websocket_mask_cython'), reason='Requires Cython')
def test_websocket_mask_cython() -> None:
    ...

def test_websocket_mask_python_empty() -> None:
    ...

@pytest.mark.skipif(not hasattr(_websocket_helpers, '_websocket_mask_cython'), reason='Requires Cython')
def test_websocket_mask_cython_empty() -> None:
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

def test_flow_control_binary(protocol: BaseProtocol, out_low_limit: WebSocketDataQueue, parser_low_limit: PatchableWebSocketReader) -> None:
    ...

def test_flow_control_multi_byte_text(protocol: BaseProtocol, out_low_limit: WebSocketDataQueue, parser_low_limit: PatchableWebSocketReader) -> None:
    ...
