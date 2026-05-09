import asyncio
import pickle
import struct
import zlib
from typing import Union, List, Tuple, Optional, Any, Iterable, Dict
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
    def parse_frame(self, data: bytes) -> List[Tuple[int, int, bytes, bool]]:
        ...

def build_frame(message: bytes, opcode: int, noheader: bool = ..., is_fin: bool = ..., compress: bool = ...) -> bytes:
    ...

def build_close_frame(code: int = ..., message: bytes = ..., noheader: bool = ...) -> bytes:
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

def test_parse_frame_header_new_data_err(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

def test_parse_frame_header_payload_size(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    ...

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

def test_close_frame_invalid_2(parser: PatchableWebSocketReader) -> None:
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

def test_websocket_mask_python() -> None:
    ...

def test_websocket_mask_cython() -> None:
    ...

def test_websocket_mask_python_empty() -> None:
    ...

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