import asyncio
import pickle
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast
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

def test_parse_frame_length0(parser: PatchableWebSocketReader) -> None:
    fin, opcode, payload, compress = parser.parse_frame(struct.pack('!BB', 1, 0))[0]
    assert (0, 1, b'', False) == (fin, opcode, payload, not not compress)

def test_parse_frame_length2(parser: PatchableWebSocketReader) -> None:
    parser.parse_frame(struct.pack('!BB', 1, 126))
    parser.parse_frame(struct.pack('!H', 4))
    res = parser.parse_frame(b'1234')
    fin, opcode, payload, compress = res[0]
    assert (0, 1, b'1234', False) == (fin, opcode, payload, not not compress)

def test_parse_frame_length2_multi_byte(parser: PatchableWebSocketReader) -> None:
    """Ensure a multi-byte length is parsed correctly."""
    expected_payload = b'1' * 32768
    parser.parse_frame(struct.pack('!BB', 1, 126))
    parser.parse_frame(struct.pack('!H', 32768))
    res = parser.parse_frame(b'1' * 32768)
    fin, opcode, payload, compress = res[0]
    assert (0, 1, expected_payload, False) == (fin, opcode, payload, not not compress)

def test_parse_frame_length2_multi_byte_multi_packet(parser: PatchableWebSocketReader) -> None:
    """Ensure a multi-byte length with multiple packets is parsed correctly."""
    expected_payload = b'1' * 32768
    assert parser.parse_frame(struct.pack('!BB', 1, 126)) == []
    assert parser.parse_frame(struct.pack('!H', 32768)) == []
    assert parser.parse_frame(b'1' * 8192) == []
    assert parser.parse_frame(b'1' * 8192) == []
    assert parser.parse_frame(b'1' * 8192) == []
    res = parser.parse_frame(b'1' * 8192)
    fin, opcode, payload, compress = res[0]
    assert len(payload) == 32768
    assert (0, 1, expected_payload, False) == (fin, opcode, payload, not not compress)

def test_parse_frame_length4(parser: PatchableWebSocketReader) -> None:
    parser.parse_frame(struct.pack('!BB', 1, 127))
    parser.parse_frame(struct.pack('!Q', 4))
    fin, opcode, payload, compress = parser.parse_frame(b'1234')[0]
    assert (0, 1, b'1234', False) == (fin, opcode, payload, not not compress)

def test_parse_frame_mask(parser: PatchableWebSocketReader) -> None:
    parser.parse_frame(struct.pack('!BB', 1, 129))
    parser.parse_frame(b'0001')
    fin, opcode, payload, compress = parser.parse_frame(b'1')[0]
    assert (0, 1, b'\x01', False) == (fin, opcode, payload, not not compress)

def test_parse_frame_header_reversed_bits(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with pytest.raises(WebSocketError):
        parser.parse_frame(struct.pack('!BB', 96, 0))

def test_parse_frame_header_control_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with pytest.raises(WebSocketError):
        parser.parse_frame(struct.pack('!BB', 8, 0))

@pytest.mark.xfail()
def test_parse_frame_header_new_data_err(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with pytest.raises(WebSocketError):
        parser.parse_frame(struct.pack('!BB', 0, 0))

def test_parse_frame_header_payload_size(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with pytest.raises(WebSocketError):
        parser.parse_frame(struct.pack('!BB', 136, 126))

@pytest.mark.parametrize(argnames='data', argvalues=[b'', bytearray(b''), memoryview(b'')], ids=['bytes', 'bytearray', 'memoryview'])
def test_ping_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader, data: Union[bytes, bytearray, memoryview]) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(1, WSMsgType.PING, b'data', False)]
        parser.feed_data(data)
        res = out._buffer[0]
        assert res == WSMessagePing(data=b'data', size=4, extra='')

def test_pong_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(1, WSMsgType.PONG, b'data', False)]
        parser.feed_data(b'')
        res = out._buffer[0]
        assert res == WSMessagePong(data=b'data', size=4, extra='')

def test_close_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(1, WSMsgType.CLOSE, b'', False)]
        parser.feed_data(b'')
        res = out._buffer[0]
        assert res == WSMessageClose(data=0, size=0, extra='')

def test_close_frame_info(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(1, WSMsgType.CLOSE, b'0112345', False)]
        parser.feed_data(b'')
        res = out._buffer[0]
        assert res == WSMessageClose(data=12337, size=7, extra='12345')

def test_close_frame_invalid(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(1, WSMsgType.CLOSE, b'1', False)]
        parser.feed_data(b'')
        exc = out.exception()
        assert isinstance(exc, WebSocketError)
        assert exc.code == WSCloseCode.PROTOCOL_ERROR

def test_close_frame_invalid_2(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    data = build_close_frame(code=1)
    with pytest.raises(WebSocketError) as ctx:
        parser._feed_data(data)
    assert ctx.value.code == WSCloseCode.PROTOCOL_ERROR

def test_close_frame_unicode_err(parser: PatchableWebSocketReader) -> None:
    data = build_close_frame(code=1000, message=b'\xf4\x90\x80\x80')
    with pytest.raises(WebSocketError) as ctx:
        parser._feed_data(data)
    assert ctx.value.code == WSCloseCode.INVALID_TEXT

def test_unknown_frame(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(1, WSMsgType.CONTINUATION, b'', False)]
        parser.feed_data(b'')
        assert isinstance(out.exception(), WebSocketError)

def test_simple_text(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    data = build_frame(b'text', WSMsgType.TEXT)
    parser._feed_data(data)
    res = out._buffer[0]
    assert res == WSMessageText(data='text', size=4, extra='')

def test_simple_text_unicode_err(parser: PatchableWebSocketReader) -> None:
    data = build_frame(b'\xf4\x90\x80\x80', WSMsgType.TEXT)
    with pytest.raises(WebSocketError) as ctx:
        parser._feed_data(data)
    assert ctx.value.code == WSCloseCode.INVALID_TEXT

def test_simple_binary(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(1, WSMsgType.BINARY, b'binary', False)]
        parser.feed_data(b'')
        res = out._buffer[0]
        assert res == WSMessageBinary(data=b'binary', size=6, extra='')

def test_fragmentation_header(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    data = build_frame(b'a', WSMsgType.TEXT)
    parser._feed_data(data[:1])
    parser._feed_data(data[1:])
    res = out._buffer[0]
    assert res == WSMessageText(data='a', size=1, extra='')

def test_continuation(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    data1 = build_frame(b'line1', WSMsgType.TEXT, is_fin=False)
    parser._feed_data(data1)
    data2 = build_frame(b'line2', WSMsgType.CONTINUATION)
    parser._feed_data(data2)
    res = out._buffer[0]
    assert res == WSMessageText(data='line1line2', size=10, extra='')

def test_continuation_with_ping(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(0, WSMsgType.TEXT, b'line1', False), (0, WSMsgType.PING, b'', False), (1, WSMsgType.CONTINUATION, b'line2', False)]
        data1 = build_frame(b'line1', WSMsgType.TEXT, is_fin=False)
        parser._feed_data(data1)
        data2 = build_frame(b'', WSMsgType.PING)
        parser._feed_data(data2)
        data3 = build_frame(b'line2', WSMsgType.CONTINUATION)
        parser._feed_data(data3)
        res = out._buffer[0]
        assert res == WSMessagePing(data=b'', size=0, extra='')
        res = out._buffer[1]
        assert res == WSMessageText(data='line1line2', size=10, extra='')

def test_continuation_err(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(0, WSMsgType.TEXT, b'line1', False), (1, WSMsgType.TEXT, b'line2', False)]
        with pytest.raises(WebSocketError):
            parser._feed_data(b'')

def test_continuation_with_close(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(0, WSMsgType.TEXT, b'line1', False), (0, WSMsgType.CLOSE, build_close_frame(1002, b'test', noheader=True), False), (1, WSMsgType.CONTINUATION, b'line2', False)]
        parser.feed_data(b'')
        res = out._buffer[0]
        assert res == WSMessageClose(data=1002, size=6, extra='test')
        res = out._buffer[1]
        assert res == WSMessageText(data='line1line2', size=10, extra='')

def test_continuation_with_close_unicode_err(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(0, WSMsgType.TEXT, b'line1', False), (0, WSMsgType.CLOSE, build_close_frame(1000, b'\xf4\x90\x80\x80', noheader=True), False), (1, WSMsgType.CONTINUATION, b'line2', False)]
        with pytest.raises(WebSocketError) as ctx:
            parser._feed_data(b'')
        assert ctx.value.code == WSCloseCode.INVALID_TEXT

def test_continuation_with_close_bad_code(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(0, WSMsgType.TEXT, b'line1', False), (0, WSMsgType.CLOSE, build_close_frame(1, b'test', noheader=True), False), (1, WSMsgType.CONTINUATION, b'line2', False)]
        with pytest.raises(WebSocketError) as ctx:
            parser._feed_data(b'')
        assert ctx.value.code == WSCloseCode.PROTOCOL_ERROR

def test_continuation_with_close_bad_payload(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(0, WSMsgType.TEXT, b'line1', False), (0, WSMsgType.CLOSE, b'1', False), (1, WSMsgType.CONTINUATION, b'line2', False)]
        with pytest.raises(WebSocketError) as ctx:
            parser._feed_data(b'')
        assert ctx.value.code, WSCloseCode.PROTOCOL_ERROR

def test_continuation_with_close_empty(out: WebSocketDataQueue, parser: PatchableWebSocketReader) -> None:
    with mock.patch.object(parser, 'parse_frame', autospec=True) as m:
        m.return_value = [(0, WSMsgType.TEXT, b'line1', False), (0, WSMsgType.CLOSE, b'', False), (1, WSMsgType.CONTINUATION, b'line2', False)]
        parser.feed_data(b'')
        res = out._buffer[0]
        assert res == WSMessageClose(data=0, size=0, extra='')
        res = out._buffer[1]
        assert res == WSMessageText(data='line1line2', size=10, extra='')
websocket_mask_data = b'some very long data for masking by websocket'
websocket_mask_mask = b'1234'
websocket_mask_masked = b'B]^Q\x11DVFH\x12_[_U\x13PPFR\x14W]A\x14\\S@_X\\T\x14SK\x13CTP@[RYV@'

def test_websocket_mask_python() -> None:
    message = bytearray(websocket_mask_data)
    _websocket_helpers._websocket_mask_python(websocket_mask_mask, message)
    assert message == websocket_mask_masked

@pytest.mark.skipif(not hasattr(_websocket_helpers, '_websocket_mask_cython'), reason='Requires Cython')
def test_websocket_mask_cython() -> None:
    message = bytearray(websocket_mask_data)
    _websocket_helpers._websocket_mask_cython(websocket_mask_mask, message)
    assert message == websocket_mask_masked
    assert _websocket_helpers.websocket_mask is _websocket_helpers._websocket_mask_cython

def test_websocket_mask_python_empty() -> None:
    message = bytearray()
    _websocket_helpers._websocket_mask_python(websocket_mask_mask, message)
    assert message == bytearray()

@pytest.mark.skipif(not hasattr(_websocket_helpers, '_websocket_mask_cython'), reason='Requires Cython')
def test_websocket_mask_cython_empty() -> None:
    message = bytearray()
    _websocket_helpers._websocket_mask_cython(websocket_mask_mask, message)
    assert message == bytearray()

def test_parse_compress_frame_single(parser: PatchableWebSocketReader) -> None:
    parser.parse_frame(struct.pack('!BB', 193, 1))
    res = parser.parse_frame(b'1')
    fin, opcode, payload, compress = res[0]
    assert (1, 1, b'1', True) == (fin, opcode, payload, not not compress)

def test_parse_compress_frame_multi(parser: PatchableWebSocketReader) -> None:
    parser.parse_frame(struct.pack('!BB', 65, 126))
    parser.parse_frame(struct.pack('!H', 4))
    res = parser.parse_frame(b'1234')
    fin, opcode, payload, compress = res[0]
    assert (0, 1, b'1234', True) == (fin, opcode, payload, not not compress)
    parser.parse_frame(struct.pack('!BB', 129, 126))
    parser.parse_frame(struct.pack('!H', 4))
    res = parser.parse_frame(b'1234')
    fin, opcode, payload, compress = res[0]
    assert (1, 1, b'1234', True) == (fin, opcode, payload, not not compress)
    parser.parse_frame(struct.pack('!BB', 129, 126))
    parser.parse_frame(struct.pack('!H', 4))
    res = parser.parse_frame(b'1234')
    fin, opcode, payload, compress = res[0]
    assert (1, 1, b'1234', False) == (fin, opcode, payload, not not compress)

def test_parse_compress_error_frame(parser: PatchableWebSocketReader) -> None:
    parser.parse_frame(struct.pack('!BB', 65, 1))
    parser.parse_frame(b'1')
    with pytest.raises(WebSocketError) as ctx:
        parser.parse_frame(struct.pack('!BB', 193, 1))
        parser.parse_frame(b'1')
    assert ctx.value.code == WSCloseCode.PROTOCOL_ERROR

def test_parse_no_compress_frame_single(out: WebSocketDataQueue) -> None:
    parser_no_compress = WebSocketReader(out, 0, compress=False)
    with pytest.raises(WebSocketError) as ctx:
        parser_no_compress.parse_frame(struct.pack('!BB', 193, 1))
        parser_no_compress.parse_frame(b'1')
    assert ctx.value.code == WSCloseCode.PROTOCOL_ERROR

def test_msg_too_large(out: WebSocketDataQueue) -> None:
    parser = WebSocketReader(out, 256, compress=False)
    data = build_frame(b'text' * 256, WSMsgType.TEXT)
    with pytest.raises(WebSocketError) as ctx:
        parser._feed_data(data)
    assert ctx.value.code == WSCloseCode.MESSAGE_TOO_BIG

def test_msg_too_large_not_fin(out: WebSocketDataQueue) -> None:
    parser = WebSocketReader(out, 256, compress=False)
    data = build_frame(b'text' * 256, WSMsgType.TEXT, is_fin=False)
    with pytest.raises(WebSocketError) as ctx:
        parser._feed_data(data)
    assert ctx.value.code == WSCloseCode.MESSAGE_TOO_BIG

def test_compressed_msg_too_large(out: WebSocketDataQueue) -> None:
    parser = WebSocketReader(out, 256, compress=True)
    data = build_frame(b'aaa' * 256, WSMsgType.TEXT, compress=True)
    with pytest.raises(WebSocketError) as ctx:
        parser._feed_data(data)
    assert ctx.value.code == WSCloseCode.MESSAGE_TOO_BIG

class TestWebSocketError:

    def test_ctor(self) -> None:
        err = WebSocketError(WSCloseCode.PROTOCOL_ERROR, 'Something invalid')
        assert err.code == WSCloseCode.PROTOCOL_ERROR
        assert str(err) == 'Something invalid'

    def test_pickle(self) -> None:
        err = WebSocketError(WSCloseCode.PROTOCOL_ERROR, 'Something invalid')
        err.foo = 'bar'
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            pickled = pickle.dumps(err, proto)
            err2 = pickle.loads(pickled)
            assert err2.code == WSCloseCode.PROTOCOL_ERROR
            assert str(err2) == 'Something invalid'
            assert err2.foo == 'bar'

def test_flow_control_binary(protocol: BaseProtocol, out_low_limit: WebSocketDataQueue, parser_low_limit: PatchableWebSocketReader) -> None:
    large_payload = b'b' * (1 + 16 * 2)
    large_payload_size = len(large_payload)
    with mock.patch.object(parser_low_limit, 'parse_frame', autospec=True) as m:
        m.return_value = [(1, WSMsgType.BINARY, large_payload, False)]
        parser_low_limit.feed_data(b'')
    res = out_low_limit._buffer[0]
    assert res == WSMessageBinary(data=large_payload, size=large_payload_size, extra='')
    assert protocol._reading_paused is True

def test_flow_control_multi_byte_text(protocol: BaseProtocol, out_low_limit: WebSocketDataQueue, parser_low_limit: PatchableWebSocketReader) -> None:
    large_payload_text = '𒀁' * (1 + 16 * 2)
    large_payload = large_payload_text.encode('utf-8')
    large_payload_size = len(large_payload)
    with mock.patch.object(parser_low_limit, 'parse_frame', autospec=True) as m:
        m.return_value = [(1, WSMsgType.TEXT, large_payload, False)]
        parser_low_limit.feed_data(b'')
    res = out_low_limit._buffer[0]
    assert res == WSMessageText(data=large_payload_text, size=large_payload_size, extra='')
    assert protocol._reading_paused is True
