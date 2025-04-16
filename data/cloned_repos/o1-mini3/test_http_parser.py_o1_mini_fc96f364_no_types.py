import asyncio
import re
from contextlib import suppress
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
from unittest import mock
from urllib.parse import quote
import pytest
from multidict import CIMultiDict
from yarl import URL
import aiohttp
from aiohttp import http_exceptions, streams
from aiohttp.base_protocol import BaseProtocol
from aiohttp.helpers import NO_EXTENSIONS
from aiohttp.http_parser import DeflateBuffer, HttpParser, HttpPayloadParser, HttpRequestParser, HttpRequestParserPy, HttpResponseParser, HttpResponseParserPy
from aiohttp.http_writer import HttpVersion
try:
    try:
        import brotlicffi as brotli
    except ImportError:
        import brotli
except ImportError:
    brotli = None
REQUEST_PARSERS: List[Type[HttpParser[Any]]] = [HttpRequestParserPy]
RESPONSE_PARSERS: List[Type[HttpParser[Any]]] = [HttpResponseParserPy]
with suppress(ImportError):
    from aiohttp.http_parser import HttpRequestParserC, HttpResponseParserC
    REQUEST_PARSERS.append(HttpRequestParserC)
    RESPONSE_PARSERS.append(HttpResponseParserC)

@pytest.fixture
def protocol():
    return mock.create_autospec(BaseProtocol, spec_set=True, instance=True)

def _gen_ids(parsers):
    return ['py-parser' if parser.__module__ == 'aiohttp.http_parser' else 'c-parser' for parser in parsers]

@pytest.fixture(params=REQUEST_PARSERS, ids=_gen_ids(REQUEST_PARSERS))
def parser(loop, protocol, request):
    return request.param(protocol, loop, 2 ** 16, max_line_size=8190, max_field_size=8190)

@pytest.fixture(params=REQUEST_PARSERS, ids=_gen_ids(REQUEST_PARSERS))
def request_cls(request):
    return request.param

@pytest.fixture(params=RESPONSE_PARSERS, ids=_gen_ids(RESPONSE_PARSERS))
def response(loop, protocol, request):
    return request.param(protocol, loop, 2 ** 16, max_line_size=8190, max_field_size=8190, read_until_eof=True)

@pytest.fixture(params=RESPONSE_PARSERS, ids=_gen_ids(RESPONSE_PARSERS))
def response_cls(request):
    return request.param

@pytest.mark.skipif(NO_EXTENSIONS, reason='Extensions available but not imported')
def test_c_parser_loaded():
    assert 'HttpRequestParserC' in dir(aiohttp.http_parser)
    assert 'HttpResponseParserC' in dir(aiohttp.http_parser)
    assert 'RawRequestMessageC' in dir(aiohttp.http_parser)
    assert 'RawResponseMessageC' in dir(aiohttp.http_parser)

def test_parse_headers(parser):
    text = b'GET /test HTTP/1.1\r\ntest: a line\r\ntest2: data\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg = messages[0][0]
    assert list(msg.headers.items()) == [('test', 'a line'), ('test2', 'data')]
    assert msg.raw_headers == ((b'test', b'a line'), (b'test2', b'data'))
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade

def test_reject_obsolete_line_folding(parser):
    text = b'GET /test HTTP/1.1\r\ntest: line\r\n Content-Length: 48\r\ntest2: data\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

@pytest.mark.skipif(NO_EXTENSIONS, reason='Only tests C parser.')
def test_invalid_character(loop, protocol, request):
    parser = HttpRequestParserC(protocol, loop, 2 ** 16, max_line_size=8190, max_field_size=8190)
    text = b'POST / HTTP/1.1\r\nHost: localhost:8080\r\nSet-Cookie: abc\x01def\r\n\r\n'
    error_detail = re.escape(":\n    \n    b'Set-Cookie: abc\\x01def'\n                     ^")
    with pytest.raises(http_exceptions.BadHttpMessage, match=error_detail):
        parser.feed_data(text)

@pytest.mark.skipif(NO_EXTENSIONS, reason='Only tests C parser.')
def test_invalid_linebreak(loop, protocol, request):
    parser = HttpRequestParserC(protocol, loop, 2 ** 16, max_line_size=8190, max_field_size=8190)
    text = b'GET /world HTTP/1.1\r\nHost: 127.0.0.1\n\r\n'
    error_detail = re.escape(":\n    \n    b'Host: 127.0.0.1\\n'\n                     ^")
    with pytest.raises(http_exceptions.BadHttpMessage, match=error_detail):
        parser.feed_data(text)

def test_cve_2023_37276(parser):
    text = b'POST / HTTP/1.1\r\nHost: localhost:8080\r\nX-Abc: \rxTransfer-Encoding: chunked\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

@pytest.mark.parametrize('rfc9110_5_6_2_token_delim', '"(),/:;<=>?@[\\]{}')
def test_bad_header_name(parser, rfc9110_5_6_2_token_delim):
    text = f'POST / HTTP/1.1\r\nhead{rfc9110_5_6_2_token_delim}er: val\r\n\r\n'.encode()
    if rfc9110_5_6_2_token_delim == ':':
        parser.feed_data(text)
        return
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

@pytest.mark.parametrize('hdr', ('Content-Length: -5', 'Content-Length: +256', 'Content-Length: ¹', 'Content-Length: 𝟙', 'Foo: abc\rdef', 'Bar: abc\ndef', 'Baz: abc\x00def', 'Foo : bar', 'Foo\t: bar', 'ÿoo: bar'))
def test_bad_headers(parser, hdr):
    text = f'POST / HTTP/1.1\r\n{hdr}\r\n\r\n'.encode()
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

def test_unpaired_surrogate_in_header_py(loop, protocol):
    parser = HttpRequestParserPy(protocol, loop, 2 ** 16, max_line_size=8190, max_field_size=8190)
    text = b'POST / HTTP/1.1\r\n\xff\r\n\r\n'
    message: Optional[bytes] = None
    try:
        parser.feed_data(text)
    except http_exceptions.InvalidHeader as e:
        message = e.message.encode('utf-8')
    assert message is not None

def test_content_length_transfer_encoding(parser):
    text = b'GET / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\nTransfer-Encoding: a\r\n\r\n' + b'apple\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

def test_bad_chunked_py(loop, protocol):
    """Test that invalid chunked encoding doesn't allow content-length to be used."""
    parser = HttpRequestParserPy(protocol, loop, 2 ** 16, max_line_size=8190, max_field_size=8190)
    text = b'GET / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n0_2e\r\n\r\n' + b'GET / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n0\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    assert isinstance(messages[0][1].exception(), http_exceptions.TransferEncodingError)

@pytest.mark.skipif('HttpRequestParserC' not in dir(aiohttp.http_parser), reason='C based HTTP parser not available')
def test_bad_chunked_c(loop, protocol):
    """C parser behaves differently. Maybe we should align them later."""
    parser = HttpRequestParserC(protocol, loop, 2 ** 16, max_line_size=8190, max_field_size=8190)
    text = b'GET / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n0_2e\r\n\r\n' + b'GET / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n0\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

def test_whitespace_before_header(parser):
    text = b'GET / HTTP/1.1\r\n\tContent-Length: 1\r\n\r\nX'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

def test_parse_headers_longline(parser):
    invalid_unicode_byte = b'\xd9'
    header_name = b'Test' + invalid_unicode_byte + b'Header' + b'A' * 8192
    text = b'GET /test HTTP/1.1\r\n' + header_name + b': test\r\n' + b'\r\n' + b'\r\n'
    with pytest.raises((http_exceptions.LineTooLong, http_exceptions.BadHttpMessage)):
        parser.feed_data(text)

@pytest.fixture
def xfail_c_parser_status(request):
    if isinstance(request.getfixturevalue('parser'), HttpRequestParserPy):
        return
    request.node.add_marker(pytest.mark.xfail(reason='Regression test for Py parser. May match C behaviour later.', raises=http_exceptions.BadStatusLine))

@pytest.mark.usefixtures('xfail_c_parser_status')
def test_parse_unusual_request_line(parser):
    text = b'#smol //a HTTP/1.3\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg, _ = messages[0]
    assert msg.compression is None
    assert not msg.upgrade
    assert msg.method == '#smol'
    assert msg.path == '//a'
    assert msg.version == (1, 3)

def test_parse(parser):
    text = b'GET /test HTTP/1.1\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg, _ = messages[0]
    assert msg.compression is None
    assert not msg.upgrade
    assert msg.method == 'GET'
    assert msg.path == '/test'
    assert msg.version == (1, 1)

async def test_parse_body(parser: HttpRequestParser) -> None:
    text = b'GET /test HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody'
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    _, payload = messages[0]
    body = await payload.read(4)
    assert body == b'body'

async def test_parse_body_with_CRLF(parser: HttpRequestParser) -> None:
    text = b'\r\nGET /test HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody'
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    _, payload = messages[0]
    body = await payload.read(4)
    assert body == b'body'

def test_parse_delayed(parser):
    text = b'GET /test HTTP/1.1\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 0
    assert not upgrade
    messages, upgrade, tail = parser.feed_data(b'\r\n')
    assert len(messages) == 1
    msg = messages[0][0]
    assert msg.method == 'GET'

def test_headers_multi_feed(parser):
    text1 = b'GET /test HTTP/1.1\r\n'
    text2 = b'test: line'
    text3 = b' continue\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text1)
    assert len(messages) == 0
    messages, upgrade, tail = parser.feed_data(text2)
    assert len(messages) == 0
    messages, upgrade, tail = parser.feed_data(text3)
    assert len(messages) == 1
    msg = messages[0][0]
    assert list(msg.headers.items()) == [('test', 'line continue')]
    assert msg.raw_headers == ((b'test', b'line continue'),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade

def test_headers_split_field(parser):
    text1 = b'GET /test HTTP/1.1\r\n'
    text2 = b't'
    text3 = b'es'
    text4 = b't: value\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text1)
    messages, upgrade, tail = parser.feed_data(text2)
    messages, upgrade, tail = parser.feed_data(text3)
    assert len(messages) == 0
    messages, upgrade, tail = parser.feed_data(text4)
    assert len(messages) == 1
    msg = messages[0][0]
    assert list(msg.headers.items()) == [('test', 'value')]
    assert msg.raw_headers == ((b'test', b'value'),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade

def test_parse_headers_multi(parser):
    text = b'GET /test HTTP/1.1\r\nSet-Cookie: c1=cookie1\r\nSet-Cookie: c2=cookie2\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg = messages[0][0]
    assert list(msg.headers.items()) == [('Set-Cookie', 'c1=cookie1'), ('Set-Cookie', 'c2=cookie2')]
    assert msg.raw_headers == ((b'Set-Cookie', b'c1=cookie1'), (b'Set-Cookie', b'c2=cookie2'))
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade

def test_conn_default_1_0(parser):
    text = b'GET /test HTTP/1.0\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close

def test_conn_default_1_1(parser):
    text = b'GET /test HTTP/1.1\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close

def test_conn_close(parser):
    text = b'GET /test HTTP/1.1\r\nconnection: close\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close

def test_conn_close_1_0(parser):
    text = b'GET /test HTTP/1.0\r\nconnection: close\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close

def test_conn_keep_alive_1_0(parser):
    text = b'GET /test HTTP/1.0\r\nconnection: keep-alive\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close

def test_conn_keep_alive_1_1(parser):
    text = b'GET /test HTTP/1.1\r\nconnection: keep-alive\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close

def test_conn_other_1_0(parser):
    text = b'GET /test HTTP/1.0\r\nconnection: test\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close

def test_conn_other_1_1(parser):
    text = b'GET /test HTTP/1.1\r\nconnection: test\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close

def test_request_chunked(parser):
    text = b'GET /test HTTP/1.1\r\ntransfer-encoding: chunked\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg, payload = messages[0]
    assert msg.chunked
    assert not upgrade
    assert isinstance(payload, streams.StreamReader)

def test_request_te_chunked_with_content_length(parser):
    text = b'GET /test HTTP/1.1\r\ncontent-length: 1234\r\ntransfer-encoding: chunked\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage, match="Transfer-Encoding can't be present with Content-Length"):
        parser.feed_data(text)

def test_request_te_chunked123(parser):
    text = b'GET /test HTTP/1.1\r\ntransfer-encoding: chunked123\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage, match='Request has invalid `Transfer-Encoding`'):
        parser.feed_data(text)

async def test_request_te_last_chunked(parser: HttpRequestParser) -> None:
    text = b'GET /test HTTP/1.1\r\nTransfer-Encoding: not, chunked\r\n\r\n1\r\nT\r\n3\r\nest\r\n0\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    assert await messages[0][1].read() == b'Test'

def test_request_te_first_chunked(parser):
    text = b'GET /test HTTP/1.1\r\nTransfer-Encoding: chunked, not\r\n\r\n1\r\nT\r\n3\r\nest\r\n0\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage, match='nvalid `Transfer-Encoding`'):
        parser.feed_data(text)

def test_conn_upgrade(parser):
    text = b'GET /test HTTP/1.1\r\nconnection: upgrade\r\nupgrade: websocket\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close
    assert msg.upgrade
    assert upgrade

def test_bad_upgrade(parser):
    """Test not upgraded if missing Upgrade header."""
    text = b'GET /test HTTP/1.1\r\nconnection: upgrade\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.upgrade
    assert not upgrade

def test_compression_empty(parser):
    text = b'GET /test HTTP/1.1\r\ncontent-encoding: \r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression is None

def test_compression_deflate(parser):
    text = b'GET /test HTTP/1.1\r\ncontent-encoding: deflate\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression == 'deflate'

def test_compression_gzip(parser):
    text = b'GET /test HTTP/1.1\r\ncontent-encoding: gzip\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression == 'gzip'

@pytest.mark.skipif(brotli is None, reason='brotli is not installed')
def test_compression_brotli(parser):
    text = b'GET /test HTTP/1.1\r\ncontent-encoding: br\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression == 'br'

def test_compression_unknown(parser):
    text = b'GET /test HTTP/1.1\r\ncontent-encoding: compress\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression is None

def test_url_connect(parser):
    text = b'CONNECT www.google.com HTTP/1.1\r\ncontent-length: 0\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg, payload = messages[0]
    assert upgrade
    assert msg.url == URL.build(authority='www.google.com')

def test_headers_connect(parser):
    text = b'CONNECT www.google.com HTTP/1.1\r\ncontent-length: 0\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg, payload = messages[0]
    assert upgrade
    assert isinstance(payload, streams.StreamReader)

def test_url_absolute(parser):
    text = b'GET https://www.google.com/path/to.html HTTP/1.1\r\ncontent-length: 0\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg, payload = messages[0]
    assert not upgrade
    assert msg.method == 'GET'
    assert msg.url == URL('https://www.google.com/path/to.html')

def test_headers_old_websocket_key1(parser):
    text = b'GET /test HTTP/1.1\r\nSEC-WEBSOCKET-KEY1: line\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

def test_headers_content_length_err_1(parser):
    text = b'GET /test HTTP/1.1\r\ncontent-length: line\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

def test_headers_content_length_err_2(parser):
    text = b'GET /test HTTP/1.1\r\ncontent-length: -1\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)
_pad: Dict[bytes, str] = {b'': 'empty', b'\x00': 'NUL', b' ': 'SP', b'  ': 'SPSP', b'\n': 'LF', b'\xa0': 'NBSP', b'\t ': 'TABSP'}

@pytest.mark.parametrize('hdr', [b'', b'foo'], ids=['name-empty', 'with-name'])
@pytest.mark.parametrize('pad2', _pad.keys(), ids=['post-' + n for n in _pad.values()])
@pytest.mark.parametrize('pad1', _pad.keys(), ids=['pre-' + n for n in _pad.values()])
def test_invalid_header_spacing(parser, pad1, pad2, hdr):
    text = b'GET /test HTTP/1.1\r\n%s%s%s: value\r\n\r\n' % (pad1, hdr, pad2)
    if pad1 == pad2 == b'' and hdr != b'':
        parser.feed_data(text)
        return
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

def test_empty_header_name(parser):
    text = b'GET /test HTTP/1.1\r\n:test\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

def test_invalid_header(parser):
    text = b'GET /test HTTP/1.1\r\ntest line\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

def test_invalid_name(parser):
    text = b'GET /test HTTP/1.1\r\ntest[]: line\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

@pytest.mark.parametrize('size', [40960, 8191])
def test_max_header_field_size(parser, size):
    name = b't' * size
    text = b'GET /test HTTP/1.1\r\n' + name + b':data\r\n\r\n'
    match = f'400, message:\n  Got more than 8190 bytes \\({size}\\) when reading'
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        parser.feed_data(text)

def test_max_header_field_size_under_limit(parser):
    name = b't' * 8190
    text = b'GET /test HTTP/1.1\r\n' + name + b':data\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.method == 'GET'
    assert msg.path == '/test'
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict({name.decode(): 'data'})
    assert msg.raw_headers == ((name, b'data'),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL('/test')

@pytest.mark.parametrize('size', [40960, 8191])
def test_max_header_value_size(parser, size):
    name = b't' * size
    text = b'GET /test HTTP/1.1\r\ndata:' + name + b'\r\n\r\n'
    match = f'400, message:\n  Got more than 8190 bytes \\({size}\\) when reading'
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        parser.feed_data(text)

def test_max_header_value_size_under_limit(parser):
    value = b'A' * 8190
    text = b'GET /test HTTP/1.1\r\ndata:' + value + b'\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.method == 'GET'
    assert msg.path == '/test'
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict({'data': value.decode()})
    assert msg.raw_headers == ((b'data', value),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL('/test')

@pytest.mark.parametrize('size', [40965, 8191])
def test_max_header_value_size_continuation(response, size):
    name = b'T' * (size - 5)
    text = b'HTTP/1.1 200 Ok\r\ndata: test\r\n ' + name + b'\r\n\r\n'
    match = f'400, message:\n  Got more than 8190 bytes \\({size}\\) when reading'
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        response.feed_data(text)

def test_max_header_value_size_continuation_under_limit(response):
    value = b'A' * 8185
    text = b'HTTP/1.1 200 Ok\r\ndata: test\r\n ' + value + b'\r\n\r\n'
    messages, upgrade, tail = response.feed_data(text)
    msg = messages[0][0]
    assert msg.code == 200
    assert msg.reason == 'Ok'
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict({'data': 'test ' + value.decode()})
    assert msg.raw_headers == ((b'data', b'test ' + value),)
    assert msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked

def test_http_request_parser(parser):
    text = b'GET /path HTTP/1.1\r\n\r\n'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.method == 'GET'
    assert msg.path == '/path'
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict()
    assert msg.raw_headers == ()
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL('/path')

def test_http_request_bad_status_line(parser):
    text = b'getpath \r\n\r\n'
    with pytest.raises(http_exceptions.BadStatusLine) as exc_info:
        parser.feed_data(text)
    assert '\\n' not in exc_info.value.message
_num: Dict[bytes, str] = {'𝟙'.encode(): 'utf8digit', '¹'.encode(): 'utf8number', '¹'.encode('latin-1'): 'latin1number'}

@pytest.mark.parametrize('nonascii_digit', _num.keys(), ids=_num.values())
def test_http_request_bad_status_line_number(parser, nonascii_digit):
    text = b'GET /digit HTTP/1.' + nonascii_digit + b'\r\n\r\n'
    with pytest.raises(http_exceptions.BadStatusLine):
        parser.feed_data(text)

def test_http_request_bad_status_line_separator(parser):
    utf8sep = 'ﷺ'.encode()
    text = b'GET /ligature HTTP/1' + utf8sep + b'1\r\n\r\n'
    with pytest.raises(http_exceptions.BadStatusLine):
        parser.feed_data(text)

def test_http_request_bad_status_line_whitespace(parser):
    text = b'GET\n/path\x0cHTTP/1.1\r\n\r\n'
    with pytest.raises(http_exceptions.BadStatusLine):
        parser.feed_data(text)

def test_http_request_message_after_close(parser):
    text = b'GET / HTTP/1.1\r\nConnection: close\r\n\r\nInvalid\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage, match='Data after `Connection: close`'):
        parser.feed_data(text)

def test_http_request_upgrade(parser):
    text = b'GET /test HTTP/1.1\r\nconnection: upgrade\r\nupgrade: websocket\r\n\r\nsome raw data'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close
    assert msg.upgrade
    assert upgrade
    assert tail == b'some raw data'

async def test_http_request_upgrade_unknown(parser: HttpRequestParser) -> None:
    text = b'POST / HTTP/1.1\r\nConnection: Upgrade\r\nContent-Length: 2\r\nUpgrade: unknown\r\nContent-Type: application/json\r\n\r\n{}'
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close
    assert msg.upgrade
    assert not upgrade
    assert not msg.chunked
    assert tail == b''
    assert await messages[0][-1].read() == b'{}'

@pytest.fixture
def xfail_c_parser_url(request):
    if isinstance(request.getfixturevalue('parser'), HttpRequestParserPy):
        return
    request.node.add_marker(pytest.mark.xfail(reason='Regression test for Py parser. May match C behaviour later.', raises=http_exceptions.InvalidURLError))

@pytest.mark.usefixtures('xfail_c_parser_url')
def test_http_request_parser_utf8_request_line(parser):
    messages, upgrade, tail = parser.feed_data(b'GET /P\xc3\xbcnktchen\xa0\xef\xb7 HTTP/1.1\r\n' + 'sTeP:  ßnek\t\xa0  \r\n\r\n'.encode())
    msg = messages[0][0]
    assert msg.method == 'GET'
    assert msg.path == '/Pünktchen\udca0\udcef\udcb7'
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict([('STEP', 'ßnek\t\xa0')])
    assert msg.raw_headers == ((b'sTeP', 'ßnek\t\xa0'.encode()),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL.build(path='/Pünktchen\udca0\udcef\udcb7', encoded=True)

def test_http_request_parser_utf8(parser):
    text = 'GET /path HTTP/1.1\r\nx-test:тест\r\n\r\n'.encode()
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.method == 'GET'
    assert msg.path == '/path'
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict([('X-TEST', 'тест')])
    assert msg.raw_headers == ((b'x-test', 'тест'.encode()),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL('/path')

def test_http_request_parser_non_utf8(parser):
    text = 'GET /path HTTP/1.1\r\nx-test:тест\r\n\r\n'.encode('cp1251')
    msg = parser.feed_data(text)[0][0][0]
    assert msg.method == 'GET'
    assert msg.path == '/path'
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict([('X-TEST', 'тест'.encode('cp1251').decode('utf8', 'surrogateescape'))])
    assert msg.raw_headers == ((b'x-test', 'тест'.encode('cp1251')),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL('/path')

def test_http_request_parser_two_slashes(parser):
    text = b'GET //path HTTP/1.1\r\n\r\n'
    msg = parser.feed_data(text)[0][0][0]
    assert msg.method == 'GET'
    assert msg.path == '//path'
    assert msg.url.path == '//path'
    assert msg.version == (1, 1)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked

@pytest.mark.parametrize('rfc9110_5_6_2_token_delim', [bytes([i]) for i in b'"(),/:;<=>?@[\\]{}'])
def test_http_request_parser_bad_method(parser, rfc9110_5_6_2_token_delim):
    with pytest.raises(http_exceptions.BadHttpMethod):
        parser.feed_data(rfc9110_5_6_2_token_delim + b'ET" /get HTTP/1.1\r\n\r\n')

def test_http_request_parser_bad_version(parser):
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(b'GET //get HT/11\r\n\r\n')

def test_http_request_parser_bad_version_number(parser):
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(b'GET /test HTTP/1.32\r\n\r\n')

def test_http_request_parser_bad_ascii_uri(parser):
    with pytest.raises(http_exceptions.InvalidURLError):
        parser.feed_data(b'GET ! HTTP/1.1\r\n\r\n')

def test_http_request_parser_bad_nonascii_uri(parser):
    with pytest.raises(http_exceptions.InvalidURLError):
        parser.feed_data(b'GET \xff HTTP/1.1\r\n\r\n')

@pytest.mark.parametrize('size', [40965, 8191])
def test_http_request_max_status_line(parser, size):
    path = b't' * (size - 5)
    match = f'400, message:\n  Got more than 8190 bytes \\({size}\\) when reading'
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        parser.feed_data(b'GET /path' + path + b' HTTP/1.1\r\n\r\n')

def test_http_request_max_status_line_under_limit(parser):
    path = b't' * (8190 - 5)
    messages, upgraded, tail = parser.feed_data(b'GET /path' + path + b' HTTP/1.1\r\n\r\n')
    msg = messages[0][0]
    assert msg.method == 'GET'
    assert msg.path == '/path' + path.decode()
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict()
    assert msg.raw_headers == ()
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL('/path' + path.decode())

def test_http_response_parser_utf8(response):
    text = 'HTTP/1.1 200 Ok\r\nx-test:тест\r\n\r\n'.encode()
    messages, upgraded, tail = response.feed_data(text)
    assert len(messages) == 1
    msg = messages[0][0]
    assert msg.version == (1, 1)
    assert msg.code == 200
    assert msg.reason == 'Ok'
    assert msg.headers == CIMultiDict([('X-TEST', 'тест')])
    assert msg.raw_headers == ((b'x-test', 'тест'.encode()),)
    assert not upgraded
    assert not tail

def test_http_response_parser_utf8_without_reason(response):
    text = 'HTTP/1.1 200 \r\nx-test:тест\r\n\r\n'.encode()
    messages, upgraded, tail = response.feed_data(text)
    assert len(messages) == 1
    msg = messages[0][0]
    assert msg.version == (1, 1)
    assert msg.code == 200
    assert msg.reason == ''
    assert msg.headers == CIMultiDict([('X-TEST', 'тест')])
    assert msg.raw_headers == ((b'x-test', 'тест'.encode()),)
    assert not upgraded
    assert not tail

def test_http_response_parser_obs_line_folding(response):
    text = b'HTTP/1.1 200 Ok\r\ntest: line\r\n continue\r\n\r\n'
    messages, upgraded, tail = response.feed_data(text)
    assert len(messages) == 1
    msg = messages[0][0]
    assert msg.version == (1, 1)
    assert msg.code == 200
    assert msg.reason == 'Ok'
    assert msg.headers == CIMultiDict([('TEST', 'line continue')])
    assert msg.raw_headers == ((b'test', b'line continue'),)
    assert not upgraded
    assert not tail

@pytest.mark.dev_mode
def test_http_response_parser_strict_obs_line_folding(response):
    text = b'HTTP/1.1 200 Ok\r\ntest: line\r\n continue\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        response.feed_data(text)

@pytest.mark.parametrize('size', [40962, 8191])
def test_http_response_parser_bad_status_line_too_long(response, size):
    reason = b't' * (size - 2)
    match = f'400, message:\n  Got more than 8190 bytes \\({size}\\) when reading'
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        response.feed_data(b'HTTP/1.1 200 Ok' + reason + b'\r\n\r\n')

def test_http_response_parser_status_line_under_limit(response):
    reason = b'O' * 8190
    messages, upgraded, tail = response.feed_data(b'HTTP/1.1 200 Ok' + reason + b'\r\n\r\n')
    msg = messages[0][0]
    assert msg.version == (1, 1)
    assert msg.code == 200
    assert msg.reason == reason.decode()

def test_http_response_parser_bad_version(response):
    with pytest.raises(http_exceptions.BadHttpMessage):
        response.feed_data(b'HT/11 200 Ok\r\n\r\n')

def test_http_response_parser_bad_version_number(response):
    with pytest.raises(http_exceptions.BadHttpMessage):
        response.feed_data(b'HTTP/12.3 200 Ok\r\n\r\n')

def test_http_response_parser_no_reason(response):
    msg = response.feed_data(b'HTTP/1.1 200\r\n\r\n')[0][0][0]
    assert msg.version == (1, 1)
    assert msg.code == 200
    assert msg.reason == ''

def test_http_response_parser_lenient_headers(response):
    messages, upgrade, tail = response.feed_data(b'HTTP/1.1 200 Ok\r\nFoo: abc\x01def\r\n\r\n')
    msg = messages[0][0]
    assert msg.headers['Foo'] == 'abc\x01def'

@pytest.mark.dev_mode
def test_http_response_parser_strict_headers(response):
    if isinstance(response, HttpResponseParserPy):
        pytest.xfail('Py parser is lenient. May update py-parser later.')
    with pytest.raises(http_exceptions.BadHttpMessage):
        response.feed_data(b'HTTP/1.1 200 Ok\r\nFoo: abc\x01def\r\n\r\n')

def test_http_response_parser_bad_crlf(response):
    """Still a lot of dodgy servers sending bad requests like this."""
    messages, upgrade, tail = response.feed_data(b'HTTP/1.0 200 OK\nFoo: abc\nBar: def\n\nBODY\n')
    msg = messages[0][0]
    assert msg.headers['Foo'] == 'abc'
    assert msg.headers['Bar'] == 'def'

async def test_http_response_parser_bad_chunked_lax(response: HttpResponseParser) -> None:
    text = b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5 \r\nabcde\r\n0\r\n\r\n'
    messages, upgrade, tail = response.feed_data(text)
    assert await messages[0][1].read(5) == b'abcde'

@pytest.mark.dev_mode
async def test_http_response_parser_bad_chunked_strict_py(loop: asyncio.AbstractEventLoop, protocol: BaseProtocol) -> None:
    """Test the Py parser with a chunked payload that has a LF in the chunk extensions."""
    response = HttpResponseParserPy(protocol, loop, 2 ** 16, max_line_size=8190, max_field_size=8190)
    text = b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5 \r\nabcde\r\n0\r\n\r\n'
    messages, upgrade, tail = response.feed_data(text)
    assert isinstance(messages[0][1].exception(), http_exceptions.TransferEncodingError)

@pytest.mark.dev_mode
@pytest.mark.skipif('HttpRequestParserC' not in dir(aiohttp.http_parser), reason='C based HTTP parser not available')
async def test_http_response_parser_bad_chunked_strict_c(loop: asyncio.AbstractEventLoop, protocol: BaseProtocol) -> None:
    """Test the C parser with a chunked payload that has a LF in the chunk extensions."""
    response = HttpResponseParserC(protocol, loop, 2 ** 16, max_line_size=8190, max_field_size=8190)
    text = b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5 \r\nabcde\r\n0\r\n\r\n'
    with pytest.raises(http_exceptions.BadHttpMessage):
        response.feed_data(text)

def test_whitespace_before_header(parser):
    text = b'GET / HTTP/1.1\r\n\tContent-Length: 1\r\n\r\nX'
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)

async def test_parse_chunked_payload_split_chunks(response: HttpResponseParser) -> None:
    network_chunks = (b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n', b'5\r\nfi', b'rst', b'\r\n', b'6', b'\r', b'\n', b'second\r', b'\n0\r\n\r\n')
    reader = response.feed_data(network_chunks[0])[0][0][1]
    for c in network_chunks[1:]:
        response.feed_data(c)
    assert response.feed_eof() is None
    assert reader.is_eof()
    assert await reader.read() == b'firstsecond'

class TestParsePayload:

    async def test_parse_eof_payload(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out)
        p.feed_data(b'data')
        p.feed_eof()
        assert out.is_eof()
        assert [bytearray(b'data')] == list(out._buffer)

    async def test_parse_length_payload_eof(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, length=4)
        p.feed_data(b'da')
        with pytest.raises(http_exceptions.ContentLengthError):
            p.feed_eof()

    async def test_parse_chunked_payload_size_error(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, chunked=True)
        with pytest.raises(http_exceptions.TransferEncodingError):
            p.feed_data(b'blah\r\n')
        assert isinstance(out.exception(), http_exceptions.TransferEncodingError)

    async def test_parse_chunked_payload_split_end(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, chunked=True)
        p.feed_data(b'4\r\nasdf\r\n0\r\n\r\n')
        assert out.is_eof()
        assert b'asdf' == b''.join(out._buffer)

    async def test_parse_chunked_payload_split_end2(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, chunked=True)
        p.feed_data(b'4\r\nasdf\r\n0\r\n\r')
        p.feed_data(b'\n')
        assert out.is_eof()
        assert b'asdf' == b''.join(out._buffer)

    async def test_parse_chunked_payload_split_end_trailers(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, chunked=True)
        p.feed_data(b'4\r\nasdf\r\n0\r\n')
        p.feed_data(b'Content-MD5: 912ec803b2ce49e4a541068d495ab570\r\n')
        p.feed_data(b'\r\n')
        assert out.is_eof()
        assert b'asdf' == b''.join(out._buffer)

    async def test_parse_chunked_payload_split_end_trailers2(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, chunked=True)
        p.feed_data(b'4\r\nasdf\r\n0\r\nContent-MD5: 912ec803b2ce49e4a541068d495ab570\r\n\r')
        p.feed_data(b'\n')
        assert out.is_eof()
        assert b'asdf' == b''.join(out._buffer)

    async def test_parse_chunked_payload_split_end_trailers3(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, chunked=True)
        p.feed_data(b'4\r\nasdf\r\n0\r\nContent-MD5: ')
        p.feed_data(b'912ec803b2ce49e4a541068d495ab570\r\n\r\n')
        assert out.is_eof()
        assert b'asdf' == b''.join(out._buffer)

    async def test_parse_chunked_payload_split_end_trailers4(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, chunked=True)
        p.feed_data(b'4\r\nasdf\r\n0\r\nC')
        p.feed_data(b'ontent-MD5: 912ec803b2ce49e4a541068d495ab570\r\n\r\n')
        assert out.is_eof()
        assert b'asdf' == b''.join(out._buffer)

    async def test_http_payload_parser_length(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, length=2)
        eof, tail = p.feed_data(b'1245')
        assert eof
        assert b'12' == out._buffer[0]
        assert b'45' == tail

    async def test_http_payload_parser_deflate(self, protocol: BaseProtocol) -> None:
        COMPRESSED = b'x\x9cKI,I\x04\x00\x04\x00\x01\x9b'
        length = len(COMPRESSED)
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, length=length, compression='deflate')
        p.feed_data(COMPRESSED)
        assert b'data' == out._buffer[0]
        assert out.is_eof()

    async def test_http_payload_parser_deflate_no_hdrs(self, protocol: BaseProtocol) -> None:
        """Tests incorrectly formed data (no zlib headers)."""
        COMPRESSED = b'KI,I\x04\x00'
        length = len(COMPRESSED)
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, length=length, compression='deflate')
        p.feed_data(COMPRESSED)
        assert b'data' == out._buffer[0]
        assert out.is_eof()

    async def test_http_payload_parser_deflate_light(self, protocol: BaseProtocol) -> None:
        COMPRESSED = b'\x18\x95KI,I\x04\x00\x04\x00\x01\x9b'
        length = len(COMPRESSED)
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, length=length, compression='deflate')
        p.feed_data(COMPRESSED)
        assert b'data' == out._buffer[0]
        assert out.is_eof()

    async def test_http_payload_parser_deflate_split(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, compression='deflate')
        p.feed_data(b'x')
        p.feed_data(b'\x9cKI,I\x04\x00\x04\x00\x01\x9b')
        p.feed_eof()
        assert b'data' == out._buffer[0]

    async def test_http_payload_parser_deflate_split_err(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, compression='deflate')
        p.feed_data(b'K')
        p.feed_data(b'I,I\x04\x00')
        p.feed_eof()
        assert b'data' == out._buffer[0]

    async def test_http_payload_parser_length_zero(self, protocol: BaseProtocol) -> None:
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, length=0)
        assert p.done
        assert out.is_eof()

    @pytest.mark.skipif(brotli is None, reason='brotli is not installed')
    async def test_http_payload_brotli(self, protocol: BaseProtocol) -> None:
        compressed = brotli.compress(b'brotli data')
        out = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        p = HttpPayloadParser(out, length=len(compressed), compression='br')
        p.feed_data(compressed)
        assert b'brotli data' == out._buffer[0]
        assert out.is_eof()

class TestDeflateBuffer:

    async def test_feed_data(self, protocol: BaseProtocol) -> None:
        buf = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        dbuf = DeflateBuffer(buf, 'deflate')
        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.decompress_sync.return_value = b'line'
        dbuf.feed_data(b'xxxx')
        assert [b'line'] == list(buf._buffer)

    async def test_feed_data_err(self, protocol: BaseProtocol) -> None:
        buf = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        dbuf = DeflateBuffer(buf, 'deflate')
        exc = ValueError()
        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.decompress_sync.side_effect = exc
        with pytest.raises(http_exceptions.ContentEncodingError):
            dbuf.feed_data(b'xsomedata')

    async def test_feed_eof(self, protocol: BaseProtocol) -> None:
        buf = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        dbuf = DeflateBuffer(buf, 'deflate')
        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.flush.return_value = b'line'
        dbuf.feed_eof()
        assert [b'line'] == list(buf._buffer)
        assert buf._eof

    async def test_feed_eof_err_deflate(self, protocol: BaseProtocol) -> None:
        buf = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        dbuf = DeflateBuffer(buf, 'deflate')
        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.flush.return_value = b'line'
        dbuf.decompressor.eof = False
        with pytest.raises(http_exceptions.ContentEncodingError):
            dbuf.feed_eof()

    async def test_feed_eof_no_err_gzip(self, protocol: BaseProtocol) -> None:
        buf = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        dbuf = DeflateBuffer(buf, 'gzip')
        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.flush.return_value = b'line'
        dbuf.decompressor.eof = False
        dbuf.feed_eof()
        assert [b'line'] == list(buf._buffer)

    async def test_feed_eof_no_err_brotli(self, protocol: BaseProtocol) -> None:
        buf = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        dbuf = DeflateBuffer(buf, 'br')
        dbuf.decompressor = mock.Mock()
        dbuf.decompressor.flush.return_value = b'line'
        dbuf.decompressor.eof = False
        dbuf.feed_eof()
        assert [b'line'] == list(buf._buffer)

    async def test_empty_body(self, protocol: BaseProtocol) -> None:
        buf = aiohttp.StreamReader(protocol, 2 ** 16, loop=asyncio.get_running_loop())
        dbuf = DeflateBuffer(buf, 'deflate')
        dbuf.feed_eof()
        assert buf.at_eof()