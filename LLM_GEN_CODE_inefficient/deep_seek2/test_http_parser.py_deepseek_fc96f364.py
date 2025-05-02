# Tests for aiohttp/protocol.py

import asyncio
import re
from contextlib import suppress
from typing import Any, Dict, Iterable, List, Type, Optional, Tuple, Union
from unittest import mock
from urllib.parse import quote

import pytest
from multidict import CIMultiDict
from yarl import URL

import aiohttp
from aiohttp import http_exceptions, streams
from aiohttp.base_protocol import BaseProtocol
from aiohttp.helpers import NO_EXTENSIONS
from aiohttp.http_parser import (
    DeflateBuffer,
    HttpParser,
    HttpPayloadParser,
    HttpRequestParser,
    HttpRequestParserPy,
    HttpResponseParser,
    HttpResponseParserPy,
)
from aiohttp.http_writer import HttpVersion

try:
    try:
        import brotlicffi as brotli
    except ImportError:
        import brotli
except ImportError:
    brotli = None

REQUEST_PARSERS: List[Type[HttpRequestParser]] = [HttpRequestParserPy]
RESPONSE_PARSERS: List[Type[HttpResponseParser]] = [HttpResponseParserPy]

with suppress(ImportError):
    from aiohttp.http_parser import HttpRequestParserC, HttpResponseParserC

    REQUEST_PARSERS.append(HttpRequestParserC)
    RESPONSE_PARSERS.append(HttpResponseParserC)


@pytest.fixture
def protocol() -> Any:
    return mock.create_autospec(BaseProtocol, spec_set=True, instance=True)


def _gen_ids(parsers: Iterable[Type[HttpParser[Any]]]) -> List[str]:
    return [
        "py-parser" if parser.__module__ == "aiohttp.http_parser" else "c-parser"
        for parser in parsers
    ]


@pytest.fixture(params=REQUEST_PARSERS, ids=_gen_ids(REQUEST_PARSERS))
def parser(
    loop: asyncio.AbstractEventLoop,
    protocol: BaseProtocol,
    request: pytest.FixtureRequest,
) -> HttpRequestParser:
    # Parser implementations
    return request.param(  # type: ignore[no-any-return]
        protocol,
        loop,
        2**16,
        max_line_size=8190,
        max_field_size=8190,
    )


@pytest.fixture(params=REQUEST_PARSERS, ids=_gen_ids(REQUEST_PARSERS))
def request_cls(request: pytest.FixtureRequest) -> Type[HttpRequestParser]:
    # Request Parser class
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=RESPONSE_PARSERS, ids=_gen_ids(RESPONSE_PARSERS))
def response(
    loop: asyncio.AbstractEventLoop,
    protocol: BaseProtocol,
    request: pytest.FixtureRequest,
) -> HttpResponseParser:
    # Parser implementations
    return request.param(  # type: ignore[no-any-return]
        protocol,
        loop,
        2**16,
        max_line_size=8190,
        max_field_size=8190,
        read_until_eof=True,
    )


@pytest.fixture(params=RESPONSE_PARSERS, ids=_gen_ids(RESPONSE_PARSERS))
def response_cls(request: pytest.FixtureRequest) -> Type[HttpResponseParser]:
    # Parser implementations
    return request.param  # type: ignore[no-any-return]


@pytest.mark.skipif(NO_EXTENSIONS, reason="Extensions available but not imported")
def test_c_parser_loaded() -> None:
    assert "HttpRequestParserC" in dir(aiohttp.http_parser)
    assert "HttpResponseParserC" in dir(aiohttp.http_parser)
    assert "RawRequestMessageC" in dir(aiohttp.http_parser)
    assert "RawResponseMessageC" in dir(aiohttp.http_parser)


def test_parse_headers(parser: HttpRequestParser) -> None:
    text = b"""GET /test HTTP/1.1\r
test: a line\r
test2: data\r
\r
"""
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg = messages[0][0]

    assert list(msg.headers.items()) == [("test", "a line"), ("test2", "data")]
    assert msg.raw_headers == ((b"test", b"a line"), (b"test2", b"data"))
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade


def test_reject_obsolete_line_folding(parser: HttpRequestParser) -> None:
    text = b"""GET /test HTTP/1.1\r
test: line\r
 Content-Length: 48\r
test2: data\r
\r
"""
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


@pytest.mark.skipif(NO_EXTENSIONS, reason="Only tests C parser.")
def test_invalid_character(
    loop: asyncio.AbstractEventLoop,
    protocol: BaseProtocol,
    request: pytest.FixtureRequest,
) -> None:
    parser = HttpRequestParserC(
        protocol,
        loop,
        2**16,
        max_line_size=8190,
        max_field_size=8190,
    )
    text = b"POST / HTTP/1.1\r\nHost: localhost:8080\r\nSet-Cookie: abc\x01def\r\n\r\n"
    error_detail = re.escape(
        r""":

    b'Set-Cookie: abc\x01def'
                     ^"""
    )
    with pytest.raises(http_exceptions.BadHttpMessage, match=error_detail):
        parser.feed_data(text)


@pytest.mark.skipif(NO_EXTENSIONS, reason="Only tests C parser.")
def test_invalid_linebreak(
    loop: asyncio.AbstractEventLoop,
    protocol: BaseProtocol,
    request: pytest.FixtureRequest,
) -> None:
    parser = HttpRequestParserC(
        protocol,
        loop,
        2**16,
        max_line_size=8190,
        max_field_size=8190,
    )
    text = b"GET /world HTTP/1.1\r\nHost: 127.0.0.1\n\r\n"
    error_detail = re.escape(
        r""":

    b'Host: 127.0.0.1\n'
                     ^"""
    )
    with pytest.raises(http_exceptions.BadHttpMessage, match=error_detail):
        parser.feed_data(text)


def test_cve_2023_37276(parser: HttpRequestParser) -> None:
    text = b"""POST / HTTP/1.1\r\nHost: localhost:8080\r\nX-Abc: \rxTransfer-Encoding: chunked\r\n\r\n"""
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


@pytest.mark.parametrize(
    "rfc9110_5_6_2_token_delim",
    r'"(),/:;<=>?@[\]{}',
)
def test_bad_header_name(
    parser: HttpRequestParser, rfc9110_5_6_2_token_delim: str
) -> None:
    text = f"POST / HTTP/1.1\r\nhead{rfc9110_5_6_2_token_delim}er: val\r\n\r\n".encode()
    if rfc9110_5_6_2_token_delim == ":":
        # Inserting colon into header just splits name/value earlier.
        parser.feed_data(text)
        return

    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


@pytest.mark.parametrize(
    "hdr",
    (
        "Content-Length: -5",  # https://www.rfc-editor.org/rfc/rfc9110.html#name-content-length
        "Content-Length: +256",
        "Content-Length: \N{SUPERSCRIPT ONE}",
        "Content-Length: \N{MATHEMATICAL DOUBLE-STRUCK DIGIT ONE}",
        "Foo: abc\rdef",  # https://www.rfc-editor.org/rfc/rfc9110.html#section-5.5-5
        "Bar: abc\ndef",
        "Baz: abc\x00def",
        "Foo : bar",  # https://www.rfc-editor.org/rfc/rfc9112.html#section-5.1-2
        "Foo\t: bar",
        "\xffoo: bar",
    ),
)
def test_bad_headers(parser: HttpRequestParser, hdr: str) -> None:
    text = f"POST / HTTP/1.1\r\n{hdr}\r\n\r\n".encode()
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_unpaired_surrogate_in_header_py(
    loop: asyncio.AbstractEventLoop, protocol: BaseProtocol
) -> None:
    parser = HttpRequestParserPy(
        protocol,
        loop,
        2**16,
        max_line_size=8190,
        max_field_size=8190,
    )
    text = b"POST / HTTP/1.1\r\n\xff\r\n\r\n"
    message = None
    try:
        parser.feed_data(text)
    except http_exceptions.InvalidHeader as e:
        message = e.message.encode("utf-8")
    assert message is not None


def test_content_length_transfer_encoding(parser: HttpRequestParser) -> None:
    text = (
        b"GET / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\nTransfer-Encoding: a\r\n\r\n"
        + b"apple\r\n"
    )
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_bad_chunked_py(
    loop: asyncio.AbstractEventLoop, protocol: BaseProtocol
) -> None:
    """Test that invalid chunked encoding doesn't allow content-length to be used."""
    parser = HttpRequestParserPy(
        protocol,
        loop,
        2**16,
        max_line_size=8190,
        max_field_size=8190,
    )
    text = (
        b"GET / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n0_2e\r\n\r\n"
        + b"GET / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n0\r\n\r\n"
    )
    messages, upgrade, tail = parser.feed_data(text)
    assert isinstance(messages[0][1].exception(), http_exceptions.TransferEncodingError)


@pytest.mark.skipif(
    "HttpRequestParserC" not in dir(aiohttp.http_parser),
    reason="C based HTTP parser not available",
)
def test_bad_chunked_c(loop: asyncio.AbstractEventLoop, protocol: BaseProtocol) -> None:
    """C parser behaves differently. Maybe we should align them later."""
    parser = HttpRequestParserC(
        protocol,
        loop,
        2**16,
        max_line_size=8190,
        max_field_size=8190,
    )
    text = (
        b"GET / HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked\r\n\r\n0_2e\r\n\r\n"
        + b"GET / HTTP/1.1\r\nHost: a\r\nContent-Length: 5\r\n\r\n0\r\n\r\n"
    )
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_whitespace_before_header(parser: HttpRequestParser) -> None:
    text = b"GET / HTTP/1.1\r\n\tContent-Length: 1\r\n\r\nX"
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_parse_headers_longline(parser: HttpRequestParser) -> None:
    invalid_unicode_byte = b"\xd9"
    header_name = b"Test" + invalid_unicode_byte + b"Header" + b"A" * 8192
    text = b"GET /test HTTP/1.1\r\n" + header_name + b": test\r\n" + b"\r\n" + b"\r\n"
    with pytest.raises((http_exceptions.LineTooLong, http_exceptions.BadHttpMessage)):
        # FIXME: `LineTooLong` doesn't seem to actually be happening
        parser.feed_data(text)


@pytest.fixture
def xfail_c_parser_status(request: pytest.FixtureRequest) -> None:
    if isinstance(request.getfixturevalue("parser"), HttpRequestParserPy):
        return
    request.node.add_marker(
        pytest.mark.xfail(
            reason="Regression test for Py parser. May match C behaviour later.",
            raises=http_exceptions.BadStatusLine,
        )
    )


@pytest.mark.usefixtures("xfail_c_parser_status")
def test_parse_unusual_request_line(parser: HttpRequestParser) -> None:
    text = b"#smol //a HTTP/1.3\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg, _ = messages[0]
    assert msg.compression is None
    assert not msg.upgrade
    assert msg.method == "#smol"
    assert msg.path == "//a"
    assert msg.version == (1, 3)


def test_parse(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg, _ = messages[0]
    assert msg.compression is None
    assert not msg.upgrade
    assert msg.method == "GET"
    assert msg.path == "/test"
    assert msg.version == (1, 1)


async def test_parse_body(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody"
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    _, payload = messages[0]
    body = await payload.read(4)
    assert body == b"body"


async def test_parse_body_with_CRLF(parser: HttpRequestParser) -> None:
    text = b"\r\nGET /test HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody"
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    _, payload = messages[0]
    body = await payload.read(4)
    assert body == b"body"


def test_parse_delayed(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 0
    assert not upgrade

    messages, upgrade, tail = parser.feed_data(b"\r\n")
    assert len(messages) == 1
    msg = messages[0][0]
    assert msg.method == "GET"


def test_headers_multi_feed(parser: HttpRequestParser) -> None:
    text1 = b"GET /test HTTP/1.1\r\n"
    text2 = b"test: line"
    text3 = b" continue\r\n\r\n"

    messages, upgrade, tail = parser.feed_data(text1)
    assert len(messages) == 0

    messages, upgrade, tail = parser.feed_data(text2)
    assert len(messages) == 0

    messages, upgrade, tail = parser.feed_data(text3)
    assert len(messages) == 1

    msg = messages[0][0]
    assert list(msg.headers.items()) == [("test", "line continue")]
    assert msg.raw_headers == ((b"test", b"line continue"),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade


def test_headers_split_field(parser: HttpRequestParser) -> None:
    text1 = b"GET /test HTTP/1.1\r\n"
    text2 = b"t"
    text3 = b"es"
    text4 = b"t: value\r\n\r\n"

    messages, upgrade, tail = parser.feed_data(text1)
    messages, upgrade, tail = parser.feed_data(text2)
    messages, upgrade, tail = parser.feed_data(text3)
    assert len(messages) == 0
    messages, upgrade, tail = parser.feed_data(text4)
    assert len(messages) == 1

    msg = messages[0][0]
    assert list(msg.headers.items()) == [("test", "value")]
    assert msg.raw_headers == ((b"test", b"value"),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade


def test_parse_headers_multi(parser: HttpRequestParser) -> None:
    text = (
        b"GET /test HTTP/1.1\r\n"
        b"Set-Cookie: c1=cookie1\r\n"
        b"Set-Cookie: c2=cookie2\r\n\r\n"
    )

    messages, upgrade, tail = parser.feed_data(text)
    assert len(messages) == 1
    msg = messages[0][0]

    assert list(msg.headers.items()) == [
        ("Set-Cookie", "c1=cookie1"),
        ("Set-Cookie", "c2=cookie2"),
    ]
    assert msg.raw_headers == (
        (b"Set-Cookie", b"c1=cookie1"),
        (b"Set-Cookie", b"c2=cookie2"),
    )
    assert not msg.should_close
    assert msg.compression is None


def test_conn_default_1_0(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.0\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close


def test_conn_default_1_1(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close


def test_conn_close(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\nconnection: close\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close


def test_conn_close_1_0(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.0\r\nconnection: close\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close


def test_conn_keep_alive_1_0(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.0\r\nconnection: keep-alive\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close


def test_conn_keep_alive_1_1(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\nconnection: keep-alive\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close


def test_conn_other_1_0(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.0\r\nconnection: test\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.should_close


def test_conn_other_1_1(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\nconnection: test\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close


def test_request_chunked(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\ntransfer-encoding: chunked\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg, payload = messages[0]
    assert msg.chunked
    assert not upgrade
    assert isinstance(payload, streams.StreamReader)


def test_request_te_chunked_with_content_length(parser: HttpRequestParser) -> None:
    text = (
        b"GET /test HTTP/1.1\r\n"
        b"content-length: 1234\r\n"
        b"transfer-encoding: chunked\r\n\r\n"
    )
    with pytest.raises(
        http_exceptions.BadHttpMessage,
        match="Transfer-Encoding can't be present with Content-Length",
    ):
        parser.feed_data(text)


def test_request_te_chunked123(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\ntransfer-encoding: chunked123\r\n\r\n"
    with pytest.raises(
        http_exceptions.BadHttpMessage,
        match="Request has invalid `Transfer-Encoding`",
    ):
        parser.feed_data(text)


async def test_request_te_last_chunked(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\nTransfer-Encoding: not, chunked\r\n\r\n1\r\nT\r\n3\r\nest\r\n0\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    # https://www.rfc-editor.org/rfc/rfc9112#section-6.3-2.4.3
    assert await messages[0][1].read() == b"Test"


def test_request_te_first_chunked(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\nTransfer-Encoding: chunked, not\r\n\r\n1\r\nT\r\n3\r\nest\r\n0\r\n\r\n"
    # https://www.rfc-editor.org/rfc/rfc9112#section-6.3-2.4.3
    with pytest.raises(
        http_exceptions.BadHttpMessage,
        match="nvalid `Transfer-Encoding`",
    ):
        parser.feed_data(text)


def test_conn_upgrade(parser: HttpRequestParser) -> None:
    text = (
        b"GET /test HTTP/1.1\r\n"
        b"connection: upgrade\r\n"
        b"upgrade: websocket\r\n\r\n"
    )
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.should_close
    assert msg.upgrade
    assert upgrade


def test_bad_upgrade(parser: HttpRequestParser) -> None:
    """Test not upgraded if missing Upgrade header."""
    text = b"GET /test HTTP/1.1\r\nconnection: upgrade\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert not msg.upgrade
    assert not upgrade


def test_compression_empty(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\ncontent-encoding: \r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression is None


def test_compression_deflate(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\ncontent-encoding: deflate\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression == "deflate"


def test_compression_gzip(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\ncontent-encoding: gzip\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression == "gzip"


@pytest.mark.skipif(brotli is None, reason="brotli is not installed")
def test_compression_brotli(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\ncontent-encoding: br\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression == "br"


def test_compression_unknown(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\ncontent-encoding: compress\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.compression is None


def test_url_connect(parser: HttpRequestParser) -> None:
    text = b"CONNECT www.google.com HTTP/1.1\r\ncontent-length: 0\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg, payload = messages[0]
    assert upgrade
    assert msg.url == URL.build(authority="www.google.com")


def test_headers_connect(parser: HttpRequestParser) -> None:
    text = b"CONNECT www.google.com HTTP/1.1\r\ncontent-length: 0\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg, payload = messages[0]
    assert upgrade
    assert isinstance(payload, streams.StreamReader)


def test_url_absolute(parser: HttpRequestParser) -> None:
    text = (
        b"GET https://www.google.com/path/to.html HTTP/1.1\r\n"
        b"content-length: 0\r\n\r\n"
    )
    messages, upgrade, tail = parser.feed_data(text)
    msg, payload = messages[0]
    assert not upgrade
    assert msg.method == "GET"
    assert msg.url == URL("https://www.google.com/path/to.html")


def test_headers_old_websocket_key1(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\nSEC-WEBSOCKET-KEY1: line\r\n\r\n"

    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_headers_content_length_err_1(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\ncontent-length: line\r\n\r\n"

    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_headers_content_length_err_2(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\ncontent-length: -1\r\n\r\n"

    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


_pad: Dict[bytes, str] = {
    b"": "empty",
    # not a typo. Python likes triple zero
    b"\000": "NUL",
    b" ": "SP",
    b"  ": "SPSP",
    # not a typo: both 0xa0 and 0x0a in case of 8-bit fun
    b"\n": "LF",
    b"\xa0": "NBSP",
    b"\t ": "TABSP",
}


@pytest.mark.parametrize("hdr", [b"", b"foo"], ids=["name-empty", "with-name"])
@pytest.mark.parametrize("pad2", _pad.keys(), ids=["post-" + n for n in _pad.values()])
@pytest.mark.parametrize("pad1", _pad.keys(), ids=["pre-" + n for n in _pad.values()])
def test_invalid_header_spacing(
    parser: HttpRequestParser, pad1: bytes, pad2: bytes, hdr: bytes
) -> None:
    text = b"GET /test HTTP/1.1\r\n%s%s%s: value\r\n\r\n" % (pad1, hdr, pad2)
    if pad1 == pad2 == b"" and hdr != b"":
        # one entry in param matrix is correct: non-empty name, not padded
        parser.feed_data(text)
        return

    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_empty_header_name(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\n:test\r\n\r\n"
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_invalid_header(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\ntest line\r\n\r\n"
    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


def test_invalid_name(parser: HttpRequestParser) -> None:
    text = b"GET /test HTTP/1.1\r\ntest[]: line\r\n\r\n"

    with pytest.raises(http_exceptions.BadHttpMessage):
        parser.feed_data(text)


@pytest.mark.parametrize("size", [40960, 8191])
def test_max_header_field_size(parser: HttpRequestParser, size: int) -> None:
    name = b"t" * size
    text = b"GET /test HTTP/1.1\r\n" + name + b":data\r\n\r\n"

    match = f"400, message:\n  Got more than 8190 bytes \\({size}\\) when reading"
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        parser.feed_data(text)


def test_max_header_field_size_under_limit(parser: HttpRequestParser) -> None:
    name = b"t" * 8190
    text = b"GET /test HTTP/1.1\r\n" + name + b":data\r\n\r\n"

    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.method == "GET"
    assert msg.path == "/test"
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict({name.decode(): "data"})
    assert msg.raw_headers == ((name, b"data"),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL("/test")


@pytest.mark.parametrize("size", [40960, 8191])
def test_max_header_value_size(parser: HttpRequestParser, size: int) -> None:
    name = b"t" * size
    text = b"GET /test HTTP/1.1\r\ndata:" + name + b"\r\n\r\n"

    match = f"400, message:\n  Got more than 8190 bytes \\({size}\\) when reading"
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        parser.feed_data(text)


def test_max_header_value_size_under_limit(parser: HttpRequestParser) -> None:
    value = b"A" * 8190
    text = b"GET /test HTTP/1.1\r\ndata:" + value + b"\r\n\r\n"

    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]
    assert msg.method == "GET"
    assert msg.path == "/test"
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict({"data": value.decode()})
    assert msg.raw_headers == ((b"data", value),)
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL("/test")


@pytest.mark.parametrize("size", [40965, 8191])
def test_max_header_value_size_continuation(
    response: HttpResponseParser, size: int
) -> None:
    name = b"T" * (size - 5)
    text = b"HTTP/1.1 200 Ok\r\ndata: test\r\n " + name + b"\r\n\r\n"

    match = f"400, message:\n  Got more than 8190 bytes \\({size}\\) when reading"
    with pytest.raises(http_exceptions.LineTooLong, match=match):
        response.feed_data(text)


def test_max_header_value_size_continuation_under_limit(
    response: HttpResponseParser,
) -> None:
    value = b"A" * 8185
    text = b"HTTP/1.1 200 Ok\r\ndata: test\r\n " + value + b"\r\n\r\n"

    messages, upgrade, tail = response.feed_data(text)
    msg = messages[0][0]
    assert msg.code == 200
    assert msg.reason == "Ok"
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict({"data": "test " + value.decode()})
    assert msg.raw_headers == ((b"data", b"test " + value),)
    assert msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked


def test_http_request_parser(parser: HttpRequestParser) -> None:
    text = b"GET /path HTTP/1.1\r\n\r\n"
    messages, upgrade, tail = parser.feed_data(text)
    msg = messages[0][0]

    assert msg.method == "GET"
    assert msg.path == "/path"
    assert msg.version == (1, 1)
    assert msg.headers == CIMultiDict()
    assert msg.raw_headers == ()
    assert not msg.should_close
    assert msg.compression is None
    assert not msg.upgrade
    assert not msg.chunked
    assert msg.url == URL("/path")


def test_http_request_bad_status_line(parser: HttpRequestParser) -> None:
    text = b"getpath \r\n\r\n"
    with pytest.raises(http_exceptions.BadStatusLine) as exc_info:
        parser.feed_data(text)
    # Check for accidentally escaped message.
    assert r"\n" not in exc_info.value.message


_num: Dict[bytes, str] = {
    # dangerous: accepted by Python int()
    # unicodedata.category("\U0001D7D9") == 'Nd'
    "\N{MATHEMATICAL DOUBLE-STRUCK DIGIT ONE}".encode(): "utf8digit",
    # only added for interop tests, refused by Python int()
    # unicodedata.category("\U000000B9") == 'No'
    "\N{SUPERSCRIPT ONE}".encode(): "utf8number",
    "\N{SUPERSCRIPT ONE}".encode("latin-1"): "latin1number",
}


@pytest.mark.parametrize("nonascii_digit", _num.keys(), ids=_num.values())
def test_http_request_bad_status_line_number(
    parser: HttpRequestParser, nonascii_digit: bytes
) -> None:
    text = b"GET /digit