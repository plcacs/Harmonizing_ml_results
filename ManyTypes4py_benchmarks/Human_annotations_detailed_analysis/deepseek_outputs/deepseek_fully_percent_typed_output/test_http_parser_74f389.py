# Tests for aiohttp/protocol.py

import asyncio
import re
from contextlib import suppress
from typing import Any, Dict, Iterable, List, Type, Optional, Tuple, Union, cast
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
   