import asyncio
import re
from contextlib import suppress
from typing import Any, Dict, Iterable, List, Optional, Type, Union
from unittest.mock import Mock
from urllib.parse import quote
import pytest
from multidict import CIMultiDict
from yarl import URL
import aiohttp
from aiohttp import http_exceptions, streams
from aiohttp.base_protocol import BaseProtocol
from aiohttp.http_parser import (DeflateBuffer, HttpParser, HttpPayloadParser,
                                HttpRequestParser, HttpRequestParserPy,
                                HttpResponseParser, HttpResponseParserPy)
from aiohttp.http_writer import HttpVersion

@pytest.fixture
def protocol() -> Mock[BaseProtocol]:
    ...

@pytest.fixture
def parser(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol], request: pytest.FixtureRequest) -> Union[HttpRequestParserPy, HttpRequestParserC]:
    ...

@pytest.fixture
def request_cls(request: pytest.FixtureRequest) -> Type[Union[HttpRequestParserPy, HttpRequestParserC]]:
    ...

@pytest.fixture
def response(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol], request: pytest.FixtureRequest) -> Union[HttpResponseParserPy, HttpResponseParserC]:
    ...

@pytest.fixture
def response_cls(request: pytest.FixtureRequest) -> Type[Union[HttpResponseParserPy, HttpResponseParserC]]:
    ...

def test_c_parser_loaded() -> None:
    ...

def test_parse_headers(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_reject_obsolete_line_folding(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.skipif(NO_EXTENSIONS, reason='Only tests C parser.')
def test_invalid_character(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol], request: pytest.FixtureRequest) -> None:
    ...

@pytest.mark.skipif(NO_EXTENSIONS, reason='Only tests C parser.')
def test_invalid_linebreak(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol], request: pytest.FixtureRequest) -> None:
    ...

def test_cve_2023_37276(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.parametrize('rfc9110_5_6_2_token_delim', '"(),/:;<=>?@[\\]{}')
def test_bad_header_name(parser: Union[HttpRequestParserPy, HttpRequestParserC], rfc9110_5_6_2_token_delim: str) -> None:
    ...

@pytest.mark.parametrize('hdr', ('Content-Length: -5', 'Content-Length: +256', 'Content-Length: ¹', 'Content-Length: 𝟙', 'Foo: abc\rdef', 'Bar: abc\ndef', 'Baz: abc\x00def', 'Foo : bar', 'Foo\t: bar', 'ÿoo: bar'))
def test_bad_headers(parser: Union[HttpRequestParserPy, HttpRequestParserC], hdr: str) -> None:
    ...

def test_unpaired_surrogate_in_header_py(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol]) -> None:
    ...

def test_content_length_transfer_encoding(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

async def test_bad_chunked_py(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol]) -> None:
    ...

@pytest.mark.skipif('HttpRequestParserC' not in dir(aiohttp.http_parser), reason='C based HTTP parser not available')
def test_bad_chunked_c(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol]) -> None:
    ...

def test_whitespace_before_header(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_parse_headers_longline(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.fixture
def xfail_c_parser_status(request: pytest.FixtureRequest) -> None:
    ...

@pytest.mark.usefixtures('xfail_c_parser_status')
def test_parse_unusual_request_line(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_parse(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

async def test_parse_body(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

async def test_parse_body_with_CRLF(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_parse_delayed(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_headers_multi_feed(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_headers_split_field(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_parse_headers_multi(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_conn_default_1_0(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_conn_default_1_1(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_conn_close(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_conn_close_1_0(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_conn_keep_alive_1_0(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_conn_keep_alive_1_1(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_conn_other_1_0(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_conn_other_1_1(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_request_chunked(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_request_te_chunked_with_content_length(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_request_te_chunked123(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

async def test_request_te_last_chunked(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_request_te_first_chunked(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_conn_upgrade(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_bad_upgrade(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_compression_empty(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_compression_deflate(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_compression_gzip(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.skipif(brotli is None, reason='brotli is not installed')
def test_compression_brotli(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_compression_unknown(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_url_connect(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_headers_connect(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_url_absolute(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_headers_old_websocket_key1(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_headers_content_length_err_1(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_headers_content_length_err_2(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.parametrize('pad2', [b'', b'foo'], ids=['name-empty', 'with-name'])
@pytest.mark.parametrize('pad1', [b'', b'foo'], ids=['pre-' + n for n in _pad.values()])
def test_invalid_header_spacing(parser: Union[HttpRequestParserPy, HttpRequestParserC], pad1: bytes, pad2: bytes, hdr: bytes) -> None:
    ...

def test_empty_header_name(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_invalid_header(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_invalid_name(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.parametrize('size', [40960, 8191])
def test_max_header_field_size(parser: Union[HttpRequestParserPy, HttpRequestParserC], size: int) -> None:
    ...

def test_max_header_field_size_under_limit(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.parametrize('size', [40960, 8191])
def test_max_header_value_size(parser: Union[HttpRequestParserPy, HttpRequestParserC], size: int) -> None:
    ...

def test_max_header_value_size_under_limit(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.parametrize('size', [40965, 8191])
def test_max_header_value_size_continuation(response: Union[HttpResponseParserPy, HttpResponseParserC], size: int) -> None:
    ...

def test_max_header_value_size_continuation_under_limit(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_request_parser(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_bad_status_line(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.parametrize('nonascii_digit', [b'\\x00', b'\\x01'], ids=['utf8digit', 'utf8number'])
def test_http_request_bad_status_line_number(parser: Union[HttpRequestParserPy, HttpRequestParserC], nonascii_digit: bytes) -> None:
    ...

def test_http_request_bad_status_line_separator(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_bad_status_line_whitespace(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_message_after_close(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_upgrade(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

async def test_http_request_upgrade_unknown(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.fixture
def xfail_c_parser_url(request: pytest.FixtureRequest) -> None:
    ...

@pytest.mark.usefixtures('xfail_c_parser_url')
def test_http_request_parser_utf8_request_line(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_parser_utf8(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_parser_non_utf8(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_parser_two_slashes(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.parametrize('rfc9110_5_6_2_token_delim', [bytes([i]) for i in b'"(),/:;<=>?@[\\]{}'])
def test_http_request_parser_bad_method(parser: Union[HttpRequestParserPy, HttpRequestParserC], rfc9110_5_6_2_token_delim: bytes) -> None:
    ...

def test_http_request_parser_bad_version(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_parser_bad_version_number(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_parser_bad_ascii_uri(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_parser_bad_nonascii_uri(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.parametrize('size', [40965, 8191])
def test_http_request_max_status_line(parser: Union[HttpRequestParserPy, HttpRequestParserC], size: int) -> None:
    ...

def test_http_request_max_status_line_under_limit(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_response_parser_utf8(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_response_parser_utf8_without_reason(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_response_parser_obs_line_folding(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

@pytest.mark.dev_mode
def test_http_response_parser_strict_obs_line_folding(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

@pytest.mark.parametrize('size', [40962, 8191])
def test_http_response_parser_bad_status_line_too_long(response: Union[HttpResponseParserPy, HttpResponseParserC], size: int) -> None:
    ...

def test_http_response_parser_status_line_under_limit(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_response_parser_bad_version(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_response_parser_bad_version_number(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_response_parser_no_reason(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_response_parser_lenient_headers(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

@pytest.mark.dev_mode
def test_http_response_parser_strict_headers(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_response_parser_bad_crlf(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

async def test_http_response_parser_bad_chunked_lax(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

@pytest.mark.dev_mode
async def test_http_response_parser_bad_chunked_strict_py(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol]) -> None:
    ...

@pytest.mark.dev_mode
@pytest.mark.skipif('HttpRequestParserC' not in dir(aiohttp.http_parser), reason='C based HTTP parser not available')
def test_http_response_parser_bad_chunked_strict_c(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol]) -> None:
    ...

async def test_http_response_parser_notchunked(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

async def test_http_response_parser_last_chunked(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_response_parser_bad(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_response_parser_code_under_100(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_response_parser_code_above_999(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_http_response_parser_code_not_int(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

@pytest.mark.parametrize('nonascii_digit', [b'\\x00', b'\\x01'], ids=['utf8digit', 'utf8number'])
def test_http_response_parser_code_not_ascii(response: Union[HttpResponseParserPy, HttpResponseParserC], nonascii_digit: bytes) -> None:
    ...

def test_http_request_chunked_payload(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_chunked_payload_and_next_message(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_http_request_chunked_payload_chunks(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_parse_chunked_payload_chunk_extension(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_parse_no_length_or_te_on_post(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol], request_cls: Type[Union[HttpRequestParserPy, HttpRequestParserC]]) -> None:
    ...

def test_parse_payload_response_without_body(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol], response_cls: Type[Union[HttpResponseParserPy, HttpResponseParserC]]) -> None:
    ...

def test_parse_length_payload(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_parse_no_length_payload(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_parse_content_length_payload_multiple(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

def test_parse_content_length_than_chunked_payload(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

@pytest.mark.parametrize('code', (204, 304, 101, 102))
def test_parse_chunked_payload_empty_body_than_another_chunked(response: Union[HttpResponseParserPy, HttpResponseParserC], code: int) -> None:
    ...

async def test_parse_chunked_payload_split_chunks(response: Union[HttpResponseParserPy, HttpResponseParserC]) -> None:
    ...

@pytest.mark.skipif(NO_EXTENSIONS, reason='Only tests C parser.')
async def test_parse_chunked_payload_with_lf_in_extensions_c_parser(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol]) -> None:
    ...

async def test_parse_chunked_payload_with_lf_in_extensions_py_parser(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol]) -> None:
    ...

def test_partial_url(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.parametrize(('uri', 'path', 'query', 'fragment'), [('/path%23frag', '/path#frag', {}, ''), ('/path%2523frag', '/path%23frag', {}, ''), ('/path?key=value%23frag', '/path', {'key': 'value#frag'}, ''), ('/path?key=value%2523frag', '/path', {'key': 'value%23frag'}, ''), ('/path#frag%20', '/path', {}, 'frag '), ('/path#frag%2520', '/path', {}, 'frag%20')])
def test_parse_uri_percent_encoded(parser: Union[HttpRequestParserPy, HttpRequestParserC], uri: str, path: str, query: Dict[str, str], fragment: str) -> None:
    ...

def test_parse_uri_utf8(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

def test_parse_uri_utf8_percent_encoded(parser: Union[HttpRequestParserPy, HttpRequestParserC]) -> None:
    ...

@pytest.mark.skipif('HttpRequestParserC' not in dir(aiohttp.http_parser), reason='C based HTTP parser not available')
def test_parse_bad_method_for_c_parser_raises(loop: asyncio.AbstractEventLoop, protocol: Mock[BaseProtocol]) -> None:
    ...

class TestParsePayload:
    async def test_parse_eof_payload(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_parse_length_payload_eof(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_parse_chunked_payload_size_error(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_parse_chunked_payload_split_end(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_parse_chunked_payload_split_end2(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_parse_chunked_payload_split_end_trailers(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_parse_chunked_payload_split_end_trailers2(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_parse_chunked_payload_split_end_trailers3(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_parse_chunked_payload_split_end_trailers4(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_http_payload_parser_length(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_http_payload_parser_deflate(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_http_payload_parser_deflate_no_hdrs(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_http_payload_parser_deflate_light(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_http_payload_parser_deflate_split(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_http_payload_parser_deflate_split_err(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_http_payload_parser_length_zero(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    @pytest.mark.skipif(brotli is None, reason='brotli is not installed')
    async def test_http_payload_brotli(self, protocol: Mock[BaseProtocol]) -> None:
        ...

class TestDeflateBuffer:
    async def test_feed_data(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_feed_data_err(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_feed_eof(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_feed_eof_err_deflate(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_feed_eof_no_err_gzip(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_feed_eof_no_err_brotli(self, protocol: Mock[BaseProtocol]) -> None:
        ...

    async def test_empty_body(self, protocol: Mock[BaseProtocol]) -> None:
        ...