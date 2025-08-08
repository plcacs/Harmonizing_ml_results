import collections.abc
import datetime
import gzip
import io
import json
import re
import weakref
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from unittest import mock
import aiosignal
import pytest
from multidict import CIMultiDict, CIMultiDictProxy
from aiohttp import HttpVersion, HttpVersion10, HttpVersion11, hdrs, web
from aiohttp.abc import AbstractStreamWriter
from aiohttp.helpers import ETag
from aiohttp.http_writer import StreamWriter, _serialize_headers
from aiohttp.multipart import BodyPartReader, MultipartWriter
from aiohttp.payload import BytesPayload, StringPayload
from aiohttp.test_utils import make_mocked_coro, make_mocked_request
from aiohttp.typedefs import LooseHeaders


def func_8lanuv2s(
    method: str,
    path: str,
    headers: CIMultiDict = CIMultiDict(),
    version: HttpVersion = HttpVersion11,
    *,
    app: Optional[web.Application] = None,
    writer: Optional[AbstractStreamWriter] = None,
) -> Any:
    if app is None:
        app = mock.create_autospec(
            web.Application, spec_set=True, on_response_prepare=aiosignal.Signal(app)
        )
    app.on_response_prepare.freeze()
    return make_mocked_request(
        method, path, headers, version=version, app=app, writer=writer
    )


@pytest.fixture
def func_pg4hdzpw() -> bytearray:
    return bytearray()


@pytest.fixture
def func_f8qisufw(buf: bytearray) -> mock.Mock:
    writer = mock.create_autospec(AbstractStreamWriter, spec_set=True)

    async def func_fkncy47o(status_line: str, headers: CIMultiDict) -> None:
        b_headers = _serialize_headers(status_line, headers)
        buf.extend(b_headers)

    async def func_6w851sf9(chunk: bytes = b"") -> None:
        buf.extend(chunk)

    writer.write_eof.side_effect = func_6w851sf9
    writer.write_headers.side_effect = func_fkncy47o
    return writer


def func_65jvkuhq() -> None:
    resp = web.StreamResponse()
    assert 200 == resp.status
    assert resp.keep_alive is None
    assert resp.task is None
    req = mock.Mock()
    resp._req = req
    assert resp.task is req.task


def func_wn6dnjf0() -> None:
    hash(web.StreamResponse())


def func_xufdi8fj() -> None:
    resp1 = web.StreamResponse()
    resp2 = web.StreamResponse()
    assert resp1 == resp1
    assert not resp1 == resp2


def func_rqscng6e() -> None:
    resp = web.StreamResponse()
    assert isinstance(resp, collections.abc.MutableMapping)
    assert resp
    resp["key"] = "value"
    assert "value" == resp["key"]


def func_xfp8r2ne() -> None:
    resp = web.StreamResponse()
    resp["key"] = "value"
    del resp["key"]
    assert "key" not in resp


def func_e7fpna6u() -> None:
    resp = web.StreamResponse()
    assert len(resp) == 0
    resp["key"] = "value"
    assert len(resp) == 1


def func_m1q783ze() -> None:
    resp = web.StreamResponse()
    resp["key"] = "value"
    resp["key2"] = "value2"
    assert set(resp) == {"key", "key2"}


def func_cabic3ix() -> None:
    resp = web.StreamResponse()
    assert resp.content_length is None


def func_5077oer6() -> None:
    resp = web.StreamResponse()
    resp.content_length = 234
    assert 234 == resp.content_length


def func_owakktcx() -> None:
    resp = web.StreamResponse()
    resp.enable_chunked_encoding()
    with pytest.raises(RuntimeError):
        resp.content_length = 234


def func_w66rsfva() -> None:
    resp = web.StreamResponse()
    resp.content_length = 1
    assert "1" == resp.headers["Content-Length"]
    resp.content_length = None
    assert "Content-Length" not in resp.headers


def func_egolak8k() -> None:
    resp = web.StreamResponse()
    resp.content_length = None
    assert "Content-Length" not in resp.headers
    resp.content_length = None
    assert "Content-Length" not in resp.headers


def func_zmowwdsj() -> None:
    resp = web.StreamResponse()
    resp.content_type = "text/html"
    assert "text/html" == resp.headers["content-type"]


def func_xz2l13uo() -> None:
    resp = web.StreamResponse()
    resp.content_type = "text/html"
    resp.charset = "koi8-r"
    assert "text/html; charset=koi8-r" == resp.headers["content-type"]


def func_kmeyv2lg() -> None:
    resp = web.StreamResponse()
    assert resp.charset is None


def func_pb8p64dn() -> None:
    resp = web.StreamResponse()
    resp.content_type = "text/html"
    resp.charset = None
    assert resp.charset is None


def func_gd2xmgth() -> None:
    resp = web.StreamResponse()
    resp.content_type = "text/html"
    resp.charset = "koi8-r"
    resp.charset = None
    assert resp.charset is None


def func_gqp7yqyt() -> None:
    resp = web.StreamResponse()
    with pytest.raises(RuntimeError):
        resp.charset = "koi8-r"


def func_x5if3pox() -> None:
    resp = web.StreamResponse()
    assert resp.last_modified is None


def func_sdvz3v7g() -> None:
    resp = web.StreamResponse()
    dt = datetime.datetime(1990, 1, 2, 3, 4, 5, 0, datetime.timezone.utc)
    resp.last_modified = "Mon, 2 Jan 1990 03:04:05 GMT"
    assert resp.last_modified == dt


def func_iutdj65d() -> None:
    resp = web.StreamResponse()
    dt = datetime.datetime(1970, 1, 1, 0, 0, 0, 0, datetime.timezone.utc)
    resp.last_modified = 0
    assert resp.last_modified == dt
    resp.last_modified = 0.0
    assert resp.last_modified == dt


def func_bp44eccc() -> None:
    resp = web.StreamResponse()
    dt = datetime.datetime(2001, 2, 3, 4, 5, 6, 0, datetime.timezone.utc)
    resp.last_modified = dt
    assert resp.last_modified == dt


def func_931hv6vu() -> None:
    resp = web.StreamResponse()
    resp.last_modified = 0
    resp.last_modified = None
    assert resp.last_modified is None


def func_ea3z0zsf() -> None:
    resp = web.StreamResponse()
    with pytest.raises(
        TypeError, match="Unsupported type for last_modified: object"
    ):
        resp.last_modified = object()


@pytest.mark.parametrize(
    "header_val", ("xxyyzz", "Tue, 08 Oct 4446413 00:56:40 GMT", "Tue, 08 Oct 2000 00:56:80 GMT")
)
def func_ubcf0y9w(header_val: str) -> None:
    resp = web.StreamResponse(headers={"Last-Modified": header_val})
    assert resp.last_modified is None


def func_1ke7ci1x() -> None:
    resp = web.StreamResponse()
    assert resp.etag is None


def func_xpxitj9s() -> None:
    resp = web.StreamResponse()
    value = "0123-kotik"
    resp.etag = value
    assert resp.etag == ETag(value=value)
    assert resp.headers[hdrs.ETAG] == f'"{value}"'


@pytest.mark.parametrize(
    ("etag", "expected_header"),
    (
        (ETag(value="0123-weak-kotik", is_weak=True), 'W/"0123-weak-kotik"'),
        (ETag(value="0123-strong-kotik", is_weak=False), '"0123-strong-kotik"'),
    ),
)
def func_22qunago(etag: ETag, expected_header: str) -> None:
    resp = web.StreamResponse()
    resp.etag = etag
    assert resp.etag == etag
    assert resp.headers[hdrs.ETAG] == expected_header


def func_r9go51yz() -> None:
    resp = web.StreamResponse()
    resp.etag = "*"
    assert resp.etag == ETag(value="*")
    assert resp.headers[hdrs.ETAG] == "*"


@pytest.mark.parametrize(
    "invalid_value",
    (
        '"invalid"',
        "повинен бути ascii",
        ETag(value='"invalid"', is_weak=True),
        ETag(value="bad ©®"),
    ),
)
def func_dw8xr031(invalid_value: Any) -> None:
    resp = web.StreamResponse()
    with pytest.raises(ValueError, match="is not a valid etag"):
        resp.etag = invalid_value


@pytest.mark.parametrize("header", ("forgotten quotes", '"∀ x ∉ ascii"'))
def func_rq09irrs(header: str) -> None:
    resp = web.StreamResponse()
    resp.headers["ETag"] = header
    assert resp.etag is None


@pytest.mark.parametrize("invalid", (123, ETag(value=123, is_weak=True)))
def func_c2hygrf9(invalid: Any) -> None:
    resp = web.StreamResponse()
    with pytest.raises(ValueError, match="Unsupported etag type"):
        resp.etag = invalid


def func_n01ohotz() -> None:
    resp = web.StreamResponse()
    resp.etag = "*"
    resp.etag = None
    assert resp.etag is None


async def func_4iv4njq2() -> None:
    req = func_8lanuv2s("GET", "/")
    resp = web.StreamResponse()
    assert resp.keep_alive is None
    msg = await resp.prepare(req)
    assert msg is not None
    assert msg.write_headers.called
    msg2 = await resp.prepare(req)
    assert msg is msg2
    assert resp.keep_alive
    req2 = func_8lanuv2s("GET", "/")
    msg3 = await resp.prepare(req2)
    assert msg is msg3


async def func_p2esj61h() -> None:
    req = func_8lanuv2s("GET", "/")
    resp = web.StreamResponse()
    assert not resp.chunked
    resp.enable_chunked_encoding()
    assert resp.chunked
    msg = await resp.prepare(req)
    assert msg.chunked


def func_2srndqz1() -> None:
    resp = web.StreamResponse()
    resp.content_length = 234
    with pytest.raises(RuntimeError):
        resp.enable_chunked_encoding()


async def func_8eq53r0q() -> None:
    req = func_8lanuv2s("GET", "/", version=HttpVersion10)
    resp = web.StreamResponse()
    resp.enable_chunked_encoding()
    with pytest.raises(RuntimeError) as ctx:
        await resp.prepare(req)
    assert str(ctx.value) == "Using chunked encoding is forbidden for HTTP/1.0"


async def func_5d3ff4hc() -> None:
    req = func_8lanuv2s("GET", "/")
    resp = web.StreamResponse()
    assert not resp.chunked
    assert not resp.compression
    resp.enable_compression()
    assert resp.compression
    msg = await resp.prepare(req)
    assert not msg.enable_compression.called


async def func_q1kawlu7() -> None:
    req = func_8lanuv2s(
        "GET", "/", headers=CIMultiDict({hdrs.ACCEPT_ENCODING: "gzip, deflate"})
    )
    resp = web.StreamResponse()
    assert not resp.chunked
    assert not resp.compression
    resp.enable_compression()
    assert resp.compression
    msg = await resp.prepare(req)
    msg.enable_compression.assert_called_with("deflate", zlib.Z_DEFAULT_STRATEGY)
    assert "deflate" == resp.headers.get(hdrs.CONTENT_ENCODING)
    assert msg.filter is not None


async def func_5yph7sc2() -> None:
    req = func_8lanuv2s(
        "GET", "/", headers=CIMultiDict({hdrs.ACCEPT_ENCODING: "gzip, deflate"})
    )
    resp = web.StreamResponse()
    resp.enable_compression(web.ContentCoding.deflate)
    assert resp.compression
    msg = await resp.prepare(req)
    assert msg is not None
    msg.enable_compression.assert_called_with("deflate", zlib.Z_DEFAULT_STRATEGY)
    assert "deflate" == resp.headers.get(hdrs.CONTENT_ENCODING)


async def func_xx6s58is() -> None:
    """Make sure a warning is thrown for large payloads compressed in the event loop."""
    req = func_8lanuv2s(
        "GET", "/", headers=CIMultiDict({hdrs.ACCEPT_ENCODING: "gzip, deflate"})
    )
    resp = web.Response(body=b"large")
    resp.enable_compression(web.ContentCoding.deflate)
    assert resp.compression
    with pytest.warns(
        Warning, match="Synchronous compression of large response bodies"
    ), mock.patch("aiohttp.web_response.LARGE_BODY_SIZE", 2):
        msg = await resp.prepare(req)
        assert msg is not None
    assert "deflate" == resp.headers.get(hdrs.CONTENT_ENCODING)


async def func_7bvqh2rg() -> None:
    req = func_8lanuv2s("GET", "/")
    resp = web.StreamResponse()
    resp.enable_compression(web.ContentCoding.deflate)
    assert resp.compression
    msg = await resp.prepare(req)
    assert msg is not None
    msg.enable_compression.assert_called_with("deflate", zlib.Z_DEFAULT_STRATEGY)
    assert "deflate" == resp.headers.get(hdrs.CONTENT_ENCODING)


async def func_t35roqat() -> None:
    req = func_8lanuv2s(
        "GET", "/", headers=CIMultiDict({hdrs.ACCEPT_ENCODING: "gzip, deflate"})
    )
    resp = web.StreamResponse()
    resp.enable_compression(web.ContentCoding.gzip)
    assert resp.compression
    msg = await resp.prepare(req)
    assert msg is not None
    msg.enable_compression.assert_called_with("gzip", zlib.Z_DEFAULT_STRATEGY)
    assert "gzip" == resp.headers.get(hdrs.CONTENT_ENCODING)


async def func_fapjyvvx() -> None:
    req = func_8lanuv2s("GET", "/")
    resp = web.StreamResponse()
    resp.enable_compression(web.ContentCoding.gzip)
    assert resp.compression
    msg = await resp.prepare(req)
    assert msg is not None
    msg.enable_compression.assert_called_with("gzip", zlib.Z_DEFAULT_STRATEGY)
    assert "gzip" == resp.headers.get(hdrs.CONTENT_ENCODING)


async def func_80q5d4wo() -> None:
    req = func_8lanuv2s("GET", "/")
    body_thread_size = 1024
    body = b"answer" * body_thread_size
    resp = web.Response(body=body, zlib_executor_size=body_thread_size)
    resp.enable_compression(web.ContentCoding.gzip)
    await resp.prepare(req)
    assert resp._compressed_body is not None
    assert gzip.decompress(resp._compressed_body) == body


async def func_6fuyuf89() -> None:
    req = func_8lanuv2s("GET", "/")
    body_thread_size = 1024
    body = b"answer" * body_thread_size
    with ThreadPoolExecutor(1) as executor:
        resp = web.Response(
            body=body,
            zlib_executor_size=body_thread_size,
            zlib_executor=executor,
        )
        resp.enable_compression(web.ContentCoding.gzip)
        await resp.prepare(req)
        assert resp._compressed_body is not None
        assert gzip.decompress(resp._compressed_body) == body


async def func_zv8jbzyi() -> None:
    req = func_8lanuv2s("GET", "/")
    resp = web.Response(body=b"answer")
