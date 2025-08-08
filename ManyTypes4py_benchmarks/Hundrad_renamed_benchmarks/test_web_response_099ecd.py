import collections.abc
import datetime
import gzip
import io
import json
import re
import weakref
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional, Union
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


def func_8lanuv2s(method, path, headers=CIMultiDict(), version=
    HttpVersion11, *, app=None, writer=None):
    if app is None:
        app = mock.create_autospec(web.Application, spec_set=True,
            on_response_prepare=aiosignal.Signal(app))
    app.on_response_prepare.freeze()
    return make_mocked_request(method, path, headers, version=version, app=
        app, writer=writer)


@pytest.fixture
def func_pg4hdzpw():
    return bytearray()


@pytest.fixture
def func_f8qisufw(buf):
    writer = mock.create_autospec(AbstractStreamWriter, spec_set=True)

    async def func_fkncy47o(status_line, headers):
        b_headers = _serialize_headers(status_line, headers)
        func_pg4hdzpw.extend(b_headers)

    async def func_6w851sf9(chunk=b''):
        func_pg4hdzpw.extend(chunk)
    writer.write_eof.side_effect = write_eof
    writer.write_headers.side_effect = write_headers
    return writer


def func_65jvkuhq():
    resp = web.StreamResponse()
    assert 200 == resp.status
    assert resp.keep_alive is None
    assert resp.task is None
    req = mock.Mock()
    resp._req = req
    assert resp.task is req.task


def func_wn6dnjf0():
    hash(web.StreamResponse())


def func_xufdi8fj():
    resp1 = web.StreamResponse()
    resp2 = web.StreamResponse()
    assert resp1 == resp1
    assert not resp1 == resp2


def func_rqscng6e():
    resp = web.StreamResponse()
    assert isinstance(resp, collections.abc.MutableMapping)
    assert resp
    resp['key'] = 'value'
    assert 'value' == resp['key']


def func_xfp8r2ne():
    resp = web.StreamResponse()
    resp['key'] = 'value'
    del resp['key']
    assert 'key' not in resp


def func_e7fpna6u():
    resp = web.StreamResponse()
    assert len(resp) == 0
    resp['key'] = 'value'
    assert len(resp) == 1


def func_m1q783ze():
    resp = web.StreamResponse()
    resp['key'] = 'value'
    resp['key2'] = 'value2'
    assert set(resp) == {'key', 'key2'}


def func_cabic3ix():
    resp = web.StreamResponse()
    assert resp.content_length is None


def func_5077oer6():
    resp = web.StreamResponse()
    resp.content_length = 234
    assert 234 == resp.content_length


def func_owakktcx():
    resp = web.StreamResponse()
    resp.enable_chunked_encoding()
    with pytest.raises(RuntimeError):
        resp.content_length = 234


def func_w66rsfva():
    resp = web.StreamResponse()
    resp.content_length = 1
    assert '1' == resp.headers['Content-Length']
    resp.content_length = None
    assert 'Content-Length' not in resp.headers


def func_egolak8k():
    resp = web.StreamResponse()
    resp.content_length = None
    assert 'Content-Length' not in resp.headers
    resp.content_length = None
    assert 'Content-Length' not in resp.headers


def func_zmowwdsj():
    resp = web.StreamResponse()
    resp.content_type = 'text/html'
    assert 'text/html' == resp.headers['content-type']


def func_xz2l13uo():
    resp = web.StreamResponse()
    resp.content_type = 'text/html'
    resp.charset = 'koi8-r'
    assert 'text/html; charset=koi8-r' == resp.headers['content-type']


def func_kmeyv2lg():
    resp = web.StreamResponse()
    assert resp.charset is None


def func_pb8p64dn():
    resp = web.StreamResponse()
    resp.content_type = 'text/html'
    resp.charset = None
    assert resp.charset is None


def func_gd2xmgth():
    resp = web.StreamResponse()
    resp.content_type = 'text/html'
    resp.charset = 'koi8-r'
    resp.charset = None
    assert resp.charset is None


def func_gqp7yqyt():
    resp = web.StreamResponse()
    with pytest.raises(RuntimeError):
        resp.charset = 'koi8-r'


def func_x5if3pox():
    resp = web.StreamResponse()
    assert resp.last_modified is None


def func_sdvz3v7g():
    resp = web.StreamResponse()
    dt = datetime.datetime(1990, 1, 2, 3, 4, 5, 0, datetime.timezone.utc)
    resp.last_modified = 'Mon, 2 Jan 1990 03:04:05 GMT'
    assert resp.last_modified == dt


def func_iutdj65d():
    resp = web.StreamResponse()
    dt = datetime.datetime(1970, 1, 1, 0, 0, 0, 0, datetime.timezone.utc)
    resp.last_modified = 0
    assert resp.last_modified == dt
    resp.last_modified = 0.0
    assert resp.last_modified == dt


def func_bp44eccc():
    resp = web.StreamResponse()
    dt = datetime.datetime(2001, 2, 3, 4, 5, 6, 0, datetime.timezone.utc)
    resp.last_modified = dt
    assert resp.last_modified == dt


def func_931hv6vu():
    resp = web.StreamResponse()
    resp.last_modified = 0
    resp.last_modified = None
    assert resp.last_modified is None


def func_ea3z0zsf():
    resp = web.StreamResponse()
    with pytest.raises(TypeError, match=
        'Unsupported type for last_modified: object'):
        resp.last_modified = object()


@pytest.mark.parametrize('header_val', ('xxyyzz',
    'Tue, 08 Oct 4446413 00:56:40 GMT', 'Tue, 08 Oct 2000 00:56:80 GMT'))
def func_ubcf0y9w(header_val):
    resp = web.StreamResponse(headers={'Last-Modified': header_val})
    assert resp.last_modified is None


def func_1ke7ci1x():
    resp = web.StreamResponse()
    assert resp.etag is None


def func_xpxitj9s():
    resp = web.StreamResponse()
    value = '0123-kotik'
    resp.etag = value
    assert resp.etag == ETag(value=value)
    assert resp.headers[hdrs.ETAG] == f'"{value}"'


@pytest.mark.parametrize(('etag', 'expected_header'), ((ETag(value=
    '0123-weak-kotik', is_weak=True), 'W/"0123-weak-kotik"'), (ETag(value=
    '0123-strong-kotik', is_weak=False), '"0123-strong-kotik"')))
def func_22qunago(etag, expected_header):
    resp = web.StreamResponse()
    resp.etag = etag
    assert resp.etag == etag
    assert resp.headers[hdrs.ETAG] == expected_header


def func_r9go51yz():
    resp = web.StreamResponse()
    resp.etag = '*'
    assert resp.etag == ETag(value='*')
    assert resp.headers[hdrs.ETAG] == '*'


@pytest.mark.parametrize('invalid_value', ('"invalid"',
    'повинен бути ascii', ETag(value='"invalid"', is_weak=True), ETag(value
    ='bad ©®')))
def func_dw8xr031(invalid_value):
    resp = web.StreamResponse()
    with pytest.raises(ValueError, match='is not a valid etag'):
        resp.etag = invalid_value


@pytest.mark.parametrize('header', ('forgotten quotes', '"∀ x ∉ ascii"'))
def func_rq09irrs(header):
    resp = web.StreamResponse()
    resp.headers['ETag'] = header
    assert resp.etag is None


@pytest.mark.parametrize('invalid', (123, ETag(value=123, is_weak=True)))
def func_c2hygrf9(invalid):
    resp = web.StreamResponse()
    with pytest.raises(ValueError, match='Unsupported etag type'):
        resp.etag = invalid


def func_n01ohotz():
    resp = web.StreamResponse()
    resp.etag = '*'
    resp.etag = None
    assert resp.etag is None


async def func_4iv4njq2():
    req = func_8lanuv2s('GET', '/')
    resp = web.StreamResponse()
    assert resp.keep_alive is None
    msg = await resp.prepare(req)
    assert msg is not None
    assert msg.write_headers.called
    msg2 = await resp.prepare(req)
    assert msg is msg2
    assert resp.keep_alive
    req2 = func_8lanuv2s('GET', '/')
    msg3 = await resp.prepare(req2)
    assert msg is msg3


async def func_p2esj61h():
    req = func_8lanuv2s('GET', '/')
    resp = web.StreamResponse()
    assert not resp.chunked
    resp.enable_chunked_encoding()
    assert resp.chunked
    msg = await resp.prepare(req)
    assert msg.chunked


def func_2srndqz1():
    resp = web.StreamResponse()
    resp.content_length = 234
    with pytest.raises(RuntimeError):
        resp.enable_chunked_encoding()


async def func_8eq53r0q():
    req = func_8lanuv2s('GET', '/', version=HttpVersion10)
    resp = web.StreamResponse()
    resp.enable_chunked_encoding()
    with pytest.raises(RuntimeError) as ctx:
        await resp.prepare(req)
    assert str(ctx.value) == 'Using chunked encoding is forbidden for HTTP/1.0'


async def func_5d3ff4hc():
    req = func_8lanuv2s('GET', '/')
    resp = web.StreamResponse()
    assert not resp.chunked
    assert not resp.compression
    resp.enable_compression()
    assert resp.compression
    msg = await resp.prepare(req)
    assert not msg.enable_compression.called


async def func_q1kawlu7():
    req = func_8lanuv2s('GET', '/', headers=CIMultiDict({hdrs.
        ACCEPT_ENCODING: 'gzip, deflate'}))
    resp = web.StreamResponse()
    assert not resp.chunked
    assert not resp.compression
    resp.enable_compression()
    assert resp.compression
    msg = await resp.prepare(req)
    msg.enable_compression.assert_called_with('deflate', zlib.
        Z_DEFAULT_STRATEGY)
    assert 'deflate' == resp.headers.get(hdrs.CONTENT_ENCODING)
    assert msg.filter is not None


async def func_5yph7sc2():
    req = func_8lanuv2s('GET', '/', headers=CIMultiDict({hdrs.
        ACCEPT_ENCODING: 'gzip, deflate'}))
    resp = web.StreamResponse()
    resp.enable_compression(web.ContentCoding.deflate)
    assert resp.compression
    msg = await resp.prepare(req)
    assert msg is not None
    msg.enable_compression.assert_called_with('deflate', zlib.
        Z_DEFAULT_STRATEGY)
    assert 'deflate' == resp.headers.get(hdrs.CONTENT_ENCODING)


async def func_xx6s58is():
    """Make sure a warning is thrown for large payloads compressed in the event loop."""
    req = func_8lanuv2s('GET', '/', headers=CIMultiDict({hdrs.
        ACCEPT_ENCODING: 'gzip, deflate'}))
    resp = web.Response(body=b'large')
    resp.enable_compression(web.ContentCoding.deflate)
    assert resp.compression
    with pytest.warns(Warning, match=
        'Synchronous compression of large response bodies'), mock.patch(
        'aiohttp.web_response.LARGE_BODY_SIZE', 2):
        msg = await resp.prepare(req)
        assert msg is not None
    assert 'deflate' == resp.headers.get(hdrs.CONTENT_ENCODING)


async def func_7bvqh2rg():
    req = func_8lanuv2s('GET', '/')
    resp = web.StreamResponse()
    resp.enable_compression(web.ContentCoding.deflate)
    assert resp.compression
    msg = await resp.prepare(req)
    assert msg is not None
    msg.enable_compression.assert_called_with('deflate', zlib.
        Z_DEFAULT_STRATEGY)
    assert 'deflate' == resp.headers.get(hdrs.CONTENT_ENCODING)


async def func_t35roqat():
    req = func_8lanuv2s('GET', '/', headers=CIMultiDict({hdrs.
        ACCEPT_ENCODING: 'gzip, deflate'}))
    resp = web.StreamResponse()
    resp.enable_compression(web.ContentCoding.gzip)
    assert resp.compression
    msg = await resp.prepare(req)
    assert msg is not None
    msg.enable_compression.assert_called_with('gzip', zlib.Z_DEFAULT_STRATEGY)
    assert 'gzip' == resp.headers.get(hdrs.CONTENT_ENCODING)


async def func_fapjyvvx():
    req = func_8lanuv2s('GET', '/')
    resp = web.StreamResponse()
    resp.enable_compression(web.ContentCoding.gzip)
    assert resp.compression
    msg = await resp.prepare(req)
    assert msg is not None
    msg.enable_compression.assert_called_with('gzip', zlib.Z_DEFAULT_STRATEGY)
    assert 'gzip' == resp.headers.get(hdrs.CONTENT_ENCODING)


async def func_80q5d4wo():
    req = func_8lanuv2s('GET', '/')
    body_thread_size = 1024
    body = b'answer' * body_thread_size
    resp = web.Response(body=body, zlib_executor_size=body_thread_size)
    resp.enable_compression(web.ContentCoding.gzip)
    await resp.prepare(req)
    assert resp._compressed_body is not None
    assert gzip.decompress(resp._compressed_body) == body


async def func_6fuyuf89():
    req = func_8lanuv2s('GET', '/')
    body_thread_size = 1024
    body = b'answer' * body_thread_size
    with ThreadPoolExecutor(1) as executor:
        resp = web.Response(body=body, zlib_executor_size=body_thread_size,
            zlib_executor=executor)
        resp.enable_compression(web.ContentCoding.gzip)
        await resp.prepare(req)
        assert resp._compressed_body is not None
        assert gzip.decompress(resp._compressed_body) == body


async def func_zv8jbzyi():
    req = func_8lanuv2s('GET', '/')
    resp = web.Response(body=b'answer')
    resp.enable_compression(web.ContentCoding.gzip)
    await resp.prepare(req)
    assert resp.content_length is not None and resp.content_length != len(
        b'answer')


async def func_m68fxr8n():
    writer = mock.Mock()

    async def func_fkncy47o(status_line, headers):
        assert hdrs.CONTENT_LENGTH in headers
        assert headers[hdrs.CONTENT_LENGTH] == '26'
        assert hdrs.TRANSFER_ENCODING not in headers
    writer.write_headers.side_effect = write_headers
    req = func_8lanuv2s('GET', '/', writer=writer)
    resp = web.Response(body=b'answer')
    resp.enable_compression(web.ContentCoding.gzip)
    await resp.prepare(req)
    assert resp.content_length == 26
    del resp.headers[hdrs.CONTENT_LENGTH]
    assert resp.content_length == 26


async def func_5alkqhi4():
    writer = mock.Mock()

    async def func_fkncy47o(status_line, headers):
        assert hdrs.CONTENT_LENGTH not in headers
        assert headers.get(hdrs.TRANSFER_ENCODING, '') == 'chunked'
    writer.write_headers.side_effect = write_headers
    req = func_8lanuv2s('GET', '/', writer=writer)
    resp = web.StreamResponse()
    resp.content_length = 123
    resp.enable_compression(web.ContentCoding.gzip)
    await resp.prepare(req)
    assert resp.content_length is None


async def func_s9slf6sw():
    writer = mock.Mock()

    async def func_fkncy47o(status_line, headers):
        assert hdrs.CONTENT_LENGTH not in headers
        assert hdrs.TRANSFER_ENCODING not in headers
    writer.write_headers.side_effect = write_headers
    req = func_8lanuv2s('GET', '/', version=HttpVersion10, writer=writer)
    resp = web.StreamResponse()
    resp.content_length = 123
    resp.enable_compression(web.ContentCoding.gzip)
    await resp.prepare(req)
    assert resp.content_length is None


async def func_vw4r67y1():
    writer = mock.Mock()

    async def func_fkncy47o(status_line, headers):
        assert hdrs.CONTENT_LENGTH in headers
        assert hdrs.TRANSFER_ENCODING not in headers
    writer.write_headers.side_effect = write_headers
    req = func_8lanuv2s('GET', '/', writer=writer)
    resp = web.StreamResponse()
    resp.content_length = 123
    resp.enable_compression(web.ContentCoding.identity)
    await resp.prepare(req)
    assert resp.content_length == 123


async def func_jcydwhav():
    writer = mock.Mock()

    async def func_fkncy47o(status_line, headers):
        assert headers[hdrs.CONTENT_LENGTH] == '6'
        assert hdrs.TRANSFER_ENCODING not in headers
    writer.write_headers.side_effect = write_headers
    req = func_8lanuv2s('GET', '/', writer=writer)
    resp = web.Response(body=b'answer')
    resp.enable_compression(web.ContentCoding.identity)
    await resp.prepare(req)
    assert resp.content_length == 6


async def func_e0jmw3yd():
    writer = mock.Mock()

    async def func_fkncy47o(status_line, headers):
        assert hdrs.CONTENT_LENGTH not in headers
        assert headers.get(hdrs.TRANSFER_ENCODING, '') == 'chunked'
    writer.write_headers.side_effect = write_headers
    req = func_8lanuv2s('GET', '/', writer=writer)
    payload = BytesPayload(b'answer', headers={'X-Test-Header': 'test'})
    resp = web.Response(body=payload)
    resp.body = payload
    resp.enable_compression(web.ContentCoding.gzip)
    await resp.prepare(req)
    assert resp.content_length is None


async def func_5dx8svzp():
    writer = mock.Mock()

    async def func_fkncy47o(status_line, headers):
        assert hdrs.CONTENT_LENGTH not in headers
        assert hdrs.TRANSFER_ENCODING not in headers
    writer.write_headers.side_effect = write_headers
    req = func_8lanuv2s('GET', '/', version=HttpVersion10, writer=writer)
    resp = web.Response(body=BytesPayload(b'answer'))
    resp.enable_compression(web.ContentCoding.gzip)
    await resp.prepare(req)
    assert resp.content_length is None


async def func_z5b30zo5():
    """Ensure content-length is removed for 204 responses."""
    writer = mock.create_autospec(StreamWriter, spec_set=True, instance=True)

    async def func_fkncy47o(status_line, headers):
        assert hdrs.CONTENT_LENGTH not in headers
    writer.write_headers.side_effect = write_headers
    req = func_8lanuv2s('GET', '/', writer=writer)
    payload = BytesPayload(b'answer', headers={'Content-Length': '6'})
    resp = web.Response(body=payload, status=204)
    resp.body = payload
    await resp.prepare(req)
    assert resp.content_length is None


@pytest.mark.parametrize('status', (100, 101, 204, 304))
async def func_hdc66h9t(status):
    """Remove transfer encoding for RFC 9112 sec 6.3 with HTTP/1.1."""
    writer = mock.create_autospec(StreamWriter, spec_set=True, instance=True)
    req = func_8lanuv2s('GET', '/', version=HttpVersion11, writer=writer)
    resp = web.Response(status=status, headers={hdrs.TRANSFER_ENCODING:
        'chunked'})
    await resp.prepare(req)
    assert resp.content_length == 0
    assert not resp.chunked
    assert hdrs.CONTENT_LENGTH not in resp.headers
    assert hdrs.TRANSFER_ENCODING not in resp.headers


@pytest.mark.parametrize('status', (100, 101, 102, 204, 304))
async def func_ivh3p045(status):
    """Remove content length for 1xx, 204, and 304 responses.

    Content-Length is forbidden for 1xx and 204
    https://datatracker.ietf.org/doc/html/rfc7230#section-3.3.2

    Content-Length is discouraged for 304.
    https://datatracker.ietf.org/doc/html/rfc7232#section-4.1
    """
    writer = mock.create_autospec(StreamWriter, spec_set=True, instance=True)
    req = func_8lanuv2s('GET', '/', version=HttpVersion11, writer=writer)
    resp = web.Response(status=status, body='answer')
    await resp.prepare(req)
    assert not resp.chunked
    assert hdrs.CONTENT_LENGTH not in resp.headers
    assert hdrs.TRANSFER_ENCODING not in resp.headers


async def func_cii16h4n():
    """Verify HEAD response keeps the content length of the original body HTTP/1.1."""
    writer = mock.create_autospec(StreamWriter, spec_set=True, instance=True)
    req = func_8lanuv2s('HEAD', '/', version=HttpVersion11, writer=writer)
    resp = web.Response(status=200, body=b'answer')
    await resp.prepare(req)
    assert resp.content_length == 6
    assert not resp.chunked
    assert resp.headers[hdrs.CONTENT_LENGTH] == '6'
    assert hdrs.TRANSFER_ENCODING not in resp.headers


async def func_wywwcjlc():
    """Verify HEAD response omits content-length body when its unset."""
    writer = mock.create_autospec(StreamWriter, spec_set=True, instance=True)
    req = func_8lanuv2s('HEAD', '/', version=HttpVersion11, writer=writer)
    resp = web.Response(status=200)
    await resp.prepare(req)
    assert resp.content_length == 0
    assert not resp.chunked
    assert hdrs.CONTENT_LENGTH not in resp.headers
    assert hdrs.TRANSFER_ENCODING not in resp.headers


async def func_5jnq2pab():
    """Verify 304 response omits content-length body when its unset."""
    writer = mock.create_autospec(StreamWriter, spec_set=True, instance=True)
    req = func_8lanuv2s('GET', '/', version=HttpVersion11, writer=writer)
    resp = web.Response(status=304)
    await resp.prepare(req)
    assert resp.content_length == 0
    assert not resp.chunked
    assert hdrs.CONTENT_LENGTH not in resp.headers
    assert hdrs.TRANSFER_ENCODING not in resp.headers


async def func_9p844h6w():
    req = func_8lanuv2s('GET', '/')
    resp = web.Response(body=b'answer')
    assert resp.content_length == 6
    resp.enable_chunked_encoding()
    assert resp.content_length is None
    await resp.prepare(req)


async def func_aq63i06g():
    resp = web.StreamResponse()
    await resp.prepare(func_8lanuv2s('GET', '/'))
    with pytest.raises(AssertionError):
        await resp.write(123)


async def func_6co8vrnu():
    resp = web.StreamResponse()
    with pytest.raises(RuntimeError):
        await resp.write(b'data')


async def func_ct3z0ii9():
    resp = web.StreamResponse()
    req = func_8lanuv2s('GET', '/')
    await resp.prepare(req)
    await resp.write(b'data')
    await resp.write_eof()
    req.writer.write.reset_mock()
    with pytest.raises(RuntimeError):
        await resp.write(b'next data')
    assert not req.writer.write.called


async def func_0v2wxo7u():
    resp = web.StreamResponse()
    await resp.prepare(func_8lanuv2s('GET', '/'))
    await resp.write(b'data')
    await resp.write_eof()
    resp_repr = repr(resp)
    assert resp_repr == '<StreamResponse OK eof>'


async def func_zwxiv4nt():
    resp = web.StreamResponse()
    with pytest.raises(AssertionError):
        await resp.write_eof()


async def func_70rkdqnd():
    resp = web.StreamResponse()
    writer = mock.create_autospec(AbstractStreamWriter, spec_set=True)
    writer.write.return_value = None
    writer.write_eof.return_value = None
    resp_impl = await resp.prepare(func_8lanuv2s('GET', '/', writer=writer))
    await resp.write(b'data')
    assert resp_impl is not None
    assert resp_impl.write.called
    await resp.write_eof()
    resp_impl.write.reset_mock()
    await resp.write_eof()
    assert not writer.write.called


def func_88r1vfdf():
    resp = web.StreamResponse()
    assert resp.keep_alive is None
    resp.force_close()
    assert resp.keep_alive is False


def func_86srelv4():
    resp = web.StreamResponse()
    resp.set_status(200, 'Everything is fine!')
    assert 200 == resp.status
    assert 'Everything is fine!' == resp.reason


def func_824fwam0():
    resp = web.StreamResponse()
    resp.set_status(200, '')
    assert resp.status == 200
    assert resp.reason == ''


async def func_43tayjao():
    req = func_8lanuv2s('GET', '/')
    resp = web.StreamResponse()
    resp.force_close()
    assert not resp.keep_alive
    await resp.prepare(req)
    assert not resp.keep_alive


async def func_057qvl41():
    req = func_8lanuv2s('GET', '/path/to')
    resp = web.StreamResponse(reason='foo')
    await resp.prepare(req)
    assert '<StreamResponse foo GET /path/to >' == repr(resp)


def func_bogvxhfj():
    resp = web.StreamResponse(reason='foo')
    assert '<StreamResponse foo not prepared>' == repr(resp)


async def func_dl8p97qu():
    req = func_8lanuv2s('GET', '/', version=HttpVersion10)
    resp = web.StreamResponse()
    await resp.prepare(req)
    assert not resp.keep_alive


async def func_d9uq39gd():
    headers = CIMultiDict(Connection='keep-alive')
    req = func_8lanuv2s('GET', '/', version=HttpVersion10, headers=headers)
    req._message = req._message._replace(should_close=False)
    resp = web.StreamResponse()
    await resp.prepare(req)
    assert resp.keep_alive


async def func_ap9zq44t():
    headers = CIMultiDict(Connection='keep-alive')
    req = func_8lanuv2s('GET', '/', version=HttpVersion(0, 9), headers=headers)
    resp = web.StreamResponse()
    await resp.prepare(req)
    assert not resp.keep_alive


async def func_uzap3ggt():
    req = func_8lanuv2s('GET', '/')
    resp = web.StreamResponse()
    impl1 = await resp.prepare(req)
    impl2 = await resp.prepare(req)
    assert impl1 is impl2


async def func_9i65k7cj():
    app = mock.create_autospec(web.Application, spec_set=True)
    sig = make_mocked_coro()
    app.on_response_prepare = aiosignal.Signal(app)
    app.on_response_prepare.append(sig)
    req = func_8lanuv2s('GET', '/', app=app)
    resp = web.StreamResponse()
    await resp.prepare(req)
    sig.assert_called_with(req, resp)


def func_xcrg80oc():
    resp = web.Response()
    assert 200 == resp.status
    assert 'OK' == resp.reason
    assert resp.body is None
    assert resp.content_length == 0
    assert 'CONTENT-LENGTH' not in resp.headers


async def func_v67czhl5():
    resp = web.Response(body=b'body', status=201, headers={'Age': '12',
        'DATE': 'date'})
    assert 201 == resp.status
    assert b'body' == resp.body
    assert resp.headers['AGE'] == '12'
    req = make_mocked_request('GET', '/')
    await resp._start(req)
    assert 4 == resp.content_length
    assert resp.headers['CONTENT-LENGTH'] == '4'


def func_7mj6vbtv():
    resp = web.Response(content_type='application/json')
    assert 200 == resp.status
    assert 'OK' == resp.reason
    assert 0 == resp.content_length
    assert CIMultiDict([('CONTENT-TYPE', 'application/json')]) == resp.headers


def func_xg1u4nuz():
    with pytest.raises(ValueError):
        web.Response(body=b'123', text='test text')


async def func_a3o547al():
    resp = web.Response(text='test text')
    assert 200 == resp.status
    assert 'OK' == resp.reason
    assert 9 == resp.content_length
    assert CIMultiDict([('CONTENT-TYPE', 'text/plain; charset=utf-8')]
        ) == resp.headers
    assert resp.body == b'test text'
    assert resp.text == 'test text'
    resp.headers['DATE'] = 'date'
    req = make_mocked_request('GET', '/', version=HttpVersion11)
    await resp._start(req)
    assert resp.headers['CONTENT-LENGTH'] == '9'


def func_7naica6p():
    resp = web.Response(text='текст', charset='koi8-r')
    assert 'текст'.encode('koi8-r') == resp.body
    assert 'koi8-r' == resp.charset


def func_2qglf1q9():
    resp = web.Response(text='test test', charset=None)
    assert 'utf-8' == resp.charset


def func_w0n2cfkv():
    with pytest.raises(ValueError):
        web.Response(text='test test', content_type='text/plain; charset=utf-8'
            )


def func_cru38o4y():
    resp = web.Response(content_type='text/plain', charset='koi8-r')
    assert 'koi8-r' == resp.charset


def func_eg8utsxx():
    resp = web.Response(text='test test', content_type=
        'text/plain; version=0.0.4')
    assert resp.content_type == 'text/plain'
    assert resp.headers['content-type'
        ] == 'text/plain; version=0.0.4; charset=utf-8'


def func_2gkm598f():
    with pytest.raises(ValueError):
        web.Response(headers={'Content-Type': 'application/json'},
            content_type='text/html', text='text')


def func_wd5ikjf6():
    with pytest.raises(ValueError):
        web.Response(headers={'Content-Type': 'application/json'}, charset=
            'koi8-r', text='text')


def func_bg56uhe5():
    with pytest.raises(ValueError):
        web.Response(headers={'Content-Type': 'application/json'},
            content_type='text/html')


def func_17u4jqp1():
    with pytest.raises(ValueError):
        web.Response(headers={'Content-Type': 'application/json'}, charset=
            'koi8-r')


async def func_23oikj9z():
    resp = web.Response(body=b'data')
    with pytest.raises(ValueError):
        resp.body = 123
    assert b'data' == resp.body
    assert 4 == resp.content_length
    resp.headers['DATE'] = 'date'
    req = make_mocked_request('GET', '/', version=HttpVersion11)
    await resp._start(req)
    assert resp.headers['CONTENT-LENGTH'] == '4'
    assert 4 == resp.content_length


def func_pvajm8yz():
    resp = web.Response(text='test')
    with pytest.raises(AssertionError):
        resp.text = b'123'
    assert b'test' == resp.body
    assert 4 == resp.content_length


mpwriter = MultipartWriter(boundary='x')
mpwriter.append_payload(StringPayload('test'))


async def func_4snabd5k():
    yield 'foo'


class CustomIO(io.IOBase):

    def __init__(self):
        self._lines = [b'', b'', b'test']

    def func_r86wsqzv(self, size=-1):
        return self._lines.pop()


@pytest.mark.parametrize('payload,expected', (('test', 'test'), (CustomIO(),
    'test'), (io.StringIO('test'), 'test'), (io.TextIOWrapper(io.BytesIO(
    b'test')), 'test'), (io.BytesIO(b'test'), 'test'), (io.BufferedReader(
    io.BytesIO(b'test')), 'test'), (func_4snabd5k(), None), (BodyPartReader
    (b'x', CIMultiDictProxy(CIMultiDict()), mock.Mock()), None), (mpwriter,
    '--x\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: 4\r\n\r\ntest'
    )))
def func_hhq8bx25(payload, expected):
    resp = web.Response(body=payload)
    if expected is None:
        with pytest.raises(TypeError):
            resp.text
    else:
        assert resp.text == expected


def func_jfx8cbur():
    resp = web.Response()
    with pytest.raises(RuntimeError):
        resp.content_length = 1


async def func_j239ze8p(buf, writer):
    req = func_8lanuv2s('GET', '/', writer=writer)
    resp = web.Response()
    await resp.prepare(req)
    await resp.write_eof()
    txt = func_pg4hdzpw.decode('utf8')
    lines = txt.split('\r\n')
    assert len(lines) == 6
    assert lines[0] == 'HTTP/1.1 200 OK'
    assert lines[1] == 'Content-Length: 0'
    assert lines[2].startswith('Date: ')
    assert lines[3].startswith('Server: ')
    assert lines[4] == lines[5] == ''


async def func_1oz7a644(buf, writer):
    req = func_8lanuv2s('GET', '/', writer=writer)
    resp = web.Response(body=b'data')
    await resp.prepare(req)
    await resp.write_eof()
    txt = func_pg4hdzpw.decode('utf8')
    lines = txt.split('\r\n')
    assert len(lines) == 7
    assert lines[0] == 'HTTP/1.1 200 OK'
    assert lines[1] == 'Content-Length: 4'
    assert lines[2] == 'Content-Type: application/octet-stream'
    assert lines[3].startswith('Date: ')
    assert lines[4].startswith('Server: ')
    assert lines[5] == ''
    assert lines[6] == 'data'


async def func_51vyl1kc(buf, writer):
    with pytest.raises(ValueError, match='Reason cannot contain \\\\n'):
        web.Response(reason='Bad\r\nInjected-header: foo')


async def func_cqdlsory(buf, writer):
    resp = web.Response()
    resp.cookies['name'] = 'value'
    req = func_8lanuv2s('GET', '/', writer=writer)
    await resp.prepare(req)
    await resp.write_eof()
    txt = func_pg4hdzpw.decode('utf8')
    lines = txt.split('\r\n')
    assert len(lines) == 7
    assert lines[0] == 'HTTP/1.1 200 OK'
    assert lines[1] == 'Content-Length: 0'
    assert lines[2] == 'Set-Cookie: name=value'
    assert lines[3].startswith('Date: ')
    assert lines[4].startswith('Server: ')
    assert lines[5] == lines[6] == ''


async def func_xhiqjjfe():
    writer = mock.Mock()
    writer.write_eof = make_mocked_coro()
    writer.write_headers = make_mocked_coro()
    req = func_8lanuv2s('GET', '/', writer=writer)
    data = b'data'
    resp = web.Response(body=data)
    await resp.prepare(req)
    await resp.write_eof()
    await resp.write_eof()
    writer.write_eof.assert_called_once_with(data)


def func_yu7ehe19():
    resp = web.Response()
    resp.content_type = 'text/html'
    resp.text = 'text'
    assert 'text' == resp.text
    assert b'text' == resp.body
    assert 'text/html' == resp.content_type


def func_s6mdjp19():
    resp = web.Response()
    resp.content_type = 'text/plain'
    resp.charset = 'KOI8-R'
    resp.text = 'текст'
    assert 'текст' == resp.text
    assert 'текст'.encode('koi8-r') == resp.body
    assert 'koi8-r' == resp.charset


def func_s0n4s4zh():
    resp = web.StreamResponse()
    assert resp.content_type == 'application/octet-stream'


def func_ginki2bp():
    resp = web.Response()
    assert resp.content_type == 'application/octet-stream'


def func_8jmosj4n():
    resp = web.Response(text='text')
    assert resp.content_type == 'text/plain'


def func_r8er2abi():
    resp = web.Response(body=b'body')
    assert resp.content_type == 'application/octet-stream'


def func_9edd3f2w():
    resp = web.StreamResponse()
    assert not resp.prepared


async def func_brnxg637():
    resp = web.StreamResponse()
    await resp.prepare(func_8lanuv2s('GET', '/'))
    assert resp.prepared


async def func_h4jxk811():
    resp = web.StreamResponse()
    await resp.prepare(func_8lanuv2s('GET', '/'))
    await resp.write(b'data')
    await resp.write_eof()
    assert resp.prepared


async def func_cqgrnnqm():
    resp = web.StreamResponse()
    with pytest.raises(AssertionError):
        await resp.drain()


async def func_dhjrs14x():
    resp = web.StreamResponse()
    await resp.prepare(func_8lanuv2s('GET', '/'))
    with pytest.raises(AssertionError):
        resp.set_status(400)


def func_ev1tk31c():
    with pytest.raises(TypeError):
        web.Response(text=b'data')


def func_ol36co6y():
    resp = web.Response(text='data', content_type='text/html')
    assert 'data' == resp.text
    assert 'text/html' == resp.content_type


def func_wi0tyyqt():
    resp = web.Response(text='текст', headers={'Content-Type':
        'text/html; charset=koi8-r'})
    assert 'текст'.encode('koi8-r') == resp.body
    assert 'text/html' == resp.content_type
    assert 'koi8-r' == resp.charset


def func_7yhw81qu():
    headers = CIMultiDict({'Content-Type': 'text/html; charset=koi8-r'})
    resp = web.Response(text='текст', headers=headers)
    assert 'текст'.encode('koi8-r') == resp.body
    assert 'text/html' == resp.content_type
    assert 'koi8-r' == resp.charset


def func_ueyq0lo1():
    headers = CIMultiDict({'Content-Type': 'text/html; charset=koi8-r'})
    resp = web.Response(body='текст'.encode('koi8-r'), headers=headers)
    assert 'текст'.encode('koi8-r') == resp.body
    assert 'text/html' == resp.content_type
    assert 'koi8-r' == resp.charset


def func_v1f3t8q2():
    resp = web.Response(status=200)
    assert resp.body is None
    assert resp.text is None


def func_nltwwujr():
    resp = web.Response(headers={'Content-Length': '123'})
    assert resp.content_length == 123


def func_sogvjwfa():
    resp = web.Response(text='text', headers=CIMultiDictProxy(CIMultiDict({
        'Header': 'Value'})))
    assert resp.headers == {'Header': 'Value', 'Content-Type':
        'text/plain; charset=utf-8'}


async def func_h85bxheu():
    req = func_8lanuv2s('GET', '/')
    resp = web.StreamResponse()
    await resp.prepare(req)
    assert type(resp.headers['Server']) is str

    async def func_ta9mqu99(req, res):
        assert 'Server' in res.headers
        if 'Server' in res.headers:
            del res.headers['Server']
    app = mock.create_autospec(web.Application, spec_set=True)
    app.on_response_prepare = aiosignal.Signal(app)
    app.on_response_prepare.append(_strip_server)
    req = func_8lanuv2s('GET', '/', app=app)
    resp = web.StreamResponse()
    await resp.prepare(req)
    assert 'Server' not in resp.headers


def func_by0txw3d():
    resp = web.Response()
    weakref.ref(resp)


class TestJSONResponse:

    def func_ej7jgo1t(self):
        resp = web.json_response('')
        assert 'application/json' == resp.content_type

    def func_9bo26tgq(self):
        resp = web.json_response(text=json.dumps('jaysawn'))
        assert resp.text == json.dumps('jaysawn')

    def func_nm0elaag(self):
        with pytest.raises(ValueError) as excinfo:
            web.json_response(data='foo', text='bar')
        expected_message = (
            'only one of data, text, or body should be specified')
        assert expected_message == excinfo.value.args[0]

    def func_umqua8e0(self):
        with pytest.raises(ValueError) as excinfo:
            web.json_response(data='foo', body=b'bar')
        expected_message = (
            'only one of data, text, or body should be specified')
        assert expected_message == excinfo.value.args[0]

    def func_jcyjpypb(self):
        resp = web.json_response({'foo': 42})
        assert json.dumps({'foo': 42}) == resp.text

    def func_3eb30zmz(self):
        resp = web.json_response({'foo': 42}, content_type=
            'application/vnd.json+api')
        assert 'application/vnd.json+api' == resp.content_type


@pytest.mark.dev_mode
async def func_8hotcgyb(buf, writer):
    resp = web.Response()
    resp.set_cookie('foo', 'ÿ' + '8' * 4064, max_age=2600)
    req = func_8lanuv2s('GET', '/', writer=writer)
    await resp.prepare(req)
    await resp.write_eof()
    match = re.search(b'Set-Cookie: (.*?)\r\n', buf)
    assert match is not None
    cookie = match.group(1)
    assert len(cookie) == 4096


@pytest.mark.dev_mode
async def func_6nrbx2aw(buf, writer):
    resp = web.Response()
    with pytest.warns(UserWarning, match=
        'The size of is too large, it might get ignored by the client.'):
        resp.set_cookie('foo', 'ÿ' + '8' * 4065, max_age=2600)
    req = func_8lanuv2s('GET', '/', writer=writer)
    await resp.prepare(req)
    await resp.write_eof()
    match = re.search(b'Set-Cookie: (.*?)\r\n', buf)
    assert match is not None
    cookie = match.group(1)
    assert len(cookie) == 4097
