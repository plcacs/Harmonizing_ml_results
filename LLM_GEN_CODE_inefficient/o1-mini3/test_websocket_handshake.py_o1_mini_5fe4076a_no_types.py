import base64
import os
from typing import List, Tuple, Dict, Any, Optional
import pytest
from aiohttp import web
from aiohttp.web_request import Request
from aiohttp.test_utils import make_mocked_request

def gen_ws_headers(protocols='', compress=0, extension_text='', server_notakeover=False, client_notakeover=False):
    key = base64.b64encode(os.urandom(16)).decode()
    hdrs: List[Tuple[str, str]] = [('Upgrade', 'websocket'), ('Connection', 'upgrade'), ('Sec-Websocket-Version', '13'), ('Sec-Websocket-Key', key)]
    if protocols:
        hdrs += [('Sec-Websocket-Protocol', protocols)]
    if compress:
        params = 'permessage-deflate'
        if compress < 15:
            params += '; server_max_window_bits=' + str(compress)
        if server_notakeover:
            params += '; server_no_context_takeover'
        if client_notakeover:
            params += '; client_no_context_takeover'
        if extension_text:
            params += '; ' + extension_text
        hdrs += [('Sec-Websocket-Extensions', params)]
    return (hdrs, key)

async def test_no_upgrade() -> None:
    ws = web.WebSocketResponse()
    req: Request = make_mocked_request('GET', '/')
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)

async def test_no_connection() -> None:
    ws = web.WebSocketResponse()
    req: Request = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket', 'Connection': 'keep-alive'})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)

async def test_protocol_version_unset() -> None:
    ws = web.WebSocketResponse()
    req: Request = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket', 'Connection': 'upgrade'})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)

async def test_protocol_version_not_supported() -> None:
    ws = web.WebSocketResponse()
    req: Request = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket', 'Connection': 'upgrade', 'Sec-Websocket-Version': '1'})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)

async def test_protocol_key_not_present() -> None:
    ws = web.WebSocketResponse()
    req: Request = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket', 'Connection': 'upgrade', 'Sec-Websocket-Version': '13'})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)

async def test_protocol_key_invalid() -> None:
    ws = web.WebSocketResponse()
    req: Request = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket', 'Connection': 'upgrade', 'Sec-Websocket-Version': '13', 'Sec-Websocket-Key': '123'})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)

async def test_protocol_key_bad_size() -> None:
    ws = web.WebSocketResponse()
    sec_key: bytes = base64.b64encode(os.urandom(2))
    val: str = sec_key.decode()
    req: Request = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket', 'Connection': 'upgrade', 'Sec-Websocket-Version': '13', 'Sec-Websocket-Key': val})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)

async def test_handshake_ok() -> None:
    hdrs, sec_key = gen_ws_headers()
    ws = web.WebSocketResponse()
    req: Request = make_mocked_request('GET', '/', headers=hdrs)
    await ws.prepare(req)
    assert ws.ws_protocol is None

async def test_handshake_protocol() -> None:
    proto: str = 'chat'
    ws = web.WebSocketResponse(protocols={'chat'})
    req: Request = make_mocked_request('GET', '/', headers=gen_ws_headers(proto)[0])
    await ws.prepare(req)
    assert ws.ws_protocol == proto

async def test_handshake_protocol_agreement() -> None:
    best_proto: str = 'worse_proto'
    wanted_protos: List[str] = ['best', 'chat', 'worse_proto']
    server_protos: str = 'worse_proto,chat'
    ws = web.WebSocketResponse(protocols=wanted_protos)
    req: Request = make_mocked_request('GET', '/', headers=gen_ws_headers(server_protos)[0])
    await ws.prepare(req)
    assert ws.ws_protocol == best_proto

async def test_handshake_protocol_unsupported(caplog: pytest.LogCaptureFixture) -> None:
    proto: str = 'chat'
    req: Request = make_mocked_request('GET', '/', headers=gen_ws_headers('test')[0])
    ws = web.WebSocketResponse(protocols=[proto])
    await ws.prepare(req)
    assert caplog.records[-1].msg == 'Client protocols %r don’t overlap server-known ones %r'
    assert ws.ws_protocol is None

async def test_handshake_compress() -> None:
    hdrs, sec_key = gen_ws_headers(compress=15)
    req: Request = make_mocked_request('GET', '/', headers=hdrs)
    ws = web.WebSocketResponse()
    await ws.prepare(req)
    assert ws.compress == 15

def test_handshake_compress_server_notakeover():
    hdrs, sec_key = gen_ws_headers(compress=15, server_notakeover=True)
    req: Request = make_mocked_request('GET', '/', headers=hdrs)
    ws = web.WebSocketResponse()
    headers: web.Headers
    compress: int
    notakeover: bool
    headers, _, compress, notakeover = ws._handshake(req)
    assert compress == 15
    assert notakeover is True
    assert 'Sec-Websocket-Extensions' in headers
    assert headers['Sec-Websocket-Extensions'] == 'permessage-deflate; server_no_context_takeover'

def test_handshake_compress_client_notakeover():
    hdrs, sec_key = gen_ws_headers(compress=15, client_notakeover=True)
    req: Request = make_mocked_request('GET', '/', headers=hdrs)
    ws = web.WebSocketResponse()
    headers: web.Headers
    compress: int
    notakeover: Optional[bool]
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' in headers
    assert headers['Sec-Websocket-Extensions'] == 'permessage-deflate', hdrs
    assert compress == 15

def test_handshake_compress_wbits():
    hdrs, sec_key = gen_ws_headers(compress=9)
    req: Request = make_mocked_request('GET', '/', headers=hdrs)
    ws = web.WebSocketResponse()
    headers: web.Headers
    compress: int
    notakeover: Optional[bool]
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' in headers
    assert headers['Sec-Websocket-Extensions'] == 'permessage-deflate; server_max_window_bits=9'
    assert compress == 9

def test_handshake_compress_wbits_error():
    hdrs, sec_key = gen_ws_headers(compress=6)
    req: Request = make_mocked_request('GET', '/', headers=hdrs)
    ws = web.WebSocketResponse()
    headers: web.Headers
    compress: int
    notakeover: Optional[bool]
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' not in headers
    assert compress == 0

def test_handshake_compress_bad_ext():
    hdrs, sec_key = gen_ws_headers(compress=15, extension_text='bad')
    req: Request = make_mocked_request('GET', '/', headers=hdrs)
    ws = web.WebSocketResponse()
    headers: web.Headers
    compress: int
    notakeover: Optional[bool]
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' not in headers
    assert compress == 0

def test_handshake_compress_multi_ext_bad():
    hdrs, sec_key = gen_ws_headers(compress=15, extension_text='bad, permessage-deflate')
    req: Request = make_mocked_request('GET', '/', headers=hdrs)
    ws = web.WebSocketResponse()
    headers: web.Headers
    compress: int
    notakeover: Optional[bool]
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' in headers
    assert headers['Sec-Websocket-Extensions'] == 'permessage-deflate'

def test_handshake_compress_multi_ext_wbits():
    hdrs, sec_key = gen_ws_headers(compress=6, extension_text=', permessage-deflate')
    req: Request = make_mocked_request('GET', '/', headers=hdrs)
    ws = web.WebSocketResponse()
    headers: web.Headers
    compress: int
    notakeover: Optional[bool]
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' in headers
    assert headers['Sec-Websocket-Extensions'] == 'permessage-deflate'
    assert compress == 15

def test_handshake_no_transfer_encoding():
    hdrs, sec_key = gen_ws_headers()
    req: Request = make_mocked_request('GET', '/', headers=hdrs)
    ws = web.WebSocketResponse()
    headers: web.Headers
    compress: int
    notakeover: Optional[bool]
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Transfer-Encoding' not in headers