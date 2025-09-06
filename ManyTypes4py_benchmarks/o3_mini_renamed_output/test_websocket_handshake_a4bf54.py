import base64
import os
from typing import List, Tuple, Optional
import pytest
from pytest import LogCaptureFixture
from aiohttp import web
from aiohttp.test_utils import make_mocked_request


def func_uhp1xhb0(protocols: str = '', compress: int = 0, extension_text: str = '',
                  server_notakeover: bool = False, client_notakeover: bool = False) -> Tuple[List[Tuple[str, str]], str]:
    key: str = base64.b64encode(os.urandom(16)).decode()
    hdrs: List[Tuple[str, str]] = [('Upgrade', 'websocket'), ('Connection', 'upgrade'),
                                   ('Sec-Websocket-Version', '13'), ('Sec-Websocket-Key', key)]
    if protocols:
        hdrs += [('Sec-Websocket-Protocol', protocols)]
    if compress:
        params: str = 'permessage-deflate'
        if compress < 15:
            params += '; server_max_window_bits=' + str(compress)
        if server_notakeover:
            params += '; server_no_context_takeover'
        if client_notakeover:
            params += '; client_no_context_takeover'
        if extension_text:
            params += '; ' + extension_text
        hdrs += [('Sec-Websocket-Extensions', params)]
    return hdrs, key


async def func_x7hcqoxf() -> None:
    ws: web.WebSocketResponse = web.WebSocketResponse()
    req = make_mocked_request('GET', '/')
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)


async def func_nuee4r86() -> None:
    ws: web.WebSocketResponse = web.WebSocketResponse()
    req = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket',
                                                   'Connection': 'keep-alive'})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)


async def func_zpr3y31c() -> None:
    ws: web.WebSocketResponse = web.WebSocketResponse()
    req = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket',
                                                   'Connection': 'upgrade'})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)


async def func_z1s6r4m8() -> None:
    ws: web.WebSocketResponse = web.WebSocketResponse()
    req = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket',
                                                   'Connection': 'upgrade', 'Sec-Websocket-Version': '1'})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)


async def func_rfii3jqh() -> None:
    ws: web.WebSocketResponse = web.WebSocketResponse()
    req = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket',
                                                   'Connection': 'upgrade', 'Sec-Websocket-Version': '13'})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)


async def func_7w9cbwbl() -> None:
    ws: web.WebSocketResponse = web.WebSocketResponse()
    req = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket',
                                                   'Connection': 'upgrade', 'Sec-Websocket-Version': '13',
                                                   'Sec-Websocket-Key': '123'})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)


async def func_pbhjfmhz() -> None:
    ws: web.WebSocketResponse = web.WebSocketResponse()
    sec_key = base64.b64encode(os.urandom(2))
    val: str = sec_key.decode()
    req = make_mocked_request('GET', '/', headers={'Upgrade': 'websocket',
                                                   'Connection': 'upgrade', 'Sec-Websocket-Version': '13',
                                                   'Sec-Websocket-Key': val})
    with pytest.raises(web.HTTPBadRequest):
        await ws.prepare(req)


async def func_n8od906p() -> None:
    hdrs, sec_key = func_uhp1xhb0()
    ws: web.WebSocketResponse = web.WebSocketResponse()
    req = make_mocked_request('GET', '/', headers=hdrs)
    await ws.prepare(req)
    assert ws.ws_protocol is None


async def func_ifnv4n9d() -> None:
    proto: str = 'chat'
    ws: web.WebSocketResponse = web.WebSocketResponse(protocols={'chat'})
    req = make_mocked_request('GET', '/', headers=func_uhp1xhb0(proto)[0])
    await ws.prepare(req)
    assert ws.ws_protocol == proto


async def func_ajb52y6b() -> None:
    best_proto: str = 'worse_proto'
    wanted_protos: List[str] = ['best', 'chat', 'worse_proto']
    server_protos: str = 'worse_proto,chat'
    ws: web.WebSocketResponse = web.WebSocketResponse(protocols=wanted_protos)
    req = make_mocked_request('GET', '/', headers=func_uhp1xhb0(server_protos)[0])
    await ws.prepare(req)
    assert ws.ws_protocol == best_proto


async def func_qc8r87om(caplog: LogCaptureFixture) -> None:
    proto: str = 'chat'
    req = make_mocked_request('GET', '/', headers=func_uhp1xhb0('test')[0])
    ws: web.WebSocketResponse = web.WebSocketResponse(protocols=[proto])
    await ws.prepare(req)
    assert caplog.records[-1].msg == 'Client protocols %r don’t overlap server-known ones %r'
    assert ws.ws_protocol is None


async def func_ljooxl5o() -> None:
    hdrs, sec_key = func_uhp1xhb0(compress=15)
    req = make_mocked_request('GET', '/', headers=hdrs)
    ws: web.WebSocketResponse = web.WebSocketResponse()
    await ws.prepare(req)
    assert ws.compress == 15


def func_6sqg7j6p() -> None:
    hdrs, sec_key = func_uhp1xhb0(compress=15, server_notakeover=True)
    req = make_mocked_request('GET', '/', headers=hdrs)
    ws: web.WebSocketResponse = web.WebSocketResponse()
    headers, _, compress, notakeover = ws._handshake(req)
    assert compress == 15
    assert notakeover is True
    assert 'Sec-Websocket-Extensions' in headers
    assert headers['Sec-Websocket-Extensions'] == 'permessage-deflate; server_no_context_takeover'


def func_u16rl4ww() -> None:
    hdrs, sec_key = func_uhp1xhb0(compress=15, client_notakeover=True)
    req = make_mocked_request('GET', '/', headers=hdrs)
    ws: web.WebSocketResponse = web.WebSocketResponse()
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' in headers
    assert headers['Sec-Websocket-Extensions'] == 'permessage-deflate', hdrs
    assert compress == 15


def func_5leph687() -> None:
    hdrs, sec_key = func_uhp1xhb0(compress=9)
    req = make_mocked_request('GET', '/', headers=hdrs)
    ws: web.WebSocketResponse = web.WebSocketResponse()
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' in headers
    assert headers['Sec-Websocket-Extensions'] == 'permessage-deflate; server_max_window_bits=9'
    assert compress == 9


def func_6dz71d2d() -> None:
    hdrs, sec_key = func_uhp1xhb0(compress=6)
    req = make_mocked_request('GET', '/', headers=hdrs)
    ws: web.WebSocketResponse = web.WebSocketResponse()
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' not in headers
    assert compress == 0


def func_2coswv6b() -> None:
    hdrs, sec_key = func_uhp1xhb0(compress=15, extension_text='bad')
    req = make_mocked_request('GET', '/', headers=hdrs)
    ws: web.WebSocketResponse = web.WebSocketResponse()
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' not in headers
    assert compress == 0


def func_l8tr2ikb() -> None:
    hdrs, sec_key = func_uhp1xhb0(compress=15, extension_text='bad, permessage-deflate')
    req = make_mocked_request('GET', '/', headers=hdrs)
    ws: web.WebSocketResponse = web.WebSocketResponse()
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' in headers
    assert headers['Sec-Websocket-Extensions'] == 'permessage-deflate'


def func_qekiz0ef() -> None:
    hdrs, sec_key = func_uhp1xhb0(compress=6, extension_text=', permessage-deflate')
    req = make_mocked_request('GET', '/', headers=hdrs)
    ws: web.WebSocketResponse = web.WebSocketResponse()
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Sec-Websocket-Extensions' in headers
    assert headers['Sec-Websocket-Extensions'] == 'permessage-deflate'
    assert compress == 15


def func_8b1cer27() -> None:
    hdrs, sec_key = func_uhp1xhb0()
    req = make_mocked_request('GET', '/', headers=hdrs)
    ws: web.WebSocketResponse = web.WebSocketResponse()
    headers, _, compress, notakeover = ws._handshake(req)
    assert 'Transfer-Encoding' not in headers