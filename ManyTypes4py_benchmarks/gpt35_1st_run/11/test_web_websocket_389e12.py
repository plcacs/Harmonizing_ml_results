from typing import Optional, Protocol

class _RequestMaker(Protocol):

    def __call__(self, method: str, path: str, headers: Optional[dict] = None, protocols: bool = False) -> None:
        ...

def test_get_extra_info(make_request: _RequestMaker, mocker: MockerFixture, ws_transport: Optional[mock.MagicMock], expected_result: str) -> None:
    valid_key: str = 'test'
    default_value: str = 'default'
    req = make_request('GET', '/')
    ws = web.WebSocketResponse()
    await ws.prepare(req)
    ws._writer = ws_transport
    assert ws.get_extra_info(valid_key, default_value) == expected_result
