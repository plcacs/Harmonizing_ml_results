from faust import App
from faust.types import TPayload, THeaders, TKey, TValue, TChannel
from typing import Any, Dict, Optional, Union, Tuple

class test_Event:

    def key(self) -> Mock:
        return Mock(name='key')

    def value(self) -> Mock:
        return Mock(name='value')

    def message(self) -> Mock:
        return Mock(name='message')

    def event(self, *, app: App, key: Mock, value: Mock, message: Mock) -> Event:
        return Event(app, key, value, {}, message)

    async def test_send(self, *, event: Event) -> None:
    
    async def test_send__USE_EXISTING_KEY_VALUE(self, *, event: Event) -> None:
    
    async def test_forward(self, *, event: Event) -> None:
    
    async def test_forward__USE_EXISTING_KEY_VALUE(self, *, event: Event) -> None:
    
    def test_attach(self, *, event: Event, app: App) -> None:
    
    async def test__send(self, *, event: Event, app: App) -> None:
    
    def test_repr(self, *, event: Event) -> None:
    
    async def test_AsyncContextManager(self, *, event: Event) -> None:
