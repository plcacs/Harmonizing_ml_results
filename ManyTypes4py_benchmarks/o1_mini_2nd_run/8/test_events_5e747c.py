import pytest
from mode.utils.mocks import AsyncMock, Mock
from faust import Event
from typing import Any, Dict, Tuple, Optional, Callable

class test_Event:

    @pytest.fixture
    def key(self) -> Mock:
        return Mock(name='key')

    @pytest.fixture
    def value(self) -> Mock:
        return Mock(name='value')

    @pytest.fixture
    def message(self) -> Mock:
        return Mock(name='message')

    @pytest.fixture
    def event(
        self,
        *,
        app: Any,
        key: Mock,
        value: Mock,
        message: Mock
    ) -> Event:
        return Event(app, key, value, {}, message)

    @pytest.mark.asyncio
    async def test_send(
        self,
        *,
        event: Event
    ) -> None:
        callback: Mock = Mock(name='callback')
        schema: Mock = Mock(name='schema')
        event._send = AsyncMock(name='event._send')
        await event.send(
            channel='chan',
            key=event.key,
            value=event.value,
            partition=3,
            timestamp=None,
            headers={'k': 'v'},
            schema=schema,
            key_serializer='kser',
            value_serializer='vser',
            callback=callback
        )
        event._send.assert_called_once_with(
            'chan',
            event.key,
            event.value,
            3,
            None,
            {'k': 'v'},
            schema,
            'kser',
            'vser',
            callback,
            force=False
        )

    @pytest.mark.asyncio
    async def test_send__USE_EXISTING_KEY_VALUE(
        self,
        *,
        event: Event
    ) -> None:
        callback: Mock = Mock(name='callback')
        event._send = AsyncMock(name='event._send')
        event.headers = {'k': 'v'}
        await event.send(
            channel='chan',
            partition=3,
            timestamp=None,
            schema=None,
            key_serializer='kser',
            value_serializer='vser',
            callback=callback
        )
        event._send.assert_called_once_with(
            'chan',
            event.key,
            event.value,
            3,
            None,
            event.headers,
            None,
            'kser',
            'vser',
            callback,
            force=False
        )

    @pytest.mark.asyncio
    async def test_forward(
        self,
        *,
        event: Event
    ) -> None:
        callback: Mock = Mock(name='callback')
        event._send = AsyncMock(name='event._send')
        await event.forward(
            channel='chan',
            key=event.key,
            value=event.value,
            headers={'k': 'v'},
            partition=3,
            timestamp=1234,
            schema=None,
            key_serializer='kser',
            value_serializer='vser',
            callback=callback
        )
        event._send.assert_called_once_with(
            'chan',
            event.key,
            event.value,
            3,
            1234,
            {'k': 'v'},
            None,
            'kser',
            'vser',
            callback,
            force=False
        )

    @pytest.mark.asyncio
    async def test_forward__USE_EXISTING_KEY_VALUE(
        self,
        *,
        event: Event
    ) -> None:
        callback: Mock = Mock(name='callback')
        event._send = AsyncMock(name='event._send')
        event.message.headers = {'k1': 'v1'}
        schema: Mock = Mock(name='schema')
        await event.forward(
            channel='chan',
            partition=3,
            timestamp=None,
            schema=schema,
            key_serializer='kser',
            value_serializer='vser',
            callback=callback
        )
        event._send.assert_called_once_with(
            'chan',
            event.message.key,
            event.message.value,
            3,
            None,
            event.message.headers,
            schema,
            'kser',
            'vser',
            callback,
            force=False
        )

    def test_attach(
        self,
        *,
        event: Event,
        app: Any
    ) -> Any:
        callback: Mock = Mock(name='callback')
        schema: Mock = Mock(name='schema')
        app._attachments.put = Mock(name='_attachments.put')
        result = event._attach(
            channel='chan',
            key=b'k',
            value=b'v',
            partition=3,
            timestamp=None,
            headers={'k': 'v'},
            schema=schema,
            key_serializer='kser',
            value_serializer='vser',
            callback=callback
        )
        app._attachments.put.assert_called_once_with(
            event.message,
            'chan',
            b'k',
            b'v',
            partition=3,
            timestamp=None,
            headers={'k': 'v'},
            schema=schema,
            key_serializer='kser',
            value_serializer='vser',
            callback=callback
        )
        assert result is app._attachments.put()

    @pytest.mark.asyncio
    async def test__send(
        self,
        *,
        event: Event,
        app: Any
    ) -> None:
        app._attachments.maybe_put = AsyncMock(name='maybe_put')
        callback: Mock = Mock(name='callback')
        schema: Mock = Mock(name='schema')
        await event._send(
            channel='chan',
            key=b'k',
            value=b'v',
            partition=4,
            timestamp=33.31234,
            headers=[('k', 'v')],
            schema=schema,
            key_serializer='kser',
            value_serializer='vser',
            callback=callback,
            force=True
        )
        app._attachments.maybe_put.assert_called_once_with(
            'chan',
            b'k',
            b'v',
            4,
            33.31234,
            [('k', 'v')],
            schema,
            'kser',
            'vser',
            callback,
            force=True
        )

    def test_repr(
        self,
        *,
        event: Event
    ) -> None:
        assert repr(event)

    @pytest.mark.asyncio
    async def test_AsyncContextManager(
        self,
        *,
        event: Event
    ) -> None:
        event.ack = Mock(name='ack')
        block_executed: bool = False
        async with event:
            block_executed = True
        assert block_executed
        event.ack.assert_called_once_with()
