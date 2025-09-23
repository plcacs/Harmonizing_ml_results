import abc
from textwrap import dedent
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Type, Union
from uuid import UUID
import httpx
from typing_extensions import Self, TypeAlias
from prefect.client.base import PrefectHttpxAsyncClient
from prefect.logging import get_logger
from prefect.server.events import messaging
from prefect.server.events.schemas.events import Event, ReceivedEvent, ResourceSpecification
if TYPE_CHECKING:
    import logging
logger = get_logger(__name__)
LabelValue: TypeAlias = Union[str, List[str]]

class EventsClient(abc.ABC):
    """The abstract interface for a Prefect Events client"""

    @abc.abstractmethod
    async def emit(self, event: Event) -> Optional[Event]:
        ...

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]) -> None:
        return None

class NullEventsClient(EventsClient):
    """A no-op implementation of the Prefect Events client for testing"""

    async def emit(self, event: Event) -> None:
        pass

class AssertingEventsClient(EventsClient):
    """An implementation of the Prefect Events client that records all events sent
    to it for inspection during tests."""
    last: Optional['AssertingEventsClient'] = None
    all: List['AssertingEventsClient'] = []
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    events: List[Event]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        AssertingEventsClient.last = self
        AssertingEventsClient.all.append(self)
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def reset(cls) -> None:
        """Reset all captured instances and their events.  For use this between tests"""
        cls.last = None
        cls.all = []

    async def emit(self, event: Event) -> Event:
        if not hasattr(self, 'events'):
            raise TypeError('Events may only be emitted while this client is being used as a context manager')
        self.events.append(event)
        return event

    async def __aenter__(self) -> Self:
        self.events = []
        return self

    async def __aexit__(self, exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]) -> None:
        return None

    @classmethod
    def emitted_events_count(cls) -> int:
        return sum((len(client.events) for client in cls.all))

    @classmethod
    def assert_emitted_event_count(cls, count: int) -> None:
        """Assert that the given number of events were emitted."""
        total_num_events = cls.emitted_events_count()
        assert total_num_events == count, f'The number of emitted events did not match the expected count: total_num_events={total_num_events!r} != count={count!r}'

    @classmethod
    def assert_emitted_event_with(cls, event: Optional[str]=None, resource: Optional[Dict[str, LabelValue]]=None, related: Optional[List[Dict[str, LabelValue]]]=None, payload: Optional[Dict[str, Any]]=None) -> None:
        """Assert that an event was emitted containing the given properties."""
        assert cls.last is not None and cls.all, 'No event client was created'
        emitted_events = [event for client in cls.all for event in reversed(client.events)]
        resource_spec = ResourceSpecification.model_validate(resource) if resource else None
        related_specs = [ResourceSpecification.model_validate(related_resource) for related_resource in related] if related else None
        mismatch_reasons: List[Tuple[str, str]] = []

        def event_matches(emitted_event: Event) -> bool:
            if event is not None and emitted_event.event != event:
                mismatch_reasons.append((f'event={event!r}', f'emitted_event.event={emitted_event.event!r}'))
                return False
            if resource_spec and (not resource_spec.matches(emitted_event.resource)):
                mismatch_reasons.append((f'resource={resource!r}', f'emitted_event.resource={emitted_event.resource!r}'))
                return False
            if related_specs:
                for related_spec in related_specs:
                    if not any((related_spec.matches(related_resource) for related_resource in emitted_event.related)):
                        mismatch_reasons.append((f'related={related!r}', f'emitted_event.related={emitted_event.related!r}'))
                        return False
            if payload and any((emitted_event.payload.get(k) != v for (k, v) in payload.items())):
                mismatch_reasons.append((f'payload={payload!r}', f'emitted_event.payload={emitted_event.payload!r}'))
                return False
            return True
        assert any((event_matches(emitted_event) for emitted_event in emitted_events)), f'''An event was not emitted matching the following criteria:\n    event={event!r}\n    resource={resource!r}\n    related={related!r}\n    payload={payload!r}\n\n# of captured events: {len(emitted_events)}\n{chr(10).join((dedent(f"""
                    Expected:
                        {expected}
                    Received:
                        {received}
                """) for (expected, received) in mismatch_reasons))}\n'''

    @classmethod
    def assert_no_emitted_event_with(cls, event: Optional[str]=None, resource: Optional[Dict[str, LabelValue]]=None, related: Optional[List[Dict[str, LabelValue]]]=None, payload: Optional[Dict[str, Any]]=None) -> None:
        try:
            cls.assert_emitted_event_with(event, resource, related, payload)
        except AssertionError:
            return
        else:
            assert False, 'An event was emitted matching the given criteria'

class PrefectServerEventsClient(EventsClient):
    _publisher: messaging.EventPublisher

    async def __aenter__(self) -> Self:
        publisher = messaging.create_event_publisher()
        self._publisher = await publisher.__aenter__()
        return self

    async def __aexit__(self, exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]) -> None:
        await self._publisher.__aexit__(exc_type, exc_val, exc_tb)
        del self._publisher
        return None

    async def emit(self, event: Event) -> ReceivedEvent:
        if not hasattr(self, '_publisher'):
            raise TypeError('Events may only be emitted while this client is being used as a context manager')
        received_event = event.receive()
        await self._publisher.publish_event(received_event)
        return received_event

class PrefectServerEventsAPIClient:
    _http_client: PrefectHttpxAsyncClient

    def __init__(self, additional_headers: Dict[str, str]={}) -> None:
        from prefect.server.api.server import create_app
        api_app = create_app()
        self._http_client = PrefectHttpxAsyncClient(transport=httpx.ASGITransport(app=api_app, raise_app_exceptions=False), headers={**additional_headers}, base_url='http://prefect-in-memory/api', enable_csrf_support=False, raise_on_all_errors=False)

    async def __aenter__(self) -> Self:
        await self._http_client.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._http_client.__aexit__(*args)

    async def pause_automation(self, automation_id: UUID) -> httpx.Response:
        return await self._http_client.patch(f'/automations/{automation_id}', json={'enabled': False})

    async def resume_automation(self, automation_id: UUID) -> httpx.Response:
        return await self._http_client.patch(f'/automations/{automation_id}', json={'enabled': True})
