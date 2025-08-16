from __future__ import annotations
from collections.abc import Coroutine
import dataclasses
from typing import TYPE_CHECKING, Any, NamedTuple, Tuple

from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import CoreState, Event, HomeAssistant, callback
from homeassistant.loader import bind_hass
from homeassistant.util.async_ import gather_with_limited_concurrency
from homeassistant.util.hass_dict import HassKey

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigFlowContext, ConfigFlowResult

FLOW_INIT_LIMIT: int = 20
DISCOVERY_FLOW_DISPATCHER: HassKey = HassKey('discovery_flow_dispatcher')

@dataclasses.dataclass(kw_only=True, slots=True)
class DiscoveryKey:
    domain: str
    key: Tuple[Any, ...]
    version: int

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> DiscoveryKey:
        if type((key := json_dict['key'])) is list:
            key = tuple(key)
        return cls(domain=json_dict['domain'], key=key, version=json_dict['version'])

@bind_hass
@callback
def async_create_flow(hass: HomeAssistant, domain: str, context: ConfigFlowContext, data: dict, *, discovery_key: DiscoveryKey = None) -> None:
    dispatcher = None
    if DISCOVERY_FLOW_DISPATCHER in hass.data:
        dispatcher = hass.data[DISCOVERY_FLOW_DISPATCHER]
    elif hass.state is not CoreState.running:
        dispatcher = hass.data[DISCOVERY_FLOW_DISPATCHER] = FlowDispatcher(hass)
        dispatcher.async_setup()
    if discovery_key:
        context = context | {'discovery_key': discovery_key}
    if not dispatcher or dispatcher.started:
        if (init_coro := _async_init_flow(hass, domain, context, data)):
            hass.async_create_background_task(init_coro, f'discovery flow {domain} {context}', eager_start=True)
        return
    dispatcher.async_create(domain, context, data)

@callback
def _async_init_flow(hass: HomeAssistant, domain: str, context: ConfigFlowContext, data: dict) -> Coroutine:
    if hass.config_entries.flow.async_has_matching_discovery_flow(domain, context, data) or hass.is_stopping:
        return None
    return hass.config_entries.flow.async_init(domain, context=context, data=data)

class PendingFlowKey(NamedTuple):
    domain: str
    source: Any

class PendingFlowValue(NamedTuple):
    context: ConfigFlowContext
    data: dict

class FlowDispatcher:
    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self.started = False
        self.pending_flows = {}

    @callback
    def async_setup(self) -> None:
        self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, self._async_start)

    async def _async_start(self, event: Event) -> None:
        pending_flows = self.pending_flows
        self.pending_flows = {}
        self.started = True
        init_coros = (init_coro for flow_key, flows in pending_flows.items() for flow_values in flows if (init_coro := _async_init_flow(self.hass, flow_key.domain, flow_values.context, flow_values.data)))
        await gather_with_limited_concurrency(FLOW_INIT_LIMIT, *init_coros)

    @callback
    def async_create(self, domain: str, context: ConfigFlowContext, data: dict) -> None:
        key = PendingFlowKey(domain, context['source'])
        values = PendingFlowValue(context, data)
        existing = self.pending_flows.setdefault(key, [])
        if not any((existing_values.data == data for existing_values in existing)):
            existing.append(values)
