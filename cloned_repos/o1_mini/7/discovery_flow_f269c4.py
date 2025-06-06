"""The discovery flow helper."""
from __future__ import annotations
from collections.abc import Coroutine
import dataclasses
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Dict, List, Tuple, Union
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import (
    CoreState,
    Event,
    HomeAssistant,
    callback,
)
from homeassistant.loader import bind_hass
from homeassistant.util.async_ import gather_with_limited_concurrency
from homeassistant.util.hass_dict import HassKey

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigFlowContext, ConfigFlowResult

FLOW_INIT_LIMIT: int = 20
DISCOVERY_FLOW_DISPATCHER: HassKey = HassKey("discovery_flow_dispatcher")


@dataclasses.dataclass(kw_only=True, slots=True)
class DiscoveryKey:
    """Serializable discovery key."""

    domain: str
    key: Union[Tuple[Any, ...], Any]
    version: int

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> DiscoveryKey:
        """Construct from JSON dict."""
        key = json_dict["key"]
        if isinstance(key, list):
            key = tuple(key)
        return cls(domain=json_dict["domain"], key=key, version=json_dict["version"])


@bind_hass
@callback
def async_create_flow(
    hass: HomeAssistant,
    domain: str,
    context: Dict[str, Any],
    data: Dict[str, Any],
    *,
    discovery_key: Optional[DiscoveryKey] = None,
) -> None:
    """Create a discovery flow."""
    dispatcher: Optional[FlowDispatcher] = None
    if DISCOVERY_FLOW_DISPATCHER in hass.data:
        dispatcher = hass.data[DISCOVERY_FLOW_DISPATCHER]
    elif hass.state is not CoreState.RUNNING:
        dispatcher = hass.data[DISCOVERY_FLOW_DISPATCHER] = FlowDispatcher(hass)
        dispatcher.async_setup()
    if discovery_key:
        context = {**context, "discovery_key": discovery_key}
    if not dispatcher or dispatcher.started:
        init_coro = _async_init_flow(hass, domain, context, data)
        if init_coro:
            hass.async_create_background_task(
                init_coro, f"discovery flow {domain} {context}", eager_start=True
            )
        return
    dispatcher.async_create(domain, context, data)


@callback
def _async_init_flow(
    hass: HomeAssistant,
    domain: str,
    context: Dict[str, Any],
    data: Dict[str, Any],
) -> Optional[Coroutine[Any, Any, ConfigFlowResult]]:
    """Create a discovery flow."""
    if (
        hass.config_entries.flow.async_has_matching_discovery_flow(domain, context, data)
        or hass.is_stopping
    ):
        return None
    return hass.config_entries.flow.async_init(domain, context=context, data=data)


class PendingFlowKey(NamedTuple):
    """Key for pending flows."""

    domain: str
    source: str


class PendingFlowValue(NamedTuple):
    """Value for pending flows."""

    context: Dict[str, Any]
    data: Dict[str, Any]


class FlowDispatcher:
    """Dispatch discovery flows."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Init the discovery dispatcher."""
        self.hass: HomeAssistant = hass
        self.started: bool = False
        self.pending_flows: Dict[PendingFlowKey, List[PendingFlowValue]] = {}

    @callback
    def async_setup(self) -> None:
        """Set up the flow dispatcher."""
        self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, self._async_start)

    async def _async_start(self, event: Event) -> None:
        """Start processing pending flows."""
        pending_flows: Dict[PendingFlowKey, List[PendingFlowValue]] = self.pending_flows
        self.pending_flows = {}
        self.started = True
        init_coros: List[Coroutine[Any, Any, Optional[ConfigFlowResult]]] = [
            _async_init_flow(
                self.hass, flow_key.domain, flow_value.context, flow_value.data
            )
            for flow_key, flows in pending_flows.items()
            for flow_value in flows
            if _async_init_flow(
                self.hass, flow_key.domain, flow_value.context, flow_value.data
            )
        ]
        await gather_with_limited_concurrency(FLOW_INIT_LIMIT, *init_coros)

    @callback
    def async_create(self, domain: str, context: Dict[str, Any], data: Dict[str, Any]) -> None:
        """Create and add or queue a flow."""
        key = PendingFlowKey(domain=domain, source=context.get("source", ""))
        value = PendingFlowValue(context=context, data=data)
        existing: List[PendingFlowValue] = self.pending_flows.setdefault(key, [])
        if not any(existing_value.data == data for existing_value in existing):
            existing.append(value)
