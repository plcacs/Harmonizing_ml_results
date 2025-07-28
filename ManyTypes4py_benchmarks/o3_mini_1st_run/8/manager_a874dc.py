"""Manager for esphome devices."""
from __future__ import annotations

import asyncio
from functools import partial
import logging
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING

from aioesphomeapi import (
    APIClient,
    APIConnectionError,
    APIVersion,
    DeviceInfo as EsphomeDeviceInfo,
    EntityInfo,
    HomeassistantServiceCall,
    InvalidAuthAPIError,
    InvalidEncryptionKeyAPIError,
    ReconnectLogic,
    RequiresEncryptionAPIError,
    UserService,
    UserServiceArgType,
)
from awesomeversion import AwesomeVersion
import voluptuous as vol

from homeassistant.components import bluetooth, tag, zeroconf
from homeassistant.const import ATTR_DEVICE_ID, CONF_MODE, EVENT_HOMEASSISTANT_CLOSE, EVENT_LOGGING_CHANGED, Platform
from homeassistant.core import Event, HomeAssistant, ServiceCall, State, callback
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import config_validation as cv, device_registry as dr, template
from homeassistant.helpers.device_registry import format_mac
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue, async_delete_issue
from homeassistant.helpers.service import async_set_service_schema
from homeassistant.helpers.template import Template
from homeassistant.util.async_ import create_eager_task

from .bluetooth import async_connect_scanner
from .const import CONF_ALLOW_SERVICE_CALLS, CONF_DEVICE_NAME, DEFAULT_ALLOW_SERVICE_CALLS, DEFAULT_URL, DOMAIN, PROJECT_URLS, STABLE_BLE_VERSION, STABLE_BLE_VERSION_STR
from .dashboard import async_get_dashboard
from .domain_data import DomainData
from .entry_data import ESPHomeConfigEntry, RuntimeEntryData

_LOGGER = logging.getLogger(__name__)


@callback
def _async_check_firmware_version(
    hass: HomeAssistant, device_info: EsphomeDeviceInfo, api_version: APIVersion
) -> None:
    """Create or delete the ble_firmware_outdated issue."""
    issue = f'ble_firmware_outdated-{device_info.mac_address}'
    if (
        not device_info.bluetooth_proxy_feature_flags_compat(api_version)
        or (device_info.project_name and device_info.project_name not in PROJECT_URLS)
        or AwesomeVersion(device_info.esphome_version) >= STABLE_BLE_VERSION
    ):
        async_delete_issue(hass, DOMAIN, issue)
        return
    async_create_issue(
        hass,
        DOMAIN,
        issue,
        is_fixable=False,
        severity=IssueSeverity.WARNING,
        learn_more_url=PROJECT_URLS.get(device_info.project_name, DEFAULT_URL),
        translation_key='ble_firmware_outdated',
        translation_placeholders={'name': device_info.name, 'version': STABLE_BLE_VERSION_STR},
    )


@callback
def _async_check_using_api_password(
    hass: HomeAssistant, device_info: EsphomeDeviceInfo, has_password: bool
) -> None:
    """Create or delete the api_password_deprecated issue."""
    issue = f'api_password_deprecated-{device_info.mac_address}'
    if not has_password:
        async_delete_issue(hass, DOMAIN, issue)
        return
    async_create_issue(
        hass,
        DOMAIN,
        issue,
        is_fixable=False,
        severity=IssueSeverity.WARNING,
        learn_more_url='https://esphome.io/components/api.html',
        translation_key='api_password_deprecated',
        translation_placeholders={'name': device_info.name},
    )


class ESPHomeManager:
    """Class to manage an ESPHome connection."""

    __slots__ = (
        'cli',
        'device_id',
        'domain_data',
        'entry',
        'entry_data',
        'hass',
        'host',
        'password',
        'reconnect_logic',
        'zeroconf_instance',
    )

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ESPHomeConfigEntry,
        host: str,
        password: str,
        cli: APIClient,
        zeroconf_instance: Any,
        domain_data: DomainData,
    ) -> None:
        """Initialize the esphome manager."""
        self.hass: HomeAssistant = hass
        self.host: str = host
        self.password: str = password
        self.entry: ESPHomeConfigEntry = entry
        self.cli: APIClient = cli
        self.device_id: Optional[str] = None
        self.domain_data: DomainData = domain_data
        self.reconnect_logic: Optional[ReconnectLogic] = None
        self.zeroconf_instance: Any = zeroconf_instance
        self.entry_data: RuntimeEntryData = entry.runtime_data

    async def on_stop(self, event: Event) -> None:
        """Cleanup the socket client on HA close."""
        await cleanup_instance(self.hass, self.entry)

    @property
    def services_issue(self) -> str:
        """Return the services issue name for this entry."""
        return f'service_calls_not_enabled-{self.entry.unique_id}'

    @callback
    def async_on_service_call(self, service: HomeassistantServiceCall) -> None:
        """Call service when user automation in ESPHome config is triggered."""
        hass: HomeAssistant = self.hass
        domain, service_name = service.service.split(".", 1)
        service_data: Dict[str, Any] = service.data.copy()
        if service.data_template:
            try:
                data_template: Dict[str, Template] = {key: Template(value, hass) for key, value in service.data_template.items()}
                service_data.update(template.render_complex(data_template, service.variables))
            except TemplateError as ex:
                _LOGGER.error("Error rendering data template %s for %s: %s", service.data_template, self.host, ex)
                return
        if service.is_event:
            device_id: Optional[str] = self.device_id
            if domain != DOMAIN:
                _LOGGER.error("Can only generate events under esphome domain! (%s)", self.host)
                return
            if service_name == "tag_scanned" and device_id is not None:
                tag_id: Any = service_data["tag_id"]
                hass.async_create_task(tag.async_scan_tag(hass, tag_id, device_id))
                return
            hass.bus.async_fire(service.service, {ATTR_DEVICE_ID: device_id, **service_data})
        elif self.entry.options.get(CONF_ALLOW_SERVICE_CALLS, DEFAULT_ALLOW_SERVICE_CALLS):
            hass.async_create_task(hass.services.async_call(domain, service_name, service_data, blocking=True))
        else:
            device_info: Optional[EsphomeDeviceInfo] = self.entry_data.device_info
            assert device_info is not None
            async_create_issue(
                hass,
                DOMAIN,
                self.services_issue,
                is_fixable=False,
                severity=IssueSeverity.WARNING,
                translation_key="service_calls_not_allowed",
                translation_placeholders={"name": device_info.friendly_name or device_info.name},
            )
            _LOGGER.error(
                "%s: Service call %s.%s: with data %s rejected; If you trust this device and want to allow access for it to make Home Assistant service calls, you can enable this functionality in the options flow",
                device_info.friendly_name or device_info.name,
                domain,
                service_name,
                service_data,
            )

    @callback
    def _send_home_assistant_state(self, entity_id: str, attribute: Optional[str], state: Optional[State]) -> None:
        """Forward Home Assistant states to ESPHome."""
        if state is None or (attribute and attribute not in state.attributes):
            return
        send_state: str = state.state
        if attribute:
            attr_val: Any = state.attributes[attribute]
            if isinstance(attr_val, bool):
                send_state = "on" if attr_val else "off"
            else:
                send_state = attr_val
        self.cli.send_home_assistant_state(entity_id, attribute, str(send_state))

    @callback
    def _send_home_assistant_state_event(self, attribute: Optional[str], event: Event) -> None:
        """Forward Home Assistant state updates to ESPHome."""
        event_data: Dict[str, Any] = event.data
        new_state: Optional[State] = event_data.get("new_state")
        old_state: Optional[State] = event_data.get("old_state")
        if new_state is None or old_state is None:
            return
        if not attribute and old_state.state == new_state.state or (attribute and old_state.attributes.get(attribute) == new_state.attributes.get(attribute)):
            return
        self._send_home_assistant_state(event.data["entity_id"], attribute, new_state)

    @callback
    def async_on_state_subscription(self, entity_id: str, attribute: Optional[str] = None) -> None:
        """Subscribe and forward states for requested entities."""
        self.entry_data.disconnect_callbacks.add(
            async_track_state_change_event(self.hass, [entity_id], partial(self._send_home_assistant_state_event, attribute))
        )
        self._send_home_assistant_state(entity_id, attribute, self.hass.states.get(entity_id))

    @callback
    def async_on_state_request(self, entity_id: str, attribute: Optional[str] = None) -> None:
        """Forward state for requested entity."""
        self._send_home_assistant_state(entity_id, attribute, self.hass.states.get(entity_id))

    async def on_connect(self) -> None:
        """Subscribe to states and list entities on successful API login."""
        try:
            await self._on_connnect()
        except APIConnectionError as err:
            _LOGGER.warning("Error getting setting up connection for %s: %s", self.host, err)
            await self.cli.disconnect()

    async def _on_connnect(self) -> None:
        """Subscribe to states and list entities on successful API login."""
        entry: ESPHomeConfigEntry = self.entry
        unique_id: Optional[str] = entry.unique_id
        entry_data: RuntimeEntryData = self.entry_data
        reconnect_logic: ReconnectLogic = self.reconnect_logic  # type: ignore
        assert reconnect_logic is not None, "Reconnect logic must be set"
        hass: HomeAssistant = self.hass
        cli: APIClient = self.cli
        stored_device_name: Optional[str] = entry.data.get(CONF_DEVICE_NAME)
        unique_id_is_mac_address: bool = bool(unique_id and ":" in unique_id)
        results: Tuple[
            EsphomeDeviceInfo, Tuple[List[EntityInfo], List[UserService]]
        ] = await asyncio.gather(
            create_eager_task(cli.device_info()), create_eager_task(cli.list_entities_services())
        )
        device_info: EsphomeDeviceInfo = results[0]
        entity_infos_services: Tuple[List[EntityInfo], List[UserService]] = results[1]
        entity_infos: List[EntityInfo] = entity_infos_services[0]
        services: List[UserService] = entity_infos_services[1]
        device_mac: str = format_mac(device_info.mac_address)
        mac_address_matches: bool = unique_id == device_mac
        if not mac_address_matches and (not unique_id_is_mac_address):
            hass.config_entries.async_update_entry(entry, unique_id=device_mac)
        if not mac_address_matches and unique_id_is_mac_address:
            _LOGGER.error(
                "Unexpected device found at %s; expected `%s` with mac address `%s`, found `%s` with mac address `%s`",
                self.host,
                stored_device_name,
                unique_id,
                device_info.name,
                device_mac,
            )
            await cli.disconnect()
            await reconnect_logic.stop()
            return
        if stored_device_name != device_info.name:
            hass.config_entries.async_update_entry(
                entry, data={**entry.data, CONF_DEVICE_NAME: device_info.name}
            )
        api_version: APIVersion = cli.api_version  # type: ignore
        assert api_version is not None, "API version must be set"
        entry_data.async_on_connect(device_info, api_version)
        if device_info.name:
            reconnect_logic.name = device_info.name
        self.device_id = _async_setup_device_registry(hass, entry, entry_data)
        entry_data.async_update_device_state()
        await entry_data.async_update_static_infos(hass, entry, entity_infos, device_info.mac_address)
        _setup_services(hass, entry_data, services)
        if device_info.bluetooth_proxy_feature_flags_compat(api_version):
            entry_data.disconnect_callbacks.add(
                async_connect_scanner(hass, entry_data, cli, device_info, self.device_id)
            )
        else:
            bluetooth.async_remove_scanner(hass, device_info.mac_address)
        if device_info.voice_assistant_feature_flags_compat(api_version) and Platform.ASSIST_SATELLITE not in entry_data.loaded_platforms:
            await self.hass.config_entries.async_forward_entry_setups(self.entry, [Platform.ASSIST_SATELLITE])
            entry_data.loaded_platforms.add(Platform.ASSIST_SATELLITE)
        cli.subscribe_states(entry_data.async_update_state)
        cli.subscribe_service_calls(self.async_on_service_call)
        cli.subscribe_home_assistant_states(self.async_on_state_subscription, self.async_on_state_request)
        entry_data.async_save_to_store()
        _async_check_firmware_version(hass, device_info, api_version)
        _async_check_using_api_password(hass, device_info, bool(self.password))

    async def on_disconnect(self, expected_disconnect: bool) -> None:
        """Run disconnect callbacks on API disconnect."""
        entry_data: RuntimeEntryData = self.entry_data
        hass: HomeAssistant = self.hass
        host: str = self.host
        name: str = entry_data.device_info.name if entry_data.device_info else host
        _LOGGER.debug("%s: %s disconnected (expected=%s), running disconnected callbacks", name, host, expected_disconnect)
        entry_data.async_on_disconnect()
        entry_data.expected_disconnect = expected_disconnect
        entry_data.stale_state = {(type(entity_state), key) for state_dict in entry_data.state.values() for key, entity_state in state_dict.items()}
        if not hass.is_stopping:
            entry_data.async_update_device_state()
        if Platform.ASSIST_SATELLITE in self.entry_data.loaded_platforms:
            await self.hass.config_entries.async_unload_platforms(self.entry, [Platform.ASSIST_SATELLITE])
            self.entry_data.loaded_platforms.remove(Platform.ASSIST_SATELLITE)

    async def on_connect_error(self, err: Exception) -> None:
        """Start reauth flow if appropriate connect error type."""
        if isinstance(err, (RequiresEncryptionAPIError, InvalidEncryptionKeyAPIError, InvalidAuthAPIError)):
            self.entry.async_start_reauth(self.hass)

    @callback
    def _async_handle_logging_changed(self, _event: Event) -> None:
        """Handle when the logging level changes."""
        self.cli.set_debug(_LOGGER.isEnabledFor(logging.DEBUG))

    async def async_start(self) -> None:
        """Start the esphome connection manager."""
        hass: HomeAssistant = self.hass
        entry: ESPHomeConfigEntry = self.entry
        entry_data: RuntimeEntryData = self.entry_data
        if entry.options.get(CONF_ALLOW_SERVICE_CALLS, DEFAULT_ALLOW_SERVICE_CALLS):
            async_delete_issue(hass, DOMAIN, self.services_issue)
        reconnect_logic: ReconnectLogic = ReconnectLogic(
            client=self.cli,
            on_connect=self.on_connect,
            on_disconnect=self.on_disconnect,
            zeroconf_instance=self.zeroconf_instance,
            name=entry.data.get(CONF_DEVICE_NAME, self.host),
            on_connect_error=self.on_connect_error,
        )
        self.reconnect_logic = reconnect_logic
        bus = hass.bus
        cleanups: Tuple[Callable[[], Any], ...] = (
            bus.async_listen(EVENT_HOMEASSISTANT_CLOSE, self.on_stop),
            bus.async_listen(EVENT_LOGGING_CHANGED, self._async_handle_logging_changed),
            reconnect_logic.stop_callback,
        )
        entry_data.cleanup_callbacks.extend(cleanups)
        infos, services = await entry_data.async_load_from_store()
        if entry.unique_id:
            await entry_data.async_update_static_infos(hass, entry, infos, entry.unique_id.upper())
        _setup_services(hass, entry_data, services)
        if entry_data.device_info is not None and entry_data.device_info.name:
            reconnect_logic.name = entry_data.device_info.name
            if entry.unique_id is None:
                hass.config_entries.async_update_entry(entry, unique_id=format_mac(entry_data.device_info.mac_address))
        await reconnect_logic.start()
        entry.async_on_unload(entry.add_update_listener(entry_data.async_update_listener))


@callback
def _async_setup_device_registry(
    hass: HomeAssistant, entry: ESPHomeConfigEntry, entry_data: RuntimeEntryData
) -> str:
    """Set up device registry feature for a particular config entry."""
    device_info: EsphomeDeviceInfo = entry_data.device_info  # type: ignore
    sw_version: str = device_info.esphome_version
    if device_info.compilation_time:
        sw_version += f" ({device_info.compilation_time})"
    configuration_url: Optional[str] = None
    if device_info.webserver_port > 0:
        entry_host: str = entry.data["host"]
        host: str = f"[{entry_host}]" if ":" in entry_host else entry_host
        configuration_url = f"http://{host}:{device_info.webserver_port}"
    elif (dashboard := async_get_dashboard(hass)) and dashboard.data and dashboard.data.get(device_info.name):
        configuration_url = f"homeassistant://hassio/ingress/{dashboard.addon_slug}"
    manufacturer: str = "espressif"
    if device_info.manufacturer:
        manufacturer = device_info.manufacturer
    model: str = device_info.model
    if device_info.project_name:
        project_name = device_info.project_name.split(".")
        manufacturer = project_name[0]
        model = project_name[1]
        sw_version = f"{device_info.project_version} (ESPHome {device_info.esphome_version})"
    suggested_area: Optional[str] = None
    if device_info.suggested_area:
        suggested_area = device_info.suggested_area
    device_registry = dr.async_get(hass)
    device_entry = device_registry.async_get_or_create(
        config_entry_id=entry.entry_id,
        configuration_url=configuration_url,
        connections={(dr.CONNECTION_NETWORK_MAC, device_info.mac_address)},
        name=entry_data.friendly_name,
        manufacturer=manufacturer,
        model=model,
        sw_version=sw_version,
        suggested_area=suggested_area,
    )
    return device_entry.id


class ServiceMetadata(NamedTuple):
    """Metadata for services."""

    description: Optional[str]
    validator: Any
    example: str
    selector: Dict[str, Any]


ARG_TYPE_METADATA: Dict[UserServiceArgType, ServiceMetadata] = {
    UserServiceArgType.BOOL: ServiceMetadata(validator=cv.boolean, example="False", selector={"boolean": None}, description=None),
    UserServiceArgType.INT: ServiceMetadata(validator=vol.Coerce(int), example="42", selector={"number": {CONF_MODE: "box"}}, description=None),
    UserServiceArgType.FLOAT: ServiceMetadata(validator=vol.Coerce(float), example="12.3", selector={"number": {CONF_MODE: "box", "step": 0.001}}, description=None),
    UserServiceArgType.STRING: ServiceMetadata(validator=cv.string, example="Example text", selector={"text": None}, description=None),
    UserServiceArgType.BOOL_ARRAY: ServiceMetadata(validator=[cv.boolean], description="A list of boolean values.", example="[True, False]", selector={"object": {}}),
    UserServiceArgType.INT_ARRAY: ServiceMetadata(validator=[vol.Coerce(int)], description="A list of integer values.", example="[42, 34]", selector={"object": {}}),
    UserServiceArgType.FLOAT_ARRAY: ServiceMetadata(validator=[vol.Coerce(float)], description="A list of floating point numbers.", example="[ 12.3, 34.5 ]", selector={"object": {}}),
    UserServiceArgType.STRING_ARRAY: ServiceMetadata(validator=[cv.string], description="A list of strings.", example="['Example text', 'Another example']", selector={"object": {}}),
}


@callback
def execute_service(entry_data: RuntimeEntryData, service: UserService, call: ServiceCall) -> None:
    """Execute a service on a node."""
    entry_data.client.execute_service(service, call.data)


def build_service_name(device_info: EsphomeDeviceInfo, service: UserService) -> str:
    """Build a service name for a node."""
    return f"{device_info.name.replace('-', '_')}_{service.name}"


@callback
def _async_register_service(
    hass: HomeAssistant, entry_data: RuntimeEntryData, device_info: EsphomeDeviceInfo, service: UserService
) -> None:
    """Register a service on a node."""
    service_name: str = build_service_name(device_info, service)
    schema: Dict[str, Any] = {}
    fields: Dict[str, Any] = {}
    for arg in service.args:
        if arg.type not in ARG_TYPE_METADATA:
            _LOGGER.error("Can't register service %s because %s is of unknown type %s", service_name, arg.name, arg.type)
            return
        metadata: ServiceMetadata = ARG_TYPE_METADATA[arg.type]
        schema[vol.Required(arg.name)] = metadata.validator
        fields[arg.name] = {
            "name": arg.name,
            "required": True,
            "description": metadata.description,
            "example": metadata.example,
            "selector": metadata.selector,
        }
    hass.services.async_register(DOMAIN, service_name, partial(execute_service, entry_data, service), vol.Schema(schema))
    async_set_service_schema(
        hass,
        DOMAIN,
        service_name,
        {"description": f"Calls the service {service.name} of the node {device_info.name}", "fields": fields},
    )


@callback
def _setup_services(hass: HomeAssistant, entry_data: RuntimeEntryData, services: List[UserService]) -> None:
    device_info: Optional[EsphomeDeviceInfo] = entry_data.device_info
    if device_info is None:
        return
    old_services: Dict[Any, UserService] = entry_data.services.copy()
    to_unregister: List[UserService] = []
    to_register: List[UserService] = []
    for service in services:
        if service.key in old_services:
            if (matching := old_services.pop(service.key)) != service:
                to_unregister.append(matching)
                to_register.append(service)
        else:
            to_register.append(service)
    to_unregister.extend(old_services.values())
    entry_data.services = {serv.key: serv for serv in services}
    for service in to_unregister:
        service_name = build_service_name(device_info, service)
        hass.services.async_remove(DOMAIN, service_name)
    for service in to_register:
        _async_register_service(hass, entry_data, device_info, service)


async def cleanup_instance(hass: HomeAssistant, entry: ESPHomeConfigEntry) -> RuntimeEntryData:
    """Cleanup the esphome client if it exists."""
    data: RuntimeEntryData = entry.runtime_data
    data.async_on_disconnect()
    for cleanup_callback in data.cleanup_callbacks:
        cleanup_callback()
    await data.async_cleanup()
    await data.client.disconnect()
    return data
