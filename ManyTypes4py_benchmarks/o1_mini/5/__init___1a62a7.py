"""Support for OwnTracks."""
from collections import defaultdict
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

from aiohttp import web
import voluptuous as vol
from homeassistant.components import cloud, mqtt, webhook
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_GPS_ACCURACY,
    ATTR_LATITUDE,
    ATTR_LONGITUDE,
    CONF_WEBHOOK_ID,
    Platform,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.typing import ConfigType
from homeassistant.setup import async_when_setup
from homeassistant.util.json import json_loads

from .config_flow import CONF_SECRET
from .const import DOMAIN
from .messages import async_handle_message, encrypt_message

_LOGGER = logging.getLogger(__name__)

CONF_MAX_GPS_ACCURACY = 'max_gps_accuracy'
CONF_WAYPOINT_IMPORT = 'waypoints'
CONF_WAYPOINT_WHITELIST = 'waypoint_whitelist'
CONF_MQTT_TOPIC = 'mqtt_topic'
CONF_REGION_MAPPING = 'region_mapping'
CONF_EVENTS_ONLY = 'events_only'
BEACON_DEV_ID = 'beacon'

PLATFORMS = [Platform.DEVICE_TRACKER]

DEFAULT_OWNTRACKS_TOPIC = 'owntracks/#'

CONFIG_SCHEMA = vol.All(
    cv.removed(CONF_WEBHOOK_ID),
    vol.Schema(
        {
            vol.Optional(DOMAIN, default={}): {
                vol.Optional(CONF_MAX_GPS_ACCURACY): vol.Coerce(float),
                vol.Optional(CONF_WAYPOINT_IMPORT, default=True): cv.boolean,
                vol.Optional(CONF_EVENTS_ONLY, default=False): cv.boolean,
                vol.Optional(CONF_MQTT_TOPIC, default=DEFAULT_OWNTRACKS_TOPIC): mqtt.valid_subscribe_topic,
                vol.Optional(CONF_WAYPOINT_WHITELIST): vol.All(cv.ensure_list, [cv.string]),
                vol.Optional(CONF_SECRET): vol.Any(
                    vol.Schema({vol.Optional(cv.string): cv.string}),
                    cv.string,
                ),
                vol.Optional(CONF_REGION_MAPPING, default={}): dict,
            }
        },
        extra=vol.ALLOW_EXTRA,
    ),
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Initialize OwnTracks component."""
    hass.data[DOMAIN] = {'config': config.get(DOMAIN, {}), 'devices': {}, 'unsub': None}
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OwnTracks entry."""
    config: Dict[str, Any] = hass.data[DOMAIN]['config']
    max_gps_accuracy: Optional[float] = config.get(CONF_MAX_GPS_ACCURACY)
    waypoint_import: bool = config.get(CONF_WAYPOINT_IMPORT)
    waypoint_whitelist: Optional[List[str]] = config.get(CONF_WAYPOINT_WHITELIST)
    secret: Union[str, Dict[str, str]] = config.get(CONF_SECRET) or entry.data.get(CONF_SECRET)
    region_mapping: Dict[str, Any] = config.get(CONF_REGION_MAPPING)
    events_only: bool = config.get(CONF_EVENTS_ONLY)
    mqtt_topic: str = config.get(CONF_MQTT_TOPIC)
    context = OwnTracksContext(
        hass,
        secret,
        max_gps_accuracy,
        waypoint_import,
        waypoint_whitelist,
        region_mapping,
        events_only,
        mqtt_topic,
    )
    webhook_id: Optional[str] = config.get(CONF_WEBHOOK_ID) or entry.data.get(CONF_WEBHOOK_ID)
    hass.data[DOMAIN]['context'] = context
    async_when_setup(hass, 'mqtt', async_connect_mqtt)
    webhook.async_register(hass, DOMAIN, 'OwnTracks', webhook_id, handle_webhook)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    hass.data[DOMAIN]['unsub'] = async_dispatcher_connect(hass, DOMAIN, async_handle_message)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload an OwnTracks config entry."""
    webhook.async_unregister(hass, entry.data.get(CONF_WEBHOOK_ID))
    unload_ok: bool = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if hass.data[DOMAIN].get('unsub'):
        hass.data[DOMAIN]['unsub']()
    return unload_ok


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Remove an OwnTracks config entry."""
    if not entry.data.get('cloudhook'):
        return
    await cloud.async_delete_cloudhook(hass, entry.data.get(CONF_WEBHOOK_ID))


async def async_connect_mqtt(hass: HomeAssistant, component: Any) -> bool:
    """Subscribe to MQTT topic."""
    context: 'OwnTracksContext' = hass.data[DOMAIN]['context']

    @callback
    def async_handle_mqtt_message(msg: mqtt.PublishMessage) -> None:
        """Handle incoming OwnTracks message."""
        try:
            message: Dict[str, Any] = json_loads(msg.payload)
        except ValueError:
            _LOGGER.error('Unable to parse payload as JSON: %s', msg.payload)
            return
        message['topic'] = msg.topic
        async_dispatcher_send(hass, DOMAIN, hass, context, message)

    await mqtt.async_subscribe(hass, context.mqtt_topic, async_handle_mqtt_message, 1)
    return True


async def handle_webhook(
    hass: HomeAssistant, webhook_id: str, request: web.Request
) -> web.Response:
    """Handle webhook callback.

    iOS sets the "topic" as part of the payload.
    Android does not set a topic but adds headers to the request.
    """
    context: 'OwnTracksContext' = hass.data[DOMAIN]['context']
    topic_base: str = re.sub('/#$', '', context.mqtt_topic)
    try:
        message: Dict[str, Any] = await request.json()
    except ValueError:
        _LOGGER.warning('Received invalid JSON from OwnTracks')
        return web.json_response([])
    if 'topic' not in message:
        headers: web.Headers = request.headers
        user: Optional[str] = headers.get('X-Limit-U')
        device: str = headers.get('X-Limit-D', user) if user else ''
        if user:
            message['topic'] = f'{topic_base}/{user}/{device}'
        elif message.get('_type') != 'encrypted':
            _LOGGER.warning(
                'No topic or user found in message. If on Android, set a username in Connection -> Identification'
            )
            return web.json_response([])
    async_dispatcher_send(hass, DOMAIN, hass, context, message)
    response = [
        {
            '_type': 'location',
            'lat': person.attributes[ATTR_LATITUDE],
            'lon': person.attributes[ATTR_LONGITUDE],
            'tid': ''.join((p[0] for p in person.name.split(' ')[:2])),
            'tst': int(person.last_updated.timestamp()),
        }
        for person in hass.states.async_all('person')
        if ATTR_LATITUDE in person.attributes and ATTR_LONGITUDE in person.attributes
    ]
    if message.get('_type') == 'encrypted' and context.secret:
        return web.json_response(
            {
                '_type': 'encrypted',
                'data': encrypt_message(
                    context.secret,
                    message['topic'],
                    json.dumps(response),
                ),
            }
        )
    return web.json_response(response)


class OwnTracksContext:
    """Hold the current OwnTracks context."""

    def __init__(
        self,
        hass: HomeAssistant,
        secret: Union[str, Dict[str, str]],
        max_gps_accuracy: Optional[float],
        import_waypoints: bool,
        waypoint_whitelist: Optional[List[str]],
        region_mapping: Dict[str, Any],
        events_only: bool,
        mqtt_topic: str,
    ) -> None:
        """Initialize an OwnTracks context."""
        self.hass: HomeAssistant = hass
        self.secret: Union[str, Dict[str, str]] = secret
        self.max_gps_accuracy: Optional[float] = max_gps_accuracy
        self.mobile_beacons_active: defaultdict[str, set[str]] = defaultdict(set)
        self.regions_entered: defaultdict[str, List[Any]] = defaultdict(list)
        self.import_waypoints: bool = import_waypoints
        self.waypoint_whitelist: Optional[List[str]] = waypoint_whitelist
        self.region_mapping: Dict[str, Any] = region_mapping
        self.events_only: bool = events_only
        self.mqtt_topic: str = mqtt_topic
        self._pending_msg: List[Dict[str, Any]] = []

    @callback
    def async_valid_accuracy(self, message: Dict[str, Any]) -> bool:
        """Check if we should ignore this message."""
        acc = message.get('acc')
        if acc is None:
            return False
        try:
            acc = float(acc)
        except ValueError:
            return False
        if acc == 0:
            _LOGGER.warning(
                'Ignoring %s update because GPS accuracy is zero: %s',
                message.get('_type', 'unknown'),
                message,
            )
            return False
        if self.max_gps_accuracy is not None and acc > self.max_gps_accuracy:
            _LOGGER.warning(
                'Ignoring %s update because expected GPS accuracy %s is not met: %s',
                message.get('_type', 'unknown'),
                self.max_gps_accuracy,
                message,
            )
            return False
        return True

    @callback
    def set_async_see(self, func: Callable[..., None]) -> None:
        """Set a new async_see function."""
        self.async_see = func
        for msg in self._pending_msg:
            func(**msg)
        self._pending_msg.clear()

    @callback
    def async_see(self, **data: Any) -> None:
        """Send a see message to the device tracker."""
        self._pending_msg.append(data)

    @callback
    def async_see_beacons(
        self, hass: HomeAssistant, dev_id: str, kwargs_param: Dict[str, Any]
    ) -> None:
        """Set active beacons to the current location."""
        kwargs = kwargs_param.copy()
        device_tracker_state = hass.states.get(f'device_tracker.{dev_id}')
        if device_tracker_state is not None:
            acc = device_tracker_state.attributes.get(ATTR_GPS_ACCURACY)
            lat = device_tracker_state.attributes.get(ATTR_LATITUDE)
            lon = device_tracker_state.attributes.get(ATTR_LONGITUDE)
            if lat is not None and lon is not None:
                kwargs['gps'] = (lat, lon)
                kwargs['gps_accuracy'] = acc
            else:
                kwargs['gps'] = None
                kwargs['gps_accuracy'] = None
        kwargs.pop('battery', None)
        for beacon in self.mobile_beacons_active[dev_id]:
            kwargs['dev_id'] = f'{BEACON_DEV_ID}_{beacon}'
            kwargs['host_name'] = beacon
            self.async_see(**kwargs)
