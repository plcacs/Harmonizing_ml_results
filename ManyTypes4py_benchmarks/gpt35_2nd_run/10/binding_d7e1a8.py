import logging
from emulated_roku import EmulatedRokuCommandHandler, EmulatedRokuServer
from homeassistant.const import EVENT_HOMEASSISTANT_START, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import CoreState, EventOrigin
from typing import Any, Dict, Optional

LOGGER: logging.Logger = logging.getLogger(__package__)
EVENT_ROKU_COMMAND: str = 'roku_command'
ATTR_COMMAND_TYPE: str = 'type'
ATTR_SOURCE_NAME: str = 'source_name'
ATTR_KEY: str = 'key'
ATTR_APP_ID: str = 'app_id'
ROKU_COMMAND_KEYDOWN: str = 'keydown'
ROKU_COMMAND_KEYUP: str = 'keyup'
ROKU_COMMAND_KEYPRESS: str = 'keypress'
ROKU_COMMAND_LAUNCH: str = 'launch'

class EmulatedRoku:
    def __init__(self, hass: Any, name: str, host_ip: str, listen_port: int, advertise_ip: str, advertise_port: int, upnp_bind_multicast: bool) -> None:
        self.hass: Any = hass
        self.roku_usn: str = name
        self.host_ip: str = host_ip
        self.listen_port: int = listen_port
        self.advertise_port: int = advertise_port
        self.advertise_ip: str = advertise_ip
        self.bind_multicast: bool = upnp_bind_multicast
        self._api_server: Optional[EmulatedRokuServer] = None
        self._unsub_start_listener: Optional[Any] = None
        self._unsub_stop_listener: Optional[Any] = None

    async def setup(self) -> bool:
        class EventCommandHandler(EmulatedRokuCommandHandler):
            def __init__(self, hass: Any) -> None:
                self.hass: Any = hass

            def on_keydown(self, roku_usn: str, key: str) -> None:
                self.hass.bus.async_fire(EVENT_ROKU_COMMAND, {ATTR_SOURCE_NAME: roku_usn, ATTR_COMMAND_TYPE: ROKU_COMMAND_KEYDOWN, ATTR_KEY: key}, EventOrigin.local)

            def on_keyup(self, roku_usn: str, key: str) -> None:
                self.hass.bus.async_fire(EVENT_ROKU_COMMAND, {ATTR_SOURCE_NAME: roku_usn, ATTR_COMMAND_TYPE: ROKU_COMMAND_KEYUP, ATTR_KEY: key}, EventOrigin.local)

            def on_keypress(self, roku_usn: str, key: str) -> None:
                self.hass.bus.async_fire(EVENT_ROKU_COMMAND, {ATTR_SOURCE_NAME: roku_usn, ATTR_COMMAND_TYPE: ROKU_COMMAND_KEYPRESS, ATTR_KEY: key}, EventOrigin.local)

            def launch(self, roku_usn: str, app_id: str) -> None:
                self.hass.bus.async_fire(EVENT_ROKU_COMMAND, {ATTR_SOURCE_NAME: roku_usn, ATTR_COMMAND_TYPE: ROKU_COMMAND_LAUNCH, ATTR_APP_ID: app_id}, EventOrigin.local)

        LOGGER.debug('Initializing emulated_roku %s on %s:%s', self.roku_usn, self.host_ip, self.listen_port)
        handler: EventCommandHandler = EventCommandHandler(self.hass)
        self._api_server = EmulatedRokuServer(self.hass.loop, handler, self.roku_usn, self.host_ip, self.listen_port, advertise_ip=self.advertise_ip, advertise_port=self.advertise_port, bind_multicast=self.bind_multicast)

        async def emulated_roku_stop(event: Any) -> None:
            LOGGER.debug('Stopping emulated_roku %s', self.roku_usn)
            self._unsub_stop_listener = None
            await self._api_server.close()

        async def emulated_roku_start(event: Any) -> None:
            try:
                LOGGER.debug('Starting emulated_roku %s', self.roku_usn)
                self._unsub_start_listener = None
                await self._api_server.start()
            except OSError:
                LOGGER.exception('Failed to start Emulated Roku %s on %s:%s', self.roku_usn, self.host_ip, self.listen_port)
                await emulated_roku_stop(None)
            else:
                self._unsub_stop_listener = self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, emulated_roku_stop)

        if self.hass.state is CoreState.running:
            await emulated_roku_start(None)
        else:
            self._unsub_start_listener = self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, emulated_roku_start)
        return True

    async def unload(self) -> bool:
        LOGGER.debug('Unloading emulated_roku %s', self.roku_usn)
        if self._unsub_start_listener:
            self._unsub_start_listener()
            self._unsub_start_listener = None
        if self._unsub_stop_listener:
            self._unsub_stop_listener()
            self._unsub_stop_listener = None
        await self._api_server.close()
        return True
