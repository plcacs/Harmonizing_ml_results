from typing import Awaitable, Optional

class EmulatedRoku:
    """Manages an emulated_roku server."""

    def __init__(self, hass: 'homeassistant.core.HomeAssistant', name: str, host_ip: str, listen_port: int, advertise_ip: str, advertise_port: int, upnp_bind_multicast: bool):
        """Initialize the properties."""
        self.hass: 'homeassistant.core.HomeAssistant' = hass
        self.roku_usn: str = name
        self.host_ip: str = host_ip
        self.listen_port: int = listen_port
        self.advertise_port: int = advertise_port
        self.advertise_ip: str = advertise_ip
        self.bind_multicast: bool = upnp_bind_multicast
        self._api_server: Optional['emulated_roku.EmulatedRokuServer'] = None
        self._unsub_start_listener: Optional[Awaitable] = None
        self._unsub_stop_listener: Optional[Awaitable] = None

    async def setup(self) -> Awaitable[bool]:
        """Start the emulated_roku server."""

        class EventCommandHandler(EmulatedRokuCommandHandler):
            """emulated_roku command handler to turn commands into events."""

            def __init__(self, hass: 'homeassistant.core.HomeAssistant'):
                self.hass: 'homeassistant.core.HomeAssistant' = hass

            # ...

    async def unload(self) -> Awaitable[bool]:
        """Unload the emulated_roku server."""
        LOGGER.debug('Unloading emulated_roku %s', self.roku_usn)
        if self._unsub_start_listener:
            self._unsub_start_listener()
            self._unsub_start_listener = None
        if self._unsub_stop_listener:
            self._unsub_stop_listener()
            self._unsub_stop_listener = None
        await self._api_server.close()
        return True
