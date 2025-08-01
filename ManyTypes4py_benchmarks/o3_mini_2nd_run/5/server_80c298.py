from __future__ import annotations
from copy import copy
import logging
import ssl
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from plexapi.client import PlexClient
from plexapi.exceptions import BadRequest, NotFound, Unauthorized
import plexapi.myplex
import plexapi.playqueue
import plexapi.server
from requests import Session
import requests.exceptions
from homeassistant.components.media_player import DOMAIN as MP_DOMAIN, MediaType
from homeassistant.const import CONF_CLIENT_ID, CONF_TOKEN, CONF_URL, CONF_VERIFY_SSL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import (
    CONF_IGNORE_NEW_SHARED_USERS,
    CONF_IGNORE_PLEX_WEB_CLIENTS,
    CONF_MONITORED_USERS,
    CONF_SERVER,
    CONF_SERVER_IDENTIFIER,
    CONF_USE_EPISODE_ART,
    DEBOUNCE_TIMEOUT,
    DEFAULT_VERIFY_SSL,
    GDM_DEBOUNCER,
    GDM_SCANNER,
    PLAYER_SOURCE,
    PLEX_NEW_MP_SIGNAL,
    PLEX_UPDATE_MEDIA_PLAYER_SESSION_SIGNAL,
    PLEX_UPDATE_MEDIA_PLAYER_SIGNAL,
    PLEX_UPDATE_SENSOR_SIGNAL,
    PLEXTV_THROTTLE,
    X_PLEX_DEVICE_NAME,
    X_PLEX_PLATFORM,
    X_PLEX_PRODUCT,
    X_PLEX_VERSION,
)
from .errors import MediaNotFound, NoServersFound, ServerNotSpecified, ShouldUpdateConfigEntry
from .helpers import get_plex_data
from .media_search import search_media
from .models import PlexSession

_LOGGER = logging.getLogger(__name__)
plexapi.X_PLEX_DEVICE_NAME = X_PLEX_DEVICE_NAME
plexapi.X_PLEX_PLATFORM = X_PLEX_PLATFORM
plexapi.X_PLEX_PRODUCT = X_PLEX_PRODUCT
plexapi.X_PLEX_VERSION = X_PLEX_VERSION


class PlexServer:
    """Manages a single Plex server connection."""

    def __init__(
        self,
        hass: HomeAssistant,
        server_config: Dict[str, Any],
        known_server_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> None:
        """Initialize a Plex server instance."""
        self.hass: HomeAssistant = hass
        self.entry_id: Optional[str] = entry_id
        self.active_sessions: Dict[str, PlexSession] = {}
        self._plex_account: Optional[plexapi.myplex.MyPlexAccount] = None
        self._plex_server: Optional[plexapi.server.PlexServer] = None
        self._created_clients: Set[str] = set()
        self._known_clients: Set[str] = set()
        self._known_idle: Set[str] = set()
        self._url: Optional[str] = server_config.get(CONF_URL)
        self._token: Optional[str] = server_config.get(CONF_TOKEN)
        self._server_name: Optional[str] = server_config.get(CONF_SERVER)
        self._verify_ssl: bool = server_config.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL)
        self._server_id: Optional[str] = known_server_id or server_config.get(CONF_SERVER_IDENTIFIER)
        self.options: Dict[str, Any] = options if options is not None else {}
        self.server_choice: Optional[str] = None
        self._accounts: List[Any] = []
        self._owner_username: Optional[str] = None
        self._plextv_clients: Optional[List[Any]] = None
        self._plextv_client_timestamp: float = 0
        self._client_device_cache: Dict[str, Optional[PlexClient]] = {}
        self._use_plex_tv: bool = self._token is not None
        self._version: Optional[str] = None
        self.async_update_platforms: Callable[[], None] = Debouncer(
            hass,
            _LOGGER,
            cooldown=DEBOUNCE_TIMEOUT,
            immediate=True,
            function=self._async_update_platforms,
            background=True,
        ).async_call
        self.thumbnail_cache: Dict[Any, Any] = {}
        if CONF_CLIENT_ID in server_config:
            plexapi.X_PLEX_IDENTIFIER = server_config[CONF_CLIENT_ID]
        plexapi.myplex.BASE_HEADERS = plexapi.reset_base_headers()
        plexapi.server.BASE_HEADERS = plexapi.reset_base_headers()

    @property
    def account(self) -> plexapi.myplex.MyPlexAccount:
        """Return a MyPlexAccount instance."""
        if not self._plex_account and self._use_plex_tv:
            try:
                self._plex_account = plexapi.myplex.MyPlexAccount(token=self._token)
            except (BadRequest, Unauthorized):
                self._use_plex_tv = False
                _LOGGER.error('Not authorized to access plex.tv with provided token')
                raise
        # Type checker assumes _plex_account is not None here.
        return self._plex_account  # type: ignore

    def plextv_clients(self) -> List[Any]:
        """Return available clients linked to Plex account."""
        if self.account is None:
            return []
        now: float = time.time()
        if now - self._plextv_client_timestamp > PLEXTV_THROTTLE:
            self._plextv_client_timestamp = now
            self._plextv_clients = [
                x for x in self.account.resources() if 'player' in x.provides and x.presence and x.publicAddressMatches
            ]
            _LOGGER.debug('Current available clients from plex.tv: %s', self._plextv_clients)
        return self._plextv_clients if self._plextv_clients is not None else []

    def connect(self) -> None:
        """Connect to a Plex server directly, obtaining direct URL if necessary."""
        config_entry_update_needed: bool = False

        def _connect_with_token() -> None:
            all_servers: List[Any] = [x for x in self.account.resources() if 'server' in x.provides]
            available_servers: List[Tuple[str, Any, Any]] = [(x.name, x.clientIdentifier, x.sourceTitle) for x in all_servers]
            if not all_servers:
                raise NoServersFound
            if not self._server_id and len(all_servers) > 1:
                raise ServerNotSpecified(available_servers)
            self.server_choice = self._server_id or available_servers[0][1]
            self._plex_server = next((x.connect(timeout=10) for x in all_servers if x.clientIdentifier == self.server_choice), None)
            if self._plex_server is None:
                raise NoServersFound

        def _connect_with_url() -> None:
            session: Optional[Session] = None
            if self._url and self._url.startswith('https') and (not self._verify_ssl):
                session = Session()
                session.verify = False
            if self._url is not None:
                self._plex_server = plexapi.server.PlexServer(self._url, self._token, session)

        def _update_plexdirect_hostname() -> bool:
            matching_servers: List[str] = [x.name for x in self.account.resources() if x.clientIdentifier == self._server_id]
            if matching_servers:
                self._plex_server = self.account.resource(matching_servers[0]).connect(timeout=10)
                return True
            _LOGGER.error('Attempt to update plex.direct hostname failed')
            return False

        if self._url:
            try:
                _connect_with_url()
            except requests.exceptions.SSLError as error:
                while error and (not isinstance(error, ssl.SSLCertVerificationError)):
                    error = error.__context__  # type: ignore
                if isinstance(error, ssl.SSLCertVerificationError):
                    domain: str = urlparse(self._url).netloc.split(':')[0]
                    if domain.endswith('plex.direct') and error.args[0].startswith(f"hostname '{domain}' doesn't match"):
                        _LOGGER.warning("Plex SSL certificate's hostname changed, updating")
                        if _update_plexdirect_hostname():
                            config_entry_update_needed = True
                        else:
                            raise Unauthorized('New certificate cannot be validated with provided token')
                    else:
                        raise
                else:
                    raise
        else:
            _connect_with_token()
        try:
            system_accounts: Any = self._plex_server.systemAccounts()  # type: ignore
            shared_users: List[Any] = self.account.users() if self.account else []
        except Unauthorized:
            _LOGGER.warning('Plex account has limited permissions, shared account filtering will not be available')
        else:
            self._accounts = []
            for user in shared_users:
                for shared_server in user.servers:
                    if shared_server.machineIdentifier == self.machine_identifier:
                        self._accounts.append(user.title)
            _LOGGER.debug('Linked accounts: %s', self.accounts)
            owner_account: Optional[str] = next((account.name for account in system_accounts if account.accountID == 1), None)  # type: ignore
            if owner_account:
                self._owner_username = owner_account
                self._accounts.append(owner_account)
                _LOGGER.debug("Server owner found: '%s'", self._owner_username)
        self._version = self._plex_server.version if self._plex_server is not None else None
        if config_entry_update_needed:
            raise ShouldUpdateConfigEntry

    @callback
    def async_refresh_entity(self, machine_identifier: str, device: Any, session: Any, source: Any) -> None:
        """Forward refresh dispatch to media_player."""
        unique_id: str = f'{self.machine_identifier}:{machine_identifier}'
        _LOGGER.debug('Refreshing %s', unique_id)
        async_dispatcher_send(
            self.hass, PLEX_UPDATE_MEDIA_PLAYER_SIGNAL.format(unique_id), device, session, source
        )

    async def async_update_session(self, payload: Dict[str, Any]) -> None:
        """Process a session payload received from a websocket callback."""
        session_payload: Dict[str, Any] = payload['PlaySessionStateNotification'][0]
        state: str = session_payload.get('state', '')
        if state == 'buffering':
            return
        session_key: int = int(session_payload['sessionKey'])
        offset: int = int(session_payload['viewOffset'])
        rating_key: int = int(session_payload['ratingKey'])
        unique_id: Optional[str] = None
        active_session: Optional[PlexSession] = None
        for uid, session in self.active_sessions.items():
            if session.session_key == session_key:
                unique_id = uid
                active_session = session
                break
        if not active_session:
            await self.async_update_platforms()
            return
        if state == 'stopped':
            self.active_sessions.pop(unique_id, None)
        else:
            active_session.state = state
            active_session.media_position = offset

        def update_with_new_media() -> None:
            """Update an existing session with new media details."""
            media: Any = self.fetch_item(rating_key)
            active_session.update_media(media)

        if active_session.media_content_id != rating_key and state in ('playing', 'paused'):
            await self.hass.async_add_executor_job(update_with_new_media)
        async_dispatcher_send(
            self.hass, PLEX_UPDATE_MEDIA_PLAYER_SESSION_SIGNAL.format(unique_id), state
        )
        async_dispatcher_send(
            self.hass, PLEX_UPDATE_SENSOR_SIGNAL.format(self.machine_identifier)
        )

    def _fetch_platform_data(self) -> Tuple[List[Any], List[Any], List[Any]]:
        """Fetch all data from the Plex server in a single method."""
        return (self._plex_server.clients(), self._plex_server.sessions(), self.plextv_clients())  # type: ignore

    async def _async_update_platforms(self) -> None:
        """Update the platform entities."""
        _LOGGER.debug('Updating devices')
        await get_plex_data(self.hass)[GDM_DEBOUNCER]()
        available_clients: Dict[str, Dict[str, Any]] = {}
        ignored_clients: Set[str] = set()
        new_clients: Set[str] = set()
        monitored_users: Set[str] = self.accounts
        known_accounts: Set[str] = set(self.option_monitored_users)
        if known_accounts:
            monitored_users = {user for user in self.option_monitored_users if self.option_monitored_users[user]['enabled']}
        if not self.option_ignore_new_shared_users:
            for new_user in self.accounts - known_accounts:
                monitored_users.add(new_user)
        try:
            devices, sessions, plextv_clients = await self.hass.async_add_executor_job(self._fetch_platform_data)
        except plexapi.exceptions.Unauthorized:
            _LOGGER.debug("Token has expired for '%s', reloading integration", self.friendly_name)
            await self.hass.config_entries.async_reload(self.entry_id)  # type: ignore
            return
        except (plexapi.exceptions.BadRequest, requests.exceptions.RequestException) as ex:
            _LOGGER.error('Could not connect to Plex server: %s (%s)', self.friendly_name, ex)
            return

        def process_device(source: str, device: Any) -> None:
            self._known_idle.discard(device.machineIdentifier)
            available_clients.setdefault(device.machineIdentifier, {'device': device})
            available_clients[device.machineIdentifier].setdefault(PLAYER_SOURCE, source)
            if (
                device.machineIdentifier not in ignored_clients
                and self.option_ignore_plexweb_clients
                and (device.product == 'Plex Web')
            ):
                ignored_clients.add(device.machineIdentifier)
                if device.machineIdentifier not in self._known_clients:
                    _LOGGER.debug('Ignoring %s %s: %s', 'Plex Web', source, device.machineIdentifier)
                return
            if device.machineIdentifier not in (self._created_clients | ignored_clients | new_clients):
                new_clients.add(device.machineIdentifier)
                _LOGGER.debug('New %s from %s: %s', device.product, source, device.machineIdentifier)

        def connect_to_client(source: str, baseurl: str, machine_identifier: str, name: str = 'Unknown') -> None:
            """Connect to a Plex client and return a PlexClient instance."""
            try:
                client: PlexClient = PlexClient(
                    server=self._plex_server,  # type: ignore
                    baseurl=baseurl,
                    identifier=machine_identifier,
                    token=self._plex_server.createToken(),  # type: ignore
                )
            except (NotFound, requests.exceptions.ConnectionError):
                _LOGGER.error('Direct client connection failed, will try again: %s (%s)', name, baseurl)
            except Unauthorized:
                _LOGGER.error('Direct client connection unauthorized, ignoring: %s (%s)', name, baseurl)
                self._client_device_cache[machine_identifier] = None
            else:
                self._client_device_cache[client.machineIdentifier] = client
                process_device(source, client)

        def connect_to_resource(resource: Any) -> None:
            """Connect to a plex.tv resource and return a Plex client."""
            try:
                client: PlexClient = resource.connect(timeout=3)
                _LOGGER.debug('Resource connection successful to plex.tv: %s', client)
            except NotFound:
                _LOGGER.info('Resource connection failed to plex.tv: %s', resource.name)
            else:
                client.proxyThroughServer(value=False, server=self._plex_server)  # type: ignore
                self._client_device_cache[client.machineIdentifier] = client
                process_device('plex.tv', client)

        def connect_new_clients() -> None:
            """Create connections to newly discovered clients."""
            for gdm_entry in get_plex_data(self.hass)[GDM_SCANNER].entries:
                machine_identifier: str = gdm_entry['data']['Resource-Identifier']
                if machine_identifier in self._client_device_cache:
                    client: Optional[PlexClient] = self._client_device_cache[machine_identifier]
                    if client is not None:
                        process_device('GDM', client)
                elif machine_identifier not in available_clients:
                    baseurl: str = f"http://{gdm_entry['from'][0]}:{gdm_entry['data']['Port']}"
                    name: str = gdm_entry['data']['Name']
                    connect_to_client('GDM', baseurl, machine_identifier, name)
            for plextv_client in plextv_clients:
                if plextv_client.clientIdentifier in self._client_device_cache:
                    client = self._client_device_cache[plextv_client.clientIdentifier]
                    if client is not None:
                        process_device('plex.tv', client)
                elif plextv_client.clientIdentifier not in available_clients:
                    connect_to_resource(plextv_client)

        def process_sessions() -> None:
            live_session_keys: Set[int] = {x.sessionKey for x in sessions}
            for unique_id, session in list(self.active_sessions.items()):
                if session.session_key not in live_session_keys:
                    _LOGGER.debug('Purging unknown session: %s', session.session_key)
                    self.active_sessions.pop(unique_id)
            for session in sessions:
                if session.TYPE == 'photo':
                    _LOGGER.debug('Photo session detected, skipping: %s', session)
                    continue
                session_username: Optional[str] = next(iter(session.usernames), None)
                player: Any = session.player
                unique_id: str = f'{self.machine_identifier}:{player.machineIdentifier}'
                if unique_id not in self.active_sessions:
                    _LOGGER.debug('Creating new Plex session: %s', session)
                    self.active_sessions[unique_id] = PlexSession(self, session)
                if session_username and session_username not in monitored_users:
                    ignored_clients.add(player.machineIdentifier)
                    _LOGGER.debug("Ignoring %s client owned by '%s'", player.product, session_username)
                    continue
                process_device('session', player)
                available_clients[player.machineIdentifier]['session'] = self.active_sessions[unique_id]

        for device in devices:
            process_device('PMS', device)

        def sync_tasks() -> None:
            connect_new_clients()
            process_sessions()
        await self.hass.async_add_executor_job(sync_tasks)
        new_entity_configs: List[Dict[str, Any]] = []
        for client_id, client_data in available_clients.items():
            if client_id in ignored_clients:
                continue
            if client_id in new_clients:
                new_entity_configs.append(client_data)
                self._created_clients.add(client_id)
            else:
                self.async_refresh_entity(client_id, client_data['device'], client_data.get('session'), client_data.get(PLAYER_SOURCE))
        self._known_clients.update(new_clients | ignored_clients)
        idle_clients: Set[str] = (self._known_clients - self._known_idle - ignored_clients).difference(available_clients)
        for client_id in idle_clients:
            self.async_refresh_entity(client_id, None, None, None)
            self._known_idle.add(client_id)
            self._client_device_cache.pop(client_id, None)
        if new_entity_configs:
            async_dispatcher_send(self.hass, PLEX_NEW_MP_SIGNAL.format(self.machine_identifier), new_entity_configs)
        async_dispatcher_send(self.hass, PLEX_UPDATE_SENSOR_SIGNAL.format(self.machine_identifier))

    @property
    def plex_server(self) -> plexapi.server.PlexServer:
        """Return the plexapi PlexServer instance."""
        return self._plex_server  # type: ignore

    @property
    def has_token(self) -> bool:
        """Return if a token is used to connect to this Plex server."""
        return self._token is not None

    @property
    def accounts(self) -> Set[str]:
        """Return accounts associated with the Plex server."""
        return set(self._accounts)

    @property
    def owner(self) -> Optional[str]:
        """Return the Plex server owner username."""
        return self._owner_username

    @property
    def version(self) -> Optional[str]:
        """Return the version of the Plex server."""
        return self._version

    @property
    def friendly_name(self) -> str:
        """Return name of connected Plex server."""
        return self._plex_server.friendlyName  # type: ignore

    @property
    def machine_identifier(self) -> str:
        """Return unique identifier of connected Plex server."""
        return self._plex_server.machineIdentifier  # type: ignore

    @property
    def url_in_use(self) -> str:
        """Return URL used for connected Plex server."""
        return self._plex_server._baseurl  # type: ignore

    @property
    def option_ignore_new_shared_users(self) -> bool:
        """Return ignore_new_shared_users option."""
        return self.options.get(MP_DOMAIN, {}).get(CONF_IGNORE_NEW_SHARED_USERS, False)

    @property
    def option_use_episode_art(self) -> bool:
        """Return use_episode_art option."""
        return self.options.get(MP_DOMAIN, {}).get(CONF_USE_EPISODE_ART, False)

    @property
    def option_monitored_users(self) -> Dict[str, Any]:
        """Return dict of monitored users option."""
        return self.options.get(MP_DOMAIN, {}).get(CONF_MONITORED_USERS, {})

    @property
    def option_ignore_plexweb_clients(self) -> bool:
        """Return ignore_plex_web_clients option."""
        return self.options.get(MP_DOMAIN, {}).get(CONF_IGNORE_PLEX_WEB_CLIENTS, False)

    @property
    def library(self) -> Any:
        """Return library attribute from server object."""
        return self._plex_server.library  # type: ignore

    def playlist(self, title: str) -> Any:
        """Return playlist from server object."""
        return self._plex_server.playlist(title)  # type: ignore

    def playlists(self) -> Any:
        """Return available playlists from server object."""
        return self._plex_server.playlists()  # type: ignore

    def create_playqueue(self, media: Any, **kwargs: Any) -> Any:
        """Create playqueue on Plex server."""
        return plexapi.playqueue.PlayQueue.create(self._plex_server, media, **kwargs)  # type: ignore

    def create_station_playqueue(self, key: Any) -> Any:
        """Create playqueue on Plex server using a radio station key."""
        return plexapi.playqueue.PlayQueue.fromStationKey(self._plex_server, key)  # type: ignore

    def get_playqueue(self, playqueue_id: Any) -> Any:
        """Retrieve existing playqueue from Plex server."""
        return plexapi.playqueue.PlayQueue.get(self._plex_server, playqueue_id)  # type: ignore

    def fetch_item(self, item: Any) -> Any:
        """Fetch item from Plex server."""
        return self._plex_server.fetchItem(item)  # type: ignore

    def lookup_media(self, media_type: str, **kwargs: Any) -> Any:
        """Lookup a piece of media."""
        media_type = media_type.lower()
        if isinstance(kwargs.get('plex_key'), int):
            key: int = kwargs['plex_key']
            try:
                return self.fetch_item(key)
            except NotFound as err:
                raise MediaNotFound(f'Media for key {key} not found') from err
        if media_type == MediaType.PLAYLIST:
            try:
                playlist_name: str = kwargs['playlist_name']
                return self.playlist(playlist_name)
            except KeyError as err:
                raise MediaNotFound("Must specify 'playlist_name' for this search") from err
            except NotFound as err:
                raise MediaNotFound(f"Playlist '{playlist_name}' not found") from err
        try:
            library_name: str = kwargs.pop('library_name')
            library_section: Any = self.library.section(library_name)
        except KeyError as err:
            raise MediaNotFound("Must specify 'library_name' for this search") from err
        except NotFound as err:
            library_sections: List[str] = [section.title for section in self.library.sections()]
            raise MediaNotFound(f"Library '{library_name}' not found in {library_sections}") from err
        _LOGGER.debug('Searching for %s in %s using: %s', media_type, library_section, kwargs)
        return search_media(media_type, library_section, **kwargs)

    @property
    def sensor_attributes(self) -> Dict[str, Any]:
        """Return active session information for use in activity sensor."""
        return {x.sensor_user: x.sensor_title for x in self.active_sessions.values()}

    def set_plex_server(self, plex_server: plexapi.server.PlexServer) -> None:
        """Set the PlexServer instance."""
        self._plex_server = plex_server

    def switch_user(self, username: str) -> PlexServer:
        """Return a shallow copy of a PlexServer as the provided user."""
        new_server: PlexServer = copy(self)
        new_server.set_plex_server(self.plex_server.switchUser(username))
        return new_server
