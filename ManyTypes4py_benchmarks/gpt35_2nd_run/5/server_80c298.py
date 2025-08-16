from __future__ import annotations
from copy import copy
import logging
import ssl
import time
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
from homeassistant.core import callback
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.dispatcher import async_dispatcher_send
from .const import CONF_IGNORE_NEW_SHARED_USERS, CONF_IGNORE_PLEX_WEB_CLIENTS, CONF_MONITORED_USERS, CONF_SERVER, CONF_SERVER_IDENTIFIER, CONF_USE_EPISODE_ART, DEBOUNCE_TIMEOUT, DEFAULT_VERIFY_SSL, GDM_DEBOUNCER, GDM_SCANNER, PLAYER_SOURCE, PLEX_NEW_MP_SIGNAL, PLEX_UPDATE_MEDIA_PLAYER_SESSION_SIGNAL, PLEX_UPDATE_MEDIA_PLAYER_SIGNAL, PLEX_UPDATE_SENSOR_SIGNAL, PLEXTV_THROTTLE, X_PLEX_DEVICE_NAME, X_PLEX_PLATFORM, X_PLEX_PRODUCT, X_PLEX_VERSION
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
    def __init__(self, hass, server_config: dict, known_server_id: str = None, options: dict = None, entry_id: str = None) -> None:
    @property
    def account(self) -> MyPlexAccount:
    def plextv_clients(self) -> list:
    def connect(self) -> None:
    @callback
    def async_refresh_entity(self, machine_identifier: str, device: PlexClient, session: PlexSession, source: str) -> None:
    async def async_update_session(self, payload: dict) -> None:
    def _fetch_platform_data(self) -> tuple:
    async def _async_update_platforms(self) -> None:
    @property
    def plex_server(self) -> PlexServer:
    @property
    def has_token(self) -> bool:
    @property
    def accounts(self) -> set:
    @property
    def owner(self) -> str:
    @property
    def version(self) -> str:
    @property
    def friendly_name(self) -> str:
    @property
    def machine_identifier(self) -> str:
    @property
    def url_in_use(self) -> str:
    @property
    def option_ignore_new_shared_users(self) -> bool:
    @property
    def option_use_episode_art(self) -> bool:
    @property
    def option_monitored_users(self) -> dict:
    @property
    def option_ignore_plexweb_clients(self) -> bool:
    @property
    def library(self) -> Library:
    def playlist(self, title: str) -> Playlist:
    def playlists(self) -> list:
    def create_playqueue(self, media: Media, **kwargs) -> PlayQueue:
    def create_station_playqueue(self, key: str) -> PlayQueue:
    def get_playqueue(self, playqueue_id: str) -> PlayQueue:
    def fetch_item(self, item: int) -> Item:
    def lookup_media(self, media_type: str, **kwargs) -> Media:
    @property
    def sensor_attributes(self) -> dict:
    def set_plex_server(self, plex_server: PlexServer) -> None:
    def switch_user(self, username: str) -> PlexServer:
