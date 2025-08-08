from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
import asyncio
from asyncio import Lock
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from json import JSONDecodeError
import logging
import secrets
import time
from typing import Any, cast
from aiohttp import ClientError, ClientResponseError, client, web
import jwt
import voluptuous as vol
from yarl import URL
from homeassistant import config_entries
from homeassistant.components import http
from homeassistant.core import HomeAssistant, callback
from homeassistant.loader import async_get_application_credentials
from homeassistant.util.hass_dict import HassKey
from .aiohttp_client import async_get_clientsession
from .network import NoURLAvailableError
_LOGGER: logging.Logger = logging.getLogger(__name__)
DATA_JWT_SECRET: str = 'oauth2_jwt_secret'
DATA_IMPLEMENTATIONS: HassKey = HassKey('oauth2_impl')
DATA_PROVIDERS: HassKey = HassKey('oauth2_providers')
AUTH_CALLBACK_PATH: str = '/auth/external/callback'
HEADER_FRONTEND_BASE: str = 'HA-Frontend-Base'
MY_AUTH_CALLBACK_PATH: str = 'https://my.home-assistant.io/redirect/oauth'
CLOCK_OUT_OF_SYNC_MAX_SEC: int = 20
OAUTH_AUTHORIZE_URL_TIMEOUT_SEC: int = 30
OAUTH_TOKEN_TIMEOUT_SEC: int = 30

@callback
def async_get_redirect_uri(hass: HomeAssistant) -> str:
    ...

class AbstractOAuth2Implementation(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def domain(self) -> str:
        ...

    @abstractmethod
    async def async_generate_authorize_url(self, flow_id: str) -> None:
        ...

    @abstractmethod
    async def async_resolve_external_data(self, external_data: Any) -> None:
        ...

    async def async_refresh_token(self, token: dict) -> dict:
        ...

    @abstractmethod
    async def _async_refresh_token(self, token: dict) -> dict:
        ...

class LocalOAuth2Implementation(AbstractOAuth2Implementation):
    def __init__(self, hass: HomeAssistant, domain: str, client_id: str, client_secret: str, authorize_url: str, token_url: str) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def domain(self) -> str:
        ...

    @property
    def redirect_uri(self) -> str:
        ...

    @property
    def extra_authorize_data(self) -> dict:
        ...

    async def async_generate_authorize_url(self, flow_id: str) -> str:
        ...

    async def async_resolve_external_data(self, external_data: dict) -> dict:
        ...

    async def _async_refresh_token(self, token: dict) -> dict:
        ...

    async def _token_request(self, data: dict) -> dict:
        ...

class AbstractOAuth2FlowHandler(config_entries.ConfigFlow, metaclass=ABCMeta):
    DOMAIN: str = ''
    VERSION: int = 1

    def __init__(self) -> None:
        ...

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        ...

    @property
    def extra_authorize_data(self) -> dict:
        ...

    async def async_generate_authorize_url(self) -> str:
        ...

    async def async_step_pick_implementation(self, user_input: dict = None) -> None:
        ...

    async def async_step_auth(self, user_input: dict = None) -> None:
        ...

    async def async_step_creation(self, user_input: dict = None) -> None:
        ...

    async def async_step_authorize_rejected(self, data: Any = None) -> None:
        ...

    async def async_oauth_create_entry(self, data: dict) -> None:
        ...

    async def async_step_user(self, user_input: dict = None) -> None:
        ...

    @classmethod
    def async_register_implementation(cls, hass: HomeAssistant, local_impl: LocalOAuth2Implementation) -> None:
        ...

@callback
def async_register_implementation(hass: HomeAssistant, domain: str, implementation: AbstractOAuth2Implementation) -> None:
    ...

async def async_get_implementations(hass: HomeAssistant, domain: str) -> dict:
    ...

async def async_get_config_entry_implementation(hass: HomeAssistant, config_entry: config_entries.ConfigEntry) -> AbstractOAuth2Implementation:
    ...

@callback
def async_add_implementation_provider(hass: HomeAssistant, provider_domain: str, async_provide_implementation: Callable) -> None:
    ...

class OAuth2AuthorizeCallbackView(http.HomeAssistantView):
    ...

class OAuth2Session:
    ...

async def async_oauth2_request(hass: HomeAssistant, token: dict, method: str, url: str, **kwargs: Any) -> Any:
    ...

@callback
def _encode_jwt(hass: HomeAssistant, data: dict) -> str:
    ...

@callback
def _decode_jwt(hass: HomeAssistant, encoded: str) -> dict:
    ...
