"""Config Flow using OAuth2.

This module exists of the following parts:
 - OAuth2 config flow which supports multiple OAuth2 implementations
 - OAuth2 implementation that works with local provided client ID/secret

"""
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
from typing import Any, Optional, Union, Dict, List
from aiohttp import ClientError, ClientResponseError, ClientSession, web
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

_LOGGER = logging.getLogger(__name__)

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
    """Return the redirect uri."""
    if 'my' in hass.config.components:
        return MY_AUTH_CALLBACK_PATH
    if (req := http.current_request.get()) is None:
        raise RuntimeError('No current request in context')
    if (ha_host := req.headers.get(HEADER_FRONTEND_BASE)) is None:
        raise RuntimeError('No header in request')
    return f'{ha_host}{AUTH_CALLBACK_PATH}'

class AbstractOAuth2Implementation(ABC):
    """Base class to abstract OAuth2 authentication."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the implementation."""

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain that is providing the implementation."""

    @abstractmethod
    async def async_generate_authorize_url(self, flow_id: str) -> str:
        """Generate a url for the user to authorize."""

    @abstractmethod
    async def async_resolve_external_data(self, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve external data to tokens."""

    async def async_refresh_token(self, token: Dict[str, Any]) -> Dict[str, Any]:
        """Refresh a token and update expires info."""
        new_token = await self._async_refresh_token(token)
        new_token['expires_in'] = int(new_token['expires_in'])
        new_token['expires_at'] = time.time() + new_token['expires_in']
        return new_token

    @abstractmethod
    async def _async_refresh_token(self, token: Dict[str, Any]) -> Dict[str, Any]:
        """Refresh a token."""

class LocalOAuth2Implementation(AbstractOAuth2Implementation):
    """Local OAuth2 implementation."""

    def __init__(self, hass: HomeAssistant, domain: str, client_id: str, client_secret: Optional[str], authorize_url: str, token_url: str) -> None:
        """Initialize local auth implementation."""
        self.hass = hass
        self._domain = domain
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorize_url = authorize_url
        self.token_url = token_url

    @property
    def name(self) -> str:
        """Name of the implementation."""
        return 'Configuration.yaml'

    @property
    def domain(self) -> str:
        """Domain providing the implementation."""
        return self._domain

    @property
    def redirect_uri(self) -> str:
        """Return the redirect uri."""
        return async_get_redirect_uri(self.hass)

    @property
    def extra_authorize_data(self) -> Dict[str, Any]:
        """Extra data that needs to be appended to the authorize url."""
        return {}

    async def async_generate_authorize_url(self, flow_id: str) -> str:
        """Generate a url for the user to authorize."""
        redirect_uri = self.redirect_uri
        return str(URL(self.authorize_url).with_query({
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': redirect_uri,
            'state': _encode_jwt(self.hass, {'flow_id': flow_id, 'redirect_uri': redirect_uri})
        }).update_query(self.extra_authorize_data))

    async def async_resolve_external_data(self, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve the authorization code to tokens."""
        return await self._token_request({
            'grant_type': 'authorization_code',
            'code': external_data['code'],
            'redirect_uri': external_data['state']['redirect_uri']
        })

    async def _async_refresh_token(self, token: Dict[str, Any]) -> Dict[str, Any]:
        """Refresh tokens."""
        new_token = await self._token_request({
            'grant_type': 'refresh_token',
            'client_id': self.client_id,
            'refresh_token': token['refresh_token']
        })
        return {**token, **new_token}

    async def _token_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a token request."""
        session: ClientSession = async_get_clientsession(self.hass)
        data['client_id'] = self.client_id
        if self.client_secret is not None:
            data['client_secret'] = self.client_secret
        _LOGGER.debug('Sending token request to %s', self.token_url)
        resp = await session.post(self.token_url, data=data)
        if resp.status >= 400:
            try:
                error_response = await resp.json()
            except (ClientError, JSONDecodeError):
                error_response = {}
            error_code = error_response.get('error', 'unknown')
            error_description = error_response.get('error_description', 'unknown error')
            _LOGGER.error('Token request for %s failed (%s): %s', self.domain, error_code, error_description)
        resp.raise_for_status()
        return cast(Dict[str, Any], await resp.json())

class AbstractOAuth2FlowHandler(config_entries.ConfigFlow, metaclass=ABCMeta):
    """Handle a config flow."""
    DOMAIN: str = ''
    VERSION: int = 1

    def __init__(self) -> None:
        """Instantiate config flow."""
        if self.DOMAIN == '':
            raise TypeError(f"Can't instantiate class {self.__class__.__name__} without DOMAIN being set")
        self.external_data: Optional[Dict[str, Any]] = None
        self.flow_impl: Optional[AbstractOAuth2Implementation] = None

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """Return logger."""

    @property
    def extra_authorize_data(self) -> Dict[str, Any]:
        """Extra data that needs to be appended to the authorize url."""
        return {}

    async def async_generate_authorize_url(self) -> str:
        """Generate a url for the user to authorize."""
        url = await self.flow_impl.async_generate_authorize_url(self.flow_id)
        return str(URL(url).update_query(self.extra_authorize_data))

    async def async_step_pick_implementation(self, user_input: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], web.Response]:
        """Handle a flow start."""
        implementations = await async_get_implementations(self.hass, self.DOMAIN)
        if user_input is not None:
            self.flow_impl = implementations[user_input['implementation']]
            return await self.async_step_auth()
        if not implementations:
            if self.DOMAIN in await async_get_application_credentials(self.hass):
                return self.async_abort(reason='missing_credentials')
            return self.async_abort(reason='missing_configuration')
        req = http.current_request.get()
        if len(implementations) == 1 and req is not None:
            self.flow_impl = list(implementations.values())[0]
            return await self.async_step_auth()
        return self.async_show_form(step_id='pick_implementation', data_schema=vol.Schema({
            vol.Required('implementation', default=list(implementations)[0]): vol.In({key: impl.name for key, impl in implementations.items()})
        }))

    async def async_step_auth(self, user_input: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], web.Response]:
        """Create an entry for auth."""
        if user_input is not None:
            self.external_data = user_input
            next_step = 'authorize_rejected' if 'error' in user_input else 'creation'
            return self.async_external_step_done(next_step_id=next_step)
        try:
            async with asyncio.timeout(OAUTH_AUTHORIZE_URL_TIMEOUT_SEC):
                url = await self.async_generate_authorize_url()
        except TimeoutError as err:
            _LOGGER.error('Timeout generating authorize url: %s', err)
            return self.async_abort(reason='authorize_url_timeout')
        except NoURLAvailableError:
            return self.async_abort(reason='no_url_available', description_placeholders={'docs_url': 'https://www.home-assistant.io/more-info/no-url-available'})
        return self.async_external_step(step_id='auth', url=url)

    async def async_step_creation(self, user_input: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], web.Response]:
        """Create config entry from external data."""
        _LOGGER.debug('Creating config entry from external data')
        try:
            async with asyncio.timeout(OAUTH_TOKEN_TIMEOUT_SEC):
                token = await self.flow_impl.async_resolve_external_data(self.external_data)
        except TimeoutError as err:
            _LOGGER.error('Timeout resolving OAuth token: %s', err)
            return self.async_abort(reason='oauth_timeout')
        except (ClientResponseError, ClientError) as err:
            _LOGGER.error('Error resolving OAuth token: %s', err)
            if isinstance(err, ClientResponseError) and err.status == HTTPStatus.UNAUTHORIZED:
                return self.async_abort(reason='oauth_unauthorized')
            return self.async_abort(reason='oauth_failed')
        if 'expires_in' not in token:
            _LOGGER.warning('Invalid token: %s', token)
            return self.async_abort(reason='oauth_error')
        try:
            token['expires_in'] = int(token['expires_in'])
        except ValueError as err:
            _LOGGER.warning('Error converting expires_in to int: %s', err)
            return self.async_abort(reason='oauth_error')
        token['expires_at'] = time.time() + token['expires_in']
        self.logger.info('Successfully authenticated')
        return await self.async_oauth_create_entry({'auth_implementation': self.flow_impl.domain, 'token': token})

    async def async_step_authorize_rejected(self, data: Optional[Dict[str, Any]] = None) -> web.Response:
        """Step to handle flow rejection."""
        return self.async_abort(reason='user_rejected_authorize', description_placeholders={'error': self.external_data['error']})

    async def async_oauth_create_entry(self, data: Dict[str, Any]) -> web.Response:
        """Create an entry for the flow."""
        return self.async_create_entry(title=self.flow_impl.name, data=data)

    async def async_step_user(self, user_input: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], web.Response]:
        """Handle a flow start."""
        return await self.async_step_pick_implementation(user_input)

    @classmethod
    def async_register_implementation(cls, hass: HomeAssistant, local_impl: AbstractOAuth2Implementation) -> None:
        """Register a local implementation."""
        async_register_implementation(hass, cls.DOMAIN, local_impl)

@callback
def async_register_implementation(hass: HomeAssistant, domain: str, implementation: AbstractOAuth2Implementation) -> None:
    """Register an OAuth2 flow implementation for an integration."""
    implementations = hass.data.setdefault(DATA_IMPLEMENTATIONS, {})
    implementations.setdefault(domain, {})[implementation.domain] = implementation

async def async_get_implementations(hass: HomeAssistant, domain: str) -> Dict[str, AbstractOAuth2Implementation]:
    """Return OAuth2 implementations for specified domain."""
    registered = hass.data.setdefault(DATA_IMPLEMENTATIONS, {}).get(domain, {})
    if DATA_PROVIDERS not in hass.data:
        return registered
    registered = dict(registered)
    for get_impl in list(hass.data[DATA_PROVIDERS].values()):
        for impl in await get_impl(hass, domain):
            registered[impl.domain] = impl
    return registered

async def async_get_config_entry_implementation(hass: HomeAssistant, config_entry: config_entries.ConfigEntry) -> AbstractOAuth2Implementation:
    """Return the implementation for this config entry."""
    implementations = await async_get_implementations(hass, config_entry.domain)
    implementation = implementations.get(config_entry.data['auth_implementation'])
    if implementation is None:
        raise ValueError('Implementation not available')
    return implementation

@callback
def async_add_implementation_provider(hass: HomeAssistant, provider_domain: str, async_provide_implementation: Callable[[HomeAssistant, str], Awaitable[List[AbstractOAuth2Implementation]]]) -> None:
    """Add an implementation provider."""
    hass.data.setdefault(DATA_PROVIDERS, {})[provider_domain] = async_provide_implementation

class OAuth2AuthorizeCallbackView(http.HomeAssistantView):
    """OAuth2 Authorization Callback View."""
    requires_auth: bool = False
    url: str = AUTH_CALLBACK_PATH
    name: str = 'auth:external:callback'

    async def get(self, request: web.Request) -> web.Response:
        """Receive authorization code."""
        if 'state' not in request.query:
            return web.Response(text='Missing state parameter')
        hass: HomeAssistant = request.app[http.KEY_HASS]
        state = _decode_jwt(hass, request.query['state'])
        if state is None:
            return web.Response(text='Invalid state. Is My Home Assistant configured to go to the right instance?', status=400)
        user_input = {'state': state}
        if 'code' in request.query:
            user_input['code'] = request.query['code']
        elif 'error' in request.query:
            user_input['error'] = request.query['error']
        else:
            return web.Response(text='Missing code or error parameter')
        await hass.config_entries.flow.async_configure(flow_id=state['flow_id'], user_input=user_input)
        _LOGGER.debug('Resumed OAuth configuration flow')
        return web.Response(headers={'content-type': 'text/html'}, text='<script>window.close()</script>')

class OAuth2Session:
    """Session to make requests authenticated with OAuth2."""

    def __init__(self, hass: HomeAssistant, config_entry: config_entries.ConfigEntry, implementation: AbstractOAuth2Implementation) -> None:
        """Initialize an OAuth2 session."""
        self.hass = hass
        self.config_entry = config_entry
        self.implementation = implementation
        self._token_lock = Lock()

    @property
    def token(self) -> Dict[str, Any]:
        """Return the token."""
        return cast(Dict[str, Any], self.config_entry.data['token'])

    @property
    def valid_token(self) -> bool:
        """Return if token is still valid."""
        return cast(float, self.token['expires_at']) > time.time() + CLOCK_OUT_OF_SYNC_MAX_SEC

    async def async_ensure_token_valid(self) -> None:
        """Ensure that the current token is valid."""
        async with self._token_lock:
            if self.valid_token:
                return
            new_token = await self.implementation.async_refresh_token(self.token)
            self.hass.config_entries.async_update_entry(self.config_entry, data={**self.config_entry.data, 'token': new_token})

    async def async_request(self, method: str, url: str, **kwargs: Any) -> ClientResponseError:
        """Make a request."""
        await self.async_ensure_token_valid()
        return await async_oauth2_request(self.hass, self.config_entry.data['token'], method, url, **kwargs)

async def async_oauth2_request(hass: HomeAssistant, token: Dict[str, Any], method: str, url: str, **kwargs: Any) -> ClientResponseError:
    """Make an OAuth2 authenticated request."""
    session: ClientSession = async_get_clientsession(hass)
    headers = kwargs.pop('headers', {})
    return await session.request(method, url, **kwargs, headers={**headers, 'authorization': f'Bearer {token["access_token"]}'})

@callback
def _encode_jwt(hass: HomeAssistant, data: Dict[str, Any]) -> str:
    """JWT encode data."""
    if (secret := hass.data.get(DATA_JWT_SECRET)) is None:
        secret = hass.data[DATA_JWT_SECRET] = secrets.token_hex()
    return jwt.encode(data, secret, algorithm='HS256')

@callback
def _decode_jwt(hass: HomeAssistant, encoded: str) -> Optional[Dict[str, Any]]:
    """JWT encode data."""
    secret = hass.data.get(DATA_JWT_SECRET)
    if secret is None:
        return None
    try:
        return jwt.decode(encoded, secret, algorithms=['HS256'])
    except jwt.InvalidTokenError:
        return None
