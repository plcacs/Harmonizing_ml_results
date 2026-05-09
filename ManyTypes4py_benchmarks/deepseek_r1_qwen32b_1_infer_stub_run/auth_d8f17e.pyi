"""Type stubs for auth_d8f17e module."""

import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import List, Any, Dict, cast, Iterable, Union, Optional, Coroutine, AnyStr

class AuthError(Exception):
    pass

class OpenIdMixin:
    """Abstract implementation of OpenID and Attribute Exchange."""
    _OPENID_ENDPOINT: str

    def authenticate_redirect(self, callback_uri: str = None, ax_attrs: List[str] = ['name', 'email', 'language', 'username']) -> None:
        ...

    async def get_authenticated_user(self, http_client: httpclient.AsyncHTTPClient = None) -> Dict:
        ...

    def _openid_args(self, callback_uri: str, ax_attrs: List[str] = [], oauth_scope: str = None) -> Dict:
        ...

    def _on_authentication_verified(self, response: bytes) -> Dict:
        ...

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        ...

class OAuthMixin:
    """Abstract implementation of OAuth 1.0 and 1.0a."""
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_VERSION: str
    _OAUTH_NO_CALLBACKS: bool

    async def authorize_redirect(self, callback_uri: str = None, extra_params: Dict = None, http_client: httpclient.AsyncHTTPClient = None) -> None:
        ...

    async def get_authenticated_user(self, http_client: httpclient.AsyncHTTPClient = None) -> Dict:
        ...

    def _oauth_request_token_url(self, callback_uri: str = None, extra_params: Dict = None) -> str:
        ...

    def _on_request_token(self, authorize_url: str, callback_uri: str, response: Any) -> None:
        ...

    def _oauth_access_token_url(self, request_token: Dict) -> str:
        ...

    def _oauth_consumer_token(self) -> Dict:
        ...

    async def _oauth_get_user_future(self, access_token: Dict) -> Dict:
        ...

    def _oauth_request_parameters(self, url: str, access_token: Dict, parameters: Dict = {}, method: str = 'GET') -> Dict:
        ...

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        ...

class OAuth2Mixin:
    """Abstract implementation of OAuth 2.0."""
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str

    def authorize_redirect(self, redirect_uri: str = None, client_id: str = None, client_secret: str = None, extra_params: Dict = None, scope: List[str] = None, response_type: str = 'code') -> None:
        ...

    def _oauth_request_token_url(self, redirect_uri: str = None, client_id: str = None, client_secret: str = None, code: str = None, extra_params: Dict = None) -> str:
        ...

    async def oauth2_request(self, url: str, access_token: str = None, post_args: Dict = None, **args: Any) -> Dict:
        ...

    def get_auth_http_client(self) -> httpclient.AsyncHTTPClient:
        ...

class TwitterMixin(OAuthMixin):
    """Twitter OAuth authentication."""
    _OAUTH_REQUEST_TOKEN_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_AUTHENTICATE_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _TWITTER_BASE_URL: str

    async def authenticate_redirect(self, callback_uri: str = None) -> None:
        ...

    async def twitter_request(self, path: str, access_token: str, post_args: Dict = None, **args: Any) -> Dict:
        ...

    def _oauth_consumer_token(self) -> Dict:
        ...

    async def _oauth_get_user_future(self, access_token: Dict) -> Dict:
        ...

class GoogleOAuth2Mixin(OAuth2Mixin):
    """Google authentication using OAuth2."""
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_USERINFO_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _OAUTH_SETTINGS_KEY: str

    def get_google_oauth_settings(self) -> Dict:
        ...

    async def get_authenticated_user(self, redirect_uri: str, code: str, client_id: str = None, client_secret: str = None) -> Dict:
        ...

class FacebookGraphMixin(OAuth2Mixin):
    """Facebook authentication using the new Graph API and OAuth2."""
    _OAUTH_ACCESS_TOKEN_URL: str
    _OAUTH_AUTHORIZE_URL: str
    _OAUTH_NO_CALLBACKS: bool
    _FACEBOOK_BASE_URL: str

    async def get_authenticated_user(self, redirect_uri: str, client_id: str, client_secret: str, code: str, extra_fields: List[str] = None) -> Dict:
        ...

    async def facebook_request(self, path: str, access_token: str = None, post_args: Dict = None, **args: Any) -> Dict:
        ...

def _oauth_signature(consumer_token: Dict, method: str, url: str, parameters: Dict = {}, token: Dict = None) -> str:
    ...

def _oauth10a_signature(consumer_token: Dict, method: str, url: str, parameters: Dict = {}, token: Dict = None) -> str:
    ...

def _oauth_escape(val: AnyStr) -> str:
    ...

def _oauth_parse_response(body: bytes) -> Dict:
    ...